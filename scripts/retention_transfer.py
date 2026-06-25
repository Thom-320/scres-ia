#!/usr/bin/env python3
"""Corrected retained-vs-frozen TRANSFER protocol (2026-06-24 audit fixes).

The earlier protocol measured the WRONG estimand: it let both arms re-learn the
current block before evaluating, i.e. the asymptotic point where reset catches up.
This runner measures TRANSFER instead:

  At the start of each disruption block k, evaluate the retained learner
  (theta accumulated from blocks 0..k-1) and the frozen baseline (theta_0),
  COLD -- before any block-k training. ΔR_k = R_retained - R_frozen is the
  head-start that accumulated learning gives on a NEW shock. Then the retained
  learner trains incrementally on block k and carries theta forward.

Fixes applied:
  #1/#3 transfer estimand (evaluate before block-k update);
  #2 clean outcome (treatment-window ReT, supply_chain.clean_metrics);
  #4 budget in BLOCKS via weekly cadence (no block-collapsing wrapper);
  #6 >=10 independent learner seeds (seed-clustered CI);
  #7 frozen = theta_0 is the no-accumulated-learning baseline; the cold eval on an
     UNSEEN block means "more training" cannot trivially explain a positive ΔR --
     it must generalise.

Track A discrete [6,3], persistent regime tape, regime hidden via mask preset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.evaluate_retained_reset_learning as ev  # noqa: E402
from supply_chain.clean_metrics import treatment_filtered_order_ret  # noqa: E402


def cluster_stats(xs: list[float]) -> dict:
    a = np.array([x for x in xs if np.isfinite(x)], dtype=float)
    n = int(a.size)
    mean = float(a.mean()) if n else float("nan")
    sem = float(a.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return {"n": n, "mean": mean, "sem": sem,
            "ci95_lo": mean - 1.96 * sem if n > 1 else float("nan"),
            "ci95_hi": mean + 1.96 * sem if n > 1 else float("nan")}


def clean_eval(args, model, regime, seed: int) -> float:
    """Run one block (episode) COLD and return treatment-window order-level ReT."""
    env = ev.build_env(args, regime=regime)
    obs, _ = env.reset(seed=seed)
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _r, term, trunc, _i = env.step(int(np.asarray(action).item()))
        done = bool(term or trunc)
    res = treatment_filtered_order_ret(env.unwrapped.sim)
    env.close()
    return float(res["mean_ret"])


def run_seed(args, seed: int, tape, n_blocks: int, train_per_block: int) -> list[float]:
    args.seed = seed
    args.online_timesteps_per_cycle = train_per_block
    with tempfile.TemporaryDirectory() as tmp:
        init = Path(tmp) / "init.zip"
        ev.build_initial_model(args, init)        # theta_0 (optional pretrain)
        frozen = ev.load_model(args, init)         # never trains
        retained = ev.load_model(args, init)       # accumulates across blocks
        deltas = []
        for k in range(n_blocks):
            regime = tape[k]
            eval_seed = 90_000 + seed * 1000 + k
            # COLD transfer eval BEFORE training on block k:
            r_frozen = clean_eval(args, frozen, regime, eval_seed)
            r_retained = clean_eval(args, retained, regime, eval_seed)
            deltas.append(r_retained - r_frozen)
            # THEN accumulate: train retained on block k, carry theta forward.
            ev.online_update(args, retained,
                             seed=eval_seed + ev.ADAPT_SEED_OFFSET, regime=regime)
    return deltas


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", default="retention_transfer")
    p.add_argument("--output-root", type=Path,
                   default=Path("outputs/benchmarks/retention_transfer"))
    p.add_argument("--reward-mode", default="control_v1")
    p.add_argument("--seeds", default="8201,8202,8203,8204,8205,8206,8207,8208,8209,8210")
    p.add_argument("--n-blocks", type=int, default=40)
    p.add_argument("--max-steps", type=int, default=12)
    p.add_argument("--train-per-block", type=int, default=150)
    p.add_argument("--rho-disruption", type=float, default=0.85)
    p.add_argument("--regime-seed", type=int, default=909)
    p.add_argument("--mask-preset", default="direct_disruption_blind")
    p.add_argument("--pretrain-timesteps", type=int, default=0)
    p.add_argument("--learning-starts", type=int, default=50)
    p.add_argument("--buffer-size", type=int, default=10_000)
    # Capacity inertia (Ed.2): make anticipation valuable so memory pays off.
    p.add_argument("--surge-inertia", action="store_true")
    p.add_argument("--surge-budget-hours", type=float, default=2016.0)
    p.add_argument("--surge-ramp-per-step", type=int, default=1)
    cli = p.parse_args()
    seeds = [int(s) for s in cli.seeds.split(",") if s.strip()]

    def base_args():
        a = ev.build_parser().parse_args([])
        a.track = "a"; a.algo = "dqn"; a.decision_cadence = "weekly"
        a.reward_mode = cli.reward_mode; a.max_steps = cli.max_steps
        a.mask_preset = cli.mask_preset
        a.pretrain_timesteps = cli.pretrain_timesteps
        a.learning_starts = cli.learning_starts; a.buffer_size = cli.buffer_size
        a.rho_disruption = cli.rho_disruption; a.rho_demand = None
        a.regime_seed = cli.regime_seed
        a.surge_inertia = cli.surge_inertia
        a.surge_budget_hours = cli.surge_budget_hours
        a.surge_ramp_per_step = cli.surge_ramp_per_step
        return a

    tape = ev.build_tape(base_args(), cli.n_blocks, seed=cli.regime_seed)
    assert tape is not None, "regime tape required (set --rho-disruption)"

    per_seed = []  # list of per-block delta lists
    for s in seeds:
        print(f"[transfer] seed {s} ...", flush=True)
        per_seed.append(run_seed(base_args(), s, tape, cli.n_blocks, cli.train_per_block))
    arr = np.array(per_seed)  # [seeds, blocks]

    # Learning curve: per-block transfer clustered over seeds.
    by_block = [cluster_stats(list(arr[:, k])) for k in range(cli.n_blocks)]
    # Overall transfer (mean per seed over blocks), and early vs late halves.
    seed_means = [float(np.nanmean(arr[i])) for i in range(len(seeds))]
    half = cli.n_blocks // 2
    early = [float(np.nanmean(arr[i, :half])) for i in range(len(seeds))]
    late = [float(np.nanmean(arr[i, half:])) for i in range(len(seeds))]
    slope = [float(np.polyfit(np.arange(cli.n_blocks), arr[i], 1)[0]) for i in range(len(seeds))]

    run_dir = cli.output_root / cli.label
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "estimand": "transfer: clean ReT(retained, cold) - ReT(theta_0, cold) on unseen block k",
        "reward_mode": cli.reward_mode, "seeds": seeds, "n_blocks": cli.n_blocks,
        "train_per_block": cli.train_per_block, "rho_disruption": cli.rho_disruption,
        "mask_preset": cli.mask_preset,
        "overall_transfer": cluster_stats(seed_means),
        "early_half": cluster_stats(early), "late_half": cluster_stats(late),
        "learning_slope_per_block": cluster_stats(slope),
        "transfer_by_block": by_block,
    }
    (run_dir / "transfer.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    ov = payload["overall_transfer"]; sl = payload["learning_slope_per_block"]
    print("\nCORRECTED TRANSFER PROTOCOL (clean ReT, cold eval, seed-clustered)")
    print(f"  overall transfer ΔR = {ov['mean']:+.4f} +/-{ov['sem']:.4f} "
          f"ci95=[{ov['ci95_lo']:+.4f},{ov['ci95_hi']:+.4f}] (n={ov['n']})")
    print(f"  early half = {payload['early_half']['mean']:+.4f}   "
          f"late half = {payload['late_half']['mean']:+.4f}")
    print(f"  learning slope/block = {sl['mean']:+.5f} +/-{sl['sem']:.5f}")
    print(f"  Saved: {run_dir / 'transfer.json'}")
    print("  Read: ΔR>0 and CI>0 => accumulated learning gives a head-start on new shocks (H1/H4).")
    print("        late>early / slope>0 => the head-start grows with exposure (H2 learning curve).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
