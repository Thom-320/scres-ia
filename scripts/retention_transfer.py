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
    cd_sigmoid_sum = 0.0
    cd_train_sum = 0.0
    n_steps = 0
    continuous = getattr(args, "track", "a") == "continuous"
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        a = (np.asarray(action, dtype=np.float32).reshape(-1) if continuous
             else int(np.asarray(action).item()))
        obs, _r, term, trunc, info = env.step(a)
        cd_sigmoid_sum += float(info.get("ret_garrido2024_sigmoid_step", 0.0))
        cd_train_sum += float(info.get("ret_garrido2024_train_step", 0.0))
        n_steps += 1
        done = bool(term or trunc)
    res = treatment_filtered_order_ret(env.unwrapped.sim)
    env.close()
    # Cost-aware CD variants are separated because the sigmoid reporting index
    # and the train-shaped index can prefer different static shifts.
    outcome = getattr(args, "outcome", "excel_ret")
    if outcome in {"cd_index", "cd_sigmoid_index"}:
        return cd_sigmoid_sum / max(n_steps, 1)
    if outcome == "cd_train_index":
        return cd_train_sum / max(n_steps, 1)
    return float(res["mean_ret"])


def run_seed(args, seed: int, tape, n_blocks: int, train_per_block: int) -> dict:
    """Three arms, all evaluated COLD on each unseen block k (before block-k data):
      frozen   = theta_0 (no learning);
      reset    = theta_0 + ONLY the previous block (single-block learning);
      retained = theta_0 + ALL prior blocks (accumulated learning).
    retained - reset isolates cross-block MEMORY L_{k-1}; retained - frozen is total.
    """
    args.seed = seed
    args.online_timesteps_per_cycle = train_per_block
    with tempfile.TemporaryDirectory() as tmp:
        init = Path(tmp) / "init.zip"
        ev.build_initial_model(args, init)
        frozen = ev.load_model(args, init)
        retained = ev.load_model(args, init)
        reset = ev.load_model(args, init)          # holds single-block training
        d_total, d_mem = [], []
        for k in range(n_blocks):
            regime = tape[k]
            eval_seed = 90_000 + seed * 1000 + k
            data_seed = eval_seed + ev.ADAPT_SEED_OFFSET
            r_frozen = clean_eval(args, frozen, regime, eval_seed)
            r_retained = clean_eval(args, retained, regime, eval_seed)
            r_reset = clean_eval(args, reset, regime, eval_seed)
            d_total.append(r_retained - r_frozen)   # any learning vs none
            d_mem.append(r_retained - r_reset)       # accumulation vs single block
            # L_{k-1} contract: retain WEIGHTS (+ target net), but clear the replay
            # buffer so memory is compressed parametric routines, not a folder of old
            # episodes. (--retain-buffer keeps the buffer as a secondary L^full lane.)
            # DQN-only: clear replay buffer so memory is parametric, not stored episodes.
            # PPO (continuous) has no replay buffer (on-policy rollouts refresh each learn()).
            if not getattr(args, "retain_buffer", False) and hasattr(retained, "replay_buffer"):
                retained.replay_buffer.reset()
            ev.online_update(args, retained, seed=data_seed, regime=regime)
            reset = ev.load_model(args, init)   # theta_0, fresh buffer
            ev.online_update(args, reset, seed=data_seed, regime=regime)
    return {"total": d_total, "mem": d_mem}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", default="retention_transfer")
    p.add_argument("--output-root", type=Path,
                   default=Path("outputs/benchmarks/retention_transfer"))
    p.add_argument("--reward-mode", default="control_v1")
    # Continuous_its (preventive-Pareto winner) lane: PPO Box action + hazard obs + holding cost.
    p.add_argument("--track", choices=("a", "continuous"), default="a")
    p.add_argument("--algo", choices=("dqn", "ppo"), default=None,
                   help="default dqn for track a, ppo for continuous")
    p.add_argument("--observation-version", default=None,
                   help="default v5 for track a, v6 for continuous")
    p.add_argument("--risk-obs", action="store_true", help="continuous: realized-risk + hazard obs")
    p.add_argument("--holding-cost", type=float, default=0.0)
    p.add_argument("--shift-cost", type=float, default=0.001)
    p.add_argument("--init-frac", type=float, default=1.0)
    p.add_argument("--n-steps", type=int, default=256, help="PPO rollout length (continuous)")
    p.add_argument("--n-epochs", type=int, default=10, help="PPO epochs (continuous)")
    p.add_argument("--learning-rate", type=float, default=3e-4)
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
    p.add_argument("--stochastic-pt", action="store_true")
    p.add_argument("--control-v2-w-fill", type=float, default=1.0)
    p.add_argument("--control-v2-w-service", type=float, default=4.0)
    p.add_argument("--control-v2-w-lost", type=float, default=2.0)
    p.add_argument("--control-v2-w-inventory", type=float, default=0.05)
    p.add_argument("--control-v2-w-shift", type=float, default=0.08)
    p.add_argument("--control-v2-w-switch", type=float, default=0.02)
    # Capacity inertia (Ed.2): make anticipation valuable so memory pays off.
    p.add_argument("--surge-inertia", action="store_true")
    p.add_argument("--surge-budget-hours", type=float, default=2016.0)
    p.add_argument("--surge-ramp-per-step", type=int, default=1)
    # Cost-augmented Cobb-Douglas frontier env (frozen candidate from the joint 6x3 gate):
    # the regime-dependent shift optimum lives in the CD index, so the OUTCOME is the CD
    # index (cd_index), not the cost-free Excel ReT (which has no headroom).
    p.add_argument("--risk-frequency-multiplier", type=float, default=1.0)
    p.add_argument("--risk-impact-multiplier", type=float, default=1.0)
    p.add_argument("--ret-g24-shift-cost", type=float, default=0.5)
    p.add_argument("--ret-g24-kappa-train-frac", type=float, default=0.2)
    p.add_argument(
        "--outcome",
        default="excel_ret",
        choices=["excel_ret", "cd_index", "cd_sigmoid_index", "cd_train_index"],
    )
    cli = p.parse_args()
    seeds = [int(s) for s in cli.seeds.split(",") if s.strip()]

    continuous = cli.track == "continuous"

    def base_args():
        a = ev.build_parser().parse_args([])
        a.track = cli.track
        a.algo = cli.algo or ("ppo" if continuous else "dqn")
        a.observation_version = cli.observation_version or ("v6" if continuous else "v5")
        a.decision_cadence = "weekly"
        a.reward_mode = cli.reward_mode
        # continuous_its knobs (read via getattr in ev.build_env)
        a.risk_obs = cli.risk_obs
        a.holding_cost = cli.holding_cost
        a.shift_cost = cli.shift_cost
        a.init_frac = cli.init_frac
        a.n_steps = cli.n_steps
        a.n_epochs = cli.n_epochs
        a.learning_rate = cli.learning_rate
        a.max_steps = cli.max_steps
        # the named masks zero discrete-obs fields; for continuous keep the hazard obs visible
        # (it IS the winning lane). Retention is then tested on the faithful winning config.
        a.mask_preset = "none" if continuous else cli.mask_preset
        a.pretrain_timesteps = cli.pretrain_timesteps
        a.learning_starts = cli.learning_starts
        a.buffer_size = cli.buffer_size
        a.stochastic_pt = cli.stochastic_pt
        a.control_v2_w_fill = cli.control_v2_w_fill
        a.control_v2_w_service = cli.control_v2_w_service
        a.control_v2_w_lost = cli.control_v2_w_lost
        a.control_v2_w_inventory = cli.control_v2_w_inventory
        a.control_v2_w_shift = cli.control_v2_w_shift
        a.control_v2_w_switch = cli.control_v2_w_switch
        a.rho_disruption = cli.rho_disruption
        a.rho_demand = None
        a.regime_seed = cli.regime_seed
        a.surge_inertia = cli.surge_inertia
        a.surge_budget_hours = cli.surge_budget_hours
        a.surge_ramp_per_step = cli.surge_ramp_per_step
        a.risk_frequency_multiplier = cli.risk_frequency_multiplier
        a.risk_impact_multiplier = cli.risk_impact_multiplier
        a.ret_g24_shift_cost = cli.ret_g24_shift_cost
        a.ret_g24_kappa_train_frac = cli.ret_g24_kappa_train_frac
        a.outcome = cli.outcome
        return a

    tape = ev.build_tape(base_args(), cli.n_blocks, seed=cli.regime_seed)
    assert tape is not None, "regime tape required (set --rho-disruption)"

    runs = []
    for s in seeds:
        print(f"[transfer] seed {s} ...", flush=True)
        runs.append(run_seed(base_args(), s, tape, cli.n_blocks, cli.train_per_block))

    def summarize(key: str) -> dict:
        arr = np.array([r[key] for r in runs])  # [seeds, blocks]
        half = cli.n_blocks // 2
        seed_means = [float(np.nanmean(arr[i])) for i in range(len(seeds))]
        early_slice = slice(0, max(1, half))
        early = [float(np.nanmean(arr[i, early_slice])) for i in range(len(seeds))]
        late = [float(np.nanmean(arr[i, half:])) for i in range(len(seeds))]
        if cli.n_blocks >= 2:
            slope = [
                float(np.polyfit(np.arange(cli.n_blocks), arr[i], 1)[0])
                for i in range(len(seeds))
            ]
        else:
            slope = [0.0 for _ in seeds]
        return {
            "overall": cluster_stats(seed_means),
            "early_half": cluster_stats(early), "late_half": cluster_stats(late),
            "learning_slope_per_block": cluster_stats(slope),
            "by_block": [cluster_stats(list(arr[:, k])) for k in range(cli.n_blocks)],
        }

    mem = summarize("mem")       # retained - reset = cross-block MEMORY (L_{k-1})
    total = summarize("total")   # retained - frozen = total learning value

    run_dir = cli.output_root / cli.label
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "primary_estimand": "memory: clean ReT(retained, cold) - ReT(reset=theta_0+1block, cold)",
        "reward_mode": cli.reward_mode, "seeds": seeds, "n_blocks": cli.n_blocks,
        "train_per_block": cli.train_per_block, "rho_disruption": cli.rho_disruption,
        "mask_preset": cli.mask_preset, "surge_inertia": cli.surge_inertia,
        "surge_budget_hours": cli.surge_budget_hours,
        "stochastic_pt": cli.stochastic_pt,
        "control_v2_weights": {
            "w_fill": cli.control_v2_w_fill,
            "w_service": cli.control_v2_w_service,
            "w_lost": cli.control_v2_w_lost,
            "w_inventory": cli.control_v2_w_inventory,
            "w_shift": cli.control_v2_w_shift,
            "w_switch": cli.control_v2_w_switch,
        },
        "outcome": cli.outcome,
        "frozen_env": {
            "risk_frequency_multiplier": cli.risk_frequency_multiplier,
            "risk_impact_multiplier": cli.risk_impact_multiplier,
            "ret_g24_shift_cost": cli.ret_g24_shift_cost,
            "ret_g24_kappa_train_frac": cli.ret_g24_kappa_train_frac,
        },
        "memory_retained_minus_reset": mem,
        "total_retained_minus_frozen": total,
    }
    (run_dir / "transfer.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    m = mem["overall"]
    t = total["overall"]
    print("\nCORRECTED TRANSFER PROTOCOL (clean ReT, cold eval, seed-clustered)")
    print(f"  MEMORY (retained - reset) ΔR = {m['mean']:+.4f} +/-{m['sem']:.4f} "
          f"ci95=[{m['ci95_lo']:+.4f},{m['ci95_hi']:+.4f}] (n={m['n']})")
    print(f"     early={mem['early_half']['mean']:+.4f} late={mem['late_half']['mean']:+.4f} "
          f"slope/block={mem['learning_slope_per_block']['mean']:+.5f}")
    print(f"  total  (retained - frozen) ΔR = {t['mean']:+.4f} +/-{t['sem']:.4f}")
    print("  Read: MEMORY>0 with CI>0 => accumulating MORE than one block helps (L_{k-1}).")
    print(f"  Saved: {run_dir / 'transfer.json'}")
    print("  Read: ΔR>0 and CI>0 => accumulated learning gives a head-start on new shocks (H1/H4).")
    print("        late>early / slope>0 => the head-start grows with exposure (H2 learning curve).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
