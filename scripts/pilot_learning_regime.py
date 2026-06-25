#!/usr/bin/env python3
"""Pilot the frozen learning regime on TRAINING tapes only.

Anti-fishing protocol (docs/PAPER_CONTRACT_2026-06-24.md): before the one-shot
held-out retained-vs-reset contrast, we confirm the persistence regime is not
dead-on-arrival -- i.e. that retained adaptation moves at all, and that the
retained-minus-reset gap tends to widen with disruption persistence rho (the H2
dose-response sanity check). This pilot sweeps ``rho_disruption`` over training
tapes and never reads the held-out evaluation seeds.

It reuses the exact contrast primitives from ``evaluate_retained_reset_learning``
so the semantics match the confirmatory run. Results here may NOT be used to retune
reward/observation/architecture -- they only decide whether the regime is worth a
powered run.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.evaluate_retained_reset_learning as ev  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", default=None)
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/benchmarks/retained_reset_learning/pilots"),
    )
    p.add_argument("--algo", choices=("dqn", "ppo"), default="dqn")
    p.add_argument(
        "--reward-mode",
        default="ReT_cd",
        help="Frozen Track A training reward (contract default ReT_cd).",
    )
    # 0.3334 ~ memoryless (1/3) for 3 disruption levels; exact 1/3 floats below the
    # validity floor, so use the smallest value at/above it.
    p.add_argument("--rhos", default="0.3334,0.6,0.9", help="rho_disruption sweep")
    p.add_argument("--cycles", type=int, default=10, help="disruption blocks per run")
    p.add_argument("--pilot-seed-base", type=int, default=4400)
    p.add_argument("--regime-seed", type=int, default=5000)
    p.add_argument(
        "--regime-seeds",
        default=None,
        help=(
            "Comma-separated tape seeds. With >1, each rho is estimated over several "
            "independent tapes and reported with a seed-CLUSTERED CI, so the dose-"
            "response is not driven by one tape's phase composition. Defaults to the "
            "single --regime-seed."
        ),
    )
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument(
        "--mask-obs-indices",
        default=None,
        help="Comma obs indices to zero (regime-observability test); v5 disruption=17,19,23.",
    )
    p.add_argument("--online-timesteps-per-cycle", type=int, default=1500)
    p.add_argument("--pretrain-timesteps", type=int, default=2000)
    p.add_argument("--learning-starts", type=int, default=100)
    p.add_argument("--buffer-size", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    return p


def base_args(args: argparse.Namespace) -> argparse.Namespace:
    """A namespace the evaluator's primitives accept, fixed across the rho sweep."""
    ns = ev.build_parser().parse_args([])
    ns.algo = args.algo
    ns.reward_mode = args.reward_mode
    ns.max_steps = args.max_steps
    ns.online_timesteps_per_cycle = args.online_timesteps_per_cycle
    ns.pretrain_timesteps = args.pretrain_timesteps
    ns.learning_starts = args.learning_starts
    ns.buffer_size = args.buffer_size
    ns.seed = args.seed
    ns.regime_seed = args.regime_seed
    ns.mask_obs_indices = args.mask_obs_indices
    ns.rho_demand = None  # memoryless: isolate disruption persistence (ablation)
    return ns


def run_one_rho(
    args: argparse.Namespace, rho: float, pilot_seeds: list[int], tape_seed: int
) -> dict[str, object]:
    started = perf_counter()
    ns = copy.copy(base_args(args))
    ns.rho_disruption = rho
    ns.regime_seed = tape_seed
    # TRAINING tape only -- never the held-out eval tape.
    tape = ev.build_tape(ns, len(pilot_seeds), seed=ns.regime_seed)
    assert tape is not None
    print(
        f"[pilot] rho={rho:.4f} start cycles={len(pilot_seeds)} "
        f"pretrain={ns.pretrain_timesteps} online_per_cycle={ns.online_timesteps_per_cycle}",
        flush=True,
    )

    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory() as tmp:
        init_path = Path(tmp) / f"init_{ns.algo}.zip"
        ev.build_initial_model(ns, init_path)
        print(f"[pilot] rho={rho:.4f} initial model ready", flush=True)
        frozen = ev.load_model(ns, init_path)
        retained = ev.load_model(ns, init_path)

        for ci, seed in enumerate(pilot_seeds):
            regime = tape[ci]
            adapt_seed = seed + ev.ADAPT_SEED_OFFSET
            cycle_started = perf_counter()
            # Frozen reference (theta_0).
            rows.append(
                ev.run_episode(
                    args=ns, condition="frozen", seed=seed, cycle_index=ci,
                    policy_fn=ev.model_policy(frozen), regime=regime,
                )
            )
            # Retained: adapt (carrying theta), then eval.
            ev.online_update(ns, retained, seed=adapt_seed, regime=regime)
            rows.append(
                ev.run_episode(
                    args=ns, condition="retained", seed=seed, cycle_index=ci,
                    policy_fn=ev.model_policy(retained), regime=regime,
                )
            )
            # Reset: reload theta_0, adapt on this block only, then eval.
            reset = ev.load_model(ns, init_path)
            ev.online_update(ns, reset, seed=adapt_seed, regime=regime)
            rows.append(
                ev.run_episode(
                    args=ns, condition="reset", seed=seed, cycle_index=ci,
                    policy_fn=ev.model_policy(reset), regime=regime,
                )
            )
            print(
                f"[pilot] rho={rho:.4f} cycle={ci + 1}/{len(pilot_seeds)} "
                f"elapsed={perf_counter() - cycle_started:.1f}s",
                flush=True,
            )

    ret_minus_reset = ev.paired_delta(
        rows, "order_level_ret_mean",
        retained_condition="retained", reset_condition="reset",
    )
    ret_minus_frozen = ev.paired_delta(
        rows, "order_level_ret_mean",
        retained_condition="retained", reset_condition="frozen",
    )
    by = {c: [] for c in ("retained", "reset", "frozen")}
    for r in rows:
        if r["condition"] in by:
            by[r["condition"]].append(float(r["order_level_ret_mean"]))
    retained_ret, reset_ret = by["retained"], by["reset"]
    cycles = np.arange(len(retained_ret), dtype=float)
    slope = (
        float(np.polyfit(cycles, retained_ret, 1)[0])
        if len(retained_ret) >= 2 and np.all(np.isfinite(retained_ret))
        else float("nan")
    )
    # Effect-size diagnostic: does the per-cycle retained-reset gap GROW as the
    # retained policy accumulates more block history? (the real effect lever)
    per_cycle_delta = [a - b for a, b in zip(retained_ret, reset_ret)]
    delta_slope = (
        float(np.polyfit(cycles, per_cycle_delta, 1)[0])
        if len(per_cycle_delta) >= 2 and np.all(np.isfinite(per_cycle_delta))
        else float("nan")
    )
    return {
        "rho_disruption": rho,
        "tape_seed": tape_seed,
        "disruption_levels": [p.disruption_level for p in tape.blocks],
        "retained_minus_reset_ret": ret_minus_reset,
        "retained_minus_frozen_ret": ret_minus_frozen,
        "retained_ret_by_cycle": retained_ret,
        "reset_ret_by_cycle": reset_ret,
        "per_cycle_retained_minus_reset": per_cycle_delta,
        "retained_minus_reset_cycle_slope": delta_slope,
        "retained_adaptation_slope": slope,
        "elapsed_seconds": perf_counter() - started,
    }


def cluster_stats(per_tape: list[float]) -> dict[str, object]:
    """Seed-clustered summary: tape seed is the inferential unit."""
    arr = np.array([x for x in per_tape if np.isfinite(x)], dtype=float)
    n = int(arr.size)
    mean = float(arr.mean()) if n else float("nan")
    sem = float(arr.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return {
        "n_tapes": n,
        "mean": mean,
        "sem": sem,
        "ci95_lo": (mean - 1.96 * sem) if n > 1 else float("nan"),
        "ci95_hi": (mean + 1.96 * sem) if n > 1 else float("nan"),
        "per_tape": [float(x) for x in per_tape],
    }


def main() -> int:
    args = build_parser().parse_args()
    rhos = [float(x) for x in args.rhos.split(",") if x.strip()]
    tape_seeds = (
        [int(x) for x in args.regime_seeds.split(",") if x.strip()]
        if args.regime_seeds
        else [args.regime_seed]
    )
    pilot_seeds = [args.pilot_seed_base + i for i in range(args.cycles)]

    per_rho: list[dict[str, object]] = []
    for rho in rhos:
        tape_results = [
            run_one_rho(args, rho, pilot_seeds, tape_seed=ts) for ts in tape_seeds
        ]
        rr = cluster_stats(
            [r["retained_minus_reset_ret"]["mean_delta"] for r in tape_results]
        )
        rf = cluster_stats(
            [r["retained_minus_frozen_ret"]["mean_delta"] for r in tape_results]
        )
        slope = cluster_stats(
            [r["retained_minus_reset_cycle_slope"] for r in tape_results]
        )
        per_rho.append(
            {
                "rho_disruption": rho,
                "tape_seeds": tape_seeds,
                "retained_minus_reset_clustered": rr,
                "retained_minus_frozen_clustered": rf,
                "retained_minus_reset_cycle_slope_clustered": slope,
                "tape_results": tape_results,
            }
        )

    label = args.label or "pilot"
    run_dir = args.output_root / label
    run_dir.mkdir(parents=True, exist_ok=True)
    online_total = sum(
        args.pretrain_timesteps + 2 * args.cycles * args.online_timesteps_per_cycle
        for _ in rhos
    ) * len(tape_seeds)
    payload = {
        "kind": "learning_regime_pilot",
        "training_only": True,
        "algo": args.algo,
        "cycles": args.cycles,
        "online_timesteps_per_cycle": args.online_timesteps_per_cycle,
        "pretrain_timesteps": args.pretrain_timesteps,
        "max_steps": args.max_steps,
        "reward_mode": args.reward_mode,
        "rhos": rhos,
        "tape_seeds": tape_seeds,
        "pilot_seed_base": args.pilot_seed_base,
        "learning_starts": args.learning_starts,
        "buffer_size": args.buffer_size,
        "estimated_dqn_block_timesteps": online_total,
        "rho_demand": "memoryless(1/3)",
        "results": per_rho,
    }
    (run_dir / "pilot.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    n_t = len(tape_seeds)
    print(f"\nLEARNING-REGIME PILOT (training tapes only, {args.algo}, {n_t} tape(s)/rho)")
    print(f"{'rho':>7} {'ret-reset(meanSEM)':>22} {'ret-frozen':>12} {'gap-slope/cyc':>14}")
    rr_means = []
    for r in per_rho:
        rr = r["retained_minus_reset_clustered"]
        rf = r["retained_minus_frozen_clustered"]
        sl = r["retained_minus_reset_cycle_slope_clustered"]
        rr_means.append(rr["mean"])
        sem = rr["sem"]
        sem_s = f"+/-{sem:.4f}" if sem == sem else "(1 tape)"
        print(
            f"{r['rho_disruption']:7.4f} {rr['mean']:+.5f} {sem_s:>10} "
            f"{rf['mean']:+12.5f} {sl['mean']:+14.5f}"
        )
    monotone = all(b >= a for a, b in zip(rr_means, rr_means[1:]))
    any_sig = any(
        (r["retained_minus_reset_clustered"]["ci95_lo"] or float("nan")) > 0
        for r in per_rho
    )
    print(f"\nret-reset clustered means by rho: {[round(x, 4) for x in rr_means]}")
    print(f"monotone non-decreasing in rho? {monotone}")
    print(f"any rho with 95% CI strictly above 0? {any_sig}")
    print(f"Saved: {run_dir / 'pilot.json'}")
    print(
        "Go/no-go (training-tape, NOT paper evidence): ret-reset positive across rho, "
        "and the dose-response trend interpretable above seed-clustered noise."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
