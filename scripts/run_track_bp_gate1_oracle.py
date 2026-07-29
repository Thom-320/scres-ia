#!/usr/bin/env python3
"""Track B-P Gate 1: static clock-policy oracle for preventive headroom.

Three no-learning policies on the `track_bp_v1` (11D) contract, CRN-paired
episode for episode:

- never_prepared:    buffer dims 0.0 every step (pure track_b neutral).
- always_prepared:   buffer dims 1.0 every step (Garrido Scenario II held
                     constant — the "strategic inventory reserves" posture).
- calendar_prepared: buffer dims 1.0 only during the `prep_window_weeks`
                     before each known exposure window (deterministic
                     supply-chain calendar: Op1 contracting rounds every
                     `calendar_cycle_weeks`), 0.0 otherwise.

All three keep the 8 track_b dims at the neutral medium (zeros), so the ONLY
difference between arms is buffer posture and its timing. Deltas of interest:

- always - never:    static preventive headroom (does holding buffers help?)
- calendar - never:  timing headroom (does anticipating the calendar help?)
- always - calendar: cost of blanket vs timed preparation (holding exposure
                     proxied by mean buffer fraction; holding is not priced
                     in reward, so ReT parity at lower holding favors timed).

Gate rule: prevention reopens only if `always - never` or `calendar - never`
is positive with a CI excluding zero on episode-level Excel ReT.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_track_b_smoke import (  # noqa: E402
    build_env_kwargs,
    build_parser as smoke_build_parser,
    save_csv,
)
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.track_bp_env import TRACK_BP_ACTION_DIM, make_track_bp_env  # noqa: E402

EVAL_EPISODE_SEED_OFFSET = 50_000


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=24)
    p.add_argument("--seed-base", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=104)
    p.add_argument("--enabled-risks", default="R11,R21,R23,R24")
    p.add_argument("--risk-frequency-by-id", default=None)
    p.add_argument("--risk-impact-by-id", default=None)
    p.add_argument("--risk-level", default="current")
    p.add_argument("--replenishment-lead-time", type=float, default=168.0)
    p.add_argument("--prep-window-weeks", type=int, default=4,
                   help="Weeks of raised buffers before each calendar exposure window.")
    p.add_argument("--calendar-cycle-weeks", type=int, default=24,
                   help="Deterministic exposure cadence in weeks (Op1 contracting = 24; "
                        "Op2 monthly deliveries = 4).")
    p.add_argument("--step-size-hours", type=float, default=168.0)
    return p


def build_args(cli: argparse.Namespace) -> argparse.Namespace:
    args = smoke_build_parser().parse_args([])
    args.risk_level = cli.risk_level
    args.faithful = True
    args.reward_mode = "control_v1"
    args.max_steps = cli.max_steps
    args.enabled_risks = cli.enabled_risks
    args.risk_frequency_by_id = cli.risk_frequency_by_id or None
    args.risk_impact_by_id = cli.risk_impact_by_id or None
    args.inventory_replenishment_lead_time = float(cli.replenishment_lead_time)
    return args


def buffer_schedule(policy: str, step: int, cli: argparse.Namespace) -> float:
    if policy == "never_prepared":
        return 0.0
    if policy == "always_prepared":
        return 1.0
    if policy == "calendar_prepared":
        cycle = int(cli.calendar_cycle_weeks)
        # Exposure windows open at steps that are multiples of `cycle`
        # (Op1 rounds fire every op1_rop = 4032h = 24 weekly steps).
        steps_until_round = (-step) % cycle
        # Raise buffers early enough for the lead time to land before the
        # round: within prep_window of the round, counting the lead.
        lead_steps = int(np.ceil(cli.replenishment_lead_time / cli.step_size_hours))
        window = int(cli.prep_window_weeks) + lead_steps
        return 1.0 if 0 < steps_until_round <= window else 0.0
    raise ValueError(f"unknown policy {policy!r}")


def run_episode(policy: str, args: argparse.Namespace, cli: argparse.Namespace,
                eval_seed: int) -> dict[str, Any]:
    kwargs = build_env_kwargs(args)
    lead = kwargs.pop("inventory_replenishment_lead_time", 168.0)
    env = make_track_bp_env(inventory_replenishment_lead_time=lead, **kwargs)
    obs, _ = env.reset(seed=eval_seed)
    terminated = truncated = False
    step = 0
    buffer_fracs: list[float] = []
    while not (terminated or truncated):
        action = np.zeros(TRACK_BP_ACTION_DIM, dtype=np.float32)
        frac = buffer_schedule(policy, step, cli)
        action[8:] = frac
        buffer_fracs.append(frac)
        obs, _r, terminated, truncated, info = env.step(action)
        step += 1
    metrics = compute_episode_metrics(env.unwrapped.sim)
    row = {
        "policy": policy,
        "eval_seed": eval_seed,
        "ret_excel": float(metrics["ret_excel"]),
        "order_ret_excel_mean": float(metrics.get("order_ret_excel_mean", 0.0)),
        "fill_rate": float(metrics.get("fill_rate", 0.0)),
        "mean_buffer_frac": float(np.mean(buffer_fracs)) if buffer_fracs else 0.0,
    }
    env.close()
    return row


def bootstrap_ci(deltas: np.ndarray, iters: int = 10_000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(deltas)
    means = np.array([
        float(np.mean(deltas[rng.integers(0, n, size=n)])) for _ in range(iters)
    ])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


POLICIES = ("never_prepared", "always_prepared", "calendar_prepared")


def main() -> None:
    cli = build_parser().parse_args()
    out = cli.output_dir
    out.mkdir(parents=True, exist_ok=True)
    args = build_args(cli)

    rows: list[dict[str, Any]] = []
    for episode_idx in range(cli.episodes):
        eval_seed = cli.seed_base + EVAL_EPISODE_SEED_OFFSET + episode_idx
        for policy in POLICIES:
            row = run_episode(policy, args, cli, eval_seed)
            row["episode"] = episode_idx + 1
            rows.append(row)
        print(f"episode {episode_idx + 1}/{cli.episodes} done", flush=True)

    save_csv(out / "gate1_rows.csv", rows)

    by_policy: dict[str, dict[int, dict[str, Any]]] = {p: {} for p in POLICIES}
    for r in rows:
        by_policy[r["policy"]][r["eval_seed"]] = r

    def paired_delta(a: str, b: str, key: str) -> np.ndarray:
        seeds = sorted(set(by_policy[a]) & set(by_policy[b]))
        return np.array([
            by_policy[a][s][key] - by_policy[b][s][key] for s in seeds
        ])

    summary: dict[str, Any] = {
        "config": {k: str(v) for k, v in vars(cli).items()},
        "resolved_env_kwargs": {
            k: repr(v) for k, v in sorted(build_env_kwargs(args).items())
        },
        "per_policy_mean": {
            p: {
                k: float(np.mean([r[k] for r in by_policy[p].values()]))
                for k in ("ret_excel", "order_ret_excel_mean", "fill_rate", "mean_buffer_frac")
            }
            for p in POLICIES
        },
    }
    for a, b, label in (
        ("always_prepared", "never_prepared", "always_minus_never"),
        ("calendar_prepared", "never_prepared", "calendar_minus_never"),
        ("always_prepared", "calendar_prepared", "always_minus_calendar"),
    ):
        for key in ("ret_excel", "order_ret_excel_mean"):
            deltas = paired_delta(a, b, key)
            lo, hi = bootstrap_ci(deltas)
            summary[f"{label}__{key}"] = {
                "mean": float(np.mean(deltas)),
                "ci95": [lo, hi],
                "n": int(len(deltas)),
                "positive": int(np.sum(deltas > 0)),
            }

    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
