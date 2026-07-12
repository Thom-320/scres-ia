#!/usr/bin/env python3
"""Diagnostic: how much of the ReT_excel_delta step reward is retroactive
rescoring of OLD orders vs genuine NEW-order contribution?

Context (2026-07-03): `_compute_ret_excel_delta` (Track A's training reward,
used by `ReT_excel_delta` and `ReT_excel_plus_cvar`) recomputes
`compute_order_level_ret_excel_formula` over ALL orders every step, passing
`current_time=env.now`. Garrido's Excel-faithful formula uses RUNNING
cumulative backorder/unattended counts that depend on `current_time`, so an
order's `ret` value can change retroactively -- purely because time passed
and the cumulative backlog state evolved, not because the agent did
anything new this step. This is faithful to Garrido's own workbook
accounting (verified 0/47546 formula mismatches, 2026-07-03), not a bug --
but it means the PER-STEP TRAINING reward is not purely attributable to the
current action, which could hurt PPO's credit assignment. This script
measures how large that effect actually is, decomposing each step's total
delta into:

    actual_delta_step   = total_now - total_old_prev
    retroactive_step    = total_old_now - total_old_prev   (old orders, revalued at the new time)
    new_contribution    = total_now - total_old_now        (genuinely new orders)

    actual_delta_step == retroactive_step + new_contribution (exact, by construction)

If |retroactive_step| is a small fraction of |actual_delta_step| across an
episode, this hypothesis is not the (main) explanation for Track A's
learning difficulty. If it's large, it's a real, additional, actionable
factor -- distinct from the critic-lag fix already tried.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_track_a_v2_conservation_3d_gate import (  # noqa: E402
    conservation_candidates,
    make_env,
)
from supply_chain.ret_thesis import compute_order_level_ret_excel_formula  # noqa: E402


def decompose_episode(
    *, family: str, phi: float, psi: float, action: tuple[float, ...], seed: int, max_steps: int
) -> list[dict[str, Any]]:
    env = make_env(family=family, phi=phi, psi=psi, max_steps=max_steps, seed=seed)
    obs, _info = env.reset(seed=seed)
    sim = env.unwrapped.sim

    # The warm-up period (~1000h) already places orders before the first real
    # step -- the reset-time order set is NOT empty. Must snapshot the true
    # post-reset baseline, not assume zero, or step 1's "actual_delta" will be
    # wrong by exactly the warm-up orders' total ReT mass.
    old_orders: list[Any] = copy.deepcopy(sim.orders)
    prev_time = float(sim.env.now)
    old_prev_summary = compute_order_level_ret_excel_formula(old_orders, current_time=prev_time)
    total_old_prev = float(old_prev_summary["mean_ret_excel"]) * float(old_prev_summary["n_orders"])

    rows: list[dict[str, Any]] = []
    step = 0
    done = truncated = False
    while not (done or truncated):
        obs, reward, done, truncated, info = env.step(np.asarray(action, dtype=np.float32))
        step += 1
        now = float(sim.env.now)
        new_orders = copy.deepcopy(sim.orders)

        full_now = compute_order_level_ret_excel_formula(new_orders, current_time=now)
        total_now = float(full_now["mean_ret_excel"]) * float(full_now["n_orders"])

        old_now = compute_order_level_ret_excel_formula(old_orders, current_time=now)
        total_old_now = float(old_now["mean_ret_excel"]) * float(old_now["n_orders"]) if old_orders else 0.0

        retroactive_step = total_old_now - total_old_prev
        new_contribution = total_now - total_old_now
        actual_delta = total_now - total_old_prev
        env_reported = float(info.get("ret_excel_components", {}).get("ret_excel_delta_step", np.nan))

        rows.append(
            {
                "step": step,
                "n_orders_old": len(old_orders),
                "n_orders_new": len(new_orders),
                "n_new_orders_this_step": len(new_orders) - len(old_orders),
                "actual_delta": actual_delta,
                "retroactive_step": retroactive_step,
                "new_contribution": new_contribution,
                "reconstruction_gap": actual_delta - (retroactive_step + new_contribution),
                "env_reported_delta": env_reported,
                "env_vs_actual_gap": (
                    env_reported - actual_delta if not np.isnan(env_reported) else float("nan")
                ),
            }
        )

        old_orders = new_orders
        total_old_prev = total_now

    env.close()
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    actual = np.array([r["actual_delta"] for r in rows], dtype=float)
    retro = np.array([r["retroactive_step"] for r in rows], dtype=float)
    new = np.array([r["new_contribution"] for r in rows], dtype=float)
    gap = np.array([r["reconstruction_gap"] for r in rows], dtype=float)
    env_gap = np.array([r["env_vs_actual_gap"] for r in rows], dtype=float)
    denom = np.sum(np.abs(actual)) or 1.0
    return {
        "n_steps": len(rows),
        "sum_abs_actual_delta": float(np.sum(np.abs(actual))),
        "sum_abs_retroactive": float(np.sum(np.abs(retro))),
        "sum_abs_new_contribution": float(np.sum(np.abs(new))),
        "retroactive_share_of_abs_magnitude": float(np.sum(np.abs(retro)) / denom),
        "max_abs_reconstruction_gap": float(np.max(np.abs(gap))) if len(gap) else 0.0,
        "max_abs_env_vs_actual_gap": (
            float(np.nanmax(np.abs(env_gap))) if len(env_gap) and not np.all(np.isnan(env_gap)) else float("nan")
        ),
        "corr_actual_vs_retroactive": (
            float(np.corrcoef(actual, retro)[0, 1]) if np.std(actual) > 0 and np.std(retro) > 0 else float("nan")
        ),
        "steps_where_sign_flipped_by_retroactive": int(
            np.sum(np.sign(new) != np.sign(actual))
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--families", default="R13,R14,R24")
    ap.add_argument("--phis", default="4")
    ap.add_argument("--psi", type=float, default=1.5)
    ap.add_argument("--seeds", default="7100,7101,7102")
    ap.add_argument("--max-steps", type=int, default=52)
    ap.add_argument(
        "--action",
        default="-0.3333,1.0,-0.3333,-0.3333,0.0,0.0",
        help="6D static action to roll out (default: op3 baseline, op9 max, rop baseline, "
        "op5 inert, shift S2 -- a representative 'reasonable policy' action, not random).",
    )
    ap.add_argument(
        "--output", default="outputs/experiments/ret_excel_delta_reward_noise_audit_2026-07-03"
    )
    args = ap.parse_args()

    families = [x.strip() for x in args.families.split(",") if x.strip()]
    phis = [float(x) for x in args.phis.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    action = tuple(float(x) for x in args.action.split(","))

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for family in families:
        for phi in phis:
            for seed in seeds:
                rows = decompose_episode(
                    family=family, phi=phi, psi=args.psi, action=action, seed=seed,
                    max_steps=args.max_steps,
                )
                for row in rows:
                    row.update({"family": family, "phi": phi, "psi": args.psi, "seed": seed})
                all_rows.extend(rows)
                summary = summarize(rows)
                summary.update({"family": family, "phi": phi, "psi": args.psi, "seed": seed})
                summaries.append(summary)
                print(
                    f"{family} phi={phi} seed={seed}: retroactive_share="
                    f"{summary['retroactive_share_of_abs_magnitude']:.4f} "
                    f"max_reconstruction_gap={summary['max_abs_reconstruction_gap']:.2e} "
                    f"max_env_vs_actual_gap={summary['max_abs_env_vs_actual_gap']:.2e}",
                    flush=True,
                )

    with (out / "step_decomposition.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    with (out / "episode_summaries.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    overall_retro_share = float(
        np.mean([s["retroactive_share_of_abs_magnitude"] for s in summaries])
    )
    overall_max_gap = float(max(s["max_abs_reconstruction_gap"] for s in summaries))
    payload = {
        "args": vars(args),
        "n_episodes": len(summaries),
        "mean_retroactive_share_of_abs_magnitude": overall_retro_share,
        "max_reconstruction_gap_across_all_episodes": overall_max_gap,
        "episode_summaries": summaries,
        "interpretation": (
            "retroactive_share close to 0 => retroactive rescoring is negligible, not a "
            "meaningful contributor to Track A's training difficulty. "
            "retroactive_share meaningfully > 0 (e.g. >0.1-0.2) => a real, additional, "
            "actionable source of non-Markovian training-reward noise, distinct from the "
            "critic-lag issue already addressed."
        ),
    }
    (out / "verdict.json").write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nOVERALL mean retroactive share of |reward| magnitude: {overall_retro_share:.4f}")
    print(f"WROTE {out}/verdict.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
