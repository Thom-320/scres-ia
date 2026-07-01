#!/usr/bin/env python3
"""Joint 6x3 (inventory x shift) Cobb-Douglas screen across risk regimes.

The marginal screen in ``screen_garrido_cost_metrics.py`` varied inventory only at
S1 and shifts only at I0, so it could not see the JOINT optimum that RL actually
optimises, and reported a regime flip that rested on 3 seeds. This runner evaluates
the FULL joint grid (6 inventory x 3 shifts = 18 configs) under the cost-augmented
Garrido-2024 index, across current/increased/severe, with paired CRN seeds, then
tests two gate criteria robustly:

  #1 interior  : joint argmax is not a buffer/shift corner.
  #2 regime-dep: joint argmax (esp. the SHIFT choice) moves with regime, and the
                 flip survives a paired-CRN CI (not seed noise).

No RL is trained. This is the cheap gate before any Track-A PPO run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

from screen_garrido_cost_metrics import ScreenConfig, _run_garrido2024_episode

INVENTORY_LEVELS = [0, 168, 336, 504, 672, 1344]
SHIFT_LEVELS = [1, 2, 3]
FAITHFUL_RAW_MODE = "kit_equivalent_order_up_to"
FAITHFUL_RAW_MULT = 2.0


def joint_descriptor(shifts: int, inventory: int) -> dict[str, Any]:
    """A full-grid (S, I) static policy descriptor for _run_garrido2024_episode."""
    profile = "I0" if inventory == 0 else f"I{inventory}"
    period = "" if inventory == 0 else str(inventory)
    return {
        "policy": f"joint_S{shifts}_I{inventory}",
        "policy_kind": "joint",
        "shifts": shifts,
        "inventory_period": period,
        "initial_buffer_profile": profile,
        "raw_material_flow_mode": FAITHFUL_RAW_MODE,
        "raw_material_order_up_to_multiplier": FAITHFUL_RAW_MULT,
    }


def _is_corner(shifts: int, inventory: int) -> bool:
    return shifts in (SHIFT_LEVELS[0], SHIFT_LEVELS[-1]) and inventory in (
        INVENTORY_LEVELS[0],
        INVENTORY_LEVELS[-1],
    )


def paired_diff_ci(a: dict[int, float], b: dict[int, float]) -> dict[str, float]:
    """Paired-CRN 95% CI of (a - b) over shared seeds."""
    seeds = sorted(set(a) & set(b))
    diffs = [a[s] - b[s] for s in seeds]
    n = len(diffs)
    mean = statistics.mean(diffs)
    sd = statistics.pstdev(diffs) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 0 else 0.0
    return {
        "mean": mean,
        "lo": mean - 1.96 * se,
        "hi": mean + 1.96 * se,
        "n": n,
        "robust": (mean - 1.96 * se) > 0.0 or (mean + 1.96 * se) < 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--regimes", type=str, default="current,increased,severe")
    ap.add_argument("--max-steps", type=int, default=104)  # ~2 years at 168h/step
    ap.add_argument("--step-hours", type=float, default=168.0)
    ap.add_argument("--shift-cost", type=float, default=0.5)
    ap.add_argument("--kappa-frac", type=float, default=0.20)
    ap.add_argument("--stochastic-pt", action="store_true", default=True)
    ap.add_argument(
        "--output",
        type=str,
        default="outputs/experiments/garrido2024_joint_grid_2026-06-26",
    )
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
    import dataclasses
    cfg = dataclasses.replace(
        ScreenConfig(),
        cd_max_steps=int(args.max_steps),
        cd_step_size_hours=float(args.step_hours),
        cd_stochastic_pt=bool(args.stochastic_pt),
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    descriptors = [
        joint_descriptor(s, i) for i in INVENTORY_LEVELS for s in SHIFT_LEVELS
    ]
    total = len(descriptors) * len(regimes) * len(seeds)
    done = 0
    for desc in descriptors:
        for regime in regimes:
            for seed in seeds:
                row = _run_garrido2024_episode(
                    desc,
                    risk_level=regime,
                    seed=seed,
                    kappa_train_frac=float(args.kappa_frac),
                    shift_cost=float(args.shift_cost),
                    risk_frequency_multiplier=1.0,
                    risk_impact_multiplier=1.0,
                    config=cfg,
                )
                row["shifts_level"] = int(desc["shifts"])
                row["inventory_level"] = int(
                    0 if desc["inventory_period"] == "" else int(desc["inventory_period"])
                )
                rows.append(row)
                done += 1
                if done % 30 == 0:
                    print(f"  {done}/{total} episodes", flush=True)

    # write rows
    rows_path = out_dir / "joint_rows.csv"
    with rows_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # per-(regime, config) means + per-seed maps
    key = "ret_garrido2024_sigmoid_total"

    def cell(regime: str, s: int, i: int) -> dict[int, float]:
        return {
            r["seed"]: float(r[key])
            for r in rows
            if r["risk_level"] == regime
            and r["shifts_level"] == s
            and r["inventory_level"] == i
        }

    summary: dict[str, Any] = {
        "description": "Joint 6x3 Cobb-Douglas screen across regimes (paired CRN).",
        "seeds": seeds,
        "regimes": regimes,
        "max_steps": cfg.cd_max_steps,
        "step_hours": cfg.cd_step_size_hours,
        "shift_cost": args.shift_cost,
        "kappa_frac": args.kappa_frac,
        "metric": key,
        "per_regime": {},
    }

    regime_argmax: dict[str, tuple[int, int]] = {}
    for regime in regimes:
        grid = {}
        for i in INVENTORY_LEVELS:
            for s in SHIFT_LEVELS:
                vals = cell(regime, s, i)
                grid[(s, i)] = statistics.mean(vals.values()) if vals else float("nan")
        (s_star, i_star), best = max(grid.items(), key=lambda kv: kv[1])
        regime_argmax[regime] = (s_star, i_star)
        # full surface as nested table for the report
        surf = {
            f"I{i}": {f"S{s}": round(grid[(s, i)], 3) for s in SHIFT_LEVELS}
            for i in INVENTORY_LEVELS
        }
        summary["per_regime"][regime] = {
            "argmax_shift": s_star,
            "argmax_inventory": i_star,
            "argmax_score": round(best, 3),
            "argmax_is_corner": _is_corner(s_star, i_star),
            "best_shift_marginal_at_argmax_inv": {
                f"S{s}": round(grid[(s, i_star)], 3) for s in SHIFT_LEVELS
            },
            "surface": surf,
        }

    # criterion #1: all regime argmaxes non-corner
    summary["criterion_1_interior"] = not any(
        _is_corner(*regime_argmax[r]) for r in regimes
    )

    # criterion #2: regime-dependent argmax + robustness of each flip vs current
    base = regimes[0]
    base_argmax = regime_argmax[base]
    flips = {}
    shift_moves = False
    for regime in regimes[1:]:
        ra = regime_argmax[regime]
        if ra != base_argmax:
            # paired CI: argmax_regime vs base_argmax, evaluated UNDER `regime`
            a = cell(regime, ra[0], ra[1])
            b = cell(regime, base_argmax[0], base_argmax[1])
            ci = paired_diff_ci(a, b)
            flips[regime] = {
                "from": f"S{base_argmax[0]}_I{base_argmax[1]}",
                "to": f"S{ra[0]}_I{ra[1]}",
                "shift_changed": ra[0] != base_argmax[0],
                "paired_ci_vs_base_argmax": {k: round(v, 4) if isinstance(v, float) else v for k, v in ci.items()},
            }
            if ra[0] != base_argmax[0]:
                shift_moves = True
    summary["criterion_2_regime_dependent"] = len(flips) > 0
    summary["criterion_2_robust_flip"] = any(f["paired_ci_vs_base_argmax"]["robust"] for f in flips.values())
    summary["shift_optimum_moves_with_regime"] = shift_moves
    summary["regime_argmax"] = {r: f"S{regime_argmax[r][0]}_I{regime_argmax[r][1]}" for r in regimes}
    summary["flips"] = flips
    summary["green_light_for_rl"] = bool(
        summary["criterion_1_interior"]
        and summary["criterion_2_regime_dependent"]
        and summary["criterion_2_robust_flip"]
    )

    sum_path = out_dir / "summary.json"
    sum_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWROTE {sum_path}")
    print(json.dumps({k: summary[k] for k in [
        "regime_argmax", "criterion_1_interior", "criterion_2_regime_dependent",
        "criterion_2_robust_flip", "shift_optimum_moves_with_regime", "green_light_for_rl",
    ]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
