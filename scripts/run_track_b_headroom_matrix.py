#!/usr/bin/env python3
"""Static-first headroom matrix for Track B environment tuning.

The goal is not to "turn risks up until PPO wins"; it is to identify stress
families where a regime-specific static oracle beats the best single constant.
Only those cells deserve PPO compute.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_track_b_independent_doe import (  # noqa: E402
    Policy,
    parse_float_list,
    parse_int_list,
    run_episode,
)


RISK_FAMILIES: dict[str, tuple[str, ...] | None] = {
    "all": None,
    "R1": ("R11", "R12", "R13", "R14"),
    "R2": ("R21", "R22", "R23", "R24"),
    "R22_R24": ("R22", "R24"),
    "R24": ("R24",),
    "mixed_no_R14": ("R11", "R12", "R13", "R21", "R22", "R23", "R24", "R3"),
}


def family_arg(name: str) -> str | None:
    risks = RISK_FAMILIES[name]
    return None if risks is None else ",".join(risks)


def mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--families", default="R2,R24,mixed_no_R14,all")
    ap.add_argument("--risk-levels", default="adaptive_benchmark_v2")
    ap.add_argument("--phis", default="1,2,4")
    ap.add_argument("--psis", default="1.0,1.5,2.0")
    ap.add_argument("--demand-mults", default="1.0")
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--max-steps", type=int, default=52)
    ap.add_argument("--step-size-hours", type=float, default=168.0)
    ap.add_argument("--risk-level", default="adaptive_benchmark_v2")
    ap.add_argument("--reward-mode", default="control_v1")
    ap.add_argument("--shifts", default="1,2,3")
    ap.add_argument("--op9-mults", default="1.0")
    ap.add_argument("--op10-mults", default="0.5,1.0,1.5,2.0")
    ap.add_argument("--op12-mults", default="0.5,1.0,1.5,2.0")
    ap.add_argument("--promote-threshold", type=float, default=0.0001)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    output_dir = args.output_dir or (
        Path("outputs/experiments")
        / f"track_b_headroom_matrix_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    families = [f.strip() for f in args.families.split(",") if f.strip()]
    unknown = sorted(set(families) - set(RISK_FAMILIES))
    if unknown:
        raise ValueError(f"Unknown risk families: {unknown}. Known={sorted(RISK_FAMILIES)}")
    phis = parse_float_list(args.phis)
    psis = parse_float_list(args.psis)
    risk_levels = [r.strip() for r in args.risk_levels.split(",") if r.strip()]
    demand_mults = parse_float_list(args.demand_mults)
    seeds = parse_int_list(args.seeds)
    policies = [
        Policy(shift=s, op9_mult=o9, op10_mult=o10, op12_mult=o12)
        for s in parse_int_list(args.shifts)
        for o9 in parse_float_list(args.op9_mults)
        for o10 in parse_float_list(args.op10_mults)
        for o12 in parse_float_list(args.op12_mults)
    ]
    print(
        f"Track B headroom matrix: {len(risk_levels)} risk levels × {len(families)} families × "
        f"{len(phis)} phi × {len(psis)} psi × {len(demand_mults)} demand × "
        f"{len(policies)} policies × {len(seeds)} seeds",
        flush=True,
    )

    seed_rows: list[dict[str, Any]] = []
    cell_policy_rows: list[dict[str, Any]] = []
    for risk_level in risk_levels:
        for family in families:
            for phi in phis:
                for psi in psis:
                    for demand_mult in demand_mults:
                        cell_key = (
                            f"{risk_level}_{family}_phi{phi:g}_psi{psi:g}_dm{demand_mult:g}"
                        )
                        for policy in policies:
                            bucket: list[dict[str, Any]] = []
                            for seed in seeds:
                                cell_args = argparse.Namespace(**vars(args))
                                cell_args.risk_level = risk_level
                                cell_args.enabled_risks = family_arg(family)
                                cell_args.risk_frequency_multiplier = float(phi)
                                cell_args.risk_impact_multiplier = float(psi)
                                cell_args.demand_mean_multiplier = float(demand_mult)
                                row = run_episode(policy, seed=seed, args=cell_args)
                                row.update(
                                    {
                                        "risk_level": risk_level,
                                        "family": family,
                                        "phi": phi,
                                        "psi": psi,
                                        "demand_mult": demand_mult,
                                        "cell": cell_key,
                                    }
                                )
                                seed_rows.append(row)
                                bucket.append(row)
                            cell_policy_rows.append(
                                {
                                    "risk_level": risk_level,
                                    "family": family,
                                    "phi": phi,
                                    "psi": psi,
                                    "demand_mult": demand_mult,
                                    "cell": cell_key,
                                    "policy": policy.label,
                                    "shift": policy.shift,
                                    "op9_mult": policy.op9_mult,
                                    "op10_mult": policy.op10_mult,
                                    "op12_mult": policy.op12_mult,
                                    "ret_excel_mean": mean([float(r["ret_excel"]) for r in bucket]),
                                    "flow_fill_rate_mean": mean([float(r["flow_fill_rate"]) for r in bucket]),
                                    "service_loss_auc_per_order_mean": mean(
                                        [float(r["service_loss_auc_per_order"]) for r in bucket]
                                    ),
                                    "lost_rate_mean": mean([float(r["lost_rate"]) for r in bucket]),
                                }
                            )
                        best = max(
                            (r for r in cell_policy_rows if r["cell"] == cell_key),
                            key=lambda r: float(r["ret_excel_mean"]),
                        )
                        print(
                            f"  {cell_key}: best={best['policy']} "
                            f"ret={float(best['ret_excel_mean']):.6f} "
                            f"flow={float(best['flow_fill_rate_mean']):.3f}",
                            flush=True,
                        )

    family_summaries: list[dict[str, Any]] = []
    for risk_level in risk_levels:
        for family in families:
            cells = sorted(
                {
                    r["cell"]
                    for r in cell_policy_rows
                    if r["risk_level"] == risk_level and r["family"] == family
                }
            )
            if not cells:
                continue
            cell_best: dict[str, dict[str, Any]] = {}
            for cell in cells:
                rows = [r for r in cell_policy_rows if r["cell"] == cell]
                cell_best[cell] = max(rows, key=lambda r: float(r["ret_excel_mean"]))
            policies_for_family = sorted(
                {
                    r["policy"]
                    for r in cell_policy_rows
                    if r["risk_level"] == risk_level and r["family"] == family
                }
            )
            const_scores = {
                policy: mean(
                    [
                        float(r["ret_excel_mean"])
                        for r in cell_policy_rows
                        if r["risk_level"] == risk_level
                        and r["family"] == family
                        and r["policy"] == policy
                    ]
                )
                for policy in policies_for_family
            }
            best_const_policy = max(const_scores, key=const_scores.get)
            best_const_score = const_scores[best_const_policy]
            oracle_score = mean([float(row["ret_excel_mean"]) for row in cell_best.values()])
            actions = {
                (
                    str(row["policy"]),
                    int(row["shift"]),
                    float(row["op9_mult"]),
                    float(row["op10_mult"]),
                    float(row["op12_mult"]),
                )
                for row in cell_best.values()
            }
            headroom = oracle_score - best_const_score
            verdict = (
                "PROMOTE"
                if headroom > float(args.promote_threshold) and len(actions) > 1
                else "NULL"
            )
            family_summaries.append(
                {
                    "risk_level": risk_level,
                    "family": family,
                    "n_cells": len(cells),
                    "oracle_ret_excel": oracle_score,
                    "best_single_constant_policy": best_const_policy,
                    "best_single_constant_ret_excel": best_const_score,
                    "oracle_minus_best_constant": headroom,
                    "best_actions_vary": len(actions) > 1,
                    "n_distinct_best_actions": len(actions),
                    "verdict": verdict,
                }
            )
    write_csv(output_dir / "seed_metrics.csv", seed_rows)
    write_csv(output_dir / "cell_policy_summary.csv", cell_policy_rows)
    write_csv(output_dir / "family_headroom_summary.csv", family_summaries)
    config = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "risk_levels": risk_levels,
        "families": families,
        "family_headroom_summary": family_summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"WROTE {output_dir}", flush=True)
    for row in family_summaries:
        print(
            "{risk_level}/{family}: headroom={h:+.6f} vary={v} verdict={verdict}".format(
                risk_level=row["risk_level"],
                family=row["family"],
                h=float(row["oracle_minus_best_constant"]),
                v=row["best_actions_vary"],
                verdict=row["verdict"],
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
