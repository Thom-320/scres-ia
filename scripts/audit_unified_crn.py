#!/usr/bin/env python3
"""Audit whether common seeds behave as strict Common Random Numbers.

The unified ladder uses the same (Cf, replication) seed for every policy. In a
DES, that is only strict CRN if exogenous streams are action-invariant. This
audit runs a small policy panel under shared seeds and compares demand and
disruption summaries. If they differ, the evaluation should be described as a
common-seed paired panel rather than identical realizations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.external_env_interface import make_dkana_thesis_faithful_env
from supply_chain.thesis_design import design_spec_for_cfi, parse_cf_range
from scripts.run_unified_thesis_evaluation import (
    base_kwargs,
    fixed_action_fn,
    per_node_action,
    thesis_design_action,
    thesis_factorized_action,
)

POLICIES = {
    "garrido_matched_DOE_baseline": ("thesis_factorized", "thesis_strict", "matched"),
    "pure_inventory_I672_S1": (
        "thesis_factorized",
        "thesis_strict",
        thesis_factorized_action(672, 1),
    ),
    "crossed_uniform_I504_S3": (
        "thesis_factorized",
        "thesis_strict",
        thesis_factorized_action(504, 3),
    ),
    "per_node_I1344_I504_I504_S3": (
        "factorized",
        "per_node",
        per_node_action(1344, 504, 504, 3),
    ),
}


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def sim_summary(sim: Any) -> dict[str, Any]:
    daily_demand = list(getattr(sim, "daily_demand", []))
    disruption_hours = getattr(sim, "cumulative_disruption_hours", None)
    if isinstance(disruption_hours, dict):
        disruption_payload: Any = {
            key: float(value) for key, value in sorted(disruption_hours.items())
        }
    else:
        disruption_payload = disruption_hours
    return {
        "total_demanded": float(getattr(sim, "total_demanded", math.nan)),
        "daily_demand_len": len(daily_demand),
        "daily_demand_hash": stable_hash(daily_demand),
        "disruption_hours_hash": stable_hash(disruption_payload),
        "disruption_hours": json.dumps(disruption_payload, sort_keys=True, default=str),
    }


def run_policy(
    args: argparse.Namespace, *, cfi: int, rep: int, policy: str
) -> dict[str, Any]:
    spec = design_spec_for_cfi(cfi)
    mode, period_mode, action = POLICIES[policy]
    if isinstance(action, str) and action == "matched":
        action = thesis_design_action(spec)
    kwargs = base_kwargs(args)
    kwargs.update(
        {
            "action_space_mode": mode,
            "inventory_period_mode": period_mode,
            "initial_action": action,
            "enabled_risks": set(spec.enabled_risks),
            "risk_overrides": dict(spec.risk_overrides),
        }
    )
    env = make_dkana_thesis_faithful_env(**kwargs)
    eval_seed = args.base_seed + cfi * 1000 + rep
    obs, info = env.reset(seed=eval_seed)
    terminated = truncated = False
    action_fn = fixed_action_fn(np.asarray(action))
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(action_fn(obs, info))
    sim = getattr(env.unwrapped, "sim", None)
    row = {
        "cfi": cfi,
        "source_cfi": spec.source_cfi,
        "family": spec.family,
        "replication": rep,
        "eval_seed": eval_seed,
        "policy": policy,
    }
    row.update(sim_summary(sim))
    env.close()
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="crn_audit")
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/benchmarks/crn_audit")
    )
    parser.add_argument("--panel-cfis", default="31,61,81")
    parser.add_argument("--replications", type=int, default=2)
    parser.add_argument("--base-seed", type=int, default=730000)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--step-size-hours", type=float, default=168.0)
    parser.add_argument("--reward-mode", default="ReT_cd_v1")
    parser.add_argument("--risk-level", default="increased")
    parser.add_argument("--observation-version", default="v5")
    parser.add_argument("--observation-mode", default="env_sdm_history_reward")
    parser.add_argument("--stochastic-pt", action="store_true", default=True)
    args = parser.parse_args()

    out_dir = args.output_root / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for cfi in parse_cf_range(args.panel_cfis):
        for rep in range(args.replications):
            for policy in POLICIES:
                rows.append(run_policy(args, cfi=cfi, rep=rep, policy=policy))

    write_csv(out_dir / "crn_audit_rows.csv", rows)

    comparison_rows = []
    for (cfi, rep), group in (
        __import__("pandas").DataFrame(rows).groupby(["cfi", "replication"])
    ):
        baseline = group.iloc[0]
        for _, row in group.iterrows():
            comparison_rows.append(
                {
                    "cfi": cfi,
                    "replication": rep,
                    "policy": row["policy"],
                    "demand_hash_matches_baseline": bool(
                        row["daily_demand_hash"] == baseline["daily_demand_hash"]
                    ),
                    "disruption_hash_matches_baseline": bool(
                        row["disruption_hours_hash"]
                        == baseline["disruption_hours_hash"]
                    ),
                    "total_demanded_delta_vs_baseline": float(
                        row["total_demanded"] - baseline["total_demanded"]
                    ),
                }
            )
    write_csv(out_dir / "crn_audit_comparisons.csv", comparison_rows)

    mismatches = [
        row
        for row in comparison_rows
        if not row["demand_hash_matches_baseline"]
        or not row["disruption_hash_matches_baseline"]
    ]
    max_abs_demand_delta = max(
        abs(row["total_demanded_delta_vs_baseline"]) for row in comparison_rows
    )
    verdict = (
        "STRICT_CRN_PASS" if not mismatches else "STRICT_CRN_FAIL_COMMON_SEED_ONLY"
    )
    lines = [
        "# CRN Audit",
        "",
        f"Verdict: `{verdict}`",
        "",
        f"Panel: `{args.panel_cfis}`, replications: `{args.replications}`, max_steps: `{args.max_steps}`.",
        "",
        f"Mismatched policy rows: `{len(mismatches)}` / `{len(comparison_rows)}`.",
        f"Max absolute total-demand delta vs baseline: `{max_abs_demand_delta:.3f}`.",
        "",
        "Interpretation: use 'common-seed paired panel' unless verdict is `STRICT_CRN_PASS`.",
        "",
    ]
    (out_dir / "CRN_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print(out_dir / "CRN_AUDIT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
