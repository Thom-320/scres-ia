#!/usr/bin/env python3
"""Static Garrido-policy fidelity and risk-stress evaluation.

This script is intentionally static-only. It answers two separate questions:

1. Fidelity: do the thesis static policies behave in this Python DES under the
   same risk rows and thesis-length horizons?
2. Stress extension: if the SAME enabled risks are made more frequent/severe,
   which thesis-static policies remain resilient?

It does not train PPO. RL should be compared later against these static results
under the same risk profile and horizon.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.config import HOURS_PER_WEEK  # noqa: E402
from supply_chain.external_env_interface import (  # noqa: E402
    THESIS_INVENTORY_PERIODS,
    get_episode_terminal_metrics,
    make_dkana_thesis_faithful_env,
)
from supply_chain.thesis_design import (  # noqa: E402
    ThesisDesignSpec,
    design_spec_for_cfi,
    parse_cf_range,
)

DEFAULT_OUTPUT_ROOT = Path("outputs/benchmarks/garrido_static_fidelity_stress")
RISK_PROFILES = (
    "thesis_pattern",
    "current",
    "increased",
    "severe",
    "severe_extended",
    "severe_training",
)
POLICY_SETS = ("minimal", "thesis_static", "with_crossed")

ROW_FIELDS = [
    "profile",
    "policy",
    "policy_kind",
    "action_space_mode",
    "inventory_period_mode",
    "cfi",
    "source_cfi",
    "family",
    "replication",
    "seed",
    "horizon_mode",
    "max_steps_used",
    "reward_total",
    "fill_rate_order_level",
    "order_level_ret_mean",
    "backorder_rate_order_level",
    "pending_backorders_count",
    "pending_backorder_qty",
    "total_demanded",
    "total_delivered",
    "cumulative_disruption_hours",
    "action",
    "common_inventory_period",
    "shifts",
]


def utc_now_label(prefix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{stamp}"


def parse_csv_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def thesis_factorized_action(period: int | None, shifts: int) -> np.ndarray:
    if period is None:
        return np.array([0, shifts - 1], dtype=np.int64)
    return np.array(
        [THESIS_INVENTORY_PERIODS.index(int(period)) + 1, shifts - 1],
        dtype=np.int64,
    )


def thesis_design_action(spec: ThesisDesignSpec) -> np.ndarray:
    period = spec.inventory_replenishment_period
    return thesis_factorized_action(None if period is None else int(period), spec.shifts)


def policy_candidates(policy_set: str) -> list[dict[str, Any]]:
    """Return static policy candidates.

    The thesis-static set contains the isolated policies Garrido used:
    inventory-only with S=1 and capacity-only with I=0. The matched DOE policy
    is per-Cf and is resolved at rollout time.
    """
    if policy_set not in POLICY_SETS:
        raise ValueError(f"Unknown policy_set={policy_set!r}")

    policies: list[dict[str, Any]] = [
        {
            "name": "garrido_matched_DOE_baseline",
            "kind": "matched_doe",
            "period": "matched",
            "shifts": "matched",
            "action": "matched",
        }
    ]

    if policy_set in ("minimal", "thesis_static", "with_crossed"):
        for period in [None, *THESIS_INVENTORY_PERIODS]:
            label = "I0" if period is None else f"I{period}"
            policies.append(
                {
                    "name": f"pure_inventory_{label}_S1",
                    "kind": "pure_inventory",
                    "period": period,
                    "shifts": 1,
                    "action": thesis_factorized_action(period, 1),
                }
            )
        for shifts in (1, 2, 3):
            policies.append(
                {
                    "name": f"pure_capacity_I0_S{shifts}",
                    "kind": "pure_capacity",
                    "period": None,
                    "shifts": shifts,
                    "action": thesis_factorized_action(None, shifts),
                }
            )

    if policy_set == "minimal":
        keep = {
            "garrido_matched_DOE_baseline",
            "pure_inventory_I0_S1",
            "pure_inventory_I672_S1",
            "pure_capacity_I0_S1",
            "pure_capacity_I0_S3",
        }
        return [policy for policy in policies if policy["name"] in keep]

    if policy_set == "with_crossed":
        for period in [None, *THESIS_INVENTORY_PERIODS]:
            label = "I0" if period is None else f"I{period}"
            for shifts in (1, 2, 3):
                policies.append(
                    {
                        "name": f"crossed_uniform_{label}_S{shifts}",
                        "kind": "crossed_uniform",
                        "period": period,
                        "shifts": shifts,
                        "action": thesis_factorized_action(period, shifts),
                    }
                )
    return policies


def risk_kwargs_for_profile(
    *,
    spec: ThesisDesignSpec,
    profile: str,
    thesis_pattern_risk_level: str,
) -> dict[str, Any]:
    """Build risk kwargs while keeping the enabled risk IDs from the thesis row."""
    if profile == "thesis_pattern":
        return {
            "risk_level": thesis_pattern_risk_level,
            "enabled_risks": set(spec.enabled_risks),
            "risk_overrides": dict(spec.risk_overrides),
        }
    if profile not in RISK_PROFILES:
        raise ValueError(f"Unknown risk profile {profile!r}")
    return {
        "risk_level": profile,
        "enabled_risks": set(spec.enabled_risks),
        "risk_overrides": {risk_id: profile for risk_id in spec.enabled_risks},
    }


def max_steps_for_spec(
    spec: ThesisDesignSpec,
    *,
    horizon_mode: str,
    fixed_max_steps: int,
    step_size_hours: float,
) -> int:
    if horizon_mode == "fixed":
        return int(fixed_max_steps)
    if horizon_mode == "thesis":
        return int(math.ceil(float(spec.horizon_hours) / float(step_size_hours)))
    raise ValueError(f"Unknown horizon_mode={horizon_mode!r}")


def rollout(
    *,
    args: argparse.Namespace,
    profile: str,
    spec: ThesisDesignSpec,
    policy: dict[str, Any],
    replication: int,
    seed: int,
) -> dict[str, Any]:
    action = policy["action"]
    if isinstance(action, str) and action == "matched":
        action = thesis_design_action(spec)
    action = np.asarray(action, dtype=np.int64)

    max_steps = max_steps_for_spec(
        spec,
        horizon_mode=args.horizon_mode,
        fixed_max_steps=args.max_steps,
        step_size_hours=args.step_size_hours,
    )
    env_kwargs = {
        "reward_mode": args.reward_mode,
        "observation_version": args.observation_version,
        "observation_mode": args.observation_mode,
        "step_size_hours": args.step_size_hours,
        "max_steps": max_steps,
        "stochastic_pt": args.stochastic_pt,
        "learn_initial_decision": False,
        "action_space_mode": "thesis_factorized",
        "inventory_period_mode": "thesis_strict",
        "initial_action": action,
        "raw_material_flow_mode": args.raw_material_flow_mode,
        "raw_material_order_up_to_multiplier": args.raw_material_order_up_to_multiplier,
    }
    env_kwargs.update(
        risk_kwargs_for_profile(
            spec=spec,
            profile=profile,
            thesis_pattern_risk_level=args.thesis_pattern_risk_level,
        )
    )

    env = make_dkana_thesis_faithful_env(**env_kwargs)
    obs, info = env.reset(seed=seed)
    terminated = truncated = False
    reward_total = 0.0
    steps = 0
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(action)
        reward_total += float(reward)
        if info.get("action_phase") == "weekly_decision":
            steps += 1

    terminal = get_episode_terminal_metrics(env)
    sim = getattr(env.unwrapped, "sim", None)
    pending_orders = getattr(sim, "pending_backorders", []) if sim is not None else []
    row = {
        "profile": profile,
        "policy": policy["name"],
        "policy_kind": policy["kind"],
        "action_space_mode": "thesis_factorized",
        "inventory_period_mode": "thesis_strict",
        "cfi": spec.cfi,
        "source_cfi": spec.source_cfi,
        "family": spec.family,
        "replication": replication,
        "seed": seed,
        "horizon_mode": args.horizon_mode,
        "max_steps_used": max_steps,
        "reward_total": reward_total,
        "fill_rate_order_level": float(terminal["fill_rate_order_level"]),
        "order_level_ret_mean": float(terminal["order_level_ret_mean"]),
        "backorder_rate_order_level": float(terminal["backorder_rate_order_level"]),
        "pending_backorders_count": float(len(pending_orders)),
        "pending_backorder_qty": float(
            sum(float(getattr(order, "remaining_qty", 0.0)) for order in pending_orders)
        ),
        "total_demanded": float(getattr(sim, "total_demanded", 0.0))
        if sim is not None
        else 0.0,
        "total_delivered": float(getattr(sim, "total_delivered", 0.0))
        if sim is not None
        else 0.0,
        "cumulative_disruption_hours": float(
            getattr(
                sim,
                "_cumulative_down_hours",
                getattr(sim, "cumulative_disruption_hours", 0.0),
            )
        )
        if sim is not None
        else 0.0,
        "action": json.dumps(action.astype(int).tolist()),
        "common_inventory_period": ""
        if policy["period"] in (None, "matched")
        else int(policy["period"]),
        "shifts": "" if policy["shifts"] == "matched" else int(policy["shifts"]),
    }
    env.close()
    return row


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return float(np.mean(vals)) if vals else float("nan")


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                str(row["profile"]),
                str(row["family"]),
                str(row["policy"]),
                str(row["policy_kind"]),
            )
        ].append(row)

    summary = []
    for (profile, family, policy, kind), bucket in sorted(groups.items()):
        summary.append(
            {
                "profile": profile,
                "family": family,
                "policy": policy,
                "policy_kind": kind,
                "episode_count": len(bucket),
                "fill_rate_order_level_mean": mean(
                    float(row["fill_rate_order_level"]) for row in bucket
                ),
                "order_level_ret_mean": mean(
                    float(row["order_level_ret_mean"]) for row in bucket
                ),
                "reward_total_mean": mean(float(row["reward_total"]) for row in bucket),
                "pending_backorder_qty_mean": mean(
                    float(row["pending_backorder_qty"]) for row in bucket
                ),
                "cumulative_disruption_hours_mean": mean(
                    float(row["cumulative_disruption_hours"]) for row in bucket
                ),
                "max_steps_used_mean": mean(float(row["max_steps_used"]) for row in bucket),
            }
        )
    return summary


def best_by(
    summary: list[dict[str, Any]],
    *,
    profile: str,
    family: str,
    kinds: set[str],
    metric: str,
) -> dict[str, Any] | None:
    candidates = [
        row
        for row in summary
        if row["profile"] == profile
        and row["family"] == family
        and row["policy_kind"] in kinds
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda row: (row[metric], row["order_level_ret_mean"]))


def summary_row(
    summary: list[dict[str, Any]], *, profile: str, family: str, policy: str
) -> dict[str, Any] | None:
    for row in summary:
        if (
            row["profile"] == profile
            and row["family"] == family
            and row["policy"] == policy
        ):
            return row
    return None


def metric_delta(
    summary: list[dict[str, Any]],
    *,
    profile: str,
    family: str,
    policy_hi: str,
    policy_lo: str,
    metric: str,
) -> float | None:
    hi = summary_row(summary, profile=profile, family=family, policy=policy_hi)
    lo = summary_row(summary, profile=profile, family=family, policy=policy_lo)
    if hi is None or lo is None:
        return None
    return float(hi[metric]) - float(lo[metric])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(out_dir: Path, args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    summary = summarize_rows(rows)
    write_csv(out_dir / "policy_family_summary.csv", summary)

    lines = [
        "# Garrido Static Fidelity and Stress",
        "",
        f"Created UTC: `{datetime.now(timezone.utc).isoformat()}`",
        f"Panel: `{args.panel_cfis}`; profiles: `{','.join(args.profiles)}`; "
        f"policy_set: `{args.policy_set}`; reps: `{args.replications}`.",
        f"Horizon mode: `{args.horizon_mode}`; fixed max_steps: `{args.max_steps}`; "
        f"reward_mode: `{args.reward_mode}`.",
        f"Raw-material flow mode: `{args.raw_material_flow_mode}`.",
        f"Order-up-to multiplier: `{args.raw_material_order_up_to_multiplier}`.",
        "",
        "Risk profile `thesis_pattern` preserves each Cf row's current/increased "
        "risk overrides from the thesis design. Other profiles keep the same "
        "enabled risk IDs but force all of them to that profile.",
        "",
        "## Best Static Policies by Profile and Family",
        "",
        "| profile | family | best pure policy | fill | ReT | pending qty |",
        "|---|---|---|---:|---:|---:|",
    ]
    for profile in args.profiles:
        for family in ("inventory", "capacity", "risk_r1", "risk_r2", "risk_r3"):
            best = best_by(
                summary,
                profile=profile,
                family=family,
                kinds={"matched_doe", "pure_inventory", "pure_capacity"},
                metric="fill_rate_order_level_mean",
            )
            if best is None:
                continue
            lines.append(
                f"| {profile} | {family} | `{best['policy']}` | "
                f"{best['fill_rate_order_level_mean']:.4f} | "
                f"{best['order_level_ret_mean']:.4f} | "
                f"{best['pending_backorder_qty_mean']:.1f} |"
            )

    lines += [
        "",
        "## H1 Risk Degradation Check",
        "",
        "Matched DOE baseline means by risk profile. A monotone decline from "
        "`current` to the declared severe profiles supports the risk-escalation "
        "mechanism before any RL claim.",
        "",
        "| profile | family | fill | ReT | disruption hours |",
        "|---|---|---:|---:|---:|",
    ]
    matched = [
        row
        for row in summary
        if row["policy"] == "garrido_matched_DOE_baseline"
    ]
    for row in sorted(matched, key=lambda r: (r["family"], r["profile"])):
        lines.append(
            f"| {row['profile']} | {row['family']} | "
            f"{row['fill_rate_order_level_mean']:.4f} | "
            f"{row['order_level_ret_mean']:.4f} | "
            f"{row['cumulative_disruption_hours_mean']:.1f} |"
        )

    lines += [
        "",
        "## H2/H3 Static Moderation Smoke",
        "",
        "H2 is probed as `pure_inventory_I672_S1 - pure_inventory_I0_S1`. "
        "H3 is probed as `pure_capacity_I0_S3 - pure_capacity_I0_S1`. "
        "These are direction checks, not final thesis-horizon hypothesis tests.",
        "",
        "| profile | family | I672-I0 fill | I672-I0 ReT | S3-S1 fill | S3-S1 ReT |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for profile in args.profiles:
        for family in ("inventory", "capacity", "risk_r1", "risk_r2", "risk_r3"):
            inv_fill = metric_delta(
                summary,
                profile=profile,
                family=family,
                policy_hi="pure_inventory_I672_S1",
                policy_lo="pure_inventory_I0_S1",
                metric="fill_rate_order_level_mean",
            )
            inv_ret = metric_delta(
                summary,
                profile=profile,
                family=family,
                policy_hi="pure_inventory_I672_S1",
                policy_lo="pure_inventory_I0_S1",
                metric="order_level_ret_mean",
            )
            cap_fill = metric_delta(
                summary,
                profile=profile,
                family=family,
                policy_hi="pure_capacity_I0_S3",
                policy_lo="pure_capacity_I0_S1",
                metric="fill_rate_order_level_mean",
            )
            cap_ret = metric_delta(
                summary,
                profile=profile,
                family=family,
                policy_hi="pure_capacity_I0_S3",
                policy_lo="pure_capacity_I0_S1",
                metric="order_level_ret_mean",
            )
            if None in (inv_fill, inv_ret, cap_fill, cap_ret):
                continue
            lines.append(
                f"| {profile} | {family} | {inv_fill:.4f} | {inv_ret:.4f} | "
                f"{cap_fill:.4f} | {cap_ret:.4f} |"
            )

    lines += [
        "",
        "## Files",
        "",
        "- `episode_metrics.csv`: raw episode rows.",
        "- `policy_family_summary.csv`: profile/family/policy means.",
        "- `manifest.json`: run configuration.",
    ]
    (out_dir / "GARRIDO_STATIC_FIDELITY_STRESS.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--panel-cfis", default="31-90")
    parser.add_argument(
        "--profiles",
        type=parse_csv_list,
        default=["thesis_pattern", "current", "increased", "severe", "severe_extended"],
        help="Comma-separated profiles. Use thesis_pattern for the original Cf overrides.",
    )
    parser.add_argument("--policy-set", choices=POLICY_SETS, default="thesis_static")
    parser.add_argument("--replications", type=int, default=30)
    parser.add_argument("--base-seed", type=int, default=721000)
    parser.add_argument("--reward-mode", default="ReT_thesis")
    parser.add_argument(
        "--raw-material-flow-mode",
        default="legacy_validated",
        choices=(
            "legacy_validated",
            "bom_total_units",
            "bom_total_units_order_up_to",
        ),
    )
    parser.add_argument("--raw-material-order-up-to-multiplier", type=float, default=2.0)
    parser.add_argument("--observation-version", default="v5")
    parser.add_argument("--observation-mode", default="env_sdm_history_reward")
    parser.add_argument("--step-size-hours", type=float, default=float(HOURS_PER_WEEK))
    parser.add_argument("--horizon-mode", choices=("fixed", "thesis"), default="fixed")
    parser.add_argument("--max-steps", type=int, default=260)
    parser.add_argument(
        "--thesis-pattern-risk-level",
        default="increased",
        help="Warm-up/default risk level used with thesis_pattern overrides.",
    )
    parser.add_argument("--stochastic-pt", action="store_true", default=True)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    unknown = set(args.profiles).difference(RISK_PROFILES)
    if unknown:
        raise ValueError(f"Unknown risk profiles: {sorted(unknown)}")

    label = args.label or utc_now_label("garrido_static_fidelity_stress")
    out_dir = args.output_root / label
    out_dir.mkdir(parents=True, exist_ok=False)

    specs = [design_spec_for_cfi(cfi) for cfi in parse_cf_range(args.panel_cfis)]
    policies = policy_candidates(args.policy_set)
    total = len(args.profiles) * len(specs) * len(policies) * args.replications
    rows: list[dict[str, Any]] = []

    csv_path = out_dir / "episode_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS)
        writer.writeheader()
        done = 0
        for profile_index, profile in enumerate(args.profiles):
            for spec in specs:
                for policy_index, policy in enumerate(policies):
                    for replication in range(args.replications):
                        seed = (
                            args.base_seed
                            + profile_index * 10_000_000
                            + spec.cfi * 10_000
                            + policy_index * 100
                            + replication
                        )
                        row = rollout(
                            args=args,
                            profile=profile,
                            spec=spec,
                            policy=policy,
                            replication=replication,
                            seed=seed,
                        )
                        rows.append(row)
                        writer.writerow(row)
                        done += 1
                        if done % max(1, args.progress_every) == 0:
                            handle.flush()
                            print(
                                f"progress {done}/{total} "
                                f"({100.0 * done / total:.1f}%)",
                                flush=True,
                            )
        handle.flush()

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "panel_cfis": args.panel_cfis,
        "profiles": args.profiles,
        "policy_set": args.policy_set,
        "replications": args.replications,
        "base_seed": args.base_seed,
        "reward_mode": args.reward_mode,
        "raw_material_flow_mode": args.raw_material_flow_mode,
        "raw_material_order_up_to_multiplier": args.raw_material_order_up_to_multiplier,
        "horizon_mode": args.horizon_mode,
        "max_steps": args.max_steps,
        "step_size_hours": args.step_size_hours,
        "stochastic_pt": args.stochastic_pt,
        "policy_count": len(policies),
        "episode_count": len(rows),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_report(out_dir, args, rows)
    print(out_dir / "GARRIDO_STATIC_FIDELITY_STRESS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
