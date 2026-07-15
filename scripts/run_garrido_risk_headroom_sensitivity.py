#!/usr/bin/env python3
"""Development-only risk sensitivity using Garrido's exact Table 6.12 profiles.

The output separates physical sensitivity, constant-policy ranking sensitivity, and
profile-tailoring value.  The latter is deliberately named ``H_profile``: it is neither
perfect-information headroom nor observable-policy headroom.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_track_a_headroom_search import Candidate, continuous_candidates
from supply_chain.continuous_its_env import make_continuous_its_track_a_env
from supply_chain.episode_metrics import compute_episode_metrics


R1 = ("R11", "R12", "R13", "R14")
R2 = ("R21", "R22", "R23", "R24")
HIGHER_IS_BETTER = ("ret_excel", "ration_ret_excel", "ret_excel_cvar10")
LOWER_IS_BETTER = (
    "lost_orders",
    "backorder_qty_final",
    "backlog_age_max",
    "service_loss_auc_ration_hours",
    "resource",
)

CF_R1 = {
    "Cf1": "--++", "Cf2": "-+--", "Cf3": "+-++", "Cf4": "+++-", "Cf5": "--+-",
    "Cf6": "++-+", "Cf7": "+--+", "Cf8": "+---", "Cf9": "-+++", "Cf10": "-+-+",
}
CF_R2 = {
    "Cf11": "+-++", "Cf12": "+---", "Cf13": "++-+", "Cf14": "+++-", "Cf15": "--++",
    "Cf16": "-+--", "Cf17": "-++-", "Cf18": "--+-", "Cf19": "-+-+", "Cf20": "++++",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _overrides(risks: tuple[str, ...], signs: str) -> dict[str, str]:
    if len(risks) != len(signs):
        raise ValueError("risk/sign length mismatch")
    return {risk: ("increased" if sign == "+" else "current") for risk, sign in zip(risks, signs)}


def build_profiles() -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = [
        {"id": "R1_current", "group": "R1_frequency", "enabled": R1, "overrides": _overrides(R1, "----"), "impact": {}},
        {"id": "R2_current", "group": "R2_frequency", "enabled": R2, "overrides": _overrides(R2, "----"), "impact": {}},
    ]
    for cf, signs in CF_R1.items():
        profiles.append({"id": cf, "group": "R1_frequency", "enabled": R1, "overrides": _overrides(R1, signs), "impact": {}})
    for cf, signs in CF_R2.items():
        profiles.append({"id": cf, "group": "R2_frequency", "enabled": R2, "overrides": _overrides(R2, signs), "impact": {}})

    # Explicit one-at-a-time profiles make the causal direction readable even though the
    # thesis fractional-factor matrices are the governing frequency design.
    for family, risks in (("R1", R1), ("R2", R2)):
        for risk in risks:
            signs = "".join("+" if r == risk else "-" for r in risks)
            profiles.append({
                "id": f"{family}_OAT_{risk}_increased",
                "group": f"{family}_one_at_a_time",
                "enabled": risks,
                "overrides": _overrides(risks, signs),
                "impact": {},
            })

    # Impact acts only where the implementation has an operational referent: recovery duration
    # (R11/R21/R22/R23) or contingent-demand quantity (R24).
    for risk in ("R11", "R21", "R22", "R23", "R24"):
        family_risks = R1 if risk in R1 else R2
        for psi in (1.0, 1.5, 2.0):
            profiles.append({
                "id": f"impact_{risk}_psi{psi:g}",
                "group": f"impact_{risk}",
                "enabled": family_risks,
                "overrides": _overrides(family_risks, "----"),
                "impact": {risk: psi},
            })

    ids = [profile["id"] for profile in profiles]
    if len(ids) != len(set(ids)):
        raise AssertionError("profile ids must be unique")
    if any("R3" in profile["enabled"] or "R3" in profile["impact"] for profile in profiles):
        raise AssertionError("R3 must remain outside the sensitivity screen")
    return profiles


def make_env(profile: dict[str, Any], *, seed: int, max_steps: int):
    env = make_continuous_its_track_a_env(
        init_frac=0.0,
        reward_mode="ReT_excel_delta",
        observation_version="v6",
        risk_level="current",
        risk_overrides=dict(profile["overrides"]),
        risk_impact_multipliers_by_id=dict(profile["impact"]),
        risk_frequency_multiplier=1.0,
        risk_impact_multiplier=1.0,
        enabled_risks=tuple(profile["enabled"]),
        risk_rng_mode="per_risk",
        stochastic_pt=False,
        max_steps=int(max_steps),
        step_size_hours=168.0,
        risk_obs=True,
        holding_cost=0.0,
        shift_cost=0.001,
    )
    env.reset(seed=int(seed))
    return env


def evaluate(profile: dict[str, Any], candidate: Candidate, *, seed: int, max_steps: int) -> dict[str, float]:
    env = make_env(profile, seed=seed, max_steps=max_steps)
    done = truncated = False
    resources: list[float] = []
    try:
        while not (done or truncated):
            _obs, _reward, done, truncated, info = env.step(np.asarray(candidate.action, dtype=np.float32))
            resources.append(float(info.get("resource_composite", candidate.resource)))
        metrics = compute_episode_metrics(env.unwrapped.sim)
        return {
            "ret_excel": float(metrics["ret_excel"]),
            "ration_ret_excel": float(metrics["ration_ret_excel"]),
            "ret_excel_cvar10": float(metrics["ret_excel_cvar10"]),
            "lost_orders": float(metrics["lost_orders"]),
            "backorder_qty_final": float(metrics["backorder_qty_final"]),
            "backlog_age_max": float(metrics["backlog_age_max"]),
            "service_loss_auc_ration_hours": float(metrics["service_loss_auc_ration_hours"]),
            "resource": float(np.mean(resources)) if resources else float(candidate.resource),
            "risk_events": float(len(env.unwrapped.sim.risk_events)),
        }
    finally:
        env.close()


def mean(rows: list[dict[str, Any]], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows])) if rows else float("nan")


def summarize_group(
    rows: list[dict[str, Any]],
    profiles: list[str],
    candidates: list[Candidate],
    seeds: list[int],
    budget: float,
    *,
    bootstrap_seed: int,
) -> dict[str, Any] | None:
    eligible = [candidate for candidate in candidates if candidate.resource <= budget + 1e-12]
    if len(eligible) < 2:
        return None
    labels = [candidate.label for candidate in eligible]
    by = {(row["profile"], row["candidate"], int(row["seed"])): row for row in rows}
    all_metrics = (*HIGHER_IS_BETTER, *LOWER_IS_BETTER)
    cubes = {
        metric: np.asarray(
            [
                [
                    [float(by[(profile, label, seed)][metric]) for seed in seeds]
                    for label in labels
                ]
                for profile in profiles
            ],
            dtype=float,
        )
        for metric in all_metrics
    }

    def argmax_label(values: np.ndarray, allowed: np.ndarray | None = None) -> int:
        indexes = np.arange(len(labels)) if allowed is None else np.flatnonzero(allowed)
        return max(indexes.tolist(), key=lambda index: (float(values[index]), labels[index]))

    def selection(sample_indexes: np.ndarray) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        means = {metric: cube[:, :, sample_indexes].mean(axis=2) for metric, cube in cubes.items()}
        ret_means = means["ret_excel"]
        robust_index = argmax_label(ret_means.mean(axis=0))
        raw_best_indexes = np.asarray(
            [argmax_label(ret_means[profile_index]) for profile_index in range(len(profiles))],
            dtype=int,
        )
        admissible_mask = np.ones(ret_means.shape, dtype=bool)
        for metric in HIGHER_IS_BETTER[1:]:
            admissible_mask &= means[metric] >= means[metric][:, [robust_index]] - 1e-12
        for metric in LOWER_IS_BETTER:
            admissible_mask &= means[metric] <= means[metric][:, [robust_index]] + 1e-12
        safe_best_indexes = np.asarray(
            [
                argmax_label(ret_means[profile_index], admissible_mask[profile_index])
                for profile_index in range(len(profiles))
            ],
            dtype=int,
        )
        return robust_index, raw_best_indexes, safe_best_indexes, admissible_mask

    full_indexes = np.arange(len(seeds), dtype=int)
    robust_index, raw_best_indexes, safe_best_indexes, admissible_mask = selection(full_indexes)
    robust = labels[robust_index]
    raw_best_by_profile = {
        profile: labels[int(index)] for profile, index in zip(profiles, raw_best_indexes)
    }
    safe_best_by_profile = {
        profile: labels[int(index)] for profile, index in zip(profiles, safe_best_indexes)
    }
    safe_candidates_by_profile = {
        profile: [label for label, allowed in zip(labels, admissible_mask[profile_index]) if allowed]
        for profile_index, profile in enumerate(profiles)
    }
    # The comparator is always admissible against itself, so every safe set is nonempty.
    ret_means = cubes["ret_excel"].mean(axis=2)
    profile_indexes = np.arange(len(profiles), dtype=int)
    raw_oracle = float(ret_means[profile_indexes, raw_best_indexes].mean())
    safe_oracle = float(ret_means[profile_indexes, safe_best_indexes].mean())
    robust_score = float(ret_means[:, robust_index].mean())

    rng = np.random.default_rng(bootstrap_seed)
    boot_raw: list[float] = []
    boot_safe: list[float] = []
    for _ in range(3000):
        sampled_indexes = rng.integers(0, len(seeds), size=len(seeds))
        sampled_robust, sampled_raw, sampled_safe, _ = selection(sampled_indexes)
        sampled_ret = cubes["ret_excel"][:, :, sampled_indexes].mean(axis=2)
        sampled_robust_score = float(sampled_ret[:, sampled_robust].mean())
        boot_raw.append(
            float(sampled_ret[profile_indexes, sampled_raw].mean() - sampled_robust_score)
        )
        boot_safe.append(
            float(sampled_ret[profile_indexes, sampled_safe].mean() - sampled_robust_score)
        )

    selected_rows = [
        by[(profile, safe_best_by_profile[profile], seed)]
        for profile in profiles for seed in seeds
    ]
    robust_rows = [by[(profile, robust, seed)] for profile in profiles for seed in seeds]
    deltas = {
        metric: mean(selected_rows, metric) - mean(robust_rows, metric)
        for metric in (*HIGHER_IS_BETTER, *LOWER_IS_BETTER)
    }
    guardrails = {
        "higher_noninferior": all(deltas[metric] >= -1e-12 for metric in HIGHER_IS_BETTER[1:]),
        "lower_noninferior": all(deltas[metric] <= 1e-12 for metric in LOWER_IS_BETTER[:-1]),
        "resource_non_superior": deltas["resource"] <= 1e-12,
    }
    raw_ci = [
        float(np.percentile(boot_raw, 2.5)),
        float(np.percentile(boot_raw, 97.5)),
    ]
    safe_ci = [
        float(np.percentile(boot_safe, 2.5)),
        float(np.percentile(boot_safe, 97.5)),
    ]
    action_set = sorted(set(safe_best_by_profile.values()))
    passes = bool(
        safe_oracle - robust_score >= 0.01
        and safe_ci[0] > 0.0
        and len(action_set) >= 2
        and all(guardrails.values())
    )
    return {
        "budget_cap": float(budget),
        "n_eligible_candidates": len(eligible),
        "best_robust_constant": robust,
        "raw_best_by_profile": raw_best_by_profile,
        "safe_best_by_profile": safe_best_by_profile,
        "safe_candidate_count_by_profile": {
            profile: len(values) for profile, values in safe_candidates_by_profile.items()
        },
        "unique_profile_optima": action_set,
        "H_profile_raw": raw_oracle - robust_score,
        "H_profile_raw_ci95": raw_ci,
        "H_profile_safe": safe_oracle - robust_score,
        "H_profile_safe_ci95": safe_ci,
        "metric_deltas_selected_minus_robust": deltas,
        "guardrails": guardrails,
        "door_pass": passes,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", default="contracts/garrido_risk_headroom_sensitivity_v1.json")
    parser.add_argument("--output", default="results/garrido_risk_headroom_sensitivity_v1/development")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--profile-prefix", default="", help="optional smoke filter")
    args = parser.parse_args()

    contract_path = Path(args.contract)
    contract = json.loads(contract_path.read_text())
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()] or list(contract["development"]["seeds"])
    max_steps = int(args.max_steps or contract["development"]["max_steps"])
    profiles = build_profiles()
    if args.profile_prefix:
        profiles = [profile for profile in profiles if profile["id"].startswith(args.profile_prefix)]
    candidates = continuous_candidates(
        contract["policy_frontier"]["buffer_fractions_of_I1344"],
        contract["policy_frontier"]["shift_levels"],
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    total = len(profiles) * len(candidates) * len(seeds)
    completed = 0
    for profile, candidate, seed in itertools.product(profiles, candidates, seeds):
        metrics = evaluate(profile, candidate, seed=seed, max_steps=max_steps)
        rows.append({
            "profile": profile["id"],
            "group": profile["group"],
            "enabled_risks": ",".join(profile["enabled"]),
            "risk_overrides": json.dumps(profile["overrides"], sort_keys=True),
            "impact_multipliers": json.dumps(profile["impact"], sort_keys=True),
            "candidate": candidate.label,
            "candidate_resource_nominal": candidate.resource,
            "action": json.dumps(candidate.action),
            "seed": seed,
            **metrics,
        })
        completed += 1
        if completed % max(1, total // 100) == 0 or completed == total:
            print(f"PROGRESS {completed}/{total} ({100.0 * completed / total:.1f}%)", flush=True)

    write_csv(output / "raw_rows.csv", rows)
    profile_ids = [profile["id"] for profile in profiles]
    candidate_labels = [candidate.label for candidate in candidates]
    by_profile_candidate = {
        (profile, candidate): [row for row in rows if row["profile"] == profile and row["candidate"] == candidate]
        for profile in profile_ids for candidate in candidate_labels
    }
    profile_summary: list[dict[str, Any]] = []
    for profile in profile_ids:
        best = max(candidate_labels, key=lambda label: mean(by_profile_candidate[(profile, label)], "ret_excel"))
        best_rows = by_profile_candidate[(profile, best)]
        profile_summary.append({
            "profile": profile,
            "group": next(item["group"] for item in profiles if item["id"] == profile),
            "best_candidate": best,
            **{f"best_{metric}": mean(best_rows, metric) for metric in (*HIGHER_IS_BETTER, *LOWER_IS_BETTER, "risk_events")},
        })
    write_csv(output / "profile_summary.csv", profile_summary)

    group_payload: dict[str, list[dict[str, Any]]] = {}
    groups = sorted(set(profile["group"] for profile in profiles))
    for group_index, group in enumerate(groups):
        members = [profile["id"] for profile in profiles if profile["group"] == group]
        summaries = []
        for budget_index, budget in enumerate(contract["policy_frontier"]["resource_budget_caps"]):
            summary = summarize_group(
                rows, members, candidates, seeds, float(budget),
                bootstrap_seed=2026071500 + 100 * group_index + budget_index,
            )
            if summary is not None:
                summaries.append(summary)
        group_payload[group] = summaries

    passing = [
        {"group": group, **summary}
        for group, summaries in group_payload.items() for summary in summaries
        if summary["door_pass"]
    ]
    result = {
        "schema_version": "garrido_risk_headroom_sensitivity_result_v1",
        "status": "DEVELOPMENT_DOOR_FOUND" if passing else "DEVELOPMENT_NO_DOOR_UNDER_TESTED_FRONTIER",
        "contract_sha256": sha256(contract_path),
        "metric": contract["metric"]["primary"],
        "black_swan_R3_scaled": False,
        "seeds": seeds,
        "max_steps": max_steps,
        "n_profiles": len(profiles),
        "n_candidates": len(candidates),
        "n_evaluations": len(rows),
        "profiles": profiles,
        "candidates": [asdict(candidate) for candidate in candidates],
        "group_budget_summaries": group_payload,
        "passing_doors": passing,
        "claim_boundary": {
            "H_PI_established": False,
            "H_obs_established": False,
            "learner_authorized": False,
            "paper2_confirmed": False,
            "paper3_authorized": False,
        },
    }
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"],
        "n_evaluations": result["n_evaluations"],
        "passing_doors": len(passing),
        "output": str(output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
