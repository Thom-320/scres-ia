#!/usr/bin/env python3
"""Deterministic design-consistency preflight for Program X / O-Scale.

The eight checks below validate only combinatorics and explicit design
boundaries.  They open no seed, simulate no stochastic tape, establish no N=2
physical parity, fit no policy and make no headroom or neural-premium claim.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from math import comb
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONTRACT = ROOT / "contracts/program_x_o_scale_amortized_control_v1.json"
DEFAULT_OUTPUT = ROOT / "results/program_x/o_scale_design_preflight_v1/result.json"


def weak_compositions(total: int, parts: int) -> Iterator[tuple[int, ...]]:
    """Yield every ordered allocation of ``total`` identical batches."""

    if total < 0:
        raise ValueError("total must be nonnegative")
    if parts < 1:
        raise ValueError("parts must be positive")
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for suffix in weak_compositions(total - first, parts - 1):
            yield (first, *suffix)


def action_count(n_products: int, batches: int) -> int:
    if n_products < 1 or batches < 0:
        raise ValueError("invalid action-space dimensions")
    return comb(n_products + batches - 1, batches)


def calendar_count(n_products: int, batches: int, horizon: int) -> int:
    if horizon < 1:
        raise ValueError("horizon must be positive")
    return action_count(n_products, batches) ** horizon


def _check(passed: bool, observed: Any, expected: Any, why: str) -> dict[str, Any]:
    return {
        "computed": True,
        "passed": bool(passed),
        "evidence": {
            "observed": observed,
            "expected": expected,
            "why_it_can_fail": why,
        },
    }


def _ordered_substrings(values: list[str], substrings: tuple[str, ...]) -> bool:
    """Return whether each expected substring occurs in the matching position."""

    return len(values) == len(substrings) and all(
        expected.lower() in observed.lower()
        for observed, expected in zip(values, substrings, strict=True)
    )


def build_payload(contract: dict[str, Any]) -> dict[str, Any]:
    mechanism = contract["mechanism"]
    demand = contract["demand_and_information"]
    learner = contract["learner_ladder"]
    metrics = contract["metric_panel"]
    thresholds = contract["prospective_thresholds"]
    branches = contract["authorization_branches"]
    planners = contract["planner_roles"]
    execution = contract["execution_policy"]

    scales = [int(value) for value in mechanism["product_scales"]]
    batches = int(mechanism["batches_per_week"])
    horizon = int(mechanism["decision_weeks"])
    expected_actions = {
        int(key): int(value)
        for key, value in mechanism["expected_action_counts"].items()
    }
    expected_calendars = {
        int(key): int(value)
        for key, value in mechanism["expected_open_loop_calendar_counts"].items()
    }

    enumerated = {n: list(weak_compositions(batches, n)) for n in scales}
    computed_actions = {n: action_count(n, batches) for n in scales}
    computed_calendars = {
        n: calendar_count(n, batches, horizon) for n in scales
    }
    unique_actions = {
        n: len(set(actions)) == len(actions) for n, actions in enumerated.items()
    }
    conserved_batches = {
        n: all(sum(action) == batches for action in actions)
        for n, actions in enumerated.items()
    }

    decoder = mechanism["action_decoder"]
    anchor = demand["n2_program_o_anchor"]
    latent = demand["latent_regime"]
    transition = latent["transition_kernel"]
    warning = demand["warning_kernel"]
    primary_metric = metrics["primary_physical_endpoint"]
    architecture = learner["primary_architecture"]

    event_order_ok = _ordered_substrings(
        demand["causal_event_order"],
        (
            "clone the predecision physical state and RNG state",
            "at t=0 draw Z_0 from the prior",
            "emit the current warning",
            "strictly half-open observation",
            "choose and lock the three-batch allocation",
            "realize current demand",
            "update the posterior from realized demand and transition",
        ),
    )

    architecture_boundaries = {
        "policy_equivariant": (
            architecture["policy_symmetry"]
            == "permutation-equivariant over product labels"
        ),
        "value_invariant": (
            architecture["value_symmetry"]
            == "permutation-invariant over product labels"
        ),
        "decoder_bound": architecture["decoder"] == "mechanism.action_decoder",
        "imitation_not_quality": (
            "cannot establish quality superiority"
            in learner["imitation_claim_boundary"]
        ),
        "amortization_does_not_require_hret": any(
            "H_ret is not required" in row
            for row in branches["amortization"]
        ),
        "gru_has_conditional_history_gate": any(
            "conditional-history" in row
            for row in branches["recurrent_representation"]
        ),
        "quality_requires_structured_residual": any(
            "observable headroom" in row
            for row in branches["quality_residual_rl"]
        ),
        "teacher_is_high_budget": (
            "natural online cost is measured"
            in planners["teacher_high_budget"]
        ),
    }

    amortization_claim = contract["claim_tiers"]["T2_neural_amortization"]
    metric_compute_boundaries = {
        "primary_higher_is_better": (
            primary_metric["direction"] == "higher_is_better"
            and primary_metric["range"] == [0.0, 1.0]
        ),
        "secondary_report_only": (
            metrics["secondary_status"]
            == (
                "PRESPECIFIED_REPORT_ONLY_CANNOT_PROMOTE_RESCUE_OR_BLOCK_"
                "THE_PRIMARY_PHYSICAL_CLAIM"
            )
        ),
        "thresholds_unopened": all(
            thresholds[key] is None
            for key in (
                "quality_sesoi",
                "noninferiority_margin",
                "operational_sla_seconds",
                "planner_break_even_query_count",
            )
        ),
        "absolute_sla_required": "absolute operational SLA" in amortization_claim,
        "p95_ratio_required": "10x lower" in amortization_claim,
        "des_calls_cannot_substitute": (
            "cannot replace either latency condition" in amortization_claim
            and " or DES calls" not in amortization_claim
        ),
    }

    resource_boundaries = {
        "all_enumerated_actions_conserve_batches": all(conserved_batches.values()),
        "decoder_nonnegative_integer": decoder["nonnegative_integer"] is True,
        "decoder_exact_sum": decoder["sum_equals_batches_per_week"] is True,
        "actual_use_is_endogenous": {
            "actual loaded departures",
            "actual payload",
            "actual vehicle-hours",
            "actual line and transport utilization",
        }.issubset(set(mechanism["endogenous_resource_outcomes"])),
        "entitlements_not_use_are_invariant": (
            "Actual utilization is an outcome"
            in mechanism["resource_interpretation"]
        ),
    }

    hmm_boundaries = {
        "uniform_prior": latent["initial_prior"] == "P(Z_0=i)=1/N for every i",
        "symmetric_transition_stay": transition["same_state"].endswith("=rho"),
        "symmetric_transition_switch": (
            "(1-rho)/(N-1)" in transition["different_state"]
        ),
        "symmetric_warning_correct": warning["correct_label"].endswith("=q"),
        "symmetric_warning_error": (
            "(1-q)/(N-1)" in warning["incorrect_label"]
        ),
        "warning_null_is_independence": (
            "q=1/N" in warning["independence_null"]
            and "independent" in warning["independence_null"]
        ),
        "iid_null_is_uniform": (
            demand["iid_regime_null"]["equivalent_symmetric_parameter"]
            == "rho=1/N"
        ),
        "causal_order": event_order_ok,
        "h4_clones_physics_and_rng": (
            "byte-identical physical" in demand["h4_identification"]
            and "RNG state" in demand["h4_identification"]
            and "Only the knowledge state" in demand["h4_identification"]
        ),
    }

    checks = {
        "c1_action_counts_match_stars_and_bars": _check(
            computed_actions == expected_actions,
            computed_actions,
            expected_actions,
            "A mismatch would invalidate the declared action-space cardinality.",
        ),
        "c2_enumeration_is_unique_and_complete": _check(
            all(unique_actions.values())
            and all(len(enumerated[n]) == computed_actions[n] for n in scales),
            {
                n: {"enumerated": len(enumerated[n]), "unique": unique_actions[n]}
                for n in scales
            },
            {n: computed_actions[n] for n in scales},
            "Duplicate or missing allocations would corrupt the cardinality check.",
        ),
        "c3_actions_decoder_and_resource_rights_are_consistent": _check(
            all(resource_boundaries.values()),
            resource_boundaries,
            {key: True for key in resource_boundaries},
            (
                "The design must conserve three batch rights exactly while treating "
                "actual utilization as an endogenous outcome."
            ),
        ),
        "c4_calendar_counts_are_arithmetically_consistent": _check(
            computed_calendars == expected_calendars,
            computed_calendars,
            expected_calendars,
            (
                "Calendar cardinality is a complexity descriptor, not evidence of "
                "planner runtime."
            ),
        ),
        "c5_n2_has_cardinality_only_and_parity_remains_pending": _check(
            computed_calendars.get(2) == 65_536
            and anchor["status"] == "PENDING_EXECUTABLE_G0_PARITY"
            and anchor["warning_consumed_by_controller"] is False
            and len(anchor["required_checks"]) >= 5
            and "only" in contract["scientific_scope"]["current_n2_evidence"],
            {
                "calendar_count": computed_calendars.get(2),
                "anchor_status": anchor["status"],
                "warning_consumed": anchor["warning_consumed_by_controller"],
                "required_executable_checks": anchor["required_checks"],
            },
            {
                "calendar_count": 65_536,
                "anchor_status": "PENDING_EXECUTABLE_G0_PARITY",
                "claim": "cardinality only; no physical parity",
            },
            (
                "Matching 4^8 does not certify transition, observation, metric, "
                "comparator or outcome parity with Program O/Q."
            ),
        ),
        "c6_hmm_warning_causality_and_h4_are_explicit": _check(
            all(hmm_boundaries.values()),
            hmm_boundaries,
            {key: True for key in hmm_boundaries},
            (
                "An accuracy scalar without a symmetric kernel, or unmatched physical "
                "state in H4, would not identify information retention."
            ),
        ),
        "c7_architecture_metrics_and_compute_claims_are_bounded": _check(
            all(architecture_boundaries.values())
            and all(metric_compute_boundaries.values()),
            {
                "architecture": architecture_boundaries,
                "metric_and_compute": metric_compute_boundaries,
            },
            {
                "architecture": {
                    key: True for key in architecture_boundaries
                },
                "metric_and_compute": {
                    key: True for key in metric_compute_boundaries
                },
            },
            (
                "Equivariance, branch-specific authorization, a high-budget teacher, "
                "secondary metric roles and an absolute latency gate prevent a "
                "manufactured neural premium."
            ),
        ),
        "c8_no_seed_or_scientific_execution_is_authorized": _check(
            contract["status"]
            == "CANDIDATE_DESIGN_DIAGNOSTIC_NOT_EXECUTABLE_NO_SEEDS_AUTHORIZED"
            and execution["seed_registry_state"]
            == "BASELINE_INVENTORY_INCOMPLETE_NO_NEW_SEEDS_AUTHORIZED"
            and execution["fresh_seeds_opened"] is False
            and execution["seed_ranges_assigned"] is False
            and execution["scientific_execution_authorized"] is False
            and learner["neural_training_authorized"] is False,
            {
                "contract_status": contract["status"],
                "seed_registry_state": execution["seed_registry_state"],
                "fresh_seeds_opened": execution["fresh_seeds_opened"],
                "seed_ranges_assigned": execution["seed_ranges_assigned"],
                "scientific_execution_authorized": execution[
                    "scientific_execution_authorized"
                ],
                "neural_training_authorized": learner[
                    "neural_training_authorized"
                ],
            },
            "candidate status plus all execution/training flags false",
            (
                "The live custody registry forbids new seeds until its inventory and "
                "PI authorization are complete."
            ),
        ),
    }

    failed = [name for name, row in checks.items() if not row["passed"]]
    return {
        "schema_version": "program_x_o_scale_design_consistency_preflight_v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_role": "DETERMINISTIC_DESIGN_CONSISTENCY_PREFLIGHT",
        "scope": (
            "DIAGNOSTIC_NO_SEEDS_NO_STOCHASTIC_TAPES_NO_SIMULATION_NO_LEARNER_"
            "NO_N2_PHYSICAL_PARITY_CLAIM"
        ),
        "claim_status": (
            "DESIGN_CONSISTENCY_PASS__NO_SCIENTIFIC_GATE_OPENED"
            if not failed
            else "DESIGN_CONSISTENCY_FAIL"
        ),
        "computed": {
            "action_counts": computed_actions,
            "open_loop_calendar_counts": computed_calendars,
            "sample_actions": {
                str(n): [
                    list(row)
                    for row in enumerated[n][: min(5, len(enumerated[n]))]
                ]
                for n in scales
            },
        },
        "consistency_checks": checks,
        "consistency_summary": {
            "all_passed": not failed,
            "n_computed": len(checks),
            "n_failed": len(failed),
            "failed": failed,
        },
        "claim_boundary": {
            "n2_cardinality_established": True,
            "n2_physical_parity_established": False,
            "headroom_established": False,
            "history_value_established": False,
            "neural_premium_established": False,
            "neural_training_authorized": False,
            "fresh_seeds_opened": False,
        },
    }


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_sealed(payload: dict[str, Any], contract_path: Path, output: Path) -> None:
    result = dict(payload)
    result["contract_path"] = str(contract_path.relative_to(ROOT))
    result["contract_sha256"] = _sha256(contract_path.read_bytes())
    canonical = json.dumps(
        result, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    result["self_sha256"] = _sha256(canonical)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    payload = build_payload(contract)
    write_sealed(payload, args.contract, args.output)
    print(payload["claim_status"])
    for n, count in payload["computed"]["action_counts"].items():
        calendars = payload["computed"]["open_loop_calendar_counts"][n]
        print(f"N={n}: actions={count}; calendars={calendars:,}")
    return 0 if payload["consistency_summary"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
