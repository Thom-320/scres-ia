#!/usr/bin/env python3
"""Fail-closed corrective audit of the concurrent Program-L terminal claim."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
FULL_RESULT = ROOT / "results" / "paper2_search" / "program_l_full_des_gate.json"
STYLIZED_RESULT = (
    ROOT / "results" / "paper2_search" / "program_l_route_recourse_screen.json"
)
RUNNER = (
    ROOT
    / "research"
    / "paper2_exhaustive_search"
    / "program_l_full_des_gate.py"
)
OUTPUT = ROOT / "results" / "paper2_search" / "program_l_corrective_audit.json"


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> int:
    full = json.loads(FULL_RESULT.read_text())
    stylized = json.loads(STYLIZED_RESULT.read_text())
    runner_text = RUNNER.read_text()
    rows = full["grid"]
    positive_observable = [row for row in rows if float(row["H_obs"]) > 0.0]
    checks = {
        "legacy_quantity_can_be_negative_so_is_not_h_pi": min(
            float(row["H_PI"]) for row in rows
        )
        < 0.0,
        "legacy_runner_now_disclaims_full_horizon_pi": "H_PI_certified" in runner_text,
        "comparator_set_is_only_three_rules": all(
            name in runner_text
            for name in ('"const_R1"', '"const_R2"', '"alternate"')
        ),
        "positive_observable_cells_are_below_practical_gate": all(
            float(row["H_obs"]) < 0.01 for row in positive_observable
        ),
        "positive_observable_cells_have_nonpositive_lcb": all(
            float(row["H_obs_ci95"][0]) <= 0.0 for row in positive_observable
        ),
        "positive_observable_cells_use_more_departures": all(
            float(row["dep_obs_minus_static"]) > 0.0 for row in positive_observable
        ),
        "stylized_headline_explicitly_fails_resource_gate": (
            stylized["gates"]["headline_cell_q0.90_cover4d"]["resource_ok"]
            is False
        ),
        "no_learner_or_virgin_claim_in_inputs": "NO learner trained"
        in full.get("note", ""),
    }
    if not all(checks.values()):
        raise RuntimeError(
            "Program-L corrective audit failed closed: "
            + ", ".join(key for key, value in checks.items() if not value)
        )

    payload = {
        "schema_version": "program_l_route_recourse_corrective_audit_v1",
        "generated_date": "2026-07-13",
        "input_sha256": {
            str(FULL_RESULT.relative_to(ROOT)): file_sha256(FULL_RESULT),
            str(STYLIZED_RESULT.relative_to(ROOT)): file_sha256(STYLIZED_RESULT),
            str(RUNNER.relative_to(ROOT)): file_sha256(RUNNER),
        },
        "machine_checks": checks,
        "legacy_development_grid": {
            "n_cells": len(rows),
            "n_tapes_per_cell": sorted({int(row["n"]) for row in rows}),
            "legacy_h_pi_label_min": min(float(row["H_PI"]) for row in rows),
            "legacy_h_pi_label_max": max(float(row["H_PI"]) for row in rows),
            "positive_h_obs_rows": positive_observable,
        },
        "corrected_interpretation": {
            "tested_heuristic_contract": "falsified_as_promotable",
            "legacy_h_pi_label": "myopic_true_state_heuristic_not_an_upper_bound",
            "route_recourse_family": "blocked_domain_fact_and_open_if_validated",
            "thesis_native_route_choice": {
                "action_present": False,
                "causal_liveness": 0.0,
                "h_pi": 0.0,
                "h_obs": 0.0,
                "ceiling_basis": "alternate-route action absent from native transition kernel",
            },
            "researcher_extension_quantitative_ceiling": None,
            "terminal_boundary_reaffirmed_by_program_l": False,
            "learner_authorized": False,
            "paper3_authorized": False,
        },
        "missing_for_family_level_closure": [
            "Garrido face validation of alternate-route and finite-fleet physics",
            "immutable preregistration and refreshed tape hashes",
            "resource-restricted full-horizon oracle or certified upper bound",
            "complete full-horizon open-loop and classical comparator frontier",
            "componentwise resource and full canonical guardrail ledgers",
            "complete action-trajectory and fixed-calendar replacement audit",
        ],
        "exact_garrido_question": (
            "Does the MFSC operator choose among at least two routes from Op8 toward "
            "the same downstream demand using one finite shared fleet; if so, what are "
            "each route's payload, outbound and return times, R22 exposure, commitment "
            "and reassignment rules, degradation persistence, and warning available "
            "before dispatch, and are 36 h each way, +24 h degraded, persistence 0.85, "
            "prevalence 0.25 and signal accuracy 0.85 plausible?"
        ),
        "verdict": (
            "REJECT_PROGRAM_L_TERMINAL_REAFFIRMATION__HEURISTIC_CONTRACT_NULL__"
            "ROUTE_FAMILY_DOMAIN_BLOCKED"
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"verdict": payload["verdict"], "checks": checks}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
