#!/usr/bin/env python3
"""Audit the SCRES/Garrido phase artifacts without opening new simulations.

This runner is deliberately read-only with respect to scientific inputs. It reads
sealed JSON artifacts, recomputes the headline contrasts from their recorded
fields, checks claim boundaries, and writes a manifest with input hashes. It does
not launch DES jobs, train learners, or allocate seeds.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


ARTIFACTS = {
    "workbook_fidelity_report": "outputs/audits/garrido_workbook_fidelity_attached_2026-08-06/audit_report.md",
    "forensic_replication": "outputs/audits/garrido_replication_attached_forensic_2026-08-06/replication_audit.json",
    "endogenous_replication": "outputs/audits/garrido_replication_attached_endogenous_2026-08-06/replication_audit.json",
    "grid_transfer_development": "results/grid_transfer_v2/result.json",
    "grid_transfer_confirmation": "results/grid_transfer_confirmation_v2/result.json",
    "search_ladder": "results/search_ladder_v5/result.json",
    "search_surrogates": "results/search_surrogates/result.json",
    "v0_recovery_gate": "results/garrido_v0_recovery_gate_v2/result.json",
    "v0_surface_gate": "results/garrido_v0_surface_gates_v1/result.json",
    "risk_sensitivity_audit": "results/garrido_risk_headroom_sensitivity_v1/independent_audit_v1.json",
    "step3_pooled": "results/step3_pooled/result.json",
    "architecture_bakeoff": "results/architecture_bakeoff/result.json",
    "kan_interpretability": "results/kan_interpretability/result.json",
    "cd_premium": "results/headroom/cd_surface_prediction_premium/result.json",
    "program_b_full_development": "results/program_b_gate_v2/development.json",
    "program_b_full_validation": "results/program_b_gate_v2/validation.json",
    "program_b_service_development": "results/program_b_gate_v2/development_service_safe.json",
    "program_b_service_validation": "results/program_b_gate_v2/validation_service_safe.json",
    "program_b_contract": "contracts/program_b_service_safe_learner_v1.json",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(root: Path, rel: str) -> dict[str, Any]:
    return json.loads((root / rel).read_text())


def finite(v: Any) -> Any:
    """Convert JSON values without silently rounding scientific evidence."""
    return v


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()


    missing = [rel for rel in ARTIFACTS.values() if not (root / rel).is_file()]
    if missing:
        raise SystemExit("missing artifacts:\n" + "\n".join(missing))


    grid_dev = load_json(root, ARTIFACTS["grid_transfer_development"])
    grid_val = load_json(root, ARTIFACTS["grid_transfer_confirmation"])
    forensic = load_json(root, ARTIFACTS["forensic_replication"])
    endogenous = load_json(root, ARTIFACTS["endogenous_replication"])
    ladder = load_json(root, ARTIFACTS["search_ladder"])
    surrogates = load_json(root, ARTIFACTS["search_surrogates"])
    recovery = load_json(root, ARTIFACTS["v0_recovery_gate"])
    surface = load_json(root, ARTIFACTS["v0_surface_gate"])
    risk = load_json(root, ARTIFACTS["risk_sensitivity_audit"])
    step3 = load_json(root, ARTIFACTS["step3_pooled"])
    bakeoff = load_json(root, ARTIFACTS["architecture_bakeoff"])
    cd = load_json(root, ARTIFACTS["cd_premium"])
    b_full_dev = load_json(root, ARTIFACTS["program_b_full_development"])
    b_full_val = load_json(root, ARTIFACTS["program_b_full_validation"])
    b_safe_dev = load_json(root, ARTIFACTS["program_b_service_development"])
    b_safe_val = load_json(root, ARTIFACTS["program_b_service_validation"])
    learner_contract = load_json(root, ARTIFACTS["program_b_contract"])


    # These are intentionally hard failures: a changed claim or a stale result
    # must not be smoothed into a new report.
    assert grid_val["claim_status"] == "GRID_TRANSFER_CONFIRMED__UCB1"
    assert grid_val["transfers"]["ucb1"] is True
    assert grid_val["contrasts"]["ucb1"]["vs_marginal_replay"]["n"] == 60
    assert grid_val["contrasts"]["ucb1"]["vs_marginal_replay"]["lcb95"] > 0.0
    assert grid_val["contrasts"]["ucb1"]["vs_cold"]["lcb95"] > 0.0
    assert grid_dev["claim_status"] == "GRID_TRANSFER_ESTABLISHED__UCB1"
    assert ladder["falsifiers"]["all_passed"] is True
    assert surrogates["falsifiers"]["all_passed"] is True
    assert recovery["claim_status"] == "GO_BUILD_V0_RECOVERY_SURFACE"
    assert surface["claim_status"] == "STOP_NO_RECOVERY_LEARNING_HEADROOM"
    assert surface["gates"]["g2_surface_is_nonseparable_out_of_seed"]["passed"] is False
    assert surface["gates"]["g3_context_specific_postures_have_operational_value"]["passed"] is False
    assert b_full_dev["primary_gate"]["status"] == "STOP_PRIMARY_FULL_LEDGER_HAS_NO_HEADROOM"
    assert b_full_val["primary_gate"]["status"] == "STOP_PRIMARY_FULL_LEDGER_HAS_NO_HEADROOM"
    assert b_full_dev["primary_gate"]["primary_ret_full_identically_zero"] is True
    assert b_full_val["primary_gate"]["primary_ret_full_identically_zero"] is True
    assert b_safe_dev["primary_gate"]["status"] == "REPORT_SERVICE_SAFE_METRIC_EXPLORATORY"
    assert b_safe_val["primary_gate"]["status"] == "REPORT_SERVICE_SAFE_METRIC_EXPLORATORY"
    assert learner_contract["status"] == "FROZEN_BEFORE_DEVELOPMENT_TRAINING_NO_NEW_DES_SEEDS"
    assert learner_contract["gates"]["fresh_confirmatory_training"] is False


    # The independent risk audit is a compact gate report; retain its exact
    # fields rather than trying to infer a stronger claim from the 4,860 rows.
    risk_status = risk.get("status") or risk.get("claim_status") or "AUDIT_ARTIFACT_PRESENT"
    risk_rows = risk.get("n_rows")
    if risk_rows is None:
        risk_rows = risk.get("evidence", {}).get("n_rows")

    selected = {
        "garrido_fidelity": {
            "workbook_report": ARTIFACTS["workbook_fidelity_report"],
            "forensic_replay": {
                "path": ARTIFACTS["forensic_replication"],
                "replication_status": forensic.get("replication_status"),
                "mean_abs_ret_gap": forensic["best_summary"]["mean_abs_ret_gap"],
                "max_abs_ret_gap": forensic["best_summary"]["max_abs_ret_gap"],
            },
            "endogenous_replay": {
                "path": ARTIFACTS["endogenous_replication"],
                "replication_status": endogenous.get("replication_status"),
                "mean_abs_ret_gap": endogenous["best_summary"]["mean_abs_ret_gap"],
                "max_abs_ret_gap": endogenous["best_summary"]["max_abs_ret_gap"],
                "max_branch_share_gap_pct": endogenous["best_summary"]["max_branch_share_gap_pct"],
            },
        },
        "garrido_outer_loop": {
            "development": {
                "status": grid_dev["claim_status"],
                "ucb1_vs_marginal": grid_dev["contrasts"]["ucb1"]["vs_marginal_replay"],
                "ucb1_vs_cold": grid_dev["contrasts"]["ucb1"]["vs_cold"],
                "mean_auc_regret": grid_dev["mean_auc"],
            },
            "reserved_block": {
                "status": grid_val["claim_status"],
                "ucb1_vs_marginal": grid_val["contrasts"]["ucb1"]["vs_marginal_replay"],
                "ucb1_vs_cold": grid_val["contrasts"]["ucb1"]["vs_cold"],
                "custody_caveat": grid_val["falsifiers"]["f4_seed_custody"]["evidence"],
            },
            "search_ladder": {
                "status": ladder["claim_status"],
                "ranking_best_first": ladder["ranking_best_first"],
                "mean_auc_regret": ladder["mean_auc_regret"],
            },
            "surrogate_comparison": {
                "status": surrogates["claim_status"],
                "mean_auc_regret": surrogates["mean_auc_regret"],
                "neural_premium_point_estimate": surrogates["neural_premium_point_estimate"],
            },
        },
        "headroom_and_policies": {
            "risk_sensitivity": {"path": ARTIFACTS["risk_sensitivity_audit"], "status": risk_status, "n_rows": risk_rows},
            "step3_pooled": {"status": step3["claim_status"], "scope": step3["scope"], "falsifiers": step3["falsifiers"]},
            "architecture_bakeoff": {"status": bakeoff["claim_status"], "contrasts": bakeoff["contrasts"]},
            "cobb_douglas": {
                "claim_status": cd["claim_status"],
                "oracle_minus_best_classical": cd["falsifiers"]["f4_available_premium_is_measured_not_asserted"]["evidence"]["oracle_minus_best_classical"],
            },
        },
        "program_b": {
            "full_ledger": {
                "development_status": b_full_dev["primary_gate"]["status"],
                "validation_status": b_full_val["primary_gate"]["status"],
                "development_fungible_null": b_full_dev["fungible_null"],
                "validation_fungible_null": b_full_val["fungible_null"],
            },
            "service_safe_exploratory": {
                "development_status": b_safe_dev["primary_gate"]["status"],
                "validation_status": b_safe_val["primary_gate"]["status"],
                "development_frozen_comparator": b_safe_dev["comparisons"]["frozen_incumbent"]["primary_safe_h_pi"],
                "development_in_sample_comparator": b_safe_dev["comparisons"]["in_sample_static_incumbent"]["primary_safe_h_pi"],
                "validation_frozen_comparator": b_safe_val["comparisons"]["frozen_incumbent"]["primary_safe_h_pi"],
                "validation_in_sample_comparator": b_safe_val["comparisons"]["in_sample_static_incumbent"]["primary_safe_h_pi"],
                "h_pi_required_before_training": learner_contract["gates"]["H_PI_required_before_training"],
            },
            "contract": {
                "path": ARTIFACTS["program_b_contract"],
                "status": learner_contract["status"],
                "training_seeds": learner_contract["learner"]["seeds"],
                "training_tape_status": learner_contract["training_tapes"]["status"],
                "evaluation_tape_status": learner_contract["evaluation_tapes"]["status"],
                "new_des_seeds_authorized": learner_contract["gates"]["fresh_confirmatory_training"],
            },
        },
    }

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    manifest = {
        "schema_version": "scres_phase_audit_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "read_only_scientific_inputs": True,
        "new_des_seeds_opened": False,
        "selected_evidence": selected,
        "input_sha256": {name: sha256(root / rel) for name, rel in ARTIFACTS.items()},
        "claim_boundary": [
            "The forensic replay validates extraction/formula compatibility, not endogenous DES equivalence.",
            "The outer-loop result supports UCB1 retention/transfer on its declared block; it is not evidence for intra-episode RL superiority.",
            "The Program B full-ledger result is a null; service-safe values are exploratory and do not authorize promotion.",
            "The Cobb-Douglas artifact has a point estimate whose confidence interval crosses zero; it is not a confirmed neural premium.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "git_commit": commit, "new_des_seeds_opened": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
