#!/usr/bin/env python3
"""Refresh hashes in the curated Paper-2 reproducibility manifest fail-closed."""
from __future__ import annotations

from datetime import date
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess


ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "research" / "paper2_exhaustive_search" / "reproducibility_manifest.json"
PACKAGES = (
    "SALib",
    "gymnasium",
    "numpy",
    "pandas",
    "sb3-contrib",
    "scikit-learn",
    "scipy",
    "simpy",
    "stable-baselines3",
    "torch",
)

REQUIRED_ARTIFACT_PATHS = {
    "research/paper2_exhaustive_search/README.md",
    "contracts/program_o_multi_ration_product_mix_v1.json",
    "contracts/program_o_exact_transducer_v1.json",
    "research/paper2_exhaustive_search/PROGRAM_O_MULTI_RATION_PREREGISTRATION_2026-07-14.md",
    "research/paper2_exhaustive_search/PROGRAM_O_EXACT_TRANSDUCER_SCREEN_FREEZE_2026-07-14.md",
    "results/program_o/affected_order_bound_v1/result.json",
    "results/program_o/exact_transducer_screen_v1/result.json",
    "research/paper2_exhaustive_search/program_o_exact_transducer_validation_freeze_20260714.json",
    "results/program_o/exact_transducer_validation_v1/result.json",
    "results/program_o/exact_transducer_validation_v1/ret_matrices.npz",
    "contracts/program_m_shared_lift_reservation_v1.json",
    "docs/external_assessments/EXTERNAL_CHATGPT_PRO_PROGRAM_M_AUDIT_PROMPT_2026-07-14.md",
    "docs/external_assessments/EXTERNAL_CHATGPT_PRO_PROGRAM_O_FULL_DES_AUDIT_PROMPT_2026-07-14.md",
    "research/paper2_exhaustive_search/PROGRAM_M_SHARED_LIFT_PREREGISTRATION_2026-07-14.md",
    "research/paper2_exhaustive_search/program_m_hpi_screen_custody_20260714.json",
    "research/paper2_exhaustive_search/program_m_hpi_screen_selection_20260714.json",
    "results/program_m/hpi_screen_v1/result.json",
    "results/program_m/hpi_screen_v1/raw_custody_bundle.tgz",
    "research/paper2_exhaustive_search/program_m_hpi_validation_verdict_20260714.json",
    "results/program_m/hpi_validation_v1/result.json",
    "results/program_m/hpi_validation_v1/raw_custody_bundle.tgz",
    "research/paper2_exhaustive_search/phase0_failure_taxonomy.json",
    "research/paper2_exhaustive_search/phase0_failure_taxonomy_validation.json",
    "research/paper2_exhaustive_search/provenance_reconciliation.md",
    "research/paper2_exhaustive_search/artifact_index.json",
    "research/paper2_exhaustive_search/source_reconstruction.md",
    "research/paper2_exhaustive_search/source_extraction_index.json",
    "research/paper2_exhaustive_search/excel_metric_reaudit_20260713.json",
    "docs/RET_EXCEL_REQUEST_SNAPSHOT_V2_CONTRACT_2026-07-14.md",
    "research/paper2_exhaustive_search/metric_governance_audit.json",
    "research/paper2_exhaustive_search/ret_excel_visible_v1_source_semantics_audit_20260714.json",
    "research/paper2_exhaustive_search/ret_excel_request_snapshot_v2_implementation_audit_20260714.json",
    "research/paper2_exhaustive_search/primary_source_literature_review.md",
    "research/paper2_exhaustive_search/literature_discovery_inclusion_inventory_20260713.json",
    "research/paper2_exhaustive_search/approach_registry.json",
    "research/paper2_exhaustive_search/historical_visible_v1_ceiling_audit_20260714.json",
    "research/paper2_exhaustive_search/candidate_intervention_ledger.json",
    "research/paper2_exhaustive_search/boundary_family_proof_ledger.json",
    "research/paper2_exhaustive_search/global_headroom_sensitivity_design_and_results.md",
    "research/paper2_exhaustive_search/global_sensitivity_portfolio_inventory.json",
    "research/paper2_exhaustive_search/program_j_request_snapshot_v2_frontier_structure_audit_20260714.json",
    "research/paper2_exhaustive_search/mtr_switch4_vps_metric_quarantine_20260714.json",
    "research/paper2_exhaustive_search/boundary_verification.json",
    "research/paper2_exhaustive_search/terminal_return_readiness.json",
    "research/paper2_exhaustive_search/terminal_return_verification.json",
    "research/paper2_exhaustive_search/comparator_completeness_audit.md",
    "research/paper2_exhaustive_search/action_trajectory_audit.md",
    "research/paper2_exhaustive_search/paper2_paper3_status.md",
    "research/paper2_exhaustive_search/paper3_claim_supersession.json",
    "research/paper2_exhaustive_search/paper_facing_claims_table.json",
    "research/paper2_exhaustive_search/paper_facing_claims_table.md",
    "research/paper2_exhaustive_search/vps_switch4_producer_precompletion_anchor_20260714.json",
    "research/paper2_exhaustive_search/mtr_switch4_deep_replay_readiness_20260714.json",
    "scripts/audit_paper2_switch4_exact_ties.py",
    "scripts/reconstruct_program_i_headroom_gsa_cells.py",
    "scripts/audit_ret_excel_visible_source_semantics.py",
    "scripts/audit_ret_excel_request_snapshot_v2.py",
    "scripts/produce_program_j_request_snapshot_v2_108cell_frontier.py",
    "scripts/launch_program_j_request_snapshot_v2_108cell.py",
    "scripts/watch_program_j_request_snapshot_v2_108cell.py",
    "scripts/screen_program_m_shared_lift_hpi.py",
    "scripts/launch_program_m_shared_lift_hpi.py",
    "scripts/watch_program_m_shared_lift_hpi.py",
    "scripts/bound_program_o_affected_orders.py",
    "scripts/screen_program_o_exact_transducer.py",
    "scripts/validate_program_o_exact_transducer.py",
    "scripts/validate_program_m_shared_lift_hpi.py",
    "scripts/launch_program_m_shared_lift_hpi_validation.py",
    "scripts/record_mtr_switch4_metric_quarantine.py",
    "scripts/validate_phase0_failure_taxonomy.py",
    "scripts/validate_paper2_switch4_producer_custody.py",
    "scripts/verify_paper2_exhaustion.py",
    "scripts/verify_paper2_terminal_return.py",
    "tests/test_paper2_exhaustive_search_registry.py",
    "tests/test_program_i_headroom_gsa_reconstruction.py",
    "tests/test_paper2_switch4_exact_ties.py",
    "tests/test_paper2_terminal_return.py",
    "tests/test_ret_excel_request_snapshot_contract.py",
    "tests/test_ret_excel_request_snapshot_v2_audit.py",
    "tests/test_ret_excel_visible_source_semantics_audit.py",
    "tests/test_program_j_request_snapshot_v2_108cell_producer.py",
    "tests/test_program_j_request_snapshot_v2_structure_audit.py",
    "tests/test_program_m_shared_lift.py",
    "tests/test_program_m_shared_lift_hpi_screen.py",
    "tests/test_program_m_shared_lift_hpi_custody.py",
    "tests/test_program_m_shared_lift_hpi_validation.py",
    "tests/test_program_m_shared_lift_hpi_validation_custody.py",
    "tests/test_program_o_affected_order_bound.py",
    "tests/test_program_o_exact_transducer.py",
    "tests/test_program_o_exact_transducer_validation.py",
    "tests/test_mtr_switch4_vps_metric_quarantine.py",
    "results/headroom_gsa/all_cells_reconstruction.json",
    "results/paper2_maintenance/request_snapshot_v2_full_frontier/verdict.json",
    "results/paper2_maintenance/request_snapshot_v2_full_frontier/raw_matrices.npz",
    "results/program_h/visible_v1_repair/observable_policy_screen.json",
    "supply_chain/ret_thesis.py",
    "supply_chain/episode_metrics.py",
    "supply_chain/garrido_replication.py",
    "supply_chain/supply_chain.py",
    "supply_chain/program_m_shared_lift.py",
}
REQUIRED_SOURCE_PATHS = {
    "/Users/thom/Downloads/Raw_data1+Re.xlsx",
    "/Users/thom/Downloads/Raw_data2+Re.xlsx",
    "/Users/thom/Downloads/Rsult_1.xlsx",
    "/Users/thom/Downloads/garrido et al 2024 factory resilience.pdf",
    "/Users/thom/Downloads/v.0_neuralNet-scres.docx",
    "/Users/thom/Downloads/v.0_neuralNet-scres.pdf",
    "/Users/thom/Library/CloudStorage/GoogleDrive-chisicathomas@gmail.com/My Drive/Supernote/Document/20_RESEARCH/PhD-Papers/garrido2024 scres+AI.pdf",
    "/Users/thom/Library/CloudStorage/GoogleDrive-chisicathomas@gmail.com/My Drive/Archive/Misc_Unsorted/Unsorted/WRAP_Theses_Garrido_Rios_2017.pdf",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def resolve_artifact(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def refresh_hashes(rows: dict[str, str]) -> dict[str, str]:
    refreshed = {}
    missing = []
    for path_text in sorted(rows):
        path = resolve_artifact(path_text)
        if not path.is_file():
            missing.append(path_text)
            continue
        refreshed[path_text] = sha256(path)
    if missing:
        raise FileNotFoundError(f"Manifest-listed files are missing: {missing}")
    return refreshed


def main() -> int:
    manifest = json.loads(MANIFEST.read_text())
    required = {
        "schema_version",
        "scientific_status",
        "paper2_confirmed",
        "paper3_authorized",
        "repository",
        "execution_scope",
        "seed_and_tape_status",
        "known_reproducibility_defects",
        "commands",
        "artifact_hashes",
        "source_hashes",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"Curated manifest is missing required fields: {missing}")

    # Never reconstruct scientific status, commands, exclusions or claim limits.
    # Only fields that can drift mechanically are refreshed here.
    manifest["generated_date"] = date.today().isoformat()
    manifest["repository"]["branch"] = git("branch", "--show-current")
    manifest["repository"]["head_input"] = git("rev-parse", "HEAD")
    branch = manifest["repository"]["branch"]
    remote_branch = f"origin/{branch}"
    manifest["repository"]["origin_current_branch_at_refresh"] = git(
        "rev-parse", remote_branch
    )
    manifest["repository"]["origin_main"] = git("rev-parse", "origin/main")
    manifest["repository"]["local_head_published"] = bool(
        git("branch", "-r", "--contains", manifest["repository"]["head_input"])
    )
    manifest["environment"] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            package: _package_version(package) for package in PACKAGES
        },
    }
    manifest["artifact_hashes"] = refresh_hashes(manifest["artifact_hashes"])
    manifest["source_hashes"] = refresh_hashes(manifest["source_hashes"])
    missing_required_artifacts = sorted(
        REQUIRED_ARTIFACT_PATHS - set(manifest["artifact_hashes"])
    )
    missing_required_sources = sorted(
        REQUIRED_SOURCE_PATHS - set(manifest["source_hashes"])
    )
    if missing_required_artifacts or missing_required_sources:
        raise ValueError(
            "Curated manifest required-set coverage failed: "
            f"artifacts={missing_required_artifacts}, sources={missing_required_sources}"
        )
    manifest["required_set_coverage"] = {
        "required_artifact_count": len(REQUIRED_ARTIFACT_PATHS),
        "required_source_count": len(REQUIRED_SOURCE_PATHS),
        "missing_required_artifacts": [],
        "missing_required_sources": [],
        "passed": True,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(MANIFEST)
    return 0


def _package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
