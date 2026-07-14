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
    "research/paper2_exhaustive_search/phase0_failure_taxonomy.json",
    "research/paper2_exhaustive_search/phase0_failure_taxonomy_validation.json",
    "research/paper2_exhaustive_search/provenance_reconciliation.md",
    "research/paper2_exhaustive_search/artifact_index.json",
    "research/paper2_exhaustive_search/source_reconstruction.md",
    "research/paper2_exhaustive_search/source_extraction_index.json",
    "research/paper2_exhaustive_search/excel_metric_reaudit_20260713.json",
    "research/paper2_exhaustive_search/primary_source_literature_review.md",
    "research/paper2_exhaustive_search/literature_discovery_inclusion_inventory_20260713.json",
    "research/paper2_exhaustive_search/approach_registry.json",
    "research/paper2_exhaustive_search/candidate_intervention_ledger.json",
    "research/paper2_exhaustive_search/boundary_family_proof_ledger.json",
    "research/paper2_exhaustive_search/boundary_verification.json",
    "research/paper2_exhaustive_search/terminal_return_readiness.json",
    "research/paper2_exhaustive_search/terminal_return_verification.json",
    "research/paper2_exhaustive_search/comparator_completeness_audit.md",
    "research/paper2_exhaustive_search/action_trajectory_audit.md",
    "research/paper2_exhaustive_search/paper2_paper3_status.md",
    "research/paper2_exhaustive_search/paper_facing_claims_table.json",
    "research/paper2_exhaustive_search/paper_facing_claims_table.md",
    "research/paper2_exhaustive_search/vps_switch4_producer_precompletion_anchor_20260714.json",
    "research/paper2_exhaustive_search/mtr_switch4_deep_replay_readiness_20260714.json",
    "scripts/audit_paper2_switch4_exact_ties.py",
    "scripts/validate_phase0_failure_taxonomy.py",
    "scripts/validate_paper2_switch4_producer_custody.py",
    "scripts/verify_paper2_exhaustion.py",
    "scripts/verify_paper2_terminal_return.py",
    "tests/test_paper2_exhaustive_search_registry.py",
    "tests/test_paper2_switch4_exact_ties.py",
    "tests/test_paper2_terminal_return.py",
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
