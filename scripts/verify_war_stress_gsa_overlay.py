#!/usr/bin/env python3
"""Verify the war-stress GSA overlay freeze without opening scientific tapes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_war_stress_gsa_overlay_manifest import build_manifest  # noqa: E402


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(repo_root: Path) -> dict[str, Any]:
    contract_path = repo_root / "contracts/war_stress_gsa_overlay_v1.json"
    parent_path = repo_root / "contracts/war_stress_timing_atlas_v1.json"
    manifest_path = (
        repo_root
        / "research/paper2_exhaustive_search/war_stress_gsa_overlay_manifest_20260716.json"
    )
    custody_path = (
        repo_root
        / "research/paper2_exhaustive_search/war_stress_timing_seed_custody_20260716.json"
    )
    failures: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    for path in (contract_path, parent_path, manifest_path, custody_path):
        require(path.is_file(), f"missing artifact: {path}")
    if failures:
        return {
            "schema_version": "war_stress_gsa_overlay_verification_v1",
            "status": "FAIL_WAR_STRESS_GSA_OVERLAY_FREEZE",
            "failures": failures,
        }

    contract = _load(contract_path)
    manifest = _load(manifest_path)
    custody = _load(custody_path)
    require(contract.get("status") == "FROZEN_BEFORE_SCIENTIFIC_SEED_ACCESS", "overlay status mismatch")
    require(_sha256(parent_path) == contract["parent_contract"]["sha256"], "parent contract hash mismatch")
    require(_sha256(manifest_path) == contract["sampling_manifest"]["sha256"], "sampling manifest hash mismatch")
    require(manifest.get("status") == "FROZEN_BEFORE_SCIENTIFIC_SEED_ACCESS", "manifest status mismatch")
    require(int(manifest["morris"]["configuration_count"]) == 570, "Morris count mismatch")
    require(int(manifest["qmc_pool"]["configuration_count"]) == 1536, "QMC count mismatch")
    ids = [row["config_id"] for family in ("morris", "qmc_pool") for row in manifest[family]["rows"]]
    require(len(ids) == len(set(ids)), "duplicate GSA configuration IDs")

    rebuilt = build_manifest()
    require(
        rebuilt["configuration_payload_sha256"] == manifest["configuration_payload_sha256"],
        "reconstructed GSA configuration payload mismatch",
    )
    require(
        manifest["configuration_payload_sha256"]
        == contract["sampling_manifest"]["configuration_payload_sha256"],
        "contract configuration payload hash mismatch",
    )

    target = contract["target"]
    require(target.get("id") == "H_timing_safe", "wrong GSA target")
    audit = contract.get("canonical_audit_standard", {})
    audit_path = repo_root / str(audit.get("path", ""))
    require(audit_path.is_file(), "canonical A1-A11 audit standard missing")
    if audit_path.is_file():
        require(_sha256(audit_path) == audit.get("sha256"), "audit standard hash mismatch")
    stochastic = contract.get("stochastic_protocol", {})
    calibration = stochastic.get("non_scientific_DES_calibration", {})
    for path_key, hash_key in (("path", "sha256"), ("runner_path", "runner_sha256")):
        path = repo_root / str(calibration.get(path_key, ""))
        require(path.is_file(), f"missing DES noise calibration artifact {path_key}")
        if path.is_file():
            require(
                _sha256(path) == calibration.get(hash_key),
                f"DES noise calibration hash mismatch for {path_key}",
            )
    surrogate_tapes = set(stochastic.get("qmc_surrogate_tapes", []))
    prim_tapes = set(stochastic.get("prim_independent_tapes", []))
    require(not surrogate_tapes & prim_tapes, "surrogate and PRIM tape blocks overlap")
    require(
        surrogate_tapes | prim_tapes == set(range(7470004, 7470013)),
        "surrogate/PRIM tape partition mismatch",
    )
    require(
        contract.get("phase_G3_sobol", {}).get("interaction_claim_authorized") is False,
        "interaction claims are not fail-closed",
    )
    require(
        int(contract.get("phase_G4_scenario_discovery", {}).get("permutation_repeats", 0))
        == 99,
        "PRIM permutation null is not frozen",
    )
    require(
        contract.get("relationship_to_parent_promotion", {}).get(
            "parent_atlas_must_execute_first"
        )
        is True,
        "GSA can run before the parent atlas",
    )
    require(
        any("regime-tailoring" in item for item in target.get("not_allowed", [])),
        "constant-tailoring target not forbidden",
    )
    require(
        contract["phase_G2_surrogate"]["failure"]
        == "no Sobol indices and no PRIM box may be reported for that stratum",
        "surrogate failure is not fail-closed",
    )
    require(
        bool(contract["relationship_to_parent_promotion"]["parent_144_cell_gate_remains_binding"]),
        "parent grid gate no longer binding",
    )
    require(not bool(contract["authorization"]["scientific_seed_access"]), "scientific seed access authorized")
    require(not bool(contract["authorization"]["learner_authorized"]), "learner authorized")
    require(not bool(custody.get("opened")), "parent development seeds already opened")
    require(int(custody.get("scientific_tape_access_count", -1)) == 0, "nonzero tape access count")

    try:
        salib_version = importlib.metadata.version("SALib")
        ema_version = importlib.metadata.version("ema-workbench")
    except importlib.metadata.PackageNotFoundError as exc:
        failures.append(f"missing required GSA dependency: {exc}")
        salib_version = "missing"
        ema_version = "missing"
    require(salib_version == "1.5.2", f"unexpected SALib version {salib_version}")
    require(ema_version == "2.5.3", f"unexpected EMA Workbench version {ema_version}")

    return {
        "schema_version": "war_stress_gsa_overlay_verification_v1",
        "status": (
            "PASS_WAR_STRESS_GSA_OVERLAY_FREEZE"
            if not failures
            else "FAIL_WAR_STRESS_GSA_OVERLAY_FREEZE"
        ),
        "contract_sha256": _sha256(contract_path),
        "manifest_sha256": _sha256(manifest_path),
        "configuration_payload_sha256": manifest["configuration_payload_sha256"],
        "morris_configurations": int(manifest["morris"]["configuration_count"]),
        "qmc_pool_configurations": int(manifest["qmc_pool"]["configuration_count"]),
        "salib_version": salib_version,
        "ema_workbench_version": ema_version,
        "scientific_seeds_opened": bool(custody.get("opened")),
        "interaction_claim_authorized": False,
        "prim_tape_block_independent": not bool(surrogate_tapes & prim_tapes),
        "DES_noise_calibration_fixture_populated": bool(calibration),
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "research/paper2_exhaustive_search/war_stress_gsa_overlay_verification_20260716.json"
        ),
    )
    args = parser.parse_args()
    verdict = verify(args.repo_root)
    rendered = json.dumps(verdict, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if verdict["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
