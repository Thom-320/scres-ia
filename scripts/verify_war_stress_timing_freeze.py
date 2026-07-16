#!/usr/bin/env python3
"""Machine-check the war-stress timing atlas design freeze and seed custody."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any


EXPECTED_MASKS = {
    "LOC_SURGE": ("R22", "R24"),
    "THEATER_CAPACITY_SURGE": ("R21", "R23", "R24"),
    "PRODUCTION_QUALITY_SURGE": ("R11", "R14", "R24"),
}
EXPECTED_PHI = (1.0, 2.0, 4.0, 8.0)
EXPECTED_PSI = (1.0, 2.0, 4.0)
EXPECTED_COUPLINGS = (
    "independent",
    "disruption_leads_surge_72h",
    "coincident",
    "surge_leads_disruption_72h",
)


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(contract_path: Path, custody_path: Path, repo_root: Path) -> dict[str, Any]:
    contract = _load(contract_path)
    custody = _load(custody_path)
    failures: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    require(
        contract.get("status") == "FROZEN_DESIGN_BEFORE_IMPLEMENTATION_AND_SEED_ACCESS",
        "contract is not frozen before implementation and seeds",
    )
    require(
        custody.get("status") == "RESERVED_BEFORE_IMPLEMENTATION_AND_FIRST_ACCESS",
        "custody status mismatch",
    )
    require(not bool(custody.get("opened")), "scientific seeds already opened")
    require(int(custody.get("scientific_tape_access_count", -1)) == 0, "nonzero tape access count")

    grid = contract["primary_grid"]
    masks = {row["id"]: tuple(row["enabled_risks"]) for row in grid["masks"]}
    phi = tuple(float(value) for value in grid["frequency_multipliers"])
    psi = tuple(float(value) for value in grid["impact_multipliers"])
    couplings = tuple(row["id"] for row in grid["coupling_modes"])
    require(masks == EXPECTED_MASKS, "risk masks differ from the frozen design")
    require(phi == EXPECTED_PHI, "frequency grid differs from the frozen design")
    require(psi == EXPECTED_PSI, "impact grid differs from the frozen design")
    require(couplings == EXPECTED_COUPLINGS, "coupling grid differs from the frozen design")

    cells = {
        (mask, frequency, impact, coupling)
        for mask, frequency, impact, coupling in itertools.product(
            masks, phi, psi, couplings
        )
    }
    require(len(cells) == 144, "expanded primary grid is not 144 unique cells")
    require(int(grid.get("cell_count", -1)) == len(cells), "declared cell count mismatch")
    require(all("R3" not in risks for risks in masks.values()), "R3 appears in the primary grid")
    require(bool(grid.get("no_posthoc_extension")), "post-hoc grid extension is not forbidden")
    require(
        "replaced" in grid.get("coupled_schedule_semantics", {}).get("native_stream_rule", ""),
        "coupled modes do not replace native non-R24 streams",
    )
    saturation = grid.get("saturation_audit", {})
    require(
        int(saturation.get("R24_pending_demand_cap_current_code", -1)) == 13000,
        "R24 saturation cap is not frozen",
    )
    require(
        "cannot enter" in saturation.get("promotion_rule", ""),
        "model-saturated cells are not excluded from promotion",
    )

    catastrophic = contract["catastrophic_scenarios"]
    require(bool(catastrophic.get("excluded_from_primary_connected_region")), "catastrophic fixtures can select the primary region")
    require(catastrophic["single_R3"].get("event_count") == 1, "single R3 fixture is not singular")
    require("NOT_R3" in catastrophic["persistent_campaign"].get("label", ""), "persistent campaign is mislabeled as R3")

    timing = contract["restricted_privileged_timing_family"]
    require(not bool(timing.get("cross_regime_posture_reversal_required")), "obsolete cross-regime reversal prerequisite remains")
    require(timing.get("same_cell_estimand"), "same-cell timing estimand missing")
    require(timing.get("claim_name") == "restricted_PI_war_timing_ceiling", "timing ceiling is misnamed")

    metrics = contract["metrics"]
    require(metrics.get("primary") == "ret_excel_request_snapshot_v2", "canonical ReT is not primary")
    require(bool(metrics["secondary_temporal_panel"].get("report_only")), "temporal panel can select")

    region = contract["promotion_and_region_rule"]
    require("at least four adjacent passing cells" in region.get("component_pass", ""), "connected-region minimum missing")
    require("single positive cell" in region.get("single_cell_positive", ""), "single-cell non-promotion missing")

    papers = contract["learner_and_papers"]
    require(papers.get("learner_seed_block") is None, "learner seed block allocated prematurely")
    require(not bool(papers.get("learner_authorized")), "learner prematurely authorized")
    require(not bool(papers.get("paper2_confirmed")), "Paper 2 prematurely confirmed")
    require(not bool(papers.get("paper3_authorized")), "Paper 3 prematurely authorized")

    blocks = custody.get("blocks", [])
    ranges: list[set[int]] = []
    for block in blocks:
        start, end, count = int(block["start"]), int(block["end"]), int(block["count"])
        require(end - start + 1 == count, f"seed count mismatch: {block['label']}")
        ranges.append(set(range(start, end + 1)))
    for left, right in itertools.combinations(ranges, 2):
        require(left.isdisjoint(right), "seed blocks overlap")
    require(all(min(values) >= 7470001 for values in ranges), "seed block escapes the new 747 series")

    declared_contract = repo_root / str(custody["contract_path"])
    declared_prereg = repo_root / str(custody["preregistration_path"])
    require(declared_contract.resolve() == contract_path.resolve(), "custody points to another contract")
    require(declared_contract.is_file(), "custody contract missing")
    require(declared_prereg.is_file(), "custody preregistration missing")
    if declared_contract.is_file():
        require(_sha256(declared_contract) == custody["contract_sha256"], "contract hash mismatch")
    if declared_prereg.is_file():
        require(_sha256(declared_prereg) == custody["preregistration_sha256"], "preregistration hash mismatch")

    return {
        "schema_version": "war_stress_timing_freeze_verification_v1",
        "status": "PASS_WAR_STRESS_TIMING_FREEZE" if not failures else "FAIL_WAR_STRESS_TIMING_FREEZE",
        "contract_sha256": _sha256(contract_path),
        "custody_sha256": _sha256(custody_path),
        "expanded_primary_cell_count": len(cells),
        "scientific_seeds_opened": bool(custody.get("opened")),
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("contracts/war_stress_timing_atlas_v1.json"),
    )
    parser.add_argument(
        "--custody",
        type=Path,
        default=Path("research/paper2_exhaustive_search/war_stress_timing_seed_custody_20260716.json"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    verdict = verify(args.contract, args.custody, args.repo_root)
    rendered = json.dumps(verdict, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if verdict["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
