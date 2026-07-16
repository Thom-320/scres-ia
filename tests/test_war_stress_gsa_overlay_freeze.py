from __future__ import annotations

import json
from pathlib import Path

from scripts.verify_war_stress_gsa_overlay import verify


ROOT = Path(__file__).resolve().parents[1]


def test_overlay_freeze_is_complete_and_unopened() -> None:
    verdict = verify(ROOT)
    assert verdict["status"] == "PASS_WAR_STRESS_GSA_OVERLAY_FREEZE"
    assert verdict["morris_configurations"] == 720
    assert verdict["qmc_pool_configurations"] == 1536
    assert verdict["scientific_seeds_opened"] is False
    assert verdict["failures"] == []


def test_overlay_manifest_contains_every_stratum_once_per_design_point() -> None:
    path = (
        ROOT
        / "research/paper2_exhaustive_search/war_stress_gsa_overlay_manifest_20260716.json"
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for family, expected_per_stratum in (("qmc_pool", 128),):
        counts: dict[tuple[str, str], int] = {}
        for row in manifest[family]["rows"]:
            key = (row["mask"], row["coupling"])
            counts[key] = counts.get(key, 0) + 1
        assert len(counts) == 12
        assert set(counts.values()) == {expected_per_stratum}

    morris_expected = {
        "LOC_SURGE": 50,
        "THEATER_CAPACITY_SURGE": 70,
        "PRODUCTION_QUALITY_SURGE": 60,
    }
    counts = {}
    for row in manifest["morris"]["rows"]:
        key = (row["mask"], row["coupling"])
        counts[key] = counts.get(key, 0) + 1
    assert len(counts) == 12
    assert all(count == morris_expected[mask] for (mask, _), count in counts.items())
