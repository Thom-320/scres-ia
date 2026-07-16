from __future__ import annotations

import json
from pathlib import Path

from scripts.verify_war_stress_gsa_execution_preflight import verify


ROOT = Path(__file__).resolve().parents[1]


def test_execution_preflight_passes_but_keeps_scientific_seeds_closed() -> None:
    result = verify(ROOT)
    assert result["status"] == "PASS_EXECUTOR_PREFLIGHT_SCIENTIFIC_BLOCKED"
    assert result["scientific_seeds_opened"] is False
    assert result["scientific_execution_authorized"] is False
    assert result["policy_templates"] == 50_202
    assert result["projected_scientific_episodes"] == 85_845_420


def test_command_manifest_has_watcher_first_and_no_scientific_authorization() -> None:
    payload = json.loads(
        (
            ROOT
            / "research/paper2_exhaustive_search/war_stress_gsa_command_manifest_20260716.json"
        ).read_text()
    )
    assert payload["benchmark"]["authorized"] is True
    assert payload["scientific"]["authorized"] is False
    assert payload["watcher"]["terminal_condition"].startswith("producer PGID/SID empty")
