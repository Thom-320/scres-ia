from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.audit_garrido_ret_workbook import rank, summarize
from scripts.verify_paper_b_metric_gate import unresolved_decisions, validate_contract


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "contracts" / "paper_b_metric_gate_v1.json"


def test_metric_gate_is_fail_closed_and_sources_are_hashed() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert validate_contract(contract) == []
    assert contract["scientific_execution_authorized"] is False
    assert len(unresolved_decisions(contract)) >= 10
    assert all(
        len(digest) == 64 for digest in contract["source_evidence"].values()
    )


def test_metric_gate_rejects_false_pass_with_unresolved_decisions() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    contract["status"] = contract["pass_status"]
    assert "passed metric gate still has unresolved decisions" in validate_contract(
        contract
    )


def test_range_clipping_is_not_observation_deletion() -> None:
    summary = summarize([0.25, 1.000155])
    assert summary["mean_clipped_0_1"] == pytest.approx(0.625)
    assert summary["mean_excluding_above_1_diagnostic_only"] == pytest.approx(0.25)
    assert summary["mean_clipped_0_1"] != pytest.approx(
        summary["mean_excluding_above_1_diagnostic_only"]
    )


def test_clipped_ranking_can_remain_stable_while_deletion_ranking_changes() -> None:
    summaries = {
        "A": summarize([0.20, 1.000155]),
        "B": summarize([0.10, 0.59, 0.60]),
    }
    assert rank(summaries, "mean") == ["A", "B"]
    assert rank(summaries, "mean_clipped_0_1") == ["A", "B"]
    assert rank(summaries, "mean_excluding_above_1_diagnostic_only") == ["B", "A"]
