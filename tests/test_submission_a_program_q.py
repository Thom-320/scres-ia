from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "papers" / "submission_a_program_q"


def test_source_of_truth_preserves_program_q_boundary() -> None:
    source = json.loads((PAPER / "source_of_truth.json").read_text())
    assert source["binding_verdict"] == "STOP_Q_NO_REPLICATED_LEARNED_ADAPTATION"
    assert source["metric"]["formula_rows"] == 47_546
    assert source["metric"]["mismatches"] == 0
    assert source["design"]["open_loop_calendars"] == 65_536
    assert source["design"]["risks_enabled"] is False
    assert source["design"]["direct_full_des_replays"] == 21_696
    latency = source["hardware_specific_latency"]
    assert latency["claim_status"] == "DESCRIPTIVE_HARDWARE_SPECIFIC_NO_OUTCOME_CLAIM"
    assert latency["observation_count"] == 24
    assert latency["reselected_structured_median_ms"] < latency["recurrent_ppo_median_ms"]
    assert any("Neural premium" in claim for claim in source["prohibited_claims"])
    assert any("Accumulated learning" in claim for claim in source["prohibited_claims"])


def test_confirmation_table_has_three_cells_and_positive_open_loop_lcbs() -> None:
    source = json.loads((PAPER / "source_of_truth.json").read_text())
    rows = source["confirmation_rows"]
    assert len(rows) == 3
    assert all(float(row[2]) > 0.01 for row in rows)
    assert all(row[6] == "10/10" for row in rows)


def test_generated_package_contains_all_planned_tables_and_figures() -> None:
    tables = sorted((PAPER / "generated" / "tables").glob("table*.tex"))
    figures = sorted((PAPER / "generated" / "figures").glob("figure*.pdf"))
    assert len(tables) == 6
    assert len(figures) == 4


def test_generator_is_byte_deterministic() -> None:
    tracked = sorted((PAPER / "generated").rglob("*")) + [PAPER / "source_of_truth.json"]
    before = {
        path.relative_to(PAPER): path.read_bytes()
        for path in tracked
        if path.is_file()
    }
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_submission_a_program_q.py")],
        cwd=ROOT,
        check=True,
    )
    after = {
        path.relative_to(PAPER): path.read_bytes()
        for path in tracked
        if path.is_file()
    }
    assert after == before


def test_deterministic_pdf_wrapper_is_tracked() -> None:
    wrapper = ROOT / "scripts" / "build_submission_a_pdf.py"
    assert wrapper.is_file()
    assert "SOURCE_DATE_EPOCH" in wrapper.read_text()
