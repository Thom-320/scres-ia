from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def test_runner_fails_before_seed_access_when_risk_evidence_is_missing(tmp_path: Path) -> None:
    output = tmp_path / "out"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_restricted_pi_timing_ceiling.py",
            "--risk-result",
            str(tmp_path / "missing-result.json"),
            "--risk-raw-rows",
            str(tmp_path / "missing-rows.csv"),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    result = json.loads((output / "result.json").read_text())
    assert result["status"] == "STOP_BEFORE_TIMING_RISK_EVIDENCE_MISSING"
    assert result["timing_seeds_opened"] is False


def test_runner_stops_without_opening_timing_seeds_on_risk_null(tmp_path: Path) -> None:
    risk_result = tmp_path / "risk-result.json"
    risk_rows = tmp_path / "rows.csv"
    risk_result.write_text(json.dumps({"status": "DEVELOPMENT_NO_DOOR_UNDER_TESTED_FRONTIER"}))
    risk_rows.write_text("unused\n")
    output = tmp_path / "out"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_restricted_pi_timing_ceiling.py",
            "--risk-result",
            str(risk_result),
            "--risk-raw-rows",
            str(risk_rows),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    result = json.loads((output / "result.json").read_text())
    assert result["status"] == "STOP_BEFORE_TIMING_NO_RISK_DOOR"
    assert result["timing_seeds_opened"] is False
