from __future__ import annotations

import json
import subprocess
import sys

import pytest

from supply_chain.garrido_replication import DEFAULT_RAW_WORKBOOKS


def test_replicate_garrido_excel_smoke_with_order_tape(tmp_path):
    if not all(path.exists() for path in DEFAULT_RAW_WORKBOOKS):
        pytest.skip("Local Garrido raw workbooks are not available.")

    output_dir = tmp_path / "replication_smoke"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/replicate_garrido_excel.py",
            "--cf-range",
            "1,11",
            "--demand-sources",
            "excel_order_tape",
            "--risk-occurrence-modes",
            "legacy_renewal",
            "--risk-attribution-sources",
            "des_events,excel_risk_tape",
            "--seed-stream-modes",
            "split",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        cwd=".",
        text=True,
        capture_output=True,
        timeout=120,
    )

    assert "Saved:" in result.stdout
    payload = json.loads((output_dir / "replication_audit.json").read_text())
    assert payload["best_config"] == {
        "demand_source": "excel_order_tape",
        "risk_occurrence_mode": "legacy_renewal",
        "risk_attribution_source": "excel_risk_tape",
        "seed_stream_mode": "split",
    }
    assert payload["gates"]["extraction_gate_passed"] is True
    assert payload["gates"]["operational_order_gate_passed"] is True
    assert payload["gates"]["operational_horizon_gate_passed"] is True
    assert payload["replication_status"] == "passed_gate"
    assert payload["gates"]["replication_status"] == "passed_gate"
    assert len(payload["order_exports"]) == 2
