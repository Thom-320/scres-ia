from __future__ import annotations

import json
import subprocess
import sys

import pytest

from supply_chain.garrido_replication import DEFAULT_RAW_WORKBOOKS


def test_run_garrido_excel_experiments_smoke(tmp_path):
    if not all(path.exists() for path in DEFAULT_RAW_WORKBOOKS):
        pytest.skip("Local Garrido raw workbooks are not available.")

    output_dir = tmp_path / "garrido_excel_static_smoke"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_garrido_excel_experiments.py",
            "--cf-range",
            "1,11",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        cwd=".",
        text=True,
        capture_output=True,
        timeout=120,
    )

    assert "mean_ret_excel_formula" in result.stdout
    payload = json.loads((output_dir / "summary.json").read_text())
    assert payload["primary_metric"] == "mean_ret_excel_formula"
    assert payload["n_cfis"] == 2
    assert payload["n_rows"] == 16
    assert payload["best_policy_overall"]["policy"] in {
        row["policy"] for row in payload["summary_by_policy"]
    }
    assert (output_dir / "rows.csv").exists()
    assert (output_dir / "summary_by_policy.csv").exists()
    assert (output_dir / "summary_by_family_policy.csv").exists()
