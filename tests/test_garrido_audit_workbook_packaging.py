from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from scripts.build_garrido_des_audit_workbooks import build_payload


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_build_payload_summarizes_replication_dir(tmp_path: Path) -> None:
    replication_dir = tmp_path / "audit"
    export_dir = replication_dir / "des_order_exports"
    replication_dir.mkdir()
    best = {
        "demand_source": "excel_order_tape",
        "risk_occurrence_mode": "thesis_window",
        "risk_attribution_source": "excel_risk_tape",
        "seed_stream_mode": "split",
    }
    row = {
        "cfi": 1,
        "family": "risk_r1",
        **best,
        "target_ret_excel_mean": 0.006,
        "sim_ret_excel_formula_mean": 0.005,
        "signed_ret_gap": -0.001,
        "abs_ret_gap": 0.001,
        "target_n_orders": 2,
        "sim_n_orders": 2,
        "n_order_gap": 0,
        "q_max_abs_gap": 0.0,
        "optj_max_abs_gap": 0.0,
        "max_branch_share_gap_pct": 0.0,
        "horizon_hours": 10.0,
        "target_horizon_hours": 10.0,
        "workbook_seed": 7,
    }
    (replication_dir / "replication_audit.json").write_text(
        json.dumps(
            {
                "formula": "formula",
                "best_config": best,
                "best_summary": {"mean_abs_ret_gap": 0.001},
                "formula_audit": {"total_rows": 47546, "total_mismatches": 0},
                "gates": {"replication_status": "passed_gate"},
                "replication_status": "passed_gate",
                "rows": [row],
            }
        ),
        encoding="utf-8",
    )
    _write_csv(
        export_dir / "CF01_excel_order_tape_thesis_window_excel_risk_tape_split.csv",
        [
            {
                "Q": 1,
                "j": 1,
                "OPTj": 1,
                "OATj": 2,
                "CTj": 1,
                "LT": 48,
                "sumBt": 0,
                "APj": 0,
                "RPj": 10,
                "DPj": 1,
                "R11_1": 1,
                "R14": 2,
                "sumUt": 0,
                "OP9": "",
                "ReT": 0.05,
                "deltaReT": 0.0,
                "excel_case": "excel_recovery",
            },
            {
                "Q": 1,
                "j": 2,
                "OPTj": 2,
                "OATj": 3,
                "CTj": 1,
                "LT": 48,
                "sumBt": 0,
                "APj": 1,
                "RPj": 0,
                "DPj": 1,
                "R11_1": 0,
                "R14": 0,
                "sumUt": 0,
                "OP9": "",
                "ReT": 0.02,
                "deltaReT": -0.03,
                "excel_case": "excel_autotomy",
            },
        ],
    )
    args = argparse.Namespace(
        replication_dir=replication_dir,
        raw_workbooks=[],
        rsult_workbook=Path("Rsult_1.xlsx"),
        selected_cfs="1",
        sample_rows_per_cf=10,
    )

    payload = build_payload(args)

    assert payload["replication"]["replication_status"] == "passed_gate"
    assert payload["cf_summary"][0]["CF"] == "CF01"
    assert payload["risk_attribution"][0]["risk_active_share"] == 0.5
    assert payload["deltas"][0]["audit_flags"] == "OK"
    assert len(payload["selected_ledgers"]) == 2
    assert "CF01" in payload["ledger_rows_by_cf"]
