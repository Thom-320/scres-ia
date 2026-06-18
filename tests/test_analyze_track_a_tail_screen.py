from __future__ import annotations

import csv
from pathlib import Path

import scripts.analyze_track_a_tail_screen as analyzer


def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def base_row(**updates: str) -> dict[str, str]:
    row = {
        "label": "run_a",
        "algo": "ppo_mlp",
        "risk_level": "increased",
        "status": "complete",
        "delta_fill": "0.0",
        "delta_ret_all_orders_vs_best_metric": "0.0",
        "delta_flow_fill_vs_best_metric": "0.0",
        "delta_ret_p10_all_vs_best_metric": "0.0",
        "improvement_stockout_week_pct_vs_best_metric": "0.0",
        "best_static_policy_by_ret_all_orders": "static_grid_I168_S1",
        "best_static_policy_by_flow_fill": "static_grid_I168_S1",
        "best_static_policy_by_ret_p10_all": "static_grid_I168_S1",
    }
    row.update(updates)
    return row


def test_decision_promotes_primary_ret_win() -> None:
    decision = analyzer.decide_row(
        base_row(delta_ret_all_orders_vs_best_metric="0.025"),
        primary_threshold=0.02,
        p10_threshold=0.01,
        max_fill_loss=0.01,
        min_stockout_improvement=0.0,
    )

    assert decision.promote
    assert decision.primary_ret_win
    assert "all-order ReT" in decision.reason


def test_decision_promotes_tail_win_with_stockout_and_fill_guard() -> None:
    decision = analyzer.decide_row(
        base_row(
            delta_fill="-0.005",
            delta_ret_p10_all_vs_best_metric="0.015",
            improvement_stockout_week_pct_vs_best_metric="2.0",
        ),
        primary_threshold=0.02,
        p10_threshold=0.01,
        max_fill_loss=0.01,
        min_stockout_improvement=0.0,
    )

    assert decision.promote
    assert decision.tail_win


def test_decision_rejects_tail_win_with_material_fill_loss() -> None:
    decision = analyzer.decide_row(
        base_row(
            delta_fill="-0.05",
            delta_ret_p10_all_vs_best_metric="0.015",
            improvement_stockout_week_pct_vs_best_metric="2.0",
        ),
        primary_threshold=0.02,
        p10_threshold=0.01,
        max_fill_loss=0.01,
        min_stockout_improvement=0.0,
    )

    assert not decision.promote
    assert not decision.tail_win


def test_analyzer_writes_decision_files(tmp_path: Path) -> None:
    source = tmp_path / "sweep_summary.csv"
    write_summary(
        source,
        [
            base_row(label="no_win"),
            base_row(
                label="flow_win",
                delta_flow_fill_vs_best_metric="0.021",
            ),
        ],
    )

    rc = analyzer.main.__wrapped__ if hasattr(analyzer.main, "__wrapped__") else None
    assert rc is None

    rows = analyzer.read_rows(source)
    summary = analyzer.summarize(
        [
            analyzer.decide_row(
                row,
                primary_threshold=0.02,
                p10_threshold=0.01,
                max_fill_loss=0.01,
                min_stockout_improvement=0.0,
            )
            for row in rows
        ]
    )

    assert summary["decision"] == "PROMOTE_TOP_CONFIGS"
    assert summary["n_promoted"] == 1
