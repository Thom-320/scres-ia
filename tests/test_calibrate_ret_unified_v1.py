from __future__ import annotations

import json
from pathlib import Path

import scripts.calibrate_ret_unified_v1 as calibrate_ret_unified_v1


def test_parameter_grid_has_expected_cardinality() -> None:
    grid = calibrate_ret_unified_v1.parameter_grid()
    assert len(grid) == 27
    assert {row["theta_sc"] for row in grid} == {0.76, 0.78, 0.80}
    assert {row["theta_bc"] for row in grid} == {0.70, 0.75, 0.80}
    assert {row["kappa"] for row in grid} == {0.10, 0.15, 0.20}


def test_parameter_grid_accepts_custom_ranges() -> None:
    grid = calibrate_ret_unified_v1.parameter_grid(
        theta_sc_grid=[0.50, 0.60],
        theta_bc_grid=[0.20],
        kappa_grid=[0.20, 0.40, 0.80],
    )
    assert len(grid) == 6
    assert {row["theta_sc"] for row in grid} == {0.50, 0.60}
    assert {row["theta_bc"] for row in grid} == {0.20}
    assert {row["kappa"] for row in grid} == {0.20, 0.40, 0.80}


def test_select_best_combo_prefers_lowest_kappa_then_tie_breaks() -> None:
    rows = [
        {
            "theta_sc": 0.76,
            "theta_bc": 0.75,
            "beta": 12.0,
            "kappa": 0.15,
            "garrido_s2_minus_s3": 2.0,
            "accepted": True,
        },
        {
            "theta_sc": 0.78,
            "theta_bc": 0.80,
            "beta": 12.0,
            "kappa": 0.10,
            "garrido_s2_minus_s3": 1.5,
            "accepted": True,
        },
        {
            "theta_sc": 0.78,
            "theta_bc": 0.75,
            "beta": 12.0,
            "kappa": 0.10,
            "garrido_s2_minus_s3": 1.0,
            "accepted": True,
        },
    ]

    selected = calibrate_ret_unified_v1.select_best_combo(rows)
    assert selected["kappa"] == 0.10
    assert selected["theta_sc"] == 0.78
    assert selected["theta_bc"] == 0.75


def test_main_writes_calibration_outputs(monkeypatch, tmp_path: Path) -> None:
    output_json = tmp_path / "ret_unified.json"
    output_grid = tmp_path / "ret_unified_grid.csv"
    output_summary = tmp_path / "ret_unified_summary.json"

    fake_args = type(
        "Args",
        (),
        {
            "output_json": output_json,
            "output_grid_csv": output_grid,
            "output_summary": output_summary,
            "risk_level": "increased",
            "stochastic_pt": True,
            "observation_version": "v4",
            "step_size_hours": 168.0,
            "max_steps": 260,
            "eval_episodes": 3,
            "seeds": [11, 22, 33],
            "theta_sc_grid": [0.78],
            "theta_bc_grid": [0.75],
            "kappa_grid": [0.10],
        },
    )()

    fake_rows = [
        {
            "theta_sc": 0.78,
            "theta_bc": 0.75,
            "beta": 12.0,
            "kappa": 0.10,
            "garrido_cf_s1_reward_mean": 10.0,
            "garrido_cf_s2_reward_mean": 12.0,
            "garrido_cf_s3_reward_mean": 11.0,
            "static_s1_reward_mean": 9.0,
            "static_s2_reward_mean": 11.0,
            "static_s3_reward_mean": 10.0,
            "garrido_rank_ok": True,
            "static_rank_ok": True,
            "garrido_s2_minus_s3": 1.0,
            "garrido_margin_ok": True,
            "accepted": True,
        }
    ]

    monkeypatch.setattr(calibrate_ret_unified_v1, "parse_args", lambda: fake_args)
    monkeypatch.setattr(
        calibrate_ret_unified_v1,
        "parameter_grid",
        lambda **kwargs: [
            {"theta_sc": 0.78, "theta_bc": 0.75, "beta": 12.0, "kappa": 0.10}
        ],
    )
    monkeypatch.setattr(
        calibrate_ret_unified_v1,
        "evaluate_combo",
        lambda params, args: fake_rows[0],
    )

    calibrate_ret_unified_v1.main()

    calibration = json.loads(output_json.read_text(encoding="utf-8"))
    summary = json.loads(output_summary.read_text(encoding="utf-8"))
    csv_header = output_grid.read_text(encoding="utf-8").splitlines()[0]

    assert calibration["theta_sc"] == 0.78
    assert calibration["theta_bc"] == 0.75
    assert calibration["kappa"] == 0.10
    assert summary["accepted_candidates"] == 1
    assert "theta_sc" in csv_header
