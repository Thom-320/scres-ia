from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

import scripts.generate_family_a_summary as family_a_summary


def _write_csv(
    path: Path, fieldnames: list[str], rows: list[dict[str, object]]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_run_bundle(
    run_dir: Path,
    *,
    reward_mode: str,
    fill_rate: float,
    static_s2_fill_rate: float,
    backorder_rate: float,
    static_s2_backorder_rate: float,
    order_level_ret: float,
    static_s2_order_level_ret: float,
    reward_total: float,
    static_s2_reward_total: float,
    shift_mix: tuple[float, float, float],
    ret_seq_kappa: float | None = None,
    year_basis: str = "thesis",
    eval_episodes: int = 10,
    seeds: list[int] | None = None,
    severe_fill_rate: float | None = None,
    severe_static_s2_fill_rate: float | None = None,
    severe_static_s3_fill_rate: float | None = None,
) -> None:
    seeds = seeds or [11, 22, 33, 44, 55]
    reward_family = (
        "resilience_index" if reward_mode == "ReT_seq_v1" else "operational_penalty"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "config": {
                    "algo": "ppo",
                    "reward_mode": reward_mode,
                    "ret_seq_kappa": ret_seq_kappa,
                    "seeds": seeds,
                    "eval_episodes": eval_episodes,
                },
                "backbone": {
                    "git_commit": "abc123",
                    "observation_version": "v1",
                    "year_basis": year_basis,
                    "risk_level": "increased",
                    "stochastic_pt": True,
                },
                "reward_contract": {
                    "reward_family": reward_family,
                },
            }
        ),
        encoding="utf-8",
    )
    rows = [
        {
            "phase": "static_screen",
            "policy": "static_s2",
            "fill_rate_mean": str(static_s2_fill_rate),
            "backorder_rate_mean": str(static_s2_backorder_rate),
            "order_level_ret_mean_mean": str(static_s2_order_level_ret),
            "reward_total_mean": str(static_s2_reward_total),
            "pct_steps_S1_mean": "0.0",
            "pct_steps_S2_mean": "100.0",
            "pct_steps_S3_mean": "0.0",
        },
        {
            "phase": "ppo_eval",
            "policy": "ppo",
            "fill_rate_mean": str(fill_rate),
            "backorder_rate_mean": str(backorder_rate),
            "order_level_ret_mean_mean": str(order_level_ret),
            "reward_total_mean": str(reward_total),
            "pct_steps_S1_mean": str(shift_mix[0]),
            "pct_steps_S2_mean": str(shift_mix[1]),
            "pct_steps_S3_mean": str(shift_mix[2]),
        },
    ]
    if severe_fill_rate is not None:
        rows.extend(
            [
                {
                    "phase": "cross_eval_severe",
                    "policy": "static_s2",
                    "fill_rate_mean": str(severe_static_s2_fill_rate),
                    "backorder_rate_mean": "0.50",
                    "order_level_ret_mean_mean": "0.16",
                    "reward_total_mean": str(static_s2_reward_total),
                    "pct_steps_S1_mean": "0.0",
                    "pct_steps_S2_mean": "100.0",
                    "pct_steps_S3_mean": "0.0",
                },
                {
                    "phase": "cross_eval_severe",
                    "policy": "static_s3",
                    "fill_rate_mean": str(severe_static_s3_fill_rate),
                    "backorder_rate_mean": "0.51",
                    "order_level_ret_mean_mean": "0.15",
                    "reward_total_mean": str(static_s2_reward_total - 1.0),
                    "pct_steps_S1_mean": "0.0",
                    "pct_steps_S2_mean": "0.0",
                    "pct_steps_S3_mean": "100.0",
                },
                {
                    "phase": "cross_eval_severe",
                    "policy": "ppo",
                    "fill_rate_mean": str(severe_fill_rate),
                    "backorder_rate_mean": "0.52",
                    "order_level_ret_mean_mean": "0.17",
                    "reward_total_mean": str(reward_total - 10.0),
                    "pct_steps_S1_mean": "70.0",
                    "pct_steps_S2_mean": "15.0",
                    "pct_steps_S3_mean": "15.0",
                },
            ]
        )
    _write_csv(run_dir / "policy_summary.csv", list(rows[0].keys()), rows)


def test_build_primary_and_severe_rows_extract_expected_metrics(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "paper_ret_seq_k020_500k"
    _write_run_bundle(
        run_dir,
        reward_mode="ReT_seq_v1",
        ret_seq_kappa=0.2,
        fill_rate=0.788,
        static_s2_fill_rate=0.792,
        backorder_rate=0.212,
        static_s2_backorder_rate=0.208,
        order_level_ret=0.2016,
        static_s2_order_level_ret=0.2014,
        reward_total=133.0,
        static_s2_reward_total=132.5,
        shift_mix=(73.0, 13.0, 14.0),
        severe_fill_rate=0.484,
        severe_static_s2_fill_rate=0.495,
        severe_static_s3_fill_rate=0.494,
    )

    primary = family_a_summary.build_primary_row(run_dir)
    severe = family_a_summary.build_severe_row(run_dir)

    assert primary["reward_family"] == "resilience_index"
    assert primary["delta_fill_vs_static_s2"] == pytest.approx(-0.004)
    assert primary["delta_order_level_ret_vs_static_s2"] == pytest.approx(0.0002)
    assert severe["severe_available"] is True
    assert severe["delta_fill_vs_static_s2"] == pytest.approx(-0.011)
    assert severe["delta_fill_vs_static_s3"] == pytest.approx(-0.010)


def test_main_writes_family_a_bundle_and_marks_secondary_non_comparable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    k020 = tmp_path / "paper_ret_seq_k020_500k"
    k010 = tmp_path / "paper_ret_seq_k010_500k"
    control = tmp_path / "paper_control_v1_500k"
    secondary = tmp_path / "final_ret_seq_v1_500k"
    output_dir = tmp_path / "family_a_summary"

    _write_run_bundle(
        k020,
        reward_mode="ReT_seq_v1",
        ret_seq_kappa=0.2,
        fill_rate=0.7883,
        static_s2_fill_rate=0.7922,
        backorder_rate=0.2117,
        static_s2_backorder_rate=0.2078,
        order_level_ret=0.1996,
        static_s2_order_level_ret=0.2015,
        reward_total=133.08,
        static_s2_reward_total=132.53,
        shift_mix=(72.9, 13.4, 13.7),
        severe_fill_rate=0.4835,
        severe_static_s2_fill_rate=0.4947,
        severe_static_s3_fill_rate=0.4940,
    )
    _write_run_bundle(
        k010,
        reward_mode="ReT_seq_v1",
        ret_seq_kappa=0.1,
        fill_rate=0.7881,
        static_s2_fill_rate=0.7923,
        backorder_rate=0.2119,
        static_s2_backorder_rate=0.2076,
        order_level_ret=0.2012,
        static_s2_order_level_ret=0.2006,
        reward_total=133.05,
        static_s2_reward_total=133.77,
        shift_mix=(65.2, 14.5, 20.3),
    )
    _write_run_bundle(
        control,
        reward_mode="control_v1",
        fill_rate=0.7820,
        static_s2_fill_rate=0.7923,
        backorder_rate=0.2180,
        static_s2_backorder_rate=0.2076,
        order_level_ret=0.1988,
        static_s2_order_level_ret=0.2006,
        reward_total=-629.36,
        static_s2_reward_total=-617.98,
        shift_mix=(45.5, 27.8, 26.7),
    )
    _write_run_bundle(
        secondary,
        reward_mode="ReT_seq_v1",
        ret_seq_kappa=0.2,
        fill_rate=0.7934,
        static_s2_fill_rate=0.7939,
        backorder_rate=0.2066,
        static_s2_backorder_rate=0.2061,
        order_level_ret=0.2007,
        static_s2_order_level_ret=0.2008,
        reward_total=133.85,
        static_s2_reward_total=133.98,
        shift_mix=(54.9, 21.7, 23.4),
        year_basis="gregorian",
        eval_episodes=20,
        seeds=[11, 22, 33, 44, 55, 66, 77, 88, 99, 100],
        severe_fill_rate=0.4870,
        severe_static_s2_fill_rate=0.4951,
        severe_static_s3_fill_rate=0.4947,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_family_a_summary.py",
            "--run-dirs",
            str(k020),
            str(k010),
            str(control),
            "--secondary-comparator",
            str(secondary),
            "--output-dir",
            str(output_dir),
        ],
    )

    family_a_summary.main()

    payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    comparator_rows = payload["secondary_comparator_rows"]
    severe_rows = payload["severe_rows"]

    assert (output_dir / "primary_comparison.csv").exists()
    assert (output_dir / "severe_cross_eval.csv").exists()
    assert (output_dir / "secondary_comparators.csv").exists()
    assert (output_dir / "summary.md").exists()
    assert payload["leader"]["label"] == "paper_ret_seq_k020_500k"
    assert comparator_rows[0]["service_metrics_comparable_to_primary"] is False
    assert comparator_rows[0]["raw_reward_comparable_to_primary"] is False
    assert "year_basis differs" in comparator_rows[0]["notes"]
    assert severe_rows[1]["severe_available"] is False
    assert severe_rows[2]["severe_available"] is False
