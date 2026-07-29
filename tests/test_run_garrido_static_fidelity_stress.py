import csv
import json

from scripts.analyze_garrido_fidelity_outputs import (
    analyze_run,
    binomial_positive_p_value,
    format_list_cell,
    iter_run_dirs,
)
from scripts.run_garrido_static_fidelity_stress import policy_candidates


def test_matched_only_policy_set_contains_only_garrido_baseline() -> None:
    policies = policy_candidates("matched_only")

    assert [policy["name"] for policy in policies] == ["garrido_matched_DOE_baseline"]
    assert policies[0]["kind"] == "matched_doe"


def test_minimal_policy_set_keeps_static_comparison_controls() -> None:
    names = {policy["name"] for policy in policy_candidates("minimal")}

    assert "garrido_matched_DOE_baseline" in names
    assert "pure_inventory_I0_S1" in names
    assert "pure_inventory_I672_S1" in names
    assert "pure_capacity_I0_S1" in names
    assert "pure_capacity_I0_S3" in names


def test_analyze_garrido_fidelity_outputs_writes_gate_summary(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rows = [
        {
            "profile": profile,
            "family": "inventory",
            "policy": "garrido_matched_DOE_baseline",
            "policy_kind": "matched_doe",
            "cfi": 31,
            "fill_rate_order_level": fill,
            "order_level_ret_mean": ret,
            "cumulative_disruption_hours": disruption,
        }
        for profile, fill, ret, disruption in [
            ("current", 0.99, 0.90, 10.0),
            ("increased", 0.90, 0.80, 20.0),
            ("severe", 0.80, 0.70, 30.0),
            ("severe_extended", 0.70, 0.60, 40.0),
        ]
    ]
    rows.extend(
        [
            {
                "profile": "thesis_pattern",
                "family": "inventory",
                "policy": "pure_inventory_I0_S1",
                "policy_kind": "pure_inventory",
                "cfi": 31,
                "fill_rate_order_level": 0.70,
                "order_level_ret_mean": 0.50,
                "cumulative_disruption_hours": 20.0,
            },
            {
                "profile": "thesis_pattern",
                "family": "inventory",
                "policy": "pure_inventory_I672_S1",
                "policy_kind": "pure_inventory",
                "cfi": 31,
                "fill_rate_order_level": 0.80,
                "order_level_ret_mean": 0.70,
                "cumulative_disruption_hours": 20.0,
            },
            {
                "profile": "thesis_pattern",
                "family": "inventory",
                "policy": "pure_capacity_I0_S1",
                "policy_kind": "pure_capacity",
                "cfi": 31,
                "fill_rate_order_level": 0.72,
                "order_level_ret_mean": 0.52,
                "cumulative_disruption_hours": 20.0,
            },
            {
                "profile": "thesis_pattern",
                "family": "inventory",
                "policy": "pure_capacity_I0_S3",
                "policy_kind": "pure_capacity",
                "cfi": 31,
                "fill_rate_order_level": 0.82,
                "order_level_ret_mean": 0.72,
                "cumulative_disruption_hours": 20.0,
            },
        ]
    )
    with (run_dir / "episode_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    payload = analyze_run(run_dir)
    saved = json.loads((run_dir / "fidelity_gate_analysis.json").read_text())

    assert payload["episode_count"] == len(rows)
    assert saved["h1"][0]["status"] == "passed"
    assert saved["h2"][0]["fill_positive"] == 1
    assert saved["h2"][0]["fill_positive_p_binom_one_sided"] == 0.5
    assert saved["h2"][0]["ret_positive"] == 1
    assert saved["h2"][0]["ret_positive_p_binom_one_sided"] == 0.5
    assert saved["h3"][0]["fill_positive"] == 1
    assert saved["h3"][0]["ret_positive"] == 1
    assert (run_dir / "FIDELITY_GATE_ANALYSIS.md").exists()
    assert iter_run_dirs([tmp_path]) == [run_dir]
    assert format_list_cell([0.123456, "increased"]) == "0.1235 -> increased"


def test_binomial_positive_p_value_is_one_sided_sign_test() -> None:
    assert binomial_positive_p_value(10, 10) == 1 / 1024
    assert binomial_positive_p_value(9, 10) == 11 / 1024
