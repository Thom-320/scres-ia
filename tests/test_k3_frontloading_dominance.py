from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.certify_k3_frontloading_dominance import (
    OUTPUT_PATH,
    _front_loaded_schedule,
    _prefix_sums,
    audit_source_semantics,
    build_certificate,
    enumerate_effective_resource_graph,
    verify_under_budget_envelope_dominance,
    visible_ledger_nonmonotonicity_counterexample,
)


ROOT = Path(__file__).resolve().parents[1]


def test_exact_effective_resource_graph_and_unique_front_loaded_schedule():
    graph = enumerate_effective_resource_graph()
    assert len(graph.states) == 61
    assert len(graph.edges) == 260
    assert len(graph.schedules) == 6_371
    assert {state for state in graph.states if state[0] == 8} == {(8, 40)}
    assert {sum(schedule) for schedule in graph.schedules} == {40}

    front = _front_loaded_schedule(graph)
    assert front == (6, 6, 6, 6, 6, 6, 4, 0)
    assert front in graph.schedules
    front_prefix = _prefix_sums(front)
    nondominated = []
    for schedule in graph.schedules:
        margins = [
            front_value - candidate_value
            for front_value, candidate_value in zip(
                front_prefix, _prefix_sums(schedule), strict=True
            )
        ]
        assert min(margins) >= 0
        if not any(margin > 0 for margin in margins):
            nondominated.append(schedule)
    assert nondominated == [front]


def test_source_semantics_fail_closed_bindings_pass():
    audit = audit_source_semantics()
    assert audit["status"] == "PASS"
    assert all(audit["checks"].values())
    assert audit["checks"]["identical_initial_inventory_one_D0"]
    assert audit["checks"]["rtape_generator_demand_is_nonnegative"]
    assert audit["checks"]["order_record_risk_periods_default_zero"]
    assert audit["checks"]["order_record_risk_indicators_default_empty"]
    assert set(audit["source_sha256"]) == {
        "contracts/program_k3_ret_budgeted_replenishment_v1.json",
        "supply_chain/program_g.py",
        "supply_chain/replenish.py",
        "supply_chain/replenish_ret.py",
        "supply_chain/ret_thesis.py",
        "supply_chain/supply_chain.py",
    }


def test_all_non_superior_resource_calendars_are_prefix_dominated():
    graph = enumerate_effective_resource_graph()
    front = _front_loaded_schedule(graph)
    audit = verify_under_budget_envelope_dominance(graph, front)
    assert audit["schedule_count_total_spend_le_budget"] == 5_758_374
    assert audit["exact_budget_schedule_count"] == 6_371
    assert audit["strictly_under_budget_schedule_count"] == 5_752_003
    assert audit["prefix_dominance_violation_count"] == 0
    assert "not described as pairwise equal-resource" in audit["resource_scope"]


def test_visible_ledger_is_not_monotone_in_completion_times_even_with_equal_losses():
    counterexample = visible_ledger_nonmonotonicity_counterexample()
    assert counterexample["status"] == "PASS_LIVE_AGGREGATOR_COUNTEREXAMPLE"
    assert counterexample["front_loaded_path"]["lost_order_ids"] == [8]
    assert counterexample["later_batched_path"]["lost_order_ids"] == [8]
    assert counterexample["front_loaded_path"]["mean_ret_excel"] == 6 / 7
    assert counterexample["later_batched_path"]["mean_ret_excel"] == 1.0
    assert "does not estimate K3 H_PI" in counterexample["claim_limit"]


def test_full_ledger_hpi_zero_and_visible_metric_scope_are_explicit():
    certificate = build_certificate()
    metric = certificate["full_ledger_metric_monotonicity"]
    assert metric["status_vectors_checked_against_live_aggregator"] == 6_561
    assert metric["coordinatewise_status_pairs_checked"] == 1_679_616
    assert metric["abstraction_mismatch_count"] == 0
    assert metric["metric_dominance_violation_count"] == 0

    full = certificate["metric_conclusions"]["frozen_k3_full_ledger"]
    assert full["metric"] == "ret_excel_full_ledger_order"
    assert full["h_pi"] == 0.0
    assert full["h_obs"] == 0.0
    assert full["learned_incremental_value_upper_bound"] == 0.0

    visible = certificate["metric_conclusions"]["ret_excel_visible_v1"]
    assert visible["result"] == "NOT_UNCONDITIONALLY_CERTIFIED_BY_THIS_THEOREM"
    assert "exact equality of lost counts" in visible["mandatory_guardrail"]
    assert "separate metric-specific proof/audit" in visible["paper2_rule"]
    assert certificate["generated_without_stochastic_tapes"] is True


def test_direct_cli_needs_no_pythonpath_and_checked_json_is_current():
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    command = [
        sys.executable,
        str(ROOT / "scripts" / "certify_k3_frontloading_dominance.py"),
        "--check",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["states"] == 61
    assert summary["schedules"] == 6_371
    assert summary["prefix_violations"] == 0
    assert OUTPUT_PATH.exists()


def test_cli_accepts_an_explicit_output_outside_repository(tmp_path):
    destination = tmp_path / "k3-certificate.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "certify_k3_frontloading_dominance.py"),
            "--output",
            str(destination),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["output"] == str(destination)
    assert json.loads(destination.read_text())["resource_graph"][
        "effective_exact_budget_schedule_count"
    ] == 6_371
