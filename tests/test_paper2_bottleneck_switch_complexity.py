import json
from hashlib import sha256

import numpy as np

from scripts.launch_paper2_switch_complexity import RESULT_ROOT
from scripts.run_paper2_bottleneck_full_frontier import (
    PRIMARY_CONTRACT_PATH,
    _contract_seed_rows,
    calendar_index,
)
from scripts.search_paper2_bottleneck_switch_complexity import (
    CONTRACT_PATH,
    EXPECTED_COUNTS,
    candidate_calendars,
    evaluate_selected_tape,
    exact_selection,
    switch_count,
)
from scripts.verify_paper2_bottleneck_switch_complexity import VERIFICATION_SCHEMA
from scripts.watch_paper2_switch_complexity import snapshot


def test_complete_at_most_three_switch_family_is_exact_and_feasible():
    rows = candidate_calendars()
    contract = json.loads(CONTRACT_PATH.read_text())
    counts = {
        switches: sum(switch_count(row) == switches for row in rows)
        for switches in EXPECTED_COUNTS
    }
    assert len(rows) == contract["candidate_family"]["candidate_count"] == 11_611
    assert counts == EXPECTED_COUNTS
    assert len(set(rows)) == 11_611
    assert rows[0] == (0,) * 24
    assert all(calendar_index(row) >= 0 for row in rows)
    assert all(
        not (
            row[index] != row[index - 1]
            and row[index - 1] != row[index - 2]
        )
        for row in rows
        for index in range(2, 24)
    )


def test_exact_fraction_selection_uses_minimum_index_for_ties():
    matrix = np.zeros((60, 11_611), dtype=float)
    matrix[:, 7] = 0.25
    matrix[:, 9] = 0.25
    sums, selected = exact_selection(matrix)
    assert selected == 7
    assert sums[7] == sums[9]
    assert sums[7] > sums[0]


def test_selected_replay_retains_calibration_guardrails_and_resources():
    primary = json.loads(PRIMARY_CONTRACT_PATH.read_text())
    first = _contract_seed_rows(primary, "calibration")[0]
    row = evaluate_selected_tape(
        0,
        int(first["seed"]),
        str(first["context_0"]),
        (0,) * 24,
    )
    guardrails = row["guardrails_and_resources"]
    assert row["split"] == "calibration"
    assert guardrails["total_token_hours"] == 4032.0
    assert guardrails["mass_residual"] == 0.0
    assert guardrails["reserve_inventory_initial"] == 10_000.0
    assert "lost_orders" in guardrails
    assert "service_loss_auc_ration_hours" in guardrails


def test_watcher_distinguishes_prestart_complete_and_failure(tmp_path):
    assert snapshot(
        tmp_path, watcher_started="2026-07-13T00:00:00+00:00"
    )["state"] == "watching_prestart"
    result = tmp_path / "result.json"
    result.write_text('{"ok":true}\n')
    result_sha = sha256(result.read_bytes()).hexdigest()
    (tmp_path / "progress.json").write_text(json.dumps({
        "stage": "complete", "output_sha256": result_sha,
    }))
    (tmp_path / "pid.json").write_text(json.dumps({
        "scientific_pid": 999_999_999, "output": str(result),
    }))
    assert snapshot(
        tmp_path, watcher_started="2026-07-13T00:00:00+00:00"
    )["state"] == "completed_unverified"
    (tmp_path / "progress.json").write_text(json.dumps({
        "stage": "complete", "output_sha256": "0" * 64,
    }))
    assert snapshot(
        tmp_path, watcher_started="2026-07-13T00:00:00+00:00"
    )["state"] == "failed_or_incomplete"


def test_contract_and_execution_scope_fail_closed():
    contract = json.loads(CONTRACT_PATH.read_text())
    rules = contract["decision_rules"]
    assert contract["calibration"]["locked_seed_access_forbidden"] is True
    assert all(rules[field] is False for field in (
        "h_pi_computed", "h_obs_computed", "w24_authorized",
        "learner_authorized", "paper2_authorized", "paper3_authorized",
    ))
    assert RESULT_ROOT.name == "switch_complexity_screen"
    assert VERIFICATION_SCHEMA == "paper2_bottleneck_switch_complexity_verification_v1"
