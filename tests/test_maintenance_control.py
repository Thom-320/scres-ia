from __future__ import annotations

import pytest

from supply_chain.maintenance_control import (
    ACTIONS, FORBIDDEN_OBSERVATIONS, OBSERVATION_KEYS, MaintenanceController,
    default_cell, make_sim, materialize_tape, periodic_policy,
    periodic_sequences, run_policy, worst_condition_policy,
)


def test_observation_schema_is_fail_closed_and_unprivileged() -> None:
    tape = materialize_tape(1200001, weeks=2)
    sim, controller, _ = make_sim(tape)
    obs = controller.observation()
    assert tuple(obs) == OBSERVATION_KEYS
    assert not set(obs).intersection(FORBIDDEN_OBSERVATIONS)
    assert all(isinstance(value, float) for value in obs.values())


def test_finite_wip_conserves_mass_and_records_blocking_or_starvation() -> None:
    tape = materialize_tape(1200002, weeks=3)
    cell = default_cell()
    cell["wip_capacity_days"] = 1
    result = run_policy(tape, periodic_policy(("PM5",)), cell=cell)
    assert result["mass_residual"] == pytest.approx(0.0, abs=1e-5)
    assert sum(result["blocked_hours"].values()) + sum(result["starved_hours"].values()) > 0


def test_all_actions_schedule_equal_preventive_resource() -> None:
    tape = materialize_tape(1200003, weeks=3)
    rows = [run_policy(tape, periodic_policy((action,))) for action in ACTIONS]
    assert {row["scheduled_pm_hours"] for row in rows} == {72.0}


def test_exogenous_consumption_is_identical_between_actions() -> None:
    tape = materialize_tape(1200004, weeks=4)
    rows = [run_policy(tape, periodic_policy((action,))) for action in ACTIONS]
    assert len({row["base_exogenous_sha256"] for row in rows}) == 1
    assert len({row["consumed_wear_sha256"] for row in rows}) == 1
    # Realization may differ, but the candidate identities/draws must not.
    candidates = [
        [(r["event_id"], r["onset_hours"], r["target"], r["realization_u"], r["repair_u"])
         for r in tape["r11_candidates"]]
        for _ in rows
    ]
    assert candidates[0] == candidates[1] == candidates[2]


def test_actions_change_future_condition_from_same_tape() -> None:
    tape = materialize_tape(1200005, weeks=3)
    sim5, c5, start5 = make_sim(tape)
    sim7, c7, start7 = make_sim(tape)
    c5.request("PM5", 0)
    c7.request("PM7", 0)
    from supply_chain.program_f import advance_including
    advance_including(sim5, start5 + 168.0)
    advance_including(sim7, start7 + 168.0)
    assert c5.condition[0] < c7.condition[0]
    assert c7.condition[2] < c5.condition[2]


def test_worst_condition_policy_returns_valid_action() -> None:
    tape = materialize_tape(1200006, weeks=1)
    _, controller, _ = make_sim(tape)
    assert worst_condition_policy(controller.observation()) in ACTIONS


def test_periodic_frontier_through_period_six_is_complete() -> None:
    rows = periodic_sequences(6)
    assert len(rows) == 1041
    assert ("PM5",) in rows
    assert ("PM5", "PM6") in rows
    assert ("PM5", "PM5") not in rows
