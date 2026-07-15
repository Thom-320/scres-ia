from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest
import numpy as np

from scripts.screen_program_o_state_rich_fit import (
    action_audit,
    connected_component_pass,
    matched_resource_frontier,
    state_counterfactual_audit,
)
from supply_chain.program_o_state_rich import (
    StateRichConfiguration,
    StateRichDecision,
    StateRichObservation,
    choose_state_rich_action,
    finite_state_rich_configurations,
    add_minority_backlog,
    state_rich_calendar,
    swap_product_channels,
)
from supply_chain.program_o_full_des import ProgramOFullDESSimulation
from supply_chain.program_o_full_des_transducer import extract_full_des_skeleton


SCHEDULER = {
    "0": ["P_H", "P_H", "P_H"],
    "1": ["P_H", "P_C", "P_H"],
    "2": ["P_C", "P_H", "P_C"],
    "3": ["P_C", "P_C", "P_C"],
}

ROOT = Path(__file__).resolve().parent.parent


def skeleton(*, same_time_batch: bool = False):
    start = 1000.0
    weeks = 3
    arrivals = []
    for week in range(weeks):
        for position, offset in enumerate((48.0, 96.0, 144.0)):
            time = start + 168.0 * week + offset
            if same_time_batch and week == 0 and position == 2:
                time = start + 168.0
            arrivals.append([time, week, position])
    order_times = []
    order_products = []
    for week in range(weeks):
        for day, offset in enumerate((30, 54, 78, 102, 126, 150)):
            order_times.append(start + 168.0 * week + offset)
            order_products.append("P_C" if (week + day) % 3 else "P_H")
    release_slots = list(range(int(start + 24), int(start + 168 * weeks + 337), 24))
    return {
        "seed": 999,
        "decision_weeks": weeks,
        "decision_start": start,
        "score_time": start + 168.0 * weeks + 336.0,
        "batch_arrivals": arrivals,
        "order_times": order_times,
        "order_quantities": [2500.0] * len(order_times),
        "order_products": order_products,
        "release_slots": release_slots,
        "opening_inventory": [5000.0, 5000.0],
        "tape_sha256": "not-consumed",
        "prefix_state_hash": "prefix",
        "skeleton_sha256": "skeleton",
    }


def observation(**changes):
    value = StateRichObservation(
        week=1,
        decision_time=1168.0,
        on_hand=(5000.0, 5000.0),
        locked_pipeline=(0.0, 0.0),
        backlog_quantity=(0.0, 0.0),
        backlog_orders=(0, 0),
        max_backlog_age=(0.0, 0.0),
        in_flight_quantity=(0.0, 0.0),
        belief_c=0.5,
        predicted_share_c=0.5,
        previous_action=1,
        remaining_decisions=3,
        observation_sha256="fixture",
    )
    return replace(value, **changes)


def test_finite_family_is_exactly_the_ten_frozen_configurations():
    configs = finite_state_rich_configurations()
    assert len(configs) == 10
    assert len({config.config_id for config in configs}) == 10
    assert [config.config_id for config in configs] == [
        "base_stock__1",
        "base_stock__2",
        "max_pressure__0",
        "max_pressure__5000",
        "min_cost_flow__1",
        "min_cost_flow__2",
        "belief_mpc__3",
        "belief_mpc__4",
        "belief_dp__3",
        "belief_dp__4",
    ]


def test_every_controller_emits_a_complete_deterministic_calendar():
    tape = skeleton()
    for config in finite_state_rich_configurations():
        first = state_rich_calendar(
            skeleton=tape,
            scheduler=SCHEDULER,
            config=config,
            regime_persistence=0.75,
            dominant_share=0.90,
        )
        second = state_rich_calendar(
            skeleton=tape,
            scheduler=SCHEDULER,
            config=config,
            regime_persistence=0.75,
            dominant_share=0.90,
        )
        assert first == second
        assert len(first[0]) == 3
        assert set(first[0]).issubset({0, 1, 2, 3})
        assert len(first[1]) == 3


def test_future_labels_quantities_and_seed_do_not_change_first_action():
    original = skeleton()
    counterfactual = {**original}
    counterfactual["seed"] = 123456789
    counterfactual["tape_sha256"] = "different"
    counterfactual["order_products"] = [
        "P_H" if value == "P_C" else "P_C"
        for value in original["order_products"]
    ]
    counterfactual["order_quantities"] = [9999.0] * len(original["order_quantities"])
    config = StateRichConfiguration("belief_mpc", 4)
    first = state_rich_calendar(
        skeleton=original,
        scheduler=SCHEDULER,
        config=config,
        regime_persistence=0.75,
        dominant_share=0.90,
    )[0]
    second = state_rich_calendar(
        skeleton=counterfactual,
        scheduler=SCHEDULER,
        config=config,
        regime_persistence=0.75,
        dominant_share=0.90,
    )[0]
    assert first[0] == second[0]


def test_same_time_prior_batch_is_locked_not_on_hand_at_decision():
    _calendar, decisions = state_rich_calendar(
        skeleton=skeleton(same_time_batch=True),
        scheduler=SCHEDULER,
        config=StateRichConfiguration("base_stock", 1),
        regime_persistence=0.75,
        dominant_share=0.90,
    )
    week_one = decisions[1].observation
    # The exact-time third slot from week zero is known committed pipeline but
    # not yet an on-hand arrival under the strict half-open observation rule.
    assert sum(week_one.locked_pipeline) == 5000.0


@pytest.mark.parametrize(
    "config",
    (
        StateRichConfiguration("base_stock", 1),
        StateRichConfiguration("min_cost_flow", 1),
        StateRichConfiguration("belief_dp", 3),
    ),
)
def test_operational_replay_state_matches_direct_full_des_events(config):
    """The compact policy state must equal the direct SimPy state pre-decision."""
    contract = json.loads(
        (ROOT / "contracts/program_o_full_des_hpi_translation_v1.json").read_text()
    )
    scheduler = contract["action"]["within_week_schedulers"][
        contract["action"]["primary_scheduler"]
    ]
    seed = 7400048  # Burned preseed parity fixture; never a fit/validation tape.
    full_skeleton, _ = extract_full_des_skeleton(
        seed=seed,
        scheduler=scheduler,
        regime_persistence=0.75,
        dominant_share=0.90,
    )
    calendar, decisions = state_rich_calendar(
        skeleton=full_skeleton.as_dict(),
        scheduler=scheduler,
        config=config,
        regime_persistence=0.75,
        dominant_share=0.90,
    )
    sim = ProgramOFullDESSimulation(
        seed=seed,
        calendar=calendar,
        scheduler=scheduler,
        regime_persistence=0.75,
        dominant_share=0.90,
    ).run_contract()
    product_index = {"P_C": 0, "P_H": 1}
    policy_arrivals = {}
    for event in sim.program_o_product_events:
        if event["event"] != "op8_arrived_sb":
            continue
        lot_id = str(event["tokens"][0]["lot_id"])
        if not lot_id.startswith("policy:w"):
            continue
        week_text, slot_text = lot_id.removeprefix("policy:w").split(":s")
        policy_arrivals[(int(week_text), int(slot_text))] = float(event["time"])

    for decision in decisions:
        now = float(decision.observation.decision_time)
        week = int(decision.observation.week)
        on_hand = [5000.0, 5000.0]
        locked = [0.0, 0.0]
        for target_week in range(week):
            labels = tuple(scheduler[str(calendar[target_week])])
            for position, product_id in enumerate(labels):
                arrival = policy_arrivals[(target_week, position)]
                target = on_hand if arrival < now - 1e-12 else locked
                target[product_index[product_id]] += 5000.0
        backlog = [0.0, 0.0]
        backlog_orders = [0, 0]
        max_age = [0.0, 0.0]
        in_flight = [0.0, 0.0]
        for order in sim.orders:
            product = product_index[str(order.requested_product_id)]
            release = order.op9_release_time
            if release is not None and float(release) < now - 1e-12:
                on_hand[product] -= float(order.quantity)
            if float(order.OPTj) >= now - 1e-12:
                continue
            if release is None or float(release) >= now - 1e-12:
                backlog[product] += float(order.quantity)
                backlog_orders[product] += 1
                max_age[product] = max(max_age[product], now - float(order.OPTj))
            elif order.OATj is None or float(order.OATj) >= now - 1e-12:
                in_flight[product] += float(order.quantity)

        observed = decision.observation
        assert observed.on_hand == pytest.approx(on_hand, abs=1e-9)
        assert observed.locked_pipeline == pytest.approx(locked, abs=1e-9)
        assert observed.backlog_quantity == pytest.approx(backlog, abs=1e-9)
        assert observed.backlog_orders == tuple(backlog_orders)
        assert observed.max_backlog_age == pytest.approx(max_age, abs=1e-9)
        assert observed.in_flight_quantity == pytest.approx(in_flight, abs=1e-9)


def test_operational_backlog_changes_state_rich_action():
    config = StateRichConfiguration("min_cost_flow", 1)
    c_pressure = observation(
        backlog_quantity=(20000.0, 0.0),
        backlog_orders=(8, 0),
        predicted_share_c=0.75,
        belief_c=0.75,
    )
    h_pressure = replace(
        c_pressure,
        backlog_quantity=(0.0, 20000.0),
        backlog_orders=(0, 8),
        predicted_share_c=0.25,
        belief_c=0.25,
    )
    action_c = choose_state_rich_action(
        c_pressure,
        config,
        scheduler=SCHEDULER,
        regime_persistence=0.75,
        dominant_share=0.90,
    )[0]
    action_h = choose_state_rich_action(
        h_pressure,
        config,
        scheduler=SCHEDULER,
        regime_persistence=0.75,
        dominant_share=0.90,
    )[0]
    assert action_c > action_h


def test_frozen_state_counterfactual_transformations_are_symmetric_and_monotone():
    original = observation(
        on_hand=(1000.0, 9000.0),
        locked_pipeline=(2000.0, 3000.0),
        backlog_quantity=(4000.0, 5000.0),
        backlog_orders=(2, 3),
        max_backlog_age=(24.0, 48.0),
        in_flight_quantity=(6000.0, 7000.0),
        belief_c=0.7,
        predicted_share_c=0.66,
        previous_action=2,
    )
    swapped = swap_product_channels(original)
    assert swapped.on_hand == (9000.0, 1000.0)
    assert swapped.backlog_orders == (3, 2)
    assert swapped.belief_c == pytest.approx(0.3)
    assert swapped.predicted_share_c == pytest.approx(0.34)
    assert swapped.previous_action == 1
    pressured = add_minority_backlog(original, action=3, scheduler=SCHEDULER)
    assert pressured.backlog_quantity == (4000.0, 10000.0)
    assert pressured.backlog_orders == (2, 4)


def test_hysteresis_preserves_previous_action_inside_frozen_band():
    state = observation(previous_action=3, predicted_share_c=0.50, belief_c=0.50)
    action = choose_state_rich_action(
        state,
        StateRichConfiguration("max_pressure", 5000),
        scheduler=SCHEDULER,
        regime_persistence=0.75,
        dominant_share=0.90,
    )[0]
    assert action == 3


def test_action_audit_rejects_a_disguised_fixed_calendar():
    assert action_audit([[2] * 8 for _ in range(48)])["passed"] is False
    diverse = [
        [((tape * 257) // (4 ** (7 - week))) % 4 for week in range(8)]
        for tape in range(48)
    ]
    assert action_audit(diverse)["passed"] is True


def test_resource_frontier_requires_one_calendar_to_dominate_every_tape():
    policy = {
        "ret_visible": np.asarray([0.8, 0.8]),
        "actual_loaded_departures": np.asarray([2.0, 2.0]),
        "actual_payload": np.asarray([20.0, 20.0]),
        "actual_downstream_vehicle_hours": np.asarray([96.0, 96.0]),
    }
    panel = {
        # Calendar 1 has adequate means but is below policy use on tape 0.
        "ret_visible": np.asarray([[0.70, 0.99, 0.60], [0.70, 0.99, 0.60]]),
        "actual_loaded_departures": np.asarray([[2.0, 1.0, 3.0], [2.0, 3.0, 3.0]]),
        "actual_payload": np.asarray([[20.0, 10.0, 30.0], [20.0, 30.0, 30.0]]),
        "actual_downstream_vehicle_hours": np.asarray(
            [[96.0, 48.0, 144.0], [96.0, 144.0, 144.0]]
        ),
    }
    frontier = matched_resource_frontier(panel=panel, policy_metrics=policy)
    assert frontier["eligible_calendar_count"] == 2
    assert frontier["calendar_index"] == 0
    assert frontier["minimum_per_tape_resource_slack"] == {
        "actual_loaded_departures": 0.0,
        "actual_payload": 0.0,
        "actual_downstream_vehicle_hours": 0.0,
    }
    assert frontier["policy_delta"] == pytest.approx(0.1)
    assert frontier["passed"] is True


def test_connected_component_requires_three_cells_spanning_both_axes():
    def row(passed):
        return {
            "mean_delta_vs_full_frontier": 0.02 if passed else 0.0,
            "favorable_tapes": 40 if passed else 0,
            "metric_guardrails_pass": passed,
            "reserved_capacity_equal": passed,
            "strict_actual_use_pass": passed,
            "resource_frontier": {"passed": passed},
            "action_trajectory": {"passed": passed},
            "state_counterfactuals": {"passed": passed},
            "information_placebos_pass": passed,
        }

    three = {
        "rho75_share75": row(True),
        "rho75_share90": row(True),
        "rho90_share75": row(True),
        "rho90_share90": row(False),
    }
    assert connected_component_pass(three)["passed"] is True
    only_one_axis = {
        "rho75_share75": row(True),
        "rho75_share90": row(True),
        "rho90_share75": row(False),
        "rho90_share90": row(False),
    }
    assert connected_component_pass(only_one_axis)["passed"] is False


def test_state_counterfactual_audit_accepts_a_symmetric_state_rule():
    config = StateRichConfiguration("min_cost_flow", 1)
    states = []
    for state in (
        observation(
            backlog_quantity=(20000.0, 0.0),
            backlog_orders=(8, 0),
            predicted_share_c=0.75,
            belief_c=0.75,
            previous_action=2,
        ),
        observation(
            backlog_quantity=(0.0, 20000.0),
            backlog_orders=(0, 8),
            predicted_share_c=0.25,
            belief_c=0.25,
            previous_action=1,
        ),
    ):
        action, objective, tied = choose_state_rich_action(
            state,
            config,
            scheduler=SCHEDULER,
            regime_persistence=0.75,
            dominant_share=0.90,
        )
        states.append(
            StateRichDecision(
                observation=state,
                action=action,
                objective=objective,
                tied_actions=tied,
            )
        )
    audit = state_counterfactual_audit(
        decisions_by_tape=[states],
        config=config,
        scheduler=SCHEDULER,
        model={"regime_persistence": 0.75, "dominant_product_share": 0.90},
    )
    assert audit["passed"] is True
