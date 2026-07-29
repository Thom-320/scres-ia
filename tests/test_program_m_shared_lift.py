from __future__ import annotations

import pytest

from supply_chain.program_m_shared_lift import (
    ACTIVATION_LEAD_HOURS,
    COMMITMENT_VEHICLE_HOURS,
    DECISION_WEEKS,
    PAYLOAD_CAPACITY_RATIONS,
    SLOT_WINDOW_HOURS,
    ProgramMSharedLiftSimulation,
    assert_program_m_observation_whitelist,
    materialize_warning_records,
)
from supply_chain.supply_chain import MFSCSimulation, OrderRecord


CALENDAR = ("RESERVE_A", "RESERVE_B") * 4


def _enabled(**overrides):
    kwargs = {
        "seed": 101,
        "horizon": 8 * 168,
        "risks_enabled": True,
        "risk_event_tape": [],
        "program_m_enabled": True,
        "reservation_calendar": CALENDAR,
        "decision_start_time": 0.0,
    }
    kwargs.update(overrides)
    return ProgramMSharedLiftSimulation(**kwargs)


def _order(destination: str, *, j: int = 1, qty: float = 2_500.0) -> OrderRecord:
    return OrderRecord(
        j=j,
        OPTj=0.0,
        quantity=qty,
        remaining_qty=0.0,
        in_flight_qty=qty,
        cssu_destination=destination,
    )


def test_flags_off_is_bit_identical_to_real_des():
    kwargs = {
        "seed": 123,
        "horizon": 2_000,
        "initial_buffers": {"op9_rations": 10_000},
        "order_fulfillment_mode": "op9_linked",
        "cssu_topology_mode": "split_v1",
        "cssu_service_rule": "FIFO_PARTIAL",
        "downstream_transport_capacity_mode": "parallel",
    }
    base = MFSCSimulation(**kwargs).run()
    extension_off = ProgramMSharedLiftSimulation(**kwargs).run()
    assert extension_off.flow_ledger() == base.flow_ledger()
    assert extension_off.cssu_delivery_events == base.cssu_delivery_events
    assert [order.CTj for order in extension_off.orders] == [order.CTj for order in base.orders]


def test_calendar_resources_are_equal_before_use_and_all_eight_actions_are_live():
    left = _enabled(reservation_calendar=("RESERVE_A",) * 8)
    right = _enabled(reservation_calendar=("RESERVE_B",) * 8)
    left.assert_complete_calendar()
    right.assert_complete_calendar()
    assert left.program_m_ledger()["resources"] == right.program_m_ledger()["resources"]
    resources = left.program_m_ledger()["resources"]
    assert resources["reserved_slots"] == DECISION_WEEKS
    assert resources["total_committed_departures"] == 8
    assert resources["empty_departures"] == 8
    assert resources["reserved_payload_capacity_rations"] == 8 * PAYLOAD_CAPACITY_RATIONS
    assert resources["reserved_vehicle_hours"] == 8 * COMMITMENT_VEHICLE_HOURS
    assert resources["total_committed_vehicle_hours"] == 384
    assert left.program_m_reservations[-1].activation_time == 7 * 168 + 24
    assert left.horizon >= left.program_m_reservations[-1].expiry_time


def test_all_eight_activation_windows_fit_before_eight_week_end_and_resources_are_invariant():
    a = _enabled(reservation_calendar=("RESERVE_A",) * 8)
    b = _enabled(reservation_calendar=("RESERVE_B",) * 8)
    episode_end = 8 * 168
    for record in a.program_m_reservations:
        assert record.activation_time == record.decision_time + ACTIVATION_LEAD_HOURS
        assert record.expiry_time == record.activation_time + SLOT_WINDOW_HOURS
        assert record.activation_time < episode_end
        assert record.expiry_time <= episode_end
    a.env.run(until=episode_end)
    b.env.run(until=episode_end)
    resources_a = a.program_m_ledger()["resources"]
    resources_b = b.program_m_ledger()["resources"]
    assert resources_a == resources_b
    assert resources_a["total_committed_departures"] == 8
    assert resources_a["empty_departures"] == 8
    assert resources_a["total_committed_vehicle_hours"] == 384

    # A loaded outcome changes utilization, never the committed envelope.
    loaded = _enabled(reservation_calendar=("RESERVE_A",) * 8)
    loaded.env.run(until=24.0)
    loaded._take_down_cssu(10, "A")
    order = _order("A")
    loaded.env.process(loaded._deliver_order_from_op9(order, order.quantity))
    loaded.env.run(until=episode_end)
    resources_loaded = loaded.program_m_ledger()["resources"]
    invariant_fields = (
        "total_committed_departures",
        "reserved_payload_capacity_rations",
        "total_committed_vehicle_hours",
    )
    assert {key: resources_loaded[key] for key in invariant_fields} == {
        key: resources_a[key] for key in invariant_fields
    }
    assert resources_loaded["loaded_departures"] == 1
    assert resources_loaded["empty_departures"] == 7


def test_protected_route_bypasses_only_reserved_destination_local_op10_and_op12():
    sim = _enabled(reservation_calendar=("RESERVE_A",) * 8)
    sim.env.run(until=24.0)
    sim._take_down_cssu(10, "A")
    sim._take_down_cssu(12, "A")
    sim._take_down_cssu(10, "B")
    a = _order("A")
    b = _order("B", j=2)

    sim.env.process(sim._deliver_order_from_op9(a, a.quantity))
    sim.env.process(sim._deliver_order_from_op9(b, b.quantity))
    sim.env.run(until=73.0)

    assert a.OATj == pytest.approx(72.0)
    assert b.OATj is None
    assert sim._is_cssu_path_down(10, "A")
    assert sim._is_cssu_path_down(12, "A")
    assert sim._is_cssu_path_down(10, "B")
    movement = sim.program_m_ledger()["movements"][0]
    assert movement["destination"] == "A"
    assert movement["payload_rations"] == pytest.approx(2_500.0)


def test_protected_route_does_not_bypass_cssu_node_or_global_route_outage():
    sim = _enabled(reservation_calendar=("RESERVE_A",) * 8)
    sim.env.run(until=24.0)
    sim._take_down_cssu(10, "A")
    sim._take_down_cssu(11, "A")
    order = _order("A")
    sim.env.process(sim._deliver_order_from_op9(order, order.quantity))
    sim.env.run(until=96.0)
    assert order.OATj is None
    sim._bring_up_cssu(11, "A")
    sim.env.run(until=121.0)
    assert order.OATj == pytest.approx(120.0)


def test_slot_expires_and_records_full_empty_commitment():
    sim = _enabled(reservation_calendar=("RESERVE_A",) * 8)
    sim.env.run(until=169.0)
    ledger = sim.program_m_ledger()
    first = sim.program_m_reservations[0]
    assert first.status == "expired_empty"
    empty = next(row for row in ledger["movements"] if row["week"] == 0)
    assert empty["loaded"] is False
    assert empty["vehicle_hours"] == COMMITMENT_VEHICLE_HOURS


def test_null_physics_is_transition_identical_across_actions():
    tape = [
        {
            "risk_id": "researcher_introduced_localized_access_disruption",
            "start_time": 180.0,
            "end_time": 252.0,
            "duration": 72.0,
            "affected_ops": [10, 12],
            "affected_cssu": "A",
        }
    ]
    common = {
        "seed": 555,
        "horizon": 8 * 168,
        "risks_enabled": True,
        "risk_event_tape": tape,
        "program_m_enabled": True,
        "decision_start_time": 0.0,
        "bypass_local_route": False,
        "initial_buffers": {"op9_rations": 20_000},
    }
    a = ProgramMSharedLiftSimulation(**common, reservation_calendar=("RESERVE_A",) * 8).run()
    b = ProgramMSharedLiftSimulation(**common, reservation_calendar=("RESERVE_B",) * 8).run()
    assert a.flow_ledger() == b.flow_ledger()
    assert a.cssu_delivery_events == b.cssu_delivery_events
    assert [order.CTj for order in a.orders] == [order.CTj for order in b.orders]


def test_conservation_after_loaded_protected_movement():
    sim = _enabled(reservation_calendar=("RESERVE_A",) * 8)
    sim.env.run(until=24.0)
    sim._take_down_cssu(10, "A")
    order = _order("A")
    sim.total_strategic_rations_injected = order.quantity
    sim.env.process(sim._deliver_order_from_op9(order, order.quantity))
    sim.env.run(until=73.0)
    ledger = sim.program_m_ledger()
    assert ledger["flow_ledger"]["ration_residual"] == pytest.approx(0.0)
    assert sum(sim.cssu_delivered.values()) == pytest.approx(order.quantity)
    assert ledger["resources"]["actual_payload_rations"] == pytest.approx(order.quantity)


def test_warning_generator_is_deterministic_joint_and_event_timing_is_unchanged():
    events = [
        {
            "risk_id": "local",
            "start_time": 200.0,
            "end_time": 272.0,
            "duration": 72.0,
            "affected_ops": [10],
            "affected_cssu": "A",
        }
    ]
    first = materialize_warning_records(
        seed=77,
        decision_start_time=0.0,
        risk_events=events,
        sensitivity=1.0,
        specificity=1.0,
    )
    second = materialize_warning_records(
        seed=77,
        decision_start_time=0.0,
        risk_events=events,
        sensitivity=1.0,
        specificity=1.0,
    )
    assert first == second
    assert (first[1].warning_A, first[1].warning_B) == (1, 0)
    assert events[0]["start_time"] == 200.0

    # Core-normalized events can carry end_time=start while duration remains
    # positive; signal truth must still use the physical duration.
    no_explicit_end = [{key: value for key, value in events[0].items() if key != "end_time"}]
    assert materialize_warning_records(
        seed=77,
        decision_start_time=0.0,
        risk_events=no_explicit_end,
        sensitivity=1.0,
        specificity=1.0,
    )[1].warning_A == 1


def test_full_des_replay_is_deterministic_under_common_random_numbers():
    tape = [
        {
            "risk_id": "researcher_introduced_localized_access_disruption",
            "start_time": 180.0,
            "end_time": 252.0,
            "duration": 72.0,
            "affected_ops": [10, 12],
            "affected_cssu": "A",
        }
    ]
    kwargs = {
        "seed": 991,
        "horizon": 8 * 168,
        "risks_enabled": True,
        "risk_event_tape": tape,
        "program_m_enabled": True,
        "reservation_calendar": ("RESERVE_A",) * 8,
        "initial_buffers": {"op9_rations": 20_000},
    }
    first = ProgramMSharedLiftSimulation(**kwargs).run()
    second = ProgramMSharedLiftSimulation(**kwargs).run()
    assert first.cssu_delivery_events == second.cssu_delivery_events
    assert first.program_m_ledger() == second.program_m_ledger()


def test_extension_event_validation_is_fail_closed_without_balancing_locations():
    valid_unbalanced = [
        {
            "risk_id": "researcher_introduced_localized_access_disruption",
            "start_time": 190.0 + week * 168,
            "duration": 24.0,
            "affected_ops": [10],
            "affected_cssu": "A",
        }
        for week in range(2)
    ]
    sim = _enabled(risk_event_tape=valid_unbalanced)
    assert [event.affected_cssu for event in sim.risk_event_tape] == ["A", "A"]

    invalid = [{**valid_unbalanced[0], "affected_ops": [9, 10]}]
    with pytest.raises(ValueError, match="only Op10/Op12"):
        _enabled(risk_event_tape=invalid)


def test_observation_is_fail_closed_and_has_no_privileged_tape_fields():
    sim = _enabled()
    obs = sim.get_program_m_observation()
    assert_program_m_observation_whitelist(obs)
    forbidden = ("seed", "tape", "future", "duration", "oracle", "latent")
    assert not any(token in key.lower() for key in obs for token in forbidden)
    with pytest.raises(AssertionError, match="non-whitelisted"):
        assert_program_m_observation_whitelist({**obs, "future_event_location": 1.0})
