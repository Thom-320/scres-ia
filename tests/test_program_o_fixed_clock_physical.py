import json
from pathlib import Path

from supply_chain.program_o_full_des import run_program_o_full_des_episode


ROOT = Path(__file__).resolve().parent.parent
PARENT_CONTRACT = json.loads(
    (ROOT / "contracts/program_o_full_des_hpi_translation_v1.json").read_text()
)
FIXED_CLOCK_CONTRACT = json.loads(
    (
        ROOT
        / "contracts/program_o_fixed_clock_physical_hobs_validation_v1.json"
    ).read_text()
)
SCHEDULER = PARENT_CONTRACT["action"]["within_week_schedulers"][
    PARENT_CONTRACT["action"]["primary_scheduler"]
]
BURNED_FIXTURE_SEED = 7400048


def test_fixed_clock_contract_is_fail_closed_before_sealed_validation():
    assert FIXED_CLOCK_CONTRACT["status"].endswith(
        "EXECUTION_FORBIDDEN_PENDING_PREFLIGHT_VERDICT"
    )
    assert FIXED_CLOCK_CONTRACT["physics"]["downstream_freight_physics_mode"] == (
        "fixed_clock_physical_v1"
    )
    assert FIXED_CLOCK_CONTRACT["primary_policy"]["config_id"] == "belief_mpc__3"
    assert FIXED_CLOCK_CONTRACT["validation_tapes"] == {
        "range": [7420049, 7420096],
        "status": "SEALED_NOT_ACCESSED",
        "open_once": True,
        "opening_forbidden_until": [
            "immutable implementation commit exists",
            "contract and source hashes are recorded",
            "preflight verdict passes",
            "watcher is ready before producer",
            "exclusive seed claim succeeds",
        ],
    }
    assert FIXED_CLOCK_CONTRACT["claim_boundary"] == {
        "h_obs_confirmed": False,
        "learned_advantage_confirmed": False,
        "learner_authorized": False,
        "paper2_confirmed": False,
        "paper3_authorized": False,
        "cobb_douglas": "outside contract",
    }


def run(calendar, mode):
    return run_program_o_full_des_episode(
        seed=BURNED_FIXTURE_SEED,
        calendar=calendar,
        scheduler=SCHEDULER,
        regime_persistence=0.75,
        dominant_share=0.90,
        downstream_freight_physics_mode=mode,
    )


def test_loaded_only_mode_remains_the_default_identity():
    _, implicit = run_program_o_full_des_episode(
        seed=BURNED_FIXTURE_SEED,
        calendar=[2] * 8,
        scheduler=SCHEDULER,
        regime_persistence=0.75,
        dominant_share=0.90,
    )
    _, explicit = run([2] * 8, "loaded_only")

    assert implicit == explicit
    assert implicit["resources"]["scheduled_downstream_missions"] == 0
    assert implicit["resources"]["scheduled_downstream_vehicle_hours"] == 0


def test_fixed_clock_executes_every_reserved_mission_loaded_or_empty():
    sim, panel = run([2] * 8, "fixed_clock_physical_v1")
    resources = panel["resources"]

    assert resources["scheduled_downstream_missions"] == 112
    assert resources["scheduled_downstream_vehicle_hours"] == 5376
    assert resources["scheduled_downstream_crew_hours"] == 5376
    assert resources["scheduled_payload_capacity"] == 112 * 2600
    assert (
        resources["actual_loaded_departures"]
        + resources["empty_downstream_missions"]
        == resources["scheduled_downstream_missions"]
    )
    assert len(sim.program_o_downstream_mission_events) == 112
    assert {
        event["kind"] for event in sim.program_o_downstream_mission_events
    } == {"loaded", "empty"}
    completed_empty = [
        event
        for event in sim.program_o_downstream_mission_events
        if event["kind"] == "empty" and event["op12_completed_at"] is not None
    ]
    assert completed_empty
    assert all(
        event["op10_completed_at"] - event["op10_started_at"] == 24
        and event["op12_completed_at"] - event["op12_started_at"] == 24
        for event in completed_empty
    )


def test_empty_missions_do_not_change_physical_or_metric_outcomes():
    for calendar in ([0] * 8, [2] * 8, [3, 0, 2, 1, 3, 2, 0, 1]):
        base_sim, base = run(calendar, "loaded_only")
        fixed_sim, fixed = run(calendar, "fixed_clock_physical_v1")

        assert fixed["metrics"] == base["metrics"]
        assert fixed["products"] == base["products"]
        assert fixed["worst_product_fill"] == base["worst_product_fill"]
        assert fixed["aggregate_state_hash"] == base["aggregate_state_hash"]
        assert fixed["conservation"] == base["conservation"]
        assert fixed_sim.program_o_actual_payload == base_sim.program_o_actual_payload


def test_fixed_clock_resources_are_calendar_independent():
    _, all_h = run([0] * 8, "fixed_clock_physical_v1")
    _, mixed = run([3, 0, 2, 1, 3, 2, 0, 1], "fixed_clock_physical_v1")

    for key in (
        "committed_action_batch_slots",
        "gross_action_production_quantity",
        "scheduled_downstream_missions",
        "scheduled_downstream_vehicle_hours",
        "scheduled_downstream_crew_hours",
        "scheduled_payload_capacity",
        "setup_hours",
    ):
        assert all_h["resources"][key] == mixed["resources"][key]


def test_fixed_clock_fungible_null_remains_exact():
    kwargs = {
        "seed": BURNED_FIXTURE_SEED,
        "scheduler": SCHEDULER,
        "regime_persistence": 0.75,
        "dominant_share": 0.90,
        "complete_substitution": True,
        "downstream_freight_physics_mode": "fixed_clock_physical_v1",
    }
    _, all_h = run_program_o_full_des_episode(calendar=[0] * 8, **kwargs)
    _, all_c = run_program_o_full_des_episode(calendar=[3] * 8, **kwargs)

    assert all_h["metrics"] == all_c["metrics"]
    assert all_h["aggregate_state_hash"] == all_c["aggregate_state_hash"]
    assert all_h["resources"] == all_c["resources"]
