import json
from pathlib import Path

import pytest

from supply_chain.dra2_experiment import materialize_tape
from supply_chain.dra2_policy_env import (
    ProgramEConvoyEnv,
    make_identity_normalizers as make_program_e_normalizers,
)
from supply_chain.program_l_route_recourse_env import (
    ProgramLRouteRecourseEnv,
    make_identity_normalizers as make_program_l_normalizers,
)


ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("seed,family", [(899011, "routine"), (899012, "op8_interruption")])
def test_route_extension_flags_off_is_bitwise_identical(seed, family):
    tape = materialize_tape(
        seed,
        family,
        4,
        "program_l_identity_burned",
        contract_id="program_e_policy_realizability_v1",
        tape_prefix="program-l-identity",
    )
    baseline = ProgramEConvoyEnv(
        [tape],
        make_program_e_normalizers(),
        episode_days=14,
        random_tapes=False,
    )
    extension = ProgramLRouteRecourseEnv(
        [tape],
        make_program_l_normalizers(),
        episode_days=14,
        random_tapes=False,
        route_recourse_enabled=False,
    )
    baseline.reset(options={"tape": tape})
    extension.reset(options={"tape": tape})

    while True:
        assert baseline.action_masks().tolist() == extension.action_masks().tolist()
        action = int(baseline.action_masks()[1])
        base_obs, base_reward, base_done, _, base_info = baseline.step(action)
        route_obs, route_reward, route_done, _, route_info = extension.step(action)
        assert base_obs.tolist() == route_obs[: len(base_obs)].tolist()
        assert base_reward == route_reward
        assert base_done == route_done
        assert base_info["effective_action"] == route_info["effective_action"]
        if base_done:
            break

    shared_terminal = set(base_info) & set(route_info)
    for key in shared_terminal:
        assert route_info[key] == base_info[key], key


def test_alternate_route_bypasses_op8_without_creating_mass():
    tape = materialize_tape(
        899013,
        "routine",
        4,
        "program_l_route2_burned",
        contract_id="program_e_policy_realizability_v1",
        tape_prefix="program-l-route2",
    )
    env = ProgramLRouteRecourseEnv(
        [tape],
        make_program_l_normalizers(),
        episode_days=14,
        random_tapes=False,
    )
    env.reset(options={"tape": tape})
    while not (
        env.sim.op8_convoy_available and float(env.sim.rations_al.level) > 0.0
    ):
        env.step(0)
    env.sim.op_down_count[8] = 1
    assert env.action_masks().tolist() == [True, False, True]

    before_wait = env.sim.op8_convoy_route_wait_hours
    _, _, _, _, info = env.step(2)
    assert info["departed"] is True
    assert info["route_id"] == "R2"
    assert env.sim.op8_convoy_route_wait_hours == before_wait
    assert env.sim.op8_convoy_metrics()["op8_convoy_resource_residual"] == 0.0
    ledger = env.sim.flow_ledger()
    assert ledger["raw_residual"] == 0.0
    assert ledger["ration_residual"] == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"route_outbound_hours": -1.0},
        {"route_return_hours": float("inf")},
        {"route_exposed_op": 0},
        {"route_exposed_op": 14},
        {"route_exposed_op": True},
    ],
)
def test_route_override_rejects_invalid_physics(kwargs):
    tape = materialize_tape(
        899014,
        "routine",
        4,
        "program_l_invalid_route_burned",
        contract_id="program_e_policy_realizability_v1",
        tape_prefix="program-l-invalid-route",
    )
    env = ProgramLRouteRecourseEnv(
        [tape],
        make_program_l_normalizers(),
        episode_days=14,
        random_tapes=False,
    )
    env.reset(options={"tape": tape})
    with pytest.raises(ValueError):
        env.sim.apply_op8_convoy_action("HOLD", **kwargs)


def test_corrective_audit_rejects_program_l_terminal_reaffirmation():
    audit = json.loads(
        (ROOT / "results/paper2_search/program_l_corrective_audit.json").read_text()
    )
    assert all(audit["machine_checks"].values())
    corrected = audit["corrected_interpretation"]
    assert corrected["terminal_boundary_reaffirmed_by_program_l"] is False
    assert corrected["route_recourse_family"] == "blocked_domain_fact_and_open_if_validated"
    assert corrected["thesis_native_route_choice"]["h_pi"] == 0.0
    assert corrected["researcher_extension_quantitative_ceiling"] is None
    assert corrected["learner_authorized"] is False
    assert corrected["paper3_authorized"] is False
