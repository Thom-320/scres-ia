from __future__ import annotations

import json
from pathlib import Path

from scripts.diagnose_program_o_state_rich_dual_resource import connected_components
from supply_chain.program_o_state_rich import (
    StateRichConfiguration,
    state_rich_calendar,
)
from tests.test_program_o_state_rich import SCHEDULER, skeleton


ROOT = Path(__file__).resolve().parent.parent


def decisions(mode: str):
    return state_rich_calendar(
        skeleton=skeleton(),
        scheduler=SCHEDULER,
        config=StateRichConfiguration("belief_mpc", 3),
        regime_persistence=0.75,
        dominant_share=0.90,
        observation_mode=mode,
    )[1]


def operational_fields(observation):
    return (
        observation.on_hand,
        observation.locked_pipeline,
        observation.backlog_quantity,
        observation.backlog_orders,
        observation.max_backlog_age,
        observation.in_flight_quantity,
    )


def test_belief_only_masks_operational_state_but_preserves_current_belief():
    real = decisions("real")
    placebo = decisions("belief_only")
    for real_row, placebo_row in zip(real, placebo):
        assert placebo_row.observation.belief_c == real_row.observation.belief_c
        assert (
            placebo_row.observation.predicted_share_c
            == real_row.observation.predicted_share_c
        )
        assert placebo_row.observation.on_hand == (0.0, 0.0)
        assert placebo_row.observation.locked_pipeline == (0.0, 0.0)
        assert placebo_row.observation.backlog_quantity == (0.0, 0.0)
        assert placebo_row.observation.backlog_orders == (0, 0)
        assert placebo_row.observation.max_backlog_age == (0.0, 0.0)
        assert placebo_row.observation.in_flight_quantity == (0.0, 0.0)


def test_operational_only_preserves_state_but_neutralizes_belief():
    real = decisions("real")
    placebo = decisions("operational_only")
    # Before the first action both rollouts share the same physical history.
    assert operational_fields(placebo[0].observation) == operational_fields(
        real[0].observation
    )
    for placebo_row in placebo:
        assert placebo_row.observation.belief_c == 0.5
        assert placebo_row.observation.predicted_share_c == 0.5


def test_stale_operational_placebo_keeps_current_belief():
    real = decisions("real")
    placebo = decisions("stale_operational_current_belief")
    assert operational_fields(placebo[2].observation) == operational_fields(
        real[0].observation
    )
    assert placebo[2].observation.belief_c == real[2].observation.belief_c
    assert (
        placebo[2].observation.predicted_share_c
        == real[2].observation.predicted_share_c
    )


def test_stale_belief_placebo_keeps_current_operational_state():
    real = decisions("real")
    placebo = decisions("current_operational_stale_belief")
    # Belief is exogenous to policy actions. Operational state follows the
    # placebo's own causal action trajectory and is intentionally not copied
    # from the real rollout.
    assert placebo[2].observation.belief_c == real[0].observation.belief_c
    assert (
        placebo[2].observation.predicted_share_c
        == real[0].observation.predicted_share_c
    )


def test_swapped_operational_placebo_keeps_current_belief():
    real = decisions("real")
    placebo = decisions("swapped_operational_current_belief")
    # The first decision has a common pre-action physical history; subsequent
    # physical histories legitimately diverge after placebo actions differ.
    for real_row, placebo_row in [(real[0], placebo[0])]:
        assert placebo_row.observation.on_hand == tuple(reversed(real_row.observation.on_hand))
        assert placebo_row.observation.backlog_quantity == tuple(reversed(real_row.observation.backlog_quantity))
        assert placebo_row.observation.belief_c == real_row.observation.belief_c
        assert (
            placebo_row.observation.predicted_share_c
            == real_row.observation.predicted_share_c
        )


def test_connected_component_requires_three_cells_spanning_both_axes():
    passed = connected_components(
        ["rho75_share75", "rho75_share90", "rho90_share75"],
        required_size=3,
    )
    assert passed["passed"] is True
    failed = connected_components(
        ["rho75_share75", "rho75_share90"], required_size=3
    )
    assert failed["passed"] is False


def test_contract_preserves_sealed_validation_and_post_hoc_claim_boundary():
    contract = json.loads(
        (
            ROOT
            / "contracts/program_o_state_rich_dual_resource_diagnostic_v1.json"
        ).read_text()
    )
    assert contract["tape_governance"]["burned_fit_min"] == 7420001
    assert contract["tape_governance"]["burned_fit_max"] == 7420048
    assert contract["tape_governance"]["sealed_validation_min"] == 7420049
    assert contract["tape_governance"]["validation_access_forbidden"] is True
    assert contract["claim_boundary"]["h_obs_confirmed"] is False
    assert contract["claim_boundary"]["learner_authorized"] is False
