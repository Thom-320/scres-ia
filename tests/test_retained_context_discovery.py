import json
from pathlib import Path

import pytest

from scripts.evaluate_program_q_replication import scheduler
from supply_chain.program_o_full_des import product_demand_tape
from supply_chain.program_o_hobs import observable_calendar
from supply_chain.retained_context_discovery import (
    arm_priors,
    build_campaign_history,
    campaign_final_posterior,
    cross_campaign_transition,
)


ROOT = Path(__file__).resolve().parent.parent


def test_north_star_preserves_program_q_boundary() -> None:
    payload = json.loads(
        (ROOT / "research/paper2_exhaustive_search/retained_learning_north_star_v1.json").read_text()
    )
    assert payload["status"] == "RESEARCH_DIRECTION_NOT_A_CLAIM"
    assert payload["evidence_ladder"][1] == "retained_knowledge_effect"
    assert "65536" in payload["historical_boundary"]
    collision = json.loads(
        (ROOT / "research/paper2_exhaustive_search/retained_learning_seed_collision_audit_v1.json").read_text()
    )
    assert collision["passed"] is True
    assert collision["collisions"] == []
    assert collision["sealed_753_namespace_untouched"] is True


def test_terminal_summary_matches_executed_verdicts() -> None:
    terminal = json.loads(
        (ROOT / "results/retained_learning_discovery_terminal_summary_v1.json").read_text()
    )
    r0 = json.loads(
        (ROOT / "results/retained_context/r0_primary_v1/result.json").read_text()
    )
    action = json.loads(
        (ROOT / "results/program_q2/action_discovery_v1/result.json").read_text()
    )
    observation = json.loads(
        (ROOT / "results/program_q2/observation_discovery_v1/result.json").read_text()
    )
    assert terminal["R0"]["verdict"] == r0["verdict"]
    assert terminal["Q2_action"]["verdict"] == action["C_verdict"]
    assert terminal["Q2_observation"]["verdict"] == observation["verdict"]
    assert terminal["Q2_observation"]["full_state_probe_authorized"] is False
    assert terminal["terminal_route"].startswith("PROGRAM_Q_FALLBACK")


def test_direct_replay_audit_passes_when_present() -> None:
    path = ROOT / "results/retained_context/r0_primary_v1/direct_replay_audit.json"
    if path.exists():
        payload = json.loads(path.read_text())
        assert payload["passed"] is True
        assert payload["episodes"] >= 1
        assert payload["max_abs_error"] <= payload["tolerance"]


def test_default_product_tape_remains_unchanged_and_override_is_explicit() -> None:
    baseline = product_demand_tape(
        7_400_048, regime_persistence=0.75, dominant_share=0.90
    )
    repeated = product_demand_tape(
        7_400_048, regime_persistence=0.75, dominant_share=0.90
    )
    assert baseline == repeated
    assert "initial_regime" not in baseline
    forced = product_demand_tape(
        7_400_048,
        regime_persistence=0.75,
        dominant_share=0.90,
        initial_regime="P_H",
    )
    assert forced["initial_regime"] == "P_H"
    assert forced["regimes"][0] == "P_H"
    with pytest.raises(ValueError):
        product_demand_tape(
            1, regime_persistence=0.75, dominant_share=0.90, initial_regime="future"
        )


def test_initial_belief_changes_only_causal_prior() -> None:
    times = [30.0 + 24.0 * day for day in range(6)]
    products = ["P_C"] * 6
    _calendar, low = observable_calendar(
        request_times=times,
        request_products=products,
        decision_start=0.0,
        decision_weeks=1,
        policy_id="belief_extreme_v1",
        initial_action=1,
        initial_belief_c=0.2,
    )
    _calendar, high = observable_calendar(
        request_times=times,
        request_products=products,
        decision_start=0.0,
        decision_weeks=1,
        policy_id="belief_extreme_v1",
        initial_action=1,
        initial_belief_c=0.8,
    )
    assert low[0].prior_request_products == high[0].prior_request_products == ()
    assert low[0].belief_c == pytest.approx(0.2)
    assert high[0].belief_c == pytest.approx(0.8)


def test_cross_campaign_null_and_direction() -> None:
    assert cross_campaign_transition(0.9, 0.5) == pytest.approx(0.5)
    assert cross_campaign_transition(0.9, 0.9) > 0.5
    assert cross_campaign_transition(0.1, 0.9) < 0.5


def test_small_history_resets_physics_and_builds_distinct_priors() -> None:
    histories = [
        build_campaign_history(
            history_root=7_570_001 + index,
            campaigns=2,
            kappa=0.9,
            scheduler=scheduler(),
            regime_persistence=0.9,
            dominant_share=0.9,
        )
        for index in range(2)
    ]
    priors = arm_priors(
        histories=histories, regime_persistence=0.9, dominant_share=0.9
    )
    assert priors["reset_posterior_0p5"][0] == (0.5, 0.5)
    assert priors["retained_posterior"][0][0] == 0.5
    assert 0.0 < priors["retained_posterior"][0][1] < 1.0
    assert priors["shuffled_posterior"][0] == priors["retained_posterior"][1]
    assert all(row.skeleton.prefix_state_hash for history in histories for row in history)


def test_campaign_final_posterior_uses_only_observed_labels() -> None:
    posterior = campaign_final_posterior(
        initial_belief_c=0.5,
        order_products=["P_C"] * 6 + ["P_H"] * 6,
        regime_persistence=0.9,
        dominant_share=0.9,
    )
    assert 0.0 < posterior < 0.5
