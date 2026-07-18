from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads(
    (ROOT / "contracts/program_s_product_mix_risk_interaction_gsa_v1.json").read_text()
)
PRIOR = json.loads(
    (
        ROOT
        / "research/paper2_exhaustive_search/program_s_prior_risk_screen_provenance_v1.json"
    ).read_text()
)
LEDGER = json.loads(
    (
        ROOT
        / "research/paper2_exhaustive_search/program_s_garrido_unexploited_mechanism_ledger_v1.json"
    ).read_text()
)
PAPER3 = json.loads(
    (ROOT / "contracts/program_s_paper3_retained_knowledge_v1.json").read_text()
)


def test_program_s_is_prospective_and_cannot_rewrite_closed_programs() -> None:
    assert CONTRACT["historical_verdicts_immutable"] == {
        "program_o": "STOP_PROGRAM_O_AFTER_CORRECTIVE_VALIDATION",
        "program_o_r": "STOP_CALIBRATION_NOT_ELIGIBLE",
        "program_q": "PASS_POWER_SELECT_N_256",
    }
    assert CONTRACT["status"] == "FROZEN_S0_IMPLEMENTATION_NO_SCIENTIFIC_SEEDS_OPENED"
    assert CONTRACT["paper3"]["authorized_before_s4_pass"] is False
    assert CONTRACT["infrastructure"]["program_q_vps_priority"] is True
    assert CONTRACT["infrastructure"]["depends_on_david"] is False


def test_program_s_seed_blocks_are_declared_but_unopened() -> None:
    for stage, key in (
        ("s1_morris", "development_tapes"),
        ("s2", "fresh_tapes"),
        ("s3", "fresh_tapes"),
        ("s4", "training_tapes"),
        ("s4", "calibration_tapes"),
        ("s4", "confirmation_tapes"),
    ):
        assert CONTRACT[stage][key]["opened"] is False
    assert CONTRACT["paper3"]["seed_block_opened"] is False


def test_program_s_action_and_masks_do_not_buy_resources() -> None:
    decision = CONTRACT["decision_contract"]
    assert decision["action_space"] == "Discrete(4)"
    assert decision["open_loop_frontier_size"] == 65536
    assert set(CONTRACT["physical_masks"]) == {
        "PRODUCTION_QUALITY_SURGE",
        "LOC_SURGE",
        "CROSS_ECHELON_SURGE",
    }
    assert CONTRACT["r3_diagnostic"]["included_in_screen"] is False
    assert CONTRACT["r3_diagnostic"]["frequency_or_impact_scaling_forbidden"] is True


def test_prior_risk_stop_is_ported_without_false_exhaustion_claim() -> None:
    assert PRIOR["terminal_verdict"] == "STOP_V1_1_BEFORE_G2_RARE_RISK_FIXTURE_NOT_POPULATED"
    assert PRIOR["verified_stage_results"]["G2_frequency_impact_map"] == "NOT_EXECUTED"
    assert PRIOR["immutable_historical_record"] is True


def test_garrido_ledger_routes_every_omitted_mechanism() -> None:
    rows = LEDGER["mechanisms"]
    assert len(rows) == 10
    assert len({row["id"] for row in rows}) == len(rows)
    assert all(row["source"] and row["theory_gate"] for row in rows)
    included = {row["id"]: row["program_s_v1_status"] for row in rows}
    assert included["nonstationary_demand_cross_functional_forecast"] == "included_only_as_operational_alarm"
    assert included["history_dependent_learning_loop"] == "paper3_protocol_only_until_s4_pass"


def test_paper3_is_frozen_but_sealed_until_program_s_paper2_pass() -> None:
    assert PAPER3["status"] == "SEALED_UNAUTHORIZED_UNTIL_PROGRAM_S_S4_PAPER2_PASS"
    assert PAPER3["seed_block"] == "7530001+"
    assert PAPER3["seed_block_opened"] is False
    assert PAPER3["reset_boundary"]["knowledge_only_may_persist"] is True
    assert "retained_empirical_bayes_belief_mpc" in PAPER3["arms"]
