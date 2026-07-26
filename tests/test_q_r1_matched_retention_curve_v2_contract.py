from __future__ import annotations

import hashlib
import json
from pathlib import Path

from supply_chain.q_r1_metaepisode_env import (
    CAMPAIGNS_PER_METAEPISODE,
    DECISIONS_PER_METAEPISODE,
    META_OBSERVATION_DIM,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "contracts/q_r1_matched_retention_curve_v2.json"
RECEIPT = ROOT / "contracts/q_r1_matched_retention_curve_v2_freeze_receipt.json"
COLLISION = (
    ROOT
    / "research/paper2_exhaustive_search"
    / "q_r1_matched_retention_curve_v2_seed_collision_audit.json"
)


def _range(bounds: list[int]) -> set[int]:
    return set(range(int(bounds[0]), int(bounds[1]) + 1))


def test_contract_is_frozen_and_splits_are_disjoint() -> None:
    contract = json.loads(CONTRACT.read_text())
    assert contract["status"] == "FROZEN_PROSPECTIVE_UNOPENED"
    splits = contract["data_splits"]
    training = _range(splits["training_history_roots"])
    selection = _range(splits["checkpoint_selection_history_roots"])
    confirmation = _range(splits["reserved_confirmation_history_roots"])
    assert training.isdisjoint(selection)
    assert training.isdisjoint(confirmation)
    assert selection.isdisjoint(confirmation)
    assert len(confirmation) == 64


def test_contract_matches_metaepisode_implementation() -> None:
    contract = json.loads(CONTRACT.read_text())
    physical = contract["physical_contract"]
    assert physical["campaigns_per_metaepisode"] == CAMPAIGNS_PER_METAEPISODE
    assert physical["decisions_per_metaepisode"] == DECISIONS_PER_METAEPISODE
    assert contract["information_rights"]["learner_inputs"].startswith(
        f"{META_OBSERVATION_DIM - 1}-dimensional"
    )
    assert physical["episode_start_retained"].startswith("true only")
    assert physical["episode_start_reset"].startswith("true at every")


def test_contract_separates_cumulative_and_within_campaign_arms() -> None:
    contract = json.loads(CONTRACT.read_text())
    roles = {arm["id"]: arm["role"] for arm in contract["arms"]}
    assert "primary learned cumulative" in roles["recurrent_ppo_retained_state"]
    assert "not eligible for cumulative-learning claim" in roles[
        "ppo_mlp_within_campaign"
    ]
    assert contract["kan_boundary"]["authorized_now"] is False
    assert contract["paper_boundary"]["submission_a_program_q_unchanged"] is True


def test_collision_audit_keeps_reserved_roots_unopened() -> None:
    audit = json.loads(COLLISION.read_text())
    assert audit["collision_clean"] is True
    assert audit["roots_opened"] is False
    assert audit["confirmation_roots_opened"] is False


def test_freeze_receipt_matches_contract_bytes() -> None:
    receipt = json.loads(RECEIPT.read_text())
    digest = hashlib.sha256(CONTRACT.read_bytes()).hexdigest()
    assert receipt["status"] == "FROZEN_PROSPECTIVE_UNOPENED"
    assert receipt["contract_sha256"] == digest
    assert receipt["frozen_before_training_roots_opened"] is True
    assert receipt["training_roots_opened"] is False
