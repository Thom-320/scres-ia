from __future__ import annotations

import json
from pathlib import Path

from supply_chain.q_r1_metaepisode_env import FACTORIAL_OBSERVATION_DIM


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "contracts/q_r1_matched_retention_factorial_v4.DRAFT.json"
COLLISION = (
    ROOT
    / "research/paper2_exhaustive_search"
    / "q_r1_matched_retention_factorial_v4_seed_collision_audit.DRAFT.json"
)


def _range(bounds: list[int]) -> set[int]:
    return set(range(int(bounds[0]), int(bounds[1]) + 1))


def test_v4_is_draft_and_cannot_open_roots() -> None:
    contract = json.loads(CONTRACT.read_text())
    assert contract["status"] == "DRAFT_PROSPECTIVE_UNOPENED_NOT_AUTHORITY"
    assert contract["data_splits"]["opened"] is False
    assert contract["execution_custody"][
        "fresh_roots_may_open_only_after_external_pre_freeze_pass"
    ]
    assert contract["boundaries"]["no_new_roots_may_open_while_status_is_draft"]


def test_v4_splits_and_optimizer_seeds_are_fresh_and_disjoint() -> None:
    contract = json.loads(CONTRACT.read_text())
    splits = contract["data_splits"]
    training = _range(splits["training_history_roots"])
    selection = _range(splits["checkpoint_selection_history_roots"])
    confirmation = _range(splits["reserved_confirmation_history_roots"])
    optimizer = set(map(int, splits["optimizer_seeds"]))
    assert len(training) == 40
    assert len(selection) == 16
    assert len(confirmation) == 64
    assert len(optimizer) == 5
    assert training.isdisjoint(selection | confirmation | optimizer)
    assert selection.isdisjoint(confirmation | optimizer)
    assert confirmation.isdisjoint(optimizer)

    collision = json.loads(COLLISION.read_text())
    assert collision["collision_clean"] is True
    assert collision["roots_opened"] is False
    assert collision["optimizer_seeds_opened"] is False
    assert collision["authority_to_open"] is False


def test_v4_freezes_runtime_and_real_checkpoint_selection() -> None:
    contract = json.loads(CONTRACT.read_text())
    training = contract["training_protocol"]
    assert training["screen_timesteps_per_seed"] == 96_000
    assert training["full_timesteps_per_seed"] == 240_000
    assert training["checkpoint_interval_timesteps"] == 24_000
    assert training["rollout_steps"] == 480
    assert training["batch_size"] == 96
    assert len(training["screen_configurations"]) == 8
    selection = training["checkpoint_selection"]
    assert selection["split"] == "checkpoint_selection_history_roots"
    assert selection["confirmation_return_used"] is False
    assert selection["oracle_return_used"] is False
    configuration = training["configuration_selection"]
    assert configuration["screen_advances"] == 2
    assert configuration["confirmation_return_used"] is False
    assert configuration["oracle_return_used"] is False


def test_v4_can_compute_north_star_and_all_factorial_estimands() -> None:
    contract = json.loads(CONTRACT.read_text())
    arms = {row["id"] for row in contract["factorial_arms_same_checkpoint"]}
    assert arms == {"P0_H0", "P1_H0", "P0_H1", "P1_H1"}
    assert contract["estimands"]["neural_premium"] == "P1_H1 - structured_retained"
    comparators = contract["structured_comparators"]
    assert comparators["retained"]["explicit_prior"] == "retained_prior_path"
    assert comparators["reset"]["explicit_prior"] == 0.5
    assert comparators["same_histories_and_skeletons_as_learner"] is True
    scope = comparators["evaluation_scope"]
    assert scope["development_checkpoint_selection"]["history_roots"] == [
        7_670_101,
        7_670_116,
    ]
    assert scope["development_checkpoint_selection"]["campaign_indices"] == [0, 1]
    budget = comparators["compute_budget"]
    assert budget["per_calendar_hard_cap_seconds"] == 600
    assert budget["on_budget_exceeded"] == "STOP_COMPUTE_BUDGET_PREDECLARED"
    assert budget["post_open_scope_reduction_forbidden"] is True
    assert FACTORIAL_OBSERVATION_DIM == 23


def test_v4_contains_all_kappa_cells_and_secondary_service_ledger() -> None:
    contract = json.loads(CONTRACT.read_text())
    assert contract["physical_contract"][
        "cross_campaign_knowledge_persistence_kappa_cells"
    ] == {"iid_null": 0.5, "dose": 0.75, "primary": 0.9}
    service = set(contract["mandatory_outputs"]["service_secondary"])
    assert {
        "worst_product_fill",
        "unresolved_orders",
        "unresolved_quantity",
        "lost_orders",
        "lost_quantity",
        "service_loss",
    } <= service
    assert contract["static_bar_protocol"][
        "computed_once_over_all_16_selection_roots"
    ]
    assert contract["static_bar_protocol"][
        "development_workers_may_not_recompute_the_bar"
    ]


def test_v4_requires_separate_immutable_freeze_and_opening_receipts() -> None:
    contract = json.loads(CONTRACT.read_text())
    custody = contract["execution_custody"]
    assert custody["development_opening_receipt_required"] is True
    assert custody["freeze_receipt_separate_and_immutable"] is True
    assert custody["execution_requires_git_metadata"] is True
    assert custody["execution_requires_clean_worktree"] is True
    assert custody["checkpoint_rows_persisted_before_next_checkpoint"] is True
    assert custody["structured_rows_and_cache_persisted_after_each_campaign"] is True
