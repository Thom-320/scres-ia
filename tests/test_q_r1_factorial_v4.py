from __future__ import annotations

import time

import numpy as np

from scripts.evaluate_program_q_replication import scheduler
from supply_chain.q_r1_comparator_v2 import ComparatorV2Config
from supply_chain.q_r1_factorial_v4 import (
    campaign_as_structured_state,
    matched_prior_paths,
    structured_pair_rows,
)
from supply_chain.retained_context_discovery import build_campaign_history


def _history(kappa: float = 0.90):
    return build_campaign_history(
        history_root=7_570_801,
        campaigns=12,
        kappa=kappa,
        scheduler=scheduler(),
        regime_persistence=0.90,
        dominant_share=0.90,
    )


def _metrics() -> dict[str, float]:
    return {
        "early_ret_complete_cohort": 0.8,
        "early_ret_visible": 0.81,
        "ret_visible": 0.82,
        "ret_full": 0.0,
        "worst_product_fill": 0.7,
        "unresolved_orders": 1.0,
        "unresolved_quantity": 2_500.0,
        "lost_orders": 0.0,
        "lost_quantity": 0.0,
        "service_loss_auc": 10.0,
    }


def test_campaign_conversion_preserves_physical_identity() -> None:
    campaign = _history()[0]
    converted = campaign_as_structured_state(campaign)
    assert converted.history_root == campaign.history_root
    assert converted.campaign_index == campaign.campaign_index
    assert converted.theta == (0.90, 0.90)
    assert converted.skeleton is campaign.skeleton
    assert converted.skeleton.skeleton_sha256 == campaign.skeleton.skeleton_sha256


def test_matched_prior_path_is_policy_independent_and_has_one_value_per_campaign() -> None:
    history = _history()
    paths = matched_prior_paths([history])
    assert len(paths) == 1
    assert len(paths[0]) == 12
    assert paths[0][0] == 0.5
    assert np.all((0.0 <= np.asarray(paths[0])) & (np.asarray(paths[0]) <= 1.0))


def test_structured_pair_emits_both_arms_and_mandatory_ledger() -> None:
    history = _history()
    paths = matched_prior_paths([history])

    def fake_builder(**kwargs):
        prior = kwargs["belief"].probability_regime_c
        return (0,) * 8, {"prior_seen": prior}

    def fake_evaluator(**_kwargs):
        return _metrics()

    rows = structured_pair_rows(
        histories=[history],
        prior_paths=paths,
        scheduler=scheduler(),
        config=ComparatorV2Config(
            horizon=4,
            conditional_paths=256,
            mode="scenario",
            worst_product_floor=0.0,
        ),
        calendar_builder=fake_builder,
        calendar_evaluator=fake_evaluator,
        cache={},
    )
    assert len(rows) == 24
    assert {row["arm"] for row in rows} == {
        "structured_reset",
        "structured_retained",
    }
    assert all(row["skeleton_sha256"] == history[row["campaign_index"]].skeleton.skeleton_sha256 for row in rows)
    assert all("service_loss_auc" in row for row in rows)
    assert sum(not row["structured_cache_hit"] for row in rows) == 23
    assert sum(row["structured_cache_hit"] for row in rows) == 1
    reset = [row for row in rows if row["arm"] == "structured_reset"]
    assert {row["explicit_prior"] for row in reset} == {0.5}


def test_structured_pair_fails_closed_when_service_is_missing() -> None:
    history = _history()

    def fake_builder(**_kwargs):
        return (0,) * 8, {}

    incomplete = _metrics()
    del incomplete["lost_quantity"]

    def fake_evaluator(**_kwargs):
        return incomplete

    with np.testing.assert_raises_regex(RuntimeError, "lost_quantity"):
        structured_pair_rows(
            histories=[history],
            prior_paths=matched_prior_paths([history]),
            scheduler=scheduler(),
            config=ComparatorV2Config(horizon=1, conditional_paths=1),
            calendar_builder=fake_builder,
            calendar_evaluator=fake_evaluator,
        )


def test_structured_budget_stops_before_starting_unaffordable_work() -> None:
    history = _history()
    calls = 0

    def should_not_run(**_kwargs):
        nonlocal calls
        calls += 1
        return (0,) * 8, {}

    with np.testing.assert_raises_regex(
        RuntimeError, "STOP_COMPUTE_BUDGET_PREDECLARED"
    ):
        structured_pair_rows(
            histories=[history],
            prior_paths=matched_prior_paths([history]),
            scheduler=scheduler(),
            config=ComparatorV2Config(horizon=1, conditional_paths=1),
            calendar_builder=should_not_run,
            calendar_evaluator=lambda **_kwargs: _metrics(),
            cache={},
            per_calendar_hard_cap_seconds=2.0,
            aggregate_hard_cap_seconds=1.0,
        )
    assert calls == 0


def test_structured_progress_is_persistable_after_each_completed_row() -> None:
    history = _history()
    snapshots: list[tuple[int, int]] = []

    rows = structured_pair_rows(
        histories=[history],
        prior_paths=matched_prior_paths([history]),
        scheduler=scheduler(),
        config=ComparatorV2Config(horizon=1, conditional_paths=1),
        calendar_builder=lambda **_kwargs: ((0,) * 8, {}),
        calendar_evaluator=lambda **_kwargs: _metrics(),
        cache={},
        progress_callback=lambda completed, cache: snapshots.append(
            (len(completed), len(cache))
        ),
    )
    assert len(rows) == 24
    assert snapshots[0] == (1, 1)
    assert snapshots[-1] == (24, 23)


def test_over_cap_unit_is_rejected_with_receipt_after_prior_rows_persist() -> None:
    history = _history()
    builder_calls = 0
    progress_snapshots: list[tuple[int, int]] = []
    rejections: list[tuple[dict[str, object], int, int]] = []

    def builder(**_kwargs):
        nonlocal builder_calls
        builder_calls += 1
        if builder_calls == 2:
            time.sleep(0.02)
        return (0,) * 8, {"builder_call": builder_calls}

    with np.testing.assert_raises_regex(
        RuntimeError, "STOP_COMPUTE_BUDGET_PREDECLARED"
    ):
        structured_pair_rows(
            histories=[history],
            prior_paths=matched_prior_paths([history]),
            scheduler=scheduler(),
            config=ComparatorV2Config(horizon=1, conditional_paths=1),
            calendar_builder=builder,
            calendar_evaluator=lambda **_kwargs: _metrics(),
            cache={},
            per_calendar_hard_cap_seconds=0.005,
            aggregate_hard_cap_seconds=1.0,
            progress_callback=lambda completed, cache: progress_snapshots.append(
                (len(completed), len(cache))
            ),
            rejection_callback=lambda receipt, completed, cache: rejections.append(
                (receipt, len(completed), len(cache))
            ),
        )

    assert progress_snapshots[-1] == (2, 1)
    assert len(rejections) == 1
    receipt, rows_before_rejection, cache_before_rejection = rejections[0]
    assert receipt["status"] == "REJECTED_OVER_CAP"
    assert receipt["action_eligible"] is False
    assert receipt["observed_seconds"] > receipt["per_calendar_hard_cap_seconds"]
    assert len(receipt["calendar_sha256"]) == 64
    assert len(receipt["diagnostics_sha256"]) == 64
    assert len(receipt["metrics_sha256"]) == 64
    assert rows_before_rejection == 2
    assert cache_before_rejection == 1
