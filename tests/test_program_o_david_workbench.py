from pathlib import Path

import numpy as np
import pytest

from research.program_o_david_workbench import (
    DEV_SEED_MIN,
    assert_development_seed,
    compact_summary,
    evaluate_policy,
    evaluate_policy_against_full_frontiers,
    integrity_report,
    swap_product_channels,
)


ROOT = Path(__file__).resolve().parent.parent


class FixedPolicy:
    label = "fixed_2"

    def reset_policy_state(self) -> None:
        return None

    def predict_action(self, observation: np.ndarray) -> int:
        assert observation.shape == (21,)
        return 2


def test_seed_guard_rejects_scientific_namespaces() -> None:
    assert assert_development_seed(DEV_SEED_MIN) == DEV_SEED_MIN
    with pytest.raises(ValueError):
        assert_development_seed(7_480_001)
    with pytest.raises(ValueError):
        assert_development_seed(748_100_001)


def test_fixed_policy_smoke_has_compact_metrics_and_integrity() -> None:
    rows = evaluate_policy(root=ROOT, policy=FixedPolicy(), seeds=[DEV_SEED_MIN])
    assert len(rows) == 3
    assert set(rows["calendar"]) == {"22222222"}
    summary = compact_summary(rows)
    assert len(summary) == 3
    assert set(summary["unique_calendars"]) == {1}
    report = integrity_report(rows)
    assert report["passed"] is True


def test_product_swap_changes_only_product_semantics() -> None:
    frame = np.linspace(0.0, 1.0, 21, dtype=np.float32)
    swapped = swap_product_channels(frame)
    assert swapped[0] == frame[1]
    assert swapped[1] == frame[0]
    assert swapped[12] == pytest.approx(1.0 - frame[12])
    assert swapped[13] == pytest.approx(1.0 - frame[13])
    assert np.array_equal(swapped[14:19], frame[[17, 16, 15, 14, 18]])
    assert np.array_equal(swapped[19:], frame[19:])
    assert np.allclose(swap_product_channels(swapped), frame)


def test_full_frontier_scoreboard_uses_all_cells_and_fixed_families() -> None:
    result = evaluate_policy_against_full_frontiers(
        root=ROOT, policy=FixedPolicy(), seeds=[DEV_SEED_MIN]
    )
    assert len(result.policy_rows) == 3
    assert len(result.scoreboard) == 3
    assert set(result.scoreboard["best_classical"].str.contains("__")) == {True}
    assert set(result.scoreboard["unique_calendars"]) == {1}
    assert all(
        row["comparator_rule"].startswith("maximum family mean")
        for row in result.diagnostics.values()
    )
