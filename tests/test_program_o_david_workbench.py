from pathlib import Path

import numpy as np
import pytest

from research.program_o_david_workbench import (
    DEV_SEED_MIN,
    assert_development_seed,
    compact_summary,
    evaluate_policy,
    integrity_report,
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
