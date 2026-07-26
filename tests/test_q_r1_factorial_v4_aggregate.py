from __future__ import annotations

from pathlib import Path

import pytest

from scripts.aggregate_q_r1_matched_retention_factorial_v4 import rank


def worker(
    directory: Path,
    config: str,
    seed: int,
    primary: float,
    premium: float,
):
    path = directory / f"{config}-{seed}.json"
    path.write_text("{}\n")
    return (
        path,
        {
            "config_id": config,
            "optimizer_seed": seed,
            "selected_checkpoint": {"timesteps": 24_000},
            "checkpoint_selection_scores": {
                "24000": [primary, premium, 0.02, -0.001, -24_000]
            },
        },
    )


def test_rank_requires_complete_config_by_seed_coverage(tmp_path: Path) -> None:
    rows = [
        worker(tmp_path, "s01", 1, 0.7, 0.01),
        worker(tmp_path, "s01", 2, 0.7, 0.01),
    ]
    with pytest.raises(ValueError, match="seed coverage"):
        rank(rows, expected_configs={"s01"}, expected_seeds={1, 2, 3})


def test_rank_uses_predeclared_metrics_then_config_id(tmp_path: Path) -> None:
    rows = [
        worker(tmp_path, "s01", seed, 0.7, 0.01) for seed in (1, 2, 3)
    ] + [
        worker(tmp_path, "s02", seed, 0.7, 0.02) for seed in (1, 2, 3)
    ]
    result = rank(
        rows,
        expected_configs={"s01", "s02"},
        expected_seeds={1, 2, 3},
    )
    assert [row["config_id"] for row in result] == ["s02", "s01"]
