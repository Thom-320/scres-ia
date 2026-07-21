from __future__ import annotations

import numpy as np

from scripts.run_program_t_action_regret_predictor import history_vector
from scripts.run_program_u_static_discovery_benchmark import (
    cem_search,
    diagonal_es_search,
    random_search,
)


class QuadraticOracle:
    def __init__(self) -> None:
        self.proposals = 0

    def score(self, calendar: tuple[int, ...]) -> float:
        self.proposals += 1
        target = np.asarray((3, 2, 3, 2, 1, 1, 0, 0))
        return -float(np.square(np.asarray(calendar) - target).sum())

    def score_many(self, calendars: list[tuple[int, ...]]) -> np.ndarray:
        return np.asarray([self.score(calendar) for calendar in calendars])


def test_static_searchers_respect_exact_candidate_budget() -> None:
    for search in (random_search, cem_search, diagonal_es_search):
        oracle = QuadraticOracle()
        result = search(oracle, budget=32, seed=17)
        assert result.proposed == 32
        assert oracle.proposals == 32
        assert len(result.trace) == 32
        assert all(left <= right for left, right in zip(result.trace, result.trace[1:]))


def test_history_vector_right_aligns_and_zero_pads() -> None:
    first = np.asarray([1.0, 2.0]); second = np.asarray([3.0, 4.0])
    result = history_vector([first, second]).reshape(8, 2)
    np.testing.assert_array_equal(result[:-2], 0.0)
    np.testing.assert_array_equal(result[-2], first)
    np.testing.assert_array_equal(result[-1], second)
