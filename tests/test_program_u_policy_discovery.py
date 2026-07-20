from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from research.paper2_exhaustive_search.program_u_efficient_cascade import (
    CandidatePoint,
    FirstTapeResult,
    batch_points,
    certification_mode,
    decide_incremental_screen,
)
from supply_chain.program_o_full_des_transducer import extract_full_des_skeleton
from supply_chain.program_u_policy_discovery import (
    EndogenousReviewProgramORetEnv,
    StaticCalendarDiscoveryEnv,
)
from supply_chain.program_u_dynamic_stress import (
    StressRegime,
    generate_dynamic_stress_tape,
)


ROOT = Path(__file__).resolve().parent.parent


def _scheduler() -> dict[str, list[str]]:
    contract = json.loads(
        (ROOT / "contracts/program_o_full_des_hpi_translation_v1.json").read_text()
    )
    key = contract["action"]["primary_scheduler"]
    return contract["action"]["within_week_schedulers"][key]


@pytest.fixture(scope="module")
def program_o_skeleton():
    value, _sim = extract_full_des_skeleton(
        seed=94800001,
        scheduler=_scheduler(),
        regime_persistence=0.75,
        dominant_share=0.90,
        downstream_freight_physics_mode="fixed_clock_physical_v1",
    )
    return value


def test_static_discovery_scores_only_after_complete_calendar() -> None:
    calls: list[tuple[tuple[int, ...], int]] = []

    def evaluator(calendar: tuple[int, ...], tape_id: int):
        calls.append((calendar, tape_id))
        return {"ret_visible": float(sum(calendar))}

    env = StaticCalendarDiscoveryEnv(
        evaluator=evaluator, tape_ids=(101, 102), horizon=3, action_count=4
    )
    obs, info = env.reset(options={"tape_id": 101})
    assert info["physical_evaluations"] == 0
    assert 101 not in obs
    for action in (3, 2):
        _obs, reward, terminated, _truncated, info = env.step(action)
        assert reward == 0.0
        assert not terminated
        assert info["physical_evaluations"] == 0
        assert calls == []
    _obs, reward, terminated, _truncated, info = env.step(1)
    assert terminated
    assert reward == 6.0
    assert calls == [((3, 2, 1), 101)]
    assert info["policy_class"] == "single_open_loop_calendar"
    assert info["physical_evaluations"] == 1


def test_static_discovery_observation_is_tape_independent() -> None:
    env = StaticCalendarDiscoveryEnv(
        evaluator=lambda calendar, tape: {"ret_visible": 0.0},
        tape_ids=(101, 999),
        horizon=2,
    )
    obs_a, _ = env.reset(options={"tape_id": 101})
    obs_b, _ = env.reset(options={"tape_id": 999})
    np.testing.assert_array_equal(obs_a, obs_b)


def test_cascade_stops_only_the_inexact_mask_and_limits_certification() -> None:
    candidates = (
        CandidatePoint("a", "production", 1.0),
        CandidatePoint("b", "production", 2.0),
        CandidatePoint("c", "loc", 3.0),
        CandidatePoint("d", "cross", 1.5),
    )
    first = (
        FirstTapeResult("a", False, 0.20, 1e-5),
        FirstTapeResult("b", True, 0.30, 0.0),
        FirstTapeResult("c", True, 0.01, 0.0),
        FirstTapeResult("d", True, 0.04, 0.0),
    )
    decision = decide_incremental_screen(
        candidates=candidates,
        first_tape=first,
        preliminary_threshold=0.02,
        certification_limit=1,
    )
    assert decision.stopped_masks == ("production",)
    assert decision.promoted_points == ("d",)
    assert decision.certification_points == ("d",)
    assert certification_mode("d", decision.certification_points) == "EXACT_FRONTIER"
    assert certification_mode("c", decision.certification_points) == "APPROXIMATE_SEARCH"


def test_worker_batches_amortize_process_startup() -> None:
    assert batch_points(("a", "b", "c", "d", "e"), batch_size=2) == (
        ("a", "b"),
        ("c", "d"),
        ("e",),
    )


def test_dynamic_stress_tape_is_crn_deterministic_and_policy_free() -> None:
    regimes = (
        StressRegime("normal", {"R11": 1.0}, {"R11": 1.0}, 0.0),
        StressRegime("surge", {"R11": 2.0}, {"R11": 1.5}, 0.3),
        StressRegime("recovery", {"R11": 0.5}, {"R11": 1.0}, 0.15),
    )
    kwargs = dict(
        seed=1234,
        regimes=regimes,
        transition_matrix=((0.7, 0.3, 0.0), (0.0, 0.6, 0.4), (0.5, 0.0, 0.5)),
        periods=50,
        potential_service_draws_per_period=200,
    )
    left = generate_dynamic_stress_tape(**kwargs)
    right = generate_dynamic_stress_tape(**kwargs)
    assert left == right
    assert left.sha256 == right.sha256
    assert set(left.regimes) <= {"normal", "surge", "recovery"}
    # The mean-one construction is statistical, not an exact per-tape rescale.
    assert abs(np.mean(left.potential_pt_multipliers) - 1.0) < 0.04


def test_zero_spread_processing_tape_is_exactly_deterministic() -> None:
    tape = generate_dynamic_stress_tape(
        seed=7,
        regimes=(StressRegime("normal", {}, {}, 0.0),),
        transition_matrix=((1.0,),),
        periods=8,
        potential_service_draws_per_period=20,
    )
    assert set(tape.potential_pt_multipliers) == {1.0}


def test_endogenous_review_spends_finite_attention_and_holds_mix(program_o_skeleton) -> None:
    env = EndogenousReviewProgramORetEnv(
        scheduler=_scheduler(),
        tape_seed_start=94800001,
        tape_seed_end=94800001,
        skeleton_factory=lambda _seed, _cell: program_o_skeleton,
        dwell_options=(1, 2, 4),
        review_budget=2,
    )
    _obs, info = env.reset()
    assert info["review_budget"] == 2
    # mix=2, dwell=2 -> encoded action 2*3+1.
    _obs, reward, terminated, _truncated, info = env.step(7)
    assert reward == 0.0
    assert not terminated
    assert info["realized_calendar"] == [2, 2]
    # Last review consumes the remaining six weeks regardless of requested dwell.
    _obs, reward, terminated, _truncated, info = env.step(9)  # mix=3, dwell=1
    assert terminated
    assert info["reviews_used"] == 2
    assert info["realized_calendar"] == [2, 2, 3, 3, 3, 3, 3, 3]
    assert info["review_trajectory"][-1]["executed_dwell"] == 6
    assert reward == info["metrics"]["ret_visible"]
