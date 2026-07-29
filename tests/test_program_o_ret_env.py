from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from supply_chain.program_o_full_des import run_program_o_full_des_episode
from supply_chain.program_o_full_des_transducer import (
    direct_full_des_vector,
    extract_full_des_skeleton,
    full_action_calendars,
    simulate_full_des_frontier,
)
from supply_chain.program_o_ret_env import (
    OBSERVATION_DIM,
    ProgramORetOnlyEnv,
)
from supply_chain.program_o_state_rich import (
    StateRichConfiguration,
    state_rich_calendar,
)
from scripts.evaluate_program_o_ret_learner import simultaneous_bootstrap


ROOT = Path(__file__).resolve().parent.parent


def scheduler() -> dict[str, list[str]]:
    contract = json.loads(
        (ROOT / "contracts/program_o_full_des_hpi_translation_v1.json").read_text()
    )
    key = contract["action"]["primary_scheduler"]
    return contract["action"]["within_week_schedulers"][key]


@pytest.fixture(scope="module")
def skeleton():
    value, _sim = extract_full_des_skeleton(
        seed=94800001,
        scheduler=scheduler(),
        regime_persistence=0.75,
        dominant_share=0.90,
        downstream_freight_physics_mode="fixed_clock_physical_v1",
    )
    return value


def test_action_override_preserves_classical_observations(skeleton) -> None:
    config = StateRichConfiguration("belief_mpc", 3)
    calendar, original = state_rich_calendar(
        skeleton=skeleton.as_dict(),
        scheduler=scheduler(),
        config=config,
        regime_persistence=0.75,
        dominant_share=0.90,
    )
    replayed, overridden = state_rich_calendar(
        skeleton=skeleton.as_dict(),
        scheduler=scheduler(),
        config=config,
        regime_persistence=0.75,
        dominant_share=0.90,
        action_overrides=calendar,
    )
    assert replayed == calendar
    assert [row.observation.observation_sha256 for row in overridden] == [
        row.observation.observation_sha256 for row in original
    ]


def test_env_has_terminal_ret_reward_only_and_no_latent_fields(skeleton) -> None:
    env = ProgramORetOnlyEnv(
        scheduler=scheduler(), tape_seed_start=94800001, tape_seed_end=94800001
    )
    observation, info = env.reset(options={"skeleton": skeleton, "tape_seed": 94800001})
    assert observation.shape == (OBSERVATION_DIM,)
    assert set(info) == {"observation_sha256"}
    calendar = (0, 1, 2, 3, 3, 2, 1, 0)
    rewards = []
    for action in calendar:
        observation, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        assert not truncated
    expected = simulate_full_des_frontier(
        skeleton=skeleton,
        scheduler=scheduler(),
        calendars=np.asarray([calendar], dtype=np.uint8),
    )
    assert rewards[:-1] == [0.0] * 7
    assert terminated
    assert rewards[-1] == pytest.approx(float(expected["ret_visible"][0]), abs=0.0)
    assert info["calendar"] == list(calendar)
    assert "ret_visible_cvar10" in info["metrics"]


def test_fixed_clock_transducer_matches_direct_full_des(skeleton) -> None:
    calendar = (0, 1, 2, 3, 3, 2, 1, 0)
    sim, panel = run_program_o_full_des_episode(
        seed=94800001,
        calendar=calendar,
        scheduler=scheduler(),
        regime_persistence=0.75,
        dominant_share=0.90,
        downstream_freight_physics_mode="fixed_clock_physical_v1",
    )
    direct = direct_full_des_vector(sim, panel)
    replay = simulate_full_des_frontier(
        skeleton=skeleton,
        scheduler=scheduler(),
        calendars=np.asarray([calendar], dtype=np.uint8),
    )
    for key, value in direct.items():
        assert float(replay[key][0]) == pytest.approx(float(value), abs=1e-8), key


def test_fungible_null_is_exact(skeleton) -> None:
    calendars = full_action_calendars()[:32]
    panel = simulate_full_des_frontier(
        skeleton=skeleton,
        scheduler=scheduler(),
        calendars=calendars,
        complete_substitution=True,
    )
    for key in ("ret_visible", "ret_full", "quantity_ret_full"):
        assert np.ptp(panel[key]) == pytest.approx(0.0, abs=0.0)


def test_invalid_override_and_seed_exhaustion(skeleton) -> None:
    with pytest.raises(ValueError):
        state_rich_calendar(
            skeleton=skeleton.as_dict(),
            scheduler=scheduler(),
            config=StateRichConfiguration("belief_mpc", 3),
            regime_persistence=0.75,
            dominant_share=0.90,
            action_overrides=(0,) * 7,
        )
    env = ProgramORetOnlyEnv(
        scheduler=scheduler(),
        tape_seed_start=1,
        tape_seed_end=1,
        skeleton_factory=lambda _seed, _cell: skeleton,
    )
    env.reset()
    with pytest.raises(RuntimeError, match="namespace exhausted"):
        env.reset()


def test_bootstrap_gate_detects_both_required_comparator_contrasts() -> None:
    rows = {}
    for cell in ("a", "b", "c"):
        learner = {"ret_visible": np.full((10, 48), 0.75)}
        open_loop = {"ret_visible": np.full((48, 4), 0.70)}
        classical = {"ret_visible": np.full((3, 48), 0.72)}
        for key in ("ret_full", "quantity_ret_full", "worst_product_fill"):
            learner[key] = np.full((10, 48), 0.81)
            open_loop[key] = np.full((48, 4), 0.79)
            classical[key] = np.full((3, 48), 0.79)
        rows[cell] = {
            "learner": learner,
            "open_loop": open_loop,
            "classical": classical,
        }
    result = simultaneous_bootstrap(rows, 200)
    assert len(result["estimates"]) == 24
    assert all(value["lcb95"] >= 0.01 for value in result["estimates"].values())
