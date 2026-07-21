from __future__ import annotations

import json

from sb3_contrib import RecurrentPPO

from scripts.evaluate_program_q_replication import scheduler
from scripts.run_program_q2_warmstart_probe import (
    CALIBRATION,
    _model,
    _teacher_episodes,
    behavior_clone,
    training_episodes_required,
)
from supply_chain.program_o_ret_env import ProgramORetOnlyEnv


def test_static_and_structured_teacher_panels_are_complete() -> None:
    payload = json.loads(CALIBRATION.read_text())
    static = _teacher_episodes("static_bc", payload)
    structured = _teacher_episodes("structured_bc", payload)
    assert len(static) == len(structured) == 3 * 48
    assert all(obs.shape == (8, 21) and actions.shape == (8,) for obs, actions in static)
    assert all(obs.shape == (8, 21) and actions.shape == (8,) for obs, actions in structured)


def test_recurrent_behavior_clone_improves_teacher_accuracy() -> None:
    env = ProgramORetOnlyEnv(scheduler=scheduler(), tape_seed_start=1, tape_seed_end=20)
    model: RecurrentPPO = _model(env, 7)
    episode = _teacher_episodes("static_bc", json.loads(CALIBRATION.read_text()))[:2]
    history = behavior_clone(model, episode, epochs=2, seed=7)
    assert len(history) == 2
    assert history[-1]["mean_nll"] <= history[0]["mean_nll"]


def test_training_namespace_accounts_for_rollout_rounding_at_each_checkpoint() -> None:
    assert training_episodes_required((0, 10_000, 30_000, 60_000)) == 7_617
