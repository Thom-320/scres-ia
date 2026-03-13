from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from supply_chain.dkana import (
    DKANA_CONFIG_FIELDS,
    DKANAPolicy,
    build_dkana_windows,
    build_mfsc_relational_state,
    build_previous_action_context,
)
from supply_chain.external_env_interface import (
    ACTION_FIELDS,
    CONTROL_CONTEXT_FIELDS,
    OBSERVATION_FIELDS,
    STATE_CONSTRAINT_FIELDS,
)


def make_synthetic_export_arrays() -> dict[str, np.ndarray]:
    num_steps = 3
    observations = np.stack(
        [
            np.linspace(0.0, 1.0, len(OBSERVATION_FIELDS), dtype=np.float32) + offset
            for offset in (0.0, 1.0, 2.0)
        ],
        axis=0,
    )
    actions = np.array(
        [
            [0.1, 0.0, -0.1, 0.2, -1.0],
            [0.2, 0.1, -0.2, 0.3, 0.0],
            [0.3, 0.2, -0.3, 0.4, 1.0],
        ],
        dtype=np.float32,
    )
    episode_ids = np.array([0, 0, 1], dtype=np.int32)
    constraint_context = np.tile(
        np.arange(len(CONTROL_CONTEXT_FIELDS), dtype=np.float32),
        (num_steps, 1),
    )
    state_constraint_context = np.stack(
        [
            np.linspace(
                10.0,
                20.0,
                len(STATE_CONSTRAINT_FIELDS),
                dtype=np.float32,
            )
            + offset
            for offset in (0.0, 1.0, 2.0)
        ],
        axis=0,
    )
    rewards = np.array([1.0, 0.5, -0.25], dtype=np.float32)
    return {
        "observations": observations,
        "actions": actions,
        "episode_ids": episode_ids,
        "constraint_context": constraint_context,
        "state_constraint_context": state_constraint_context,
        "rewards": rewards,
    }


def test_build_mfsc_relational_state_preserves_feature_order() -> None:
    observation = np.linspace(0.0, 1.0, len(OBSERVATION_FIELDS), dtype=np.float32)
    state_constraints = np.linspace(
        10.0,
        20.0,
        len(STATE_CONSTRAINT_FIELDS),
        dtype=np.float32,
    )
    row_matrix = build_mfsc_relational_state(observation, state_constraints)

    assert row_matrix.shape == (
        len(OBSERVATION_FIELDS) + len(STATE_CONSTRAINT_FIELDS),
        3,
    )
    assert np.array_equal(
        row_matrix[:, 0],
        np.arange(row_matrix.shape[0], dtype=np.float32),
    )
    assert np.all(row_matrix[:, 1] == 0.0)
    assert np.allclose(
        row_matrix[:, 2],
        np.concatenate([observation, state_constraints]),
    )


def test_build_previous_action_context_resets_per_episode() -> None:
    actions = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ],
        dtype=np.float32,
    )
    episode_ids = np.array([0, 0, 1], dtype=np.int32)
    previous_actions = build_previous_action_context(actions, episode_ids)

    assert np.allclose(previous_actions[0], 0.0)
    assert np.allclose(previous_actions[1], actions[0])
    assert np.allclose(previous_actions[2], 0.0)


def test_build_dkana_windows_left_pads_history_and_context() -> None:
    export_arrays = make_synthetic_export_arrays()
    dataset = build_dkana_windows(window_size=2, **export_arrays)
    row_count = len(OBSERVATION_FIELDS) + len(STATE_CONSTRAINT_FIELDS)

    assert dataset.row_matrices.shape == (3, 2, row_count, 3)
    assert dataset.config_context.shape == (3, 2, len(DKANA_CONFIG_FIELDS))
    assert dataset.action_targets.shape == (3, len(ACTION_FIELDS))
    assert dataset.reward_targets is not None
    assert dataset.time_mask.tolist() == [
        [False, True],
        [True, True],
        [False, True],
    ]
    assert np.allclose(dataset.config_context[0, 1, -len(ACTION_FIELDS) :], 0.0)
    assert np.allclose(
        dataset.config_context[1, 1, -len(ACTION_FIELDS) :],
        export_arrays["actions"][0],
    )
    assert dataset.config_fields == DKANA_CONFIG_FIELDS


def test_dkana_policy_forward_returns_distribution() -> None:
    export_arrays = make_synthetic_export_arrays()
    dataset = build_dkana_windows(window_size=2, **export_arrays)
    model = DKANAPolicy(
        config_dim=dataset.config_context.shape[-1],
        action_dim=len(ACTION_FIELDS),
        max_sequence_length=2,
    )

    distribution = model(
        torch.from_numpy(dataset.row_matrices[:2]),
        torch.from_numpy(dataset.config_context[:2]),
        torch.from_numpy(dataset.time_mask[:2]),
    )

    assert distribution.mean.shape == (2, len(ACTION_FIELDS))
    assert distribution.stddev.shape == (2, len(ACTION_FIELDS))
    assert torch.all(distribution.stddev > 0)


def test_build_dkana_dataset_script_writes_numpy_outputs(tmp_path: Path) -> None:
    export_dir = tmp_path / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    export_arrays = make_synthetic_export_arrays()
    np.save(export_dir / "observations.npy", export_arrays["observations"])
    np.save(export_dir / "actions.npy", export_arrays["actions"])
    np.save(export_dir / "episode_ids.npy", export_arrays["episode_ids"])
    np.save(export_dir / "constraint_context.npy", export_arrays["constraint_context"])
    np.save(
        export_dir / "state_constraint_context.npy",
        export_arrays["state_constraint_context"],
    )
    np.save(export_dir / "rewards.npy", export_arrays["rewards"])
    (export_dir / "env_spec.json").write_text(
        json.dumps({"observation_fields": list(OBSERVATION_FIELDS)}),
        encoding="utf-8",
    )
    (export_dir / "state_constraint_fields.json").write_text(
        json.dumps({"fields": list(STATE_CONSTRAINT_FIELDS)}),
        encoding="utf-8",
    )

    command = [
        sys.executable,
        "scripts/build_dkana_dataset.py",
        "--input-dir",
        str(export_dir),
        "--output-dir",
        str(tmp_path / "dkana"),
        "--window-size",
        "2",
    ]
    completed = subprocess.run(
        command,
        cwd="/Users/thom/Desktop/Universidad_Codigo/proyecto_grarrido_scres+ia",
        check=True,
        capture_output=True,
        text=True,
    )

    assert "dkana_row_matrices.npy" in completed.stdout
    metadata = json.loads((tmp_path / "dkana" / "metadata.json").read_text())
    assert metadata["row_matrices_shape"] == [
        3,
        2,
        len(OBSERVATION_FIELDS) + len(STATE_CONSTRAINT_FIELDS),
        3,
    ]
    assert metadata["config_context_shape"] == [3, 2, len(DKANA_CONFIG_FIELDS)]
