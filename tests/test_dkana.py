from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from supply_chain.dkana import (
    DKANA_CONFIG_FIELDS,
    DKANAOnlinePolicyAdapter,
    DKANAPolicy,
    build_causal_attention_mask,
    build_dkana_config_fields,
    build_dkana_windows,
    build_enumeration_map,
    build_mfsc_relational_state,
    build_prefixed_variable_names,
    build_previous_action_context,
)
from supply_chain.external_env_interface import (
    ACTION_FIELDS,
    ACTION_FIELDS_TRACK_B_V1,
    CONTROL_CONTEXT_FIELDS,
    OBSERVATION_FIELDS_V3,
    OBSERVATION_FIELDS_V7,
    REWARD_TERM_FIELDS,
    STATE_CONSTRAINT_FIELDS,
    get_track_b_env_spec,
    make_dkana_track_b_env,
    make_track_b_env,
)


def make_synthetic_export_arrays() -> dict[str, np.ndarray]:
    num_steps = 3
    observations = np.stack(
        [
            np.linspace(0.0, 1.0, len(OBSERVATION_FIELDS_V3), dtype=np.float32) + offset
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
    observation = np.linspace(0.0, 1.0, len(OBSERVATION_FIELDS_V3), dtype=np.float32)
    state_constraints = np.linspace(
        10.0,
        20.0,
        len(STATE_CONSTRAINT_FIELDS),
        dtype=np.float32,
    )
    row_matrix = build_mfsc_relational_state(observation, state_constraints)

    assert row_matrix.shape == (
        len(OBSERVATION_FIELDS_V3) + len(STATE_CONSTRAINT_FIELDS),
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


def test_build_mfsc_relational_state_can_emit_temporal_relations() -> None:
    observation = np.array([1.0, 2.0], dtype=np.float32)
    previous_observation = np.array([2.0, 2.0], dtype=np.float32)
    state_constraints = np.array([4.0], dtype=np.float32)
    previous_state_constraints = np.array([3.0], dtype=np.float32)
    observation_fields = ("obs_a", "obs_b")
    state_fields = ("state_c",)
    variable_names = build_prefixed_variable_names(observation_fields, state_fields)
    enumeration_map = build_enumeration_map(variable_names)

    row_matrix = build_mfsc_relational_state(
        observation,
        state_constraints,
        observation_fields=observation_fields,
        state_constraint_fields=state_fields,
        enumeration_map=enumeration_map,
        previous_observation=previous_observation,
        previous_state_constraint_vector=previous_state_constraints,
        relation_mode="temporal_delta",
    )

    assert row_matrix.shape == (6, 3)
    temporal_rows = row_matrix[3:]
    assert temporal_rows[:, 1].tolist() == [
        float(enumeration_map.relation_index("<")),
        float(enumeration_map.relation_index("=")),
        float(enumeration_map.relation_index(">")),
    ]
    assert temporal_rows[:, 2].tolist() == [2.0, 2.0, 3.0]


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
    row_count = len(OBSERVATION_FIELDS_V3) + len(STATE_CONSTRAINT_FIELDS)

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


def test_build_dkana_windows_supports_track_b_context_fields() -> None:
    export_arrays = make_synthetic_export_arrays()
    export_arrays["actions"] = np.column_stack(
        [
            export_arrays["actions"],
            np.array([0.4, 0.5, 0.6], dtype=np.float32),
            np.array([-0.4, -0.5, -0.6], dtype=np.float32),
        ]
    )
    export_arrays["observations"] = np.stack(
        [
            np.linspace(0.0, 1.0, len(OBSERVATION_FIELDS_V7), dtype=np.float32) + offset
            for offset in (0.0, 1.0, 2.0)
        ],
        axis=0,
    )

    dataset = build_dkana_windows(
        window_size=2,
        observation_fields=OBSERVATION_FIELDS_V7,
        **export_arrays,
    )

    expected_config_fields = build_dkana_config_fields(len(ACTION_FIELDS_TRACK_B_V1))
    assert dataset.config_context.shape == (3, 2, len(expected_config_fields))
    assert dataset.action_targets.shape == (3, len(ACTION_FIELDS_TRACK_B_V1))
    assert dataset.config_fields == expected_config_fields
    assert dataset.variable_names[: len(OBSERVATION_FIELDS_V7)] == tuple(
        f"obs::{field_name}" for field_name in OBSERVATION_FIELDS_V7
    )


def test_build_dkana_windows_temporal_delta_adds_relation_rows() -> None:
    export_arrays = make_synthetic_export_arrays()
    dataset = build_dkana_windows(
        window_size=2,
        relation_mode="temporal_delta",
        **export_arrays,
    )
    base_row_count = len(OBSERVATION_FIELDS_V3) + len(STATE_CONSTRAINT_FIELDS)

    assert dataset.relation_mode == "temporal_delta"
    assert dataset.row_matrices.shape == (3, 2, base_row_count * 2, 3)
    second_step_temporal_rows = dataset.row_matrices[1, 1, base_row_count:]
    assert set(second_step_temporal_rows[:, 1].tolist()) == {
        2.0,
    }


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


def test_dkana_policy_uses_last_true_timestep_with_left_padding() -> None:
    export_arrays = make_synthetic_export_arrays()
    dataset = build_dkana_windows(window_size=2, **export_arrays)
    model = DKANAPolicy(
        config_dim=dataset.config_context.shape[-1],
        action_dim=len(ACTION_FIELDS),
        max_sequence_length=2,
    )
    model.eval()

    captured: dict[str, torch.Tensor] = {}
    original_decoder_forward = model.decoder.forward

    def capture_decoder_input(inputs: torch.Tensor) -> torch.Tensor:
        captured["policy_embedding"] = inputs.detach().clone()
        return original_decoder_forward(inputs)

    model.decoder.forward = capture_decoder_input  # type: ignore[method-assign]
    with torch.no_grad():
        model(
            torch.from_numpy(dataset.row_matrices[:1]),
            torch.from_numpy(dataset.config_context[:1]),
            torch.from_numpy(dataset.time_mask[:1]),
        )

        batch_size, sequence_length, row_count, _ = dataset.row_matrices[:1].shape
        row_tensor = torch.from_numpy(dataset.row_matrices[:1])
        config_tensor = torch.from_numpy(dataset.config_context[:1])
        row_latent = model.row_encoder(row_tensor)
        config_latent = model.config_encoder(config_tensor).unsqueeze(2)
        local_tokens = torch.cat([row_latent, config_latent], dim=2)
        local_tokens = local_tokens.reshape(
            batch_size * sequence_length,
            row_count + 1,
            model.latent_dim,
        )
        local_tokens = model.local_position_encoding(local_tokens)
        local_mask = build_causal_attention_mask(
            local_tokens.shape[1], local_tokens.device
        )
        local_context = model.local_attention(local_tokens, mask=local_mask)
        state_embeddings = local_context[:, -1, :].reshape(
            batch_size,
            sequence_length,
            model.latent_dim,
        )
        state_embeddings = model.global_position_encoding(state_embeddings)
        global_mask = build_causal_attention_mask(
            sequence_length, state_embeddings.device
        )
        time_mask = torch.from_numpy(dataset.time_mask[:1])
        valid_query = time_mask.unsqueeze(2)
        valid_key = time_mask.unsqueeze(1)
        self_attention = torch.eye(
            sequence_length,
            dtype=torch.bool,
            device=state_embeddings.device,
        ).unsqueeze(0)
        combined_mask = global_mask.unsqueeze(0).expand(batch_size, -1, -1).clone()
        combined_mask |= ~valid_key
        combined_mask = torch.where(valid_query, combined_mask, ~self_attention)
        global_mask = combined_mask.repeat_interleave(model.num_heads, dim=0)
        global_context = model.global_attention(
            state_embeddings,
            mask=global_mask,
        )

    assert torch.allclose(captured["policy_embedding"], global_context[:, -1, :])


def test_dkana_online_adapter_returns_track_b_action() -> None:
    env = make_track_b_env(max_steps=1)
    obs, info = env.reset(seed=123)
    spec = get_track_b_env_spec()
    row_count = 2 * (len(spec.observation_fields) + len(STATE_CONSTRAINT_FIELDS))
    model = DKANAPolicy(
        config_dim=len(CONTROL_CONTEXT_FIELDS) + len(ACTION_FIELDS_TRACK_B_V1),
        action_dim=len(ACTION_FIELDS_TRACK_B_V1),
        latent_dim=32,
        num_heads=4,
        max_rows=row_count + 1,
        max_sequence_length=2,
    )
    adapter = DKANAOnlinePolicyAdapter(
        model,
        window_size=2,
        observation_fields=spec.observation_fields,
        state_constraint_fields=STATE_CONSTRAINT_FIELDS,
        action_dim=len(ACTION_FIELDS_TRACK_B_V1),
        relation_mode="temporal_delta",
    )

    action = adapter(obs, info)

    assert action.shape == (len(ACTION_FIELDS_TRACK_B_V1),)
    assert np.all(action >= -1.0)
    assert np.all(action <= 1.0)
    env.close()


def test_dkana_track_b_env_emits_context_window_in_info() -> None:
    env = make_dkana_track_b_env(max_steps=2, dkana_window_size=3)
    obs, info = env.reset(seed=123)
    row_count = 2 * (len(OBSERVATION_FIELDS_V7) + len(STATE_CONSTRAINT_FIELDS))
    config_dim = len(CONTROL_CONTEXT_FIELDS) + len(ACTION_FIELDS_TRACK_B_V1)

    assert obs.shape == (len(OBSERVATION_FIELDS_V7),)
    assert info["dkana_row_matrices"].shape == (3, row_count, 3)
    assert info["dkana_config_context"].shape == (3, config_dim)
    assert info["dkana_time_mask"].tolist() == [False, False, True]
    assert info["dkana_context"]["relation_to_index"] == {"=": 0, "<": 1, ">": 2}
    assert info["dkana_context"]["relation_mode"] == "temporal_delta"

    action = np.linspace(-1.0, 1.0, len(ACTION_FIELDS_TRACK_B_V1), dtype=np.float32)
    _, _, _, _, step_info = env.step(action)

    assert step_info["dkana_time_mask"].tolist() == [False, True, True]
    np.testing.assert_allclose(
        step_info["dkana_config_context"][-1, -len(ACTION_FIELDS_TRACK_B_V1) :],
        action,
        atol=1e-6,
    )
    env.close()


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
        json.dumps({"observation_fields": list(OBSERVATION_FIELDS_V3)}),
        encoding="utf-8",
    )
    (export_dir / "state_constraint_fields.json").write_text(
        json.dumps({"fields": list(STATE_CONSTRAINT_FIELDS)}),
        encoding="utf-8",
    )
    (export_dir / "constraint_context_fields.json").write_text(
        json.dumps({"fields": list(CONTROL_CONTEXT_FIELDS)}),
        encoding="utf-8",
    )
    (export_dir / "reward_terms_fields.json").write_text(
        json.dumps({"fields": list(REWARD_TERM_FIELDS), "formula": "unit_test"}),
        encoding="utf-8",
    )
    (export_dir / "action_fields.json").write_text(
        json.dumps({"fields": list(ACTION_FIELDS)}),
        encoding="utf-8",
    )
    (export_dir / "metadata.json").write_text(
        json.dumps(
            {
                "reward_mode": "ReT_unified_v1",
                "observation_version": "v3",
                "frame_stack": 1,
            }
        ),
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
        "--relation-mode",
        "temporal_delta",
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
        2 * (len(OBSERVATION_FIELDS_V3) + len(STATE_CONSTRAINT_FIELDS)),
        3,
    ]
    assert metadata["relation_mode"] == "temporal_delta"
    assert metadata["config_context_shape"] == [3, 2, len(DKANA_CONFIG_FIELDS)]
    assert metadata["reward_mode"] == "ReT_unified_v1"
    assert metadata["observation_version"] == "v3"
    assert metadata["action_fields"] == list(ACTION_FIELDS)
    assert metadata["reward_term_fields"] == list(REWARD_TERM_FIELDS)
