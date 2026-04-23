#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.dkana import build_dkana_windows


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def load_json_if_exists(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        return load_json(path)
    return default


def resolve_output_dir(
    input_dir: Path, output_dir: Path | None, window_size: int
) -> Path:
    if output_dir is not None:
        return output_dir
    return input_dir / f"dkana_seq_w{window_size}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build fixed-length DKANA windows from exported MFSC trajectories."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/data_export"),
        help="Directory produced by export_trajectories_for_david.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination directory for DKANA-ready numpy arrays.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=12,
        help="Number of historical steps per DKANA sequence window.",
    )
    parser.add_argument(
        "--relation-mode",
        choices=["equality", "temporal_delta"],
        default="equality",
        help=(
            "MRC relation encoding. 'equality' emits x = value rows. "
            "'temporal_delta' also emits x <, =, or > previous_value rows."
        ),
    )
    args = parser.parse_args()

    output_dir = resolve_output_dir(args.input_dir, args.output_dir, args.window_size)
    output_dir.mkdir(parents=True, exist_ok=True)

    observations = np.load(args.input_dir / "observations.npy")
    actions = np.load(args.input_dir / "actions.npy")
    episode_ids = np.load(args.input_dir / "episode_ids.npy")
    constraint_context = np.load(args.input_dir / "constraint_context.npy")
    state_constraint_context = np.load(args.input_dir / "state_constraint_context.npy")
    rewards_path = args.input_dir / "rewards.npy"
    rewards = np.load(rewards_path) if rewards_path.exists() else None

    env_spec = load_json(args.input_dir / "env_spec.json")
    metadata_payload = load_json_if_exists(args.input_dir / "metadata.json", {})
    constraint_field_payload = load_json_if_exists(
        args.input_dir / "constraint_context_fields.json",
        {"fields": []},
    )
    state_field_payload = load_json_if_exists(
        args.input_dir / "state_constraint_fields.json",
        {"fields": []},
    )
    reward_field_payload = load_json_if_exists(
        args.input_dir / "reward_terms_fields.json",
        {"fields": [], "formula": None},
    )
    action_field_payload = load_json_if_exists(
        args.input_dir / "action_fields.json",
        {"fields": []},
    )
    direct_action_field_payload = load_json_if_exists(
        args.input_dir / "direct_action_context_fields.json",
        {"fields": []},
    )
    dataset = build_dkana_windows(
        observations=observations,
        actions=actions,
        episode_ids=episode_ids,
        constraint_context=constraint_context,
        state_constraint_context=state_constraint_context,
        rewards=rewards,
        window_size=args.window_size,
        observation_fields=tuple(env_spec["observation_fields"]),
        state_constraint_fields=tuple(state_field_payload["fields"]),
        relation_mode=args.relation_mode,
    )

    np.save(output_dir / "dkana_row_matrices.npy", dataset.row_matrices)
    np.save(output_dir / "dkana_config_context.npy", dataset.config_context)
    np.save(output_dir / "dkana_action_targets.npy", dataset.action_targets)
    np.save(output_dir / "dkana_time_mask.npy", dataset.time_mask)
    if dataset.reward_targets is not None:
        np.save(output_dir / "dkana_reward_targets.npy", dataset.reward_targets)

    metadata = {
        "source_dir": str(args.input_dir),
        "window_size": args.window_size,
        "relation_mode": dataset.relation_mode,
        "reward_mode": metadata_payload.get("reward_mode"),
        "observation_version": metadata_payload.get("observation_version"),
        "frame_stack": metadata_payload.get("frame_stack", 1),
        "row_matrices_shape": list(dataset.row_matrices.shape),
        "config_context_shape": list(dataset.config_context.shape),
        "action_targets_shape": list(dataset.action_targets.shape),
        "time_mask_shape": list(dataset.time_mask.shape),
        "reward_targets_shape": (
            list(dataset.reward_targets.shape)
            if dataset.reward_targets is not None
            else None
        ),
        "variable_names": list(dataset.variable_names),
        "config_fields": list(dataset.config_fields),
        "action_fields": list(action_field_payload["fields"]),
        "constraint_context_fields": list(constraint_field_payload["fields"]),
        "state_constraint_fields": list(state_field_payload["fields"]),
        "reward_term_fields": list(reward_field_payload["fields"]),
        "reward_formula": reward_field_payload.get("formula"),
        "direct_action_context_fields": list(direct_action_field_payload["fields"]),
        "relation_to_index": dataset.relation_to_index,
        "env_spec": env_spec,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as file_obj:
        json.dump(metadata, file_obj, indent=2)

    print(f"Wrote DKANA dataset to {output_dir}")
    print(f"  dkana_row_matrices.npy  shape={dataset.row_matrices.shape}")
    print(f"  dkana_config_context.npy shape={dataset.config_context.shape}")
    print(f"  dkana_action_targets.npy shape={dataset.action_targets.shape}")
    print(f"  dkana_time_mask.npy shape={dataset.time_mask.shape}")
    if dataset.reward_targets is not None:
        print(f"  dkana_reward_targets.npy shape={dataset.reward_targets.shape}")


if __name__ == "__main__":
    main()
