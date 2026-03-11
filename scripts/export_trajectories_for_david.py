#!/usr/bin/env python3
"""
Export DES observation trajectories for external model training (e.g., DKANA).

Generates numpy arrays from the shift_control environment using random policy.
David can load these directly into his PyTorch pipeline.

Usage:
    python scripts/export_trajectories_for_david.py
    python scripts/export_trajectories_for_david.py --episodes 200 --output-dir data_for_dkana
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.external_env_interface import (
    get_shift_control_constraint_context,
    get_shift_control_env_spec,
    make_shift_control_env,
    spec_to_dict,
)

STATE_CONSTRAINT_FIELDS = (
    "raw_material_wdc",
    "raw_material_al",
    "rations_al",
    "rations_sb",
    "rations_sb_dispatch",
    "rations_cssu",
    "rations_theatre",
    "total_inventory",
    "op3_total_dispatch_cap",
    "op3_per_material_dispatch_cap",
    "op9_dispatch_cap",
    "assembly_line_available",
    "any_location_available",
    "op9_available",
    "op11_available",
    "fill_rate",
    "backorder_rate",
    "time_fraction",
    "pending_batch_fraction",
    "contingent_demand_fraction",
)

REWARD_TERM_FIELDS = (
    "reward_total",
    "service_loss_step",
    "shift_cost_step",
    "disruption_fraction_step",
    "ret_thesis_corrected_step",
)


def build_state_constraint_vector(state_context: dict[str, object]) -> np.ndarray:
    inventory_detail = state_context["inventory_detail"]
    assert isinstance(inventory_detail, dict)
    values = [
        float(inventory_detail["raw_material_wdc"]),
        float(inventory_detail["raw_material_al"]),
        float(inventory_detail["rations_al"]),
        float(inventory_detail["rations_sb"]),
        float(inventory_detail["rations_sb_dispatch"]),
        float(inventory_detail["rations_cssu"]),
        float(inventory_detail["rations_theatre"]),
        float(state_context["total_inventory"]),
        float(state_context["op3_total_dispatch_cap"]),
        float(state_context["op3_per_material_dispatch_cap"]),
        float(state_context["op9_dispatch_cap"]),
        float(bool(state_context["assembly_line_available"])),
        float(bool(state_context["any_location_available"])),
        float(bool(state_context["op9_available"])),
        float(bool(state_context["op11_available"])),
        float(state_context["fill_rate"]),
        float(state_context["backorder_rate"]),
        float(state_context["time_fraction"]),
        float(state_context["pending_batch_fraction"]),
        float(state_context["contingent_demand_fraction"]),
    ]
    return np.array(values, dtype=np.float32)


def build_reward_term_vector(info: dict[str, object], reward: float) -> np.ndarray:
    return np.array(
        [
            float(reward),
            float(info.get("service_loss_step", 0.0)),
            float(info.get("shift_cost_step", 0.0)),
            float(info.get("disruption_fraction_step", 0.0)),
            float(info.get("ret_thesis_corrected_step", 0.0)),
        ],
        dtype=np.float32,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export trajectories for external models."
    )
    parser.add_argument(
        "--episodes", type=int, default=100, help="Number of episodes to collect."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/data_export"))
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument(
        "--risk-level", default="current", choices=["current", "increased", "severe"]
    )
    parser.add_argument(
        "--reward-mode",
        default="control_v1",
        choices=["ReT_thesis", "control_v1"],
        help="Reward mode used while collecting trajectories for external models.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_obs = []
    all_actions = []
    all_rewards = []
    all_episode_ids = []
    all_constraint_context = []
    all_state_constraint_context = []
    all_reward_terms = []
    episode_lengths = []
    constraint_context = get_shift_control_constraint_context()
    constraint_vector = np.array(
        [
            constraint_context["base_control_parameters"]["op3_q"],
            constraint_context["base_control_parameters"]["op3_rop"],
            constraint_context["base_control_parameters"]["op9_q_min"],
            constraint_context["base_control_parameters"]["op9_q_max"],
            constraint_context["base_control_parameters"]["op9_rop"],
            constraint_context["inventory_multiplier_range"]["min"],
            constraint_context["inventory_multiplier_range"]["max"],
            -0.33,
            0.33,
        ],
        dtype=np.float32,
    )

    for ep in range(args.episodes):
        env = make_shift_control_env(
            risk_level=args.risk_level,
            reward_mode=args.reward_mode,
        )
        obs, _ = env.reset(seed=args.seed_start + ep)
        ep_obs, ep_actions, ep_rewards = [obs.copy()], [], []
        done, truncated = False, False

        while not (done or truncated):
            state_constraint_context = env.get_state_constraint_context()
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            ep_obs.append(obs.copy())
            ep_actions.append(action.copy())
            ep_rewards.append(reward)
            all_state_constraint_context.append(
                build_state_constraint_vector(state_constraint_context)
            )
            all_reward_terms.append(build_reward_term_vector(info, reward))

        T = len(ep_actions)
        episode_lengths.append(T)
        all_obs.extend(ep_obs[:T])  # align with actions
        all_actions.extend(ep_actions)
        all_rewards.extend(ep_rewards)
        all_episode_ids.extend([ep] * T)
        all_constraint_context.extend([constraint_vector.copy()] * T)

        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}/{args.episodes} done (T={T} steps)")

    # Save as numpy
    np.save(args.output_dir / "observations.npy", np.array(all_obs, dtype=np.float32))
    np.save(args.output_dir / "actions.npy", np.array(all_actions, dtype=np.float32))
    np.save(args.output_dir / "rewards.npy", np.array(all_rewards, dtype=np.float32))
    np.save(
        args.output_dir / "episode_ids.npy", np.array(all_episode_ids, dtype=np.int32)
    )
    np.save(
        args.output_dir / "constraint_context.npy",
        np.array(all_constraint_context, dtype=np.float32),
    )
    np.save(
        args.output_dir / "state_constraint_context.npy",
        np.array(all_state_constraint_context, dtype=np.float32),
    )
    np.save(
        args.output_dir / "reward_terms.npy",
        np.array(all_reward_terms, dtype=np.float32),
    )

    # Save env spec as JSON
    spec = get_shift_control_env_spec()
    with (args.output_dir / "env_spec.json").open("w") as f:
        json.dump(spec_to_dict(spec), f, indent=2)
    with (args.output_dir / "constraint_context.json").open("w") as f:
        json.dump(constraint_context, f, indent=2)
    with (args.output_dir / "state_constraint_fields.json").open("w") as f:
        json.dump({"fields": list(STATE_CONSTRAINT_FIELDS)}, f, indent=2)
    with (args.output_dir / "reward_terms_fields.json").open("w") as f:
        json.dump(
            {
                "reward_mode": args.reward_mode,
                "fields": list(REWARD_TERM_FIELDS),
                "formula": (
                    "reward_total = -(w_bo * service_loss_step + "
                    "w_cost * shift_cost_step + w_disr * disruption_fraction_step)"
                    if args.reward_mode == "control_v1"
                    else "See env_experimental_shifts.py for ReT_thesis reward details."
                ),
            },
            f,
            indent=2,
        )

    # Save metadata
    meta = {
        "episodes": args.episodes,
        "total_steps": len(all_rewards),
        "episode_lengths": episode_lengths,
        "risk_level": args.risk_level,
        "reward_mode": args.reward_mode,
        "obs_shape": [len(all_obs), 15],
        "action_shape": [len(all_actions), 5],
        "constraint_context_shape": [
            len(all_constraint_context),
            len(constraint_vector),
        ],
        "state_constraint_context_shape": [
            len(all_state_constraint_context),
            len(STATE_CONSTRAINT_FIELDS),
        ],
        "reward_terms_shape": [len(all_reward_terms), len(REWARD_TERM_FIELDS)],
        "policy": "random",
        "note": (
            "Collected with random policy. Use for offline training or as baseline data. "
            "constraint_context contains explicit action constraints and base control parameters "
            "that are enforced by the environment but not encoded inside obs_t. "
            "state_constraint_context contains per-step state-dependent feasibility signals. "
            "reward_terms contains the step reward decomposition requested for external models."
        ),
    }
    with (args.output_dir / "metadata.json").open("w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nExported {len(all_rewards):,} steps from {args.episodes} episodes")
    print(f"  observations.npy  shape=({len(all_obs)}, 15)")
    print(f"  actions.npy       shape=({len(all_actions)}, 5)")
    print(f"  rewards.npy       shape=({len(all_rewards)},)")
    print(f"  episode_ids.npy   shape=({len(all_episode_ids)},)")
    print(
        f"  constraint_context.npy shape=({len(all_constraint_context)}, {len(constraint_vector)})"
    )
    print(
        "  state_constraint_context.npy "
        f"shape=({len(all_state_constraint_context)}, {len(STATE_CONSTRAINT_FIELDS)})"
    )
    print(
        f"  reward_terms.npy   shape=({len(all_reward_terms)}, {len(REWARD_TERM_FIELDS)})"
    )
    print("  env_spec.json     (environment contract)")
    print("  constraint_context.json (explicit constraint metadata)")
    print("  state_constraint_fields.json (per-step state-constraint schema)")
    print("  reward_terms_fields.json (reward decomposition schema)")
    print("  metadata.json     (collection metadata)")
    print(f"\nSaved to: {args.output_dir}")


if __name__ == "__main__":
    main()
