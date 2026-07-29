#!/usr/bin/env python3
"""Program L campaign harness for persistent/reset/frozen PPO arms.

Large training is impossible unless a Gate-2 verdict explicitly promotes PPO.
``--allow-smoke-without-gate`` exists only for tiny pipeline tests and is recorded
in every manifest.  This runner uses fixed calibration normalization and never
opens the virgin confirmatory universe.
"""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import io
import json
from pathlib import Path
import sys
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_l_static_panel import build_calibration_tapes, observable_heuristic  # noqa: E402
from supply_chain.l_program_env import (  # noqa: E402
    BUFFER_LEVELS,
    CampaignTape,
    FixedNormalizerStats,
    GarridoLearningEnv,
    RewardScales,
    materialize_campaign_tape,
)


ARMS = (
    "frozen",
    "persistent_weights",
    "reset_local",
    "persistent_full",
    "scratch_matched_onpolicy",
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def generate_tapes(
    *, split: str, count: int, seed_base: int, horizon_weeks: int
) -> list[CampaignTape]:
    families = ("R1", "R2", "mixed")
    levels = ("current", "increased")
    return [
        CampaignTape(
            campaign_id=f"l-{split}-{index:03d}-{seed_base + index}",
            family=families[index % 3],
            risk_level=levels[(index // 3) % 2],
            base_seed=seed_base + index,
            horizon_weeks=horizon_weeks,
            split=split,
        )
        for index in range(count)
    ]


class CampaignSequenceEnv(gym.Wrapper):
    """Cycle through a frozen tape list whenever SB3 requests a reset."""

    def __init__(
        self,
        *,
        tapes: list[CampaignTape],
        buffer_level: int,
        normalizer: FixedNormalizerStats,
        reward_scales: RewardScales,
        lambda_shift: float,
    ) -> None:
        if not tapes:
            raise ValueError("CampaignSequenceEnv requires at least one tape.")
        horizon = {tape.horizon_weeks for tape in tapes}
        if len(horizon) != 1:
            raise ValueError("All sequence tapes must share one horizon.")
        self.tapes = list(tapes)
        self.buffer_level = int(buffer_level)
        self._index = 0
        base = GarridoLearningEnv(
            max_steps=next(iter(horizon)),
            buffer_level=buffer_level,
            normalizer=normalizer,
            reward_scales=reward_scales,
            lambda_shift=lambda_shift,
        )
        super().__init__(base)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        tape = self.tapes[self._index % len(self.tapes)]
        self._index += 1
        return self.env.reset(
            seed=tape.base_seed,
            options={
                "campaign_tape": tape,
                "buffer_level": self.buffer_level,
                "initial_state_seed": tape.base_seed,
                "initial_shift": 1,
            },
        )


def batch_size_for(n_steps: int) -> int:
    for candidate in (64, 32, 16, 8, 4, 2):
        if candidate <= n_steps and n_steps % candidate == 0:
            return candidate
    return n_steps


def make_sequence_env(
    tapes: list[CampaignTape],
    buffer_level: int,
    normalizer: FixedNormalizerStats,
    reward_scales: RewardScales,
    lambda_shift: float,
) -> CampaignSequenceEnv:
    return CampaignSequenceEnv(
        tapes=tapes,
        buffer_level=buffer_level,
        normalizer=normalizer,
        reward_scales=reward_scales,
        lambda_shift=lambda_shift,
    )


def build_model(
    *,
    tapes: list[CampaignTape],
    buffer_level: int,
    normalizer: FixedNormalizerStats,
    reward_scales: RewardScales,
    lambda_shift: float,
    learner_seed: int,
) -> PPO:
    n_steps = tapes[0].horizon_weeks
    vec = DummyVecEnv(
        [
            lambda: make_sequence_env(
                tapes, buffer_level, normalizer, reward_scales, lambda_shift
            )
        ]
    )
    return PPO(
        "MlpPolicy",
        vec,
        learning_rate=3e-4,
        n_steps=n_steps,
        batch_size=batch_size_for(n_steps),
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.98,
        clip_range=0.2,
        ent_coef=0.0,
        policy_kwargs={
            "net_arch": {"pi": [64, 64], "vf": [64, 64]},
            "activation_fn": torch.nn.Tanh,
        },
        seed=learner_seed,
        verbose=0,
        device="cpu",
    )


def load_model(
    path: Path,
    *,
    tapes: list[CampaignTape],
    buffer_level: int,
    normalizer: FixedNormalizerStats,
    reward_scales: RewardScales,
    lambda_shift: float,
) -> PPO:
    vec = DummyVecEnv(
        [
            lambda: make_sequence_env(
                tapes, buffer_level, normalizer, reward_scales, lambda_shift
            )
        ]
    )
    return PPO.load(path, env=vec, device="cpu")


def reset_optimizer_state(model: PPO) -> None:
    """Remove Adam moments while preserving actor/critic parameters and groups."""
    model.policy.optimizer.state.clear()


def torch_state_digest(state: Any) -> str:
    buffer = io.BytesIO()
    torch.save(state, buffer)
    return sha256(buffer.getvalue()).hexdigest()


def checkpoint_manifest(
    model: PPO,
    *,
    path: Path,
    arm: str,
    buffer_level: int,
    learner_seed: int,
    cycle: int,
    parent_sha256: str | None,
) -> dict[str, Any]:
    model.save(path)
    model_path = path if path.suffix == ".zip" else path.with_suffix(".zip")
    file_sha = sha256(model_path.read_bytes()).hexdigest()
    return {
        "arm": arm,
        "buffer_level": buffer_level,
        "learner_seed": learner_seed,
        "cycle": cycle,
        "path": str(model_path),
        "file_sha256": file_sha,
        "parent_sha256": parent_sha256,
        "policy_state_sha256": torch_state_digest(model.policy.state_dict()),
        "optimizer_state_sha256": torch_state_digest(
            model.policy.optimizer.state_dict()
        ),
    }


def train_on_tape(
    model: PPO,
    *,
    tape: CampaignTape,
    buffer_level: int,
    normalizer: FixedNormalizerStats,
    reward_scales: RewardScales,
    lambda_shift: float,
) -> None:
    env = DummyVecEnv(
        [
            lambda: make_sequence_env(
                [tape], buffer_level, normalizer, reward_scales, lambda_shift
            )
        ]
    )
    model.set_env(env)
    model.learn(
        total_timesteps=tape.horizon_weeks,
        reset_num_timesteps=False,
        progress_bar=False,
    )
    env.close()


def evaluate_model(
    model: PPO,
    *,
    arm: str,
    buffer_level: int,
    learner_seed: int,
    cycle: int,
    probes: list[CampaignTape],
    normalizer: FixedNormalizerStats,
    reward_scales: RewardScales,
    lambda_shift: float,
) -> list[dict[str, Any]]:
    rows = []
    for tape in probes:
        env = GarridoLearningEnv(
            max_steps=tape.horizon_weeks,
            buffer_level=buffer_level,
            normalizer=normalizer,
            reward_scales=reward_scales,
            lambda_shift=lambda_shift,
        )
        try:
            obs, _info = env.reset(
                seed=tape.base_seed,
                options={
                    "campaign_tape": tape,
                    "buffer_level": buffer_level,
                    "initial_state_seed": tape.base_seed,
                    "initial_shift": 1,
                },
            )
            term = trunc = False
            while not (term or trunc):
                action, _state = model.predict(obs, deterministic=True)
                obs, _reward, term, trunc, _info = env.step(int(action))
            metrics = env.terminal_metrics()
            if "ret_excel" not in metrics:
                raise RuntimeError("Program L evaluation requires ret_excel.")
            rows.append(
                {
                    "arm": arm,
                    "buffer_level": buffer_level,
                    "learner_seed": learner_seed,
                    "cycle": cycle,
                    "probe_id": tape.campaign_id,
                    "probe_sha256": tape.digest(),
                    "probe_family": tape.family,
                    "probe_risk_level": tape.risk_level,
                    **metrics,
                }
            )
        finally:
            env.close()
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate1-dir", type=Path, required=True)
    parser.add_argument("--gate2-verdict", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--buffers", default=",".join(map(str, BUFFER_LEVELS)))
    parser.add_argument("--learner-seeds", default="1,2,3,4,5")
    parser.add_argument("--cycles", type=int, default=8)
    parser.add_argument("--fixed-probes", type=int, default=60)
    parser.add_argument("--pretrain-timesteps", type=int, default=5000)
    parser.add_argument("--lambda-shift", type=float, default=0.25)
    parser.add_argument("--allow-smoke-without-gate", action="store_true")
    args = parser.parse_args()

    gate2 = json.loads(args.gate2_verdict.read_text(encoding="utf-8"))
    promoted = bool(gate2.get("promoted_to_ppo", False))
    if not promoted and not args.allow_smoke_without_gate:
        raise SystemExit("Gate 2 did not promote PPO; training is blocked.")
    if args.allow_smoke_without_gate and (args.cycles > 2 or args.fixed_probes > 3):
        raise ValueError("Smoke override is limited to <=2 cycles and <=3 probes.")

    manifest = json.loads((args.gate1_dir / "manifest.json").read_text())
    calibration_tapes = [CampaignTape.from_mapping(row) for row in manifest["tapes"]]
    horizon = int(manifest["horizon_weeks"])
    normalizer = FixedNormalizerStats.from_mapping(
        json.loads((args.gate1_dir / "fixed_normalizer.json").read_text())
    )
    reward_scales = RewardScales.from_mapping(
        json.loads((args.gate1_dir / "reward_scales.json").read_text())
    )
    training_tapes = [
        materialize_campaign_tape(tape)
        for tape in generate_tapes(
            split="training",
            count=args.cycles,
            seed_base=710_000,
            horizon_weeks=horizon,
        )
    ]
    probes = [
        materialize_campaign_tape(tape)
        for tape in generate_tapes(
            split="fixed_probe",
            count=args.fixed_probes,
            seed_base=720_000,
            horizon_weeks=horizon,
        )
    ]
    buffers = [int(value) for value in args.buffers.split(",") if value]
    learner_seeds = [int(value) for value in args.learner_seeds.split(",") if value]
    if any(value not in BUFFER_LEVELS for value in buffers):
        raise ValueError(f"Unknown buffer in {buffers}.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    for buffer_level in buffers:
        for learner_seed in learner_seeds:
            run_dir = args.output_dir / f"I{buffer_level}" / f"seed{learner_seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            theta0_path = run_dir / "theta0.zip"
            theta0 = build_model(
                tapes=calibration_tapes,
                buffer_level=buffer_level,
                normalizer=normalizer,
                reward_scales=reward_scales,
                lambda_shift=args.lambda_shift,
                learner_seed=learner_seed,
            )
            if args.pretrain_timesteps > 0:
                theta0.learn(
                    total_timesteps=args.pretrain_timesteps,
                    reset_num_timesteps=False,
                    progress_bar=False,
                )
            theta0.save(theta0_path)
            theta0.get_env().close()

            models = {
                arm: load_model(
                    theta0_path,
                    tapes=training_tapes,
                    buffer_level=buffer_level,
                    normalizer=normalizer,
                    reward_scales=reward_scales,
                    lambda_shift=args.lambda_shift,
                )
                for arm in ("frozen", "persistent_weights", "persistent_full")
            }
            parent_hash: dict[str, str | None] = {arm: None for arm in ARMS}
            for cycle_index, tape in enumerate(training_tapes, start=1):
                reset_model = load_model(
                    theta0_path,
                    tapes=[tape],
                    buffer_level=buffer_level,
                    normalizer=normalizer,
                    reward_scales=reward_scales,
                    lambda_shift=args.lambda_shift,
                )
                reset_optimizer_state(reset_model)
                train_on_tape(
                    reset_model,
                    tape=tape,
                    buffer_level=buffer_level,
                    normalizer=normalizer,
                    reward_scales=reward_scales,
                    lambda_shift=args.lambda_shift,
                )

                reset_optimizer_state(models["persistent_weights"])
                train_on_tape(
                    models["persistent_weights"],
                    tape=tape,
                    buffer_level=buffer_level,
                    normalizer=normalizer,
                    reward_scales=reward_scales,
                    lambda_shift=args.lambda_shift,
                )
                train_on_tape(
                    models["persistent_full"],
                    tape=tape,
                    buffer_level=buffer_level,
                    normalizer=normalizer,
                    reward_scales=reward_scales,
                    lambda_shift=args.lambda_shift,
                )
                cycle_models = {**models, "reset_local": reset_model}
                for arm, model in cycle_models.items():
                    ckpt_path = run_dir / f"{arm}_cycle{cycle_index}.zip"
                    ckpt = checkpoint_manifest(
                        model,
                        path=ckpt_path,
                        arm=arm,
                        buffer_level=buffer_level,
                        learner_seed=learner_seed,
                        cycle=cycle_index,
                        parent_sha256=parent_hash[arm],
                    )
                    parent_hash[arm] = ckpt["file_sha256"]
                    checkpoints.append(ckpt)
                    all_rows.extend(
                        evaluate_model(
                            model,
                            arm=arm,
                            buffer_level=buffer_level,
                            learner_seed=learner_seed,
                            cycle=cycle_index,
                            probes=probes,
                            normalizer=normalizer,
                            reward_scales=reward_scales,
                            lambda_shift=args.lambda_shift,
                        )
                    )
                reset_model.get_env().close()

            # Matched-compute control: start again from theta0 only after the
            # sequential arms are complete, traverse the identical campaign
            # multiset in a canonical balanced order, and consume exactly one
            # on-policy campaign update per tape. No historical rollout is reused.
            scratch = load_model(
                theta0_path,
                tapes=training_tapes,
                buffer_level=buffer_level,
                normalizer=normalizer,
                reward_scales=reward_scales,
                lambda_shift=args.lambda_shift,
            )
            balanced_order = sorted(
                training_tapes,
                key=lambda tape: (tape.family, tape.risk_level, tape.campaign_id),
            )
            for tape in balanced_order:
                reset_optimizer_state(scratch)
                train_on_tape(
                    scratch,
                    tape=tape,
                    buffer_level=buffer_level,
                    normalizer=normalizer,
                    reward_scales=reward_scales,
                    lambda_shift=args.lambda_shift,
                )
            scratch_ckpt = checkpoint_manifest(
                scratch,
                path=run_dir / f"scratch_matched_onpolicy_cycle{args.cycles}.zip",
                arm="scratch_matched_onpolicy",
                buffer_level=buffer_level,
                learner_seed=learner_seed,
                cycle=args.cycles,
                parent_sha256=None,
            )
            checkpoints.append(scratch_ckpt)
            all_rows.extend(
                evaluate_model(
                    scratch,
                    arm="scratch_matched_onpolicy",
                    buffer_level=buffer_level,
                    learner_seed=learner_seed,
                    cycle=args.cycles,
                    probes=probes,
                    normalizer=normalizer,
                    reward_scales=reward_scales,
                    lambda_shift=args.lambda_shift,
                )
            )
            scratch.get_env().close()
            for model in models.values():
                model.get_env().close()

    write_csv(args.output_dir / "probe_rows.csv", all_rows)
    (args.output_dir / "checkpoints.json").write_text(
        json.dumps(checkpoints, indent=2, sort_keys=True), encoding="utf-8"
    )
    output_manifest = {
        "kind": "l_program_campaign_cycles",
        "contract_id": "garrido_learning_v1",
        "gate2_promoted": promoted,
        "smoke_override": bool(args.allow_smoke_without_gate),
        "buffers": buffers,
        "learner_seeds": learner_seeds,
        "cycles": args.cycles,
        "fixed_probes": args.fixed_probes,
        "pretrain_timesteps": args.pretrain_timesteps,
        "lambda_shift": args.lambda_shift,
        "normalizer": normalizer.payload(),
        "reward_scales": {
            "late_backlog_hours": reward_scales.late_backlog_hours,
            "total_backlog_hours": reward_scales.total_backlog_hours,
            "extra_shift_hours": reward_scales.extra_shift_hours,
        },
        "training_tapes": [row.payload(include_hash=True) for row in training_tapes],
        "fixed_probes_tapes": [row.payload(include_hash=True) for row in probes],
        "virgin_tapes_opened": False,
        "primary_metric": "ret_excel",
        "forbidden_metric_substitution": "order_level_ret_mean",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(output_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "rows": len(all_rows),
                "checkpoints": len(checkpoints),
                "virgin_tapes_opened": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
