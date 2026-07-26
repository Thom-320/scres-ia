#!/usr/bin/env python3
"""Train and evaluate the frozen Q-R1 matched-retention RecurrentPPO design.

The primary comparison uses one frozen checkpoint twice.  The retained arm
carries its recurrent state through all twelve campaigns.  The reset arm zeros
that state at each physical campaign boundary.  Both arms see the same
observations, campaign skeletons, actions space, weights, and deterministic
evaluation procedure.

Confirmation roots are deliberately unsupported here.  They remain sealed
until development selection and the prospective power audit are complete.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_program_q_replication import scheduler  # noqa: E402
from supply_chain.q_r1_metaepisode_env import (  # noqa: E402
    CAMPAIGNS_PER_METAEPISODE,
    DECISIONS_PER_METAEPISODE,
    QRetainedMetaEpisodeEnv,
)
from supply_chain.retained_context_discovery import build_campaign_history  # noqa: E402


CONTRACT_PATH = ROOT / "contracts/q_r1_matched_retention_curve_v2.json"
RECEIPT_PATH = (
    ROOT / "contracts/q_r1_matched_retention_curve_v2_freeze_receipt.json"
)
REGIME_PERSISTENCE = 0.75
DOMINANT_SHARE = 0.90


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*command: str) -> str:
    return subprocess.check_output(
        ["git", *command], cwd=ROOT, text=True
    ).strip()


def _load_authority() -> tuple[dict[str, Any], dict[str, Any]]:
    contract = json.loads(CONTRACT_PATH.read_text())
    receipt = json.loads(RECEIPT_PATH.read_text())
    if contract["status"] != "FROZEN_PROSPECTIVE_UNOPENED":
        raise RuntimeError("matched-retention contract is not frozen and unopened")
    if receipt["contract_sha256"] != _sha256(CONTRACT_PATH):
        raise RuntimeError("contract bytes no longer match the freeze receipt")
    if receipt["confirmation_roots_opened"]:
        raise RuntimeError("freeze receipt unexpectedly marks confirmation roots opened")
    return contract, receipt


def _root_range(contract: dict[str, Any], split: str) -> list[int]:
    key = {
        "training": "training_history_roots",
        "selection": "checkpoint_selection_history_roots",
    }[split]
    lower, upper = contract["data_splits"][key]
    return list(range(int(lower), int(upper) + 1))


def build_histories(roots: list[int], kappa: float) -> tuple[tuple, ...]:
    sched = scheduler()
    return tuple(
        build_campaign_history(
            history_root=root,
            campaigns=CAMPAIGNS_PER_METAEPISODE,
            kappa=float(kappa),
            scheduler=sched,
            regime_persistence=REGIME_PERSISTENCE,
            dominant_share=DOMINANT_SHARE,
        )
        for root in roots
    )


def make_env(roots: list[int], kappa: float, sampling_seed: int) -> QRetainedMetaEpisodeEnv:
    return QRetainedMetaEpisodeEnv(
        histories=build_histories(roots, kappa),
        scheduler=scheduler(),
        regime_persistence=REGIME_PERSISTENCE,
        dominant_share=DOMINANT_SHARE,
        sampling_seed=sampling_seed,
    )


def evaluate_same_weights(
    model: Any,
    *,
    roots: list[int],
    kappa: float,
    reset_at_boundaries: bool,
) -> list[dict[str, Any]]:
    """Roll the same frozen model over a deterministic ordered history set."""
    env = make_env(roots, kappa, sampling_seed=0)
    rows: list[dict[str, Any]] = []
    arm = (
        "recurrent_ppo_reset_state_same_weights"
        if reset_at_boundaries
        else "recurrent_ppo_retained_state"
    )
    for history_index, root in enumerate(roots):
        observation, _ = env.reset(options={"history_index": history_index})
        state = None
        episode_start = np.asarray([True], dtype=bool)
        terminated = False
        decisions = 0
        crossed_boundaries = 0
        while not terminated:
            action, state = model.predict(
                observation,
                state=state,
                episode_start=episode_start,
                deterministic=True,
            )
            action_value = int(np.asarray(action).item())
            observation, reward, terminated, truncated, info = env.step(action_value)
            if truncated:
                raise RuntimeError("Q-R1 metaepisode may not truncate")
            decisions += 1
            if info.get("campaign_complete"):
                rows.append(
                    {
                        "history_root": int(root),
                        "kappa": float(kappa),
                        "campaign_index": int(info["campaign_index"]),
                        "arm": arm,
                        "early_ret_complete_cohort": float(reward),
                        "ret_visible": float(info["ret_visible"]),
                        "worst_product_fill": float(info["worst_product_fill"]),
                        "unresolved_orders": float(info["unresolved_orders"]),
                        "lost_orders": float(info["lost_orders"]),
                        "calendar": list(map(int, info["calendar"])),
                        "skeleton_sha256": info["skeleton_sha256"],
                        "prefix_state_hash": info["prefix_state_hash"],
                    }
                )
            boundary = bool(info.get("campaign_boundary"))
            if boundary:
                crossed_boundaries += 1
            if reset_at_boundaries and boundary:
                state = None
                episode_start = np.asarray([True], dtype=bool)
            else:
                episode_start = np.asarray([False], dtype=bool)
        if decisions != DECISIONS_PER_METAEPISODE:
            raise RuntimeError(f"history {root} produced {decisions} decisions")
        if crossed_boundaries != CAMPAIGNS_PER_METAEPISODE - 1:
            raise RuntimeError(
                f"history {root} crossed {crossed_boundaries} physical boundaries"
            )
    return rows


def _config(contract: dict[str, Any], config_id: str) -> dict[str, Any]:
    matches = [
        row
        for row in contract["training"]["screen_configurations"]
        if row["id"] == config_id
    ]
    if len(matches) != 1:
        raise ValueError(f"unknown or duplicate config id: {config_id}")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "screen"), required=True)
    parser.add_argument("--config-id", default="s01")
    parser.add_argument("--optimizer-seed", type=int, default=7_661_001)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite {args.output_dir}")

    contract, receipt = _load_authority()
    cfg = _config(contract, args.config_id)
    allowed_seeds = set(map(int, contract["data_splits"]["optimizer_seeds"]))
    if args.mode == "screen" and args.optimizer_seed not in allowed_seeds:
        raise SystemExit("screen optimizer seed is outside the frozen set")

    from sb3_contrib import RecurrentPPO  # noqa: PLC0415

    if args.mode == "smoke":
        roots = [7_570_801]
        kappa_values = [0.90]
        timesteps = int(args.timesteps or 192)
        claim_status = "BURNED_INSTRUMENT_SMOKE_NO_CLAIM"
    else:
        roots = _root_range(contract, "training")
        kappa_values = [0.50, 0.75, 0.90]
        timesteps = int(
            args.timesteps or contract["training"]["screen_timesteps_per_seed"]
        )
        claim_status = "DEVELOPMENT_NO_CONFIRMATORY_CLAIM"

    training_histories = tuple(
        history
        for kappa in kappa_values
        for history in build_histories(roots, kappa)
    )
    train_env = QRetainedMetaEpisodeEnv(
        histories=training_histories,
        scheduler=scheduler(),
        regime_persistence=REGIME_PERSISTENCE,
        dominant_share=DOMINANT_SHARE,
        sampling_seed=args.optimizer_seed,
    )
    policy_kwargs = {
        "lstm_hidden_size": int(cfg["hidden_size"]),
        "net_arch": {"pi": [64, 64], "vf": [64, 64]},
    }
    model = RecurrentPPO(
        "MlpLstmPolicy",
        train_env,
        seed=args.optimizer_seed,
        n_steps=int(contract["training"]["rollout_steps"]),
        batch_size=int(contract["training"]["batch_size"]),
        learning_rate=float(cfg["learning_rate"]),
        gamma=float(cfg["gamma"]),
        gae_lambda=float(cfg["gae_lambda"]),
        ent_coef=float(cfg["entropy"]),
        normalize_advantage=bool(contract["training"]["normalize_advantage"]),
        policy_kwargs=policy_kwargs,
        verbose=0,
    )
    started = time.perf_counter()
    model.learn(total_timesteps=timesteps, progress_bar=False)
    training_seconds = time.perf_counter() - started

    evaluation_roots = (
        [7_570_801]
        if args.mode == "smoke"
        else _root_range(contract, "selection")
    )
    rows: list[dict[str, Any]] = []
    for kappa in kappa_values:
        rows.extend(
            evaluate_same_weights(
                model,
                roots=evaluation_roots,
                kappa=kappa,
                reset_at_boundaries=False,
            )
        )
        rows.extend(
            evaluate_same_weights(
                model,
                roots=evaluation_roots,
                kappa=kappa,
                reset_at_boundaries=True,
            )
        )

    args.output_dir.mkdir(parents=True)
    checkpoint = args.output_dir / "model"
    model.save(checkpoint)
    checkpoint_path = checkpoint.with_suffix(".zip")
    (args.output_dir / "rows.json").write_text(
        json.dumps(rows, indent=1, sort_keys=True) + "\n"
    )
    payload = {
        "schema_version": "q_r1_matched_retention_curve_v2_run",
        "claim_status": claim_status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "contract_sha256": receipt["contract_sha256"],
        "execution_commit": _git("rev-parse", "HEAD"),
        "worktree_clean_before_output": not bool(_git("status", "--short")),
        "config": cfg,
        "optimizer_seed": int(args.optimizer_seed),
        "timesteps": timesteps,
        "training_roots": roots,
        "evaluation_roots": evaluation_roots,
        "kappa_values": kappa_values,
        "training_seconds": training_seconds,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "row_count": len(rows),
        "same_checkpoint_both_memory_arms": True,
        "confirmation_roots_opened": False,
        "kan_evaluated": False,
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(payload, indent=1, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
