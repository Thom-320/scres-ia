#!/usr/bin/env python3
"""Run the auditable Q-R1 matched-retention factorial v4.

The only mode authorized while the contract remains a draft is
``instrument-preflight``. It uses the burned root declared by the contract and
emits every arm, ledger field, and estimand without opening development.

Development modes are implemented but fail closed until the contract is frozen,
an immutable freeze receipt exists, and an opening receipt is written before
the first development root is materialized.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
from itertools import product
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_program_q_replication import scheduler  # noqa: E402
from scripts.run_q_r1_comparator_v2_frozen_pareto import load_freeze  # noqa: E402
from supply_chain.program_o_full_des_transducer import (  # noqa: E402
    simulate_full_des_frontier,
)
from supply_chain.q_r1_factorial_v4 import (  # noqa: E402
    campaign_as_structured_state,
    matched_prior_paths,
    structured_pair_rows,
)
from supply_chain.q_r1_metaepisode_env import (  # noqa: E402
    CAMPAIGNS_PER_METAEPISODE,
    DECISIONS_PER_METAEPISODE,
    QRetainedMetaEpisodeEnv,
)
from supply_chain.q_r1_retained_learning import evaluate_calendar  # noqa: E402
from supply_chain.retained_context_discovery import (  # noqa: E402
    CampaignSpec,
    build_campaign_history,
)


CONTRACT_PATH = ROOT / "contracts/q_r1_matched_retention_factorial_v4.DRAFT.json"
FREEZE_RECEIPT_PATH = (
    ROOT / "contracts/q_r1_matched_retention_factorial_v4_freeze_receipt.json"
)
COMPARATOR_FREEZE_PATH = ROOT / "contracts/q_r1_comparator_v2_frozen_c256_v1.json"
RHO = 0.90
SHARE = 0.90
KAPPAS = (0.50, 0.75, 0.90)
FACTORIAL_ARMS = (
    ("P0_H0", False, True),
    ("P1_H0", True, True),
    ("P0_H1", False, False),
    ("P1_H1", True, False),
)
PRIMARY = "early_ret_complete_cohort"
ALL_CALENDARS = np.asarray(tuple(product(range(4), repeat=8)), dtype=np.uint8)


class PredeclaredComputeBudgetExceeded(RuntimeError):
    """The frozen comparator exceeded its predeclared compute budget."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def runtime_receipt() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in ("numpy", "scipy", "simpy", "gymnasium", "stable-baselines3", "sb3-contrib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "execution_commit": git("rev-parse", "HEAD"),
        "git_status_porcelain": git("status", "--porcelain"),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "argv": list(sys.argv),
    }


def load_authority(mode: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
    contract = json.loads(CONTRACT_PATH.read_text())
    if mode == "instrument-preflight":
        if contract["status"] != "DRAFT_PROSPECTIVE_UNOPENED_NOT_AUTHORITY":
            raise RuntimeError("instrument preflight expects the reviewed draft")
        if contract["data_splits"]["opened"]:
            raise RuntimeError("draft unexpectedly marks development roots open")
        return contract, None
    if contract["status"] != "FROZEN_PROSPECTIVE_UNOPENED":
        raise RuntimeError("development is forbidden until the contract is frozen")
    if not FREEZE_RECEIPT_PATH.exists():
        raise RuntimeError("development requires a separate freeze receipt")
    receipt = json.loads(FREEZE_RECEIPT_PATH.read_text())
    if receipt.get("contract_sha256") != sha256(CONTRACT_PATH):
        raise RuntimeError("contract bytes do not match the freeze receipt")
    if receipt.get("status") != "FROZEN_PROSPECTIVE_UNOPENED":
        raise RuntimeError("freeze receipt does not authorize development")
    return contract, receipt


def integer_range(bounds: list[int]) -> list[int]:
    return list(range(int(bounds[0]), int(bounds[1]) + 1))


def build_histories(roots: list[int], kappas: tuple[float, ...]) -> tuple[tuple[CampaignSpec, ...], ...]:
    sched = scheduler()
    return tuple(
        build_campaign_history(
            history_root=root,
            campaigns=CAMPAIGNS_PER_METAEPISODE,
            kappa=kappa,
            scheduler=sched,
            regime_persistence=RHO,
            dominant_share=SHARE,
        )
        for kappa in kappas
        for root in roots
    )


def constant_paths(
    histories: tuple[tuple[CampaignSpec, ...], ...], value: float
) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for _ in history) for history in histories)


def make_env(
    histories: tuple[tuple[CampaignSpec, ...], ...],
    *,
    retained_prior: bool,
    sampling_seed: int,
) -> QRetainedMetaEpisodeEnv:
    paths = (
        matched_prior_paths(histories)
        if retained_prior
        else constant_paths(histories, 0.5)
    )
    return QRetainedMetaEpisodeEnv(
        histories=histories,
        scheduler=scheduler(),
        regime_persistence=RHO,
        dominant_share=SHARE,
        sampling_seed=sampling_seed,
        prior_paths=paths,
        expose_prior_feature=True,
    )


def evaluate_neural_arm(
    model: Any,
    *,
    histories: tuple[tuple[CampaignSpec, ...], ...],
    arm: str,
    retained_prior: bool,
    reset_hidden_at_boundaries: bool,
    checkpoint_sha256: str,
) -> list[dict[str, Any]]:
    env = make_env(histories, retained_prior=retained_prior, sampling_seed=0)
    rows: list[dict[str, Any]] = []
    for history_index, history in enumerate(histories):
        observation, _ = env.reset(options={"history_index": history_index})
        state = None
        episode_start = np.asarray([True], dtype=bool)
        terminated = False
        decisions = 0
        while not terminated:
            action, state = model.predict(
                observation,
                state=state,
                episode_start=episode_start,
                deterministic=True,
            )
            observation, reward, terminated, truncated, info = env.step(
                int(np.asarray(action).item())
            )
            if truncated:
                raise RuntimeError("factorial metaepisode may not truncate")
            decisions += 1
            if info.get("campaign_complete"):
                rows.append(
                    {
                        "history_root": int(history[0].history_root),
                        "campaign_index": int(info["campaign_index"]),
                        "kappa": float(history[0].kappa),
                        "arm": arm,
                        "checkpoint_sha256": checkpoint_sha256,
                        "explicit_prior": float(info["explicit_prior"]),
                        "calendar": list(map(int, info["calendar"])),
                        "skeleton_sha256": info["skeleton_sha256"],
                        "prefix_state_hash": info["prefix_state_hash"],
                        PRIMARY: float(reward),
                        "early_ret_visible": float(info["early_ret_visible"]),
                        "ret_visible": float(info["ret_visible"]),
                        "ret_full": float(info["ret_full"]),
                        "whole_campaign_ret": float(info["whole_campaign_ret"]),
                        "worst_product_fill": float(info["worst_product_fill"]),
                        "unresolved_orders": float(info["unresolved_orders"]),
                        "unresolved_quantity": float(info["unresolved_quantity"]),
                        "lost_orders": float(info["lost_orders"]),
                        "lost_quantity": float(info["lost_quantity"]),
                        "service_loss": float(info["service_loss"]),
                        "gross_policy_batch_slots": float(
                            info["gross_policy_batch_slots"]
                        ),
                        "gross_production_quantity": float(
                            info["gross_production_quantity"]
                        ),
                        "charged_daily_dispatch_slots": float(
                            info["charged_daily_dispatch_slots"]
                        ),
                        "charged_downstream_vehicle_hours": float(
                            info["charged_downstream_vehicle_hours"]
                        ),
                    }
                )
            boundary = bool(info.get("campaign_boundary"))
            if reset_hidden_at_boundaries and boundary:
                state = None
                episode_start = np.asarray([True], dtype=bool)
            else:
                episode_start = np.asarray([False], dtype=bool)
        if decisions != DECISIONS_PER_METAEPISODE:
            raise RuntimeError("factorial arm emitted the wrong decision count")
    return rows


def filter_histories(
    histories: tuple[tuple[CampaignSpec, ...], ...],
    *,
    roots: set[int],
    campaign_indices: set[int] | None = None,
) -> tuple[tuple[CampaignSpec, ...], ...]:
    selected: list[tuple[CampaignSpec, ...]] = []
    for history in histories:
        if int(history[0].history_root) not in roots:
            continue
        if campaign_indices is None:
            selected.append(history)
        else:
            selected.append(
                tuple(
                    campaign
                    for campaign in history
                    if int(campaign.campaign_index) in campaign_indices
                )
            )
    return tuple(selected)


def build_static_bar(
    histories: tuple[tuple[CampaignSpec, ...], ...],
    *,
    campaign_indices: set[int] | None,
) -> dict[str, Any]:
    sched = scheduler()
    cache: dict[str, np.ndarray] = {}
    labels: list[np.ndarray] = []
    identities: list[dict[str, Any]] = []
    for history in histories:
        for campaign in history:
            if campaign_indices is not None and campaign.campaign_index not in campaign_indices:
                continue
            digest = campaign.skeleton.skeleton_sha256
            if digest not in cache:
                panel = simulate_full_des_frontier(
                    skeleton=campaign.skeleton,
                    scheduler=sched,
                    calendars=ALL_CALENDARS,
                    include_q_r1_metrics=True,
                )
                cache[digest] = np.asarray(panel[PRIMARY], dtype=float)
            labels.append(cache[digest])
            identities.append(
                {
                    "history_root": campaign.history_root,
                    "campaign_index": campaign.campaign_index,
                    "kappa": campaign.kappa,
                    "skeleton_sha256": digest,
                }
            )
    stacked = np.vstack(labels)
    index = int(stacked.mean(axis=0).argmax())
    return {
        "schema_version": "q_r1_factorial_v4_static_bar",
        "calendar": ALL_CALENDARS[index].astype(int).tolist(),
        "frontier_row": index,
        "selection_campaigns": len(labels),
        "unique_skeletons": len(cache),
        "selected_before_any_arm_grading": True,
        "identities": identities,
    }


def static_rows(
    histories: tuple[tuple[CampaignSpec, ...], ...],
    *,
    calendar: list[int],
) -> list[dict[str, Any]]:
    sched = scheduler()
    rows: list[dict[str, Any]] = []
    for history in histories:
        for campaign in history:
            physical = campaign_as_structured_state(campaign)
            metrics = evaluate_calendar(
                campaign=physical,
                calendar=calendar,
                scheduler=sched,
            )
            rows.append(
                {
                    "history_root": campaign.history_root,
                    "campaign_index": campaign.campaign_index,
                    "kappa": campaign.kappa,
                    "arm": "best_static_frozen",
                    "calendar": list(calendar),
                    "skeleton_sha256": campaign.skeleton.skeleton_sha256,
                    "prefix_state_hash": campaign.skeleton.prefix_state_hash,
                    **metrics,
                }
            )
    return rows


def evaluate_structured(
    histories: tuple[tuple[CampaignSpec, ...], ...],
    *,
    campaign_indices: set[int],
    contract: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _freeze, config = load_freeze(COMPARATOR_FREEZE_PATH)
    selected = filter_histories(
        histories,
        roots={int(history[0].history_root) for history in histories},
        campaign_indices=campaign_indices,
    )
    paths = matched_prior_paths(histories)
    path_index = {
        (int(history[0].history_root), float(history[0].kappa)): path
        for history, path in zip(histories, paths, strict=True)
    }
    selected_paths = tuple(
        tuple(
            path_index[(int(history[0].history_root), float(history[0].kappa))][
                campaign.campaign_index
            ]
            for campaign in history
        )
        for history in selected
    )
    cache: dict[
        tuple[str, str, str],
        tuple[tuple[int, ...], dict[str, object], dict[str, Any]],
    ] = {}
    started = time.perf_counter()
    rows = structured_pair_rows(
        histories=selected,
        prior_paths=selected_paths,
        scheduler=scheduler(),
        config=config,
        cache=cache,
    )
    elapsed = time.perf_counter() - started
    budget = contract["structured_comparators"]["compute_budget"]
    if elapsed > float(budget["development_sequential_hard_cap_seconds"]):
        raise PredeclaredComputeBudgetExceeded(
            "STOP_COMPUTE_BUDGET_PREDECLARED"
        )
    for row in rows:
        row["service_loss"] = row["service_loss_auc"]
        row["whole_campaign_ret"] = row["ret_visible"]
    return rows, {
        "elapsed_seconds": elapsed,
        "cache_entries": len(cache),
        "rows": len(rows),
        "config_id": config.config_id,
        "cache_reuse": len(rows) - len(cache),
    }


def indexed_mean_delta(
    rows: list[dict[str, Any]], left: str, right: str
) -> dict[str, Any]:
    indexed = {
        (
            str(row["arm"]),
            int(row["history_root"]),
            float(row["kappa"]),
            int(row["campaign_index"]),
        ): float(row[PRIMARY])
        for row in rows
    }
    keys = sorted(
        {
            (root, kappa, campaign)
            for arm, root, kappa, campaign in indexed
            if arm == left
        }
        & {
            (root, kappa, campaign)
            for arm, root, kappa, campaign in indexed
            if arm == right
        }
    )
    values = np.asarray(
        [
            indexed[(left, *key)] - indexed[(right, *key)]
            for key in keys
        ],
        dtype=float,
    )
    return {
        "left": left,
        "right": right,
        "paired_rows": len(values),
        "mean": float(values.mean()) if len(values) else None,
        "values": values.tolist(),
    }


def estimands(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output = {
        "explicit_context_value": indexed_mean_delta(rows, "P1_H0", "P0_H0"),
        "recurrent_residual_given_context": indexed_mean_delta(
            rows, "P1_H1", "P1_H0"
        ),
        "raw_recurrent_memory_value": indexed_mean_delta(rows, "P0_H1", "P0_H0"),
        "total_retained_neural_treatment": indexed_mean_delta(
            rows, "P1_H1", "P0_H0"
        ),
        "structured_retained_value": indexed_mean_delta(
            rows, "structured_retained", "structured_reset"
        ),
        "neural_premium": indexed_mean_delta(
            rows, "P1_H1", "structured_retained"
        ),
    }
    recurrent = output["recurrent_residual_given_context"]["mean"]
    raw = output["raw_recurrent_memory_value"]["mean"]
    output["interaction"] = {
        "mean": None if recurrent is None or raw is None else recurrent - raw
    }
    return output


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("instrument-preflight", "development-worker"),
        required=True,
    )
    parser.add_argument("--config-id", default="s01")
    parser.add_argument("--optimizer-seed", type=int, default=7_672_001)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite {args.output_dir}")

    contract, freeze_receipt = load_authority(args.mode)
    runtime = runtime_receipt()
    if runtime["git_status_porcelain"]:
        raise RuntimeError("runner requires a clean worktree before output creation")

    configs = {
        str(row["id"]): row
        for row in contract["training_protocol"]["screen_configurations"]
    }
    if args.config_id not in configs:
        raise ValueError("config id is outside the contract")
    cfg = configs[args.config_id]

    if args.mode == "instrument-preflight":
        scope = contract["structured_comparators"]["evaluation_scope"][
            "instrument_preflight"
        ]
        training_roots = integer_range(scope["history_roots"])
        evaluation_roots = list(training_roots)
        kappas = tuple(map(float, scope["kappa_cells"]))
        timesteps = int(contract["training_protocol"]["rollout_steps"])
        checkpoint_interval = timesteps
        claim_status = "BURNED_INSTRUMENT_PREFLIGHT_NO_CLAIM"
        opening_name = "instrument_preflight_receipt.json"
        static_campaigns = set(map(int, scope["campaign_indices"]))
    else:
        training_roots = integer_range(
            contract["data_splits"]["training_history_roots"]
        )
        evaluation_roots = integer_range(
            contract["data_splits"]["checkpoint_selection_history_roots"]
        )
        kappas = KAPPAS
        timesteps = int(contract["training_protocol"]["screen_timesteps_per_seed"])
        checkpoint_interval = int(
            contract["training_protocol"]["checkpoint_interval_timesteps"]
        )
        claim_status = "DEVELOPMENT_NO_CONFIRMATORY_CLAIM"
        opening_name = "development_opening_receipt.json"
        static_campaigns = None
        allowed = set(map(int, contract["data_splits"]["optimizer_seeds"]))
        if args.optimizer_seed not in allowed:
            raise ValueError("optimizer seed is outside the frozen contract")

    args.output_dir.mkdir(parents=True)
    opening_receipt = {
        "schema_version": "q_r1_factorial_v4_opening_receipt",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "claim_status": claim_status,
        "contract": str(CONTRACT_PATH.relative_to(ROOT)),
        "contract_sha256": sha256(CONTRACT_PATH),
        "freeze_receipt_sha256": (
            None if freeze_receipt is None else sha256(FREEZE_RECEIPT_PATH)
        ),
        "training_roots_opened": training_roots,
        "selection_roots_opened": evaluation_roots,
        "kappa_cells": kappas,
        "optimizer_seed": int(args.optimizer_seed),
        "runtime": runtime,
        "confirmation_roots_opened": False,
    }
    write_json(args.output_dir / opening_name, opening_receipt)

    training_histories = build_histories(training_roots, kappas)
    evaluation_histories = (
        training_histories
        if evaluation_roots == training_roots
        else build_histories(evaluation_roots, kappas)
    )
    static_bar = build_static_bar(
        evaluation_histories,
        campaign_indices=static_campaigns,
    )
    write_json(args.output_dir / "static_bar.json", static_bar)

    from sb3_contrib import RecurrentPPO  # noqa: PLC0415

    training_env = make_env(
        training_histories,
        retained_prior=True,
        sampling_seed=args.optimizer_seed,
    )
    model = RecurrentPPO(
        "MlpLstmPolicy",
        training_env,
        seed=args.optimizer_seed,
        n_steps=int(contract["training_protocol"]["rollout_steps"]),
        batch_size=int(contract["training_protocol"]["batch_size"]),
        learning_rate=float(cfg["learning_rate"]),
        gamma=float(cfg["gamma"]),
        gae_lambda=float(cfg["gae_lambda"]),
        ent_coef=float(cfg["entropy"]),
        normalize_advantage=bool(
            contract["training_protocol"]["normalize_advantage"]
        ),
        policy_kwargs={
            "lstm_hidden_size": int(cfg["hidden_size"]),
            "net_arch": {
                "pi": list(contract["training_protocol"]["policy_net_arch"]),
                "vf": list(contract["training_protocol"]["value_net_arch"]),
            },
        },
        verbose=0,
    )
    checkpoints = list(range(0, timesteps + 1, checkpoint_interval))
    if checkpoints[-1] != timesteps:
        checkpoints.append(timesteps)
    checkpoint_rows: dict[int, list[dict[str, Any]]] = {}
    checkpoint_receipts: list[dict[str, Any]] = []
    completed = 0
    started = time.perf_counter()
    for step in checkpoints:
        if step > completed:
            model.learn(
                total_timesteps=step - completed,
                reset_num_timesteps=False,
                progress_bar=False,
            )
            completed = step
        model_path = args.output_dir / f"checkpoint_t{step:06d}"
        model.save(model_path)
        archive = model_path.with_suffix(".zip")
        checkpoint_sha = sha256(archive)
        rows: list[dict[str, Any]] = []
        for arm, prior, reset_hidden in FACTORIAL_ARMS:
            rows.extend(
                evaluate_neural_arm(
                    model,
                    histories=evaluation_histories,
                    arm=arm,
                    retained_prior=prior,
                    reset_hidden_at_boundaries=reset_hidden,
                    checkpoint_sha256=checkpoint_sha,
                )
            )
        for row in rows:
            row["timesteps"] = step
            row["config_id"] = args.config_id
            row["optimizer_seed"] = args.optimizer_seed
        checkpoint_rows[step] = rows
        checkpoint_receipts.append(
            {
                "timesteps": step,
                "path": archive.name,
                "sha256": checkpoint_sha,
                "factorial_arms": [row[0] for row in FACTORIAL_ARMS],
            }
        )

    scope = contract["structured_comparators"]["evaluation_scope"][
        "instrument_preflight"
        if args.mode == "instrument-preflight"
        else "development_checkpoint_selection"
    ]
    structured_histories = filter_histories(
        evaluation_histories,
        roots=set(integer_range(scope["history_roots"])),
    )
    structured, structured_receipt = evaluate_structured(
        structured_histories,
        campaign_indices=set(map(int, scope["campaign_indices"])),
        contract=contract,
    )
    bar_rows = static_rows(
        evaluation_histories,
        calendar=list(map(int, static_bar["calendar"])),
    )

    structured_mean = {
        (
            int(row["history_root"]),
            float(row["kappa"]),
            int(row["campaign_index"]),
        ): float(row[PRIMARY])
        for row in structured
        if row["arm"] == "structured_retained"
    }

    def selection_key(step: int) -> tuple[float, float, float, int]:
        rows = checkpoint_rows[step]
        retained = [row for row in rows if row["arm"] == "P1_H1"]
        primary = float(np.mean([row[PRIMARY] for row in retained]))
        premium_values = [
            float(row[PRIMARY])
            - structured_mean[
                (
                    int(row["history_root"]),
                    float(row["kappa"]),
                    int(row["campaign_index"]),
                )
            ]
            for row in retained
            if (
                int(row["history_root"]),
                float(row["kappa"]),
                int(row["campaign_index"]),
            )
            in structured_mean
        ]
        premium = float(np.mean(premium_values)) if premium_values else float("-inf")
        total = estimands(rows)["total_retained_neural_treatment"]["mean"]
        return primary, premium, float(total), -step

    selected_step = max(checkpoints, key=selection_key)
    selected_checkpoint = next(
        row for row in checkpoint_receipts if row["timesteps"] == selected_step
    )
    selected_rows = checkpoint_rows[selected_step] + structured + bar_rows
    result = {
        "schema_version": "q_r1_matched_retention_factorial_v4_run",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "claim_status": claim_status,
        "mode": args.mode,
        "contract_sha256": sha256(CONTRACT_PATH),
        "config_id": args.config_id,
        "optimizer_seed": args.optimizer_seed,
        "rho": RHO,
        "kappa_cells": kappas,
        "training_roots": training_roots,
        "evaluation_roots": evaluation_roots,
        "static_bar": {
            "calendar": static_bar["calendar"],
            "frontier_row": static_bar["frontier_row"],
            "sha256": sha256(args.output_dir / "static_bar.json"),
        },
        "checkpoints": checkpoint_receipts,
        "selected_checkpoint": selected_checkpoint,
        "selection_rule": contract["training_protocol"]["checkpoint_selection"],
        "structured_comparator": structured_receipt,
        "estimands": estimands(selected_rows),
        "arm_counts": {
            arm: sum(row["arm"] == arm for row in selected_rows)
            for arm in (
                "P0_H0",
                "P1_H0",
                "P0_H1",
                "P1_H1",
                "structured_reset",
                "structured_retained",
                "best_static_frozen",
            )
        },
        "same_checkpoint_hash_all_neural_arms": len(
            {
                row["checkpoint_sha256"]
                for row in selected_rows
                if str(row["arm"]).startswith("P")
            }
        )
        == 1,
        "confirmation_roots_opened": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(args.output_dir / "rows.json", selected_rows)
    write_json(args.output_dir / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
