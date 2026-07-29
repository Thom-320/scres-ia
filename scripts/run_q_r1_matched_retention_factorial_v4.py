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
from collections import Counter
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


CONTRACT_PATH = ROOT / "contracts/q_r1_matched_retention_factorial_v4.json"
FREEZE_RECEIPT_PATH = (
    ROOT / "contracts/q_r1_matched_retention_factorial_v4_freeze_receipt.json"
)
COMPARATOR_FREEZE_PATH = ROOT / "contracts/q_r1_comparator_v2_frozen_c256_v1.json"
STRUCTURED_AMENDMENT_PATH = (
    ROOT / "contracts/q_r1_factorial_v4_shared_structured_amendment_v1.json"
)
STRUCTURED_AMENDMENT_FREEZE_PATH = (
    ROOT
    / "contracts/q_r1_factorial_v4_shared_structured_amendment_v1_freeze_receipt.json"
)
FULL_PHASE_AMENDMENT_PATH = (
    ROOT / "contracts/q_r1_factorial_v4_full_phase_runner_amendment_v1.json"
)
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


def authoritative_static_bar_sha256(
    *,
    mode: str,
    output_dir: Path,
    static_bar_path: Path | None,
) -> str:
    """Hash the static bar that is authoritative for the active mode."""
    path = output_dir / "static_bar.json" if mode == "static-bar" else static_bar_path
    if path is None or not path.is_file():
        raise RuntimeError("authoritative static bar is missing")
    return sha256(path)


def json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def development_timesteps(contract: dict[str, Any], phase: str) -> int:
    """Resolve the frozen development budget for the explicitly named phase."""
    if phase == "screen":
        return int(contract["training_protocol"]["screen_timesteps_per_seed"])
    if phase == "full":
        return int(contract["training_protocol"]["full_timesteps_per_seed"])
    raise ValueError(f"unknown development phase: {phase}")


def validate_full_screen_selection(
    path: Path | None,
    *,
    contract: dict[str, Any],
    config_id: str,
) -> dict[str, Any]:
    """Fail closed unless a full worker belongs to the frozen screen advance."""
    if path is None or not path.is_file():
        raise RuntimeError("full worker requires the screen selection artifact")
    selection = json.loads(path.read_text())
    if selection.get("phase") != "screen":
        raise RuntimeError("full worker screen selection has the wrong phase")
    if selection.get("contract_sha256") != sha256(CONTRACT_PATH):
        raise RuntimeError("full worker screen selection contract mismatch")
    if config_id not in set(map(str, selection.get("advanced_config_ids", []))):
        raise RuntimeError("full worker config did not advance from the screen")
    expected_advances = int(
        contract["training_protocol"]["configuration_selection"]["screen_advances"]
    )
    if len(selection.get("advanced_config_ids", [])) != expected_advances:
        raise RuntimeError("screen selection advance count mismatch")
    return selection


def validate_shared_static_bar(
    *,
    static_bar_path: Path,
    completion_receipt_path: Path,
    opening_receipt_path: Path,
    expected_contract_sha256: str,
    expected_roots: list[int],
    expected_campaigns: int,
    expected_opening_mode: str = "static-bar",
) -> dict[str, Any]:
    """Load the one authoritative static bar and verify its custody chain."""
    opening = json.loads(opening_receipt_path.read_text())
    completion = json.loads(completion_receipt_path.read_text())
    static_bar = json.loads(static_bar_path.read_text())
    if opening.get("mode") != expected_opening_mode:
        raise RuntimeError("development opening receipt has the wrong mode")
    if opening.get("contract_sha256") != expected_contract_sha256:
        raise RuntimeError("development opening receipt contract mismatch")
    if completion.get("schema_version") != (
        "q_r1_factorial_v4_static_bar_completion_receipt"
    ):
        raise RuntimeError("static bar completion receipt schema mismatch")
    if completion.get("mode") != expected_opening_mode:
        raise RuntimeError("static bar completion receipt mode mismatch")
    if completion.get("contract_sha256") != expected_contract_sha256:
        raise RuntimeError("static bar completion receipt contract mismatch")
    if completion.get("opening_receipt_sha256") != sha256(opening_receipt_path):
        raise RuntimeError("static bar completion receipt opening hash mismatch")
    actual_bar_sha = sha256(static_bar_path)
    if completion.get("static_bar_sha256") != actual_bar_sha:
        raise RuntimeError("static bar artifact hash mismatch")
    if completion.get("identities_sha256") != json_sha256(
        static_bar.get("identities")
    ):
        raise RuntimeError("static bar identity digest mismatch")
    static_roots = sorted(
        {int(row["history_root"]) for row in static_bar.get("identities", [])}
    )
    if static_roots != list(map(int, expected_roots)):
        raise RuntimeError("static bar does not cover all selection roots")
    if static_bar.get("selection_campaigns") != int(expected_campaigns):
        raise RuntimeError("static bar selection campaign count mismatch")
    if completion.get("selection_roots") != static_roots:
        raise RuntimeError("static bar completion root coverage mismatch")
    if completion.get("selection_campaigns") != int(expected_campaigns):
        raise RuntimeError("static bar completion campaign count mismatch")
    if completion.get("calendar") != static_bar.get("calendar"):
        raise RuntimeError("static bar completion calendar mismatch")
    if completion.get("frontier_row") != static_bar.get("frontier_row"):
        raise RuntimeError("static bar completion frontier-row mismatch")
    return static_bar


def load_shared_structured_authority() -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify the prospective amendment that hoists invariant structured work."""
    if not STRUCTURED_AMENDMENT_PATH.is_file():
        raise RuntimeError("shared structured amendment is missing")
    if not STRUCTURED_AMENDMENT_FREEZE_PATH.is_file():
        raise RuntimeError("shared structured amendment is not frozen")
    amendment = json.loads(STRUCTURED_AMENDMENT_PATH.read_text())
    receipt = json.loads(STRUCTURED_AMENDMENT_FREEZE_PATH.read_text())
    if amendment.get("status") != "DRAFT_PROSPECTIVE_UNOPENED_NOT_AUTHORITY":
        raise RuntimeError("shared structured amendment content marker mismatch")
    if receipt.get("status") != "FROZEN_PROSPECTIVE_UNOPENED":
        raise RuntimeError("shared structured freeze status mismatch")
    if receipt.get("amendment_sha256") != sha256(STRUCTURED_AMENDMENT_PATH):
        raise RuntimeError("shared structured amendment hash mismatch")
    if receipt.get("base_contract_sha256") != sha256(CONTRACT_PATH):
        raise RuntimeError("shared structured base-contract hash mismatch")
    if receipt.get("confirmation_roots_opened") is not False:
        raise RuntimeError("shared structured freeze does not keep confirmation sealed")
    return amendment, receipt


def validate_shared_structured_bar(
    *,
    rows_path: Path,
    completion_receipt_path: Path,
    opening_receipt_path: Path,
    expected_contract_sha256: str,
    expected_amendment_sha256: str,
    expected_roots: list[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load the sole structured artifact and verify coverage and custody."""
    opening = json.loads(opening_receipt_path.read_text())
    completion = json.loads(completion_receipt_path.read_text())
    rows = json.loads(rows_path.read_text())
    if opening.get("schema_version") != "q_r1_shared_structured_opening_v1":
        raise RuntimeError("shared structured opening schema mismatch")
    if opening.get("base_contract_sha256") != expected_contract_sha256:
        raise RuntimeError("shared structured opening base-contract mismatch")
    if opening.get("amendment_sha256") != expected_amendment_sha256:
        raise RuntimeError("shared structured opening amendment mismatch")
    if completion.get("schema_version") != "q_r1_shared_structured_completion_v1":
        raise RuntimeError("shared structured completion schema mismatch")
    if completion.get("opening_receipt_sha256") != sha256(opening_receipt_path):
        raise RuntimeError("shared structured completion opening hash mismatch")
    if completion.get("base_contract_sha256") != expected_contract_sha256:
        raise RuntimeError("shared structured completion base-contract mismatch")
    if completion.get("amendment_sha256") != expected_amendment_sha256:
        raise RuntimeError("shared structured completion amendment mismatch")
    if completion.get("structured_rows_sha256") != sha256(rows_path):
        raise RuntimeError("shared structured artifact hash mismatch")
    if completion.get("rows_digest_sha256") != json_sha256(rows):
        raise RuntimeError("shared structured row digest mismatch")
    if not isinstance(rows, list) or len(rows) != 192:
        raise RuntimeError("shared structured row count mismatch")
    if Counter(str(row.get("arm")) for row in rows) != Counter(
        {"structured_reset": 96, "structured_retained": 96}
    ):
        raise RuntimeError("shared structured arm coverage mismatch")
    if sorted({int(row["history_root"]) for row in rows}) != list(
        map(int, expected_roots)
    ):
        raise RuntimeError("shared structured root coverage mismatch")
    if {float(row["kappa"]) for row in rows} != set(KAPPAS):
        raise RuntimeError("shared structured kappa coverage mismatch")
    if {int(row["campaign_index"]) for row in rows} != {0, 1}:
        raise RuntimeError("shared structured campaign coverage mismatch")
    identities = [
        {
            "history_root": int(row["history_root"]),
            "kappa": float(row["kappa"]),
            "campaign_index": int(row["campaign_index"]),
            "arm": str(row["arm"]),
            "skeleton_sha256": str(row["skeleton_sha256"]),
            "prefix_state_hash": str(row["prefix_state_hash"]),
        }
        for row in rows
    ]
    if completion.get("identities_sha256") != json_sha256(identities):
        raise RuntimeError("shared structured identity digest mismatch")
    if completion.get("confirmation_roots_opened") is not False:
        raise RuntimeError("shared structured artifact opened confirmation roots")
    return rows, completion


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
        if FREEZE_RECEIPT_PATH.exists():
            raise RuntimeError(
                "instrument preflight is closed after the contract freeze"
            )
        if contract["status"] != "DRAFT_PROSPECTIVE_UNOPENED_NOT_AUTHORITY":
            raise RuntimeError("instrument preflight expects the reviewed draft")
        if contract["data_splits"]["opened"]:
            raise RuntimeError("draft unexpectedly marks development roots open")
        return contract, None
    if contract["status"] != "DRAFT_PROSPECTIVE_UNOPENED_NOT_AUTHORITY":
        raise RuntimeError("frozen contract content marker is unexpected")
    if not FREEZE_RECEIPT_PATH.exists():
        raise RuntimeError("development requires a separate freeze receipt")
    receipt = json.loads(FREEZE_RECEIPT_PATH.read_text())
    if receipt.get("contract_sha256") != sha256(CONTRACT_PATH):
        raise RuntimeError("contract bytes do not match the freeze receipt")
    if receipt.get("status") != "FROZEN_PROSPECTIVE_UNOPENED":
        raise RuntimeError("freeze receipt does not authorize development")
    if receipt.get("reviewed_contract_internal_status") != contract["status"]:
        raise RuntimeError("freeze receipt content-status marker mismatch")
    if receipt.get("fresh_development_roots_opened") is not False:
        raise RuntimeError("freeze receipt does not attest closed development roots")
    if receipt.get("confirmation_roots_opened") is not False:
        raise RuntimeError("freeze receipt does not attest closed confirmation roots")
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
            metrics["service_loss"] = metrics["service_loss_auc"]
            metrics["whole_campaign_ret"] = metrics["ret_visible"]
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
    progress_dir: Path,
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
    budget = contract["structured_comparators"]["compute_budget"]

    def persist_progress(
        completed_rows: list[dict[str, Any]],
        completed_cache: dict[
            tuple[str, str, str],
            tuple[tuple[int, ...], dict[str, object], dict[str, Any]],
        ],
    ) -> None:
        partial_rows = progress_dir / "structured_rows.partial.json"
        write_json(partial_rows, completed_rows)
        cache_payload = [
            {
                "key": list(key),
                "calendar": list(value[0]),
                "diagnostics": value[1],
                "metrics": value[2],
            }
            for key, value in sorted(completed_cache.items())
        ]
        partial_cache = progress_dir / "structured_cache.partial.json"
        write_json(partial_cache, cache_payload)
        write_json(
            progress_dir / "structured_progress.json",
            {
                "complete": False,
                "rows_persisted": len(completed_rows),
                "cache_entries_persisted": len(completed_cache),
                "rows_sha256": sha256(partial_rows),
                "cache_sha256": sha256(partial_cache),
                "confirmation_roots_opened": False,
            },
        )

    def persist_rejection(
        rejection: dict[str, Any],
        completed_rows: list[dict[str, Any]],
        completed_cache: dict[
            tuple[str, str, str],
            tuple[tuple[int, ...], dict[str, object], dict[str, Any]],
        ],
    ) -> None:
        persist_progress(completed_rows, completed_cache)
        receipt = {
            **rejection,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "contract_sha256": sha256(CONTRACT_PATH),
            "comparator_freeze_sha256": sha256(COMPARATOR_FREEZE_PATH),
            "rows_persisted_before_rejection": len(completed_rows),
            "cache_entries_persisted_before_rejection": len(completed_cache),
            "confirmation_roots_opened": False,
        }
        write_json(
            progress_dir / "structured_rejected_over_cap.json",
            receipt,
        )

    started = time.perf_counter()
    rows = structured_pair_rows(
        histories=selected,
        prior_paths=selected_paths,
        scheduler=scheduler(),
        config=config,
        cache=cache,
        per_calendar_hard_cap_seconds=float(
            budget["per_calendar_hard_cap_seconds"]
        ),
        aggregate_hard_cap_seconds=float(
            budget["development_sequential_hard_cap_seconds"]
        ),
        progress_callback=persist_progress,
        rejection_callback=persist_rejection,
    )
    elapsed = time.perf_counter() - started
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
        choices=("instrument-preflight", "static-bar", "development-worker"),
        required=True,
    )
    parser.add_argument("--config-id", default="s01")
    parser.add_argument("--optimizer-seed", type=int, default=7_672_001)
    parser.add_argument(
        "--development-phase", choices=("screen", "full"), default="screen"
    )
    parser.add_argument("--screen-selection", type=Path)
    parser.add_argument("--static-bar-path", type=Path)
    parser.add_argument("--static-bar-completion-receipt", type=Path)
    parser.add_argument("--development-opening-receipt", type=Path)
    parser.add_argument("--structured-bar-path", type=Path)
    parser.add_argument("--structured-bar-completion-receipt", type=Path)
    parser.add_argument("--structured-bar-opening-receipt", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite {args.output_dir}")
    if args.mode != "development-worker" and args.development_phase != "screen":
        raise ValueError("development phase applies only to development workers")

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
        expected_preflight_seed = int(
            contract["data_splits"]["instrument_preflight_optimizer_seed"]
        )
        if args.optimizer_seed != expected_preflight_seed:
            raise ValueError(
                "instrument preflight seed differs from the contracted burned seed"
            )
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
        timesteps = development_timesteps(contract, args.development_phase)
        checkpoint_interval = int(
            contract["training_protocol"]["checkpoint_interval_timesteps"]
        )
        claim_status = "DEVELOPMENT_NO_CONFIRMATORY_CLAIM"
        opening_name = "development_opening_receipt.json"
        static_campaigns = None
        if args.mode == "development-worker":
            allowed = set(map(int, contract["data_splits"]["optimizer_seeds"]))
            if args.optimizer_seed not in allowed:
                raise ValueError("optimizer seed is outside the frozen contract")
            if args.development_phase == "full":
                validate_full_screen_selection(
                    args.screen_selection,
                    contract=contract,
                    config_id=args.config_id,
                )

    shared_static_bar: dict[str, Any] | None = None
    shared_structured_rows: list[dict[str, Any]] | None = None
    shared_structured_completion: dict[str, Any] | None = None
    if args.mode == "development-worker":
        if (
            args.development_opening_receipt is None
            or not args.development_opening_receipt.is_file()
        ):
            raise RuntimeError(
                "development worker requires the static-bar opening receipt"
            )
        if args.static_bar_path is None or not args.static_bar_path.is_file():
            raise RuntimeError(
                "development worker requires the shared static-bar artifact"
            )
        if (
            args.static_bar_completion_receipt is None
            or not args.static_bar_completion_receipt.is_file()
        ):
            raise RuntimeError(
                "development worker requires the static-bar completion receipt"
            )
        expected_roots = integer_range(
            contract["data_splits"]["checkpoint_selection_history_roots"]
        )
        shared_static_bar = validate_shared_static_bar(
            static_bar_path=args.static_bar_path,
            completion_receipt_path=args.static_bar_completion_receipt,
            opening_receipt_path=args.development_opening_receipt,
            expected_contract_sha256=sha256(CONTRACT_PATH),
            expected_roots=expected_roots,
            expected_campaigns=len(expected_roots)
            * CAMPAIGNS_PER_METAEPISODE
            * len(KAPPAS),
        )
        amendment, _amendment_receipt = load_shared_structured_authority()
        if (
            args.structured_bar_path is None
            or not args.structured_bar_path.is_file()
            or args.structured_bar_completion_receipt is None
            or not args.structured_bar_completion_receipt.is_file()
            or args.structured_bar_opening_receipt is None
            or not args.structured_bar_opening_receipt.is_file()
        ):
            raise RuntimeError(
                "development worker requires the frozen shared structured artifact"
            )
        shared_structured_rows, shared_structured_completion = (
            validate_shared_structured_bar(
                rows_path=args.structured_bar_path,
                completion_receipt_path=args.structured_bar_completion_receipt,
                opening_receipt_path=args.structured_bar_opening_receipt,
                expected_contract_sha256=sha256(CONTRACT_PATH),
                expected_amendment_sha256=sha256(STRUCTURED_AMENDMENT_PATH),
                expected_roots=expected_roots,
            )
        )

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
    if args.mode in {"instrument-preflight", "static-bar"}:
        write_json(args.output_dir / opening_name, opening_receipt)
    else:
        assert args.development_opening_receipt is not None
        write_json(
            args.output_dir / "worker_opening_reference.json",
            {
                "opening_receipt": str(args.development_opening_receipt),
                "opening_receipt_sha256": sha256(
                    args.development_opening_receipt
                ),
                "static_bar_completion_receipt": str(
                    args.static_bar_completion_receipt
                ),
                "static_bar_completion_receipt_sha256": sha256(
                    args.static_bar_completion_receipt
                ),
                "contract_sha256": sha256(CONTRACT_PATH),
                "development_phase": args.development_phase,
                "screen_selection": (
                    None
                    if args.screen_selection is None
                    else str(args.screen_selection)
                ),
                "screen_selection_sha256": (
                    None
                    if args.screen_selection is None
                    else sha256(args.screen_selection)
                ),
                "structured_bar": str(args.structured_bar_path),
                "structured_bar_sha256": sha256(args.structured_bar_path),
                "structured_bar_completion_receipt_sha256": sha256(
                    args.structured_bar_completion_receipt
                ),
                "structured_amendment_sha256": sha256(
                    STRUCTURED_AMENDMENT_PATH
                ),
            },
        )

    training_histories = (
        ()
        if args.mode == "static-bar"
        else build_histories(training_roots, kappas)
    )
    evaluation_histories = (
        training_histories
        if training_histories and evaluation_roots == training_roots
        else build_histories(evaluation_roots, kappas)
    )
    if args.mode in {"instrument-preflight", "static-bar"}:
        static_bar = build_static_bar(
            evaluation_histories,
            campaign_indices=static_campaigns,
        )
        write_json(args.output_dir / "static_bar.json", static_bar)
        static_bar_path = args.output_dir / "static_bar.json"
        opening_path = args.output_dir / opening_name
        completion_receipt = {
            "schema_version": (
                "q_r1_factorial_v4_static_bar_completion_receipt"
            ),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "mode": args.mode,
            "claim_status": claim_status,
            "contract_sha256": sha256(CONTRACT_PATH),
            "opening_receipt_sha256": sha256(opening_path),
            "static_bar_sha256": sha256(static_bar_path),
            "identities_sha256": json_sha256(static_bar["identities"]),
            "selection_roots": sorted(
                {
                    int(row["history_root"])
                    for row in static_bar["identities"]
                }
            ),
            "selection_campaigns": int(static_bar["selection_campaigns"]),
            "calendar": static_bar["calendar"],
            "frontier_row": int(static_bar["frontier_row"]),
            "immutable": True,
        }
        write_json(
            args.output_dir / "static_bar_completion_receipt.json",
            completion_receipt,
        )
        validate_shared_static_bar(
            static_bar_path=static_bar_path,
            completion_receipt_path=(
                args.output_dir / "static_bar_completion_receipt.json"
            ),
            opening_receipt_path=opening_path,
            expected_contract_sha256=sha256(CONTRACT_PATH),
            expected_roots=completion_receipt["selection_roots"],
            expected_campaigns=int(static_bar["selection_campaigns"]),
            expected_opening_mode=args.mode,
        )
    else:
        assert args.static_bar_path is not None
        assert shared_static_bar is not None
        static_bar = shared_static_bar
        write_json(
            args.output_dir / "static_bar_reference.json",
            {
                "path": str(args.static_bar_path),
                "sha256": sha256(args.static_bar_path),
                "completion_receipt_sha256": sha256(
                    args.static_bar_completion_receipt
                ),
                "calendar": static_bar["calendar"],
                "frontier_row": static_bar["frontier_row"],
            },
        )

    if args.mode == "static-bar":
        result = {
            "schema_version": "q_r1_factorial_v4_static_bar_run",
            "claim_status": "DEVELOPMENT_OPENING_NO_LEARNER_RESULT",
            "contract_sha256": sha256(CONTRACT_PATH),
            "static_bar_sha256": authoritative_static_bar_sha256(
                mode=args.mode,
                output_dir=args.output_dir,
                static_bar_path=args.static_bar_path,
            ),
            "static_bar_completion_receipt_sha256": sha256(
                args.output_dir / "static_bar_completion_receipt.json"
            ),
            "training_roots_opened": training_roots,
            "selection_roots_opened": evaluation_roots,
            "confirmation_roots_opened": False,
            "learner_evaluated": False,
            "structured_comparator_evaluated": False,
        }
        write_json(args.output_dir / "result.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

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
        checkpoint_rows_path = (
            args.output_dir / f"checkpoint_rows_t{step:06d}.json"
        )
        write_json(checkpoint_rows_path, rows)
        checkpoint_receipts.append(
            {
                "timesteps": step,
                "path": archive.name,
                "sha256": checkpoint_sha,
                "rows_path": checkpoint_rows_path.name,
                "rows_sha256": sha256(checkpoint_rows_path),
                "factorial_arms": [row[0] for row in FACTORIAL_ARMS],
            }
        )
        write_json(
            args.output_dir / "checkpoint_progress.json",
            {
                "schema_version": "q_r1_factorial_v4_checkpoint_progress",
                "completed_timesteps": [
                    row["timesteps"] for row in checkpoint_receipts
                ],
                "checkpoint_receipts": checkpoint_receipts,
                "structured_evaluation_started": False,
                "confirmation_roots_opened": False,
            },
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
    if args.mode == "development-worker":
        assert shared_structured_rows is not None
        assert shared_structured_completion is not None
        structured = shared_structured_rows
        structured_receipt = {
            "mode": "shared_immutable_artifact",
            "elapsed_seconds": 0.0,
            "rows": len(structured),
            "source_elapsed_seconds": float(
                shared_structured_completion["elapsed_seconds"]
            ),
            "source_rows_sha256": sha256(args.structured_bar_path),
            "source_completion_receipt_sha256": sha256(
                args.structured_bar_completion_receipt
            ),
            "source_opening_receipt_sha256": sha256(
                args.structured_bar_opening_receipt
            ),
            "amendment_sha256": sha256(STRUCTURED_AMENDMENT_PATH),
            "confirmation_roots_opened": False,
        }
        write_json(
            args.output_dir / "structured_bar_reference.json",
            structured_receipt,
        )
    else:
        structured, structured_receipt = evaluate_structured(
            structured_histories,
            campaign_indices=set(map(int, scope["campaign_indices"])),
            contract=contract,
            progress_dir=args.output_dir,
        )
    write_json(args.output_dir / "structured_rows.json", structured)
    write_json(
        args.output_dir / "structured_progress.json",
        {
            **structured_receipt,
            "complete": True,
            "rows_sha256": sha256(args.output_dir / "structured_rows.json"),
        },
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

    def selection_key(step: int) -> tuple[float, float, float, float, int]:
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
        point_estimands = estimands(rows)
        total = float(point_estimands["total_retained_neural_treatment"]["mean"])
        iid_rows = [row for row in rows if float(row["kappa"]) == 0.5]
        iid_effect = estimands(iid_rows)["total_retained_neural_treatment"]["mean"]
        return primary, premium, total, -abs(float(iid_effect)), -step

    selected_step = max(checkpoints, key=selection_key)
    selected_checkpoint = next(
        row for row in checkpoint_receipts if row["timesteps"] == selected_step
    )
    selected_rows = checkpoint_rows[selected_step] + structured + bar_rows
    all_checkpoint_rows = [
        row
        for receipt in checkpoint_receipts
        for row in json.loads(
            (args.output_dir / str(receipt["rows_path"])).read_text()
        )
    ]
    result = {
        "schema_version": "q_r1_matched_retention_factorial_v4_run",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "claim_status": claim_status,
        "mode": args.mode,
        "development_phase": (
            args.development_phase
            if args.mode == "development-worker"
            else None
        ),
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
            "sha256": authoritative_static_bar_sha256(
                mode=args.mode,
                output_dir=args.output_dir,
                static_bar_path=args.static_bar_path,
            ),
        },
        "checkpoints": checkpoint_receipts,
        "selected_checkpoint": selected_checkpoint,
        "checkpoint_selection_scores": {
            str(step): list(map(float, selection_key(step))) for step in checkpoints
        },
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
    write_json(args.output_dir / "checkpoint_rows.json", all_checkpoint_rows)
    write_json(args.output_dir / "rows.json", selected_rows)
    result["checkpoint_rows_sha256"] = sha256(
        args.output_dir / "checkpoint_rows.json"
    )
    write_json(args.output_dir / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
