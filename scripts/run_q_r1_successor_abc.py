#!/usr/bin/env python3
"""Run the repaired Q-R1 A/B/C estimands without overwriting burned results.

A is a two-decision treatment followed by natural reset-MPC replanning on the
state actually reached. B sustains each arm's controller for the campaign. C
reproduces the historical calendar splice and is diagnostic-only.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_program_q_replication import scheduler  # noqa: E402
from supply_chain.program_t_full_des_mpc import FullDEST0Config  # noqa: E402
from supply_chain.program_t_joint_belief import ExactJointBelief  # noqa: E402
from supply_chain.q_r1_retained_learning import (  # noqa: E402
    ACTUAL_RESOURCE_KEYS,
    ESTIMANDS,
    RESOURCE_KEYS,
    PhysicalCampaignState,
    common_continuation_calendar,
    controller_calendar,
    controller_calendar_from_prefix,
    evaluate_calendar,
)
from supply_chain.retained_context_discovery import (  # noqa: E402
    ARMS as HISTORICAL_ARMS,
    arm_priors,
    build_campaign_history,
)


ARMS = HISTORICAL_ARMS + ("delayed_posterior",)
DEPLOYABLE_ARMS = ("retained_posterior", "reset_posterior_0p5")


def contract_identity(contract: dict[str, object]) -> str:
    payload = dict(contract)
    payload["contract_identity_sha256"] = None
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def load_and_validate_contract(args: argparse.Namespace) -> tuple[dict[str, object] | None, str | None]:
    """Bind a prospective shard to the frozen successor contract."""
    if args.contract is None:
        if args.burned or args.allow_dirty_smoke:
            return None, None
        raise RuntimeError("prospective Q-R1 runs require --contract")
    contract_path = args.contract.resolve()
    contract = json.loads(contract_path.read_text())
    if contract.get("status") != "FROZEN_PROSPECTIVE_UNOPENED":
        raise RuntimeError("Q-R1 contract is not frozen and unopened")
    planner = contract["selected_universal_planner"]
    expected = {
        "horizon": args.horizon,
        "mode": args.mode,
        "particles": args.particles,
        "belief_integration": args.belief_integration,
    }
    if any(planner.get(key) != value for key, value in expected.items()):
        raise RuntimeError("runner planner does not match the frozen contract")
    if args.campaigns != contract.get("campaigns_per_history"):
        raise RuntimeError("campaign count does not match the frozen contract")
    shard = [args.seed_start, args.seed_start + args.histories - 1]
    allowed = [row["history_roots"] for row in contract.get("shards", [])]
    if shard not in allowed:
        raise RuntimeError("requested shard is not an exact frozen shard")
    identity = contract_identity(contract)
    if contract.get("contract_identity_sha256") != identity:
        raise RuntimeError("Q-R1 contract identity digest mismatch")
    return contract, identity


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def runtime_receipt() -> dict[str, object]:
    packages = {}
    for name in ("numpy", "scipy", "simpy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "commit": git_value("rev-parse", "HEAD"),
        "git_status_porcelain": git_value("status", "--porcelain"),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "argv": sys.argv,
    }


def fixed_theta_belief(prior_c: float) -> ExactJointBelief:
    """Known rho90/share90 model with no access to the current latent state."""
    return ExactJointBelief.from_theta_marginal(
        (0.0, 0.0, 1.0), probability_regime_c=float(prior_c)
    )


def delayed_paths(retained: list[tuple[float, ...]]) -> list[tuple[float, ...]]:
    output = []
    for path in retained:
        output.append(tuple(0.5 if index < 2 else path[index - 1] for index in range(len(path))))
    return output


def clustered_interval(
    values: dict[int, list[float]], *, seed: int, draws: int = 10_000
) -> tuple[float, float]:
    roots = sorted(values)
    means = np.asarray([np.mean(values[root]) for root in roots], dtype=float)
    if len(means) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(means), size=(draws, len(means)))
    distribution = means[sampled].mean(axis=1)
    return tuple(map(float, np.quantile(distribution, (0.025, 0.975))))


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    indexed = {
        (
            row["estimand"],
            row["kappa"],
            row["history_root"],
            row["campaign_index"],
            row["arm"],
        ): row
        for row in rows
    }
    output: dict[str, object] = {}
    for estimand in ESTIMANDS:
        by_kappa: dict[str, object] = {}
        for kappa in (0.5, 0.75, 0.9):
            roots = sorted(
                {
                    int(row["history_root"])
                    for row in rows
                    if row["estimand"] == estimand and row["kappa"] == kappa
                }
            )
            by_arm: dict[str, object] = {}
            for arm in ARMS:
                visible: dict[int, list[float]] = defaultdict(list)
                complete: dict[int, list[float]] = defaultdict(list)
                total: dict[int, list[float]] = defaultdict(list)
                fill_by_root: dict[int, list[float]] = defaultdict(list)
                fills: list[float] = []
                unresolved: list[float] = []
                unresolved_quantity: list[float] = []
                lost: list[float] = []
                lost_quantity: list[float] = []
                service: list[float] = []
                favorable: list[bool] = []
                scheduled_error = 0.0
                for root in roots:
                    campaigns = sorted(
                        int(row["campaign_index"])
                        for row in rows
                        if row["estimand"] == estimand
                        and row["kappa"] == kappa
                        and int(row["history_root"]) == root
                        and row["arm"] == arm
                        and int(row["campaign_index"]) > 0
                    )
                    for campaign in campaigns:
                        target = indexed[(estimand, kappa, root, campaign, arm)]
                        reset = indexed[(estimand, kappa, root, campaign, "reset_posterior_0p5")]
                        visible_delta = float(target["early_ret_visible"]) - float(reset["early_ret_visible"])
                        complete_delta = float(target["early_ret_complete_cohort"]) - float(reset["early_ret_complete_cohort"])
                        visible[root].append(visible_delta)
                        complete[root].append(complete_delta)
                        total[root].append(float(target["ret_visible"]) - float(reset["ret_visible"]))
                        favorable.append(complete_delta > 0.0)
                        fill_delta = float(target["worst_product_fill"]) - float(reset["worst_product_fill"])
                        fills.append(fill_delta)
                        fill_by_root[root].append(fill_delta)
                        unresolved.append(float(target["unresolved_orders"]) - float(reset["unresolved_orders"]))
                        unresolved_quantity.append(float(target["unresolved_quantity"]) - float(reset["unresolved_quantity"]))
                        lost.append(float(target["lost_orders"]) - float(reset["lost_orders"]))
                        lost_quantity.append(float(target["lost_quantity"]) - float(reset["lost_quantity"]))
                        service.append(float(target["service_loss_auc"]) - float(reset["service_loss_auc"]))
                        for key in RESOURCE_KEYS:
                            scheduled_error = max(
                                scheduled_error,
                                abs(float(target[key]) - float(reset[key])),
                            )
                visible_values = [value for root in roots for value in visible[root]]
                complete_values = [value for root in roots for value in complete[root]]
                total_values = [value for root in roots for value in total[root]]
                visible_ci = clustered_interval(visible, seed=20260731)
                complete_ci = clustered_interval(complete, seed=20260732)
                fill_ci = clustered_interval(fill_by_root, seed=20260733)
                by_arm[arm] = {
                    "mean_early_ret_visible_delta": float(np.mean(visible_values)),
                    "early_ret_visible_clustered_ci95": list(visible_ci),
                    "mean_early_ret_complete_cohort_delta": float(np.mean(complete_values)),
                    "early_ret_complete_cohort_clustered_ci95": list(complete_ci),
                    "early_ret_complete_cohort_delta_quantiles": {
                        "q10": float(np.quantile(complete_values, 0.10)),
                        "q50": float(np.quantile(complete_values, 0.50)),
                        "q90": float(np.quantile(complete_values, 0.90)),
                    },
                    "mean_total_ret_visible_delta": float(np.mean(total_values)),
                    "favorable_complete_cohort_fraction": float(np.mean(favorable)),
                    "mean_worst_product_fill_delta": float(np.mean(fills)),
                    "worst_product_fill_clustered_ci95": list(fill_ci),
                    "mean_unresolved_orders_delta": float(np.mean(unresolved)),
                    "max_unresolved_orders_delta": float(np.max(unresolved)),
                    "mean_unresolved_quantity_delta": float(np.mean(unresolved_quantity)),
                    "mean_lost_orders_delta": float(np.mean(lost)),
                    "mean_lost_quantity_delta": float(np.mean(lost_quantity)),
                    "mean_service_loss_auc_delta": float(np.mean(service)),
                    "max_scheduled_resource_error": float(scheduled_error),
                    "n_pairs": len(complete_values),
                }
            by_kappa[str(kappa)] = by_arm
        output[estimand] = by_kappa
    return output


def run(args: argparse.Namespace, output_dir: Path) -> dict[str, object]:
    runtime = runtime_receipt()
    if runtime["git_status_porcelain"] and not args.allow_dirty_smoke:
        raise RuntimeError("scientific Q-R1 runs require a clean worktree")
    contract, contract_sha256 = load_and_validate_contract(args)
    sched = scheduler()
    config = FullDEST0Config(
        horizon=args.horizon,
        mode=args.mode,
        particles=args.particles,
        worst_product_floor=0.70,
        belief_integration=args.belief_integration,
    )
    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for kappa in (0.5, 0.75, 0.9):
        histories = [
            build_campaign_history(
                history_root=args.seed_start + index,
                campaigns=args.campaigns,
                kappa=kappa,
                scheduler=sched,
                regime_persistence=0.90,
                dominant_share=0.90,
            )
            for index in range(args.histories)
        ]
        priors = arm_priors(
            histories=histories,
            regime_persistence=0.90,
            dominant_share=0.90,
        )
        priors["delayed_posterior"] = delayed_paths(priors["retained_posterior"])
        for history_index, history in enumerate(histories):
            for campaign in history:
                if time.perf_counter() - started > args.hard_cap_seconds:
                    raise TimeoutError("Q-R1 successor hard cap exceeded")
                physical = PhysicalCampaignState(
                    history_root=campaign.history_root,
                    campaign_index=campaign.campaign_index,
                    persistence_mode="iid" if kappa == 0.5 else f"binary_{kappa}",
                    theta=(0.90, 0.90),
                    initial_regime=campaign.initial_regime,
                    skeleton=campaign.skeleton,
                )
                beliefs = {
                    arm: fixed_theta_belief(priors[arm][history_index][campaign.campaign_index])
                    for arm in ARMS
                }
                sustained: dict[str, tuple[int, ...]] = {}
                details: dict[tuple[str, str], dict[str, object]] = {}
                for arm in ARMS:
                    sustained[arm], details[("sustained_control", arm)] = controller_calendar(
                        campaign=physical,
                        belief=beliefs[arm],
                        scheduler=sched,
                        config=config,
                    )
                reset_calendar = sustained["reset_posterior_0p5"]
                calendars: dict[tuple[str, str], tuple[int, ...]] = {}
                for arm in ARMS:
                    calendars[("sustained_control", arm)] = sustained[arm]
                    prefix = sustained[arm][:2]
                    calendars[("prefix_natural_replanning", arm)], details[("prefix_natural_replanning", arm)] = controller_calendar_from_prefix(
                        campaign=physical,
                        treatment_prefix=prefix,
                        continuation_belief=beliefs["reset_posterior_0p5"],
                        scheduler=sched,
                        config=config,
                    )
                    calendars[("historical_splice", arm)] = common_continuation_calendar(
                        sustained[arm], reset_calendar
                    )
                    details[("historical_splice", arm)] = {
                        "config_id": "historical_calendar_splice_diagnostic_only",
                        "online_ms": 0.0,
                    }
                for estimand in ESTIMANDS:
                    for arm in ARMS:
                        metrics = evaluate_calendar(
                            campaign=physical,
                            calendar=calendars[(estimand, arm)],
                            scheduler=sched,
                        )
                        rows.append(
                            {
                                "estimand": estimand,
                                "claim_eligible": estimand != "historical_splice",
                                "kappa": kappa,
                                "history_root": campaign.history_root,
                                "campaign_index": campaign.campaign_index,
                                "tape_seed": campaign.tape_seed,
                                "arm": arm,
                                "initial_belief_c": priors[arm][history_index][campaign.campaign_index],
                                "calendar": list(calendars[(estimand, arm)]),
                                "skeleton_sha256": campaign.skeleton.skeleton_sha256,
                                "prefix_state_hash": campaign.skeleton.prefix_state_hash,
                                "online_ms": float(details[(estimand, arm)].get("online_ms", 0.0)),
                                **metrics,
                            }
                        )
    payload = {
        "schema_version": "q_r1_successor_abc_v1",
        "claim_status": (
            "SMOKE_NO_CLAIM"
            if args.allow_dirty_smoke
            else ("BURNED_DEVELOPMENT_NO_CLAIM" if args.burned else "PROSPECTIVE_CONFIRMATION")
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "history_roots": [args.seed_start, args.seed_start + args.histories - 1],
        "histories": args.histories,
        "campaigns_per_history": args.campaigns,
        "planner": config.config_id,
        "contract_sha256": contract_sha256,
        "contract_schema": None if contract is None else contract.get("schema_version"),
        "estimand_boundaries": {
            "prefix_natural_replanning": "two treatment actions, then reset-belief policy replans on each reached state",
            "sustained_control": "arm controller acts for all eight decisions",
            "historical_splice": "diagnostic-only calendar splice; never claim-eligible",
        },
        "deployable_arms": list(DEPLOYABLE_ARMS),
        "oracle_boundary": "oracle initial context is an information upper bound and never deployable",
        "summary": summarize(rows),
        "rows": rows,
        "elapsed_seconds": time.perf_counter() - started,
        "runtime": runtime,
    }
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    try:
        result_label = str(result_path.relative_to(ROOT))
    except ValueError:
        result_label = str(result_path)
    receipt = {
        "result_sha256": sha256(result_path),
        "result": result_label,
        "row_count": len(rows),
        "historical_artifacts_overwritten": False,
    }
    (output_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--histories", type=int, default=32)
    parser.add_argument("--campaigns", type=int, default=12)
    parser.add_argument("--horizon", type=int, choices=(1, 3, 4, 6, 8), default=3)
    parser.add_argument("--mode", choices=("scenario", "robust", "constraint_aware"), default="scenario")
    parser.add_argument("--particles", type=int, default=16)
    parser.add_argument("--belief-integration", choices=("mc", "stratified"), default="stratified")
    parser.add_argument("--hard-cap-seconds", type=float, default=1800.0)
    parser.add_argument("--burned", action="store_true")
    parser.add_argument("--allow-dirty-smoke", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    payload = run(args, output_dir)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
