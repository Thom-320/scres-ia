#!/usr/bin/env python3
"""Pre-PPO headroom gate for ``preventive_reserve_v2``.

The runner evaluates only fixed and observable policies.  It cannot launch PPO.
Its terminal verdict requires (i) a perfect-warning upper bound, (ii) imperfect
warning value over a shuffled placebo, (iii) physical reserve use, and (iv)
Pareto non-dominance against the static reserve frontier.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from supply_chain.l_program_env import CampaignTape, materialize_campaign_tape
from supply_chain.v2_preventive_env import PreventiveReserveV2Env


ARMS = {
    "static_0": ("none", 0),
    "static_15000": ("none", 1),
    "static_30000": ("none", 2),
    "perfect_warning": ("perfect", 0),
    "imperfect_warning": ("imperfect", 0),
    "shuffled_placebo": ("shuffled_placebo", 0),
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _paired_bootstrap(
    values: np.ndarray, *, seed: int = 20260710, draws: int = 10_000
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    samples = values[rng.integers(0, n, size=(draws, n))].mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(samples, 0.025)),
        float(np.quantile(samples, 0.975)),
    )


def _policy_action(arm: str, observation: np.ndarray) -> int:
    if arm == "static_0":
        return 0
    if arm == "static_15000":
        return 1
    if arm == "static_30000":
        return 2
    return 2 if float(observation[3]) > 0.5 else 0


def run_arm(
    *,
    arm: str,
    tape: CampaignTape,
    warning_seed: int,
    reserve_cost: float,
    contract_id: str,
    physical_downstream: bool,
) -> dict[str, Any]:
    warning_mode, initial_index = ARMS[arm]
    env = PreventiveReserveV2Env(
        horizon_weeks=tape.horizon_weeks,
        holding_cost_per_unit_day=reserve_cost,
        contract_id=contract_id,
        replenishment_transport_mode=(
            "physical_downstream" if physical_downstream else "fixed_lead"
        ),
        replenishment_lead_hours=(48.0 if physical_downstream else 336.0),
    )
    observation, reset_info = env.reset(
        seed=tape.base_seed,
        options={
            "campaign_tape": tape,
            "warning_mode": warning_mode,
            "warning_seed": warning_seed,
            "initial_target_index": initial_index,
        },
    )
    while True:
        action = _policy_action(arm, observation)
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    metrics = env.terminal_metrics()
    row = {
        "arm": arm,
        "campaign_id": tape.campaign_id,
        "tape_sha256": tape.digest(),
        "base_seed": tape.base_seed,
        "warning_sha256": reset_info["warning_schedule"]["sha256"],
        **metrics,
    }
    env.close()
    return _jsonable(row)


def _dominated(candidate: dict[str, float], other: dict[str, float]) -> bool:
    weak = (
        other["service_loss"] <= candidate["service_loss"]
        and other["inventory_time"] <= candidate["inventory_time"]
        and other["ret_excel"] >= candidate["ret_excel"]
    )
    strict = (
        other["service_loss"] < candidate["service_loss"]
        or other["inventory_time"] < candidate["inventory_time"]
        or other["ret_excel"] > candidate["ret_excel"]
    )
    return weak and strict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-tapes", type=int, default=30)
    parser.add_argument("--horizon-weeks", type=int, default=52)
    parser.add_argument("--seed-start", type=int, default=772000)
    parser.add_argument("--reserve-cost", type=float, default=1.0)
    parser.add_argument("--contract-id", default="preventive_reserve_v2")
    parser.add_argument("--physical-downstream", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/preventive_reserve_v2/gate1"),
    )
    parser.add_argument(
        "--screen-only",
        action="store_true",
        help="Run diagnostics but never emit a promotion verdict.",
    )
    args = parser.parse_args()
    if args.n_tapes < 4:
        raise ValueError("At least four paired tapes are required.")
    args.output.mkdir(parents=True, exist_ok=True)

    tapes: list[CampaignTape] = []
    for index in range(args.n_tapes):
        family = "R2" if index % 2 == 0 else "mixed"
        level = "current" if (index // 2) % 2 == 0 else "increased"
        seed = args.seed_start + index
        tape = CampaignTape(
            campaign_id=f"v2-cal-{seed}",
            family=family,
            risk_level=level,
            base_seed=seed,
            horizon_weeks=args.horizon_weeks,
            split="calibration",
        )
        tapes.append(materialize_campaign_tape(tape))

    rows: list[dict[str, Any]] = []
    for tape in tapes:
        warning_seed = tape.base_seed + 90_001
        for arm in ARMS:
            rows.append(
                run_arm(
                    arm=arm,
                    tape=tape,
                    warning_seed=warning_seed,
                    reserve_cost=args.reserve_cost,
                    contract_id=args.contract_id,
                    physical_downstream=bool(args.physical_downstream),
                )
            )

    by_key = {(row["campaign_id"], row["arm"]): row for row in rows}
    campaign_ids = [tape.campaign_id for tape in tapes]

    def vector(arm: str, field: str) -> np.ndarray:
        return np.asarray(
            [float(by_key[(campaign_id, arm)][field]) for campaign_id in campaign_ids],
            dtype=np.float64,
        )

    loss_field = "service_loss_auc_ration_hours"
    perfect_gain = (
        vector("static_0", loss_field)
        - vector("perfect_warning", loss_field)
    ) / np.maximum(vector("static_0", loss_field), 1.0)
    alert_gain = (
        vector("shuffled_placebo", loss_field)
        - vector("imperfect_warning", loss_field)
    ) / np.maximum(vector("shuffled_placebo", loss_field), 1.0)
    perfect_stats = _paired_bootstrap(perfect_gain)
    alert_stats = _paired_bootstrap(alert_gain, seed=20260711)

    summary: dict[str, dict[str, float]] = {}
    for arm in ARMS:
        summary[arm] = {
            "service_loss": float(vector(arm, loss_field).mean()),
            "inventory_time": float(
                vector(arm, "emergency_reserve_inventory_time").mean()
            ),
            "ret_excel": float(vector(arm, "ret_excel").mean()),
            "reserve_issued": float(
                vector(arm, "emergency_reserve_units_issued").mean()
            ),
        }

    deployable = ("static_0", "static_15000", "static_30000")
    perfect_dominated = any(
        _dominated(summary["perfect_warning"], summary[arm]) for arm in deployable
    )
    imperfect_dominated = any(
        _dominated(summary["imperfect_warning"], summary[arm]) for arm in deployable
    )
    reserve_live_fraction = float(
        np.mean(vector("perfect_warning", "emergency_reserve_units_issued") > 0.0)
    )
    perfect_within_static15_inventory = (
        summary["perfect_warning"]["inventory_time"]
        <= 1.02 * summary["static_15000"]["inventory_time"]
    )
    perfect_pass = (
        perfect_stats[1] > 0.0
        and perfect_stats[0] >= 0.05
        and perfect_within_static15_inventory
        and not perfect_dominated
    )
    alert_pass = alert_stats[1] > 0.0 and alert_stats[0] >= 0.05
    liveness_pass = reserve_live_fraction >= 0.50
    pareto_pass = not imperfect_dominated
    promoted = perfect_pass and alert_pass and liveness_pass and pareto_pass
    if args.screen_only:
        verdict = "SCREEN_ONLY_NO_PROMOTION_AUTHORITY"
    elif not perfect_pass:
        verdict = "STOP_NO_PREVENTIVE_HEADROOM"
    elif not alert_pass:
        verdict = "STOP_WARNING_NOT_ACTIONABLE"
    elif not liveness_pass:
        verdict = "STOP_RESERVE_MECHANISM_NOT_LIVE"
    elif not pareto_pass:
        verdict = "STOP_ALERT_POLICY_STATIC_DOMINATED"
    else:
        verdict = "PROMOTE_TO_PPO_PILOT"

    manifest = {
        "contract_id": args.contract_id,
        "replenishment_transport_mode": (
            "physical_downstream" if args.physical_downstream else "fixed_lead"
        ),
        "preregistered_gate": {
            "perfect_warning_relative_service_gain_vs_static0_min": 0.05,
            "perfect_inventory_time_le_static15_plus_2pct": True,
            "perfect_must_be_pareto_nondominated_vs_static_frontier": True,
            "imperfect_vs_placebo_relative_service_gain_min": 0.05,
            "paired_ci95_lower_must_exceed_zero": True,
            "reserve_liveness_fraction_min": 0.50,
            "imperfect_must_be_pareto_nondominated_vs_static_frontier": True,
        },
        "screen_only": bool(args.screen_only),
        "n_tapes": args.n_tapes,
        "horizon_weeks": args.horizon_weeks,
        "seed_start": args.seed_start,
        "arms": list(ARMS),
        "summary": summary,
        "contrasts": {
            "perfect_vs_static_0_relative_service_gain": {
                "mean": perfect_stats[0],
                "ci95": [perfect_stats[1], perfect_stats[2]],
                "pass": perfect_pass,
            },
            "imperfect_vs_shuffled_relative_service_gain": {
                "mean": alert_stats[0],
                "ci95": [alert_stats[1], alert_stats[2]],
                "pass": alert_pass,
            },
        },
        "reserve_live_fraction": reserve_live_fraction,
        "liveness_pass": liveness_pass,
        "perfect_static_dominated": perfect_dominated,
        "perfect_within_static15_inventory": perfect_within_static15_inventory,
        "imperfect_static_dominated": imperfect_dominated,
        "pareto_pass": pareto_pass,
        "promoted_to_ppo": bool(promoted and not args.screen_only),
        "verdict": verdict,
    }
    rows_path = args.output / "episode_rows.json"
    rows_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    manifest["episode_rows_sha256"] = sha256(rows_path.read_bytes()).hexdigest()
    (args.output / "verdict.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
