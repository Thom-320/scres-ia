#!/usr/bin/env python3
"""Run Garrido Q2 as a between-simulation closed loop.

The learner chooses a WRAP configuration, observes the completed DES run, and
then chooses the next configuration.  It never changes a simulation while an
episode is running.  The default learner is the linear null because the Q1
neural promotion gate is currently held by the WRAP fidelity and metric gates.

This runner is intentionally a development harness.  It writes the full
retained/reset trace and keeps the claim status at HOLD until a promoted Q1
model, a complete oracle, and the behavioral WRAP gate are all available.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
from typing import Any, Callable, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_garrido_fig5_surrogate import (  # noqa: E402
    design_features,
    fit_kan,
    fit_mlp,
)
from supply_chain.config import INVENTORY_BUFFERS, THESIS_FAITHFUL_PROTOCOL  # noqa: E402
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.garrido_thesis_design import DESIGN, Configuration  # noqa: E402
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402

FAMILY_RISKS = {
    "R1r": ("R11", "R12", "R13", "R14"),
    "R2r": ("R21", "R22", "R23", "R24"),
    "R3": ("R3",),
}
DEFAULT_ORACLE = ROOT / "results/garrido_drivers_per_configuration/result.json"
DEFAULT_Q1 = ROOT / "results/garrido_fig5_surrogate/result.json"


@dataclass(frozen=True)
class Candidate:
    cf: int
    cfg: Configuration
    seed: int

    def feature_row(self) -> dict[str, Any]:
        return {
            "cf": self.cf,
            "family": self.cfg.risk_family,
            "pattern": self.cfg.risk_pattern,
            "seed": self.seed,
            "rho": {
                "buffer_hours": self.cfg.buffer_hours,
                "shifts": self.cfg.shifts,
            },
        }


def candidate_table(*, fallback_seed_base: int) -> dict[int, Candidate]:
    return {
        cf: Candidate(
            cf=cf,
            cfg=cfg,
            seed=(cfg.seed if cfg.seed is not None else fallback_seed_base + cfg.base_index),
        )
        for cf, cfg in DESIGN.items()
    }


def load_oracle(path: Path) -> tuple[dict[int, float], dict[str, Any]]:
    """Load a development oracle without treating it as author data."""
    payload = json.loads(path.read_text())
    values: dict[int, float] = {}
    for row in payload.get("rows", []):
        cf = int(row["cf"])
        if "ret_excel" in row:
            values[cf] = float(row["ret_excel"])
        elif isinstance(row.get("ours"), dict) and "ret_excel" in row["ours"]:
            values[cf] = float(row["ours"]["ret_excel"])
    metadata = {
        "path": str(path),
        "claim_status": payload.get("claim_status"),
        "schema_version": payload.get("schema_version"),
        "n_values": len(values),
    }
    return values, metadata


def run_des_configuration(
    candidate: Candidate,
    *,
    horizon_hours: float | None = None,
) -> dict[str, float | int | str]:
    """Run one candidate under the strict thesis physical settings."""
    cfg = candidate.cfg
    buffers = None
    period = None
    if cfg.buffer_hours:
        level = INVENTORY_BUFFERS[cfg.buffer_hours]
        buffers = {
            "op3_rm": float(level["op3_rm"]),
            "op5_rm": float(level["op5_rm"]),
            "op9_rations": float(level["op9_rations"]),
        }
        period = float(cfg.buffer_hours)
    run_horizon = float(cfg.horizon_hours if horizon_hours is None else horizon_hours)
    if run_horizon <= 0.0:
        raise ValueError("horizon_hours must be positive")
    sim = MFSCSimulation(
        shifts=cfg.shifts,
        initial_buffers=buffers,
        inventory_replenishment_period=period,
        seed=candidate.seed,
        horizon=run_horizon,
        risks_enabled=True,
        risk_level="current",
        enabled_risks=set(FAMILY_RISKS[cfg.risk_family]),
        risk_overrides={risk: "increased" for risk in cfg.increased_risks},
        strict_exogenous_crn=True,
        year_basis=THESIS_FAITHFUL_PROTOCOL["year_basis"],
        warmup_trigger=THESIS_FAITHFUL_PROTOCOL["warmup_trigger"],
        r14_defect_mode=THESIS_FAITHFUL_PROTOCOL["r14_defect_mode"],
    )
    sim.step(action=None, step_hours=run_horizon)
    metrics = compute_episode_metrics(sim)
    return {
        "cf": candidate.cf,
        "seed": candidate.seed,
        "horizon_hours": run_horizon,
        "warmup_hours": float(sim.warmup_time),
        "ret_excel": float(metrics["ret_excel"]),
        "fill_rate": float(metrics["fill_rate"]),
        "flow_fill_rate": float(metrics["flow_fill_rate"]),
        "backorder_qty_final": float(metrics["backorder_qty_final"]),
        "service_loss_auc_ration_hours": float(
            metrics["service_loss_auc_ration_hours"]
        ),
        "n_orders": float(metrics["n_orders"]),
    }


class BetweenRunLearner:
    """A small, auditable selector whose only retained state is observations."""

    def __init__(self, kind: str, *, seed: int, retained: bool, update: bool = True):
        if kind not in {"linear", "backprop", "kan"}:
            raise ValueError(f"unsupported learner: {kind}")
        self.kind = kind
        self.seed = int(seed)
        self.retained = bool(retained)
        self.update_enabled = bool(update)
        self.observations: list[tuple[int, float]] = []

    def start_campaign(self) -> None:
        if not self.retained:
            self.observations = []

    def observe(self, cf: int, ret_excel: float) -> None:
        if self.update_enabled:
            self.observations.append((int(cf), float(ret_excel)))

    def select(
        self,
        candidates: Iterable[int],
        candidate_rows: dict[int, dict[str, Any]],
    ) -> int:
        available = sorted(set(int(cf) for cf in candidates))
        if not available:
            raise ValueError("no unobserved candidates remain")
        if len(self.observations) < 2:
            return available[0]

        observed_ids = [cf for cf, _ in self.observations]
        observed_rows = [candidate_rows[cf] for cf in observed_ids]
        target = np.asarray([value for _, value in self.observations], dtype=np.float64)
        x_train = design_features(observed_rows)
        x_test = design_features([candidate_rows[cf] for cf in available])

        if self.kind == "linear":
            from sklearn.linear_model import LinearRegression

            model = LinearRegression().fit(x_train, target)
            predictions = model.predict(x_test)
        elif self.kind == "backprop":
            predictions, _ = fit_mlp(
                x_train,
                target,
                x_test,
                seed=self.seed,
                classify=False,
            )
        else:
            predictions, _ = fit_kan(
                x_train,
                target,
                x_test,
                seed=self.seed,
                classify=False,
            )
        # Stable tie-breaking by Cf makes the policy reproducible and auditable.
        return min(
            zip(available, predictions.tolist()),
            key=lambda item: (-float(item[1]), int(item[0])),
        )[0]


def campaign_groups(
    candidates: dict[int, Candidate],
    *,
    order: str,
    max_campaigns: int | None,
    seed: int,
) -> list[tuple[str, list[int]]]:
    grouped: dict[str, list[int]] = {family: [] for family in FAMILY_RISKS}
    for cf, candidate in candidates.items():
        grouped[candidate.cfg.risk_family].append(cf)
    campaigns = [(family, sorted(cfs)) for family, cfs in grouped.items()]
    if order == "shuffled":
        random.Random(seed).shuffle(campaigns)
    if max_campaigns is not None:
        campaigns = campaigns[: max(0, int(max_campaigns))]
    return campaigns


def _best_regret(
    observed: Iterable[float],
    *,
    oracle_best: float | None,
) -> tuple[float | None, float | None]:
    best = max(observed, default=None)
    if best is None or oracle_best is None:
        return best, None
    return best, float(oracle_best - best)


def run_arm(
    arm_name: str,
    candidates: dict[int, Candidate],
    campaigns: list[tuple[str, list[int]]],
    *,
    budget: int,
    learner_kind: str,
    learner_seed: int,
    retained: bool,
    update: bool,
    horizon_hours: float | None,
    oracle: dict[int, float],
    simulate: Callable[[Candidate], dict[str, Any]],
) -> dict[str, Any]:
    learner = BetweenRunLearner(
        learner_kind,
        seed=learner_seed,
        retained=retained,
        update=update,
    )
    records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for campaign_index, (family, campaign_candidates) in enumerate(campaigns):
        learner.start_campaign()
        remaining = set(campaign_candidates)
        observed_values: list[float] = []
        oracle_best = (
            max((oracle[cf] for cf in campaign_candidates), default=None)
            if all(cf in oracle for cf in campaign_candidates)
            else None
        )
        for step in range(min(int(budget), len(remaining))):
            if arm_name == "ofat":
                cf = min(remaining)
            else:
                cf = learner.select(remaining, {key: c.feature_row() for key, c in candidates.items()})
            remaining.remove(cf)
            outcome = simulate(candidates[cf])
            ret = float(outcome["ret_excel"])
            observed_values.append(ret)
            if arm_name != "ofat":
                learner.observe(cf, ret)
            best_so_far, regret = _best_regret(observed_values, oracle_best=oracle_best)
            records.append(
                {
                    "arm": arm_name,
                    "campaign_index": campaign_index,
                    "campaign_family": family,
                    "step": step,
                    "cf": cf,
                    "retained": retained,
                    "update_enabled": update,
                    "outcome": outcome,
                    "best_so_far": best_so_far,
                    "simple_regret": regret,
                    "learner_observation_count": len(learner.observations),
                }
            )
        best_so_far, regret = _best_regret(observed_values, oracle_best=oracle_best)
        summaries.append(
            {
                "campaign_index": campaign_index,
                "campaign_family": family,
                "budget": min(int(budget), len(campaign_candidates)),
                "n_runs": len(observed_values),
                "best_so_far": best_so_far,
                "oracle_best": oracle_best,
                "simple_regret": regret,
            }
        )
    return {
        "arm": arm_name,
        "learner": learner_kind,
        "retained": retained,
        "update_enabled": update,
        "campaigns": summaries,
        "records": records,
    }


def summarize_comparison(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    retained = arms.get("retained")
    reset = arms.get("reset")
    if retained is None or reset is None:
        return {"status": "HOLD_COMPARISON_ARMS_MISSING"}
    retained_by_family = {
        item["campaign_family"]: item for item in retained["campaigns"]
    }
    reset_by_family = {item["campaign_family"]: item for item in reset["campaigns"]}
    common = sorted(set(retained_by_family) & set(reset_by_family))
    if not common:
        return {"status": "HOLD_NO_COMMON_CAMPAIGNS"}
    retained_best = np.asarray(
        [retained_by_family[family]["best_so_far"] for family in common], dtype=float
    )
    reset_best = np.asarray(
        [reset_by_family[family]["best_so_far"] for family in common], dtype=float
    )
    result: dict[str, Any] = {
        "status": "DEVELOPMENT_PAIRED_CAMPAIGN_COMPARISON",
        "campaigns": common,
        "retained_minus_reset_best_so_far": float(np.mean(retained_best - reset_best)),
        "retained_best_so_far_mean": float(np.mean(retained_best)),
        "reset_best_so_far_mean": float(np.mean(reset_best)),
    }
    retained_regret = [
        retained_by_family[family]["simple_regret"]
        for family in common
        if retained_by_family[family]["simple_regret"] is not None
    ]
    reset_regret = [
        reset_by_family[family]["simple_regret"]
        for family in common
        if reset_by_family[family]["simple_regret"] is not None
    ]
    if retained_regret and reset_regret:
        result["retained_minus_reset_regret"] = float(
            np.mean(retained_regret) - np.mean(reset_regret)
        )
    else:
        result["regret_status"] = "HOLD_ORACLE_INCOMPLETE"
    return result


def load_q1_gate(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "promotion_eligible": False}
    payload = json.loads(path.read_text())
    decision = payload.get("q1_decision", {})
    return {
        "exists": True,
        "path": str(path),
        "claim_status": payload.get("claim_status"),
        "decision": decision.get("decision"),
        "selected_model_before_gates": decision.get("selected_model_before_gates"),
        "promotion_eligible": bool(decision.get("promotion_eligible", False)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/garrido_wrap_q2/result.json")
    parser.add_argument("--oracle-json", type=Path, default=DEFAULT_ORACLE)
    parser.add_argument("--q1-json", type=Path, default=DEFAULT_Q1)
    parser.add_argument("--learner", choices=("auto", "linear", "backprop", "kan"), default="auto")
    parser.add_argument("--allow-development-learner", action="store_true")
    parser.add_argument("--budget", type=int, default=6)
    parser.add_argument("--horizon-hours", type=float, default=None)
    parser.add_argument("--campaign-order", choices=("fixed", "shuffled"), default="fixed")
    parser.add_argument("--max-campaigns", type=int, default=None)
    parser.add_argument("--fallback-seed-base", type=int, default=900_000)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.budget <= 0:
        raise SystemExit("--budget must be positive")
    candidates = candidate_table(fallback_seed_base=args.fallback_seed_base)
    campaigns = campaign_groups(
        candidates,
        order=args.campaign_order,
        max_campaigns=args.max_campaigns,
        seed=args.seed,
    )
    oracle, oracle_metadata = load_oracle(args.oracle_json)
    q1 = load_q1_gate(args.q1_json)
    learner = args.learner
    if learner == "auto":
        learner = (
            q1.get("selected_model_before_gates")
            if q1.get("promotion_eligible")
            else "linear"
        )
    if learner != "linear" and not args.allow_development_learner:
        raise SystemExit(
            "neural learner is not promoted; pass --allow-development-learner only "
            "for a development run"
        )
    if learner not in {"linear", "backprop", "kan"}:
        raise SystemExit(f"invalid resolved learner: {learner}")

    output: dict[str, Any] = {
        "schema_version": "garrido_wrap_q2_closed_loop_v1",
        "contract_id": "garrido_wrap_scres_ai_v1",
        "claim_status": "HOLD_WRAP_BEHAVIORAL_FIDELITY",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "question": "Can between-run retained state choose WRAP configurations better than OFAT and reset state under the same budget?",
        "learner": learner,
        "learner_requested": args.learner,
        "q1_gate": q1,
        "oracle": oracle_metadata,
        "oracle_complete": all(cf in oracle for cf in candidates),
        "campaign_order": args.campaign_order,
        "campaigns": [
            {"family": family, "cf": cfs, "budget": min(args.budget, len(cfs))}
            for family, cfs in campaigns
        ],
        "same_seed_policy": "same Candidate.seed per Cf across arms; no future event access",
        "physical_protocol": dict(THESIS_FAITHFUL_PROTOCOL),
    }

    if args.dry_run:
        output["status"] = "DRY_RUN_NO_DES_EXECUTED"
        output["arms"] = {}
        output["comparison"] = {"status": "DRY_RUN"}
    else:
        cache: dict[tuple[int, float | None], dict[str, Any]] = {}

        def simulate(candidate: Candidate) -> dict[str, Any]:
            key = (candidate.cf, args.horizon_hours)
            if key not in cache:
                cache[key] = run_des_configuration(
                    candidate,
                    horizon_hours=args.horizon_hours,
                )
            return dict(cache[key])

        arm_payloads: dict[str, dict[str, Any]] = {}
        for arm_name, retained, update in (
            ("ofat", False, False),
            ("retained", True, True),
            ("reset", False, True),
            ("no_update", True, False),
        ):
            arm_payloads[arm_name] = run_arm(
                arm_name,
                candidates,
                campaigns,
                budget=args.budget,
                learner_kind=learner,
                learner_seed=args.seed,
                retained=retained,
                update=update,
                horizon_hours=args.horizon_hours,
                oracle=oracle,
                simulate=simulate,
            )
        output["status"] = "DEVELOPMENT_Q2_CLOSED_LOOP"
        output["simulation_cache_entries"] = len(cache)
        output["arms"] = arm_payloads
        output["comparison"] = summarize_comparison(
            {"retained": arm_payloads["retained"], "reset": arm_payloads["reset"]}
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"Saved: {args.output}")
    print(f"Claim status: {output['claim_status']}")
    print(f"Learner: {learner}; campaigns: {len(campaigns)}; budget: {args.budget}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

