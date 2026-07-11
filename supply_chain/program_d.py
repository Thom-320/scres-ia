"""Shared, fail-closed utilities for Program D D1-v2."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import BACKORDER_PRIORITY_RULE_OPTIONS, HOURS_PER_WEEK
from .episode_metrics import compute_episode_metrics
from .l_program_env import CampaignTape, GarridoLearningEnv, materialize_campaign_tape

PROGRAM_D_CONTRACT = "program_d_d1_v2"
RULES = tuple(BACKORDER_PRIORITY_RULE_OPTIONS)
FAMILIES = ("R1", "R2", "mixed")
LEVELS = ("current", "increased")
PROXY_PATH = Path(__file__).resolve().parent / "data" / "garrido_proxy_v1_freeze_2026-07-10.json"


def make_tapes(split: str, seed_start: int, n: int = 30, horizon_weeks: int = 104) -> list[CampaignTape]:
    if n != 30:
        raise ValueError("The frozen calibration design requires exactly 30 tapes per split.")
    cells = [(family, level) for family in FAMILIES for level in LEVELS]
    tapes: list[CampaignTape] = []
    for i in range(n):
        family, level = cells[i // 5]
        seed = seed_start + i
        raw = CampaignTape(
            campaign_id=f"d1v2-{split}-{family}-{level}-{seed}",
            family=family,
            risk_level=level,
            base_seed=seed,
            horizon_weeks=horizon_weeks,
            split=split,
            contract_version=PROGRAM_D_CONTRACT,
        )
        tapes.append(materialize_campaign_tape(raw))
    return tapes


def exogenous_hash(sim: Any, treatment_start: float) -> dict[str, str]:
    risks = [
        {
            "id": str(e.risk_id), "start": round(float(e.start_time - treatment_start), 9),
            "end": round(float(e.end_time - treatment_start), 9),
            "ops": list(map(int, e.affected_ops)), "magnitude": round(float(e.magnitude), 9),
        }
        for e in sim.risk_events if float(e.start_time) >= treatment_start - 1e-9
    ]
    demand = [
        (round(float(t - treatment_start), 9), round(float(q), 9))
        for t, q in getattr(sim, "daily_demand", []) if float(t) >= treatment_start - 1e-9
    ]
    def digest(value: Any) -> str:
        return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"risk_sha256": digest(risks), "demand_sha256": digest(demand)}


def run_constant(tape: CampaignTape, rule: str) -> dict[str, Any]:
    if rule not in RULES:
        raise ValueError(rule)
    env = GarridoLearningEnv(max_steps=tape.horizon_weeks, buffer_level=0)
    env.reset(seed=tape.base_seed, options={"campaign_tape": tape, "buffer_level": 0, "initial_state_seed": tape.base_seed, "initial_shift": 1})
    env.sim.set_backorder_priority_rule(rule)
    backlog_auc = 0.0
    final_info: dict[str, Any] = {}
    for _ in range(tape.horizon_weeks):
        _, _, terminated, truncated, final_info = env.step(0)
        backlog_auc += float(env.sim.pending_backorder_qty) * HOURS_PER_WEEK
        if terminated or truncated:
            break
    metrics = compute_episode_metrics(env.sim, treatment_start=env._treatment_start)
    ledger = env.sim.flow_ledger()
    metrics.update({
        "backlog_auc_ration_hours": backlog_auc,
        "raw_mass_residual": float(ledger["raw_residual"]),
        "ration_mass_residual": float(ledger["ration_residual"]),
        "mass_balance_residual": max(abs(float(ledger["raw_residual"])), abs(float(ledger["ration_residual"]))),
        "priority_rule_change_count": len(env.sim.backorder_priority_rule_events),
    })
    metrics.update(exogenous_hash(env.sim, env._treatment_start))
    return metrics


def paired_bootstrap(values: Iterable[float], *, seed: int, n_boot: int = 10000) -> dict[str, Any]:
    x = np.asarray(list(values), dtype=float)
    if x.size == 0:
        return {"mean": float("nan"), "ci95": [float("nan"), float("nan")]}
    rng = np.random.default_rng(seed)
    boot = np.mean(x[rng.integers(0, x.size, size=(n_boot, x.size))], axis=1)
    return {"mean": float(np.mean(x)), "ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]}
