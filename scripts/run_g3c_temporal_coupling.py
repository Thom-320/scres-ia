#!/usr/bin/env python3
"""Canonical G3c burned-only preflight runner.

The runner compares a causal two-state hysteresis policy with the frozen myopic
equivariant rule under minimum dwell.  It accepts only the burned contention
block and requires the v2 contract explicitly.  No confirmatory role is exposed
here: opening fresh roots needs a later, separately versioned authority.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from statistics import NormalDist
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.config import HOURS_PER_DAY, HOURS_PER_WEEK  # noqa: E402
from supply_chain.config import THESIS_FAITHFUL_PROTOCOL as P  # noqa: E402
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.seed_custody import custody_falsifier, module_manifest  # noqa: E402
from supply_chain.service_first_metric import claimant_fills  # noqa: E402
from supply_chain.scientific_payload import (  # noqa: E402
    canonical_scientific_payload,
    scientific_payload_sha256,
)
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402

R1R = ("R11", "R12", "R13", "R14")
R2R = ("R21", "R22", "R23", "R24")
REGIMES = {
    "R1r+R2r|base": (R1R + R2R, {}, {}),
    "R1r+R2r|freq3_imp2": (R1R + R2R, {"R23": 3.0}, {"R23": 2.0}),
}
DWELL_LEVELS = (1, 6, 12)
TREATMENT_LEVELS = (6, 12)
CONSTANT_SHARES = tuple(round(i / 10.0, 1) for i in range(1, 10))
PRIMARY = "worst_claimant_fill"
GUARDRAIL_MARGINS = {
    "flow_fill_rate": 0.005,
    "lost_orders": 0.50,
    "backorder_qty_final_relative": 0.010,
}
SESOI = 0.010
SEED_BASE = 5_200_001
BURNED_SEEDS = tuple(range(5_200_001, 5_200_017))
STEP_HOURS = 24.0
HORIZON_WEEKS = 52
TAU_IN = 0.10
TAU_OUT = 0.02
TARGET_HIGH = 0.9
TARGET_LOW = 0.1
TARGET_NEUTRAL = 0.5
N_BOOT = 5_000
MAX_SEEDS_PER_CELL = 96
POWER_TARGET = 0.90
ALPHA = 0.05
Z_POWER = NormalDist().inv_cdf(POWER_TARGET)
NORM_CRITICAL = NormalDist().inv_cdf(1.0 - ALPHA / 4.0)
REFERENCE = Path("results/headroom/g3_obs_conversion_v2/result.json")
CONTRACT_SCHEMA = "g3c_burned_preflight_v2"
MODULES = (
    "scripts/run_g3c_temporal_coupling.py",
    "supply_chain/g3c_temporal.py",
    "supply_chain/supply_chain.py",
    "supply_chain/config.py",
    "supply_chain/episode_metrics.py",
    "supply_chain/arm_runner.py",
    "supply_chain/scientific_payload.py",
    "supply_chain/seed_custody.py",
    "supply_chain/service_first_metric.py",
)


def myopic_target(unmet_a: float, unmet_b: float, *, wrong: bool = False,
                  tolerance: float = 1e-9) -> float:
    """Frozen state-feedback rule; ``wrong`` is its claimant-direction control."""
    delta = float(unmet_a) - float(unmet_b)
    if abs(delta) <= tolerance:
        target = TARGET_NEUTRAL
    elif delta > 0.0:
        target = TARGET_HIGH
    else:
        target = TARGET_LOW
    if not wrong:
        return float(target)
    # Mirror through the registered levels, never by arithmetic: `1.0 - 0.9` is
    # 0.09999999999999998, which is not TARGET_LOW. Against an exact 1e-9 comparison that makes
    # the wrong-claimant arm request an undeclared split, and re-request it at every step.
    return {TARGET_HIGH: TARGET_LOW, TARGET_LOW: TARGET_HIGH,
            TARGET_NEUTRAL: TARGET_NEUTRAL}[target]


def normalized_unmet_delta(unmet_a: float, unmet_b: float) -> float:
    total = max(float(unmet_a) + float(unmet_b), 1.0)
    return (float(unmet_a) - float(unmet_b)) / total


def hysteresis_target(state: int, delta: float, *, tau_in: float = TAU_IN,
                      tau_out: float = TAU_OUT) -> tuple[int, float]:
    """Update a causal two-state hysteresis controller.

    State ``1`` serves A, ``-1`` serves B, and ``0`` is neutral.  The outer
    threshold enters a state; the inner threshold releases it.  The thresholds
    are fixed in the v2 contract and never tuned on the burned tapes.
    """
    if tau_in <= tau_out or tau_out < 0.0:
        raise ValueError("hysteresis requires tau_in > tau_out >= 0")
    if state not in {-1, 0, 1}:
        raise ValueError("hysteresis state must be -1, 0, or 1")
    delta = float(delta)
    if state == 1 and delta < tau_out:
        state = 0
    elif state == -1 and delta > -tau_out:
        state = 0
    if state == 0:
        if delta >= tau_in:
            state = 1
        elif delta <= -tau_in:
            state = -1
    target = TARGET_HIGH if state == 1 else TARGET_LOW if state == -1 else TARGET_NEUTRAL
    return state, float(target)


def _build(seed: int, risks, freq, impact, *, dwell: int | None) -> MFSCSimulation:
    kwargs = dict(
        shifts=1,
        initial_buffers={n: 0.0 for n in ("op3_rm", "op5_rm", "op9_rations")},
        inventory_replenishment_period=0.0,
        seed=seed,
        horizon=float(HORIZON_WEEKS * HOURS_PER_WEEK),
        risks_enabled=True,
        risk_level="current",
        enabled_risks=set(risks),
        risk_frequency_multipliers_by_id=dict(freq) or None,
        risk_impact_multipliers_by_id=dict(impact) or None,
        cssu_topology_mode="split_v1",
        cssu_allocation_a=TARGET_NEUTRAL,
        cssu_service_rule="FIFO_PARTIAL",
        cssu_reallocate_unused=False,
        cssu_switch_cost_rations=0.0,
        order_fulfillment_mode="op9_linked",
        op9_dispatch_policy="fixed_clock_daily",
        strict_exogenous_crn=True,
        year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"],
        r14_defect_mode=P["r14_defect_mode"],
    )
    if dwell is not None:
        kwargs["cssu_min_dwell_days"] = int(dwell)
    return MFSCSimulation(**kwargs)


def _panel(sim: MFSCSimulation) -> dict[str, float]:
    panel = compute_episode_metrics(sim)
    fills = claimant_fills(sim)
    return {
        PRIMARY: float(min(fills.values())) if fills else float("nan"),
        "flow_fill_rate": float(panel.get("flow_fill_rate", float("nan"))),
        "lost_orders": float(panel.get("lost_orders", float("nan"))),
        "backorder_qty_final": float(panel.get("backorder_qty_final", float("nan"))),
        "switches": float(getattr(sim, "cssu_switch_count", 0)),
        "blocked_by_dwell": float(getattr(sim, "cssu_blocked_by_dwell_count", 0)),
        "alpha_final": float(sim.cssu_allocation_a),
    }


def episode_policy(seed: int, risks, freq, impact, *, dwell: int | None,
                   mode: str, constant_share: float | None = None,
                   retain_trace: bool = False) -> dict[str, object]:
    """Run one frozen policy on one tape and retain compact decision evidence."""
    valid = {"myopic", "hysteresis", "placebo", "wrong_claimant", "constant"}
    if mode not in valid:
        raise ValueError(f"unknown G3c policy mode: {mode}")
    if mode == "constant" and constant_share is None:
        raise ValueError("constant policy requires constant_share")
    sim = _build(seed, risks, freq, impact, dwell=dwell)
    placebo_rng = np.random.default_rng(seed ^ 0x9E3779B9)
    trace: list[dict[str, float | str | int]] = []
    n_decisions = 0
    alphas: list[float] = []
    done = False
    hysteresis_state = 0
    while not done:
        unmet_a = float(sim.cssu_demanded.get("A", 0.0)) - float(sim.cssu_delivered.get("A", 0.0))
        unmet_b = float(sim.cssu_demanded.get("B", 0.0)) - float(sim.cssu_delivered.get("B", 0.0))
        delta = normalized_unmet_delta(unmet_a, unmet_b)
        if mode == "myopic":
            target = myopic_target(unmet_a, unmet_b)
        elif mode == "wrong_claimant":
            target = myopic_target(unmet_a, unmet_b, wrong=True)
        elif mode == "placebo":
            target = float(placebo_rng.choice((TARGET_HIGH, TARGET_NEUTRAL, TARGET_LOW)))
        elif mode == "constant":
            target = float(constant_share)
        else:
            hysteresis_state, target = hysteresis_target(hysteresis_state, delta)
        if target != TARGET_NEUTRAL:
            n_decisions += 1
        if retain_trace:
            trace.append({
                "time": float(sim.env.now),
                "unmet_a": unmet_a,
                "unmet_b": unmet_b,
                "normalized_delta": delta,
                "target": float(target),
                "mode": mode,
                "hysteresis_state": int(hysteresis_state),
            })
        action = None
        if (sim._pending_cssu_action is None
                and abs(float(sim.cssu_allocation_a) - target) > 1e-9):
            action = {"cssu_allocation_a": target}
        _, _, done, _ = sim.step(action=action, step_hours=STEP_HOURS)
        alphas.append(float(sim.cssu_allocation_a))
    panel = _panel(sim)
    scientific_payload = canonical_scientific_payload(sim, panel)
    return {
        "seed": int(seed),
        "dwell": None if dwell is None else int(dwell),
        "mode": mode,
        "constant_share": constant_share,
        "metrics": panel,
        "scientific_payload_sha256": scientific_payload_sha256(scientific_payload),
        "n_steps": len(trace),
        "n_decisions": n_decisions,
        "trace": trace,
        "alpha_sd": float(np.std(alphas)) if alphas else 0.0,
    }


def _metric(row: dict[str, object], name: str) -> float:
    return float(row["metrics"][name])  # type: ignore[index]


def paired_bootstrap(values: np.ndarray, n_boot: int, rng: np.random.Generator) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("cannot bootstrap an empty paired vector")
    draws = values[rng.integers(0, values.size, size=(int(n_boot), values.size))].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "sd": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "lcb95": float(np.percentile(draws, 2.5)),
        "ucb95": float(np.percentile(draws, 97.5)),
        "n": int(values.size),
        "n_boot": int(n_boot),
    }


def required_n(sd: float, target: float) -> int:
    if target <= 0.0:
        raise ValueError("power target must be positive")
    if not np.isfinite(sd) or sd <= 0.0:
        return 1
    return int(np.ceil(((Z_POWER + NORM_CRITICAL) * float(sd) / float(target)) ** 2))


def mde_at_n(sd: float, n: int) -> float:
    if not np.isfinite(sd) or sd <= 0.0:
        return 0.0
    return float((Z_POWER + NORM_CRITICAL) * float(sd) / np.sqrt(max(int(n), 1)))


def check_myopic_trace(trace: list[dict[str, object]]) -> dict[str, object]:
    mismatches: list[dict[str, float]] = []
    for row in trace:
        expected = myopic_target(float(row["unmet_a"]), float(row["unmet_b"]))
        if abs(float(row["target"]) - expected) > 1e-9:
            mismatches.append({"time": float(row["time"]), "expected": expected,
                               "observed": float(row["target"])})
            if len(mismatches) >= 5:
                break
    return {"passed": not mismatches, "mismatches": mismatches,
            "future_symbols_read": [], "policy": "myopic_equivariant_tau0"}


def validate_guardrail_margins(margins: dict[str, float]) -> dict[str, object]:
    required = set(GUARDRAIL_MARGINS)
    missing = sorted(required - set(margins))
    nonpositive = sorted(k for k, v in margins.items()
                         if k in required and float(v) <= 0.0)
    return {"passed": not missing and not nonpositive,
            "missing": missing, "nonpositive": nonpositive,
            "stochastic_margins": {k: float(margins[k]) for k in sorted(required)}}


def guardrail_harm(reference: list[dict[str, object]], candidate: list[dict[str, object]],
                   metric: str) -> np.ndarray:
    if metric == "flow_fill_rate":
        return np.asarray([_metric(r, metric) - _metric(c, metric)
                           for r, c in zip(reference, candidate)], dtype=float)
    if metric == "lost_orders":
        return np.asarray([_metric(c, metric) - _metric(r, metric)
                           for r, c in zip(reference, candidate)], dtype=float)
    if metric == "backorder_qty_final_relative":
        return np.asarray([
            (_metric(c, "backorder_qty_final") - _metric(r, "backorder_qty_final"))
            / max(abs(_metric(r, "backorder_qty_final")), 1.0)
            for r, c in zip(reference, candidate)
        ], dtype=float)
    raise ValueError(f"unknown guardrail: {metric}")


def _cell_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [{k: v for k, v in row.items() if k != "trace"} for row in rows]


def _load_contract(path: Path) -> dict[str, object]:
    try:
        contract = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot load JSON contract {path}: {exc}") from exc
    if contract.get("schema_version") != CONTRACT_SCHEMA:
        raise SystemExit(f"G3c runner requires {CONTRACT_SCHEMA}")
    return contract


def _check_frozen_plan(contract: dict[str, object]) -> dict[str, object]:
    mechanism = contract.get("mechanism", {})
    policy = contract.get("policy", {})
    power = contract.get("power", {})
    expected = {
        "level_derivation_evidence": "results/headroom/g3c_dwell_inertia/result.json",
        "levels_days": list(DWELL_LEVELS),
        "treatment_levels_days": list(TREATMENT_LEVELS),
        "null_level_days": 1,
        "hysteresis_tau_in_normalized_unmet": TAU_IN,
        "hysteresis_tau_out_normalized_unmet": TAU_OUT,
        "contrasts": 4,
        "simultaneous_method": "Bonferroni over 2 treatment levels x 2 regimes",
        "power_target": POWER_TARGET,
        "mde_target": SESOI,
        "max_seeds_per_cell": MAX_SEEDS_PER_CELL,
    }
    observed = {
        "level_derivation_evidence": contract.get("level_derivation_evidence"),
        "levels_days": mechanism.get("levels_days"),
        "treatment_levels_days": mechanism.get("treatment_levels_days"),
        "null_level_days": mechanism.get("null_level_days"),
        "hysteresis_tau_in_normalized_unmet": policy.get("hysteresis_tau_in_normalized_unmet"),
        "hysteresis_tau_out_normalized_unmet": policy.get("hysteresis_tau_out_normalized_unmet"),
        "contrasts": power.get("contrasts"),
        "simultaneous_method": power.get("simultaneous_method"),
        "power_target": power.get("power_target"),
        "mde_target": power.get("mde_target"),
        "max_seeds_per_cell": power.get("max_seeds_per_cell"),
    }
    mismatches = {k: {"expected": v, "observed": observed[k]}
                  for k, v in expected.items() if observed[k] != v}
    return {"passed": not mismatches, "mismatches": mismatches,
        "plan_sha256": sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()}


def _run_rows(seeds: list[int], risks, freq, impact, *, dwell: int | None,
              mode: str, constant_share: float | None = None,
              retain_trace_for_first: bool = False) -> list[dict[str, object]]:
    """Run a cell without retaining O(N * horizon) traces in memory."""
    return [episode_policy(
        seed, risks, freq, impact, dwell=dwell, mode=mode,
        constant_share=constant_share,
        retain_trace=retain_trace_for_first and index == 0,
    ) for index, seed in enumerate(seeds)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--run-role", choices=("BURNED_PREFLIGHT",), required=True)
    parser.add_argument("--replay-of", required=True)
    parser.add_argument("--seed-base", type=int, default=SEED_BASE)
    parser.add_argument("--seeds", type=int, default=len(BURNED_SEEDS))
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--output", type=Path,
                        default=Path("results/headroom/g3c_preflight_burned_v2/result.json"))
    args = parser.parse_args()
    if not args.contract.is_file():
        parser.error(f"contract does not exist: {args.contract}")
    contract = _load_contract(args.contract)
    if args.replay_of != "contention_headroom":
        parser.error("BURNED_PREFLIGHT requires --replay-of contention_headroom")
    if args.seed_base != SEED_BASE or args.seeds != len(BURNED_SEEDS):
        parser.error("BURNED_PREFLIGHT must use exactly seeds 5200001-5200016")
    if not REFERENCE.is_file():
        parser.error(f"reference artifact does not exist: {REFERENCE}")
    seeds = [args.seed_base + i for i in range(args.seeds)]
    started = time.perf_counter()
    rng = np.random.default_rng(20260805)

    cells: dict[str, dict[int, dict[str, list[dict[str, object]]]]] = {}
    for regime, (risks, freq, impact) in REGIMES.items():
        cells[regime] = {}
        for dwell in DWELL_LEVELS:
            cells[regime][dwell] = {
                "myopic": _run_rows(
                    seeds, risks, freq, impact, dwell=dwell, mode="myopic",
                    retain_trace_for_first=(dwell == 1)),
                "hysteresis": _run_rows(
                    seeds, risks, freq, impact, dwell=dwell, mode="hysteresis"),
            }
        cells[regime][1]["placebo"] = _run_rows(
            seeds, risks, freq, impact, dwell=1, mode="placebo")
        cells[regime][1]["wrong_claimant"] = _run_rows(
            seeds, risks, freq, impact, dwell=1, mode="wrong_claimant")
        for share in CONSTANT_SHARES:
            cells[regime][1].setdefault("constants", {})[str(share)] = [
                *_run_rows(seeds, risks, freq, impact, dwell=1, mode="constant",
                           constant_share=share)]
        print(f"  {regime}: {len(seeds)} seeds x dwell {DWELL_LEVELS}")

    null_identity: dict[str, object] = {"checks": 0, "mismatches": 0, "examples": []}
    for regime, (risks, freq, impact) in REGIMES.items():
        explicit = cells[regime][1]["myopic"]
        legacy = _run_rows(seeds, risks, freq, impact, dwell=None, mode="myopic")
        for exp, old in zip(explicit, legacy):
            null_identity["checks"] += 1
            if exp["scientific_payload_sha256"] != old["scientific_payload_sha256"]:
                null_identity["mismatches"] += 1
                if len(null_identity["examples"]) < 5:
                    null_identity["examples"].append({"regime": regime, "seed": exp["seed"]})
    null_identity["passed"] = null_identity["mismatches"] == 0

    contrast_results: dict[str, dict[str, object]] = {}
    for regime in REGIMES:
        null = cells[regime][1]["myopic"]
        for dwell in TREATMENT_LEVELS:
            candidate = cells[regime][dwell]["hysteresis"]
            key = f"{regime}|dwell={dwell}"
            primary_diff = np.asarray([_metric(c, PRIMARY) - _metric(n, PRIMARY)
                                       for n, c in zip(null, candidate)], dtype=float)
            guardrails = {}
            for metric, delta in GUARDRAIL_MARGINS.items():
                stat = paired_bootstrap(guardrail_harm(null, candidate, metric), args.n_boot, rng)
                stat.update({"delta": float(delta), "passes": bool(stat["ucb95"] <= delta)})
                guardrails[metric] = stat
            contrast_results[key] = {
                "regime": regime,
                "dwell": dwell,
                "primary_hysteresis_minus_myopic": paired_bootstrap(primary_diff, args.n_boot, rng),
                "guardrails": guardrails,
                "n": len(candidate),
            }

    power_cells: dict[str, dict[str, object]] = {}
    for key, contrast in contrast_results.items():
        primary = contrast["primary_hysteresis_minus_myopic"]
        targets = {"primary_worst_claimant_fill": {"sd": primary["sd"], "target": SESOI}}
        for metric, stat in contrast["guardrails"].items():
            targets[metric] = {"sd": stat["sd"], "target": stat["delta"]}
        requirements = {
            name: {"sd": float(spec["sd"]), "target": float(spec["target"]),
                   "mde_at_observed_n": mde_at_n(float(spec["sd"]), len(seeds)),
                   "required_n": required_n(float(spec["sd"]), float(spec["target"]))}
            for name, spec in targets.items()
        }
        power_cells[key] = {
            "observed_n": len(seeds),
            "required_n_max": max(v["required_n"] for v in requirements.values()),
            "within_max_budget": all(v["required_n"] <= MAX_SEEDS_PER_CELL
                                      for v in requirements.values()),
            "requirements": requirements,
        }
    power_plan = {
        "z_power": Z_POWER,
        "z_simultaneous_95": NORM_CRITICAL,
        "contrasts": 4,
        "max_seeds_per_cell": MAX_SEEDS_PER_CELL,
        "power_target": POWER_TARGET,
        "alpha": ALPHA,
        "simultaneous_method": "Bonferroni over 2 treatment levels x 2 regimes",
        "cells": power_cells,
        "powered_within_budget": all(v["within_max_budget"] for v in power_cells.values()),
    }

    trace = cells["R1r+R2r|base"][1]["myopic"][0]["trace"]
    f3_trace = check_myopic_trace(trace)  # type: ignore[arg-type]
    constant_evidence = {}
    f3_pass = True
    for regime in REGIMES:
        constants = cells[regime][1]["constants"]
        best_share = max(CONSTANT_SHARES,
                         key=lambda share: np.mean([_metric(row, PRIMARY)
                                                    for row in constants[str(share)]]))
        myopic = cells[regime][1]["myopic"]
        best = constants[str(best_share)]
        stat = paired_bootstrap(np.asarray([_metric(m, PRIMARY) - _metric(c, PRIMARY)
                                            for m, c in zip(myopic, best)]), args.n_boot, rng)
        stat["best_constant_share"] = best_share
        constant_evidence[regime] = stat
        f3_pass = f3_pass and bool(stat["lcb95"] > 0.0)

    f4_values = {}
    f5_values = {}
    for regime in REGIMES:
        myopic = cells[regime][1]["myopic"]
        placebo = cells[regime][1]["placebo"]
        wrong = cells[regime][1]["wrong_claimant"]
        f4_values[regime] = paired_bootstrap(np.asarray([_metric(m, PRIMARY) - _metric(p, PRIMARY)
                                                          for m, p in zip(myopic, placebo)]),
                                             args.n_boot, rng)
        f5_values[regime] = paired_bootstrap(np.asarray([_metric(m, PRIMARY) - _metric(w, PRIMARY)
                                                          for m, w in zip(myopic, wrong)]),
                                             args.n_boot, rng)
    f4_pass = all(v["lcb95"] > 0.0 for v in f4_values.values())
    f5_pass = all(v["lcb95"] > 0.0 for v in f5_values.values())
    f6 = validate_guardrail_margins(GUARDRAIL_MARGINS)
    f7 = _check_frozen_plan(contract)
    f8 = custody_falsifier(seeds, replay_of=args.replay_of, exclude=args.output)
    f9_pass = all(bool(stat["passes"])
                  for contrast in contrast_results.values()
                  for stat in contrast["guardrails"].values())
    f2_evidence = {
        f"{regime}|dwell={dwell}": {
            "blocked_total": sum(row["metrics"]["blocked_by_dwell"]
                                  for row in cells[regime][dwell]["myopic"]),
            "switches_total": sum(row["metrics"]["switches"]
                                   for row in cells[regime][dwell]["myopic"]),
            "null_blocked_total": sum(row["metrics"]["blocked_by_dwell"]
                                       for row in cells[regime][1]["myopic"]),
            "null_switches_total": sum(row["metrics"]["switches"]
                                        for row in cells[regime][1]["myopic"]),
        }
        for regime in REGIMES for dwell in TREATMENT_LEVELS
    }
    f2_pass = all(v["blocked_total"] > 0 and v["switches_total"] < v["null_switches_total"]
                  for v in f2_evidence.values())

    falsifiers: dict[str, object] = {
        "f1_null_arm_payload_identity": {"passed": bool(null_identity["passed"]),
                                          "evidence": null_identity},
        "f2_min_dwell_actually_binds_at_treatment_levels": {"passed": bool(f2_pass),
                                                          "evidence": f2_evidence},
        "f3_incumbent_beats_best_constant": {"passed": bool(f3_pass),
                                               "evidence": constant_evidence},
        "f4_uninformed_placebo_fails": {"passed": bool(f4_pass),
                                         "evidence": f4_values},
        "f5_wrong_claimant_fails": {"passed": bool(f5_pass), "evidence": f5_values},
        "f6_guardrails_use_signed_margins": {"passed": bool(f6["passed"]),
                                               "evidence": f6},
        "f7_power_frozen_before_execution": {"passed": bool(f7["passed"]),
                                               "evidence": {"plan": f7,
                                                             "myopic_trace": f3_trace}},
        "f8_no_fresh_seeds_before_authority": f8,
        "f9_no_gain_by_abandonment": {"passed": bool(f9_pass),
                                       "evidence": {"contrasts": contrast_results,
                                                     "margins": GUARDRAIL_MARGINS}},
    }
    required = [v for v in falsifiers.values()
                if isinstance(v, dict) and not v.get("not_applicable")]
    all_passed = all(bool(v.get("passed")) for v in required)
    falsifiers["all_passed"] = all_passed
    falsifiers["not_applicable"] = sorted(
        k for k, v in falsifiers.items()
        if k != "all_passed" and isinstance(v, dict) and v.get("not_applicable"))

    if not all_passed:
        if not null_identity["passed"] or not f2_pass or not f3_pass or not f6["passed"] or not f7["passed"]:
            verdict = "STOP_G3C_INSTRUMENT_INVALID"
        elif not f4_pass or not f5_pass:
            verdict = "STOP_G3C_DIRECTIONALITY_NOT_ESTABLISHED"
        elif not f9_pass:
            verdict = "STOP_G3C_GUARDRAIL"
        else:
            verdict = "STOP_G3C_INSTRUMENT_INVALID"
    elif not power_plan["powered_within_budget"]:
        verdict = "STOP_G3C_UNDERPOWERED"
    else:
        verdict = "G3C_PREFLIGHT_POWER_SUFFICIENT_NO_FRESH_SEEDS"

    payload = {
        "schema_version": "g3c_temporal_coupling_preflight_v2",
        "claim_status": verdict,
        "scientific_verdict": verdict,
        "scope": "BURNED_PREFLIGHT_NO_FRESH_SEEDS_NO_ADJUDICATION_NO_LEARNER",
        "run_role": args.run_role,
        "replay_of": args.replay_of,
        "execution_authorization": "PI explicit burned-only authorization in G3c v2 amendment",
        "primary_metric": PRIMARY,
        "primary_contrast": "hysteresis_minus_myopic",
        "sesoi": SESOI,
        "guardrail_margins": GUARDRAIL_MARGINS,
        "seeds": seeds,
        "regimes": list(REGIMES),
        "dwell_levels": list(DWELL_LEVELS),
        "treatment_levels": list(TREATMENT_LEVELS),
        "hysteresis": {"tau_in": TAU_IN, "tau_out": TAU_OUT},
        "level_derivation_evidence": contract.get("level_derivation_evidence"),
        "constant_grid": list(CONSTANT_SHARES),
        "step_hours": STEP_HOURS,
        "horizon_weeks": HORIZON_WEEKS,
        "power_plan": power_plan,
        "cells": {
            regime: {
                str(dwell): {
                    mode: (_cell_summary(rows) if mode != "constants" else {
                        share: _cell_summary(share_rows)
                        for share, share_rows in rows.items()
                    })
                    for mode, rows in modes.items()
                }
                for dwell, modes in by_dwell.items()
            }
            for regime, by_dwell in cells.items()
        },
        "contrasts": contrast_results,
        "falsifiers": falsifiers,
        "module_manifest": module_manifest(MODULES, script=__file__),
        "scope_caveat": "G3c temporal coupling on split_v1 CSSU A/B; no E* per-node expansion and no learner.",
        "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(
        payload,
        args.output,
        contract=args.contract,
        reference=REFERENCE,
        stamp_extra={"run_role": args.run_role, "replay_of": args.replay_of,
                     "module_manifest": payload["module_manifest"]},
    )
    print(f"\nverdict: {verdict}")
    for name, value in falsifiers.items():
        if name in {"all_passed", "not_applicable"} or not isinstance(value, dict):
            continue
        label = "NO APLICA" if value.get("not_applicable") else ("PASA" if value["passed"] else "FALLA")
        print(f"  {name}: {label}")
    for key, cell in power_cells.items():
        print(f"  {key}: required_n_max={cell['required_n_max']} <= {MAX_SEEDS_PER_CELL}: "
              f"{cell['within_max_budget']}")
    print(f"sealed: {args.output} ({digest[:16]}…)")
    return 0 if verdict == "G3C_PREFLIGHT_POWER_SUFFICIENT_NO_FRESH_SEEDS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
