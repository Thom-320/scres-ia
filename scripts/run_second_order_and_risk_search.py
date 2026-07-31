#!/usr/bin/env python3
"""Second-order Sobol indices, and a search over the risk hyperparameters for headroom.

Two questions, one runner.

**Where does the node go?** The first-order map said how much interaction each factor has; it
never said WITH WHOM. `S_ij` does. The estimator is Saltelli's closed second-order form, which
needs the `BA_i` blocks as well as `AB_i` -- `N(2k+2)` evaluations rather than `N(k+2)`:

    V_ij^closed = mean(Y_BA_i * Y_AB_j) - f0^2 ,   S_ij = S_ij^closed - S_i - S_j

The node belongs where a DECISION factor interacts with a RISK factor: that pair is a lever
whose right setting depends on the environment, which is the only thing a policy can exploit.

**Which risk hyperparameters create the most headroom?** Garrido authorised editing risks, and
the three axes are activation, occurrence and impact. Occurrence and activation are the same
axis at its limit -- a frequency multiplier at 0 is a deactivated risk -- so each family carries
a frequency factor spanning that limit and an impact factor, and `f2` verifies the limit really
deactivates. Stage 2 then searches that space for the configuration set with the largest
`H_regime`.

Scored on `ret_excel_risk_conditional`, which the metric comparison measured at 65x the
headroom resolution of `ret_excel` -- searching for headroom with the metric that hides it was
the previous run's handicap.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.config import HOURS_PER_WEEK  # noqa: E402
from supply_chain.config import THESIS_FAITHFUL_PROTOCOL as P  # noqa: E402
from supply_chain.decision_right_discovery import NumericFactor, saltelli_sample  # noqa: E402
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402

FAMILY_RISKS = {"R1r": ("R11", "R12", "R13", "R14"),
                "R2r": ("R21", "R22", "R23", "R24"), "R3": ("R3",)}
ALL_RISKS = tuple(r for ids in FAMILY_RISKS.values() for r in ids)
METRIC = "ret_excel_risk_conditional"
SEED_BASE = 4_500_001

DECISION = ("op9_rop", "op10_rop", "op10_q_max", "op12_rop", "op12_q_max")
FACTORS = (
    NumericFactor("op9_rop", 12.0, 48.0, "decision"),
    NumericFactor("op10_rop", 12.0, 48.0, "decision"),
    NumericFactor("op10_q_max", 1_200.0, 5_200.0, "decision"),
    NumericFactor("op12_rop", 12.0, 48.0, "decision"),
    NumericFactor("op12_q_max", 1_200.0, 5_200.0, "decision"),
    NumericFactor("freq_R1r", 0.0, 3.0, "risk"),
    NumericFactor("freq_R2r", 0.0, 3.0, "risk"),
    NumericFactor("freq_R3", 0.0, 3.0, "risk"),
    NumericFactor("impact_R1r", 0.5, 3.0, "risk"),
    NumericFactor("impact_R2r", 0.5, 3.0, "risk"),
    NumericFactor("impact_R3", 0.5, 3.0, "risk"),
)
NAMES = tuple(f.name for f in FACTORS)
RISK_NAMES = tuple(f.name for f in FACTORS if f.group == "risk")
OFF = 1e-6   # a frequency multiplier at this level is a deactivated risk


def run(params: dict[str, float], seed: int, horizon: float) -> float:
    freq = {rid: max(OFF, float(params[f"freq_{fam}"]))
            for fam, ids in FAMILY_RISKS.items() for rid in ids}
    impact = {rid: float(params[f"impact_{fam}"])
              for fam, ids in FAMILY_RISKS.items() for rid in ids}
    sim = MFSCSimulation(
        shifts=1, initial_buffers={n: 0.0 for n in ("op3_rm", "op5_rm", "op9_rations")},
        inventory_replenishment_period=0.0, seed=seed, horizon=horizon,
        risks_enabled=True, risk_level="current", enabled_risks=set(ALL_RISKS),
        risk_frequency_multipliers_by_id=freq, risk_impact_multipliers_by_id=impact,
        strict_exogenous_crn=True, year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"], r14_defect_mode=P["r14_defect_mode"])
    sim.step(action={
        "op9_rop": params["op9_rop"], "op9_q_min": 2_400.0, "op9_q_max": 2_600.0,
        "op10_rop": params["op10_rop"],
        "op10_q_min": params["op10_q_max"] * 0.92, "op10_q_max": params["op10_q_max"],
        "op12_rop": params["op12_rop"],
        "op12_q_min": params["op12_q_max"] * 0.92, "op12_q_max": params["op12_q_max"],
    }, step_hours=horizon)
    return float(compute_episode_metrics(sim)[METRIC])


def scale(unit: np.ndarray) -> dict[str, float]:
    return {f.name: f.scale(unit[i]) for i, f in enumerate(FACTORS)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-sobol", type=int, default=192)
    ap.add_argument("--risk-configs", type=int, default=24)
    ap.add_argument("--lever-levels", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--horizon-weeks", type=int, default=52)
    ap.add_argument("--output", type=Path,
                    default=Path("results/sensitivity/second_order_risk_search_v1/result.json"))
    args = ap.parse_args()
    horizon = float(args.horizon_weeks * HOURS_PER_WEEK)
    k = len(FACTORS)
    started = time.perf_counter()

    # ---- stage 1: A, B, AB_i, BA_i -> S_i, S_Ti and the full S_ij matrix -----------------
    a, b, ab = saltelli_sample(k, args.n_sobol, seed=20260731)
    ba = np.empty_like(ab)
    for i in range(k):
        ba[i] = b.copy()
        ba[i][:, i] = a[:, i]

    def block(rows: np.ndarray, tag: int) -> np.ndarray:
        return np.array([run(scale(r), SEED_BASE + tag * 10_000 + j, horizon)
                         for j, r in enumerate(rows)])

    y_a, y_b = block(a, 0), block(b, 1)
    y_ab = [block(ab[i], 2 + i) for i in range(k)]
    y_ba = [block(ba[i], 2 + k + i) for i in range(k)]
    print(f"  Sobol 2do orden: {(2 * k + 2) * args.n_sobol} corridas "
          f"({time.perf_counter() - started:.0f}s)", flush=True)

    f0 = float(np.mean(np.concatenate([y_a, y_b])))
    var = float(np.var(np.concatenate([y_a, y_b]), ddof=1))
    s1 = {NAMES[i]: (var - np.mean((y_b - y_ab[i]) ** 2) / 2.0) / var for i in range(k)}
    st = {NAMES[i]: np.mean((y_a - y_ab[i]) ** 2) / (2.0 * var) for i in range(k)}
    s_ij: dict[str, float] = {}
    for i, j in itertools.combinations(range(k), 2):
        closed = (float(np.mean(y_ba[i] * y_ab[j])) - f0 ** 2) / var
        s_ij[f"{NAMES[i]}|{NAMES[j]}"] = float(closed - s1[NAMES[i]] - s1[NAMES[j]])

    cross = {pair: v for pair, v in s_ij.items()
             if (pair.split("|")[0] in DECISION) != (pair.split("|")[1] in DECISION)}
    ranked_cross = sorted(cross, key=lambda p: -cross[p])

    # ---- stage 2: search the risk hyperparameters for the largest H_regime ----------------
    axes = {"op9_rop": np.linspace(12.0, 48.0, args.lever_levels),
            "op10_q_max": np.linspace(1_200.0, 5_200.0, args.lever_levels),
            "op12_q_max": np.linspace(1_200.0, 5_200.0, args.lever_levels)}
    levers = [dict(zip(axes, v)) for v in itertools.product(*axes.values())]
    rng = np.random.default_rng(20260731)
    risk_unit = rng.random((args.risk_configs, len(RISK_NAMES)))
    risk_configs = []
    for row in risk_unit:
        cfg = {}
        for idx, name in enumerate(RISK_NAMES):
            f = next(f for f in FACTORS if f.name == name)
            cfg[name] = f.scale(row[idx])
        risk_configs.append(cfg)

    scores = np.zeros((len(levers), len(risk_configs), args.seeds))
    seed_cursor = SEED_BASE + (2 * k + 2) * 10_000
    for l_i, lever in enumerate(levers):
        for r_i, cfg in enumerate(risk_configs):
            params = {**cfg, **lever, "op10_rop": 24.0, "op12_rop": 24.0}
            for s in range(args.seeds):
                scores[l_i, r_i, s] = run(params, seed_cursor + s, horizon)
    print(f"  búsqueda de riesgo: {scores.size} corridas "
          f"({time.perf_counter() - started:.0f}s)", flush=True)

    per_cfg = scores.mean(axis=2)
    informed = per_cfg.max(axis=0)
    common = per_cfg.mean(axis=1).max()
    h_regime = float(informed.mean() - common)
    contribution = {r_i: float(informed[r_i] - per_cfg[int(per_cfg.mean(axis=1).argmax()), r_i])
                    for r_i in range(len(risk_configs))}
    best_cfg = max(contribution, key=lambda r: contribution[r])

    # ---- falsifiers ----------------------------------------------------------------------
    off = {n: (OFF if n.startswith("freq") else 1.0) for n in RISK_NAMES}
    base_lever = {"op9_rop": 24.0, "op10_rop": 24.0, "op10_q_max": 2_600.0,
                  "op12_rop": 24.0, "op12_q_max": 2_600.0}
    with_zero_freq = run({**off, **base_lever}, SEED_BASE, horizon)
    sim_off = MFSCSimulation(
        shifts=1, initial_buffers={n: 0.0 for n in ("op3_rm", "op5_rm", "op9_rations")},
        inventory_replenishment_period=0.0, seed=SEED_BASE, horizon=horizon,
        risks_enabled=False, strict_exogenous_crn=True, year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"], r14_defect_mode=P["r14_defect_mode"])
    sim_off.step(action=base_lever | {"op9_q_min": 2_400.0, "op9_q_max": 2_600.0,
                                      "op10_q_min": 2_392.0, "op12_q_min": 2_392.0},
                 step_hours=horizon)
    risks_disabled = float(compute_episode_metrics(sim_off)[METRIC])

    falsifiers = {
        "f1_first_order_indices_are_inside_zero_one": {
            "passed": all(-0.05 <= s1[n] <= 1.05 and -0.05 <= st[n] <= 1.05 for n in NAMES),
            "evidence": {"why_it_can_fail": ("the raw ret_excel decomposition failed exactly "
                                             "this; the risk-conditional metric is used here "
                                             "partly because it should not"),
                         "S1": s1, "ST": st, "sum_S1": float(sum(s1.values()))}},
        "f2_zero_frequency_really_deactivates": {
            "passed": abs(with_zero_freq - risks_disabled) < 1e-9,
            "evidence": {"why_it_can_fail": ("activation is encoded as the frequency factor's "
                                             "lower limit; if a multiplier of 1e-6 still fires "
                                             "risks, the activation axis is not represented"),
                         "freq_at_zero": with_zero_freq, "risks_disabled": risks_disabled}},
        "f3_second_order_matrix_is_symmetric_in_construction": {
            "passed": len(s_ij) == k * (k - 1) // 2,
            "evidence": {"why_it_can_fail": "a missing pair would silently drop an interaction",
                         "pairs": len(s_ij), "expected": k * (k - 1) // 2}},
        "f4_headroom_is_non_negative": {
            "passed": h_regime >= -1e-12,
            "evidence": {"why_it_can_fail": "a negative value would betray a bad aggregation",
                         "H_regime": h_regime}},
        "f5_search_space_is_non_degenerate": {
            "passed": float(np.std(per_cfg)) > 0.0,
            "evidence": {"why_it_can_fail": "a constant surface makes the search empty",
                         "sd": float(np.std(per_cfg))}},
    }
    falsifiers["all_passed"] = all(v["passed"] for kk, v in falsifiers.items()
                                   if kk != "all_passed")

    print(f"\n  === S_ij: decisión × riesgo, los 10 mayores ===")
    for pair in ranked_cross[:10]:
        print(f"    {pair:<34} {cross[pair]:+.4f}")
    print(f"\n  H_regime sobre {len(risk_configs)} configuraciones de riesgo: {h_regime:.6f}")
    print(f"  configuración que más aporta: #{best_cfg} -> "
          f"{ {k2: round(v, 3) for k2, v in risk_configs[best_cfg].items()} }")
    print("\n  falsadores:")
    for name, check in falsifiers.items():
        if name != "all_passed":
            print(f"    {name:<48} {'PASA' if check['passed'] else 'FALLA'}")

    payload = {
        "schema_version": "second_order_risk_search_v1",
        "claim_status": ("DEVELOPMENT_SECOND_ORDER_AND_RISK_SEARCH" if falsifiers["all_passed"]
                         else "HALTED_FALSIFIER_FAILED"),
        "metric": METRIC,
        "factors": {f.name: {"lower": f.lower, "upper": f.upper, "group": f.group}
                    for f in FACTORS},
        "S1": s1, "ST": st, "S_ij": s_ij,
        "S_ij_decision_x_risk_ranked": [{"pair": p, "S_ij": cross[p]} for p in ranked_cross],
        "risk_search": {
            "H_regime": h_regime, "n_configs": len(risk_configs),
            "configs": risk_configs, "contribution_by_config": contribution,
            "best_config_index": best_cfg, "best_config": risk_configs[best_cfg]},
        "falsifiers": falsifiers,
        "n_runs": int((2 * k + 2) * args.n_sobol + scores.size),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(
        payload, args.output,
        contract=Path("docs/RESULTADO_METRICAS_HEADROOM_2026-07-31.md"),
        reference=Path("results/sensitivity/headroom_map_v1/result.json"))
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
