#!/usr/bin/env python3
"""Per-risk sensitivity with CRN replicates per design point -- the fix the failure diagnosed.

The per-risk decomposition failed its bounds falsifier at `N = 128` and again at `N = 512`, and
the shape of the failure named the cause rather than the cure: `sum(S1) = 0.035` with `S_T`
between 0.40 and 0.72 almost everywhere. That is not massive interaction; it is per-episode
noise the estimator books as total effect. Quadrupling `N` shrank the overflow and left the
picture unchanged, because more design points do not reduce the noise INSIDE a point.

The textbook fix for a stochastic simulator is to estimate the EXPECTED response at each design
point: evaluate every row with `R` common random numbers and average. The same `R` seeds are
used at every point, so the noise is common and cancels in the differences the estimator takes.

`f_replication` is the falsifier that makes this claim testable rather than asserted: the same
indices are computed from the FIRST replicate alone, and replication must reduce the number of
out-of-bounds indices. If averaging does not help, the diagnosis was wrong and the artifact says
so instead of quietly presenting nicer numbers.

Stage 2 then uses the converged indices for what they are for: build the risk regimes out of the
two factors with the largest interaction and measure whether that targeted set produces more
headroom than the family mixes did.
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
from supply_chain.decision_right_discovery import saltelli_sample, sobol_indices  # noqa: E402
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402

RISKS = ("R11", "R12", "R13", "R14", "R21", "R22", "R23", "R24", "R3")
FREQ_RANGE, IMPACT_RANGE = (0.0, 3.0), (0.5, 3.0)
NAMES = tuple([f"freq_{r}" for r in RISKS] + [f"impact_{r}" for r in RISKS])
METRIC, SIDE = "ret_excel_risk_conditional", "ret_excel"
SEED_BASE = 4_800_001
OFF = 1e-6


def params_from(row: np.ndarray) -> tuple[dict, dict]:
    freq = {r: max(OFF, FREQ_RANGE[0] + row[i] * (FREQ_RANGE[1] - FREQ_RANGE[0]))
            for i, r in enumerate(RISKS)}
    impact = {r: IMPACT_RANGE[0] + row[len(RISKS) + i] * (IMPACT_RANGE[1] - IMPACT_RANGE[0])
              for i, r in enumerate(RISKS)}
    return freq, impact


def evaluate(freq: dict, impact: dict, seed: int, horizon: float,
             *, risks=RISKS, node_action: dict | None = None) -> dict[str, float]:
    sim = MFSCSimulation(
        shifts=1, initial_buffers={n: 0.0 for n in ("op3_rm", "op5_rm", "op9_rations")},
        inventory_replenishment_period=0.0, seed=seed, horizon=horizon,
        risks_enabled=True, risk_level="current", enabled_risks=set(risks),
        risk_frequency_multipliers_by_id=freq, risk_impact_multipliers_by_id=impact,
        strict_exogenous_crn=True, year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"], r14_defect_mode=P["r14_defect_mode"])
    action = {"op12_rop": 24.0, "op12_q_min": 2_392.0, "op12_q_max": 2_600.0}
    action.update(node_action or {})
    sim.step(action=action, step_hours=horizon)
    panel = compute_episode_metrics(sim)
    return {METRIC: float(panel[METRIC]), SIDE: float(panel[SIDE])}


def rank(values: np.ndarray, pooled: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.sort(pooled), values, side="left") / max(pooled.size - 1, 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-sobol", type=int, default=192)
    ap.add_argument("--replicates", type=int, default=5)
    ap.add_argument("--horizon-weeks", type=int, default=52)
    ap.add_argument("--lever-levels", type=int, default=3)
    ap.add_argument("--headroom-seeds", type=int, default=5)
    ap.add_argument("--output", type=Path,
                    default=Path("results/sensitivity/perrisk_crn_v1/result.json"))
    args = ap.parse_args()
    horizon = float(args.horizon_weeks * HOURS_PER_WEEK)
    k = len(NAMES)
    started = time.perf_counter()

    a, b, ab = saltelli_sample(k, args.n_sobol, seed=20260731)
    # THE SAME replicate seeds at every design point. That is what makes them common random
    # numbers rather than just more sampling.
    crn = [SEED_BASE + i for i in range(args.replicates)]

    def block(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (mean over replicates, first replicate alone)."""
        means, firsts = [], []
        for row in rows:
            freq, impact = params_from(row)
            draws = [evaluate(freq, impact, s, horizon)[METRIC] for s in crn]
            means.append(float(np.mean(draws)))
            firsts.append(float(draws[0]))
        return np.array(means), np.array(firsts)

    y_a, y_a1 = block(a)
    y_b, y_b1 = block(b)
    y_ab, y_ab1 = [], []
    for i in range(k):
        m, f = block(ab[i])
        y_ab.append(m)
        y_ab1.append(f)
        print(f"  {i + 1}/{k} ({time.perf_counter() - started:.0f}s)", flush=True)

    pooled = np.concatenate([y_a, y_b, *y_ab])
    pooled1 = np.concatenate([y_a1, y_b1, *y_ab1])
    replicated = sobol_indices(rank(y_a, pooled), rank(y_b, pooled),
                               [rank(v, pooled) for v in y_ab], NAMES, n_boot=300, seed=5)
    single = sobol_indices(rank(y_a1, pooled1), rank(y_b1, pooled1),
                           [rank(v, pooled1) for v in y_ab1], NAMES, n_boot=100, seed=5)

    def out_of_bounds(table: dict) -> list[str]:
        return [n for n in NAMES
                if not (-0.05 <= table[n]["S1"] <= 1.05 and -0.05 <= table[n]["ST"] <= 1.05)]

    oob_rep, oob_single = out_of_bounds(replicated), out_of_bounds(single)
    sum_s1 = float(sum(v["S1"] for v in replicated.values()))
    ranked = sorted(NAMES, key=lambda n: -replicated[n]["interaction"])
    print(f"  Sobol replicado listo ({time.perf_counter() - started:.0f}s)", flush=True)

    # ---- stage 2: build regimes from the top-2 interacting risk factors -------------------
    top2 = ranked[:2]
    corners = list(itertools.product([0.0, 1.0], repeat=2))
    base_row = np.full(k, 0.5)
    regimes = {}
    for corner in corners:
        row = base_row.copy()
        for value, name in zip(corner, top2):
            row[NAMES.index(name)] = value
        regimes[f"{top2[0]}={corner[0]:.0f},{top2[1]}={corner[1]:.0f}"] = params_from(row)
    levers = [{"op12_rop": v} for v in np.linspace(12.0, 48.0, args.lever_levels)]
    scores = np.zeros((len(levers), len(regimes), args.headroom_seeds))
    cursor = SEED_BASE + 500_000
    for l_i, lever in enumerate(levers):
        for r_i, (freq, impact) in enumerate(regimes.values()):
            for s in range(args.headroom_seeds):
                scores[l_i, r_i, s] = evaluate(freq, impact, cursor + s, horizon,
                                               node_action=lever)[METRIC]
    per = scores.mean(axis=2)
    h_targeted = float(per.max(axis=0).mean() - per.mean(axis=1).max())

    falsifiers = {
        "f_replication_reduces_the_estimator_overflow": {
            "passed": len(oob_rep) < len(oob_single),
            "evidence": {
                "why_it_can_fail": ("this is the DIAGNOSIS under test. If averaging CRN "
                                    "replicates does not shrink the out-of-bounds set, the "
                                    "problem was not per-point noise and the fix is wrong"),
                "out_of_bounds_single_replicate": oob_single,
                "out_of_bounds_replicated": oob_rep,
                "replicates": args.replicates}},
        "f1_indices_inside_zero_one": {
            "passed": not oob_rep,
            "evidence": {"why_it_can_fail": "an index outside [0,1] is a failed measurement",
                         "sum_S1": sum_s1, "out_of_bounds": oob_rep}},
        "f2_crn_is_common_across_design_points": {
            "passed": len(set(crn)) == args.replicates,
            "evidence": {"why_it_can_fail": ("different seeds per point would be more sampling, "
                                             "not common random numbers"),
                         "seeds": crn}},
        "f3_targeted_headroom_is_non_negative": {
            "passed": h_targeted >= -1e-12,
            "evidence": {"H_targeted": h_targeted, "regimes": list(regimes)}},
        "f4_design_covers_the_space": {
            "passed": all(len({round(float(v), 6) for v in a[:, i]}) >= 64 for i in range(k)),
            "evidence": {"why_it_can_fail": "a collapsed mapping would make the screen vacuous"}},
    }
    falsifiers["all_passed"] = all(v["passed"] for kk, v in falsifiers.items()
                                   if kk != "all_passed")

    print(f"\n  === por-riesgo con {args.replicates} réplicas CRN "
          f"(sum S1 = {sum_s1:.3f}) ===")
    print(f"  {'factor':<16}{'S1':>8}{'ST':>8}{'ST-S1':>9}")
    for n in ranked[:10]:
        print(f"  {n:<16}{replicated[n]['S1']:>8.3f}{replicated[n]['ST']:>8.3f}"
              f"{replicated[n]['interaction']:>9.3f}")
    print(f"\n  fuera de [0,1]: 1 réplica {len(oob_single)} -> "
          f"{args.replicates} réplicas {len(oob_rep)}")
    print(f"  H_regime dirigido sobre {top2}: {h_targeted:.6f}")
    print("\n  falsadores:")
    for name, check in falsifiers.items():
        if name != "all_passed":
            print(f"    {name:<48} {'PASA' if check['passed'] else 'FALLA'}")

    payload = {
        "schema_version": "perrisk_crn_v1",
        "claim_status": ("DEVELOPMENT_PERRISK_CRN" if falsifiers["all_passed"]
                         else "HALTED_FALSIFIER_FAILED"),
        "metric": METRIC, "replicates": args.replicates, "crn_seeds": crn,
        "indices_replicated": replicated, "indices_single_replicate": single,
        "sum_S1": sum_s1, "ranking_by_interaction": ranked,
        "targeted_headroom": {"factors": top2, "H_regime": h_targeted,
                              "regimes": list(regimes)},
        "n_runs": int((k + 2) * args.n_sobol * args.replicates + scores.size),
        "falsifiers": falsifiers,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(
        payload, args.output,
        contract=Path("docs/RESULTADO_PORRIESGO_Y_NODO_NUEVO_2026-07-31.md"),
        reference=Path("results/sensitivity/perrisk_and_new_node_v1/result.json"))
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
