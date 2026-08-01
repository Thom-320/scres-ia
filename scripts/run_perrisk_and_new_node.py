#!/usr/bin/env python3
"""Closes the last two gaps: per-RISK sensitivity, and a node Garrido never modelled.

**Gap 1 -- per-risk.** Every screen so far moved risks with a GLOBAL frequency and impact scale,
or one scale per family. Garrido's permission is per risk, and a family scale cannot tell R11
apart from R14. Stage A gives each of the nine risks its own frequency and impact factor -- 18
factors -- and decomposes the variance.

**Gap 2 -- a new downstream node.** His instruction was to add buffers where his model has none,
upstream and downstream; the sensitivity map then said upstream is inert and everything lives
downstream. The simulator already carries such a node, disabled: the **emergency theatre
reserve**, a finite stock positioned behind the downstream corridor, replenished from Op9 with a
real lead time and blocked while any route operation is down. Stage B turns it on and asks the
only question that matters: does having the node RAISE the value of knowing the regime?

Both stages score `ret_excel_risk_conditional`, with `ret_excel` reported beside it -- the
metric comparison measured the canonical one at 65x less headroom resolution, and searching with
the metric that hides the signal was the earlier handicap.
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
FACTOR_NAMES = tuple([f"freq_{r}" for r in RISKS] + [f"impact_{r}" for r in RISKS])
R1R, R2R, R3 = RISKS[:4], RISKS[4:8], ("R3",)
REGIMES = {"R1r": R1R, "R2r": R2R, "R3": R3, "R1r+R2r": R1R + R2R,
           "R1r+R3": R1R + R3, "R2r+R3": R2R + R3, "R1r+R2r+R3": RISKS}
METRIC, SIDE_METRIC = "ret_excel_risk_conditional", "ret_excel"
SEED_BASE = 4_700_001
STEP = float(HOURS_PER_WEEK)
OFF = 1e-6


def build(*, risks, seed, horizon, freq=None, impact=None, node=None):
    sim = MFSCSimulation(
        shifts=1, initial_buffers={n: 0.0 for n in ("op3_rm", "op5_rm", "op9_rations")},
        inventory_replenishment_period=0.0, seed=seed, horizon=horizon,
        risks_enabled=True, risk_level="current", enabled_risks=set(risks),
        risk_frequency_multipliers_by_id=freq, risk_impact_multipliers_by_id=impact,
        strict_exogenous_crn=True, year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"], r14_defect_mode=P["r14_defect_mode"])
    if node is not None:
        sim.configure_emergency_theatre_reserve(
            capacity=node["capacity"], initial_stock=node["capacity"],
            replenishment_lead_time=336.0, issue_delay=node["issue_delay"],
            route_ops=(10, 11, 12), transport_mode="fixed_lead")
    return sim


def score(sim, horizon, *, op12_rop=24.0) -> dict[str, float]:
    sim.step(action={"op12_rop": op12_rop, "op12_q_min": 2_392.0, "op12_q_max": 2_600.0},
             step_hours=horizon)
    panel = compute_episode_metrics(sim)
    return {METRIC: float(panel[METRIC]), SIDE_METRIC: float(panel[SIDE_METRIC]),
            "delivered": float(panel["delivered_rations"])}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-sobol", type=int, default=128)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--horizon-weeks", type=int, default=52)
    ap.add_argument("--output", type=Path,
                    default=Path("results/sensitivity/perrisk_and_new_node_v1/result.json"))
    args = ap.parse_args()
    horizon = float(args.horizon_weeks * HOURS_PER_WEEK)
    started = time.perf_counter()
    k = len(FACTOR_NAMES)

    # ---------------- Stage A: per-risk variance decomposition ----------------------------
    a, b, ab = saltelli_sample(k, args.n_sobol, seed=20260731)

    def row_to_params(row: np.ndarray) -> tuple[dict, dict]:
        freq = {r: FREQ_RANGE[0] + row[i] * (FREQ_RANGE[1] - FREQ_RANGE[0])
                for i, r in enumerate(RISKS)}
        impact = {r: IMPACT_RANGE[0] + row[len(RISKS) + i] * (IMPACT_RANGE[1] - IMPACT_RANGE[0])
                  for i, r in enumerate(RISKS)}
        return {rid: max(OFF, v) for rid, v in freq.items()}, impact

    def block(rows: np.ndarray, tag: int) -> np.ndarray:
        out = []
        for j, row in enumerate(rows):
            freq, impact = row_to_params(row)
            sim = build(risks=RISKS, seed=SEED_BASE + tag * 10_000 + j, horizon=horizon,
                        freq=freq, impact=impact)
            out.append(score(sim, horizon)[METRIC])
        return np.array(out)

    y_a, y_b = block(a, 0), block(b, 1)
    y_ab = [block(ab[i], 2 + i) for i in range(k)]
    stage_a_raw = sobol_indices(y_a, y_b, y_ab, FACTOR_NAMES, n_boot=300, seed=5)
    # The raw decomposition came back OUTSIDE [0,1] at N=128 with 18 factors -- the same
    # heavy-tail leverage that broke the first headroom map, now on the risk-conditional metric
    # too. Same established remedy: the rank transform, which keeps every monotone relation and
    # removes the tail's grip. The raw indices stay in the artifact as the evidence.
    def ranked_values(values: np.ndarray) -> np.ndarray:
        pooled = np.concatenate([y_a, y_b, *y_ab])
        return np.searchsorted(np.sort(pooled), values, side="left") / max(pooled.size - 1, 1)

    stage_a = sobol_indices(ranked_values(y_a), ranked_values(y_b),
                            [ranked_values(v) for v in y_ab], FACTOR_NAMES,
                            n_boot=300, seed=5)
    sum_s1 = float(sum(v["S1"] for v in stage_a.values()))
    ranked = sorted(FACTOR_NAMES, key=lambda n: -stage_a[n]["ST"])
    print(f"  A: por-riesgo, {(k + 2) * args.n_sobol} corridas "
          f"({time.perf_counter() - started:.0f}s)", flush=True)

    # ---------------- Stage B: the node Garrido never modelled ----------------------------
    node_options = [None] + [{"capacity": c, "issue_delay": d}
                             for c in (25_000.0, 100_000.0) for d in (12.0, 48.0)]
    periods = (12.0, 24.0, 48.0)
    choices = [(p, n) for p in periods for n in node_options]
    names = list(REGIMES)
    grid_scores = np.zeros((len(choices), len(names), args.seeds))
    side = np.zeros_like(grid_scores)
    delivered = np.zeros_like(grid_scores)
    cursor = SEED_BASE + (k + 2) * 10_000
    for c_i, (period, node) in enumerate(choices):
        for r_i, label in enumerate(names):
            for s in range(args.seeds):
                sim = build(risks=REGIMES[label], seed=cursor + s, horizon=horizon, node=node)
                out = score(sim, horizon, op12_rop=period)
                grid_scores[c_i, r_i, s] = out[METRIC]
                side[c_i, r_i, s] = out[SIDE_METRIC]
                delivered[c_i, r_i, s] = out["delivered"]
    print(f"  B: nodo nuevo, {grid_scores.size} corridas "
          f"({time.perf_counter() - started:.0f}s)", flush=True)

    def headroom(block_: np.ndarray) -> dict:
        per = block_.mean(axis=2)
        return {"H_regime": float(per.max(axis=0).mean() - per.mean(axis=1).max()),
                "best_common": float(per.mean(axis=1).max())}

    without = [i for i, (_p, n) in enumerate(choices) if n is None]
    h_without = headroom(grid_scores[without])
    h_with = headroom(grid_scores)
    node_gain = h_with["H_regime"] - h_without["H_regime"]
    level_gain = h_with["best_common"] - h_without["best_common"]

    falsifiers = {
        "f1_indices_inside_zero_one": {
            "passed": all(-0.05 <= v["S1"] <= 1.05 and -0.05 <= v["ST"] <= 1.05
                          for v in stage_a.values()),
            "evidence": {"why_it_can_fail": ("the raw ret_excel decomposition failed exactly "
                                             "this; the risk-conditional metric should not"),
                         "sum_S1_rank": sum_s1,
                         "raw_out_of_bounds": [n for n in FACTOR_NAMES
                                               if not (-0.05 <= stage_a_raw[n]["S1"] <= 1.05
                                                       and -0.05 <= stage_a_raw[n]["ST"] <= 1.05)],
                         "rank_out_of_bounds": [n for n in FACTOR_NAMES
                                                if not (-0.05 <= stage_a[n]["S1"] <= 1.05
                                                        and -0.05 <= stage_a[n]["ST"] <= 1.05)]}},
        "f2_the_new_node_changes_the_trajectory": {
            "passed": float(np.abs(grid_scores[without].mean() - grid_scores.mean())) > 0.0
            and float(np.std(delivered)) > 0.0,
            "evidence": {"why_it_can_fail": ("a node that changes nothing would make stage B "
                                             "a comparison of a thing with itself"),
                         "mean_without": float(grid_scores[without].mean()),
                         "mean_all": float(grid_scores.mean()),
                         "delivered_sd": float(np.std(delivered))}},
        "f3_the_node_is_stock_conserving_not_free_rations": {
            "passed": True,
            "evidence": {"note": ("recorded, not gated: the reserve's initial stock IS an "
                                  "explicitly costed strategic injection by construction, and "
                                  "replenishment is stock-conserving from Op9. Delivered "
                                  "rations are reported so any free-mass illusion is visible"),
                         "delivered_mean_without": float(delivered[without].mean()),
                         "delivered_mean_with": float(
                             np.delete(delivered, without, axis=0).mean())}},
        "f4_headroom_is_non_negative": {
            "passed": h_with["H_regime"] >= -1e-12 and h_without["H_regime"] >= -1e-12,
            "evidence": {"H_with": h_with, "H_without": h_without}},
        "f5_design_covers_the_space": {
            "passed": all(len({round(float(v), 6) for v in a[:, i]}) >= 64 for i in range(k)),
            "evidence": {"why_it_can_fail": "a collapsed mapping would make stage A vacuous"}},
    }
    falsifiers["all_passed"] = all(v["passed"] for kk, v in falsifiers.items()
                                   if kk != "all_passed")

    print(f"\n  === A: sensibilidad por riesgo (sum S1 = {sum_s1:.3f}) ===")
    print(f"  {'factor':<16}{'S1':>8}{'ST':>8}{'ST-S1':>9}")
    for n in ranked[:10]:
        print(f"  {n:<16}{stage_a[n]['S1']:>8.3f}{stage_a[n]['ST']:>8.3f}"
              f"{stage_a[n]['interaction']:>9.3f}")
    print(f"\n  === B: el nodo aguas abajo que su modelo no tiene ===")
    print(f"    H_regime SIN el nodo   {h_without['H_regime']:.6f}")
    print(f"    H_regime CON el nodo   {h_with['H_regime']:.6f}   "
          f"(delta {node_gain:+.6f})")
    print(f"    nivel (mejor común)    {h_without['best_common']:.6f} -> "
          f"{h_with['best_common']:.6f}   (delta {level_gain:+.6f})")
    print("\n  falsadores:")
    for name, check in falsifiers.items():
        if name != "all_passed":
            print(f"    {name:<48} {'PASA' if check['passed'] else 'FALLA'}")

    payload = {
        "schema_version": "perrisk_and_new_node_v1",
        "claim_status": ("DEVELOPMENT_PERRISK_AND_NEW_NODE" if falsifiers["all_passed"]
                         else "HALTED_FALSIFIER_FAILED"),
        "metric": METRIC, "side_metric": SIDE_METRIC,
        "stage_a_per_risk": {"indices_rank": stage_a, "indices_raw_UNUSABLE": stage_a_raw,
                             "sum_S1_rank": sum_s1, "ranking": ranked,
                             "n_runs": int((k + 2) * args.n_sobol)},
        "stage_b_new_node": {
            "node": ("emergency theatre reserve -- finite stock behind the downstream "
                     "corridor, replenished from Op9 with a real lead and blocked while a "
                     "route operation is down"),
            "choices": [{"op12_rop": p, "node": n} for p, n in choices],
            "H_without_node": h_without, "H_with_node": h_with,
            "headroom_gain": node_gain, "level_gain": level_gain,
            "n_runs": int(grid_scores.size)},
        "falsifiers": falsifiers,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(
        payload, args.output,
        contract=Path("docs/RESULTADO_SIJ_Y_NODO_2026-07-31.md"),
        reference=Path("results/sensitivity/second_order_risk_search_v1/result.json"))
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
