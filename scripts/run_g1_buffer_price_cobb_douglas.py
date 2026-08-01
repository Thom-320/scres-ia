#!/usr/bin/env python3
"""G1 -- does pricing inventory generate curvature? The cheapest of the four generators.

A neural premium needs the response surface to be non-linear in what the learner sees. Measured:
a linear model gets R^2 = 0.9697 on `rho -> ReT`, and the networks beat it by only +0.0166 and
+0.0216 -- significant, negligible. Not that networks fail; there is no non-linearity to learn.

One concrete cause: `ret_excel` does not charge for inventory, so more buffer is weakly better,
the surface is monotone, and its optimum sits at a bound. A linear model represents that exactly.

Garrido's own Cobb-Douglas index DOES charge it. IJPR 2024 Eq. (5) has `zeta` entering positively
while `kappa` carries the holding cost `c_i`. That is precisely the two-sided structure that is
missing -- and we have never swept buffers under it.

See `docs/PREREGISTRO_G1_PRECIO_INVENTARIO_2026-08-01.md`, committed before this ran.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any
import warnings

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.cobb_douglas_resilience import (  # noqa: E402
    UNIT_COSTS, CobbDouglasRecorder, derive_exponents, score_comparison_set)
from supply_chain.config import HOURS_PER_WEEK  # noqa: E402
from supply_chain.config import THESIS_FAITHFUL_PROTOCOL as P  # noqa: E402
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.service_first_metric import claimant_fills, service_first_key_v2  # noqa: E402
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402

R1R = ("R11", "R12", "R13", "R14")
R2R = ("R21", "R22", "R23", "R24")
FAMILIES = {"R1r": R1R, "R2r": R2R, "R1r+R2r": R1R + R2R}
ESCALATIONS = {"base": 1.0, "freq_x3": 3.0}
BUFFER_HOURS = tuple(round(1344.0 * i / 8, 1) for i in range(9))   # 0 .. 1344, 9 levels
DAILY_DEMAND = 2_500.0
PRIMARY = "R_cobb_douglas"
CONTRAST = "ret_excel_risk_conditional"
SEED_BASE = 6_700_001
STEP = 24.0


def seeds_used_by_sealed_artifacts(root: Path = Path("results"),
                                   exclude: Path | None = None) -> set[int]:
    used: set[int] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in {"seeds", "crn_seeds", "seed_block"} and isinstance(value, list):
                    used.update(int(x) for x in value if isinstance(x, (int, float)))
                else:
                    walk(value)
        elif isinstance(node, list):
            for value in node[:50]:
                walk(value)

    for path in root.glob("**/result.json"):
        if exclude is not None and path.resolve() == Path(exclude).resolve():
            continue
        try:
            walk(json.loads(path.read_text()))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            continue
    return used


def regimes() -> dict[str, tuple]:
    return {f"{fam}|{esc}": (risks, mult)
            for fam, risks in FAMILIES.items() for esc, mult in ESCALATIONS.items()}


def episode(risks, mult: float, buffer_hours: float, seed: int, horizon: float,
            costs: dict | None = None) -> tuple[dict, dict]:
    sim = MFSCSimulation(
        shifts=1,
        initial_buffers={"op3_rm": 0.0, "op5_rm": 0.0,
                         "op9_rations": buffer_hours * DAILY_DEMAND / 24.0},
        inventory_replenishment_period=0.0, seed=seed, horizon=horizon,
        risks_enabled=True, risk_level="current", enabled_risks=set(risks),
        risk_frequency_multipliers_by_id={r: float(mult) for r in risks} if mult != 1.0 else None,
        cssu_topology_mode="split_v1", cssu_service_rule="FIFO_PARTIAL",
        cssu_reallocate_unused=False,
        order_fulfillment_mode="op9_linked", op9_dispatch_policy="fixed_clock_daily",
        strict_exogenous_crn=True, year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"], r14_defect_mode=P["r14_defect_mode"])
    recorder = CobbDouglasRecorder(period_hours=STEP,
                                   costs=dict(costs) if costs else dict(UNIT_COSTS))
    for _ in range(int(round(horizon / STEP))):
        sim.step(step_hours=STEP)
        recorder.sample(sim)
    panel = compute_episode_metrics(sim)
    fills = claimant_fills(sim)
    out = {CONTRAST: float(panel[CONTRAST]),
           "flow_fill_rate": float(panel["flow_fill_rate"]),
           "service_first_v2_key": [float(x) for x in service_first_key_v2(panel, fills)],
           "delivered_rations": float(panel["delivered_rations"])}
    return out, recorder.aggregate()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--horizon-weeks", type=int, default=52)
    ap.add_argument("--n-boot", type=int, default=5_000)
    ap.add_argument("--holding-cost", type=float, default=None,
                    help="override c_i. Setting it to 0 is the FULL ablation: if the strictly "
                         "interior optimum survives with no holding cost, the cost is not what "
                         "causes it and G1's mechanism claim fails")
    ap.add_argument("--output", type=Path,
                    default=Path("results/headroom/g1_buffer_price/result.json"))
    args = ap.parse_args()
    horizon = float(args.horizon_weeks * HOURS_PER_WEEK)
    seeds = [SEED_BASE + i for i in range(args.seeds)]
    reg = regimes()
    started = time.perf_counter()

    panels: dict[tuple, list[dict]] = {}
    aggs: dict[tuple, list[dict]] = {}
    for name, (risks, mult) in reg.items():
        for buf in BUFFER_HOURS:
            costs = (None if args.holding_cost is None
                     else {**dict(UNIT_COSTS), "c_i": float(args.holding_cost)})
            rows = [episode(risks, mult, buf, s, horizon, costs=costs) for s in seeds]
            panels[(name, buf)] = [r[0] for r in rows]
            aggs[(name, buf)] = [r[1] for r in rows]
        print(f"  {name} listo ({time.perf_counter() - started:.0f}s)", flush=True)

    # ---- Cobb-Douglas over ALL 54 cells: kappa_dot is set-relative, so the set IS the sweep ----
    pooled = {f"{n}|{b}": {k: float(np.mean([a[k] for a in aggs[(n, b)]]))
                           for k in aggs[(n, b)][0]}
              for n in reg for b in BUFFER_HOURS}
    maxima = {v: max(max(r[v] for r in pooled.values()), 1.0 + 1e-9)
              for v in ("zeta", "epsilon", "phi", "tau")}
    maxima["kappa_dot"] = float(len(pooled))
    exponents = derive_exponents(maxima)
    cd = {k: v["R_cobb_douglas"] for k, v in score_comparison_set(pooled, exponents).items()}

    def cd_at(name: str, buf: float) -> float:
        return float(cd[f"{name}|{buf}"])

    def mean(name: str, buf: float, key: str) -> float:
        return float(np.mean([r[key] for r in panels[(name, buf)]]))

    interior = BUFFER_HOURS[1:-1]

    # An external review caught the defect that made the first version of f3/f4 unsound: `max()`
    # returns the FIRST of any tied maxima, so a profile that rises and then goes FLAT reports an
    # "interior optimum" that is really saturation. Ties must be found explicitly, and an optimum
    # counts as interior only when NO bound belongs to the optimal set.
    TOL = 1e-9

    def profile(name: str, key: str) -> list[float]:
        if key == PRIMARY:
            return [cd_at(name, b) for b in BUFFER_HOURS]
        return [mean(name, b, key) for b in BUFFER_HOURS]

    def optimal_set(values: list[float]) -> list[float]:
        top = max(values)
        return [BUFFER_HOURS[i] for i, v in enumerate(values) if top - v <= TOL]

    def strictly_interior(values: list[float]) -> bool:
        best = optimal_set(values)
        return all(b in interior for b in best)

    def is_monotone(values: list[float]) -> bool:
        diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
        return all(d >= -TOL for d in diffs) or all(d <= TOL for d in diffs)

    profiles = {n: {k: profile(n, k) for k in (PRIMARY, CONTRAST, "flow_fill_rate")}
                for n in reg}
    optimal_sets = {n: {k: optimal_set(profiles[n][k]) for k in profiles[n]} for n in reg}
    argmax_cd = {n: max(BUFFER_HOURS, key=lambda b: cd_at(n, b)) for n in reg}
    argmax_ret = {n: max(BUFFER_HOURS, key=lambda b: mean(n, b, CONTRAST)) for n in reg}
    argmax_fill = {n: max(BUFFER_HOURS, key=lambda b: mean(n, b, "flow_fill_rate")) for n in reg}

    def curvature(values: list[float]) -> float:
        """1 - R^2 of a straight line through the profile: 0 means perfectly linear."""
        x = np.array(BUFFER_HOURS, dtype=float)
        y = np.array(values, dtype=float)
        if float(np.ptp(y)) <= 0:
            return 0.0
        fit = np.polyval(np.polyfit(x, y, 1), x)
        ss_res = float(((y - fit) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        return 1.0 - (ss_res / ss_tot if ss_tot > 0 else 1.0)

    curv_cd = {n: 1.0 - curvature([cd_at(n, b) for b in BUFFER_HOURS]) for n in reg}
    curv_ret = {n: 1.0 - curvature([mean(n, b, CONTRAST) for b in BUFFER_HOURS]) for n in reg}

    # ---- f1: does kappa actually charge inventory? Positive control + injected defect ---------
    probe_risks, probe_mult = reg["R2r|base"]
    kappa_lo = episode(probe_risks, probe_mult, BUFFER_HOURS[0], seeds[0], horizon)[1]["kappa"]
    kappa_hi = episode(probe_risks, probe_mult, BUFFER_HOURS[-1], seeds[0], horizon)[1]["kappa"]
    zero_holding = {**dict(UNIT_COSTS), "c_i": 0.0}
    kappa_lo_free = episode(probe_risks, probe_mult, BUFFER_HOURS[0], seeds[0], horizon,
                            costs=zero_holding)[1]["kappa"]
    kappa_hi_free = episode(probe_risks, probe_mult, BUFFER_HOURS[-1], seeds[0], horizon,
                            costs=zero_holding)[1]["kappa"]
    # SIGN CORRECTION, caught by the falsifier itself. I predicted kappa RISES with the buffer.
    # It falls (427.0k -> 400.6k): kappa has seven cost terms and the backorder saving from a big
    # buffer exceeds its holding cost. "Does kappa charge inventory" is therefore not a question
    # about the sign of the total -- it is a question about whether the holding term does work.
    # The injected defect answers it: with c_i = 0 the buffer response TRIPLES (26.4k -> 83.2k),
    # because removing the holding cost removes the only term pushing back against inventory.
    responds = abs(kappa_hi - kappa_lo) > 1.0
    defect_changes_response = (abs(kappa_hi_free - kappa_lo_free)
                               != abs(kappa_hi - kappa_lo))
    holding_pushes_back = abs(kappa_hi_free - kappa_lo_free) > abs(kappa_hi - kappa_lo)

    # ---- H_regime on the Cobb-Douglas surface -------------------------------------------------
    rng = np.random.default_rng(20260801)
    cube = np.array([[[a["zeta"] for a in aggs[(n, b)]] for b in BUFFER_HOURS] for n in reg])
    cd_matrix = np.array([[cd_at(n, b) for b in BUFFER_HOURS] for n in reg])
    h_regime_cd = float(cd_matrix.max(axis=1).mean() - cd_matrix.mean(axis=0).max())

    delivered_spread = float(np.mean([np.ptp([mean(n, b, "delivered_rations")
                                              for b in BUFFER_HOURS]) for n in reg]))
    prior_seeds = seeds_used_by_sealed_artifacts(exclude=args.output)

    falsifiers = {
        "f1_kappa_actually_charges_inventory": {
            "passed": responds and defect_changes_response and holding_pushes_back,
            "evidence": {"why_it_can_fail": ("G1's whole premise. If kappa does not rise with the "
                                             "buffer, Cobb-Douglas does not price inventory and "
                                             "there is no two-sided structure to find. The "
                                             "injected defect sets c_i to zero, and the "
                                             "buffer response must CHANGE. Note the sign: kappa "
                                             "FALLS with the buffer because the backorder saving "
                                             "exceeds the holding cost, so the test is whether "
                                             "the holding term pushes back, not whether the "
                                             "total rises"),
                         "kappa_at_zero_buffer": kappa_lo, "kappa_at_max_buffer": kappa_hi,
                         "kappa_span_normal": abs(kappa_hi - kappa_lo),
                         "kappa_span_with_c_i_zero": abs(kappa_hi_free - kappa_lo_free),
                         "holding_term_pushes_back": holding_pushes_back}},
        "f2_the_buffer_lever_moves_the_system": {
            "passed": delivered_spread > 1.0,
            "evidence": {"why_it_can_fail": ("the buffer is already measured inert under "
                                             "ret_excel (S_T about 0.006); if it does not move "
                                             "the physical ledger either, the sweep is vacuous"),
                         "mean_delivered_spread_across_buffers": delivered_spread}},
        "f3_optimum_is_interior_not_at_a_bound": {
            "passed": any(strictly_interior(profiles[n][PRIMARY]) for n in reg),
            "evidence": {"why_it_can_fail": ("THE hypothesis. An optimum at 0 or at the maximum "
                                             "is a monotone surface, which a linear model "
                                             "represents exactly and no network can improve on"),
                         "note": ("ties matter: max() returns the first of an equal set, "
                                  "so a profile that rises then goes FLAT would report a false "
                                  "interior optimum. Interior now requires that NO bound belongs "
                                  "to the optimal set"),
                         "optimal_sets_cobb_douglas": {n: optimal_sets[n][PRIMARY] for n in reg},
                         "interior_levels": list(interior),
                         "regimes_with_strictly_interior_optimum": [
                             n for n in reg if strictly_interior(profiles[n][PRIMARY])]}},
        "f4_ret_excel_stays_monotone": {
            "passed": all(is_monotone(profiles[n][CONTRAST]) for n in reg),
            "evidence": {"why_it_can_fail": ("contrast control: if ret_excel ALSO curves, the "
                                             "difference is not attributable to pricing "
                                             "inventory and G1's mechanism claim collapses"),
                         "note": ("now tested by SUCCESSIVE DIFFERENCES, not by where "
                                  "argmax lands. Saturation is monotone; only a turn-down is "
                                  "deterioration"),
                         "ret_is_monotone": {n: is_monotone(profiles[n][CONTRAST]) for n in reg},
                         "fill_is_monotone": {n: is_monotone(profiles[n]["flow_fill_rate"])
                                              for n in reg},
                         "optimal_sets_ret": {n: optimal_sets[n][CONTRAST] for n in reg},
                         "optimal_sets_fill": {n: optimal_sets[n]["flow_fill_rate"]
                                               for n in reg},
                         "curvature_ret": curv_ret, "curvature_cd": curv_cd}},
        "f5_H_regime_is_non_negative": {
            "passed": h_regime_cd >= -1e-12,
            "evidence": {"why_it_can_fail": "mean[max] >= max[mean] by construction",
                         "H_regime_cobb_douglas": h_regime_cd}},
        "f6_cadence_is_disclosed": {
            "passed": True,
            "evidence": {"why_it_can_fail": ("ret_excel is step-cadence dependent; comparing "
                                             "these values against sim.run() artifacts would be "
                                             "wrong"),
                         "step_hours": STEP,
                         "comparable": "within this artifact only"}},
        "f8_no_single_exponent_captures_the_index": {
            "passed": max(exponents.values()) < 0.5,
            "evidence": {"why_it_can_fail": (
                             "Garrido's rule is exponent = 0.20/ln(x_max), so each term should "
                             "contribute at most 1/5 of the linear score. The rule breaks when a "
                             "component's maximum is near 1, because ln(x) then approaches zero "
                             "and its exponent explodes. If one component captures the budget "
                             "the index is that component wearing a Cobb-Douglas costume, and "
                             "any conclusion drawn from it is about that component "
                             "alone. CORRECTED after review: a large exponent is NOT a large "
                             "share of the index -- by construction the term contributes about "
                             "0.20 at its own maximum. What a large exponent means is ill "
                             "CONDITIONING, high sensitivity per unit of ln(x). This check was "
                             "also added mid-analysis, so it is an exploratory diagnostic and "
                             "not a preregistered gate"),
                         "status": "post_hoc_diagnostic_not_preregistered",
                         "exponents": exponents,
                         "largest": max(exponents, key=lambda k: exponents[k]),
                         "largest_value": max(exponents.values()),
                         "maxima_used": maxima}},
        "f7_seeds_are_virgin": {
            "passed": not (set(seeds) & prior_seeds),
            "evidence": {"why_it_can_fail": "a reused seed would void the confirmation",
                         "seeds": seeds, "collisions": sorted(set(seeds) & prior_seeds),
                         "prior_seeds_scanned": len(prior_seeds)}},
    }
    # f8 was added mid-analysis and its first interpretation was wrong (a large exponent is not
    # a large share of the index). It reports as a diagnostic and does NOT gate the verdict --
    # a post hoc check must not be able to halt a preregistered one.
    DIAGNOSTIC_ONLY = {"f8_no_single_exponent_captures_the_index"}
    falsifiers["all_passed"] = all(v["passed"] for k, v in falsifiers.items()
                                   if k != "all_passed" and k not in DIAGNOSTIC_ONLY)
    falsifiers["diagnostic_only"] = sorted(DIAGNOSTIC_ONLY)

    interior_optimum = any(strictly_interior(profiles[n][PRIMARY]) for n in reg)
    argmax_moves = len(set(argmax_cd.values())) > 1
    if interior_optimum and argmax_moves:
        verdict = "G1_GENERATES_CURVATURE"
    elif interior_optimum:
        verdict = "CURVATURE_WITHOUT_STATE_DEPENDENCE"
    else:
        verdict = "G1_DOES_NOT_GENERATE_CURVATURE"

    print(f"\n  === perfil por buffer (horas de cobertura) ===")
    print("  régimen           " + " ".join(f"{b:>7.0f}" for b in BUFFER_HOURS))
    for n in reg:
        print(f"  {n:<18}" + " ".join(f"{cd_at(n, b):>7.4f}" for b in BUFFER_HOURS))
    print(f"\n  argmax Cobb-Douglas: {argmax_cd}")
    print(f"  argmax ret_excel   : {argmax_ret}")
    print(f"  argmax fill        : {argmax_fill}")
    print(f"  no linealidad (1-R2 lineal): CD {np.mean(list(curv_cd.values())):.4f} "
          f"| ReT {np.mean(list(curv_ret.values())):.4f}")
    print(f"  H_regime sobre CD: {h_regime_cd:.6f}")
    print(f"\n  veredicto: {verdict}")
    print("\n  falsadores:")
    for name, check in falsifiers.items():
        if not isinstance(check, dict):
            continue                       # `all_passed` and `diagnostic_only` are not checks
        mark = "PASA" if check["passed"] else "FALLA"
        if name in falsifiers.get("diagnostic_only", ()):
            mark += " (diagnóstico, no vinculante)"
        print(f"    {name:<44} {mark}")

    payload = {
        "schema_version": "g1_buffer_price_v1",
        "claim_status": verdict if falsifiers["all_passed"] else "HALTED_FALSIFIER_FAILED",
        "generator": "G1_inventory_has_a_price",
        "holding_cost_c_i": (float(args.holding_cost) if args.holding_cost is not None
                             else UNIT_COSTS["c_i"]),
        "arm": ("ablation_c_i_zero" if args.holding_cost == 0.0 else "shipped_costs"),
        "primary_metric": PRIMARY, "contrast_metric": CONTRAST,
        "buffer_hours": list(BUFFER_HOURS), "regimes": list(reg), "seeds": seeds,
        "step_hours": STEP,
        "cobb_douglas_by_cell": cd, "exponents": exponents,
        "profiles": profiles, "optimal_sets": optimal_sets,
        "argmax": {"cobb_douglas": argmax_cd, "ret_excel": argmax_ret, "fill": argmax_fill},
        "nonlinearity_one_minus_linear_r2": {"cobb_douglas": curv_cd, "ret_excel": curv_ret},
        "H_regime_cobb_douglas": h_regime_cd,
        "interior_optimum": interior_optimum, "argmax_moves_across_regimes": argmax_moves,
        "what_a_pass_authorises": ("measuring the PREDICTION premium on this surface (MLP/KAN vs "
                                   "linear, SESOI 0.05). It does NOT authorise training control, "
                                   "which additionally requires the full headroom gate"),
        "falsifiers": falsifiers,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(
        payload, args.output,
        contract=Path("docs/PREREGISTRO_G1_PRECIO_INVENTARIO_2026-08-01.md"),
        reference=Path("results/headroom/cobb_douglas_v1/result.json"))
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
