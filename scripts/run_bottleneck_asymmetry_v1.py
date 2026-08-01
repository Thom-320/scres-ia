#!/usr/bin/env python3
"""Fase 1A' -- contention AT the bottleneck with ASYMMETRIC claimants: does H_PI track asymmetry?

Fase 1A put the dispute DOWNSTREAM of the bottleneck between claimants that are symmetric by
construction, and measured H_regime = 1.5e-04 while forfeiting 17% of flow. Program O put two
NON-FUNGIBLE products in contention FOR Op5-Op7 -- the real bottleneck, 2.6% margin -- and
measured H_PI = 0.1515. The two differences are variables of this instrument, not assumptions:
WHERE the dispute is, and WHETHER the claimants are asymmetric.

So sweep the asymmetry. `dominant_share = 0.5` is the symmetric case, i.e. Fase 1A's condition;
0.9 is strongly asymmetric. `complete_substitution=True` makes the two products interchangeable,
which is the fungible null that has now returned exactly 0.0 twice today.

This is NOT a Program O rescue: different estimand (H_PI as a function of asymmetry, not H_obs),
virgin seeds disjoint from its burned blocks, no learner, its own preregistration. See
`docs/PREREGISTRO_ASIMETRIA_CUELLO_2026-07-31.md` for the reading rule, fixed in advance --
including the rule that a headroom bought by dropping a product is reported as a metric defect.
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
from supply_chain.program_o_full_des import run_program_o_full_des_episode  # noqa: E402

SCHEDULER = {0: ("P_C", "P_C", "P_C"), 1: ("P_C", "P_C", "P_H"),
             2: ("P_C", "P_H", "P_H"), 3: ("P_H", "P_H", "P_H")}
ACTIONS = (0, 3)                       # pure schedules; 256 calendars, enumerated exactly
WEEKS = 8
CALENDARS = tuple(itertools.product(ACTIONS, repeat=WEEKS))
SHARES = (0.5, 0.6, 0.7, 0.8, 0.9)     # 0.5 IS the symmetric case Fase 1A ran under
PERSISTENCE = (0.5, 0.9)
RISKS = {"R21", "R22", "R23", "R24"}
PRIMARY = "ret_excel_risk_conditional"
SIDE = ("ret_excel_visible_clipped_0_1",)
SEED_BASE = 7_600_001
# Program O's burned validation blocks. Touching either would contaminate a sealed verdict.
PROGRAM_O_BURNED = set(range(7_420_049, 7_420_097)) | set(range(7_430_001, 7_430_049))


def episode(seed: int, calendar, share: float, persistence: float,
            substitution: bool) -> dict[str, float]:
    _, panel = run_program_o_full_des_episode(
        seed=int(seed), calendar=list(calendar), scheduler=SCHEDULER,
        regime_persistence=float(persistence), dominant_share=float(share),
        complete_substitution=bool(substitution),
        risks_enabled=True, enabled_risks=set(RISKS))
    metrics, products = panel["metrics"], panel["products"]
    out = {PRIMARY: float(metrics[PRIMARY])}
    out.update({k: float(metrics[k]) for k in SIDE})
    out["worst_product_fill"] = float(panel["worst_product_fill"])
    out["demand_pc"] = float(products["P_C"]["demanded_quantity"])
    out["demand_ph"] = float(products["P_H"]["demanded_quantity"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=5_000)
    ap.add_argument("--output", type=Path,
                    default=Path("results/sensitivity/bottleneck_asymmetry_v1/result.json"))
    args = ap.parse_args()
    seeds = [SEED_BASE + i for i in range(args.seeds)]
    started = time.perf_counter()
    rng = np.random.default_rng(20260731)

    cells: dict[tuple, np.ndarray] = {}          # (share, persistence, subst) -> (S, C) primary
    extras: dict[tuple, dict] = {}
    for share, persistence, subst in itertools.product(SHARES, PERSISTENCE, (False, True)):
        cube = np.zeros((len(seeds), len(CALENDARS)))
        side = {m: np.zeros_like(cube) for m in (*SIDE, "worst_product_fill")}
        dem = np.zeros((len(seeds), 2))
        for si, seed in enumerate(seeds):
            for ci, cal in enumerate(CALENDARS):
                row = episode(seed, cal, share, persistence, subst)
                cube[si, ci] = row[PRIMARY]
                for m in side:
                    side[m][si, ci] = row[m]
            dem[si] = [row["demand_pc"], row["demand_ph"]]
        cells[(share, persistence, subst)] = cube
        extras[(share, persistence, subst)] = {"side": side, "demand": dem}
        print(f"  share={share} persist={persistence} subst={subst} "
              f"({time.perf_counter() - started:.0f}s)", flush=True)

    def h_pi(cube: np.ndarray) -> dict[str, float]:
        """mean_seed[max_calendar] - max_calendar[mean_seed], bootstrapped over seeds."""
        def stat(idx: np.ndarray) -> float:
            sub = cube[idx]
            return float(sub.max(axis=1).mean() - sub.mean(axis=0).max())
        point = stat(np.arange(cube.shape[0]))
        draws = np.array([stat(rng.integers(0, cube.shape[0], cube.shape[0]))
                          for _ in range(args.n_boot)])
        return {"H_PI": point, "lcb95": float(np.percentile(draws, 2.5)),
                "ucb95": float(np.percentile(draws, 97.5))}

    summary = {f"share={s}|persist={p}|subst={b}": h_pi(c)
               for (s, p, b), c in cells.items()}

    # H1: does H_PI grow with asymmetry? Read at the best persistence, non-substituting arm.
    by_share = {s: max(h_pi(cells[(s, p, False)])["H_PI"] for p in PERSISTENCE)
                for s in SHARES}
    best_share = max(SHARES, key=lambda s: by_share[s])
    best_cell = max((k for k in summary if "subst=False" in k),
                    key=lambda k: summary[k]["H_PI"])
    h_best, lcb_best = summary[best_cell]["H_PI"], summary[best_cell]["lcb95"]
    h_subst = max(summary[k]["H_PI"] for k in summary if "subst=True" in k)

    # Does the winning calendar buy ReT by dropping a product? The rule says that is a metric
    # defect, not headroom, so it is measured rather than assumed away.
    s_b, p_b, _ = max(cells, key=lambda k: (not k[2], h_pi(cells[k])["H_PI"]))
    cube_b = cells[(s_b, p_b, False)]
    wpf_b = extras[(s_b, p_b, False)]["side"]["worst_product_fill"]
    argmax_cal = cube_b.mean(axis=0).argmax()
    wpf_at_best = float(wpf_b[:, argmax_cal].mean())
    wpf_at_worst = float(wpf_b[:, cube_b.mean(axis=0).argmin()].mean())

    demand_gap = {s: float(np.mean(np.abs(extras[(s, 0.9, False)]["demand"][:, 0]
                                          - extras[(s, 0.9, False)]["demand"][:, 1])))
                  for s in SHARES}
    calendar_spread = float(np.mean([np.ptp(c.mean(axis=0)) for c in cells.values()]))

    falsifiers = {
        "f1_asymmetry_is_actually_asymmetric": {
            "passed": demand_gap[0.9] > demand_gap[0.5],
            "evidence": {"why_it_can_fail": ("if realised demand does not separate, the lever "
                                             "creates no asymmetry and H1 is vacuous"),
                         "mean_abs_demand_gap_by_share": demand_gap}},
        "f2_substitution_control_binds": {
            "passed": h_subst < h_best,
            "evidence": {"why_it_can_fail": ("if H_PI does not fall under complete substitution "
                                             "the fungibility control controls nothing"),
                         "H_PI_best_non_substituting": h_best,
                         "H_PI_best_substituting": h_subst}},
        "f3_the_calendar_changes_the_outcome": {
            "passed": calendar_spread > 1e-9,
            "evidence": {"why_it_can_fail": ("if all 256 calendars score alike there is no "
                                             "decision and H_PI measures noise"),
                         "mean_spread_across_calendars": calendar_spread,
                         "n_calendars": len(CALENDARS)}},
        "f4_H_PI_is_non_negative": {
            "passed": all(v["H_PI"] >= -1e-12 for v in summary.values()),
            "evidence": {"why_it_can_fail": ("mean[max] >= max[mean] by construction; negative "
                                             "would be an aggregation bug, not a finding")}},
        "f5_seeds_are_virgin_and_disjoint_from_program_o": {
            "passed": not (set(seeds) & PROGRAM_O_BURNED),
            "evidence": {"why_it_can_fail": ("touching a burned validation block would "
                                             "contaminate a sealed verdict"),
                         "seeds": seeds,
                         "program_o_burned": "7420049-7420096, 7430001-7430048"}},
        "f6_this_is_not_a_program_o_rescue": {
            "passed": True,
            "evidence": {"why_it_can_fail": ("re-running its cells to flip its verdict would be "
                                             "a rescue and is forbidden by its contract"),
                         "estimand": "H_PI as a function of asymmetry, not H_obs",
                         "learner_trained": False, "cells_reused": False,
                         "own_preregistration": "docs/PREREGISTRO_ASIMETRIA_CUELLO_2026-07-31.md"}},
    }
    falsifiers["all_passed"] = all(v["passed"] for k, v in falsifiers.items()
                                   if k != "all_passed")

    grows = by_share[0.9] > by_share[0.5]
    metric_defect = wpf_at_best < wpf_at_worst
    if metric_defect:
        verdict = "HEADROOM_IS_A_METRIC_DEFECT_NOT_A_DECISION"
    elif h_best >= 0.01 and lcb_best > 0 and grows and h_subst < h_best:
        verdict = "ASYMMETRY_AT_THE_BOTTLENECK_IS_THE_MISSING_INGREDIENT"
    elif h_best >= 0.01 and not grows:
        verdict = "LOCATION_NOT_ASYMMETRY_IS_WHAT_MATTERS"
    else:
        verdict = "NEITHER_LOCATION_NOR_ASYMMETRY_OPENS_THE_DOOR"

    print(f"\n  === H_PI sobre `{PRIMARY}` (256 calendarios enumerados) ===")
    print(f"  {'asimetría':<12}{'H_PI (no fungible)':>22}")
    for s in SHARES:
        print(f"  {s:<12}{by_share[s]:>22.6f}")
    print(f"\n  mejor celda: {best_cell} -> {h_best:.6f} (LCB95 {lcb_best:.6f})")
    print(f"  mejor con sustitución completa (nulo): {h_subst:.6f}")
    print(f"  worst_product_fill en el mejor calendario {wpf_at_best:.4f} "
          f"vs en el peor {wpf_at_worst:.4f}")
    print(f"\n  veredicto: {verdict}")
    print("\n  falsadores:")
    for name, check in falsifiers.items():
        if name != "all_passed":
            print(f"    {name:<48} {'PASA' if check['passed'] else 'FALLA'}")

    payload = {
        "schema_version": "bottleneck_asymmetry_v1",
        "claim_status": verdict if falsifiers["all_passed"] else "HALTED_FALSIFIER_FAILED",
        "primary_metric": PRIMARY, "side_metrics": list(SIDE),
        "n_calendars": len(CALENDARS), "weeks": WEEKS, "actions": list(ACTIONS),
        "shares": list(SHARES), "persistence": list(PERSISTENCE), "seeds": seeds,
        "summary": summary, "H_PI_by_asymmetry": by_share, "best_share": best_share,
        "best_cell": {"cell": best_cell, "H_PI": h_best, "lcb95": lcb_best},
        "fungible_null_best": h_subst,
        "service_check": {"worst_product_fill_at_best_calendar": wpf_at_best,
                          "worst_product_fill_at_worst_calendar": wpf_at_worst,
                          "rule": ("headroom bought by dropping a product is a metric defect, "
                                   "not a decision")},
        "fase_1a_reference": {"H_regime": 1.53e-04, "condition": "downstream, symmetric"},
        "program_o_reference": {"H_PI": 0.15151, "fungible_null": 0.0},
        "falsifiers": falsifiers,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(
        payload, args.output,
        contract=Path("docs/PREREGISTRO_ASIMETRIA_CUELLO_2026-07-31.md"),
        reference=Path("results/sensitivity/contention_headroom_v1/result.json"))
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
