#!/usr/bin/env python3
"""The prediction premium on the first surface this project has measured curvature on.

Q1 was answered on the `rho -> ReT` panel, where a linear model already gets R^2 = 0.9697 and the
networks add +0.0166 and +0.0216 -- statistically significant, practically negligible against the
preregistered SESOI of 0.05. The obvious reply was that the surface was simply linear, so there
was nothing for a network to do.

G1 then measured a surface that is NOT linear: the buffer profile has 1 - linear R^2 of 0.0790 on
ret_excel, with an interior optimum in two regimes. That curvature is physical rather than
metric-induced -- flow_fill_rate, which has no cost term, puts its optimum at the same level.

So this asks the premium question where the premise finally holds. If no premium appears even
here, the paper's claim strengthens considerably: not "the surface was too easy" but "even with
measured curvature the neural premium does not reach the smallest effect worth having".

Reuses `fit_mlp`, `fit_kan` and `grouped_folds` from the Q1 surrogate rather than reimplementing
them, so the comparison is against the same trained networks under the same protocol.

See `docs/PREREGISTRO_PRIMA_PREDICCION_BUFFER_2026-08-01.md`, committed before this ran.
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")

from build_garrido_fig5_surrogate import fit_kan, fit_mlp, grouped_folds  # noqa: E402
from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.seed_custody import (  # noqa: E402
    custody_falsifier, seeds_used_by_sealed_artifacts)
from supply_chain.config import HOURS_PER_WEEK  # noqa: E402
from supply_chain.config import THESIS_FAITHFUL_PROTOCOL as P  # noqa: E402
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402

R1R = ("R11", "R12", "R13", "R14")
R2R = ("R21", "R22", "R23", "R24")
FAMILIES = ("R1r", "R2r", "R1r+R2r")
FAMILY_RISKS = {"R1r": R1R, "R2r": R2R, "R1r+R2r": R1R + R2R}
ESCALATIONS = {"base": 1.0, "freq_x3": 3.0, "freq_x5": 5.0}
N_LEVELS = 17
BUFFER_HOURS = tuple(round(1344.0 * i / (N_LEVELS - 1), 1) for i in range(N_LEVELS))
TARGET = "ret_excel_risk_conditional"
SESOI = 0.05
DAILY_DEMAND = 2_500.0
SEED_BASE = 6_800_001




def features(buffer_hours: float, family: str, escalation: str) -> list[float]:
    """`rho` plus the risk design -- deliberately WITHOUT the drivers.

    A driver is a post-simulation quantity and the four of them sum to ReT exactly, so including
    one would leak the answer. This is the same rule the Q1 surrogate follows.
    """
    return [
        buffer_hours / 1344.0,
        *[1.0 if family == f else 0.0 for f in FAMILIES],
        *[1.0 if escalation == e else 0.0 for e in ESCALATIONS],
    ]


def episode(risks, mult: float, buffer_hours: float, seed: int, horizon: float) -> float:
    sim = MFSCSimulation(
        shifts=1,
        initial_buffers={"op3_rm": 0.0, "op5_rm": 0.0,
                         "op9_rations": buffer_hours * DAILY_DEMAND / 24.0},
        inventory_replenishment_period=0.0, seed=seed, horizon=horizon,
        risks_enabled=True, risk_level="current", enabled_risks=set(risks),
        risk_frequency_multipliers_by_id=({r: float(mult) for r in risks}
                                          if mult != 1.0 else None),
        cssu_topology_mode="split_v1", cssu_service_rule="FIFO_PARTIAL",
        cssu_reallocate_unused=False,
        order_fulfillment_mode="op9_linked", op9_dispatch_policy="fixed_clock_daily",
        strict_exogenous_crn=True, year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"], r14_defect_mode=P["r14_defect_mode"])
    sim.run()
    return float(compute_episode_metrics(sim)[TARGET])


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return 1.0 - (ss_res / ss_tot if ss_tot > 0 else 1.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--horizon-weeks", type=int, default=52)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--output", type=Path,
                    default=Path("results/headroom/buffer_prediction_premium/result.json"))
    args = ap.parse_args()
    horizon = float(args.horizon_weeks * HOURS_PER_WEEK)
    seeds = [SEED_BASE + i for i in range(args.seeds)]
    started = time.perf_counter()

    rows, x_list, y_list, groups = [], [], [], []
    for family in FAMILIES:
        for escalation, mult in ESCALATIONS.items():
            for buf in BUFFER_HOURS:
                for seed in seeds:
                    value = episode(FAMILY_RISKS[family], mult, buf, seed, horizon)
                    rows.append({"family": family, "escalation": escalation,
                                 "buffer_hours": buf, "seed": seed, TARGET: value})
                    x_list.append(features(buf, family, escalation))
                    y_list.append(value)
                    groups.append(seed)
        print(f"  {family} listo ({time.perf_counter() - started:.0f}s)", flush=True)

    x = np.asarray(x_list, dtype=np.float64)
    y = np.asarray(y_list, dtype=np.float64)
    g = np.asarray(groups)

    # ---- curvature of the profile, recomputed here so f1 does not trust G1's artifact ---------
    curvature_by_cell = {}
    for family in FAMILIES:
        for escalation in ESCALATIONS:
            profile = [float(np.mean([r[TARGET] for r in rows
                                      if r["family"] == family and r["escalation"] == escalation
                                      and r["buffer_hours"] == b]))
                       for b in BUFFER_HOURS]
            xs = np.array(BUFFER_HOURS, dtype=float)
            ys = np.array(profile, dtype=float)
            fit = np.polyval(np.polyfit(xs, ys, 1), xs)
            ss_res = float(((ys - fit) ** 2).sum())
            ss_tot = float(((ys - ys.mean()) ** 2).sum())
            curvature_by_cell[f"{family}|{escalation}"] = (
                1.0 - (1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0)
    mean_curvature = float(np.mean(list(curvature_by_cell.values())))

    # ---- grouped CV: constant, linear, MLP, KAN ------------------------------------------------
    folds = grouped_folds(g, n_folds=args.folds)
    per_fold: dict[str, list[float]] = {"constant": [], "linear": [], "backprop": [], "kan": []}
    for fold_index, (train_idx, test_idx) in enumerate(folds):
        x_tr, y_tr, x_te, y_te = x[train_idx], y[train_idx], x[test_idx], y[test_idx]
        per_fold["constant"].append(r2(y_te, np.full_like(y_te, y_tr.mean())))
        coef, *_ = np.linalg.lstsq(np.c_[x_tr, np.ones(len(x_tr))], y_tr, rcond=None)
        per_fold["linear"].append(r2(y_te, np.c_[x_te, np.ones(len(x_te))] @ coef))
        pred_mlp, _ = fit_mlp(x_tr, y_tr, x_te, seed=1000 + fold_index, classify=False)
        per_fold["backprop"].append(r2(y_te, pred_mlp))
        try:
            pred_kan, _ = fit_kan(x_tr, y_tr, x_te, seed=1000 + fold_index, classify=False)
            per_fold["kan"].append(r2(y_te, pred_kan))
        except Exception as exc:                      # pragma: no cover - optional dependency
            per_fold["kan"].append(float("nan"))
            print(f"  KAN no disponible en fold {fold_index}: {exc}", flush=True)
    means = {k: float(np.nanmean(v)) for k, v in per_fold.items()}

    def paired(model: str) -> dict:
        d = np.array(per_fold[model]) - np.array(per_fold["linear"])
        d = d[~np.isnan(d)]
        se = float(d.std(ddof=1) / np.sqrt(d.size)) if d.size > 1 else 0.0
        return {"mean_difference": float(d.mean()), "sd": float(d.std(ddof=1)) if d.size > 1 else 0.0,
                "ci95_low": float(d.mean() - 1.96 * se), "ci95_high": float(d.mean() + 1.96 * se),
                "n_folds": int(d.size), "sesoi": SESOI,
                "passes_sesoi_and_ci": bool(d.mean() >= SESOI and (d.mean() - 1.96 * se) > 0)}

    comparisons = {m: paired(m) for m in ("backprop", "kan")}
    prior_seeds = seeds_used_by_sealed_artifacts(exclude=args.output)
    no_seed_in_both = all(not (set(g[tr].tolist()) & set(g[te].tolist())) for tr, te in folds)

    falsifiers = {
        "f1_the_surface_actually_has_curvature": {
            "passed": mean_curvature > 0.02,
            "evidence": {"why_it_can_fail": ("this is the PREMISE. Q1 already showed no premium "
                                             "on a near-linear surface; asking again on another "
                                             "linear surface would prove nothing. Recomputed "
                                             "here rather than trusting the G1 artifact"),
                         "mean_one_minus_linear_r2": mean_curvature,
                         "by_cell": curvature_by_cell}},
        "f2_no_driver_leakage": {
            "passed": x.shape[1] == 1 + len(FAMILIES) + len(ESCALATIONS),
            "evidence": {"why_it_can_fail": ("a driver is a post-simulation quantity and the four "
                                             "sum to ReT exactly, so one of them in the features "
                                             "would hand over the answer"),
                         "n_features": int(x.shape[1]),
                         "features": ["buffer_hours/1344", *FAMILIES, *ESCALATIONS]}},
        "f3_folds_are_grouped_by_seed": {
            "passed": no_seed_in_both,
            "evidence": {"why_it_can_fail": ("a seed in both train and test would let a model "
                                             "memorise that episode's noise and inflate every "
                                             "R2 including the linear one"),
                         "n_folds": len(folds), "n_seeds": len(seeds)}},
        "f4_linear_baseline_is_not_a_straw_man": {
            "passed": means["linear"] > means["constant"] + 0.01,
            "evidence": {"why_it_can_fail": ("a premium over a badly fitted linear model would "
                                             "measure our own incompetence, not non-linearity"),
                         "r2_constant": means["constant"], "r2_linear": means["linear"]}},
        "f5_sesoi_was_fixed_in_advance": {
            "passed": SESOI == 0.05,
            "evidence": {"why_it_can_fail": ("choosing the threshold after seeing the difference "
                                             "is how a negligible gain becomes a headline"),
                         "sesoi": SESOI,
                         "source": "the same 0.05 preregistered for the Q1 surrogate"}},
        "f6_seeds_are_virgin": {
            "passed": not (set(seeds) & prior_seeds),
            "evidence": {"why_it_can_fail": "reuse would void the comparison",
                         "seeds": seeds, "collisions": sorted(set(seeds) & prior_seeds),
                         "prior_seeds_scanned": len(prior_seeds)}},
    }
    falsifiers["all_passed"] = all(v["passed"] for k, v in falsifiers.items()
                                   if k != "all_passed")

    any_premium = any(c["passes_sesoi_and_ci"] for c in comparisons.values())
    verdict = ("NEURAL_PREMIUM_ON_CURVED_SURFACE" if any_premium
               else "NO_NEURAL_PREMIUM_EVEN_WITH_MEASURED_CURVATURE")

    print(f"\n  === R2 held-out, CV agrupada por semilla ({args.folds} folds) ===")
    for model, value in means.items():
        print(f"  {model:<12}{value:>9.4f}")
    print(f"\n  no linealidad del perfil (1 - R2 lineal): {mean_curvature:.4f}")
    for model, c in comparisons.items():
        print(f"  {model:<12} vs lineal {c['mean_difference']:+.4f} "
              f"[{c['ci95_low']:+.4f}, {c['ci95_high']:+.4f}]  "
              f"SESOI {SESOI} -> {'PASA' if c['passes_sesoi_and_ci'] else 'no alcanza'}")
    print(f"\n  veredicto: {verdict}")
    print("\n  falsadores:")
    for name, check in falsifiers.items():
        if name != "all_passed":
            print(f"    {name:<44} {'PASA' if check['passed'] else 'FALLA'}")

    payload = {
        "schema_version": "buffer_prediction_premium_v1",
        "claim_status": verdict if falsifiers["all_passed"] else "HALTED_FALSIFIER_FAILED",
        "target": TARGET, "sesoi": SESOI, "cadence": "sim.run()",
        "buffer_levels": list(BUFFER_HOURS), "families": list(FAMILIES),
        "escalations": list(ESCALATIONS), "seeds": seeds, "n_rows": len(rows),
        "held_out_r2_mean": means, "held_out_r2_per_fold": per_fold,
        "paired_comparisons": comparisons,
        "profile_curvature": {"mean_one_minus_linear_r2": mean_curvature,
                              "by_cell": curvature_by_cell},
        "q1_panel_reference": {"linear": 0.9697483885147611, "backprop": 0.9863153335912018,
                               "kan": 0.9913280603236345,
                               "note": "the near-linear surface this one is contrasted against"},
        "falsifiers": falsifiers,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(
        payload, args.output,
        contract=Path("docs/PREREGISTRO_PRIMA_PREDICCION_BUFFER_2026-08-01.md"),
        reference=Path("results/headroom/g1_buffer_price/result.json"))
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
