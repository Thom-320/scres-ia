#!/usr/bin/env python3
"""The prediction premium on the Cobb-Douglas surface -- the one that actually curves.

Three repairs an external review demanded, all of them real:

1. THE BASELINE WAS ADDITIVE. Features were buffer plus one-hot family plus one-hot escalation,
   with no `buffer x family` or `buffer x escalation` products. A network could then "win" merely
   by representing interactions that a better CLASSICAL model also represents. The premium is now
   measured against the best of {additive linear, linear with interactions, quadratic}.

2. THE INTERVAL USED 1.96 ON FIVE FOLDS. With 4 degrees of freedom the correct multiplier is
   t(0.975, 4) = 2.776, so the old interval was too narrow. The sign does not change; the
   arithmetic was still wrong.

3. THE "CURVATURE BELOW NOISE" CLAIM WAS NOT MEASURED. I compared 0.0763, a lack-of-fit statistic
   on profile MEANS, against 0.3174, predictive error on individual EPISODES. Those are different
   scales and the comparison does not support the claim. It is withdrawn and replaced by the
   quantity that IS comparable: the CELL-MEAN ORACLE, a model that knows each cell's true mean.
   Its R^2 is the ceiling any function class can reach, so `R2_oracle - R2_best_classical` is the
   maximum premium available. If that gap is below the SESOI, no model can earn one.

And the target is Cobb-Douglas, because the corrected G1 showed it is the surface with a strictly
interior optimum while ret_excel and flow_fill_rate merely saturate. The previous premium run
measured the monotone surface and was framed as if it were the curved one.

See `docs/PREREGISTRO_PRIMA_CD_2026-08-01.md`, committed before this ran.
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
from supply_chain.cobb_douglas_resilience import (  # noqa: E402
    UNIT_COSTS, CobbDouglasRecorder, derive_exponents, resilience_index)
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
SESOI = 0.05
DAILY_DEMAND = 2_500.0
SEED_BASE = 6_900_001
STEP = 24.0
T_CRIT = {4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 9: 2.262}      # t(0.975, df)


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


def base_features(buf: float, family: str, escalation: str) -> list[float]:
    return [buf / 1344.0,
            *[1.0 if family == f else 0.0 for f in FAMILIES],
            *[1.0 if escalation == e else 0.0 for e in ESCALATIONS]]


def rich_features(buf: float, family: str, escalation: str) -> list[float]:
    """Additive terms PLUS interactions and a quadratic -- the honest classical competitor."""
    b = buf / 1344.0
    fam = [1.0 if family == f else 0.0 for f in FAMILIES]
    esc = [1.0 if escalation == e else 0.0 for e in ESCALATIONS]
    return [b, b * b, *fam, *esc, *[b * f for f in fam], *[b * e for e in esc]]


def episode(risks, mult: float, buf: float, seed: int, horizon: float):
    sim = MFSCSimulation(
        shifts=1,
        initial_buffers={"op3_rm": 0.0, "op5_rm": 0.0,
                         "op9_rations": buf * DAILY_DEMAND / 24.0},
        inventory_replenishment_period=0.0, seed=seed, horizon=horizon,
        risks_enabled=True, risk_level="current", enabled_risks=set(risks),
        risk_frequency_multipliers_by_id=({r: float(mult) for r in risks}
                                          if mult != 1.0 else None),
        cssu_topology_mode="split_v1", cssu_service_rule="FIFO_PARTIAL",
        cssu_reallocate_unused=False,
        order_fulfillment_mode="op9_linked", op9_dispatch_policy="fixed_clock_daily",
        strict_exogenous_crn=True, year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"], r14_defect_mode=P["r14_defect_mode"])
    recorder = CobbDouglasRecorder(period_hours=STEP, costs=dict(UNIT_COSTS))
    for _ in range(int(round(horizon / STEP))):
        sim.step(step_hours=STEP)
        recorder.sample(sim)
    panel = compute_episode_metrics(sim)
    return recorder.aggregate(), float(panel["ret_excel_risk_conditional"])


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return 1.0 - (ss_res / ss_tot if ss_tot > 0 else 1.0)


def ols(x_tr, y_tr, x_te):
    coef, *_ = np.linalg.lstsq(np.c_[x_tr, np.ones(len(x_tr))], y_tr, rcond=None)
    return np.c_[x_te, np.ones(len(x_te))] @ coef


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--horizon-weeks", type=int, default=52)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--output", type=Path,
                    default=Path("results/headroom/cd_surface_prediction_premium/result.json"))
    args = ap.parse_args()
    horizon = float(args.horizon_weeks * HOURS_PER_WEEK)
    seeds = [SEED_BASE + i for i in range(args.seeds)]
    started = time.perf_counter()

    cells, ret_by_row, index = {}, [], []
    for family in FAMILIES:
        for escalation, mult in ESCALATIONS.items():
            for buf in BUFFER_HOURS:
                for seed in seeds:
                    agg, ret = episode(FAMILY_RISKS[family], mult, buf, seed, horizon)
                    cells[(family, escalation, buf, seed)] = agg
                    ret_by_row.append(ret)
                    index.append((family, escalation, buf, seed))
        print(f"  {family} listo ({time.perf_counter() - started:.0f}s)", flush=True)

    # kappa_dot is SET-RELATIVE and the maxima that set the exponents are too, so building the
    # target over all rows -- as the first draft did -- leaks the test fold into its own label.
    # An external review caught this before it ran. The target is therefore constructed INSIDE
    # each fold from TRAINING rows only, and the frozen transform is applied to the test rows.
    aggs = [cells[k] for k in index]
    y_ret = np.array(ret_by_row)
    x_base = np.array([base_features(b, f, e) for (f, e, b, _) in index])
    x_rich = np.array([rich_features(b, f, e) for (f, e, b, _) in index])
    g = np.array([s for (_, _, _, s) in index])
    cell_key = [(f, e, b) for (f, e, b, _) in index]

    def target_from_training(train_idx: np.ndarray) -> tuple[np.ndarray, dict]:
        """Freeze exponents and the kappa normaliser on TRAIN, then score every row with them."""
        maxima = {v: max(max(aggs[i][v] for i in train_idx), 1.0 + 1e-9)
                  for v in ("zeta", "epsilon", "phi", "tau")}
        maxima["kappa_dot"] = float(len(train_idx))
        exps = derive_exponents(maxima)
        total = float(sum(aggs[i]["kappa"] for i in train_idx))
        scale = float(len(train_idx)) / total if total > 0 else 1.0
        values = np.array([
            resilience_index({"zeta": a["zeta"], "epsilon": a["epsilon"], "phi": a["phi"],
                              "tau": a["tau"], "kappa_dot": max(a["kappa"] * scale, 1e-9)},
                             exps)["R_cobb_douglas"]
            for a in aggs])
        return values, exps

    folds = grouped_folds(g, n_folds=args.folds)
    models = ("constant", "linear_additive", "linear_interactions", "spline_buffer",
              "tree", "oracle_cell_mean", "backprop", "kan")
    per_fold: dict[str, list[float]] = {m: [] for m in models}
    exponents_by_fold = []

    def spline_features(rows: list[int]) -> np.ndarray:
        """Piecewise-linear in the buffer with knots at the quartiles -- a classical curve fit."""
        knots = [336.0, 672.0, 1008.0]
        out = []
        for i in rows:
            f, e, b, _ = index[i]
            out.append([*base_features(b, f, e),
                        *[max(0.0, (b - k) / 1344.0) for k in knots]])
        return np.asarray(out)

    def tree_predict(x_tr, y_tr, x_te, depth: int = 4):
        """A small CART grown on the same features: the classical rule the review asked for."""
        def build(idx, d):
            if d == 0 or len(idx) < 8 or float(y_tr[idx].std()) < 1e-12:
                return float(y_tr[idx].mean())
            best = None
            for col in range(x_tr.shape[1]):
                values = np.unique(x_tr[idx, col])
                for thr in values[:-1]:
                    left = idx[x_tr[idx, col] <= thr]
                    right = idx[x_tr[idx, col] > thr]
                    if len(left) < 4 or len(right) < 4:
                        continue
                    sse = float(((y_tr[left] - y_tr[left].mean()) ** 2).sum()
                                + ((y_tr[right] - y_tr[right].mean()) ** 2).sum())
                    if best is None or sse < best[0]:
                        best = (sse, col, thr, left, right)
            if best is None:
                return float(y_tr[idx].mean())
            _, col, thr, left, right = best
            return (col, thr, build(left, d - 1), build(right, d - 1))

        tree = build(np.arange(len(y_tr)), depth)

        def walk(node, row):
            while isinstance(node, tuple):
                col, thr, lo, hi = node
                node = lo if row[col] <= thr else hi
            return node

        return np.array([walk(tree, r) for r in x_te])

    for fi, (tr, te) in enumerate(folds):
        y, exps = target_from_training(tr)
        exponents_by_fold.append(exps)
        cell_mean = {}
        for key in set(cell_key[i] for i in tr):
            rows = [i for i in tr if cell_key[i] == key]
            cell_mean[key] = float(y[rows].mean())
        # The oracle knows each cell's mean FROM TRAINING ROWS ONLY -- a test cell unseen in
        # training falls back to the global training mean rather than to its own value.
        y_oracle_te = np.array([cell_mean.get(cell_key[i], float(y[tr].mean())) for i in te])

        per_fold["constant"].append(r2(y[te], np.full(len(te), y[tr].mean())))
        per_fold["linear_additive"].append(r2(y[te], ols(x_base[tr], y[tr], x_base[te])))
        per_fold["linear_interactions"].append(r2(y[te], ols(x_rich[tr], y[tr], x_rich[te])))
        sp_tr, sp_te = spline_features(list(tr)), spline_features(list(te))
        per_fold["spline_buffer"].append(r2(y[te], ols(sp_tr, y[tr], sp_te)))
        per_fold["tree"].append(r2(y[te], tree_predict(x_base[tr], y[tr], x_base[te])))
        per_fold["oracle_cell_mean"].append(r2(y[te], y_oracle_te))
        pred, _ = fit_mlp(x_base[tr], y[tr], x_base[te], seed=2000 + fi, classify=False)
        per_fold["backprop"].append(r2(y[te], pred))
        try:
            pred_k, _ = fit_kan(x_base[tr], y[tr], x_base[te], seed=2000 + fi, classify=False)
            per_fold["kan"].append(r2(y[te], pred_k))
        except Exception as exc:                        # pragma: no cover
            per_fold["kan"].append(float("nan"))
            print(f"  KAN no disponible ({exc})", flush=True)
    means = {m: float(np.nanmean(v)) for m, v in per_fold.items()}

    classical = ("linear_additive", "linear_interactions", "spline_buffer", "tree")
    best_classical = max(classical, key=lambda m: means[m])
    t_crit = T_CRIT.get(args.folds - 1, 2.776)

    def paired(model: str, baseline: str) -> dict:
        d = np.array(per_fold[model]) - np.array(per_fold[baseline])
        d = d[~np.isnan(d)]
        se = float(d.std(ddof=1) / np.sqrt(d.size)) if d.size > 1 else 0.0
        low = float(d.mean() - t_crit * se)
        return {"baseline": baseline, "mean_difference": float(d.mean()),
                "ci95_low": low, "ci95_high": float(d.mean() + t_crit * se),
                "t_critical": t_crit, "df": int(d.size - 1), "sesoi": SESOI,
                "passes_sesoi_and_ci": bool(d.mean() >= SESOI and low > 0)}

    comparisons = {m: paired(m, best_classical) for m in ("backprop", "kan")}
    available = paired("oracle_cell_mean", best_classical)

    y_diag, _ = target_from_training(np.arange(len(index)))
    diag_mean = {}
    for key in set(cell_key):
        rows = [i for i, k in enumerate(cell_key) if k == key]
        diag_mean[key] = float(y_diag[rows].mean())
    curv = {}
    for family in FAMILIES:
        for escalation in ESCALATIONS:
            prof = [diag_mean[(family, escalation, b)] for b in BUFFER_HOURS]
            xs, ys = np.array(BUFFER_HOURS), np.array(prof)
            fit = np.polyval(np.polyfit(xs, ys, 1), xs)
            ss_tot = float(((ys - ys.mean()) ** 2).sum())
            curv[f"{family}|{escalation}"] = (float(((ys - fit) ** 2).sum() / ss_tot)
                                              if ss_tot > 0 else 0.0)
    strict_interior = {}
    for family in FAMILIES:
        for escalation in ESCALATIONS:
            prof = [diag_mean[(family, escalation, b)] for b in BUFFER_HOURS]
            top = max(prof)
            best = [BUFFER_HOURS[i] for i, v in enumerate(prof) if top - v <= 1e-12]
            strict_interior[f"{family}|{escalation}"] = all(
                b not in (BUFFER_HOURS[0], BUFFER_HOURS[-1]) for b in best)

    prior_seeds = seeds_used_by_sealed_artifacts(exclude=args.output)
    falsifiers = {
        "f1_target_is_the_curved_surface": {
            "passed": any(strict_interior.values()),
            "evidence": {"why_it_can_fail": ("the previous premium run measured ret_excel, which "
                                             "the corrected G1 showed is MONOTONE, while framing "
                                             "it as the curved surface. If Cobb-Douglas has no "
                                             "strictly interior optimum here either, this run "
                                             "repeats that mistake"),
                         "strictly_interior_by_cell": strict_interior,
                         "profile_lack_of_fit": curv}},
        "f2_baseline_includes_interactions": {
            "passed": (x_rich.shape[1] > x_base.shape[1]
                       and len(classical) >= 4),
            "evidence": {"why_it_can_fail": ("an additive baseline lets a network 'win' by "
                                             "representing interactions a better classical model "
                                             "also represents; the premium must be over the BEST "
                                             "classical competitor"),
                         "n_features_additive": int(x_base.shape[1]),
                         "n_features_rich": int(x_rich.shape[1]),
                         "classical_competitors": list(classical),
                         "best_classical": best_classical,
                         "r2_by_classical": {m: means[m] for m in classical}}},
        "f3_inference_uses_t_not_normal": {
            "passed": abs(t_crit - 1.96) > 0.1,
            "evidence": {"why_it_can_fail": ("1.96 on five folds understates the interval; with "
                                             "4 df the multiplier is 2.776. The old run used "
                                             "1.96 and its intervals were too narrow"),
                         "t_critical": t_crit, "df": args.folds - 1}},
        "f4_available_premium_is_measured_not_asserted": {
            "passed": available["mean_difference"] >= -1e-9,
            "evidence": {"why_it_can_fail": (
                             "I previously compared 0.0763, a lack-of-fit on profile MEANS, "
                             "against 0.3174, predictive error on individual EPISODES, and "
                             "claimed curvature sits below noise. Different scales, claim "
                             "withdrawn. The cell-mean ORACLE is the comparable quantity: its "
                             "R2 is the ceiling any function of these features can reach, so "
                             "this gap is the maximum premium ANY model could earn"),
                         "oracle_minus_best_classical": available}},
        "f7_target_is_built_from_training_rows_only": {
            "passed": len(exponents_by_fold) == len(folds)
                      and any(exponents_by_fold[0] != e for e in exponents_by_fold[1:]),
            "evidence": {"why_it_can_fail": (
                             "kappa_dot is set-relative and the maxima that set the exponents "
                             "are too, so building the target over ALL rows leaks the test fold "
                             "into its own label. An external review caught this before the run. "
                             "Exponents and the kappa normaliser are now frozen on TRAIN inside "
                             "each fold -- if they were identical across folds the freezing "
                             "would not be happening"),
                         "exponents_by_fold": exponents_by_fold}},
        "f5_folds_grouped_by_seed": {
            "passed": all(not (set(g[tr].tolist()) & set(g[te].tolist())) for tr, te in folds),
            "evidence": {"why_it_can_fail": "a shared seed would inflate every R2"}},
        "f6_seeds_are_virgin": {
            "passed": not (set(seeds) & prior_seeds),
            "evidence": {"why_it_can_fail": "reuse would void the comparison", "seeds": seeds,
                         "collisions": sorted(set(seeds) & prior_seeds)}},
    }
    falsifiers["all_passed"] = all(v["passed"] for k, v in falsifiers.items()
                                   if k != "all_passed")

    any_premium = any(c["passes_sesoi_and_ci"] for c in comparisons.values())
    premium_possible = available["mean_difference"] >= SESOI
    verdict = ("NEURAL_PREMIUM_ON_CD_SURFACE" if any_premium
               else "NO_PREMIUM_AND_NONE_WAS_AVAILABLE" if not premium_possible
               else "PREMIUM_WAS_AVAILABLE_AND_NOT_CAPTURED")

    print(f"\n  === R2 held-out sobre Cobb-Douglas, CV agrupada por semilla ===")
    for m, v in means.items():
        print(f"  {m:<22}{v:>9.4f}")
    print(f"\n  mejor clásico: {best_classical}")
    print(f"  prima DISPONIBLE (oráculo − mejor clásico): {available['mean_difference']:+.4f} "
          f"[{available['ci95_low']:+.4f}, {available['ci95_high']:+.4f}]")
    for m, c in comparisons.items():
        print(f"  {m:<12} {c['mean_difference']:+.4f} "
              f"[{c['ci95_low']:+.4f}, {c['ci95_high']:+.4f}] "
              f"-> {'PASA' if c['passes_sesoi_and_ci'] else 'no alcanza SESOI'}")
    print(f"\n  veredicto: {verdict}")
    print("\n  falsadores:")
    for name, check in falsifiers.items():
        if isinstance(check, dict):
            print(f"    {name:<46} {'PASA' if check['passed'] else 'FALLA'}")

    payload = {
        "schema_version": "cd_surface_prediction_premium_v1",
        "claim_status": verdict if falsifiers["all_passed"] else "HALTED_FALSIFIER_FAILED",
        "target": "R_cobb_douglas", "sesoi": SESOI, "cadence_hours": STEP,
        "buffer_levels": list(BUFFER_HOURS), "families": list(FAMILIES),
        "escalations": list(ESCALATIONS), "seeds": seeds, "n_rows": len(index),
        "held_out_r2_mean": means, "held_out_r2_per_fold": per_fold,
        "best_classical": best_classical,
        "available_premium_oracle_minus_classical": available,
        "neural_comparisons": comparisons,
        "profile_lack_of_fit": curv, "strictly_interior_by_cell": strict_interior,
        "exponents_by_fold": exponents_by_fold,
        "supersedes": ("results/headroom/buffer_prediction_premium/result.json -- same question "
                       "on the monotone ret_excel surface, with an additive baseline and 1.96 "
                       "intervals"),
        "falsifiers": falsifiers,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(
        payload, args.output,
        contract=Path("docs/PREREGISTRO_PRIMA_CD_2026-08-01.md"),
        reference=Path("results/headroom/g1_buffer_price/result.json"))
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
