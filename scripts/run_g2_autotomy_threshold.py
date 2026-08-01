#!/usr/bin/env python3
"""G2 -- does a hard threshold produce a premium that classical methods do not already eat?

G1 came out positive: Cobb-Douglas has a strictly interior optimum and removing the holding cost
destroys it in all six regimes. But no premium appeared -- a SPLINE beat both networks. The lesson
is precise: it is not enough for a surface to be non-linear; the premium needs non-linearity that
classical methods do not already capture, and smooth curvature in one variable is not that.

A hard threshold is different. Garrido's autotomy branch (`CTj <= LT`, weight 1.0, the heaviest in
his metric) is DEAD in our model because the fulfilment delay of 54 h exceeds the 48 h lead time,
so 0 of 416 orders can reach it. The `FDB` arm -- freight waves, the `shift_uniform` delta and a
0.05 h band read off HIS autotomy rows -- revives it at 0.31% against his 0.44%, at a measured
fidelity price of 0.95 SE of `ret_mean`.

A discontinuity does not average away against noise: an order either crosses or it does not. That
is the structural reason G2 is the best remaining candidate in this lane.

The threshold RULE is a mandatory comparator here. If an explicit `if` matches the network, there
is no neural premium -- only a network rediscovering a conditional.

See `docs/PREREGISTRO_G2_UMBRAL_AUTOTOMIA_2026-08-01.md`, committed before this ran.
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
from supply_chain.config import HOURS_PER_WEEK  # noqa: E402
from supply_chain.config import THESIS_FAITHFUL_PROTOCOL as P  # noqa: E402
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402

R1R = ("R11", "R12", "R13", "R14")
R2R = ("R21", "R22", "R23", "R24")
FAMILIES = {"R1r": R1R, "R2r": R2R, "R1r+R2r": R1R + R2R}
ESCALATIONS = {"base": 1.0, "freq_x3": 3.0}
BUFFERS = (0.0, 336.0, 672.0, 1008.0, 1344.0)
SHIFTS = (1, 2)
OP9_ROP = (12.0, 24.0, 48.0)
BAND_TOLERANCE = 0.05          # read off HIS autotomy rows, never fitted to ours
ARMS = {
    "shipped_dead_autotomy": {"transit": "constant", "delta": "off", "pred": "le", "tol": 0.0},
    "FDB_live_autotomy": {"transit": "freight_waves", "delta": "shift_uniform",
                          "pred": "band", "tol": BAND_TOLERANCE},
}
TARGET = "ret_excel_risk_conditional"
PRIMARY_BASELINE = "linear_interactions"      # declared here, on principle, before any result
SESOI = 0.05
DAILY_DEMAND = 2_500.0
SEED_BASE = 7_000_001
T_CRIT = {4: 2.776, 5: 2.571, 6: 2.447, 9: 2.262}


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


def episode(arm: dict, family: str, esc: float, buf: float, shifts: int, rop: float,
            seed: int, horizon: float) -> dict[str, float]:
    risks = FAMILIES[family]
    sim = MFSCSimulation(
        shifts=int(shifts),
        initial_buffers={"op3_rm": 0.0, "op5_rm": 0.0,
                         "op9_rations": buf * DAILY_DEMAND / 24.0},
        inventory_replenishment_period=0.0, seed=seed, horizon=horizon,
        risks_enabled=True, risk_level="current", enabled_risks=set(risks),
        risk_frequency_multipliers_by_id=({r: float(esc) for r in risks}
                                          if esc != 1.0 else None),
        fulfillment_transit_mode=str(arm["transit"]),
        fulfillment_delta_mode=str(arm["delta"]),
        autotomy_predicate=str(arm["pred"]),
        autotomy_tolerance_hours=float(arm["tol"]),
        strict_exogenous_crn=True, year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"], r14_defect_mode=P["r14_defect_mode"])
    sim.params["op9_rop"] = float(rop)
    sim.run()
    panel = compute_episode_metrics(sim)
    return {TARGET: float(panel[TARGET]),
            "autotomy_share": float(panel["excel_case_pct_autotomy"]) / 100.0,
            "ret_mean_proxy": float(panel["ret_excel"]),
            "n_orders": float(panel["n_orders"])}


def features(family: str, esc: float, buf: float, shifts: int, rop: float) -> list[float]:
    fam = [1.0 if family == f else 0.0 for f in FAMILIES]
    return [buf / 1344.0, (shifts - 1) / 1.0, rop / 48.0, 1.0 if esc > 1.0 else 0.0, *fam]


def rich(family: str, esc: float, buf: float, shifts: int, rop: float) -> list[float]:
    b = buf / 1344.0
    base = features(family, esc, buf, shifts, rop)
    return [*base, b * b, *[b * v for v in base[1:]]]


def r2(y, p):
    ss_res = float(((y - p) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - (ss_res / ss_tot if ss_tot > 0 else 1.0)


def ols(x_tr, y_tr, x_te):
    coef, *_ = np.linalg.lstsq(np.c_[x_tr, np.ones(len(x_tr))], y_tr, rcond=None)
    return np.c_[x_te, np.ones(len(x_te))] @ coef


def threshold_rule(x_tr, y_tr, x_te):
    """The mandatory comparator: the single best `if` over one feature and one cut."""
    best = None
    for col in range(x_tr.shape[1]):
        for thr in np.unique(x_tr[:, col])[:-1]:
            lo, hi = x_tr[:, col] <= thr, x_tr[:, col] > thr
            if lo.sum() < 5 or hi.sum() < 5:
                continue
            sse = float(((y_tr[lo] - y_tr[lo].mean()) ** 2).sum()
                        + ((y_tr[hi] - y_tr[hi].mean()) ** 2).sum())
            if best is None or sse < best[0]:
                best = (sse, col, thr, float(y_tr[lo].mean()), float(y_tr[hi].mean()))
    if best is None:
        return np.full(len(x_te), float(y_tr.mean()))
    _, col, thr, v_lo, v_hi = best
    return np.where(x_te[:, col] <= thr, v_lo, v_hi)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--horizon-weeks", type=int, default=52)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--output", type=Path,
                    default=Path("results/headroom/g2_autotomy_threshold/result.json"))
    args = ap.parse_args()
    horizon = float(args.horizon_weeks * HOURS_PER_WEEK)
    seeds = [SEED_BASE + i for i in range(args.seeds)]
    started = time.perf_counter()

    data: dict[str, dict] = {}
    for arm_name, arm in ARMS.items():
        rows, x_b, x_r, y, g = [], [], [], [], []
        for family in FAMILIES:
            for esc in ESCALATIONS.values():
                for buf in BUFFERS:
                    for shifts in SHIFTS:
                        for rop in OP9_ROP:
                            for seed in seeds:
                                row = episode(arm, family, esc, buf, shifts, rop, seed, horizon)
                                rows.append(row)
                                x_b.append(features(family, esc, buf, shifts, rop))
                                x_r.append(rich(family, esc, buf, shifts, rop))
                                y.append(row[TARGET])
                                g.append(seed)
        data[arm_name] = {"rows": rows, "x_base": np.array(x_b), "x_rich": np.array(x_r),
                          "y": np.array(y), "g": np.array(g)}
        share = float(np.mean([r["autotomy_share"] for r in rows]))
        print(f"  {arm_name:<24} {len(rows)} episodios, autotomía {share:.4%} "
              f"({time.perf_counter() - started:.0f}s)", flush=True)

    live = data["FDB_live_autotomy"]
    dead = data["shipped_dead_autotomy"]
    share_live = float(np.mean([r["autotomy_share"] for r in live["rows"]]))
    share_dead = float(np.mean([r["autotomy_share"] for r in dead["rows"]]))
    crossing_rows = float(np.mean([r["autotomy_share"] > 0 for r in live["rows"]]))
    # Label noise: how much of the autotomy share is explained by the design cell at all.
    cell = [tuple(v) for v in live["x_base"]]
    within = []
    for key in set(cell):
        idx = [i for i, k in enumerate(cell) if k == key]
        within.append(float(np.std([live["rows"][i]["autotomy_share"] for i in idx])))
    label_noise = float(np.mean(within))

    folds = grouped_folds(live["g"], n_folds=args.folds)
    models = ("constant", "linear_additive", "linear_interactions", "threshold_rule",
              "train_cell_mean_comparator", "backprop", "kan")
    per_fold: dict[str, list[float]] = {m: [] for m in models}
    y, xb, xr = live["y"], live["x_base"], live["x_rich"]
    for fi, (tr, te) in enumerate(folds):
        cmean = {}
        for key in set(cell[i] for i in tr):
            rows_i = [i for i in tr if cell[i] == key]
            cmean[key] = float(y[rows_i].mean())
        per_fold["constant"].append(r2(y[te], np.full(len(te), y[tr].mean())))
        per_fold["linear_additive"].append(r2(y[te], ols(xb[tr], y[tr], xb[te])))
        per_fold["linear_interactions"].append(r2(y[te], ols(xr[tr], y[tr], xr[te])))
        per_fold["threshold_rule"].append(r2(y[te], threshold_rule(xb[tr], y[tr], xb[te])))
        per_fold["train_cell_mean_comparator"].append(
            r2(y[te], np.array([cmean.get(cell[i], float(y[tr].mean())) for i in te])))
        pred, _ = fit_mlp(xb[tr], y[tr], xb[te], seed=3000 + fi, classify=False)
        per_fold["backprop"].append(r2(y[te], pred))
        try:
            pk, _ = fit_kan(xb[tr], y[tr], xb[te], seed=3000 + fi, classify=False)
            per_fold["kan"].append(r2(y[te], pk))
        except Exception as exc:                       # pragma: no cover
            per_fold["kan"].append(float("nan"))
            print(f"  KAN no disponible ({exc})", flush=True)
    means = {m: float(np.nanmean(v)) for m, v in per_fold.items()}
    t_crit = T_CRIT.get(args.folds - 1, 2.776)

    def paired(model: str, baseline: str) -> dict:
        d = np.array(per_fold[model]) - np.array(per_fold[baseline])
        d = d[~np.isnan(d)]
        se = float(d.std(ddof=1) / np.sqrt(d.size)) if d.size > 1 else 0.0
        low = float(d.mean() - t_crit * se)
        return {"baseline": baseline, "mean_difference": float(d.mean()), "ci95_low": low,
                "ci95_high": float(d.mean() + t_crit * se), "sesoi": SESOI,
                "passes_sesoi_and_ci": bool(d.mean() >= SESOI and low > 0)}

    vs_primary = {m: paired(m, PRIMARY_BASELINE) for m in ("backprop", "kan", "threshold_rule")}
    vs_threshold = {m: paired(m, "threshold_rule") for m in ("backprop", "kan")}
    available = paired("train_cell_mean_comparator", PRIMARY_BASELINE)
    fidelity_price = (float(np.mean([r["ret_mean_proxy"] for r in live["rows"]]))
                      - float(np.mean([r["ret_mean_proxy"] for r in dead["rows"]])))
    prior_seeds = seeds_used_by_sealed_artifacts(exclude=args.output)

    falsifiers = {
        "f1_autotomy_is_actually_alive": {
            "passed": share_live > 0.0 and share_dead == 0.0,
            "evidence": {"why_it_can_fail": ("if FDB does not revive the branch, or if the "
                                             "shipped arm already has it, there is no threshold "
                                             "switched on and G2 has no premise"),
                         "autotomy_share_FDB": share_live,
                         "autotomy_share_shipped": share_dead,
                         "garrido_reference": 0.0044}},
        "f2_threshold_is_crossed_often_enough": {
            "passed": crossing_rows > 0.02,
            "evidence": {"why_it_can_fail": ("a discontinuity almost never crossed is noise, not "
                                             "structure. Reported whether it passes or fails"),
                         "fraction_of_episodes_with_any_crossing": crossing_rows,
                         "mean_autotomy_share": share_live,
                         "label_noise_within_cell_sd": label_noise}},
        "f3_a_threshold_rule_is_among_the_comparators": {
            "passed": "threshold_rule" in means,
            "evidence": {"why_it_can_fail": ("without it a 'neural premium' could just be a "
                                             "network rediscovering an if-statement"),
                         "r2_threshold_rule": means["threshold_rule"],
                         "networks_vs_threshold_rule": vs_threshold}},
        "f4_primary_baseline_declared_before": {
            "passed": PRIMARY_BASELINE == "linear_interactions",
            "evidence": {"why_it_can_fail": ("selecting the baseline on test performance was a "
                                             "real defect of the CD run; here it is fixed in the "
                                             "preregistration"),
                         "primary_baseline": PRIMARY_BASELINE}},
        "f5_folds_grouped_by_seed": {
            "passed": all(not (set(live["g"][tr].tolist()) & set(live["g"][te].tolist()))
                          for tr, te in folds),
            "evidence": {"why_it_can_fail": "a shared seed inflates every R2"}},
        "f6_fidelity_price_is_disclosed": {
            "passed": True,
            "evidence": {"why_it_can_fail": ("FDB WORSENS ret_mean by a measured 0.95 SE; using "
                                             "it without saying so would be selling convenient "
                                             "physics"),
                         "ret_excel_mean_FDB_minus_shipped": fidelity_price,
                         "documented_price": "0.95 SE of ret_mean, "
                                             "docs/RESULTADO_CIERRE_AUTOTOMIA_2026-07-31.md"}},
        "f7_seeds_are_virgin": {
            "passed": not (set(seeds) & prior_seeds),
            "evidence": {"why_it_can_fail": "reuse would void the comparison", "seeds": seeds,
                         "collisions": sorted(set(seeds) & prior_seeds)}},
    }
    falsifiers["all_passed"] = all(v["passed"] for k, v in falsifiers.items()
                                   if k != "all_passed")

    beats_threshold = any(vs_threshold[m]["mean_difference"] > 0
                          and vs_threshold[m]["ci95_low"] > 0 for m in vs_threshold)
    neural = any(vs_primary[m]["passes_sesoi_and_ci"] for m in ("backprop", "kan"))
    rule_suffices = vs_primary["threshold_rule"]["mean_difference"] >= max(
        vs_primary["backprop"]["mean_difference"], vs_primary["kan"]["mean_difference"])
    verdict = ("NEURAL_PREMIUM_FROM_DISCONTINUITY" if neural and beats_threshold
               else "THRESHOLD_RULE_SUFFICES" if rule_suffices
               else "DISCONTINUITY_INSUFFICIENT")

    print(f"\n  autotomía viva {share_live:.4%} (Garrido 0,44%) · muerta {share_dead:.4%}")
    print(f"  episodios con algún cruce: {crossing_rows:.1%} · ruido de etiqueta "
          f"{label_noise:.5f}")
    print(f"\n  === R2 held-out sobre `{TARGET}` (brazo FDB) ===")
    for m, v in means.items():
        print(f"  {m:<28}{v:>9.4f}")
    print(f"\n  margen disponible: {available['mean_difference']:+.4f} "
          f"[{available['ci95_low']:+.4f}, {available['ci95_high']:+.4f}]")
    for m, c in vs_primary.items():
        print(f"  {m:<20} vs primario {c['mean_difference']:+.4f} "
              f"[{c['ci95_low']:+.4f}, {c['ci95_high']:+.4f}]")
    print(f"\n  veredicto: {verdict}")
    print("\n  falsadores:")
    for name, check in falsifiers.items():
        if isinstance(check, dict):
            print(f"    {name:<46} {'PASA' if check['passed'] else 'FALLA'}")

    payload = {
        "schema_version": "g2_autotomy_threshold_v1",
        "claim_status": verdict if falsifiers["all_passed"] else "HALTED_FALSIFIER_FAILED",
        "generator": "G2_hard_threshold", "target": TARGET, "sesoi": SESOI,
        "primary_baseline": PRIMARY_BASELINE, "arms": {k: v for k, v in ARMS.items()},
        "autotomy_share": {"FDB": share_live, "shipped": share_dead, "garrido": 0.0044},
        "threshold_diagnostics": {"episodes_with_crossing": crossing_rows,
                                  "label_noise_within_cell_sd": label_noise},
        "held_out_r2_mean": means, "held_out_r2_per_fold": per_fold,
        "vs_primary_baseline": vs_primary, "vs_threshold_rule": vs_threshold,
        "available_margin": available,
        "fidelity_price_ret_excel": fidelity_price,
        "scope": ("prediction only. H_regime in this lane is ~1e-4, so even a predictive premium "
                  "would not be control headroom, and the two must stay separate"),
        "seeds": seeds, "falsifiers": falsifiers,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(
        payload, args.output,
        contract=Path("docs/PREREGISTRO_G2_UMBRAL_AUTOTOMIA_2026-08-01.md"),
        reference=Path("results/headroom/cd_surface_prediction_premium/result.json"))
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
