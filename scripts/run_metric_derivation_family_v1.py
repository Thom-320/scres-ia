#!/usr/bin/env python3
"""Every defensible derivation of Garrido's two metrics, measured on one surface.

The PI asked which derivation reaches the bar. Trying variants until one crosses and reporting
only that one is p-hacking with metric degrees of freedom. So the whole space is declared in the
preregistration, every member is measured here, every member is reported, and Holm-Bonferroni over
K = 162 is paid.

Every axis moves a parameter Garrido chose or repairs a documented defect:

  * `0.5/RP` exceeds 1 whenever RP < 0.5 h -- one order scored 73.9 -- because RP is in hours and
    the coefficient is dimensionless. That is an error, not a preference.
  * `Re^min = 0` makes the non-recovery branch score exactly zero always, so `(DP-RP)/CT` never
    enters the index at all.
  * The Excel formula carries no `CT <= LT` guard; the thesis Eq. 5.5 does. Both are his.
  * His section 6.2 asks for drivers his own index did not consider.

One DES pass per (config, context, seed) yields every ReT variant, because they all read the same
per-order records.

Contract: docs/PREREGISTRO_FAMILIA_DERIVACIONES_METRICA_2026-08-06.md
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import itertools
import json
import math
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
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.ret_thesis import (  # noqa: E402
    compute_fill_rate_from_orders,
    compute_order_level_ret_excel_request_snapshot_ledger,
    order_has_ret_risk_indicator,
)
from supply_chain.seed_custody import custody_falsifier, module_manifest  # noqa: E402
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402

R1R = ("R11", "R12", "R13", "R14")
R2R = ("R21", "R22", "R23", "R24")
CONTEXTS = {
    "R1r": (R1R, {}), "R2r": (R2R, {}), "R1r+R2r": (R1R + R2R, {}),
    "R1r|esc": (R1R, {r: 3.0 for r in R1R}),
    "R2r|esc": (R2R, {r: 3.0 for r in R2R}),
    "R1r+R2r|esc": (R1R + R2R, {r: 3.0 for r in R1R + R2R}),
}
FACTORS = {
    "buffer_hours": (0.0, 168.0, 336.0, 504.0, 672.0, 1344.0),
    "shifts": (1, 2, 3),
    "op9_rop": (12.0, 24.0, 36.0, 48.0),
    "op12_rop": (12.0, 24.0, 36.0, 48.0),
}
CONFIGS = tuple(dict(zip(FACTORS, c)) for c in itertools.product(*FACTORS.values()))
SEED_BASE, WEEKS = 5_300_001, 52
GATE = 0.05

#: The declared family. Order matters only for reporting.
AXES = {
    "population": ("visible", "full_ledger", "risk_conditional"),
    "guard": ("excel", "thesis"),
    "recovery": ("raw", "dimensionless", "bounded"),
    "re_min": (0.0, 0.25),
    "clip": ("none", "unit"),
    "aggregation": ("order", "quantity"),
}
VARIANTS = [dict(zip(AXES, combo)) for combo in itertools.product(*AXES.values())]
MODULES = ("supply_chain/supply_chain.py", "supply_chain/ret_thesis.py",
           "supply_chain/episode_metrics.py", "supply_chain/seed_custody.py")


def order_rows(sim) -> list[dict]:
    """Per-order records, with the POPULATION and the Bt/Ut snapshot taken from the contract.

    Three attempts at reimplementing the ledger failed f4 in three different ways -- naive running
    counts, the wrong population, and reading the Excel branch as a CT-vs-LT guard. The lesson is
    not to reimplement it: `compute_order_level_ret_excel_request_snapshot_ledger` already defines
    which rows are visible and what Bt/Ut each one saw, so this calls it and joins back to the
    order objects for the AP/RP/DP the declared axes need. Only the axes vary; the ledger does not.
    """
    all_orders = list(getattr(sim, "orders", []) or [])
    fill = compute_fill_rate_from_orders(all_orders)
    ledger = compute_order_level_ret_excel_request_snapshot_ledger(all_orders)
    by_j = {int(getattr(o, "j", 0) or 0): o for o in all_orders}
    risk_cases = {"excel_autotomy", "excel_recovery", "excel_risk_no_recovery"}

    rows = []
    for lr in ledger["ret_rows"]:
        o = by_j.get(int(lr["j"]))
        if o is None:
            continue
        ct = getattr(o, "CTj", None)
        rows.append({
            "j": int(lr["j"]), "delivered": True, "lost": False,
            "ct": None if ct is None else float(ct),
            "lt": float(getattr(o, "LTj", 48.0) or 48.0),
            "ap": float(getattr(o, "APj", 0.0) or 0.0),
            "rp": float(getattr(o, "RPj", 0.0) or 0.0),
            "dp": float(getattr(o, "DPj", 0.0) or 0.0),
            "qty": float(lr["quantity"]), "fill": fill,
            "bt": int(lr["sum_bt"]), "ut": int(lr["sum_ut"]),
            "risk": str(lr["case"]) in risk_cases,
            "ledger_ret": float(lr["ret"]), "ledger_case": str(lr["case"]),
        })
    # full_ledger scores every GENERATED order; the ones the ledger omits are the unserved, at 0.
    visible_j = {r["j"] for r in rows}
    for o in all_orders:
        j = int(getattr(o, "j", 0) or 0)
        if j in visible_j:
            continue
        rows.append({"j": j, "delivered": False, "lost": bool(getattr(o, "lost", False)),
                     "ct": None, "lt": float(getattr(o, "LTj", 48.0) or 48.0),
                     "ap": 0.0, "rp": 0.0, "dp": 0.0,
                     "qty": float(getattr(o, "quantity", 0.0) or 0.0), "fill": fill,
                     "bt": 0, "ut": 0, "risk": False,
                     "ledger_ret": 0.0, "ledger_case": "excel_unfulfilled"})
    return rows


def score_order(row: dict, v: dict) -> tuple[float, str]:
    """One order under one variant.

    The two guards are genuinely different formulas, not one with an extra condition:

      excel  -- IF(AVG(risk)>0, IF(APj>0, APj/LT, 0.5/RPj), 1-((Bt+Ut)/j)).  The branch is on
                whether risk was ACTIVE. A no-risk order goes to the running fill-rate term and
                scores near 1. Reading this as a CT-vs-LT branch sends every order through
                recovery and collapses the episode from 0.55 to 0.002 -- f4 caught exactly that.
      thesis -- Eq. 5.5's four branches, gated on CT vs LT, with the episode-level Re(FR_t).
    """
    if not row["delivered"]:
        return 0.0, "unfulfilled"
    ap, rp, dp, lt, ct = row["ap"], row["rp"], row["dp"], row["lt"], row["ct"]

    def recovery_term() -> float:
        if rp <= 0.0:
            return 0.0
        if v["recovery"] == "raw":
            return 0.5 * (1.0 / rp)                 # his coefficient, RP in hours
        if v["recovery"] == "dimensionless":
            return 0.5 * (lt / rp)                  # RP measured in lead times
        return 0.5 / (1.0 + rp / lt)                # bounded by 0.5, monotone in lateness

    if v["guard"] == "excel":
        if row["risk"]:
            if ap > 0.0:
                value, branch = ap / max(lt, 1e-9), "autotomy"
            else:
                value, branch = recovery_term(), "recovery"
        else:
            value = 1.0 - (row["bt"] + row["ut"]) / max(row["j"], 1)
            branch = "fill_rate"
    else:
        if ap > 0.0 and ct is not None and ct <= lt:
            value, branch = 1.0 * (ap / max(lt, 1e-9)), "autotomy"
        elif ct is not None and ct > lt and rp > 0.0:
            value, branch = recovery_term(), "recovery"
        elif ct is not None and ct > lt:
            value = float(v["re_min"]) * ((dp - rp) / max(ct, 1e-9))
            branch = "non_recovery"
        else:
            value, branch = float(row["fill"]), "fill_rate"

    if v["clip"] == "unit":
        return min(1.0, max(0.0, value)), branch
    return max(0.0, value), branch


def score_episode(rows: list[dict], v: dict) -> float:
    if v["population"] == "visible":
        pool = [r for r in rows if r["delivered"] and not r["lost"]]
    elif v["population"] == "full_ledger":
        pool = rows
    else:
        pool = [r for r in rows if r["delivered"] and r["risk"]]
    if not pool:
        return 0.0
    scored = [(score_order(r, v)[0], r["qty"]) for r in pool]
    if v["aggregation"] == "order":
        return float(np.mean([s for s, _ in scored]))
    total = sum(q for _, q in scored)
    return float(sum(s * q for s, q in scored) / total) if total > 0 else 0.0


def episode(config, context, seed, horizon):
    risks, freq = CONTEXTS[context]
    sim = MFSCSimulation(
        shifts=int(config["shifts"]),
        initial_buffers={"op3_rm": 0.0, "op5_rm": 0.0,
                         "op9_rations": float(config["buffer_hours"]) * 2_500.0 / 24.0},
        inventory_replenishment_period=0.0, seed=seed, horizon=horizon,
        risks_enabled=True, risk_level="current", enabled_risks=set(risks),
        risk_frequency_multipliers_by_id=dict(freq) or None,
        strict_exogenous_crn=True, year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"], r14_defect_mode=P["r14_defect_mode"])
    sim.params["op9_rop"] = float(config["op9_rop"])
    sim.params["op12_rop"] = float(config["op12_rop"])
    sim.run()
    rows = order_rows(sim)
    base = {"population": "visible", "guard": "excel", "recovery": "raw",
            "re_min": 0.0, "clip": "none", "aggregation": "order"}
    # Per-order identity against the contract's own ledger: far sharper than comparing means,
    # which can agree while every individual row is wrong.
    worst = max((abs(score_order(r, base)[0] - r["ledger_ret"])
                 for r in rows if r["delivered"]), default=0.0)
    return [score_episode(rows, v) for v in VARIANTS], float(worst)


def h_regime(per_ctx) -> float:
    stacked = []
    for mean in per_ctx:
        m = np.asarray(mean, dtype=float)
        lo, hi = m.min(), m.max()
        stacked.append((m - lo) / (hi - lo) if hi > lo else np.zeros_like(m))
    s = np.stack(stacked)
    return float(s.max(axis=1).mean() - s.mean(axis=0).max())


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted values, monotone-enforced."""
    k = len(pvalues)
    order = sorted(range(k), key=lambda i: pvalues[i])
    adj, running = [0.0] * k, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (k - rank) * pvalues[i]))
        adj[i] = running
    return adj


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--replay-of", required=True)
    ap.add_argument("--output", type=Path,
                    default=Path("results/metric_derivation_family/result.json"))
    args = ap.parse_args()
    started = time.perf_counter()
    seeds = [SEED_BASE + i for i in range(args.seeds)]
    horizon = float(WEEKS * HOURS_PER_WEEK)
    contexts = list(CONTEXTS)
    rng = np.random.default_rng(20260806)
    print(f"  {len(VARIANTS)} variantes x {len(CONFIGS)} configs x {len(contexts)} contextos "
          f"x {len(seeds)} semillas", flush=True)

    surface = {}          # (ctx, seed) -> array (variants, configs)
    worst_anchor_dev = 0.0
    for ctx in contexts:
        for seed in seeds:
            cols = []
            for cfg in CONFIGS:
                v, dev = episode(cfg, ctx, seed, horizon)
                cols.append(v)
                worst_anchor_dev = max(worst_anchor_dev, dev)
            surface[(ctx, seed)] = np.array(cols).T
        print(f"  {ctx} listo ({time.perf_counter() - started:.0f}s)", flush=True)

    rows = []
    for vi, v in enumerate(VARIANTS):
        per_ctx = [np.mean([surface[(c, s)][vi] for s in seeds], axis=0) for c in contexts]
        point = h_regime(per_ctx)
        draws = np.array([
            h_regime([np.mean([surface[(c, seeds[i])][vi]
                               for i in rng.integers(0, len(seeds), len(seeds))], axis=0)
                      for c in contexts]) for _ in range(args.n_boot)])
        lcb = float(np.percentile(draws, 2.5))
        # One-sided p for H0: H_regime <= GATE, from the bootstrap distribution.
        pval = float(np.mean(draws <= GATE))
        rows.append({**v, "H_regime": point, "lcb95_raw": lcb, "p_one_sided": pval})

    adj = holm([r["p_one_sided"] for r in rows])
    for r, a in zip(rows, adj):
        r["p_holm"] = a
        r["passes"] = bool(r["lcb95_raw"] >= GATE and a < 0.05)

    winners = [r for r in rows if r["passes"]]
    best = max(rows, key=lambda r: r["H_regime"])
    verdict = ("DEFENSIBLE_DERIVATION_REACHES_THE_BAR" if winners
               else "NO_DEFENSIBLE_DERIVATION_REACHES_THE_BAR")

    anchor_check = {"max_abs_per_order_deviation": worst_anchor_dev,
                    "baseline_variant": "visible/excel/raw/re_min=0/no-clip/order-mean",
                    "reference": "compute_order_level_ret_excel_request_snapshot_ledger"}

    falsifiers = {
        "f2_all_variants_reported": {
            "passed": len(rows) == len(VARIANTS) == 144,
            "evidence": {"why_it_can_fail": "a variant dropped in silence would make the "
                                            "multiplicity correction too weak and the family "
                                            "incomplete", "n": len(rows)}},
        "f3_the_family_separates": {
            "passed": len({round(r["H_regime"], 9) for r in rows}) > 1,
            "evidence": {"why_it_can_fail": "identical H across every variant would mean we are "
                                            "measuring the estimator, not the metric"}},
        "f4_baseline_is_identical_to_the_contract_ledger": {
            "passed": worst_anchor_dev < 1e-12,
            "evidence": {"why_it_can_fail": "the baseline variant must reproduce the contract's "
                                            "ledger ORDER BY ORDER, not just in the mean -- means "
                                            "can agree while every row is wrong. Three earlier "
                                            "reimplementations failed this three different ways",
                         **anchor_check}},
        "f5_multiplicity_applied": {
            "passed": all("p_holm" in r for r in rows),
            "evidence": {"why_it_can_fail": "an uncorrected LCB95 compared against the bar would "
                                            "be exactly the shopping this design exists to avoid",
                         "method": "Holm-Bonferroni", "k": len(rows)}},
        "f6_no_fresh_seeds": custody_falsifier(seeds, replay_of=args.replay_of,
                                               exclude=args.output),
    }
    falsifiers["all_passed"] = all(
        v["passed"] for k, v in falsifiers.items()
        if k != "all_passed" and isinstance(v, dict) and not v.get("not_applicable"))

    print(f"\n  === las 10 mejores de {len(rows)} ===")
    for r in sorted(rows, key=lambda r: -r["H_regime"])[:10]:
        tag = " <-- PASA" if r["passes"] else ""
        print(f"    H {r['H_regime']:+.5f} lcb {r['lcb95_raw']:+.5f} holm {r['p_holm']:.3f}  "
              f"{r['population']:<16}{r['guard']:<7}{r['recovery']:<14}"
              f"min{r['re_min']} {r['clip']:<5}{r['aggregation']}{tag}")
    print(f"\n  ancla f4 · máxima desviación por pedido: {worst_anchor_dev:.2e}")
    print(f"\n  veredicto: {verdict}   (máximo {best['H_regime']:+.5f} contra umbral {GATE})\n")
    for name, f in falsifiers.items():
        if name == "all_passed" or not isinstance(f, dict):
            continue
        label = "NO APLICA" if f.get("not_applicable") else ("PASA" if f["passed"] else "FALLA")
        print(f"    {name:<44} {label}")

    payload = {
        "schema_version": "metric_derivation_family_v1",
        "claim_status": verdict,
        "scope": "DEVELOPMENT_ON_BURNED_TAPES_NO_ADJUDICATION",
        "run_role": "CACHE_ANALYSIS", "replay_of": args.replay_of,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "module_manifest": module_manifest(MODULES, script=__file__),
        "gate": GATE, "multiplicity": "Holm-Bonferroni", "k_variants": len(rows),
        "axes": {k: list(v) for k, v in AXES.items()},
        "seeds": seeds, "contexts": contexts, "n_configurations": len(CONFIGS),
        "variants": rows, "best": best, "winners": winners,
        "anchor_check": anchor_check, "falsifiers": falsifiers,
        "commitment": ("The full table enters the manuscript whichever variant wins; a crossing "
                       "variant is reported with its position in the family and its correction."),
        "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(payload, args.output, contract=args.contract,
                            reference=Path("results/endpoint_headroom_atlas/result.json"))
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
