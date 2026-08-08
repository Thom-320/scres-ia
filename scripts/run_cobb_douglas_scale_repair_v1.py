#!/usr/bin/env python3
"""Repair Garrido's index for its two measured scale defects, and report what that buys.

Contract: docs/PREREGISTRO_COBB_DOUGLAS_REPARACION_ESCALA_2026-08-08.md, amended by
docs/ENMIENDA_FALSADOR_F5_REPARACION_ESCALA_2026-08-08.md. Both were committed before this file
existed.

DEFECT A -- under his assumption (6) the cost term is not a cost term. All seven coefficients at 1
is harmless when the decision variables share a scale; in his APP model inventory is ~4.5x
production per period. In ours it is 181x, so over the 10,368 cells of the burned cache inventory
holds 85.650% of kappa and backorders 13.734%, and corr(kappa, zeta + epsilon) = 0.999993. With the
derived exponents that puts an effective -0.368 on ln(zeta) where his construction puts +0.014:
sign inverted, 26x the magnitude. His own section 5 relaxes assumption (6). Axis D relaxes it far
enough to matter -- the economic grid we already have stops at x5, and at c_i = 0.5 inventory still
holds 62.9% of kappa.

DEFECT B -- the rule `share/ln(x_max)` makes the exponent inverse to dynamic range. Published, that
gives kappa_dot 7.38x zeta. On our maxima it gives tau 47.5x and kappa_dot 31.3x, because both are
ratios near 1 and the rule equates each argument at its maximum against a floor of x = 1 that
neither ever approaches. Axis E equalises over the RANGE instead, `share/(ln x_max - ln x_min)`,
which reduces exactly to his rule when x_min = 1 -- f3 proves that against his five published
numbers, and it is what makes this his index repaired rather than a metric of ours.

THE PREDICTION, RECORDED BEFORE RUNNING (f4): over_range should LOWER H_regime, because it
de-weights kappa_dot -- the only component carrying any headroom, 0.00187 -- toward zeta and phi,
whose measured headroom is exactly 0.

FIRST PASS RETIRED, AND WHY. The first execution returned SCALE_REPAIR_REACHES_THE_BAR on four
`holding_decoupled` variants that respected the share bound. It is kept at
`results/cobb_douglas_scale_repair/result.RETIRED_incomplete_independence_gate.json` because a
withdrawn result is retained and labelled, never deleted. It was withdrawn because f2 as
implemented was narrower than f2 as written: it asked only whether `scale_neutral` decoupled from
zeta, when the contract's own defect statement names epsilon -- "the cost term is not a cost term:
it is zeta + epsilon again" -- and f2's own text predicts this failure verbatim. Widened to every
variable at every cost level, the measurement says `holding_decoupled` moves kappa_dot's
correlation with ln zeta from +0.976 to +0.175 and its correlation with ln epsilon from +0.218 to
+0.968. It does not decouple the cost term; it re-points the duplicate at the variable with the
widest log-range. A repair cannot be credited with a crossing it produced that way, so
independence joins the share bound as a disqualification.

Development on burned tapes. Adjudicates nothing, opens no seeds.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import glob
import itertools
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.cobb_douglas_resilience import (  # noqa: E402
    COST_COMPONENT_KEYS, GARRIDO_2024_EXPONENTS, SIGNS, kappa_dot, validate_costs,
)
from supply_chain.seed_custody import custody_falsifier, module_manifest  # noqa: E402

BASE_VARS = ("zeta", "epsilon", "phi", "tau", "kappa_dot")
SERVICE_VAR = "upsilon"
SIGNS_EXT = dict(SIGNS, upsilon=-1.0)
FLOORS = {"zeta": 1.0, "epsilon": 1.0, "phi": 1.0, "tau": 1.0,
          "upsilon": 1.0, "kappa_dot": 1e-9}

AXES = {
    "costs": ("garrido_c1", "holding_decoupled", "scale_neutral"),
    "exponents": ("at_max", "over_range"),
    "variables": ("his_five", "no_tau", "plus_service"),
    "kappa_set": ("within", "global"),
}
#: The six cells that already exist in the sealed family, and are therefore not new search.
DUPLICATES_OF_SEALED = {("garrido_c1", "at_max")}

GATE, N_BOOT = 0.05, 200
#: Declared in f2 of the preregistration. A kappa_dot that correlates above this with another term
#: of the same index is that term again under a different name, not an independent cost variable.
INDEPENDENCE_MAX_ABS_CORR = 0.90
K_ALREADY_POOLED = 158                       # family A (144) + family B (14 scored, 4 excluded)
MODULES = ("supply_chain/cobb_douglas_resilience.py", "supply_chain/arm_runner.py",
           "supply_chain/seed_custody.py")

#: His own maxima, read off his published exponents by inverting `0.20/ln(x_max)`. Used only by f3.
HIS_MAXIMA = {v: math.exp(0.20 / e) for v, e in GARRIDO_2024_EXPONENTS.items()}


def variable_set(level: str) -> tuple[str, ...]:
    if level == "his_five":
        return BASE_VARS
    if level == "no_tau":
        return tuple(v for v in BASE_VARS if v != "tau")
    return BASE_VARS + (SERVICE_VAR,)


def share_for(variables) -> float:
    """His 0.20 is 1/5 because he has five variables; it moves with the count, not with taste."""
    return 1.0 / len(variables)


def exponents_from(maxima, minima, variables, scheme: str) -> dict[str, float]:
    """`at_max` is his rule. `over_range` is his rule with the x_min = 1 assumption removed.

    A term's contribution to DISCRIMINATING between configurations is its span over the comparison
    set, `a * (ln x_max - ln x_min)`, not its value at the maximum against a floor. The two
    coincide exactly when x_min = 1, which is why over_range reproduces his published numbers on
    his own data and diverges on ours, where tau and kappa_dot never come near the floor.
    """
    share = share_for(variables)
    out: dict[str, float] = {}
    for v in variables:
        hi = float(maxima[v])
        if scheme == "at_max":
            span = math.log(hi)
        else:
            lo = max(float(minima[v]), FLOORS[v])
            span = math.log(hi) - math.log(lo)
        if span <= 0.0:
            raise ValueError(f"{v}: span {span:.6g} <= 0; {share:.4f}/span is undefined")
        out[v] = share / span
    return out


def cost_vector(level: str, means: dict[str, float]) -> dict[str, float]:
    """Three cost vectors, none with a free parameter -- all three fall out of the cache.

    `garrido_c1` is his assumption (6). `holding_decoupled` is the single-parameter minimum that
    removes the domination and nothing else: it prices a unit of held inventory at the ratio that
    makes the holding bill equal the production bill on average. `scale_neutral` does that for all
    four active components, so no term dominates kappa by unit magnitude -- which is what
    "isolating kappa_dot from the influence of the cost parameters" would require to hold.
    """
    costs = {k: 1.0 for k in COST_COMPONENT_KEYS}
    if level == "garrido_c1":
        return costs
    if level == "holding_decoupled":
        costs["c_i"] = means["mean_regular_production"] / means["mean_inventory"]
        return costs
    for cost_key, comp in COST_COMPONENT_KEYS.items():
        m = means.get(comp, 0.0)
        costs[cost_key] = (1.0 / m) if m > 0 else 1.0   # inert components stay at 1
    return costs


def score(values, exponents, variables) -> np.ndarray:
    linear = np.zeros(len(values[variables[0]]))
    for v in variables:
        floored = np.maximum(np.asarray(values[v], dtype=float), FLOORS[v])
        linear = linear + SIGNS_EXT[v] * exponents[v] * np.log(floored)
    return 1.0 / (1.0 + np.exp(-linear))


def h_regime(per_ctx) -> float:
    norm = []
    for m in per_ctx:
        m = np.asarray(m, dtype=float)
        lo, hi = m.min(), m.max()
        norm.append((m - lo) / (hi - lo) if hi > lo else np.zeros_like(m))
    stacked = np.stack(norm)
    return float(stacked.max(axis=1).mean() - stacked.mean(axis=0).max())


def pearson(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    xc, yc = x - x.mean(), y - y.mean()
    den = math.sqrt(float((xc * xc).sum()) * float((yc * yc).sum()))
    return float((xc * yc).sum() / den) if den > 0 else 0.0


def load_grid(aggregates_path: Path, surface_dir: Path):
    raw = json.loads(aggregates_path.read_text())
    aggregates = {(k.rsplit("|", 1)[0], int(k.rsplit("|", 1)[1])): v for k, v in raw.items()}
    panel = {}
    for path in glob.glob(str(surface_dir / "**" / "*.json"), recursive=True):
        d = json.loads(Path(path).read_text())
        if d.get("context") is None or not d.get("cells"):
            continue
        panel[(d["context"], int(d["seed"]))] = d["cells"]
    cells = sorted(set(aggregates) & set(panel))
    if not cells:
        raise SystemExit(f"{aggregates_path} and {surface_dir} share no (context, seed)")
    n_cfg = len(aggregates[cells[0]])
    for key in cells:
        if len(panel[key]) != n_cfg or len(aggregates[key]) != n_cfg:
            raise SystemExit(f"{key}: {len(panel[key])} surface vs {len(aggregates[key])} aggregate")
    return aggregates, panel, cells, n_cfg


def run_grid(name, aggregates, panel, cells, n_cfg, rng):
    contexts = sorted({c for c, _ in cells})
    seeds = sorted({s for _, s in cells})

    # Component means over the whole grid -- the cost vectors are derived from these, once, and
    # are therefore a property of the declared comparison set rather than of any one variant.
    comps = list(COST_COMPONENT_KEYS.values())
    means = {c: float(np.mean([a[c] for key in cells for a in aggregates[key]])) for c in comps}

    # DEFECT A, measured on this grid before any variant is scored.
    shares = {c: [] for c in comps}
    for key in cells:
        for a in aggregates[key]:
            k = sum(a[c] for c in comps)
            for c in comps:
                shares[c].append(a[c] / k if k > 0 else 0.0)
    share_mean = {c: float(np.mean(v)) for c, v in shares.items()}
    flat_kappa_c1 = [sum(a[c] for c in comps) for key in cells for a in aggregates[key]]
    flat_zeta = [a["zeta"] for key in cells for a in aggregates[key]]
    flat_eps = [a["epsilon"] for key in cells for a in aggregates[key]]
    defect_a = {
        "kappa_component_share_under_c1": share_mean,
        "corr_kappa_zeta": pearson(flat_kappa_c1, flat_zeta),
        "corr_kappa_zeta_plus_epsilon": pearson(flat_kappa_c1,
                                                [z + e for z, e in zip(flat_zeta, flat_eps)]),
        "inventory_to_production_ratio": means["mean_inventory"] / means["mean_regular_production"],
    }

    variants, decoupling = [], {}
    for cost_lvl, exp_lvl, var_lvl, kap_lvl in itertools.product(*AXES.values()):
        variables = variable_set(var_lvl)
        costs = validate_costs(cost_vector(cost_lvl, means))

        kappa = {key: [sum(costs[ck] * a[cc] for ck, cc in COST_COMPONENT_KEYS.items())
                       for a in aggregates[key]] for key in cells}
        flat = [k for key in cells for k in kappa[key]]
        gscale = len(flat) / sum(flat) if sum(flat) else 1.0

        surfaces = {}
        for key in cells:
            kd_w = kappa_dot({str(i): k for i, k in enumerate(kappa[key])})
            kd = ([kd_w[str(i)] for i in range(n_cfg)] if kap_lvl == "within"
                  else [k * gscale for k in kappa[key]])
            surfaces[key] = {
                "zeta": [float(a["zeta"]) for a in aggregates[key]],
                "epsilon": [float(a["epsilon"]) for a in aggregates[key]],
                "phi": [float(a["phi"]) for a in aggregates[key]],
                "tau": [float(a["tau"]) for a in aggregates[key]],
                SERVICE_VAR: [float(c["panel"]["demanded_rations"] - c["panel"]["delivered_rations"])
                              for c in panel[key]],
                "kappa_dot": kd,
            }

        # f2: does re-pricing actually make kappa_dot an INDEPENDENT term?
        #
        # The first pass asked this only of `scale_neutral` and only against zeta, which is
        # narrower than what f2's own text declares: "re-weighting toward epsilon can leave kappa
        # tracking epsilon instead of decoupling it. If it fails, axis D does not repair what it
        # says it repairs." The preregistration names epsilon explicitly, in the sentence that
        # states the defect -- "the cost term is not a cost term: it is zeta + epsilon again". So
        # the check runs against EVERY other variable in the index, at every cost level. Widening
        # it is implementing the contract, not moving its goalposts.
        if (cost_lvl, kap_lvl) not in decoupling:
            fk = [math.log(max(k, FLOORS["kappa_dot"]))
                  for key in cells for k in surfaces[key]["kappa_dot"]]
            against = {}
            for other in BASE_VARS:
                if other == "kappa_dot":
                    continue
                flat_other = [math.log(max(v, FLOORS[other]))
                              for key in cells for v in surfaces[key][other]]
                against[other] = pearson(fk, flat_other)
            worst = max(against, key=lambda v: abs(against[v]))
            decoupling[(cost_lvl, kap_lvl)] = {
                "corr_ln_kappa_dot_vs": against,
                "corr_ln_kappa_dot_ln_zeta": against["zeta"],
                "corr_ln_kappa_dot_ln_epsilon": against["epsilon"],
                "worst_duplicate": worst,
                "worst_abs_corr": abs(against[worst]),
                "independent": bool(abs(against[worst]) < INDEPENDENCE_MAX_ABS_CORR),
                "costs": costs,
            }

        maxima = {v: max(max(surfaces[k][v]) for k in cells) for v in variables}
        minima = {v: min(min(surfaces[k][v]) for k in cells) for v in variables}
        row = {"costs": cost_lvl, "exponents": exp_lvl, "variables": var_lvl,
               "kappa_set": kap_lvl, "n_variables": len(variables),
               "is_new": (cost_lvl, exp_lvl) not in DUPLICATES_OF_SEALED}
        try:
            used = exponents_from(maxima, minima, variables, exp_lvl)
        except ValueError as exc:
            variants.append({**row, "H_regime": None, "excluded": str(exc)})
            continue

        share = share_for(variables)
        term_max = {v: abs(used[v] * math.log(max(maxima[v], FLOORS[v]))) for v in variables}
        per_ctx = [np.mean([score(surfaces[(c, s)], used, variables) for s in seeds], axis=0)
                   for c in contexts]
        point = h_regime(per_ctx)
        draws = np.asarray([
            h_regime([np.mean([score(surfaces[(c, s)], used, variables)
                               for s in (seeds[i] for i in rng.integers(0, len(seeds), len(seeds)))],
                              axis=0) for c in contexts])
            for _ in range(N_BOOT)])
        variants.append({
            **row, "H_regime": point, "lcb95": float(np.percentile(draws, 2.5)),
            "p_not_above_gate": float(np.mean(draws <= GATE)),
            "respects_share_bound": bool(all(t <= share + 1e-9 for t in term_max.values())),
            "share_bound": share, "max_term_magnitude": max(term_max.values()),
            "term_magnitudes": term_max, "exponents_used": used,
            "cost_vector": costs,
        })

    return {"grid": name, "contexts": contexts, "seeds": seeds, "n_configs": n_cfg,
            "n_cells": len(cells), "component_means": means, "defect_a": defect_a,
            "decoupling": {f"{c}|{k}": v for (c, k), v in decoupling.items()},
            "variants": variants}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--amendment", type=Path, required=True)
    ap.add_argument("--replay-of", required=True)
    ap.add_argument("--sealed-family", type=Path,
                    default=Path("results/cobb_douglas_variant_family/result.json"))
    ap.add_argument("--output", type=Path,
                    default=Path("results/cobb_douglas_scale_repair/result.json"))
    args = ap.parse_args()
    started = time.perf_counter()
    rng = np.random.default_rng(20260808)

    grids = {}
    for name, agg, surf in (
        ("wrap288_v1", "results/cobb_douglas_component_headroom/aggregates.json",
         "results/surface_cache/wrap288_v1"),
        ("wrap288_compat_extended_v1",
         "results/cobb_douglas_component_headroom_extended/aggregates.json",
         "results/surface_cache/wrap288_compat_extended_v1"),
    ):
        print(f"\n  rejilla {name}")
        a, p, c, n = load_grid(Path(agg), Path(surf))
        print(f"    {len(c)} celdas · {n} configuraciones")
        grids[name] = run_grid(name, a, p, c, n, rng)
        d = grids[name]["defect_a"]
        print(f"    cuota de inventario en kappa (c=1) : {d['kappa_component_share_under_c1']['mean_inventory']:.5f}")
        print(f"    corr(kappa, zeta+epsilon)          : {d['corr_kappa_zeta_plus_epsilon']:.6f}")
        print(f"    inventario / produccion            : {d['inventory_to_production_ratio']:.1f}x")

    # PRIMARY GRID. The extended grid is where headroom could live; the 288 is the control the
    # amendment defines. Multiplicity is paid on the primary, over the pooled family.
    primary = grids["wrap288_compat_extended_v1"]
    live = [v for v in primary["variants"] if v.get("H_regime") is not None]
    new_live = [v for v in live if v["is_new"]]

    k = K_ALREADY_POOLED + len(new_live)
    order = sorted(range(len(new_live)), key=lambda i: new_live[i]["p_not_above_gate"])
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (k - rank) * new_live[idx]["p_not_above_gate"]))
        new_live[idx]["holm_adjusted_p"] = running

    # Independence is a property of the cost vector and the kappa set, so it attaches to every
    # variant sharing that pair.
    for v in live:
        dec_row = primary["decoupling"][f"{v['costs']}|{v['kappa_set']}"]
        v["kappa_dot_independent"] = dec_row["independent"]
        v["kappa_dot_worst_duplicate"] = dec_row["worst_duplicate"]
        v["kappa_dot_worst_abs_corr"] = dec_row["worst_abs_corr"]

    crossing = [v for v in new_live if v["lcb95"] >= GATE and v["holm_adjusted_p"] < 0.05]
    winners = [v for v in crossing
               if v["respects_share_bound"] and v["kappa_dot_independent"]]
    best = max(live, key=lambda v: v["H_regime"])

    verdict = ("SCALE_REPAIR_REACHES_THE_BAR" if winners
               else "ONLY_NON_INDEPENDENT_COST_TERMS_REACH_THE_BAR"
               if crossing and all(not v["kappa_dot_independent"] for v in crossing)
               else "ONLY_MISSCALED_VARIANTS_REACH_THE_BAR_AGAIN" if crossing
               else "SCALE_REPAIR_DOES_NOT_CREATE_HEADROOM")

    # ---- falsifiers -----------------------------------------------------------------------
    def cell(grid, **sel):
        for v in grids[grid]["variants"]:
            if all(v.get(kk) == vv for kk, vv in sel.items()):
                return v
        return {}

    anchor = cell("wrap288_v1", costs="garrido_c1", exponents="at_max",
                  variables="his_five", kappa_set="within")
    sealed_h = None
    if args.sealed_family.exists():
        for v in json.loads(args.sealed_family.read_text())["variants"]:
            if (v.get("exponents") == "ours" and v.get("variables") == "his_five"
                    and v.get("kappa_set") == "within"):
                sealed_h = v
    anchor_ok = bool(
        sealed_h is not None and anchor
        and abs(anchor["H_regime"] - sealed_h["H_regime"]) < 1e-9
        and abs(anchor["lcb95"] - sealed_h["lcb95"]) < 1e-9
        and abs(anchor["max_term_magnitude"] - sealed_h["max_term_magnitude"]) < 1e-9
        and anchor["respects_share_bound"] == sealed_h["respects_share_bound"])

    # f3: over_range with x_min = 1 must reproduce his five published exponents on his maxima.
    his_repro = exponents_from(HIS_MAXIMA, {v: 1.0 for v in BASE_VARS}, BASE_VARS, "over_range")
    f3_err = {v: abs(his_repro[v] - GARRIDO_2024_EXPONENTS[v]) for v in BASE_VARS}

    # f4: the recorded prediction -- over_range should LOWER H_regime, paired within cost/vars/set.
    paired = []
    for v in live:
        if v["exponents"] != "over_range":
            continue
        mate = cell("wrap288_compat_extended_v1", costs=v["costs"], exponents="at_max",
                    variables=v["variables"], kappa_set=v["kappa_set"])
        if mate.get("H_regime") is not None:
            paired.append({"costs": v["costs"], "variables": v["variables"],
                           "kappa_set": v["kappa_set"], "at_max": mate["H_regime"],
                           "over_range": v["H_regime"],
                           "delta": v["H_regime"] - mate["H_regime"]})
    n_down = sum(1 for p in paired if p["delta"] < 0)

    dec = primary["decoupling"]
    f2_vals = {kk: {"worst_duplicate": vv["worst_duplicate"],
                    "worst_abs_corr": vv["worst_abs_corr"],
                    "independent": vv["independent"]} for kk, vv in dec.items()}

    falsifiers = {
        "f1_premise_A_holds": {
            "passed": all(g["defect_a"]["kappa_component_share_under_c1"]["mean_inventory"] >= 0.50
                          for g in grids.values()),
            "evidence": {"why_it_can_fail": "if kappa were not dominated by inventory under c=1 "
                                            "there would be no defect A and the whole cost axis "
                                            "would be repairing nothing",
                         "per_grid": {n: g["defect_a"] for n, g in grids.items()}}},
        "f2_the_repair_decouples": {
            "passed": any(v["independent"] for kk, v in f2_vals.items()
                          if not kk.startswith("garrido_c1")),
            "evidence": {"why_it_can_fail": "epsilon spans 6.40 in ln against zeta's 1.28, so "
                                            "re-weighting toward epsilon can leave kappa tracking "
                                            "epsilon instead of decoupling it. That is the failure "
                                            "this falsifier named in advance, and it is the one "
                                            "that occurred",
                         "threshold": INDEPENDENCE_MAX_ABS_CORR,
                         "checked_against": [v for v in BASE_VARS if v != "kappa_dot"],
                         "per_cost_level": f2_vals, "detail": dec}},
        "f3_over_range_reduces_to_his_rule": {
            "passed": all(e < 5e-4 for e in f3_err.values()),
            "evidence": {"why_it_can_fail": "this is the check that over_range is HIS rule with a "
                                            "degenerate assumption removed rather than a metric of "
                                            "ours. If it does not reproduce his five published "
                                            "numbers on his own maxima with x_min = 1, the reading "
                                            "of his rule is wrong and axis E is withdrawn",
                         "his_maxima_recovered": HIS_MAXIMA,
                         "reproduced": his_repro, "published": GARRIDO_2024_EXPONENTS,
                         "abs_error": f3_err}},
        "f4_direction_was_predicted_first": {
            "passed": True,
            "evidence": {"prediction_recorded_in": str(args.contract),
                         "prediction": "over_range LOWERS H_regime relative to at_max",
                         "why_it_matters": "the prediction runs against what we would like, because "
                                           "over_range de-weights kappa_dot -- the only component "
                                           "with H > 0 -- toward zeta and phi, whose H is 0",
                         "pairs_measured": len(paired), "pairs_that_fell": n_down,
                         "prediction_held": n_down == len(paired) and len(paired) > 0,
                         "pairs": paired},
            "note": ("recorded, not gating: a violated prediction is reported as a violation, "
                     "never re-read as a confirmation")},
        "f5a_baseline_reproduces_the_sealed_family": {
            "passed": anchor_ok,
            "evidence": {"why_it_can_fail": "the baseline cell IS the sealed family's "
                                            "ours/his_five/within variant. If the reimplementation "
                                            "does not return its exact numbers, the 30 new "
                                            "variants are not comparable with the 158 old ones and "
                                            "the combined table means nothing",
                         "amended_by": str(args.amendment),
                         "sealed": {kk: sealed_h.get(kk) for kk in
                                    ("H_regime", "lcb95", "max_term_magnitude",
                                     "respects_share_bound")} if sealed_h else None,
                         "reimplemented": {kk: anchor.get(kk) for kk in
                                           ("H_regime", "lcb95", "max_term_magnitude",
                                            "respects_share_bound")}}},
        "f5b_manufactured_headroom_is_declared_as_misscaling": {
            "passed": True,
            "evidence": {"why_it_can_fail": "a repair that lifts H on the control grid while "
                                            "breaking its own share bound is a scale error, not a "
                                            "finding; one that lifts it while respecting the bound "
                                            "is a finding and is reported as one",
                         "control_grid": "wrap288_v1",
                         "lifted_breaking_bound": [
                             {kk: v[kk] for kk in ("costs", "exponents", "variables", "kappa_set",
                                                   "H_regime", "max_term_magnitude", "share_bound")}
                             for v in grids["wrap288_v1"]["variants"]
                             if v.get("H_regime") and v["H_regime"] > 1e-9
                             and not v["respects_share_bound"]],
                         "lifted_respecting_bound": [
                             {kk: v[kk] for kk in ("costs", "exponents", "variables", "kappa_set",
                                                   "H_regime", "max_term_magnitude", "share_bound")}
                             for v in grids["wrap288_v1"]["variants"]
                             if v.get("H_regime") and v["H_regime"] > 1e-9
                             and v["respects_share_bound"]]}},
        "f6_crossers_survive_both_disqualifications": {
            "passed": all(v["respects_share_bound"] and v["kappa_dot_independent"]
                          for v in crossing),
            "evidence": {"why_it_can_fail": "two ways, and either is fatal to a crossing. The rule "
                                            "share/ln(x_max) exists so no term exceeds its share at "
                                            "its own maximum, and a crosser that breaks it is his "
                                            "index MISAPPLIED -- that is what disqualified the four "
                                            "crossers of the sealed family. And a crosser whose "
                                            "kappa_dot duplicates another term of the same index is "
                                            "not measuring cost at all, which is the defect axis D "
                                            "exists to repair; a repair cannot be credited with a "
                                            "crossing it produced by re-pointing the defect",
                         "crossing": [{kk: v[kk] for kk in
                                       ("costs", "exponents", "variables", "kappa_set", "H_regime",
                                        "lcb95", "respects_share_bound", "max_term_magnitude",
                                        "kappa_dot_independent", "kappa_dot_worst_duplicate",
                                        "kappa_dot_worst_abs_corr")}
                                      for v in crossing]}},
        "f7_multiplicity_applied": {
            "passed": k >= K_ALREADY_POOLED + len(new_live) and len(new_live) > 0,
            "evidence": {"why_it_can_fail": "correcting 30 against 30 would understate the search; "
                                            "Holm runs over the pooled family",
                         "k_pooled": k, "already_pooled": K_ALREADY_POOLED,
                         "new_scored": len(new_live),
                         "duplicates_of_sealed_not_counted_as_new":
                             len(live) - len(new_live)}},
        "f8_no_fresh_seeds": custody_falsifier(primary["seeds"], replay_of=args.replay_of,
                                               exclude=args.output),
    }
    falsifiers["all_passed"] = all(
        v["passed"] for kk, v in falsifiers.items()
        if kk != "all_passed" and isinstance(v, dict) and not v.get("not_applicable"))

    print(f"\n  rejilla primaria: {primary['grid']}  ({len(new_live)} variantes nuevas de "
          f"{len(live)} medidas)\n")
    for v in sorted(live, key=lambda x: -x["H_regime"])[:14]:
        if v in winners:
            tag = "  <-- CRUZA"
        elif v in crossing:
            why = ("kappa_dot ~= " + v["kappa_dot_worst_duplicate"]
                   if not v["kappa_dot_independent"] else "cota rota")
            tag = f"  <-- cruza DESCALIFICADA ({why})"
        else:
            tag = "" if v["is_new"] else "  (duplicado de la sellada)"
        print(f"    H {v['H_regime']:+.5f} lcb {v['lcb95']:+.5f}  "
              f"{v['costs']:<18}{v['exponents']:<11}{v['variables']:<14}{v['kappa_set']:<7}"
              f"cota {'ok ' if v['respects_share_bound'] else 'ROTA'} "
              f"indep {'si' if v['kappa_dot_independent'] else 'NO'}{tag}")
    print(f"\n  prediccion f4: over_range baja H en {n_down}/{len(paired)} pares")
    print(f"  veredicto: {verdict}   (maximo {best['H_regime']:+.5f} contra umbral {GATE})\n")
    for name, f in falsifiers.items():
        if name == "all_passed" or not isinstance(f, dict):
            continue
        # `not_applicable` carries passed=None. Printing that as FALLA is how a clean custody
        # replay reads as a custody failure, which is the opposite of the truth.
        label = ("NO APLICA" if f.get("not_applicable")
                 else "PASA" if f["passed"] else "FALLA")
        print(f"    {name:<50} {label}")

    payload = {
        "schema_version": "cobb_douglas_scale_repair_v1",
        "claim_status": verdict,
        "scope": "DEVELOPMENT_ON_BURNED_TAPES_NO_ADJUDICATION",
        "run_role": "CACHE_ANALYSIS", "replay_of": args.replay_of,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "module_manifest": module_manifest(MODULES, script=__file__),
        "axes": AXES, "gate": GATE, "k_pooled": k,
        "primary_grid": primary["grid"], "control_grid": "wrap288_v1",
        "amendment_path": str(args.amendment),
        "defect_a_cost_term_duplicates_inventory": {
            n: g["defect_a"] for n, g in grids.items()},
        "defect_b_exponent_is_inverse_to_range": {
            "his_published_exponents": GARRIDO_2024_EXPONENTS,
            "his_maxima_recovered_by_inverting_the_rule": HIS_MAXIMA,
            "our_exponents_at_max": {
                v: e for v, e in (cell(primary["grid"], costs="garrido_c1", exponents="at_max",
                                       variables="his_five",
                                       kappa_set="within").get("exponents_used") or {}).items()},
            "our_exponents_over_range": {
                v: e for v, e in (cell(primary["grid"], costs="garrido_c1", exponents="over_range",
                                       variables="his_five",
                                       kappa_set="within").get("exponents_used") or {}).items()}},
        "grids": grids, "crossing_the_gate": crossing, "winners": winners, "best": best,
        "prediction_f4": {"pairs": paired, "n_pairs": len(paired), "n_fell": n_down},
        "falsifiers": falsifiers, "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(payload, args.output, contract=args.contract,
                            reference=args.sealed_family)
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
