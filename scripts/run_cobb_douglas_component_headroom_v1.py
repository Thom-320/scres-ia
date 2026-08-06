#!/usr/bin/env python3
"""Is the Cobb-Douglas zero a property of the chain, or an artefact of aggregation?

The ported index measured H_regime exactly 0. But R is a weighted product of five components whose
weights are wildly unequal -- Garrido's own fit gives kappa_dot an exponent of 0.1771 against
zeta's 0.024, roughly 7x -- and the exponents are SET-RELATIVE, derived as 0.20/ln(x_max) on
whatever set you calibrate against. On top of that, our port already records that tau is exactly 0
in 88 of 108 calibration episodes, because the thesis operating point carries enough stock that net
requirements never go positive.

So a scalar zero has at least three innocent explanations that are not "the chain has no
regime-dependent headroom":

  * the weight sits on the one component whose optimum does not move;
  * a component is dead in our chain and contributes nothing either way;
  * components with opposite signs move together and cancel inside the product.

This measures H_regime for each of the five components SEPARATELY, on the same surfaces and with
the same estimator, and reports them beside the scalar. If a component carries headroom the scalar
does not, the zero is about the index. If none does, the zero is about the chain -- and that is a
much stronger statement than we have been entitled to make.

kappa_dot is set-relative: the comparison set is declared here as all configurations within one
(context, seed), which is the set an experimenter actually chooses between.

Development on burned tapes. Adjudicates nothing.
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
from supply_chain.cobb_douglas_resilience import (  # noqa: E402
    SIGNS, VARIABLES, CobbDouglasRecorder, derive_exponents, kappa_dot, resilience_index,
)
from supply_chain.config import HOURS_PER_WEEK  # noqa: E402
from supply_chain.config import THESIS_FAITHFUL_PROTOCOL as P  # noqa: E402
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
RAW_LEVELS = (0.0, 17_500.0, 70_000.0, 140_000.0)
GRIDS = {
    "wrap288_v1": {
        "buffer_hours": (0.0, 168.0, 336.0, 504.0, 672.0, 1344.0),
        "shifts": (1, 2, 3),
        "op9_rop": (12.0, 24.0, 36.0, 48.0),
        "op12_rop": (12.0, 24.0, 36.0, 48.0),
    },
}
GRIDS["wrap288_compat_extended_v1"] = dict(GRIDS["wrap288_v1"],
                                           op3_rm=RAW_LEVELS, op5_rm=RAW_LEVELS)
FACTORS = GRIDS["wrap288_v1"]
CONFIGS = tuple(dict(zip(FACTORS, c)) for c in itertools.product(*FACTORS.values()))


def use_grid(grid_id: str) -> None:
    global FACTORS, CONFIGS
    FACTORS = GRIDS[grid_id]
    CONFIGS = tuple(dict(zip(FACTORS, c)) for c in itertools.product(*FACTORS.values()))
SEED_BASE, WEEKS, PERIOD_HOURS = 5_300_001, 52, 24.0
MODULES = ("supply_chain/supply_chain.py", "supply_chain/cobb_douglas_resilience.py",
           "supply_chain/config.py", "supply_chain/seed_custody.py")


def episode(config, context, seed, horizon):
    risks, freq = CONTEXTS[context]
    sim = MFSCSimulation(
        shifts=int(config["shifts"]),
        initial_buffers={"op3_rm": float(config.get("op3_rm", 0.0)),
                         "op5_rm": float(config.get("op5_rm", 0.0)),
                         "op9_rations": float(config["buffer_hours"]) * 2_500.0 / 24.0},
        inventory_replenishment_period=0.0, seed=seed, horizon=horizon,
        risks_enabled=True, risk_level="current", enabled_risks=set(risks),
        risk_frequency_multipliers_by_id=dict(freq) or None,
        strict_exogenous_crn=True, year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"], r14_defect_mode=P["r14_defect_mode"])
    sim.params["op9_rop"] = float(config["op9_rop"])
    sim.params["op12_rop"] = float(config["op12_rop"])
    rec = CobbDouglasRecorder(period_hours=PERIOD_HOURS)
    done = False
    while not done:
        _, _, done, _ = sim.step(step_hours=PERIOD_HOURS)
        rec.sample(sim)
    return rec.aggregate()


def h_regime(values_by_ctx, sign=+1.0) -> float:
    """Same estimator as everywhere else: mean over seeds BEFORE the max over actions."""
    per_ctx = []
    for mean in values_by_ctx:
        m = sign * np.asarray(mean, dtype=float)
        lo, hi = m.min(), m.max()
        per_ctx.append((m - lo) / (hi - lo) if hi > lo else np.zeros_like(m))
    stacked = np.stack(per_ctx)
    return float(stacked.max(axis=1).mean() - stacked.mean(axis=0).max())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--grid", default="wrap288_v1", choices=tuple(GRIDS))
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--replay-of", required=True)
    ap.add_argument("--output", type=Path,
                    default=Path("results/cobb_douglas_component_headroom/result.json"))
    args = ap.parse_args()
    use_grid(args.grid)
    started = time.perf_counter()
    seeds = [SEED_BASE + i for i in range(args.seeds)]
    print(f"  rejilla {args.grid}: {len(CONFIGS):,} configuraciones")
    horizon = float(WEEKS * HOURS_PER_WEEK)
    contexts = list(CONTEXTS)

    # The aggregates cost ~950 s to produce. Cache them: a defect in the arithmetic downstream
    # must cost a rerun of the arithmetic, not a rerun of the simulator.
    cache = args.output.parent / "aggregates.json"
    aggregates: dict[tuple[str, int], list[dict]] = {}
    if cache.exists():
        raw = json.loads(cache.read_text())
        # rsplit, not split: context names contain "|" themselves (R1r|esc).
        aggregates = {(k.rsplit("|", 1)[0], int(k.rsplit("|", 1)[1])): v for k, v in raw.items()}
        print(f"  agregados leídos de {cache} ({len(aggregates)} celdas)")
    else:
        for ctx in contexts:
            for seed in seeds:
                aggregates[(ctx, seed)] = [episode(c, ctx, seed, horizon) for c in CONFIGS]
            print(f"  {ctx} listo ({time.perf_counter() - started:.0f}s)", flush=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({f"{c}|{s}": v for (c, s), v in aggregates.items()}))

    # Exponents from OUR maxima, by Garrido's own rule. Copying his five numbers would rescale
    # every term by orders of magnitude, because they encode HIS observed maxima.
    maxima = {v: max(float(a[v]) for cell in aggregates.values() for a in cell)
              for v in VARIABLES if v != "kappa_dot"}
    kd_all = []
    for cell in aggregates.values():
        kd = kappa_dot({str(i): a["kappa"] for i, a in enumerate(cell)})
        kd_all.append([kd[str(i)] for i in range(len(cell))])
    maxima["kappa_dot"] = max(max(row) for row in kd_all)
    exponents = derive_exponents(maxima)
    print("\n  exponentes derivados sobre NUESTROS máximos:")
    for v in VARIABLES:
        print(f"    {v:<10} max={maxima[v]:.4g}  exponente={exponents[v]:.5f}")

    # Per-component and scalar surfaces, per (context, seed).
    comp_surface = {v: {} for v in VARIABLES}
    scalar_surface = {}
    for k, (cell, kd_row) in enumerate(zip(aggregates.values(), kd_all)):
        key = list(aggregates)[k]
        for v in VARIABLES:
            comp_surface[v][key] = np.array(
                [kd_row[i] if v == "kappa_dot" else float(a[v]) for i, a in enumerate(cell)])
        scalar_surface[key] = np.array([
            resilience_index({vv: (kd_row[i] if vv == "kappa_dot" else float(a[vv]))
                              for vv in VARIABLES}, exponents)["R_cobb_douglas"]
            for i, a in enumerate(cell)])

    rows = {}
    for v in VARIABLES:
        per_ctx = [np.mean([comp_surface[v][(c, s)] for s in seeds], axis=0) for c in contexts]
        rows[v] = {"H_regime": h_regime(per_ctx, SIGNS[v]), "sign": SIGNS[v],
                   "exponent": exponents[v], "max": maxima[v],
                   "mean_spread": float(np.mean([m.max() - m.min() for m in per_ctx])),
                   "dead_fraction": float(np.mean(
                       [np.mean(np.abs(m) < 1e-12) for m in per_ctx]))}
    per_ctx_R = [np.mean([scalar_surface[(c, s)] for s in seeds], axis=0) for c in contexts]
    scalar_h = h_regime(per_ctx_R, +1.0)

    # The first version of this rule was `best > 3 * max(scalar, 1e-9)`, which with a scalar of
    # exactly 0 fires on ANY non-zero component -- it could not fail in the case it existed to
    # judge. A component only reveals hidden headroom if it clears the same bar every other gate
    # in this project uses.
    GATE = 0.05
    best_component = max(rows, key=lambda v: rows[v]["H_regime"])
    hidden = rows[best_component]["H_regime"] >= GATE > scalar_h
    verdict = ("SCALAR_ZERO_IS_AN_AGGREGATION_ARTIFACT" if hidden
               else "NO_COMPONENT_CARRIES_HEADROOM_EITHER")

    falsifiers = {
        "f1_exponents_are_ours_not_his": {
            "passed": all(abs(exponents[v] - h) > 1e-9 for v, h in
                          (("zeta", 0.024), ("epsilon", 0.026), ("phi", 0.04),
                           ("tau", 0.06), ("kappa_dot", 0.1771))),
            "evidence": {"why_it_can_fail": "his five exponents encode HIS observed maxima; "
                                            "reusing them on our chain would rescale every term "
                                            "by orders of magnitude. If ours came out identical, "
                                            "the derivation is not running",
                         "ours": exponents, "his": {"zeta": 0.024, "epsilon": 0.026, "phi": 0.04,
                                                    "tau": 0.06, "kappa_dot": 0.1771}}},
        "f2_components_are_not_all_dead": {
            "passed": any(r["mean_spread"] > 0 for r in rows.values()),
            "evidence": {"why_it_can_fail": "a component that never moves across configurations "
                                            "cannot carry headroom and would make its zero "
                                            "uninformative rather than a finding",
                         "dead_fraction": {v: rows[v]["dead_fraction"] for v in VARIABLES}}},
        "f3_no_fresh_seeds": custody_falsifier(seeds, replay_of=args.replay_of,
                                               exclude=args.output),
    }
    falsifiers["all_passed"] = all(
        v["passed"] for k, v in falsifiers.items()
        if k != "all_passed" and isinstance(v, dict) and not v.get("not_applicable"))

    print(f"\n  escalar R : H_regime {scalar_h:+.5f}")
    print("  por componente (signo ya aplicado):")
    for v, r in sorted(rows.items(), key=lambda kv: -kv[1]["H_regime"]):
        print(f"    {v:<10} H {r['H_regime']:+.5f}  peso {r['exponent']:.4f}  "
              f"spread {r['mean_spread']:.4g}  muerto {r['dead_fraction']:.0%}")
    print(f"\n  veredicto: {verdict}\n")
    for name, f in falsifiers.items():
        if name == "all_passed" or not isinstance(f, dict):
            continue
        label = "NO APLICA" if f.get("not_applicable") else ("PASA" if f["passed"] else "FALLA")
        print(f"    {name:<38} {label}")

    payload = {
        "schema_version": "cobb_douglas_component_headroom_v1",
        "claim_status": verdict,
        "scope": "DEVELOPMENT_ON_BURNED_TAPES_NO_ADJUDICATION",
        "run_role": "CACHE_ANALYSIS", "replay_of": args.replay_of,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "module_manifest": module_manifest(MODULES, script=__file__),
        "grid_id": args.grid,
        "comparison_set_for_kappa_dot": "all configurations within one (context, seed)",
        "exponents_ours": exponents, "maxima_ours": maxima,
        "gate": 0.05, "best_component": best_component,
        "scalar_h_regime": scalar_h, "components": rows,
        "weight_concentration": {
            "note": ("share of total exponent mass carried by each component; the index is "
                     "dominated by the two variables with the smallest observed range"),
            "shares": {v: rows[v]["exponent"] / sum(r["exponent"] for r in rows.values())
                       for v in rows}},
        "seeds": seeds, "contexts": contexts, "n_configurations": len(CONFIGS),
        "falsifiers": falsifiers, "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(payload, args.output, contract=args.contract,
                            reference=Path("results/endpoint_headroom_atlas/result.json"))
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
