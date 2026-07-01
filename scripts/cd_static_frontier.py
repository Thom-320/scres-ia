#!/usr/bin/env python3
"""Stage 1 — Static Cobb-Douglas (CD) same-bar frontier screen across an env-variant ladder.

Per Plan 2026-06-27. Evaluates the 18 Track-A [6 inventory x 3 shift] STATIC policies under the
faithful CD env (Garrido-2024 sigmoid index) across regimes, for each env variant in a
fewest-change ladder. Scores whether a real DECISION FRONTIER exists ON THE CD BAR -- with NO RL
outcome (anti outcome-shopping). Picks the env variant(s) with a genuine interior, regime-dependent
CD optimum + real CD headroom.

CD metric per (config, regime) = mean over steps of `ret_garrido2024_sigmoid_step` (in (0,1]).
Mirrors `scripts/screen_garrido_cost_metrics.py::_run_garrido2024_episode` env construction so the
CD index is identical, but adds the env-variant + demand knobs and the frontier scoring.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.env_experimental_shifts import MFSCGymEnvShifts
from supply_chain.config import THESIS_FAITHFUL_PROTOCOL as P, INVENTORY_BUFFERS

INVENTORY_LEVELS = [0, 168, 336, 504, 672, 1344]
SHIFT_LEVELS = [1, 2, 3]
MAX_CORNER = "S3_I1344"
COLLAPSE_FLOOR = 0.02      # CD sigmoid mean below this = collapsed
OFF_SAT_LO, OFF_SAT_HI = 0.05, 0.95

# Fewest-change env ladder. Each entry overlays knobs on the faithful CD env.
# (name, risk_freq, risk_impact, stochastic_pt, demand_mult)
DEFAULT_LADDER = [
    ("faithful",  1.0, 1.0, False, 1.0),   # 1:1 thesis-faithful
    ("risk_a",    1.0, 1.5, False, 1.0),   # +impact (env_b_headroom promising cell)
    ("risk_b",    2.0, 1.0, False, 1.0),   # +frequency
    ("spt",       1.0, 1.5, True,  1.0),   # +stochastic processing times
    ("demand",    1.0, 1.5, False, 1.1),   # +mean demand (note: variable demand NOT wired)
]


def run_cd_episode(shifts, period, seed, regime, *, phi, psi, stochastic_pt, demand_mult,
                   kappa_train_frac, shift_cost, max_steps, step_size, calibration_path=None):
    bufs = {k: float(v) for k, v in INVENTORY_BUFFERS[period].items()} if period else None
    extra = {}
    if calibration_path:
        extra["ret_g24_calibration_path"] = str(calibration_path)
    env = MFSCGymEnvShifts(
        reward_mode="ReT_garrido2024", observation_version="v4",
        step_size_hours=float(step_size), max_steps=int(max_steps), risk_level=str(regime),
        stochastic_pt=bool(stochastic_pt), year_basis="thesis", warmup_trigger="op9_arrival",
        downstream_q_source="figure_6_2", r14_defect_mode="thesis_strict_op6",
        risk_occurrence_mode="thesis_window", risk_frequency_multiplier=float(phi),
        risk_impact_multiplier=float(psi), demand_mean_multiplier=float(demand_mult),
        raw_material_flow_mode=P["raw_material_flow_mode"],
        raw_material_order_up_to_multiplier=float(P["raw_material_order_up_to_multiplier"]),
        ret_g24_kappa_train_frac=float(kappa_train_frac), ret_g24_shift_cost=float(shift_cost),
        **extra,
    )
    env.reset(seed=int(seed), options={
        "initial_buffers": bufs, "initial_shifts": int(shifts),
        "inventory_replenishment_period": (float(period) if period else None),
    })
    action = {"assembly_shifts": int(shifts)}
    sig_sum = 0.0
    comps = {}
    n = 0
    done = False
    while not done:
        _, _r, term, trunc, info = env.step(action)
        sig_sum += float(info["ret_garrido2024_sigmoid_step"])
        comps = info
        n += 1
        done = bool(term or trunc)
    ret = env.sim.compute_order_level_ret() if env.sim is not None else {}
    env.close()
    return {
        "cd_sigmoid_mean": sig_sum / max(n, 1),
        "zeta_avg": float(comps.get("zeta_avg", math.nan)),
        "epsilon_avg": float(comps.get("epsilon_avg", math.nan)),
        "phi_avg": float(comps.get("phi_avg", math.nan)),
        "tau_avg": float(comps.get("tau_avg", math.nan)),
        "kappa_dot": float(comps.get("kappa_dot", math.nan)),
        "ret_excel": float(ret.get("mean_ret_excel_formula", math.nan)),
        "fill_rate": float(ret.get("fill_rate_order_level", math.nan)),
        # resource axes for the Stage-2 Pareto
        "extra_shifts": float(shifts - 1),
        "buffer_units": float(sum(INVENTORY_BUFFERS[period].values())) if period else 0.0,
    }


def panel_variant(seeds, regimes, variant, *, kappa_train_frac, shift_cost, max_steps, step_size,
                  calibration_path=None):
    _name, phi, psi, spt, dem = variant
    rows = []
    for period_i in INVENTORY_LEVELS:
        for s in SHIFT_LEVELS:
            for regime in regimes:
                accs = {}
                for seed in seeds:
                    m = run_cd_episode(s, period_i, seed, regime, phi=phi, psi=psi,
                                       stochastic_pt=spt, demand_mult=dem,
                                       kappa_train_frac=kappa_train_frac, shift_cost=shift_cost,
                                       max_steps=max_steps, step_size=step_size,
                                       calibration_path=calibration_path)
                    for k, v in m.items():
                        accs.setdefault(k, []).append(float(v))
                row = {"policy": f"S{s}_I{period_i}", "regime": regime}
                for k, vs in accs.items():
                    row[k] = statistics.mean(v for v in vs if not math.isnan(v)) if any(
                        not math.isnan(v) for v in vs) else math.nan
                rows.append(row)
    return rows


def score_variant(rows, regimes, metric="cd_sigmoid_mean"):
    policies = sorted({r["policy"] for r in rows})
    by = {(r["policy"], r["regime"]): r for r in rows}
    def mean_metric(pol):
        return statistics.mean(by[(pol, rg)][metric] for rg in regimes)
    robust = max(policies, key=mean_metric)
    oracle = {rg: max(policies, key=lambda p: by[(p, rg)][metric]) for rg in regimes}
    gaps = {rg: by[(oracle[rg], rg)][metric] - by[(robust, rg)][metric] for rg in regimes}
    mean_gap = statistics.mean(gaps.values())
    argmax_div = len(set(oracle.values()))
    corner_free = any(oracle[rg] != MAX_CORNER for rg in regimes)
    mild = regimes[0]
    mild_cd = by[(oracle[mild], mild)][metric]
    off_sat = OFF_SAT_LO <= mild_cd <= OFF_SAT_HI
    worst_oracle = min(by[(oracle[rg], rg)][metric] for rg in regimes)
    collapse_guard = worst_oracle > COLLAPSE_FLOOR
    eligible = corner_free and argmax_div >= 2 and off_sat and collapse_guard
    return {
        "robust_static": robust, "oracle_by_regime": oracle,
        "cd_gap_by_regime": {k: round(v, 5) for k, v in gaps.items()},
        "mean_cd_gap": round(mean_gap, 5), "argmax_diversity": argmax_div,
        "corner_free": corner_free, "mild_oracle_cd": round(mild_cd, 4),
        "off_saturation": off_sat, "worst_oracle_cd": round(worst_oracle, 5),
        "collapse_guard": collapse_guard, "eligible": eligible,
        "oracle_ret_excel_by_regime": {
            rg: round(max(by[(p, rg)]["ret_excel"] for p in policies), 5) for rg in regimes},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes", default="current,increased,severe")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--variants", default="all",
                    help="comma names from the ladder, or 'all'")
    ap.add_argument("--cells", default=None,
                    help="Override the ladder with explicit phi:psi war cells, "
                         "e.g. '2:1,3:1,4:1.5,5:2' (spt off, demand 1.0).")
    ap.add_argument("--kappa-train-frac", type=float, default=0.2)
    ap.add_argument("--shift-cost", type=float, default=0.5)
    ap.add_argument("--max-steps", type=int, default=52)
    ap.add_argument("--step-size", type=float, default=168.0)
    ap.add_argument("--calibration-path", default=None,
                    help="CD calibration JSON (recalibrated per cost level in Stage 2).")
    ap.add_argument("--output", default="outputs/experiments/cd_static_frontier_2026-06-27")
    args = ap.parse_args()
    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if args.cells:
        ladder = []
        for tok in args.cells.split(","):
            tok = tok.strip()
            if not tok:
                continue
            phi_s, psi_s = tok.split(":")
            phi_v, psi_v = float(phi_s), float(psi_s)
            ladder.append((f"war_phi{phi_v}_psi{psi_v}", phi_v, psi_v, False, 1.0))
    elif args.variants == "all":
        ladder = DEFAULT_LADDER
    else:
        want = {v.strip() for v in args.variants.split(",")}
        ladder = [v for v in DEFAULT_LADDER if v[0] in want]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    results = []
    for i, variant in enumerate(ladder, 1):
        name = variant[0]
        rows = panel_variant(seeds, regimes, variant, kappa_train_frac=args.kappa_train_frac,
                             shift_cost=args.shift_cost, max_steps=args.max_steps,
                             step_size=args.step_size, calibration_path=args.calibration_path)
        with (out / f"panel_{name}.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        sc = score_variant(rows, regimes)
        sc.update({"variant": name, "phi": variant[1], "psi": variant[2],
                   "stochastic_pt": variant[3], "demand_mult": variant[4]})
        results.append(sc)
        print(f"  [{i}/{len(ladder)}] {name:9} mean_cd_gap={sc['mean_cd_gap']:.4f} "
              f"div={sc['argmax_diversity']} corner_free={sc['corner_free']} "
              f"off_sat={sc['off_saturation']} mild_cd={sc['mild_oracle_cd']:.3f} "
              f"robust={sc['robust_static']} oracle={list(sc['oracle_by_regime'].values())} "
              f"ELIGIBLE={sc['eligible']}", flush=True)

    eligible = [r for r in results if r["eligible"]]
    pool = eligible or results
    ranked = sorted(pool, key=lambda r: (-r["mean_cd_gap"], r["phi"] + r["psi"]))
    summary = {"regimes": regimes, "seeds": seeds, "kappa_train_frac": args.kappa_train_frac,
               "shift_cost": args.shift_cost, "max_steps": args.max_steps,
               "results": results, "promising": [r["variant"] for r in ranked[:2]],
               "n_eligible": len(eligible)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== PROMISING CD ENV CELLS (eligible, deepest CD gap; fewest-change tiebreak) ===")
    for r in ranked[:2]:
        print(f"  {r['variant']}: phi={r['phi']} psi={r['psi']} spt={r['stochastic_pt']} "
              f"dem={r['demand_mult']}  cd_gap={r['mean_cd_gap']:.4f} div={r['argmax_diversity']} "
              f"oracle={list(r['oracle_by_regime'].values())}")
    print(f"  eligible: {len(eligible)}/{len(results)}")
    print(f"\nWROTE {out}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
