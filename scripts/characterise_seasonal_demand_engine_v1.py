#!/usr/bin/env python3
"""The five falsifiers of the Paper 2 seasonal demand engine, run through the sealed primitives.

Preregistration: docs/PREREGISTRO_DEMANDA_ESTACIONAL_P2_2026-08-07.md
Engine: supply_chain/demand_seasonal.py, Garrido/Pongutá/García-Reyes (2024) IJPR §3.2.

g1 is the one that protects the frozen path: with the switch off the realised order series must be
identical, order for order. It runs FIRST and halts everything if it fails, because every other
number would be read off a contaminated simulator.

Development. No learner. No custody seeds opened.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from supply_chain.arm_runner import run_falsifiers, scored_orders, seal_and_write  # noqa: E402
from supply_chain.demand_seasonal import SeasonalDemandContract  # noqa: E402
from supply_chain.external_env_interface import make_thesis_aligned_training_env  # noqa: E402
from supply_chain.seed_custody import module_manifest  # noqa: E402

MODULES = ("supply_chain/arm_runner.py", "supply_chain/demand_seasonal.py",
           "supply_chain/supply_chain.py", "supply_chain/env_experimental_shifts.py")
WEEK_H = 168.0
ARMS = {
    "thesis_uniform": {"demand_process": "thesis_uniform"},
    "garrido_seasonal_v1": {"demand_process": "garrido_seasonal_v1"},
    "garrido_seasonal_no_forecast": {"demand_process": "garrido_seasonal_v1",
                                     "demand_forecast_visibility": "hidden"},
    "forecast_shuffled": {"demand_process": "garrido_seasonal_v1",
                          "demand_forecast_visibility": "shuffled"},
}


def acf(x: np.ndarray, lag: int) -> float:
    if len(x) <= lag + 1:
        return float("nan")
    a, b = x[:-lag], x[lag:]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def rollout(seed: int, **kw):
    env = make_thesis_aligned_training_env(**kw)
    env.reset(seed=seed)
    done = False
    while not done:
        _, _, term, trunc, _ = env.step(env.action_space.sample() * 0)
        done = term or trunc
    return env.sim


def order_fingerprint(sim) -> list[tuple[float, float]]:
    return [(round(float(o.OPTj), 6), round(float(o.quantity), 6))
            for o in scored_orders(sim)]


def weekly_series(sim) -> np.ndarray:
    t0, t1 = float(sim.warmup_time), float(sim.env.now)
    n = int((t1 - t0) // WEEK_H)
    if n < 1:
        return np.zeros(0)
    out = np.zeros(n)
    for o in scored_orders(sim):
        k = int((float(o.OPTj) - t0) // WEEK_H)
        if 0 <= k < n:
            out[k] += float(o.quantity)
    return out


def run_arm(name: str, seeds: list[int]) -> dict:
    rows = []
    period = SeasonalDemandContract().period_weeks
    for s in seeds:
        sim = rollout(s, **ARMS[name])
        w = weekly_series(sim)
        if len(w) < period * 3:
            continue
        row = {"seed": s, "n_weeks": len(w), "weekly_mean": float(w.mean()),
               "weekly_sd": float(w.std(ddof=1)), "acf1": acf(w, 1),
               "acf_season": acf(w, period), "acf_half_season": acf(w, period // 2)}
        ds = getattr(sim, "demand_seasonal", None)
        if ds is not None:
            row["alpha"], row["gamma"] = ds.alpha, ds.gamma
            hist = [h for h in ds.forecast_history if np.isfinite(h["gr"])]
            # GR issued at week k forecasts week k+1. Score it against the demand that ACTUALLY
            # arrived in k+1, which is the only honest test of a forecast.
            gr = np.array([h["gr"] for h in hist[:-1]], dtype=float)
            nxt = np.array([h["realised"] for h in hist[1:]], dtype=float)
            m = np.isfinite(gr) & np.isfinite(nxt)
            row["forecast_corr"] = (float(np.corrcoef(gr[m], nxt[m])[0, 1])
                                    if m.sum() > 2 and np.std(gr[m]) > 1e-9 else float("nan"))
            row["forecast_mape"] = (float(np.mean(np.abs(gr[m] - nxt[m]) / np.maximum(nxt[m], 1e-9)))
                                    if m.sum() else float("nan"))
        rows.append(row)

    def agg(k):
        v = np.array([r[k] for r in rows if k in r], dtype=float)
        v = v[np.isfinite(v)]
        if not v.size:
            return {"mean": float("nan"), "se": float("nan"), "n": 0}
        return {"mean": float(v.mean()),
                "se": float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0,
                "n": int(len(v)),
                "min": float(v.min()), "max": float(v.max()),
                "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0}

    wm, wsd = agg("weekly_mean"), agg("weekly_sd")
    return {
        "n_episodes": len(rows), "per_episode": rows,
        "weekly_mean": wm, "weekly_sd": wsd,
        "weekly_cv": (wsd["mean"] / wm["mean"]) if wm["mean"] else float("nan"),
        "acf1": agg("acf1"), "acf_season": agg("acf_season"),
        "acf_half_season": agg("acf_half_season"),
        "alpha": agg("alpha"), "gamma": agg("gamma"),
        "forecast_corr": agg("forecast_corr"), "forecast_mape": agg("forecast_mape"),
        "mean_n_weeks": float(np.mean([r["n_weeks"] for r in rows])) if rows else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--seed0", type=int, default=8_600_001)
    ap.add_argument("--output", type=Path,
                    default=Path("results/demand_seasonal_engine/result.json"))
    ap.add_argument("--contract", type=Path,
                    default=Path("docs/PREREGISTRO_DEMANDA_ESTACIONAL_P2_2026-08-07.md"))
    args = ap.parse_args()
    seeds = [args.seed0 + i for i in range(args.seeds)]
    t0 = time.time()

    # g1 FIRST, on two seeds. If the frozen path moved, nothing else may be read.
    g1_pairs = []
    for s in seeds[:2]:
        a = order_fingerprint(rollout(s))
        b = order_fingerprint(rollout(s, demand_process="thesis_uniform"))
        g1_pairs.append({"seed": s, "n": len(a), "identical": a == b})
    g1_ok = all(p["identical"] for p in g1_pairs)
    if not g1_ok:
        payload = {"schema_version": "seasonal_demand_engine_v1",
                   "claim_status": "HALT_NATIVE_PATH_CONTAMINATED",
                   "scope": "DEVELOPMENT_HALT", "run_role": "DIAGNOSTIC", "seeds": seeds,
                   "primary_metric": "weekly_cv_and_seasonal_acf",
                   "falsifiers": {"all_passed": False,
                                  "g1_native_path_unchanged": {"passed": False,
                                                               "evidence": g1_pairs}},
                   "module_manifest": module_manifest(MODULES, script=__file__)}
        d = seal_and_write(payload, args.output, contract=args.contract,
                           reference=Path("results/demand_process/result.json"))
        print(f"HALT: g1 falló. -> {args.output} ({d[:16]}…)")
        return 1

    out = {n: run_arm(n, seeds) for n in ARMS}
    season = out["garrido_seasonal_v1"]
    contract = SeasonalDemandContract()
    band = 2.0 / np.sqrt(max(season["mean_n_weeks"], 1.0))

    def g1():
        return g1_ok, g1_pairs

    def g2():
        """Weekly CV inside the band declared around Figure 3's implied 21.3%."""
        cv = season["weekly_cv"]
        return bool(0.15 <= cv <= 0.28), {
            "weekly_cv": cv, "band": [0.15, 0.28],
            "thesis_uniform_cv": out["thesis_uniform"]["weekly_cv"],
            "garrido_figure3_implied_cv": 174.51 / 819.13}

    def g3():
        """THE central falsifier: positive autocorrelation at the seasonal lag, outside the iid
        band. If Eq (1) with random alpha and gamma destroys the phase, the engine does not create
        the state it was built for."""
        r = season["acf_season"]["mean"]
        return bool(r > band), {
            "acf_at_seasonal_lag": r, "lag_weeks": contract.period_weeks,
            "iid_band": float(band), "acf1": season["acf1"]["mean"],
            "acf_half_season": season["acf_half_season"]["mean"],
            "thesis_uniform_acf_season": out["thesis_uniform"]["acf_season"]["mean"]}

    def g4():
        """alpha and gamma cover [0,1] without clustering."""
        a, g = season["alpha"], season["gamma"]
        ok = all(0.40 <= v["mean"] <= 0.60 and v["sd"] > 0.20 for v in (a, g))
        return bool(ok), {"alpha": a, "gamma": g}

    def g5():
        """The forecast informs but is not clairvoyant. CAN FAIL FROM BOTH SIDES: ~0 means GR
        carries nothing and the shuffled placebo could never lose; ~1 means it is an oracle in
        disguise and any advantage would be an artefact."""
        c = season["forecast_corr"]["mean"]
        return bool(0.0 < c < 1.0 and np.isfinite(c)), {
            "corr_gr_vs_next_week_realised": c,
            "se": season["forecast_corr"]["se"],
            "mape": season["forecast_mape"]["mean"]}

    fals = run_falsifiers({"g1_native_path_unchanged": g1,
                           "g2_weekly_cv_in_band": g2,
                           "g3_seasonal_acf_positive_outside_iid_band": g3,
                           "g4_alpha_gamma_cover_unit_interval": g4,
                           "g5_forecast_informative_but_imperfect": g5})

    verdict = ("ENGINE_READY_FOR_HEADROOM_GATE" if fals["all_passed"]
               else "SEASONAL_ENGINE_DOES_NOT_PRODUCE_PHASE"
               if not fals["g3_seasonal_acf_positive_outside_iid_band"]["passed"]
               else "ENGINE_PARTIAL")

    payload = {
        "schema_version": "seasonal_demand_engine_v1", "claim_status": verdict,
        "scope": "DEVELOPMENT_ENGINE_CHARACTERISATION_NO_ADJUDICATION_NO_LEARNER",
        "run_role": "DIAGNOSTIC", "primary_metric": "weekly_cv_and_seasonal_acf",
        "seeds": seeds, "seasonal_contract": contract.__dict__,
        "seasonal_profile": list(contract.profile()),
        "seasonal_profile_cv": contract.profile_cv(),
        "arms": out, "falsifiers": fals, "elapsed_seconds": time.time() - t0,
        "module_manifest": module_manifest(MODULES, script=__file__),
        "preregistration": str(args.contract),
        "declared_source_ambiguities": [
            "Eq (1) taken literally sums the trend update twice (F+2d); we implement the standard "
            "Holt one-step forecast F+d and expose double_trend for the literal reading",
            "Figure 3 reports kurtosis -1.88 with skewness 5.19, which no distribution can have "
            "(kurtosis >= skewness^2 + 1); we calibrate on mean, sd, min and max instead",
            "the Makridakis 36-value seed series is not transcribed, so the period-12 profile is "
            "OUR reconstruction of Figure 3's shape at OUR scale",
        ],
    }
    digest = seal_and_write(payload, args.output, contract=args.contract,
                            reference=Path("results/demand_process/result.json"))

    print(f"\n{'arm':32}{'CV sem':>9}{'media':>10}{'acf1':>8}{'acf_est':>9}")
    for k, v in out.items():
        print(f"{k:32}{v['weekly_cv']:9.4f}{v['weekly_mean']['mean']:10.0f}"
              f"{v['acf1']['mean']:8.3f}{v['acf_season']['mean']:9.3f}")
    print(f"\nbanda iid ±{band:.4f}   alpha {season['alpha']['mean']:.3f}±{season['alpha']['sd']:.3f}"
          f"   gamma {season['gamma']['mean']:.3f}±{season['gamma']['sd']:.3f}"
          f"   corr(GR, real t+1) {season['forecast_corr']['mean']:.3f}")
    for k, v in fals.items():
        if k != "all_passed":
            print(f"  {k:48} {'PASA' if v['passed'] else 'FALLA'}")
    print(f"\n{verdict}\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if fals["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
