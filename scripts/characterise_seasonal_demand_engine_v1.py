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
from supply_chain.demand_seasonal import SeasonalDemandContract, SeasonalDemandProcess  # noqa: E402
from supply_chain.external_env_interface import make_thesis_aligned_training_env  # noqa: E402
from supply_chain.seed_custody import module_manifest  # noqa: E402

MODULES = ("supply_chain/arm_runner.py", "supply_chain/demand_seasonal.py",
           "supply_chain/supply_chain.py", "supply_chain/env_experimental_shifts.py")
WEEK_H = 168.0
ARMS = {
    "thesis_uniform": {"demand_process": "thesis_uniform"},
    "garrido_seasonal_v1": {"demand_process": "garrido_seasonal_v1"},
    "garrido_holt_winters_observable": {
        "demand_process": "garrido_seasonal_v1",
        "demand_seasonal_contract": {"forecast_mode": "holt_winters_observable"},
    },
    "garrido_seasonal_no_forecast": {
        "demand_process": "garrido_seasonal_v1",
        "demand_seasonal_contract": {"forecast_mode": "holt_winters_observable"},
        "demand_forecast_visibility": "hidden",
    },
    "forecast_shuffled": {
        "demand_process": "garrido_seasonal_v1",
        "demand_seasonal_contract": {"forecast_mode": "holt_winters_observable"},
        "demand_forecast_visibility": "shuffled",
    },
}


def acf(x: np.ndarray, lag: int) -> float:
    if len(x) <= lag + 1:
        return float("nan")
    a, b = x[:-lag], x[lag:]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def forecast_skill(actual: np.ndarray, forecast: np.ndarray, period: int) -> dict:
    """Score a one-step forecast at the following realised observation."""
    n = min(len(actual) - 1, len(forecast))
    y = np.asarray(actual[1:1 + n], dtype=float)
    p = np.asarray(forecast[:n], dtype=float)
    finite = np.isfinite(y) & np.isfinite(p)
    y, p = y[finite], p[finite]
    if not len(y):
        return {"n": 0, "mase": float("nan"), "mase_naive": float("nan"),
                "mase_seasonal_naive": float("nan"), "rmse": float("nan"),
                "bias": float("nan"), "corr": float("nan"), "cross_corr_lags": {}}
    naive = np.abs(actual[1:] - actual[:-1])
    seasonal = (np.abs(actual[period:] - actual[:-period])
                if len(actual) > period else np.asarray([], dtype=float))
    errors = p - y
    mae = float(np.mean(np.abs(errors)))
    naive_scale = max(float(np.nanmean(naive)), 1e-12)
    seasonal_scale = (float(np.nanmean(seasonal)) if len(seasonal)
                      else float("nan"))
    lags = {}
    for lag in range(-2, 3):
        if lag < 0:
            a, b = p[-lag:], y[:lag]
        elif lag > 0:
            a, b = p[:-lag], y[lag:]
        else:
            a, b = p, y
        lags[str(lag)] = (
            float(np.corrcoef(a, b)[0, 1])
            if len(a) > 2 and np.std(a) > 1e-12 and np.std(b) > 1e-12
            else float("nan")
        )
    return {
        "n": int(len(y)),
        # Keep `mase` as the ordinary naive-scaled MASE for compatibility, while exposing both
        # baselines required by the amended reading protocol.
        "mase": float(mae / naive_scale),
        "mase_naive": float(mae / naive_scale),
        "mase_seasonal_naive": (float(mae / max(seasonal_scale, 1e-12))
                                if np.isfinite(seasonal_scale) else float("nan")),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "bias": float(np.mean(errors)),
        "corr": float(np.corrcoef(p, y)[0, 1]) if len(y) > 2 else float("nan"),
        "cross_corr_lags": lags,
    }


def synthetic_forecast_diagnostics() -> dict:
    """Locate phase on an impulse plus known-period sinusoid before production runs."""
    n, period = 96, 12
    t = np.arange(n, dtype=float)
    actual = 2_500.0 + 250.0 * np.sin(2 * np.pi * t / period)
    actual[36] += 500.0
    contract = SeasonalDemandContract(
        forecast_mode="holt_winters_observable",
        alpha_range=(0.35, 0.35), gamma_range=(0.20, 0.20), seasonal_beta=0.20,
    )
    process = SeasonalDemandProcess(contract, np.random.default_rng(20260808))
    for i, value in enumerate(actual):
        process.observe((i + 1) * WEEK_H, float(value))
    hist = process.forecast_history[period:]
    return forecast_skill(
        np.asarray([row["realised"] for row in hist], dtype=float),
        np.asarray([row["gr"] for row in hist], dtype=float), period,
    )


def sampler_diagnostics(n: int = 2_000) -> dict:
    """Instrument-only U[0,1] test; these draws are not scientific episode seeds."""
    rng = np.random.default_rng(20260808)
    alpha, gamma = rng.uniform(0.0, 1.0, size=(2, n))

    def one(values):
        ordered = np.sort(values)
        grid = (np.arange(1, len(values) + 1) - 0.5) / len(values)
        return {"n": int(len(values)), "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)), "min": float(values.min()),
                "max": float(values.max()),
                "ks_uniform": float(np.max(np.abs(ordered - grid)))}

    return {"instrument_seed": 20260808, "alpha": one(alpha), "gamma": one(gamma)}


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
        if ds is not None and ds.forecast_mode == "holt_winters_observable":
            row["alpha"], row["gamma"] = ds.alpha, ds.gamma
            hist = [h for h in ds.forecast_history if np.isfinite(h["gr"])]
            gr = np.asarray([h["gr"] for h in hist], dtype=float)
            realised = np.asarray([h["realised"] for h in hist], dtype=float)
            if name == "forecast_shuffled":
                # Same marginal signal, destroyed temporal alignment.
                gr = np.random.default_rng(20260808 + int(s)).permutation(gr)
            skill = forecast_skill(realised, gr, period)
            row["forecast_corr"] = skill["corr"]
            row["forecast_mase"] = skill["mase"]
            row["forecast_mase_naive"] = skill["mase_naive"]
            row["forecast_mase_seasonal_naive"] = skill["mase_seasonal_naive"]
            row["forecast_rmse"] = skill["rmse"]
            row["forecast_bias"] = skill["bias"]
            row["forecast_cross_corr_lags"] = skill["cross_corr_lags"]
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
        "forecast_corr": agg("forecast_corr"), "forecast_mase": agg("forecast_mase"),
        "forecast_mase_naive": agg("forecast_mase_naive"),
        "forecast_mase_seasonal_naive": agg("forecast_mase_seasonal_naive"),
        "forecast_rmse": agg("forecast_rmse"), "forecast_bias": agg("forecast_bias"),
        "mean_n_weeks": float(np.mean([r["n_weeks"] for r in rows])) if rows else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--seed0", type=int, default=8_600_001)
    ap.add_argument("--output", type=Path,
                    default=Path("results/demand_seasonal_engine/result.json"))
    ap.add_argument("--contract", type=Path,
                    default=Path("docs/ENMIENDA_DEMANDA_ESTACIONAL_P2_2026-08-08.md"))
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
    observable = out["garrido_holt_winters_observable"]
    contract = SeasonalDemandContract()
    band = 2.0 / np.sqrt(max(season["mean_n_weeks"], 1.0))
    sampler = sampler_diagnostics()
    synthetic = synthetic_forecast_diagnostics()

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
        """Test the sampler over many instrument draws, not a 12-episode mean."""
        a, g = sampler["alpha"], sampler["gamma"]
        ok = all(0.47 <= v["mean"] <= 0.53 and 0.27 <= v["sd"] <= 0.31
                 and v["ks_uniform"] < 0.05 for v in (a, g))
        return bool(ok), sampler

    def g5():
        """Score our observable Holt-Winters extension at the declared horizon.

        MASE IS PART OF THE JUDGEMENT, NOT JUST OF THE REPORT. The governing amendment
        (ENMIENDA_DEMANDA_ESTACIONAL_P2_2026-08-08) says g5 "se juzga ... mediante MASE frente a
        naive y seasonal-naive". Gating only on `corr > shuffled` would pass a forecast that is
        WORSE than a naive one, which is the outcome the amendment exists to exclude: an instrument
        beating a phase-destroying placebo is a low bar, and beating the naive baseline is the one
        that licenses calling it informative. MASE < 1 against both baselines is that bar, and it is
        written here before the run.
        """
        c = observable["forecast_corr"]["mean"]
        shuffled = out["forecast_shuffled"]["forecast_corr"]["mean"]
        mase_n = observable["forecast_mase_naive"]["mean"]
        mase_s = observable["forecast_mase_seasonal_naive"]["mean"]
        beats_naive = bool(np.isfinite(mase_n) and mase_n < 1.0)
        beats_seasonal_naive = bool(np.isfinite(mase_s) and mase_s < 1.0)
        ok = (0.0 < c < 1.0 and np.isfinite(c) and c > shuffled
              and beats_naive and beats_seasonal_naive)
        return bool(ok), {
            "beats_naive": beats_naive,
            "beats_seasonal_naive": beats_seasonal_naive,
            "bar": "MASE < 1 against both baselines, plus corr strictly between 0 and 1 and above the shuffled placebo",
            "corr_gr_vs_next_week_realised": c,
            "se": observable["forecast_corr"]["se"],
            "mase": observable["forecast_mase"]["mean"],
            "mase_naive": observable["forecast_mase_naive"]["mean"],
            "mase_seasonal_naive": observable["forecast_mase_seasonal_naive"]["mean"],
            "rmse": observable["forecast_rmse"]["mean"],
            "bias": observable["forecast_bias"]["mean"],
            "shuffled_corr": shuffled,
            "synthetic": synthetic,
        }

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
        "scope": "DEVELOPMENT_SENSITIVITY_NO_ADJUDICATION_NO_LEARNER",
        "run_role": "DEVELOPMENT_SENSITIVITY", "execution_role": "DIAGNOSTIC",
        "primary_metric": "weekly_cv_and_seasonal_acf_and_forecast_skill",
        "seeds": seeds, "seasonal_contract": contract.__dict__,
        "seasonal_profile": list(contract.profile()),
        "seasonal_profile_cv": contract.profile_cv(),
        "sampler_diagnostics": sampler,
        "synthetic_forecast_diagnostics": synthetic,
        "arms": out, "falsifiers": fals, "elapsed_seconds": time.time() - t0,
        "module_manifest": module_manifest(MODULES, script=__file__),
        "preregistration": str(args.contract),
        "declared_source_ambiguities": [
            "GR is used by Garrido as a gross-requirements generator/input; forecast skill is a "
            "separate researcher-defined estimand",
            "the Holt-Winters observable mode is our extension and is never called a repair of "
            "the source equation",
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
    print(f"\nbanda iid ±{band:.4f}   sampler alpha {sampler['alpha']['mean']:.3f}±{sampler['alpha']['sd']:.3f}"
          f"   gamma {sampler['gamma']['mean']:.3f}±{sampler['gamma']['sd']:.3f}"
          f"   corr(HW GR, real t+1) {observable['forecast_corr']['mean']:.3f}")
    for k, v in fals.items():
        if k != "all_passed":
            print(f"  {k:48} {'PASA' if v['passed'] else 'FALLA'}")
    print(f"\n{verdict}\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if fals["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
