#!/usr/bin/env python3
"""Is the demand process static, variable, or variable-and-predictable? Measured, not derived.

WHY THIS EXISTS. Asked whether we have static or variable demand, I read `config.py` and computed
the weekly CV by hand. That is exactly the move that manufactured fake defects on 2026-07-30, so
the hand figure is not a result and does not enter any artifact. This measures the REALIZED series
through the sealed primitives instead.

WHAT IT SEPARATES, and why the distinction decides the project's central claim:

  * VARIABLE   — does the realized weekly demand move at all?
  * PREDICTABLE — does it move with MEMORY? An iid process is variable but carries no state to
                  condition on, so from the demand side there is nothing a state-dependent policy
                  can beat a constant with. That is a structural reason for H_regime = 0 that we
                  have never written down, and it is testable: the lag-1 autocorrelation.

TWO ENVIRONMENTS, because they do not share a demand process:

  * thesis-native (`risk_level='current'`)          -> DEMAND is drawn iid, no regime scaling
  * track_b       (`risk_level='adaptive_benchmark_v2'`) -> `_sample_calendar_demand_quantity`
    multiplies by the regime's `demand_scale` (0.95..1.12), so demand IS coupled to a persistent
    state. Track B is also the only environment where a neural signal has appeared.

Development. No custody seeds are opened; the episode seeds are the block this lane already uses.
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

from supply_chain.arm_runner import scored_orders, seal_and_write  # noqa: E402
from supply_chain.config import ASSEMBLY_RATE, DEMAND, HOURS_PER_SHIFT  # noqa: E402
from supply_chain.external_env_interface import (  # noqa: E402
    make_thesis_aligned_training_env,
    make_track_b_env,
)
from supply_chain.seed_custody import module_manifest  # noqa: E402

MODULES = ("supply_chain/arm_runner.py", "supply_chain/supply_chain.py",
           "supply_chain/external_env_interface.py", "supply_chain/config.py")
WEEK_H = 168.0
RATIONS_PER_SHIFT = ASSEMBLY_RATE * HOURS_PER_SHIFT
ENVS = {"thesis_native": make_thesis_aligned_training_env, "track_b": make_track_b_env}


def demand_series(sim) -> list[dict]:
    """Every scored order as a demand event. `scored_orders` owns the population."""
    return [
        {"t": float(o.OPTj), "qty": float(o.quantity), "contingent": bool(o.contingent)}
        for o in scored_orders(sim)
    ]


def weekly(events: list[dict], t0: float, t1: float, contingent: bool = True) -> np.ndarray:
    """Bin into complete weeks. Partial trailing weeks are dropped, not padded with zeros --
    a padded partial week reads as a demand collapse and would fake variance."""
    n = int((t1 - t0) // WEEK_H)
    if n < 1:
        return np.zeros(0)
    out = np.zeros(n)
    for e in events:
        if not contingent and e["contingent"]:
            continue
        k = int((e["t"] - t0) // WEEK_H)
        if 0 <= k < n:
            out[k] += e["qty"]
    return out


def acf(x: np.ndarray, lag: int) -> float:
    """Sample autocorrelation. NaN when the series cannot support the lag or does not vary."""
    if len(x) <= lag + 1:
        return float("nan")
    a, b = x[:-lag], x[lag:]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def run_env(name: str, seeds: list[int]) -> dict:
    rows, per_ep = [], []
    for s in seeds:
        env = ENVS[name]()
        env.reset(seed=s)
        done = False
        while not done:
            _, _, term, trunc, _ = env.step(env.action_space.sample() * 0)
            done = term or trunc
        sim = env.sim
        ev = demand_series(sim)
        if not ev:
            continue
        t0, t1 = float(sim.warmup_time), float(sim.env.now)
        w_all, w_reg = weekly(ev, t0, t1, True), weekly(ev, t0, t1, False)
        if len(w_all) < 6:
            continue
        rows.append(ev)
        cap_s1 = RATIONS_PER_SHIFT * DEMAND["operating_days_per_week"]
        per_ep.append({
            "seed": s, "n_weeks": len(w_all),
            "weeks_over_capacity_S1": float(np.mean(w_all > cap_s1)),
            "weekly_mean": float(w_all.mean()), "weekly_sd": float(w_all.std(ddof=1)),
            "weekly_mean_regular": float(w_reg.mean()),
            "weekly_sd_regular": float(w_reg.std(ddof=1)),
            "acf1": acf(w_all, 1), "acf2": acf(w_all, 2), "acf4": acf(w_all, 4),
            "acf1_regular": acf(w_reg, 1),
            "contingent_share_qty": float(1.0 - w_reg.sum() / max(w_all.sum(), 1e-9)),
        })
    flat = [e for ev in rows for e in ev]
    reg_q = np.array([e["qty"] for e in flat if not e["contingent"]])
    con_q = np.array([e["qty"] for e in flat if e["contingent"]])

    def agg(k):
        v = np.array([p[k] for p in per_ep], dtype=float)
        v = v[~np.isnan(v)]
        if not len(v):
            return {"mean": float("nan"), "se": float("nan"), "n": 0}
        return {"mean": float(v.mean()),
                "se": float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0,
                "n": int(len(v))}

    wm, wsd = agg("weekly_mean"), agg("weekly_sd")
    wmr, wsdr = agg("weekly_mean_regular"), agg("weekly_sd_regular")
    return {
        "n_episodes": len(per_ep),
        "per_episode": per_ep,
        "regular_order_qty": {
            "n": int(reg_q.size), "min": float(reg_q.min()) if reg_q.size else float("nan"),
            "max": float(reg_q.max()) if reg_q.size else float("nan"),
            "mean": float(reg_q.mean()) if reg_q.size else float("nan"),
        },
        "contingent_order_qty": {
            "n": int(con_q.size), "min": float(con_q.min()) if con_q.size else float("nan"),
            "max": float(con_q.max()) if con_q.size else float("nan"),
            "mean": float(con_q.mean()) if con_q.size else float("nan"),
        },
        "weekly_mean": wm, "weekly_sd": wsd,
        "weekly_cv": (wsd["mean"] / wm["mean"]) if wm["mean"] else float("nan"),
        "weekly_cv_regular_only": (wsdr["mean"] / wmr["mean"]) if wmr["mean"] else float("nan"),
        "weekly_mean_regular_only": wmr["mean"],
        "acf1": agg("acf1"), "acf2": agg("acf2"), "acf4": agg("acf4"),
        "acf1_regular": agg("acf1_regular"),
        "contingent_share_qty": agg("contingent_share_qty"),
        "weekly_capacity": {
            f"S{s}": float(s * RATIONS_PER_SHIFT * DEMAND["operating_days_per_week"])
            for s in (1, 2, 3)
        },
        "weeks_over_capacity_S1": agg("weeks_over_capacity_S1"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--seed0", type=int, default=8_600_001)
    ap.add_argument("--output", type=Path,
                    default=Path("results/demand_process/result.json"))
    ap.add_argument("--contract", type=Path,
                    default=Path("docs/PREREGISTRO_PROCESO_DEMANDA_2026-08-07.md"))
    args = ap.parse_args()

    seeds = [args.seed0 + i for i in range(args.seeds)]
    t0 = time.time()
    out = {name: run_env(name, seeds) for name in ENVS}

    tn, tb = out["thesis_native"], out["track_b"]

    def f1():
        """Regular order quantities stay inside the contract's U(a, b).

        CAN FAIL: a multiplier or a regime scale applied to a nominally thesis-native env would
        push draws outside [2400, 2600]. CAN PASS: exact bounds, nothing to straddle."""
        r = tn["regular_order_qty"]
        ok = bool(r["n"] > 0 and r["min"] >= DEMAND["a"] and r["max"] <= DEMAND["b"])
        return ok, r

    def f2():
        """The thesis-native weekly series VARIES. CAN FAIL if sd is 0, which is what 'static
        demand' would literally mean and is the hypothesis under test."""
        sd = tn["weekly_sd"]["mean"]
        return bool(sd > 0.0), {"weekly_sd": sd}

    def f3():
        """Lag-1 autocorrelation of the thesis-native weekly series is inside the iid band
        |r| < 2/sqrt(n_weeks). This is the SCIENTIFIC test, not a sanity check.

        CAN FAIL: a demand process with memory (trend, seasonality, regime coupling) sits outside
        the band, and that would mean there IS a demand state to condition on. CAN PASS: an iid
        process sits inside it. Judged on the magnitude against its own band, never on a sign --
        rule R6."""
        nw = float(np.mean([p["n_weeks"] for p in tn["per_episode"]])) if tn["per_episode"] else 0.0
        band = 2.0 / np.sqrt(max(nw, 1.0))
        r = tn["acf1"]["mean"]
        return bool(abs(r) < band), {"acf1": r, "iid_band": float(band), "mean_n_weeks": nw}

    def f4():
        """track_b and thesis-native are DIFFERENT processes: track_b couples demand to a
        persistent regime, so its weekly CV should exceed the thesis-native one.

        CAN FAIL: if adaptive_benchmark_v2 does not actually move realized demand, the two CVs
        coincide and the 'Track B has a demand state' reading collapses."""
        a, b = tn["weekly_cv"], tb["weekly_cv"]
        return bool(b > a), {"thesis_native_cv": a, "track_b_cv": b}

    from supply_chain.arm_runner import run_falsifiers
    fals = run_falsifiers({"f1_regular_draws_inside_contract_bounds": f1,
                           "f2_weekly_series_varies": f2,
                           "f3_lag1_acf_inside_iid_band": f3,
                           "f4_track_b_is_a_different_process": f4})

    payload = {
        "schema_version": "demand_process_measurement_v1",
        "claim_status": "DEMAND_PROCESS_CHARACTERISED",
        "scope": "DEVELOPMENT_DIAGNOSTIC_NO_ADJUDICATION_NO_LEARNER",
        "run_role": "DIAGNOSTIC",
        "primary_metric": "weekly_demand_cv_and_lag1_autocorrelation",
        "seeds": seeds,
        "contract_demand": dict(DEMAND),
        "rations_per_shift": float(RATIONS_PER_SHIFT),
        "environments": out,
        "falsifiers": fals,
        "elapsed_seconds": time.time() - t0,
        "module_manifest": module_manifest(MODULES, script=__file__),
        "why": ("Asked whether demand is static or variable, the honest answer needs the realized "
                "series, not a hand-derived CV off config.py. Variable and PREDICTABLE are "
                "different questions: an iid process varies without offering any state to "
                "condition on, which would be a structural reason a constant cannot be beaten "
                "from the demand side."),
    }
    digest = seal_and_write(payload, args.output, contract=args.contract,
                            reference=Path("results/track_b_nonneural/result.json"))

    print(f"\n{'':16}{'CV sem':>9}{'media sem':>11}{'acf1':>8}{'acf1 reg':>10}{'%conting':>10}")
    for k, v in out.items():
        print(f"{k:16}{v['weekly_cv']:9.4f}{v['weekly_mean']['mean']:11.0f}"
              f"{v['acf1']['mean']:8.3f}{v['acf1_regular']['mean']:10.3f}"
              f"{100*v['contingent_share_qty']['mean']:10.1f}")
    print(f"\ncapacidad semanal  S1={tn['weekly_capacity']['S1']:.0f}  "
          f"S2={tn['weekly_capacity']['S2']:.0f}  S3={tn['weekly_capacity']['S3']:.0f}")
    for k, v in fals.items():
        if k != "all_passed":
            print(f"  {k:44} {'PASA' if v['passed'] else 'FALLA'}")
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if fals["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
