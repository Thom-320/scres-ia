#!/usr/bin/env python3
"""Program G G1 — central-cell oracle-to-observable conversion (pre-RL, no learner).

Measures, on the V1.2 two-CSSU shared-transport contract:
  H_PI  = best fixed calendar - clairvoyant per-tape oracle   (spatial headroom)
  H_obs = best fixed calendar - observable signal policy       (converted headroom)
  eta   = H_obs / H_PI                                         (conversion efficiency)
plus wrong-CSSU and shuffled-signal placebos to isolate the value of knowing WHERE.
Service-loss (unmet rations) is primary; lower is better. Exact 3^4 oracle. No PPO,
no virgin tapes. Autonomy: docs/PROGRAM_G_AUTONOMY_AUTHORIZATION_2026-07-12.json.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.program_g import (
    ACTIONS, central_cell, cover_signal_policy, enumerate_oracle, materialize_tape,
    mpc_policy, periodic_calendars, signal_hysteresis_policy, simulate,
)


def boot_ci(x, n=2000, seed=7):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    means = [rng.choice(x, size=len(x), replace=True).mean() for _ in range(n)]
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def wrong_cssu(seq):
    swap = {"A": "B", "B": "A", "HOLD": "HOLD"}
    return tuple(swap[a] for a in seq)


# arm -> (persistent tempo, convoy physical memory, observable policy may read the signal)
ARM_SPEC = {
    "Base": (False, False, False),
    "T":    (True,  False, False),
    "TR":   (True,  True,  False),
    "TS":   (True,  False, True),
    "TRS":  (True,  True,  True),
}


def observable_best(t, arm, use_signal):
    losses = [simulate(t, cover_signal_policy(t, arm, use_signal=use_signal), arm=arm).service_loss]
    if use_signal:
        losses.append(simulate(t, mpc_policy(t, arm), arm=arm).service_loss)
        losses.append(simulate(t, signal_hysteresis_policy(t), arm=arm).service_loss)
    return min(losses)


def eval_arm(arm, cell, weeks, n):
    persistent, _memory, use_signal = ARM_SPEC[arm]
    cal = [materialize_tape(990001 + i, cell, weeks, persistent=persistent) for i in range(n)]
    hold = [materialize_tape(1000001 + i, cell, weeks, persistent=persistent) for i in range(n)]
    cals = periodic_calendars(weeks)
    cal_loss = np.array([[simulate(t, c, arm=arm).service_loss for t in cal] for c in cals])
    best_cal = cals[int(cal_loss.mean(axis=1).argmin())]                 # frozen on calibration
    static = np.array([simulate(t, best_cal, arm=arm).service_loss for t in hold])
    oracle = np.array([enumerate_oracle(t, arm=arm)[0] for t in hold])
    obs = np.array([observable_best(t, arm, use_signal) for t in hold])
    cover = np.array([simulate(t, cover_signal_policy(t, arm, use_signal=use_signal), arm=arm).service_loss for t in hold])
    wrong = np.array([simulate(t, wrong_cssu(cover_signal_policy(t, arm, use_signal=use_signal)), arm=arm).service_loss for t in hold])
    missions_obs = np.mean([simulate(t, cover_signal_policy(t, arm, use_signal=use_signal), arm=arm).convoy_missions for t in hold])
    missions_static = np.mean([simulate(t, best_cal, arm=arm).convoy_missions for t in hold])
    H_PI, H_obs = static - oracle, static - obs
    return {
        "arm": arm, "uses_signal": use_signal, "persistent_tempo": persistent,
        "best_fixed_calendar_frozen_on_calib": list(best_cal),
        "service_loss_mean": {"oracle": float(oracle.mean()), "best_static": float(static.mean()),
                              "best_observable": float(obs.mean()), "wrong_cssu_placebo": float(wrong.mean())},
        "H_PI_ci95": boot_ci(H_PI), "H_obs_ci95": boot_ci(H_obs),
        "eta": float(H_obs.sum() / max(H_PI.sum(), 1e-9)),
        "signal_beats_wrong_cssu_ci95": boot_ci(wrong - cover) if use_signal else [0.0, 0.0, 0.0],
        "convoy_missions_obs_vs_static": [float(missions_obs), float(missions_static)],
    }


def main() -> int:
    weeks = 4
    n = 200
    cell = central_cell()
    arms = ["TRS", "TS", "TR", "T", "Base"]
    ladder = {arm: eval_arm(arm, cell, weeks, n) for arm in arms}

    trs = ladder["TRS"]
    _, pi_lo, _ = trs["H_PI_ci95"]; _, obs_lo, _ = trs["H_obs_ci95"]
    _, where_lo, _ = trs["signal_beats_wrong_cssu_ci95"]
    interp = ("G1_SIGNAL_CONVERTS_SPATIAL_HEADROOM_OOS"
              if (pi_lo > 0 and obs_lo > 0 and where_lo > 0 and trs["eta"] >= 0.30)
              else "G1_SPATIAL_HEADROOM_PRESENT_NOT_OBSERVABLY_CONVERTED" if pi_lo > 0
              else "G1_NO_MATERIAL_SPATIAL_HEADROOM")
    out = {"gate": "PROGRAM_G_G1_LADDER_CALIB_HOLDOUT", "cell": cell, "weeks": weeks,
           "n_calibration": n, "n_holdout": n, "calibration_seed_start": 990001,
           "holdout_seed_start": 1000001, "interpretation": interp,
           "ladder": ladder, "ppo_trained": False, "virgin_tapes_opened": 0,
           "note": ("Honest OOS: best static frozen on calibration, all metrics on disjoint holdout. "
                    "Service-loss (unmet rations) is a disclosed Program-G proxy, not ret_excel. "
                    "IN-SAMPLE central cell only; the 24-cell screen + >=2 adjacent cells still gate a learner.")}
    output = Path("results/program_g/g1"); output.mkdir(parents=True, exist_ok=True)
    (output / "verdict.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"interpretation": interp, **{a: {k: ladder[a][k] for k in
          ("H_PI_ci95", "H_obs_ci95", "eta", "signal_beats_wrong_cssu_ci95",
           "convoy_missions_obs_vs_static")} for a in arms}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
