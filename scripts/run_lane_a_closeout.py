#!/usr/bin/env python3
"""Lane A closeout (rigor): are there ANY same-variable regimes/metrics with a moving buffer optimum?

A0 showed buffer 0.15 is ReT-optimal across R2 intensities (no frontier). This closes the remaining
genuinely-different angles before declaring Lane A exhausted:
  - tail metric: best buffer for SERVICE-LOSS TAIL (not just mean ReT) per regime — does timing the tail
    create a frontier even if the mean-level optimum is constant?
  - demand surge (R24-only): does prepositioning before contingent-demand surges move the optimum?
  - manufacturing (R1-only) and mixed families: does the optimal lever/level move across families?
If the best buffer is the SAME constant across every regime AND metric -> no frontier, Lane A exhausted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from supply_chain.continuous_its_env import make_continuous_its_track_a_env
from supply_chain.episode_metrics import compute_episode_metrics

REGIMES = {
    "R2_phi4": (4.0, ["R21", "R22", "R23", "R24"]),
    "R24_surge_phi4": (4.0, ["R24"]),
    "R1_mfg_phi4": (4.0, ["R11", "R12", "R13", "R14"]),
    "mixed_phi4": (4.0, None),
}
BUFS = [0.0, 0.05, 0.10, 0.15, 0.25, 0.50]
SEEDS = 3
MAXSTEPS = 104


def evalb(buf, phi, enabled, seed):
    common = dict(reward_mode="ReT_excel_delta", observation_version="v6", risk_level="current",
                  risk_frequency_multiplier=phi, risk_impact_multiplier=1.5, stochastic_pt=False,
                  max_steps=MAXSTEPS, step_size_hours=168.0, risk_obs=True)
    if enabled:
        common["enabled_risks"] = enabled
    env = make_continuous_its_track_a_env(init_frac=buf, **common)
    env.reset(seed=seed)
    a = np.array([buf, -1.0], dtype=np.float32)
    done = trunc = False
    while not (done or trunc):
        _, _r, done, trunc, _i = env.step(a)
    m = compute_episode_metrics(env.unwrapped.sim)
    return float(m["ret_excel"]), float(m["service_loss_auc_ration_hours"])


def main() -> int:
    out = Path("outputs/experiments/lane_a_closeout_2026-06-28"); out.mkdir(parents=True, exist_ok=True)
    res = {}
    print(f"{'regime':16} {'metric':6} " + " ".join(f"b{b:.2f}" for b in BUFS) + "  BEST")
    for name, (phi, en) in REGIMES.items():
        ret = {b: [] for b in BUFS}
        tail = {b: [] for b in BUFS}
        for b in BUFS:
            for s in range(SEEDS):
                r, t = evalb(b, phi, en, 7000 + s)
                ret[b].append(r); tail[b].append(t)
        retm = {b: float(np.mean(v)) for b, v in ret.items()}
        tailm = {b: float(np.mean(v)) for b, v in tail.items()}
        best_ret = max(retm, key=retm.get)         # higher ReT better
        best_tail = min(tailm, key=tailm.get)       # lower service-loss-tail better
        res[name] = {"ret_mean": retm, "tail_mean": tailm, "best_ret_buf": best_ret, "best_tail_buf": best_tail}
        print(f"{name:16} {'ReT':6} " + " ".join(f"{retm[b]:.3f}" for b in BUFS) + f"  b{best_ret:.2f}")
        print(f"{name:16} {'tail':6} " + " ".join(f"{tailm[b]/1e6:.2f}" for b in BUFS) + f"  b{best_tail:.2f}")

    best_ret_bufs = {res[n]["best_ret_buf"] for n in res}
    best_tail_bufs = {res[n]["best_tail_buf"] for n in res}
    moving = len(best_ret_bufs) > 1 or len(best_tail_bufs) > 1
    (out / "closeout.json").write_text(json.dumps(res, indent=2, default=float))
    print(f"\nbest-ReT buffer set across regimes:  {sorted(best_ret_bufs)}")
    print(f"best-tail buffer set across regimes: {sorted(best_tail_bufs)}")
    print(f"\n=> {'MOVING OPTIMUM found -> a frontier may exist; investigate' if moving else 'CONSTANT OPTIMUM across all regimes & metrics -> NO frontier; Lane A exhausted'}")
    print("LANE A CLOSEOUT DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
