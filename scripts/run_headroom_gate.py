#!/usr/bin/env python3
"""Lane A0 — robust non-stationary headroom gate (static-only, no training).

Decides whether a same-variables dynamic win is even POSSIBLE under non-stationary R2 intensity. The
2-seed smoke suggested the optimal buffer moves with intensity (+5% headroom), but the per-op campaign
smoke found oracle = best constant. This gate settles it with 5 seeds + bootstrap CIs.

For each disruption intensity in {calm, R2-φ2, R2-φ4, R2-φ6} we sweep a buffer grid (single shared
continuous_its buffer AND per-op Op9-only) at S1, over N CRN seeds, and compute:
  - per-intensity best constant buffer (the oracle component),
  - best SINGLE constant across the intensity mix (the static deployment baseline),
  - oracle = mean over intensities of each intensity's best,
  - headroom = oracle - best_single_constant, with a bootstrap CI over seeds.
Verdict: headroom CI lower bound > 0 => a regime-adaptive policy can beat any constant => proceed Lane A.
Else => the opening is illusory => record null, shift weight to Lane B / Track B.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from supply_chain.continuous_its_env import (
    make_continuous_its_track_a_env, make_per_op_buffer_track_a_env)
from supply_chain.episode_metrics import compute_episode_metrics

R2 = ["R21", "R22", "R23", "R24"]
REGIMES = [("calm", 1.0, None), ("R2_phi2", 2.0, R2), ("R2_phi4", 4.0, R2), ("R2_phi6", 6.0, R2)]


def eval_buffer(mode, buf, phi, enabled, seed, max_steps):
    """One episode ReT for a constant buffer at S1. mode='single' or 'per_op' (Op9-only)."""
    common = dict(reward_mode="ReT_excel_delta", observation_version="v6", risk_level="current",
                  risk_frequency_multiplier=phi, risk_impact_multiplier=1.5, stochastic_pt=False,
                  max_steps=max_steps, step_size_hours=168.0, risk_obs=True)
    if enabled:
        common["enabled_risks"] = enabled
    if mode == "single":
        env = make_continuous_its_track_a_env(init_frac=buf, **common)
        act = np.array([buf, -1.0], dtype=np.float32)
    else:
        env = make_per_op_buffer_track_a_env(**common)
        act = np.array([0.0, 0.0, buf, -1.0], dtype=np.float32)  # op3=op5=0, op9=buf, S1
    env.reset(seed=seed)
    done = trunc = False
    while not (done or trunc):
        _, _r, done, trunc, _i = env.step(act)
    return float(compute_episode_metrics(env.unwrapped.sim)["ret_excel"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("single", "per_op"), default="single")
    ap.add_argument("--bufs", default="0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.50")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=7000)
    ap.add_argument("--max-steps", type=int, default=104)
    ap.add_argument("--mix", default="R2_phi2,R2_phi4,R2_phi6", help="intensities in the non-stationary mix")
    ap.add_argument("--output", default="outputs/experiments/headroom_gate_2026-06-28")
    args = ap.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    bufs = [float(x) for x in args.bufs.split(",")]
    mix = args.mix.split(",")
    reg = {name: (phi, en) for name, phi, en in REGIMES}

    # per (intensity, buffer): array of per-seed ReT
    table = {}
    for name in mix:
        phi, en = reg[name]
        table[name] = {}
        for buf in bufs:
            rets = [eval_buffer(args.mode, buf, phi, en, args.seed0 + s, args.max_steps)
                    for s in range(args.seeds)]
            table[name][buf] = rets
            print(f"  {name:9} buf={buf:.2f}: ReT={np.mean(rets):.4f}±{np.std(rets):.4f}", flush=True)

    # per-seed: best-single-constant (one buf maximizing mean-over-intensities) and oracle (per-intensity best)
    def seed_means(buf):
        return np.array([np.mean([table[name][buf][s] for name in mix]) for s in range(args.seeds)])
    const_mean_by_buf = {buf: float(np.mean(seed_means(buf))) for buf in bufs}
    best_const_buf = max(const_mean_by_buf, key=const_mean_by_buf.get)

    # bootstrap headroom = oracle - best_single_constant, over seeds
    rng = np.random.default_rng(0)
    headrooms = []
    for _ in range(2000):
        idx = rng.integers(0, args.seeds, args.seeds)
        oracle = np.mean([max(np.mean([table[name][buf][i] for i in idx]) for buf in bufs) for name in mix])
        bc = max(np.mean([np.mean([table[name][buf][i] for name in mix]) for i in idx]) for buf in bufs)
        headrooms.append(oracle - bc)
    h = np.array(headrooms)
    ci_lo, ci_hi = float(np.percentile(h, 2.5)), float(np.percentile(h, 97.5))
    headroom_mean = float(np.mean(h))
    real = ci_lo > 0

    per_intensity_best = {name: max(bufs, key=lambda b: np.mean(table[name][b])) for name in mix}
    summary = {"args": vars(args), "mode": args.mode, "mix": mix,
               "per_intensity_best_buf": per_intensity_best,
               "best_single_constant_buf": best_const_buf,
               "headroom_mean": headroom_mean, "headroom_ci": [ci_lo, ci_hi],
               "opening_real": bool(real),
               "table_means": {n: {b: float(np.mean(v)) for b, v in d.items()} for n, d in table.items()}}
    (out / f"gate_{args.mode}.json").write_text(json.dumps(summary, indent=2, default=float))

    print(f"\n=== A0 HEADROOM GATE ({args.mode} buffer, mix={mix}, {args.seeds} seeds, h{args.max_steps}) ===")
    print(f"per-intensity best buffer: {per_intensity_best}")
    print(f"best SINGLE constant buffer: {best_const_buf}")
    print(f"headroom (oracle - best constant) = {headroom_mean:+.5f}  CI95=[{ci_lo:+.5f},{ci_hi:+.5f}]")
    print(f"\n=> {'OPENING REAL: a regime-adaptive policy can beat any constant -> proceed Lane A' if real else 'NO HEADROOM: oracle ~= best constant -> opening illusory; record null, weight to Lane B / Track B'}")
    print(f"WROTE {out}/gate_{args.mode}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
