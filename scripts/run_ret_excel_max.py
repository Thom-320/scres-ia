#!/usr/bin/env python3
"""Sprint 0 raw-ReT headroom GATE (P8) + dense free/charged static frontier (P7).

Decides — cheaply, before any GPU — whether a RAW Excel-ReT win over static policies is physically
possible, or whether objective 1 must become "budget-constrained ReT". Eval-only (no training):

  1. Evaluate the DENSE static frontier: buffer_frac in {0.00,0.05,...,1.00} x shift {S1,S2,S3} (21x3=63)
     on Excel ReT + CVaR95 + resource_composite, over fixed eval seeds, on the winning-lane env
     (continuous_its, v6+risk_obs/hazard, war phi4/psi1.5, h104, holding_cost=0).
  2. Report best FREE static (max Excel, any resource), best CHARGED static at <= dynamic resource,
     and the Pareto frontier.
  3. Verdict: raw_ret_headroom = best_free_static_excel - dynamic_excel_ref. If > 0 beyond noise, a raw
     Excel-ReT win is impossible (a free max-buffer static already tops it) -> pivot to budget-constrained.

Dynamic reference (Kaggle mixed10/60k confirmed): excel 0.0021425, resource 0.241.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from supply_chain.continuous_its_env import make_continuous_its_track_a_env
from supply_chain.episode_metrics import compute_episode_metrics

SHIFT_SIGS = {1: -1.0, 2: 0.0, 3: 1.0}


def build(phi, psi, max_steps, init_frac):
    return make_continuous_its_track_a_env(
        reward_mode="ReT_excel_delta", observation_version="v6", risk_level="current",
        risk_frequency_multiplier=float(phi), risk_impact_multiplier=float(psi), stochastic_pt=False,
        max_steps=int(max_steps), step_size_hours=168.0, init_frac=init_frac, risk_obs=True,
        holding_cost=0.0, shift_cost=0.0)


def _cvar(sl):
    s = sorted(x for x in sl if x == x)
    k = max(1, int(round(0.05 * len(s))))
    return float(np.mean(s[-k:])) if s else float("nan")


def resource_composite(frac, shift):
    return 0.5 * float(np.clip(frac, 0.0, 1.0)) + 0.5 * (float(int(shift) - 1) / 2.0)


def eval_static(phi, psi, max_steps, frac, shift, episodes, seed0):
    sig = SHIFT_SIGS[shift]
    excels, sl = [], []
    for ep in range(episodes):
        env = build(phi, psi, max_steps, init_frac=frac)
        env.reset(seed=seed0 + ep)
        done = trunc = False
        a = np.array([frac, sig], dtype=np.float32)
        while not (done or trunc):
            _, _r, done, trunc, _i = env.step(a)
        m = compute_episode_metrics(env.unwrapped.sim)
        excels.append(float(m.get("ret_excel", np.nan)))
        sl.append(float(m.get("service_loss_auc_ration_hours", np.nan)))
    return {"frac": frac, "shift": shift, "label": f"f{frac:.2f}_S{shift}",
            "excel": float(np.nanmean(excels)), "excel_sem": float(np.nanstd(excels) / np.sqrt(max(1, episodes))),
            "cvar": _cvar(sl), "resource": resource_composite(frac, shift)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phi", type=float, default=4.0)
    ap.add_argument("--psi", type=float, default=1.5)
    ap.add_argument("--max-steps", type=int, default=104)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=9000)
    ap.add_argument("--n-fracs", type=int, default=21, help="dense buffer grid (21 -> 0.00..1.00 step .05)")
    ap.add_argument("--dynamic-excel-ref", type=float, default=0.0021424619)
    ap.add_argument("--dynamic-resource-ref", type=float, default=0.24117719)
    ap.add_argument("--output", default="outputs/experiments/ret_excel_max_gate_2026-06-28")
    args = ap.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    fracs = [round(i / (args.n_fracs - 1), 4) for i in range(args.n_fracs)]

    statics = []
    for frac in fracs:
        for shift in (1, 2, 3):
            r = eval_static(args.phi, args.psi, args.max_steps, frac, shift, args.episodes, args.seed0)
            statics.append(r)
            print(f"  {r['label']:10} excel={r['excel']:.5f}±{r['excel_sem']:.5f} "
                  f"cvar={r['cvar']:.2e} res={r['resource']:.3f}", flush=True)

    best_free = max(statics, key=lambda s: s["excel"])
    le = [s for s in statics if s["resource"] <= args.dynamic_resource_ref + 1e-9]
    best_charged = max(le, key=lambda s: s["excel"]) if le else None
    # Pareto frontier (max excel as resource increases)
    pareto, best_so_far = [], -1.0
    for s in sorted(statics, key=lambda s: s["resource"]):
        if s["excel"] > best_so_far:
            pareto.append(s); best_so_far = s["excel"]

    headroom = best_free["excel"] - args.dynamic_excel_ref
    raw_ret_possible = headroom <= best_free["excel_sem"] + 1e-9  # within noise => maybe possible

    summary = {"args": vars(args), "statics": statics, "best_free_static": best_free,
               "best_charged_static_le_dyn_resource": best_charged,
               "dynamic_ref": {"excel": args.dynamic_excel_ref, "resource": args.dynamic_resource_ref},
               "raw_ret_headroom": headroom, "raw_ret_win_possible": bool(raw_ret_possible),
               "pareto_frontier": pareto}
    (out / "gate.json").write_text(json.dumps(summary, indent=2, default=float))

    print(f"\n=== RAW-ReT HEADROOM GATE (war φ{args.phi}/ψ{args.psi}, h{args.max_steps}, {args.episodes} eps) ===")
    print(f"DYNAMIC ref: excel={args.dynamic_excel_ref:.5f} at resource={args.dynamic_resource_ref:.3f}")
    print(f"BEST FREE static:    {best_free['label']} excel={best_free['excel']:.5f}±{best_free['excel_sem']:.5f} "
          f"res={best_free['resource']:.3f}")
    if best_charged:
        print(f"BEST static @<=dyn-res: {best_charged['label']} excel={best_charged['excel']:.5f} res={best_charged['resource']:.3f}")
    print(f"raw-ReT headroom (best_free - dynamic) = {headroom:+.5f}  (best_free SEM {best_free['excel_sem']:.5f})")
    if raw_ret_possible:
        print("=> RAW-ReT WIN PLAUSIBLE: dynamic is within noise of the best free static. Pursue objective 1 (raw ReT).")
    else:
        print("=> RAW-ReT WIN UNLIKELY: a FREE static tops the dynamic on Excel ReT. "
              "Pivot objective 1 -> BUDGET-CONSTRAINED ReT (Lagrangian). Lead with the Pareto win.")
    print(f"WROTE {out}/gate.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
