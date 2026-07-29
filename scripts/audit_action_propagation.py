#!/usr/bin/env python3
"""Does the agent's action actually change the environment, and with what latency?

Answers the user's correctness question directly: when Pepe picks a buffer fraction / shift, (a) does
the simulator state change, (b) immediately or after a lag, and (c) does it change the OUTCOME
(inventory / fill / ReT)? If actions don't propagate, no policy could ever win and "reactive" would
be an artifact, so this is a gate on everything else.

Mechanism (from code): the continuous wrapper's _set_targets() sets sim.inventory_buffer_targets and
IMMEDIATELY tops up the containers (raw_material_wdc/al, rations_sb) via container.put(shortfall) at
decision time (lead=0 in this path). So a RAISE should be instant; a LOWER (frac->0 clears targets)
should decay via consumption. The shift lever changes op3_q/batch_size in the action dict at once.

Four checks + an Excel trace:
  1. target_propagation: after a step, sim.inventory_buffer_targets == frac x I1344 (same step)?
  2. container_jump_on_raise: frac 0->1 -> container level jumps within the same step? (latency)
  3. decay_on_lower: frac 1->0 -> container decays over later steps? (no instant removal)
  4. outcome_divergence: constant frac=0 vs frac=1 -> different total_inventory, fill, ReT?
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from supply_chain.continuous_its_env import make_continuous_its_track_a_env, _I1344, _BUFFER_KEYS
from supply_chain.episode_metrics import compute_episode_metrics

CONTAINERS = {"op3_rm": "raw_material_wdc", "op5_rm": "raw_material_al", "op9_rations": "rations_sb"}
SHIFT_SIGS = {1: -1.0, 2: 0.0, 3: 1.0}


def _levels(sim):
    return {k: float(getattr(sim, attr).level) for k, attr in CONTAINERS.items()}


def trace(regime, phi, psi, max_steps, schedule, shift_sig=0.0):
    """Run a deterministic frac schedule; record per-step action, target, container levels, outcome."""
    env = make_continuous_its_track_a_env(
        reward_mode="ReT_excel_delta", observation_version="v6", risk_level=regime,
        risk_frequency_multiplier=phi, risk_impact_multiplier=psi, stochastic_pt=False,
        max_steps=max_steps, step_size_hours=168.0, init_frac=0.0)
    env.reset(seed=1)
    sim = env.unwrapped.sim
    rows = []
    for t in range(max_steps):
        frac = float(schedule(t))
        _, _r, done, trunc, info = env.step(np.array([frac, shift_sig], dtype=np.float32))
        tgt = dict(getattr(sim, "inventory_buffer_targets", {}) or {})
        lv = _levels(sim)
        rows.append({
            "week": t, "now_h": float(sim.env.now), "frac_cmd": frac,
            "shift": int(info.get("continuous_its_shift", -1)),
            "target_op9": float(tgt.get("op9_rations", 0.0)),
            "expected_target_op9": frac * float(_I1344["op9_rations"]),
            "lvl_wdc": lv["op3_rm"], "lvl_al": lv["op5_rm"], "lvl_rations": lv["op9_rations"],
            "total_inventory": float(info.get("total_inventory", np.nan)),
            "new_delivered": float(info.get("new_delivered", np.nan)),
            "new_demanded": float(info.get("new_demanded", np.nan)),
            "backorder_qty": float(info.get("pending_backorder_qty", np.nan)),
        })
        if done or trunc:
            break
    m = compute_episode_metrics(sim)
    return rows, m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="current")
    ap.add_argument("--phi", type=float, default=4.0)
    ap.add_argument("--psi", type=float, default=1.5)
    ap.add_argument("--max-steps", type=int, default=52)
    ap.add_argument("--raise-week", type=int, default=10)
    ap.add_argument("--lower-week", type=int, default=30)
    ap.add_argument("--output", default="outputs/audits/action_propagation_2026-06-27")
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # step-change schedule: 0 until raise_week, 1 until lower_week, then 0
    def step_sched(t):
        if t < args.raise_week:
            return 0.0
        if t < args.lower_week:
            return 1.0
        return 0.0

    rows, m_step = trace(args.regime, args.phi, args.psi, args.max_steps, step_sched)

    # checks
    def at(week):
        return next((r for r in rows if r["week"] == week), None)

    # 1. target propagation (same-step): target == expected for all rows with frac>0
    prop_ok = all(abs(r["target_op9"] - r["expected_target_op9"]) < 1.0
                  for r in rows if r["frac_cmd"] > 1e-6)
    # 2. container jump on raise: level at raise_week vs week before
    pre = at(args.raise_week - 1)
    post = at(args.raise_week)
    jump = (post["lvl_rations"] - pre["lvl_rations"]) if (pre and post) else float("nan")
    target_at_raise = post["expected_target_op9"] if post else float("nan")
    # same-step put(shortfall) then consumption draws down within the week -> ~instant if it reaches
    # most of target the same step (0.8 accounts for in-week consumption of a high-throughput node).
    jump_reaches_target = bool(post and post["lvl_rations"] >= 0.8 * target_at_raise)
    # 3. lower: strategic target cleared instantly (frac->0 -> targets={}); physical level then drifts
    #    down only as fast as consumption (slow here because fill is downstream-bottlenecked).
    target_cleared = all(r["target_op9"] == 0.0 for r in rows if r["week"] >= args.lower_week)
    after = [r["lvl_rations"] for r in rows if r["week"] >= args.lower_week]
    decays = len(after) > 2 and after[-1] < after[0] - 1.0

    # 4. outcome divergence: constant frac0 vs frac1 (no risk schedule change; same seed)
    rows0, m0 = trace(args.regime, args.phi, args.psi, args.max_steps, lambda t: 0.0)
    rows1, m1 = trace(args.regime, args.phi, args.psi, args.max_steps, lambda t: 1.0)
    inv0 = float(np.nanmean([r["total_inventory"] for r in rows0]))
    inv1 = float(np.nanmean([r["total_inventory"] for r in rows1]))
    fill0 = float(m0.get("fill_rate", np.nan))
    fill1 = float(m1.get("fill_rate", np.nan))
    ret0 = float(m0.get("ret_excel", np.nan))
    ret1 = float(m1.get("ret_excel", np.nan))
    inv_diverges = abs(inv1 - inv0) > 1.0
    outcome_diverges = (abs(ret1 - ret0) > 1e-6) or (abs(fill1 - fill0) > 1e-4)

    # shift contrast (S1 vs S3, constant frac=0.5) — effect shows in delivered_rations (no shift-hours
    # metric exists; Track A shift has weak authority over delivered fill = the F11 downstream bottleneck)
    _, mS1 = trace(args.regime, args.phi, args.psi, args.max_steps, lambda t: 0.5, shift_sig=-1.0)
    _, mS3 = trace(args.regime, args.phi, args.psi, args.max_steps, lambda t: 0.5, shift_sig=1.0)
    del1 = float(mS1.get("delivered_rations", np.nan))
    del3 = float(mS3.get("delivered_rations", np.nan))
    shift_changes_delivered = abs(del3 - del1) > 1.0

    # export Excel trace (the step-change run) so the user can SEE it
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        xlsx = out / "action_propagation_trace.xlsx"
        with pd.ExcelWriter(xlsx) as xl:
            df.to_excel(xl, sheet_name="step_change_trace", index=False)
            pd.DataFrame(rows0).to_excel(xl, sheet_name="const_frac0", index=False)
            pd.DataFrame(rows1).to_excel(xl, sheet_name="const_frac1", index=False)
        xlsx_msg = str(xlsx)
    except Exception as e:  # pragma: no cover
        import csv
        with open(out / "action_propagation_trace.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        xlsx_msg = f"{out}/action_propagation_trace.csv (pandas unavailable: {e})"

    print(f"\n=== ACTION->ENVIRONMENT PROPAGATION ({args.regime} φ{args.phi}/ψ{args.psi}, h{args.max_steps}) ===")
    print("[1] target propagation (same step): "
          f"{'OK - target == frac x I1344 instantly' if prop_ok else 'BROKEN - target != commanded'}")
    print(f"[2] container jump on RAISE (week {args.raise_week}, frac 0->1):")
    print(f"    rations level {pre['lvl_rations']:.0f} -> {post['lvl_rations']:.0f} "
          f"(+{jump:.0f}); target={target_at_raise:.0f} "
          f"-> {'INSTANT (reaches target same step)' if jump_reaches_target else 'LAGGED/partial'}")
    print(f"[3] LOWER (week {args.lower_week}, frac 1->0): "
          f"target cleared instantly={target_cleared}; physical level {after[0]:.0f}->{after[-1]:.0f} "
          f"({'decays via consumption' if decays else 'persists (consumption slow vs stock; fill is downstream-bottlenecked)'})")
    print("[4] outcome divergence (constant frac=0 vs frac=1, same seed):")
    print(f"    mean total_inventory: {inv0:.0f} vs {inv1:.0f}  ({'DIVERGES' if inv_diverges else 'SAME'})")
    print(f"    fill_rate: {fill0:.4f} vs {fill1:.4f} | ret_excel: {ret0:.5f} vs {ret1:.5f}  "
          f"({'OUTCOME CHANGES' if outcome_diverges else 'NO OUTCOME CHANGE'})")
    print(f"[shift] S1 vs S3 (frac=0.5): delivered_rations {del1:.0f} vs {del3:.0f}  "
          f"({'shift moves delivered (weakly)' if shift_changes_delivered else 'no change'})")
    verdict = prop_ok and jump_reaches_target and target_cleared and inv_diverges and outcome_diverges
    print(f"\n=> {'ACTIONS PROPAGATE: target=instant, raise=same-step, lower=target-cleared, OUTCOMES change' if verdict else 'PROPAGATION ISSUE - investigate'}")
    print(f"WROTE trace: {xlsx_msg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
