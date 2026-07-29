#!/usr/bin/env python3
"""Garrido's prescribed cheap gate: buffer targets only, dispatch frozen, under CRN.

Development screen only. Cannot open confirmation universes and cannot authorize RL.

Garrido specified this gate in `docs/garrido_meeting_decision_variables_2026-07-03/
main.tex`: freeze `op10_q`/`op12_q`, vary ONLY the buffer targets under common random
numbers, and measure whether headroom exists BEFORE building or training anything. On
2026-07-28 he then authorised widening buffers to nodes not currently modelled, upstream
and downstream: *"poner buffers en los nodos no considerados es valido, perfectamente
valido"*.

Adding those nodes is new physics. This screen is the cheap precondition: it measures the
clairvoyant ceiling of the buffer family that ALREADY exists (`op3_rm`, `op5_rm`,
`op9_rations` -- raw material at the WDC, raw material at the assembly line, and rations
at the supply base). If perfect information over the existing buffer family buys nothing,
that is strong prior evidence about what more buffer nodes of the same kind can buy, and
it is evidence obtained without writing a single line of new physics.

    B    = max_g  mean_t ReT(g, t)     best fixed buffer posture (static bar)
    C    = mean_t max_g ReT(g, t)      clairvoyant per-tape ceiling
    H_PI = C - B                       perfect-information headroom

H_PI is a DIAGNOSTIC CEILING, not learner opportunity: the clairvoyant selects per tape
using outcomes no deployable policy observes. A large H_PI would not authorise training;
a near-zero H_PI does close the family, because no policy of any kind can exceed it.

CAVEAT ON THE CURRENT BUFFER MECHANISM. `_deliver_buffer_top_up` (supply_chain.py:1137)
satisfies a shortfall with an unmatched `container.put(shortfall)`: there is no upstream
`.get()`, containers are effectively unbounded, and the injected quantity carries no price.
Mass still balances -- `flow_ledger()` counts the injection as a source term
(supply_chain.py:1956-1970) -- so this is not an arithmetic leak; it is an exogenous,
uncapacitated, unpriced supply. That biases this screen IN FAVOUR of large buffers. A
null measured under this generous mechanism is therefore conservative: physically sourced
buffers can only be worse.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.config import HOURS_PER_WEEK, INVENTORY_BUFFERS  # noqa: E402
from supply_chain.config import THESIS_FAITHFUL_PROTOCOL as P  # noqa: E402
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402

# Thesis buffer ladder, Table 6.16: the five replenishment periods give five target
# levels per node. The grid is that ladder plus a zero-buffer floor, so the screen
# spans "no strategic buffer at all" to the largest level the thesis contemplates.
LADDER = [168, 336, 504, 672, 1344]
NODES = ["op3_rm", "op5_rm", "op9_rations"]
LEVELS = {n: [0.0] + [float(INVENTORY_BUFFERS[k][n]) for k in LADDER] for n in NODES}


def evaluate(buffers: dict[str, float], tape: int, horizon_weeks: int) -> dict[str, float]:
    """One full-DES episode at a fixed buffer posture, CRN-locked to `tape`.

    Dispatch is left at its thesis default and never written, which is exactly the
    "congelar op10_q/op12_q" condition Garrido asked for.
    """
    sim = MFSCSimulation(
        shifts=1,
        initial_buffers=dict(buffers),
        seed=tape,
        horizon=float(horizon_weeks) * HOURS_PER_WEEK,
        risks_enabled=True,
        risk_level="current",
        strict_exogenous_crn=True,
        year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"],
        r14_defect_mode=P["r14_defect_mode"],
    )
    sim.step(action=None, step_hours=float(horizon_weeks) * HOURS_PER_WEEK)
    m = compute_episode_metrics(sim)
    ledger = sim.flow_ledger()
    return {
        "ret_excel": float(m["ret_excel"]),
        # Uncensored companion. Under the thesis risk regime `ret_excel` scores each
        # policy on a DIFFERENT population -- the omitted-order fraction ranges 3.9%
        # to 18.6% across shift postures -- and loses ordinal discrimination entirely
        # (it ranks the worst posture first). `ret_excel_full_ledger` applies the same
        # formula over every order and stays monotone. Both are reported; neither is
        # selected after the fact.
        "ret_excel_full_ledger": float(m["ret_excel_full_ledger"]),
        "ret_excel_visible_n": float(m["ret_excel_visible_n"]),
        "ret_excel_omitted_n": float(m["ret_excel_omitted_n"]),
        "flow_fill_rate": float(m["flow_fill_rate"]),
        "lost_orders": float(m["lost_orders"]),
        "service_loss_auc": float(m["service_loss_auc_ration_hours"]),
        # The exogenous-injection counters. These are what a physically sourced
        # mechanism must drive to zero; reported so the bias is quantified, not assumed.
        "strategic_raw_injected": float(sim.total_strategic_raw_injected),
        "strategic_rations_injected": float(sim.total_strategic_rations_injected),
        "raw_residual": float(ledger["raw_residual"]),
        "ration_residual": float(ledger["ration_residual"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tapes", type=int, default=12)
    ap.add_argument("--seed-start", type=int, default=1_220_001)
    ap.add_argument("--horizon-weeks", type=int, default=52)
    ap.add_argument("--output-dir", type=Path,
                    default=Path("results/authority_ladder/buffer_gate"))
    args = ap.parse_args()

    tapes = [args.seed_start + i for i in range(args.tapes)]
    grid = [dict(zip(NODES, combo))
            for combo in itertools.product(*(LEVELS[n] for n in NODES))]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    METRICS = ("ret_excel", "ret_excel_full_ledger")
    rows = {k: np.zeros((len(grid), len(tapes)), dtype=np.float64) for k in METRICS}
    fill = np.zeros((len(grid), len(tapes)), dtype=np.float64)
    injected = np.zeros_like(fill)
    omit_frac = np.zeros_like(fill)
    max_resid = 0.0
    for gi, buffers in enumerate(grid):
        for ti, tape in enumerate(tapes):
            r = evaluate(buffers, tape, args.horizon_weeks)
            for k in METRICS:
                rows[k][gi, ti] = r[k]
            fill[gi, ti] = r["flow_fill_rate"]
            injected[gi, ti] = (r["strategic_raw_injected"]
                                + r["strategic_rations_injected"])
            seen = r["ret_excel_visible_n"] + r["ret_excel_omitted_n"]
            omit_frac[gi, ti] = r["ret_excel_omitted_n"] / max(seen, 1.0)
            max_resid = max(max_resid, abs(r["raw_residual"]),
                            abs(r["ration_residual"]))
        if (gi + 1) % 24 == 0:
            print(f"  {gi + 1}/{len(grid)} ({time.perf_counter() - started:.0f}s)",
                  flush=True)

    rng = np.random.default_rng(20260728)
    by_metric: dict[str, dict] = {}
    for k in METRICS:
        arr = rows[k]
        per_setting_mean = arr.mean(axis=1)
        bg = int(per_setting_mean.argmax())
        gap = arr.max(axis=0) - arr[bg, :]
        boot = np.array([rng.choice(gap, len(tapes), True).mean()
                         for _ in range(10_000)])
        by_metric[k] = {
            "best_setting": grid[bg],
            "static_bar_mean": float(per_setting_mean[bg]),
            "clairvoyant_ceiling_mean": float(arr.max(axis=0).mean()),
            "h_pi": float(arr.max(axis=0).mean() - per_setting_mean[bg]),
            "h_pi_lcb95": float(np.quantile(boot, 0.05)),
            "grid_span_mean": float(per_setting_mean.max() - per_setting_mean.min()),
            "zero_buffer_mean": float(per_setting_mean[0]),
            "max_buffer_mean": float(per_setting_mean[len(grid) - 1]),
            "best_is_all_max": bool(bg == len(grid) - 1),
            "n_tapes_with_strictly_better_setting": int((gap > 1e-12).sum()),
        }

    # Primary reporting stays on the canonical metric; the companion is disclosed.
    per_setting_mean = rows["ret_excel"].mean(axis=1)
    best_g = int(per_setting_mean.argmax())
    static_bar = float(per_setting_mean[best_g])
    ceiling = float(rows["ret_excel"].max(axis=0).mean())
    per_tape_gap = rows["ret_excel"].max(axis=0) - rows["ret_excel"][best_g, :]
    boot = np.array([rng.choice(per_tape_gap, len(tapes), True).mean()
                     for _ in range(10_000)])

    # Is the optimum interior, or does the screen simply want the largest buffer
    # everywhere? The latter is the signature of the unpriced exogenous source.
    zero_g = 0  # all-zero buffers is the first product entry
    max_g = len(grid) - 1  # all-largest is the last

    payload = {
        "schema_version": "authority_ladder_buffer_gate_v1",
        "claim_status": "DEVELOPMENT_SCREEN_NO_CLAIM",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "gate_source": "docs/garrido_meeting_decision_variables_2026-07-03/main.tex",
        "metric": "ret_excel",
        "nodes": NODES,
        "levels": LEVELS,
        "n_settings": len(grid),
        "tapes": tapes,
        "horizon_weeks": args.horizon_weeks,
        "dispatch_frozen": True,
        "best_setting": grid[best_g],
        "static_bar_mean_ret": static_bar,
        "clairvoyant_ceiling_mean_ret": ceiling,
        "h_pi": ceiling - static_bar,
        "h_pi_lcb95": float(np.quantile(boot, 0.05)),
        "n_tapes_with_strictly_better_setting": int((per_tape_gap > 1e-12).sum()),
        "zero_buffer_mean_ret": float(per_setting_mean[zero_g]),
        "max_buffer_mean_ret": float(per_setting_mean[max_g]),
        "buffer_span_mean_ret": float(per_setting_mean.max() - per_setting_mean.min()),
        "best_is_all_max": bool(best_g == max_g),
        "mean_exogenous_injected_at_best": float(injected[best_g].mean()),
        "mean_fill_at_best": float(fill[best_g].mean()),
        "max_abs_mass_residual": max_resid,
        "by_metric": by_metric,
        "mean_omitted_order_fraction_at_best": float(omit_frac[best_g].mean()),
        "omitted_fraction_span_across_grid": float(
            omit_frac.mean(axis=1).max() - omit_frac.mean(axis=1).min()),
        "h_pi_is_diagnostic_ceiling_not_learner_opportunity": True,
        "bias_note": ("buffers are topped up from an exogenous, uncapacitated, unpriced "
                      "source, so this screen is biased toward large buffers; a null "
                      "here is conservative"),
        "elapsed_seconds": time.perf_counter() - started,
    }
    body = json.dumps(payload, indent=1, sort_keys=True)
    payload["self_sha256"] = sha256(body.encode()).hexdigest()
    path = args.output_dir / "screen_result.json"
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")

    print(f"\nsettings={len(grid)} tapes={len(tapes)}")
    print(f"  best posture         = {grid[best_g]}")
    print(f"  static bar           = {static_bar:.8f}")
    print(f"  clairvoyant ceiling  = {ceiling:.8f}")
    print(f"  H_PI                 = {ceiling - static_bar:.6g}  "
          f"LCB95={payload['h_pi_lcb95']:.6g}")
    print(f"  ReT span over grid   = {payload['buffer_span_mean_ret']:.6g}  "
          f"(zero={payload['zero_buffer_mean_ret']:.6f} "
          f"max={payload['max_buffer_mean_ret']:.6f})")
    print(f"  best is all-max      = {payload['best_is_all_max']}")
    print(f"  exogenous injected   = {payload['mean_exogenous_injected_at_best']:.0f}")
    print(f"  omitted-order frac   = {payload['mean_omitted_order_fraction_at_best']:.4f} "
          f"(span across grid {payload['omitted_fraction_span_across_grid']:.4f})")
    print("\n  metric comparison (censored vs uncensored):")
    for k, v in by_metric.items():
        print(f"    {k:24s} bar={v['static_bar_mean']:.8f} "
              f"H_PI={v['h_pi']:.6g} LCB95={v['h_pi_lcb95']:.6g} "
              f"span={v['grid_span_mean']:.6g}")
        print(f"    {'':24s} best={v['best_setting']}")
    print(f"-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
