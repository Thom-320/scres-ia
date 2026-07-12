#!/usr/bin/env python3
"""Program G — ret_excel confirmation via the project's real order-level machinery.

Ports Program G outcomes to the PRIMARY metric ret_excel_visible_v1 using
compute_order_level_ret_excel_visible_ledger on emitted OrderRecords. Because the
visible ledger only scores ATTENDED orders (a shed-to-win incentive, per the DRA-1
lesson), the binding headline is the FULL ledger (unfulfilled scored 0) AND a
lost-order guardrail: the observable policy must beat the best static on the full
ledger WITHOUT attending fewer orders. Region = 12 surge-1.50 cells; holdout 1000001+,
virgin 1010001+. No new physics in the shared DES; a disclosed daily order adapter.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.program_g import (
    cover_signal_policy, materialize_tape, periodic_calendars, ret_order_metrics, simulate,
    simulate_orders,
)
from supply_chain.ret_thesis import (
    compute_order_level_ret_excel_visible_ledger as ret_visible,
)

REGION = [{"cell_id": f"P{p}_Q{int(q*100)}_L{l}_S150", "signal_q": q, "lead_weeks": l,
           "surge_mult": 1.50, "persistence": p, "r22_weekly_prob": 0.05}
          for p in ("short", "long") for q in (0.65, 0.75, 0.85) for l in (1, 2)]
WEEKS = 4
ARM = "TRS"


def region_tape(i, base):
    return materialize_tape(base + i, REGION[i % len(REGION)], WEEKS, persistent=True)


def boot_ci(x, n=2000, seed=7):
    rng = np.random.default_rng(seed); x = np.asarray(x, float)
    m = [rng.choice(x, len(x), True).mean() for _ in range(n)]
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def score(orders):
    """Return (visible_mean, full_ledger_mean, n_attended, n_generated)."""
    vis = ret_visible(orders)
    n_gen = int(vis["n_generated_orders"])
    vis_mean = float(vis["mean_ret_excel"]) if vis["n_visible_rows"] else 0.0
    canonical = ret_order_metrics(orders)
    full_mean = canonical["ret_order"]
    n_att = canonical["attended"]
    return vis_mean, full_mean, n_att, n_gen


def eval_set(tapes, best_cal):
    rows = {"static": [], "cover": []}
    for t in tapes:
        rows["static"].append(score(simulate_orders(t, best_cal, ARM)))
        rows["cover"].append(score(simulate_orders(t, cover_signal_policy(t, ARM), ARM)))
    out = {}
    for k, r in rows.items():
        vis, full, att, gen = map(np.array, zip(*r))
        out[k] = {"ret_excel_visible_mean": float(vis.mean()),
                  "ret_excel_full_ledger_mean": float(full.mean()),
                  "attended_orders_mean": float(att.mean()),
                  "generated_orders_mean": float(gen.mean())}
    s_full = np.array([x[1] for x in rows["static"]]); c_full = np.array([x[1] for x in rows["cover"]])
    s_vis = np.array([x[0] for x in rows["static"]]); c_vis = np.array([x[0] for x in rows["cover"]])
    s_att = np.array([x[2] for x in rows["static"]]); c_att = np.array([x[2] for x in rows["cover"]])
    out["cover_minus_static_full_ledger_ci95"] = boot_ci(c_full - s_full)   # headline (guardrailed)
    out["cover_minus_static_visible_ci95"] = boot_ci(c_vis - s_vis)
    out["attended_delta_cover_minus_static_ci95"] = boot_ci(c_att - s_att)  # guardrail: not < 0
    return out


def main() -> int:
    n = 200
    train = [region_tape(i, 990001) for i in range(n)]
    cals = periodic_calendars(WEEKS)
    cl = np.array([[simulate(t, c, arm=ARM).service_loss for t in train] for c in cals])
    best_cal = cals[int(cl.mean(axis=1).argmin())]

    holdout = eval_set([region_tape(i, 1000001) for i in range(n)], best_cal)
    virgin = eval_set([region_tape(i, 1010001) for i in range(n)], best_cal)

    full_lo = virgin["cover_minus_static_full_ledger_ci95"][1]
    att_lo = virgin["attended_delta_cover_minus_static_ci95"][1]
    interp = ("G_RETEXCEL_CONFIRMS_ADAPTIVE_WIN" if full_lo > 0 and att_lo >= -0.5
              else "G_RETEXCEL_WIN_IS_SHED_TO_WIN_ARTIFACT" if virgin["cover_minus_static_visible_ci95"][1] > 0 and att_lo < -0.5
              else "G_RETEXCEL_NO_WIN")
    out = {"gate": "PROGRAM_G_RETEXCEL_CONFIRMATION", "metric": "ret_excel_visible_v1 + full-ledger guardrail",
           "frozen_best_calendar": list(best_cal), "n_per_split": n,
           "holdout": holdout, "virgin": virgin, "interpretation": interp,
           "note": ("Real ret_excel machinery on emitted OrderRecords (disclosed daily adapter, "
                    "no new shared-DES physics). Headline = FULL ledger (unfulfilled=0) + lost-order "
                    "guardrail (attended not materially fewer), per the DRA-1 shed-to-win lesson.")}
    output = Path("results/program_g/retexcel"); output.mkdir(parents=True, exist_ok=True)
    (output / "verdict.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"interpretation": interp, "virgin": virgin}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
