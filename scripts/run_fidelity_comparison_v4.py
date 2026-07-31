#!/usr/bin/env python3
"""The six-moment concordance screen against `fidelity_reference_v4`, with its own artifact.

**Why this script exists.** The manuscript's validation table cited
`results/metric_audit/r14_seed_arms_v1/result.json`, which cannot support it: that artifact
is HALTED (`..._FALSIFIER_FAILED`), it is scored against the v3 reference rather than the v4
the section declares, its `scored_moments` EXCLUDES `scored_orders_per_year` -- the row the
section printed in bold -- and it swept the pre-amendment epsilon band. The published `d_k`
did not come from it either: they were recomputed by hand against v4, outside the pipeline
(`ret_mean` R1r 1.249 in the artifact against 1.52 in the section). So the table had no
source. This runner gives it one.

**What it is, and is not.** One arm: the shipped defaults. There is nothing to dominate and
no verdict to render, so none is computed -- `non_dominated` over a single cell is a
tautology, and the master contract's output is a SET, not a score. This is a
DEVELOPMENT-LEVEL CONCORDANCE SCREEN, not a behavioural validation:

* his sheets are ten DESIGNED configurations per family, not replicates, so `s_k` is a
  between-configuration spread and `d_k` is a *descriptive standardized discrepancy*, not an
  inferential test statistic;
* the sheets are not even exchangeable among themselves -- CF1 and CF2 run ~19.8 thesis
  years against ~9.8 for the other eight, and CF6/CF10 are near-replicates of one another;
* configurations, seeds, horizons and scored windows are NOT matched between his runs and
  ours.

All three facts are measured and written into the artifact so the caveat travels with the
number rather than living in a paragraph someone can drop.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.arm_runner import (  # noqa: E402
    THESIS_YEAR_HOURS, aggregate, build_reference, episode_moments, run_falsifiers,
    scored_orders, seal_and_write,
)
from supply_chain.config import HOURS_PER_WEEK, LEAD_TIME_PROMISE  # noqa: E402
from supply_chain.config import GARRIDO_FULFILLMENT_DELAY_HOURS as DELAY  # noqa: E402
from supply_chain.config import THESIS_FAITHFUL_PROTOCOL as P  # noqa: E402
from supply_chain.fidelity_moments import MOMENT_NAMES  # noqa: E402
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402

FAMILIES = {"R1r": ("R11", "R12", "R13", "R14"), "R2r": ("R21", "R22", "R23", "R24")}
ROOTS = tuple(3_500_001 + i for i in range(12))   # virgin: no artifact under results/ uses them
ARM = "shipped_defaults"
EXPECTED_REFERENCE_SHA = "32e23a79b43f76a7"       # v4, first 16
# The window convention, now matched on both sides: `max(OPTj) - min(OPTj)` over the scored
# population. `arm_runner` used `horizon - warmup` until this run measured the two apart by
# 1.5% (R1r) and 2.2% (R2r) -- larger than R1r's own gap to the reference, so the convention
# was deciding the d_k. The abandoned convention is still computed and reported as a
# sensitivity, so the size of that choice stays visible in the artifact.


def run_episode(*, family: str, seed: int, horizon: float) -> MFSCSimulation:
    sim = MFSCSimulation(
        shifts=1,
        initial_buffers={n: 0.0 for n in ("op3_rm", "op5_rm", "op9_rations")},
        inventory_replenishment_period=0.0, seed=seed, horizon=horizon,
        risks_enabled=True, risk_level="current", enabled_risks=set(FAMILIES[family]),
        risk_overrides={r: "increased" for r in FAMILIES[family]},
        strict_exogenous_crn=True, year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"], r14_defect_mode=P["r14_defect_mode"])
    sim.step(action=None, step_hours=horizon)
    return sim


def garrido_delayed_share(workbook_dir: Path) -> dict:
    """His delayed share, under the SAME rule applied to our orders, sealed here.

    Reuses `build_fidelity_reference_v4.read_sheet` rather than re-implementing the header
    detection -- CF2 is only reachable through it. "Delayed" is `k >= 1` in
    `CTj = 48 + k*24 + delta`, i.e. `CTj >= 72`, the same lattice step tested on our side.
    Without this the manuscript's "83.5% of his" would be an unsealed side-computation, which
    is the defect this whole runner exists to remove.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_fidelity_reference_v4 import WORKBOOKS, read_sheet  # noqa: E402

    out: dict[str, dict] = {}
    for family, sheets in (("R1r", range(1, 11)), ("R2r", range(11, 21))):
        shares, n_tot, n_del = [], 0, 0
        for i in sheets:
            for wb in WORKBOOKS:
                path = workbook_dir / wb
                if not path.exists():
                    continue
                d = read_sheet(path, f"CF{i}")
                if d is None or not len(d):
                    continue
                ct = [float(v) for v in d["CTj"].tolist() if v == v]
                dd = sum(1 for c in ct if c >= 72.0)
                shares.append(dd / len(ct))
                n_tot, n_del = n_tot + len(ct), n_del + dd
                break
        out[family] = {"mean_over_sheets": float(np.mean(shares)),
                       "pooled": n_del / n_tot, "n_sheets": len(shares), "n_rows": n_tot,
                       "rule": "CTj >= 72 h, i.e. k >= 1 in CTj = 48 + k*24 + delta"}
    return out


def reference_shape(blob: dict) -> dict:
    """What the reference is made of, measured -- not asserted in prose."""
    per = blob["per_sheet"]
    out: dict[str, dict] = {}
    for family, ref in blob["reference_by_family"].items():
        sheets = ref["_sheets"]
        win = {s: per[s]["_window_years"] for s in sheets}
        # The closest pair on the six moments, in units of each moment's family spread.
        closest, best = None, float("inf")
        for i, a in enumerate(sheets):
            for b in sheets[i + 1:]:
                d = max(abs(per[a][m] - per[b][m]) / (ref[m]["spread"] or 1e-12)
                        for m in MOMENT_NAMES)
                if d < best:
                    closest, best = (a, b), d
        out[family] = {
            "n_sheets": len(sheets),
            "window_years_min": min(win.values()), "window_years_max": max(win.values()),
            "window_years_by_sheet": win,
            "closest_pair": list(closest or ()),
            "closest_pair_max_moment_gap_in_spreads": best,
            "sheets_are_designed_configurations_not_replicates": True,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contract", type=Path,
                    default=Path("docs/MANUSCRIPT_MODEL_VALIDATION_SECTION_2026-07-31.md"))
    ap.add_argument("--reference", type=Path,
                    default=Path("results/metric_audit/fidelity_reference_v4/result.json"))
    ap.add_argument("--roots", nargs="+", type=int, default=list(ROOTS))
    ap.add_argument("--horizon-weeks", type=int, default=52)
    ap.add_argument("--workbook-dir", type=Path, default=Path.home() / "Downloads")
    ap.add_argument("--output", type=Path,
                    default=Path("results/metric_audit/fidelity_comparison_v4/result.json"))
    args = ap.parse_args()

    horizon = float(args.horizon_weeks * HOURS_PER_WEEK)
    ref_blob = json.loads(args.reference.read_text())
    t0 = time.perf_counter()

    per_family: dict[str, list] = {}
    window: dict[str, dict] = {}
    ctj: dict[str, dict] = {}
    for family in FAMILIES:
        rows, alt_rate, floor, mins, delayed, on_grid = [], [], 0, [], 0, 0
        legacy_rate: list[float] = []
        total = 0
        for seed in args.roots:
            sim = run_episode(family=family, seed=seed, horizon=horizon)
            rows.append(episode_moments(sim))
            orders = [o for o in scored_orders(sim)
                      if not bool(getattr(o, "lost", False))
                      and getattr(o, "OATj", None) is not None]
            opt = [float(o.OPTj) for o in orders]
            span = max(max(opt) - min(opt), 1e-9)
            alt_rate.append(len(orders) / (span / THESIS_YEAR_HOURS))
            legacy = max(horizon - float(sim.warmup_time), 1e-9)
            legacy_rate.append(len(orders) / (legacy / THESIS_YEAR_HOURS))
            cts = [float(o.CTj) for o in orders if o.CTj is not None]
            total += len(cts)
            mins.append(min(cts) if cts else float("nan"))
            floor += sum(1 for c in cts if c < float(DELAY) - 1e-6)
            for c in cts:
                if c > float(DELAY) + 1e-6:
                    delayed += 1
                    if abs((c - float(LEAD_TIME_PROMISE)) % 24.0) < 1e-6:
                        on_grid += 1
        per_family[family] = rows
        ours = float(np.mean([r["scored_orders_per_year"] for r in rows]))
        alt, legacy_mean = float(np.mean(alt_rate)), float(np.mean(legacy_rate))
        window[family] = {
            "rate_adopted_last_minus_first_optj": ours,   # matches v4's estimator
            "rate_recomputed_here_as_a_check": alt,
            "rate_abandoned_horizon_minus_warmup": legacy_mean,
            "relative_gap": abs(legacy_mean - ours) / max(ours, 1e-9),
            "note": ("his window ends at the last order; the abandoned convention ended at "
                     "the horizon. The gap below is the size of that choice"),
        }
        ctj[family] = {
            "min_ctj_by_root": mins,
            "orders_below_shipped_delay": floor,
            "orders_with_ctj": total,
            "delayed_orders": delayed,
            "delayed_share": (delayed / total) if total else float("nan"),
            "delayed_on_24h_grid_from_LT": on_grid,
            "grid_share": (on_grid / delayed) if delayed else float("nan"),
        }
        print(f"  {family} ({time.perf_counter() - t0:.0f}s)", flush=True)

    reference = {f: build_reference(ref_blob, f) for f in FAMILIES}
    cells = {f: {ARM: aggregate(per_family[f], reference[f])} for f in FAMILIES}
    shape = reference_shape(ref_blob)
    his_delayed = garrido_delayed_share(args.workbook_dir)
    for f in FAMILIES:
        ctj[f]["garrido_delayed_share"] = his_delayed[f]

    checks = {
        "f1_reference_is_v4_and_complete": lambda: (
            (str(ref_blob.get("schema_version")) == "fidelity_reference_v4"
             and not ref_blob.get("sheets_missing")
             and str(ref_blob.get("self_sha256", ""))[:16] == EXPECTED_REFERENCE_SHA),
            {"why_it_can_fail": "pointing this runner at v3, or at a rebuilt v4 with a new seal",
             "schema": ref_blob.get("schema_version"),
             "sheets_missing": ref_blob.get("sheets_missing"),
             "self_sha256_16": str(ref_blob.get("self_sha256", ""))[:16],
             "expected": EXPECTED_REFERENCE_SHA}),
        "f2_rate_estimator_is_the_reference_estimator": lambda: (
            max(abs(ref_blob["per_sheet"][s]["scored_orders_per_year"]
                    - ref_blob["per_sheet"][s]["_n_rows"]
                    / ref_blob["per_sheet"][s]["_window_years"])
                for s in ref_blob["per_sheet"]) < 1e-9,
            {"why_it_can_fail": ("this reconstructs the REFERENCE's own rate from its stored "
                                 "row count and window and checks it against the value it "
                                 "published. If v4 used a different denominator than the one "
                                 "our side now mirrors, this disagrees and the row is not "
                                 "comparable across sides"),
             "max_abs_error": max(
                 abs(ref_blob["per_sheet"][s]["scored_orders_per_year"]
                     - ref_blob["per_sheet"][s]["_n_rows"]
                     / ref_blob["per_sheet"][s]["_window_years"])
                 for s in ref_blob["per_sheet"]),
             "convention_sensitivity_measured_not_gated": {
                 f: window[f]["relative_gap"] for f in FAMILIES},
             "note": ("the abandoned `horizon - warmup` convention differs from the adopted "
                      f"one by up to {max(window[f]['relative_gap'] for f in FAMILIES):.1%}, "
                      "which is why it was abandoned rather than tolerated")}),
        "f3_no_moment_is_degenerate": lambda: (
            all(v == v for f in FAMILIES for v in cells[f][ARM]["discrepancies"].values()),
            {"why_it_can_fail": ("a reference moment with zero between-sheet spread carries "
                                 "no scale and comes back NaN; the row could not be quoted"),
             "d_k": {f: cells[f][ARM]["discrepancies"] for f in FAMILIES}}),
        "f4_autotomy_is_structurally_unreachable": lambda: (
            all(min(m for m in ctj[f]["min_ctj_by_root"]) >= float(LEAD_TIME_PROMISE)
                for f in FAMILIES)
            and all(r["autotomy_share"] == 0.0 for f in FAMILIES for r in per_family[f]),
            {"why_it_can_fail": ("if any order completed at or below LT the autotomy branch "
                                 "would fire and the section's stated cause for that gap "
                                 "would be wrong"),
             "shipped_delay_hours": float(DELAY), "lead_time_promise": float(LEAD_TIME_PROMISE),
             "min_ctj": {f: min(ctj[f]["min_ctj_by_root"]) for f in FAMILIES},
             "orders_below_shipped_delay": {f: ctj[f]["orders_below_shipped_delay"]
                                            for f in FAMILIES}}),
        "f5_delayed_orders_ride_the_24h_grid": lambda: (
            all(ctj[f]["delayed_orders"] > 0 and ctj[f]["grid_share"] > 0.99
                for f in FAMILIES),
            {"why_it_can_fail": ("the grid claim is measured on the DELAYED subpopulation "
                                 "only; if those orders left the 24 h lattice, or if there "
                                 "were none, the cadence claim would not hold"),
             "delayed": {f: ctj[f]["delayed_orders"] for f in FAMILIES},
             "grid_share": {f: ctj[f]["grid_share"] for f in FAMILIES},
             "scope_note": ("the modal order completes at the shipped constant "
                            f"{float(DELAY)} h, which is NOT on the lattice measured from "
                            f"LT = {float(LEAD_TIME_PROMISE)}")}),
    }
    fals = run_falsifiers(checks)

    print(f"\n  === seis momentos contra v4 ({ARM}, {len(args.roots)} raíces) ===")
    print(f"  {'momento':<24}{'R1r nuestro':>14}{'d_k':>8}{'R1r ref':>12}"
          f"{'R2r nuestro':>14}{'d_k':>8}{'R2r ref':>12}")
    for m in MOMENT_NAMES:
        r1, r2 = cells["R1r"][ARM], cells["R2r"][ARM]
        print(f"  {m:<24}{r1['moments'][m]:>14.4f}{r1['discrepancies'][m]:>8.2f}"
              f"{reference['R1r'][m].mean:>12.4f}"
              f"{r2['moments'][m]:>14.4f}{r2['discrepancies'][m]:>8.2f}"
              f"{reference['R2r'][m].mean:>12.4f}")
    print("\n  falsadores:")
    for k, v in fals.items():
        if k != "all_passed":
            print(f"    {k:<44} {'PASA' if v['passed'] else 'FALLA'}")

    payload = {
        "schema_version": "fidelity_comparison_v4",
        "claim_status": ("DEVELOPMENT_CONCORDANCE_SCREEN" if fals["all_passed"]
                         else "HALTED_FALSIFIER_FAILED"),
        "what_this_is_not": (
            "NOT a behavioural validation. Configurations, seeds, horizons and scored "
            "windows are unmatched, and the reference sheets are designed configurations, "
            "not replicates. d_k is a DESCRIPTIVE STANDARDIZED DISCREPANCY."),
        "arm": ARM, "roots": list(args.roots), "horizon_hours": horizon,
        "moments_scored": list(MOMENT_NAMES),
        "results": cells if fals["all_passed"] else None,
        "results_withheld_note": (None if fals["all_passed"]
                                  else "computed but NOT reported: a falsifier failed"),
        "reference_shape": shape,
        "window_conventions": window,
        "ctj_diagnostics": ctj,
        "no_verdict_reason": (
            "one arm. non_dominated() over a single cell is a tautology and the contract's "
            "output is a set, not a score, so no dominance verdict is rendered here."),
        "falsifiers": fals,
        "per_episode": per_family,
        "elapsed_seconds": time.perf_counter() - t0,
    }
    digest = seal_and_write(payload, args.output, contract=args.contract,
                            reference=args.reference, stamp_extra={"arm": ARM})
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if fals["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
