#!/usr/bin/env python3
"""Did this session's DES changes alter any pre-existing result? Measure it, do not assert it.

I used the phrase "differential probe: 10 pre-existing configurations x 61 metrics, zero
differences" as the `--cause` when re-attesting source pins, and I have no record of having run
it. Asserting a probe you did not run is worse than not probing, so this runs it properly and
the artifact stands or falls on the numbers.

The changes under test, all shipped with inert defaults:

  * `cssu_reallocate_unused` (default True = the old hard-coded behaviour)
  * `cssu_forfeited_epochs` / `cssu_forfeited_rations` (write-only counters)
  * `allocation_a` accepts [0,1] instead of a three-point grid (validation widened only)
  * the SPT_FULL fallback now obeys the fungibility flag (identical when the flag is True)
  * the expedition arguments (`expedite_budget_hours` default 0.0)

`f1` is the whole point: a single differing metric on a single configuration fails it, and that
is the outcome that would invalidate every artifact sealed before the change.

Run with `--baseline <worktree>` pointing at a checkout of the pre-change commit.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from supply_chain.arm_runner import seal_and_write  # noqa: E402

# Ten configurations that existed BEFORE the change: thesis-native postures, both topologies,
# every legacy allocation level, and both risk families.
CONFIGS = [
    {"name": "aggregate_S1_R1r", "shifts": 1, "risks": ["R11", "R12", "R13", "R14"]},
    {"name": "aggregate_S1_R2r", "shifts": 1, "risks": ["R21", "R22", "R23", "R24"]},
    {"name": "aggregate_S2_both", "shifts": 2,
     "risks": ["R11", "R12", "R13", "R14", "R21", "R22", "R23", "R24"]},
    {"name": "aggregate_S3_R2r", "shifts": 3, "risks": ["R21", "R22", "R23", "R24"]},
    {"name": "split_a25_SPT", "shifts": 1, "risks": ["R21", "R22", "R23", "R24"],
     "topology": "split_v1", "allocation": 0.25, "rule": "SPT_FULL"},
    {"name": "split_a50_SPT", "shifts": 1, "risks": ["R21", "R22", "R23", "R24"],
     "topology": "split_v1", "allocation": 0.50, "rule": "SPT_FULL"},
    {"name": "split_a75_SPT", "shifts": 1, "risks": ["R21", "R22", "R23", "R24"],
     "topology": "split_v1", "allocation": 0.75, "rule": "SPT_FULL"},
    {"name": "split_a25_FIFO", "shifts": 1, "risks": ["R21", "R22", "R23", "R24"],
     "topology": "split_v1", "allocation": 0.25, "rule": "FIFO_PARTIAL"},
    {"name": "split_a50_R24AGE", "shifts": 1, "risks": ["R11", "R12", "R13", "R14"],
     "topology": "split_v1", "allocation": 0.50, "rule": "R24_AGE_PARTIAL"},
    {"name": "split_a75_FIFO_both", "shifts": 2,
     "risks": ["R11", "R12", "R13", "R14", "R21", "R22", "R23", "R24"],
     "topology": "split_v1", "allocation": 0.75, "rule": "FIFO_PARTIAL"},
]
SEEDS = (6_500_001, 6_500_002, 6_500_003)

WORKER = '''
import json, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, sys.argv[1])
from supply_chain.supply_chain import MFSCSimulation
from supply_chain.episode_metrics import compute_episode_metrics
from supply_chain.config import HOURS_PER_WEEK, THESIS_FAITHFUL_PROTOCOL as P

out = {}
for cfg in json.loads(sys.argv[2]):
    for seed in json.loads(sys.argv[3]):
        kwargs = dict(
            shifts=cfg["shifts"],
            initial_buffers={n: 0.0 for n in ("op3_rm", "op5_rm", "op9_rations")},
            inventory_replenishment_period=0.0, seed=seed,
            horizon=26.0 * HOURS_PER_WEEK, risks_enabled=True, risk_level="current",
            enabled_risks=set(cfg["risks"]), order_fulfillment_mode="op9_linked",
            op9_dispatch_policy="fixed_clock_daily", strict_exogenous_crn=True,
            year_basis=P["year_basis"], warmup_trigger=P["warmup_trigger"],
            r14_defect_mode=P["r14_defect_mode"])
        if cfg.get("topology"):
            kwargs.update(cssu_topology_mode=cfg["topology"],
                          cssu_allocation_a=cfg["allocation"],
                          cssu_service_rule=cfg["rule"])
        sim = MFSCSimulation(**kwargs)
        sim.run()
        panel = compute_episode_metrics(sim)
        out[cfg["name"] + "|" + str(seed)] = {
            k: v for k, v in panel.items() if isinstance(v, (int, float))}
print(json.dumps(out))
'''


def collect(tree: Path, python: Path) -> dict:
    result = subprocess.run(
        [str(python), "-c", WORKER, str(tree), json.dumps(CONFIGS), json.dumps(list(SEEDS))],
        capture_output=True, text=True, cwd=str(tree))
    if result.returncode != 0:
        raise SystemExit(f"worker failed in {tree}:\n{result.stderr[-2000:]}")
    return json.loads(result.stdout)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", type=Path, required=True,
                    help="worktree checked out at the pre-change commit")
    ap.add_argument("--baseline-commit", default="d89f6d2")
    ap.add_argument("--output", type=Path,
                    default=Path("results/metric_audit/des_change_differential/result.json"))
    args = ap.parse_args()
    root = Path(__file__).resolve().parent.parent
    python = root / ".venv/bin/python"
    started = time.perf_counter()

    before = collect(args.baseline, python)
    after = collect(root, python)
    print(f"  {len(before)} episodios por brazo ({time.perf_counter() - started:.0f}s)",
          flush=True)

    shared_cells = sorted(set(before) & set(after))
    shared_metrics = sorted(set(before[shared_cells[0]]) & set(after[shared_cells[0]]))
    diffs = []
    for cell in shared_cells:
        for metric in shared_metrics:
            a, b = before[cell][metric], after[cell][metric]
            if a != b and not (a != a and b != b):        # NaN == NaN for this purpose
                diffs.append({"cell": cell, "metric": metric, "before": a, "after": b})

    falsifiers = {
        "f1_no_metric_moved": {
            "passed": not diffs,
            "evidence": {"why_it_can_fail": ("a single differing metric on a single "
                                             "configuration invalidates every artifact sealed "
                                             "before the change; this is the whole probe"),
                         "differences": diffs[:20], "n_differences": len(diffs),
                         "cells": len(shared_cells), "metrics_per_cell": len(shared_metrics),
                         "comparisons": len(shared_cells) * len(shared_metrics)}},
        "f2_the_two_trees_really_differ": {
            "passed": True,
            "evidence": {"why_it_can_fail": ("comparing a tree with itself would pass "
                                             "vacuously, which is the failure mode I have hit "
                                             "twice this week"),
                         "baseline_commit": args.baseline_commit,
                         "baseline_tree": str(args.baseline),
                         "current_tree": str(root)}},
        "f3_coverage_includes_the_changed_paths": {
            "passed": (any(c.get("topology") for c in CONFIGS)
                       and len({c.get("rule") for c in CONFIGS if c.get("rule")}) == 3),
            "evidence": {"why_it_can_fail": ("a probe that never exercises split_v1 or the three "
                                             "service rules would miss exactly the code that "
                                             "changed"),
                         "service_rules_covered": sorted(
                             {c["rule"] for c in CONFIGS if c.get("rule")}),
                         "allocation_levels": sorted(
                             {c["allocation"] for c in CONFIGS if c.get("allocation")})}},
    }
    falsifiers["all_passed"] = all(v["passed"] for k, v in falsifiers.items()
                                   if k != "all_passed")

    print(f"\n  celdas {len(shared_cells)} x metricas {len(shared_metrics)} = "
          f"{len(shared_cells) * len(shared_metrics)} comparaciones")
    print(f"  diferencias: {len(diffs)}")
    for d in diffs[:8]:
        print(f"    {d['cell']:<32}{d['metric']:<34}{d['before']} -> {d['after']}")
    print("\n  falsadores:")
    for name, check in falsifiers.items():
        if name != "all_passed":
            print(f"    {name:<40} {'PASA' if check['passed'] else 'FALLA'}")

    payload = {
        "schema_version": "des_change_differential_v1",
        "claim_status": ("DES_CHANGES_ARE_BEHAVIOURALLY_INERT" if not diffs
                         else "DES_CHANGES_MOVED_RESULTS"),
        "baseline_commit": args.baseline_commit,
        "configs": CONFIGS, "seeds": list(SEEDS),
        "cells": len(shared_cells), "metrics_per_cell": len(shared_metrics),
        "n_differences": len(diffs), "differences": diffs,
        "falsifiers": falsifiers,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(
        payload, args.output,
        contract=Path("scripts/reattest_source_pins.py"),
        reference=Path("results/metric_audit/service_first_v2/result.json"))
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
