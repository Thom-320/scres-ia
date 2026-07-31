#!/usr/bin/env python3
"""Does the canonical v2 metric still compute what it computed when it was frozen?

`ret_excel_request_snapshot_v2` was frozen on 2026-07-14 (`ff5e4a8`), and its governance audit
pinned the sha256 of every implementation source. Four of those files have since changed, so
five separate attestations across the repository fail. The question the failures pose is NOT
"do the hashes match" -- they plainly do not -- but **whether the metric still returns the same
numbers**. Re-attesting without answering that would launder a silent change to the primary
endpoint; answering it first makes re-attestation a recorded decision rather than a shrug.

The substantive change is `4111cbc` (2026-07-17, Program Q) inside
`compute_order_level_ret_excel_request_snapshot_ledger`: the causal boundary for counting
`Bt`/`Ut` moved from the row key `(j, OPTj)` to the **request time**, and two options appeared
(`same_time_precedence`, `force_reconstruct`). That is exactly the kind of edit that can be
either an equivalent restatement or a redefinition of the endpoint, and only measurement tells
them apart.

Method: load the FROZEN `ret_thesis.py` out of `ff5e4a8` beside the current one, run both over
the SAME order objects from the same simulations, and compare per-order `ReT` exactly. One
population, two implementations, no re-simulation in between -- so any difference is the metric
and nothing else.

Two falsifiers, each able to fail:

* `f1_frozen_module_is_the_frozen_bytes` -- the loaded module's sha256 must equal the hash the
  governance audit pinned. If the extraction silently loaded HEAD's file instead, every
  comparison below would be a tautology, which is how this class of check usually dies.
* `f2_the_changed_branch_was_exercised` -- **this one already caught me.** The first run
  reported 3,289 rows and zero differences with `orders_needing_reconstruction: 0`: every order
  carried captured `Bt`/`Ut`, so BOTH implementations short-circuited to `captured_at_request`
  and the branch that changed was never entered. The equivalence was real and worthless. The
  comparison now runs in two strata -- as shipped, and with the captured fields cleared so the
  frozen code's own condition sends both down the reconstruction path -- and the falsifier
  fails unless the second stratum has rows.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.arm_runner import scored_orders, seal_and_write  # noqa: E402
from supply_chain.config import HOURS_PER_WEEK  # noqa: E402
from supply_chain.config import THESIS_FAITHFUL_PROTOCOL as P  # noqa: E402
from supply_chain.ret_thesis import (  # noqa: E402
    compute_order_level_ret_excel_request_snapshot_ledger as current_ledger,
)
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402

FROZEN_COMMIT = "ff5e4a8"
FROZEN_PATH = "supply_chain/ret_thesis.py"
GOVERNANCE = Path(
    "research/paper2_exhaustive_search/"
    "ret_excel_request_snapshot_v2_implementation_audit_20260714.json"
)
FAMILIES = {"R1r": ("R11", "R12", "R13", "R14"), "R2r": ("R21", "R22", "R23", "R24")}
ROOTS = tuple(3_600_001 + i for i in range(8))


def load_frozen(tmpdir: Path) -> tuple[object, str]:
    """Import the frozen file as a member of the CURRENT package, so relative imports work."""
    blob = subprocess.run(["git", "show", f"{FROZEN_COMMIT}:{FROZEN_PATH}"],
                          check=True, capture_output=True).stdout
    path = tmpdir / "frozen_ret_thesis.py"
    path.write_bytes(blob)
    spec = importlib.util.spec_from_file_location("supply_chain._frozen_ret_thesis", path)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "supply_chain"
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, sha256(blob).hexdigest()


def run_episode(family: str, seed: int, horizon: float) -> MFSCSimulation:
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roots", nargs="+", type=int, default=list(ROOTS))
    ap.add_argument("--horizon-weeks", type=int, default=52)
    ap.add_argument("--output", type=Path, default=Path(
        "results/metric_audit/v2_metric_freeze_equivalence/result.json"))
    args = ap.parse_args()
    horizon = float(args.horizon_weeks * HOURS_PER_WEEK)

    with tempfile.TemporaryDirectory() as raw:
        frozen, frozen_sha = load_frozen(Path(raw))
        frozen_ledger = frozen.compute_order_level_ret_excel_request_snapshot_ledger

        pinned = json.loads(GOVERNANCE.read_text())["implementation_sources"][FROZEN_PATH]
        # Two strata: `as_shipped` keeps the captured Bt/Ut (both implementations
        # short-circuit); `reconstructed` clears them so the frozen code's OWN condition sends
        # both down the branch that changed. Only the second tests the edit.
        strata = {name: {"rows": [], "n": 0, "n_differing": 0, "max_abs_diff": 0.0}
                  for name in ("as_shipped", "reconstructed")}
        distinct_ret: set[float] = set()

        def compare(orders, now: float, stratum: str, family: str, seed: int) -> None:
            a = current_ledger(orders, current_time=now)["ret_values"]
            b = frozen_ledger(orders, current_time=now)["ret_values"]
            s = strata[stratum]
            if len(a) != len(b):
                s["rows"].append({"family": family, "seed": seed,
                                  "error": f"row count {len(a)} vs {len(b)}"})
                s["n_differing"] += max(len(a), len(b))
                return
            diffs = [abs(float(x) - float(y)) for x, y in zip(a, b)]
            s["n"] += len(diffs)
            s["n_differing"] += sum(1 for d in diffs if d > 0.0)
            s["max_abs_diff"] = max([s["max_abs_diff"]] + diffs)
            distinct_ret.update(round(float(x), 9) for x in a)
            s["rows"].append({"family": family, "seed": seed, "n": len(diffs),
                              "n_differing": sum(1 for d in diffs if d > 0.0),
                              "max_abs_diff": max(diffs) if diffs else 0.0})

        for family in FAMILIES:
            for seed in args.roots:
                sim = run_episode(family, seed, horizon)
                orders = scored_orders(sim)
                now = float(sim.env.now)
                compare(orders, now, "as_shipped", family, seed)
                for order in orders:  # the captured path is done with; force the other one
                    order.ret_bt_at_request = None
                    order.ret_ut_at_request = None
                compare(orders, now, "reconstructed", family, seed)

        rows = {name: s.pop("rows") for name, s in strata.items()}
        n_rows = sum(s["n"] for s in strata.values())
        n_diff = sum(s["n_differing"] for s in strata.values())
        worst = max(s["max_abs_diff"] for s in strata.values())

    falsifiers = {
        "f1_frozen_module_is_the_frozen_bytes": {
            "passed": frozen_sha == pinned,
            "evidence": {
                "why_it_can_fail": ("if the extraction loaded HEAD's file, or the wrong commit, "
                                    "every comparison below would be a tautology"),
                "loaded_sha256": frozen_sha, "governance_pinned_sha256": pinned,
                "commit": FROZEN_COMMIT},
        },
        "f2_the_changed_branch_was_exercised": {
            "passed": bool(strata["reconstructed"]["n"] > 1000 and len(distinct_ret) > 20),
            "evidence": {
                "why_it_can_fail": ("the FIRST run of this script passed a weaker version of "
                                    "this check with 3,289 rows, zero differences and ZERO "
                                    "orders on the reconstruction path: both implementations "
                                    "short-circuited to captured_at_request and the edited "
                                    "branch was never entered. An equivalence measured off "
                                    "the changed code proves nothing"),
                "rows_by_stratum": {k: v["n"] for k, v in strata.items()},
                "distinct_ret_values": len(distinct_ret)},
        },
    }
    falsifiers["all_passed"] = all(v["passed"] for k, v in falsifiers.items()
                                   if k != "all_passed")
    equivalent = bool(n_diff == 0 and falsifiers["all_passed"])

    for name, s in strata.items():
        print(f"  {name:<16} filas {s['n']:>6}  difieren {s['n_differing']:>6}  "
              f"max|Δ| {s['max_abs_diff']}")
    for name, check in falsifiers.items():
        if name != "all_passed":
            print(f"  {name:<40} {'PASA' if check['passed'] else 'FALLA'}")

    payload = {
        "schema_version": "v2_metric_freeze_equivalence_v1",
        "claim_status": ("V2_METRIC_UNCHANGED_SINCE_FREEZE" if equivalent
                         else "V2_METRIC_CHANGED_SINCE_FREEZE"),
        "question": ("do the post-freeze edits to ret_thesis.py change what the canonical v2 "
                     "metric returns, on one population scored by both implementations?"),
        "frozen_commit": FROZEN_COMMIT, "frozen_path": FROZEN_PATH,
        "roots": list(args.roots), "horizon_hours": horizon,
        "rows_compared": n_rows, "rows_differing": n_diff, "max_abs_diff": worst,
        "by_stratum": strata, "per_cell": rows, "falsifiers": falsifiers,
        "scope": ("shipped defaults, both risk families, 8 roots each. It does NOT cover the "
                  "non-default `same_time_precedence='snapshot_before_events'` or "
                  "`force_reconstruct=True` paths, which are new and have no frozen "
                  "counterpart to compare against."),
    }
    digest = seal_and_write(payload, args.output, contract=GOVERNANCE, reference=GOVERNANCE)
    print(f"\n  {payload['claim_status']}\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
