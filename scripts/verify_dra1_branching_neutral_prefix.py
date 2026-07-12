#!/usr/bin/env python3
"""VERIFIER re-test: DRA-1 exact branching from a NEUTRAL prefix (V3 confound check).

Codex's branching (`run_dra1_exact_branching.py`) samples branch states from the
best-admissible constant prefix (allocation_a=0.25), which systematically congests
CSSU-A (V3 finding), making 0.25 self-confirming and suppressing action diversity.
This independent re-test samples states from a NEUTRAL 0.50 prefix (and 0.75) and
asks: once the prefix bias is removed, does the optimal allocation
(a) become symmetric / state-contingent (favouring whichever node is recoverable),
(b) expose non-negligible oracle ReT headroom?
If the optimum stays state-independent with negligible headroom under a neutral
prefix, the DRA-1 STOP is confound-free.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_dra1_exact_branching import (  # noqa: E402
    ALLOCATION_LEVELS,
    SERVICE_RULES,
    branch,
    select_state,
)

FRONTIER = Path("results/program_d/dra1_static_frontier")


def run_for_prefix(tapes: list, base: tuple[float, str]) -> dict:
    states = [select_state(tape, base) for tape in tapes]
    rows = []
    for tape, state in zip(tapes, states):
        for allocation in ALLOCATION_LEVELS:
            for rule in SERVICE_RULES:
                res = branch(tape, state, base, (allocation, rule))
                rows.append({"tape_id": tape["tape_id"], "category": state["category"],
                             "allocation_a": allocation, "service_rule": rule,
                             "a_backlog": float(state.get("cssu_A_backlog_count", 0.0)),
                             "b_backlog": float(state.get("cssu_B_backlog_count", 0.0)), **res})
    # oracle best per state
    by_state: dict[str, list] = {}
    for r in rows:
        by_state.setdefault(r["tape_id"], []).append(r)
    oracle = []
    for sid, rs in by_state.items():
        baseline = next(r for r in rs if float(r["allocation_a"]) == base[0] and r["service_rule"] == base[1])
        best = max(rs, key=lambda r: (float(r["long_ret"]), float(r["long_clipped"])))
        # which node is more backlogged in this state, and does optimal serve the OTHER (write-off)?
        a_bk, b_bk = best["a_backlog"], best["b_backlog"]
        stressed = "A" if a_bk > b_bk else ("B" if b_bk > a_bk else "tie")
        # allocation_a>0.5 favours serving A; <0.5 favours B
        served_more = "A" if float(best["allocation_a"]) > 0.5 else ("B" if float(best["allocation_a"]) < 0.5 else "even")
        oracle.append({"allocation_a": float(best["allocation_a"]), "stressed": stressed,
                       "served_more": served_more,
                       "delta_ret": float(best["long_ret"]) - float(baseline["long_ret"])})
    return {"n": len(oracle),
            "allocation_counts": dict(Counter(o["allocation_a"] for o in oracle)),
            "stressed_counts": dict(Counter(o["stressed"] for o in oracle)),
            # key: when A is stressed does optimum serve B, and vice-versa? (write-off pattern)
            "served_vs_stressed": {f"{s}->{sv}": n for (s,sv),n in Counter((o["stressed"], o["served_more"]) for o in oracle).items()},
            "mean_delta_ret": sum(o["delta_ret"] for o in oracle) / max(len(oracle), 1),
            "max_delta_ret": max(o["delta_ret"] for o in oracle)}


def main() -> int:
    tapes = json.loads((FRONTIER / "calibration_tapes.json").read_text())
    out = {}
    for label, base in [("prefix_0.50", (0.50, "SPT_FULL")),
                        ("prefix_0.75", (0.75, "SPT_FULL"))]:
        print(f"[verify] running {label} ...", flush=True)
        out[label] = run_for_prefix(tapes, base)
        print(json.dumps({label: out[label]}, indent=2), flush=True)
    Path("results/program_d/dra1_branching_neutral_prefix_verify.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    # verdict
    print("\n=== VERIFIER READING ===")
    for label, r in out.items():
        alloc = r["allocation_counts"]
        diverse = sum(1 for v in alloc.values() if v / r["n"] >= 0.15) >= 2 and max(alloc.values()) / r["n"] <= 0.85
        print(f"{label}: alloc={alloc} | diversity={'PASS' if diverse else 'FAIL'} | "
              f"mean_delta_ret={r['mean_delta_ret']:.2e} | stressed={r['stressed_counts']}")
        print(f"   served_vs_stressed={r['served_vs_stressed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
