#!/usr/bin/env python3
"""VERIFIER (V5): counterbalanced, symmetric-relabel closure of the DRA-1 branching.

Codex's rebuttal to V4 is correct: my neutral-prefix check did not control for the
mild environmental B-attractor, so it could not cleanly separate
  (i) a state-INDEPENDENT constant optimum, from
  (ii) a state-CONTINGENT "serve/write-off the stressed node" rule that a constant
       allocation_a cannot replicate.

This test does the definitive thing:
- generate branch states from THREE prefixes (0.25 -> A-stressed, 0.75 -> B-stressed,
  0.50 -> mixed) so both congestion directions appear (counterbalanced);
- relabel actions symmetrically as share_to_stressed = allocation_a if A is the more
  backlogged node, else 1 - allocation_a;
- compare the BEST fixed share_to_stressed rule (a DYNAMIC allocation_a that flips with
  which node is stressed) against the BEST fixed constant allocation_a, on the same
  pooled states, with bootstrap grouped by tape.

If the stressed-frame rule beats the best constant (CI95>0, material), DRA-1 has
adaptive value and must NOT stop. If not, the STOP is finally confound-free.
"""
from __future__ import annotations

import json
from pathlib import Path
import statistics
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_dra1_exact_branching import branch, select_state  # noqa: E402

FRONTIER = Path("results/program_d/dra1_static_frontier")
SHARES = (0.25, 0.50, 0.75)
RULE = "SPT_FULL"


def collect(tapes: list, prefixes: list[float]) -> list[dict]:
    """One row per (tape, prefix-state) with long_ret for each allocation_a and the
    stressed-node label."""
    rows = []
    for pfx in prefixes:
        base = (pfx, RULE)
        for tape in tapes:
            state = select_state(tape, base)
            a_bk = float(state.get("cssu_A_backlog_count", 0.0))
            b_bk = float(state.get("cssu_B_backlog_count", 0.0))
            if abs(a_bk - b_bk) < 1e-9:
                continue  # skip ties: stressed node undefined
            stressed = "A" if a_bk > b_bk else "B"
            ret = {}
            for a in SHARES:
                res = branch(tape, state, base, (a, RULE))
                ret[a] = float(res["long_ret"])
            rows.append({"tape": tape["tape_id"], "prefix": pfx, "stressed": stressed, "ret": ret})
    return rows


def grouped_bootstrap(per_tape: dict[str, list[float]], seed: int, n: int = 10000) -> dict:
    tapes = list(per_tape)
    means = np.array([statistics.mean(per_tape[t]) for t in tapes])
    rng = np.random.default_rng(seed)
    boot = np.array([means[rng.integers(0, len(means), len(means))].mean() for _ in range(n)])
    return {"mean": float(means.mean()), "ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]}


def main() -> int:
    tapes = json.loads((FRONTIER / "calibration_tapes.json").read_text())
    rows = collect(tapes, [0.25, 0.50, 0.75])
    n_A = sum(r["stressed"] == "A" for r in rows)
    n_B = sum(r["stressed"] == "B" for r in rows)
    print(f"[v5] pooled states: {len(rows)} (A-stressed={n_A}, B-stressed={n_B})", flush=True)

    # Best CONSTANT allocation_a (same action regardless of stress).
    const_mean = {a: statistics.mean(r["ret"][a] for r in rows) for a in SHARES}
    best_const_a = max(const_mean, key=const_mean.get)

    # Best fixed SHARE_TO_STRESSED rule: allocation_a = s if A-stressed else 1-s.
    def share_ret(r, s):
        a = s if r["stressed"] == "A" else round(1 - s, 2)
        return r["ret"][a]
    share_mean = {s: statistics.mean(share_ret(r, s) for r in rows) for s in SHARES}
    best_share = max(share_mean, key=share_mean.get)

    # Delta: best stressed-frame rule vs best constant, grouped bootstrap by tape.
    per_tape_delta: dict[str, list[float]] = {}
    for r in rows:
        d = share_ret(r, best_share) - r["ret"][best_const_a]
        per_tape_delta.setdefault(r["tape"], []).append(d)
    delta = grouped_bootstrap(per_tape_delta, 0xD5A)

    # Is the stressed-frame optimum itself state-contingent, or a fixed share?
    per_state_best_share = []
    for r in rows:
        per_state_best_share.append(max(SHARES, key=lambda s: share_ret(r, s)))
    from collections import Counter
    share_opt_counts = dict(Counter(per_state_best_share))

    out = {
        "kind": "dra1_stressed_frame_v5",
        "pooled_states": len(rows), "A_stressed": n_A, "B_stressed": n_B,
        "constant_allocation_mean_ret": const_mean, "best_constant_allocation_a": best_const_a,
        "share_to_stressed_mean_ret": share_mean, "best_share_to_stressed": best_share,
        "best_share_minus_best_constant": delta,
        "per_state_optimal_share_counts": share_opt_counts,
        "verdict": (
            "ADAPTIVE_VALUE_FOUND_REOPEN" if delta["ci95"][0] > 1e-4
            else "STOP_CONFIRMED_NO_STRESSED_FRAME_HEADROOM"
        ),
    }
    Path("results/program_d/dra1_stressed_frame_v5.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
