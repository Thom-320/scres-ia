#!/usr/bin/env python3
"""Deployable residual against the frozen c256 comparator, on burned roots.

Replaces the contaminated construction in scripts/run_q_r1_d3_residual_bound.py, which
took ``best = max(candidates, key=early_ret_2w)`` over a pool that included placebos and
oracle arms -- a per-episode selection on the realized outcome.  The external audits
measured the damage: placebos won 421/528 selected episodes.

Two rules here, both structural rather than advisory:

1. **Allowlist.** Only deployable arms can enter a deployable bound.  Placebo and oracle
   quantities are computed but are emitted under separate, explicitly labelled keys and
   can never appear in a deployable number.
2. **No selection on the evaluated outcome.** The open-loop calendar is chosen
   leave-one-root-out: the calendar scored on root r is the one that maximises mean
   cohort ReT over every root except r.  The per-campaign maximum over the 4^8 frontier is
   reported only as a clairvoyant CEILING, never as an achievable policy.

Bounds emitted, separated (the "five separated bounds" the plan asks for):

  A  ceiling_minus_frozen      clairvoyant upper bound  -- NOT deployable
  B  frozen_minus_open_loop    value of structured feedback over the best fixed calendar
  C  open_loop_minus_reset     value of a good fixed calendar over the reset comparator
  D  ceiling_minus_open_loop   headroom over open loop -- NOT deployable
  E  frozen_minus_reset        the retained effect already measured by the Pareto

Bound A is the gate quantity: it is the most any controller, learned or otherwise, could
convert on these campaigns.  If A is null, no learner can help here and M1/M2/M4 stop.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_program_q_replication import scheduler  # noqa: E402
from supply_chain.program_o_full_des_transducer import simulate_full_des_frontier  # noqa: E402
from supply_chain.retained_context_discovery import build_campaign_history  # noqa: E402


OBJECTIVE = "early_ret_complete_cohort"
GUARDRAILS = ("early_worst_product_fill", "early_unresolved_orders")
MODE_TO_KAPPA = {"binary_0.9": 0.90, "binary_0.75": 0.75}
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20_260_722


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clustered(deltas_by_root: dict[int, list[float]], *, seed: int = BOOTSTRAP_SEED) -> dict:
    roots = sorted(deltas_by_root)
    values = np.asarray([float(np.mean(deltas_by_root[root])) for root in roots], dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.asarray(
        [float(rng.choice(values, size=len(values), replace=True).mean()) for _ in range(BOOTSTRAP_DRAWS)],
        dtype=float,
    )
    return {
        "mean": float(values.mean()),
        "lcb95_one_sided": float(np.quantile(boot, 0.05)),
        "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "n_roots": len(roots),
        "n_pairs": int(sum(len(v) for v in deltas_by_root.values())),
    }


def run(pareto_path: Path, *, verify_tolerance: float) -> dict:
    payload = json.loads(pareto_path.read_text())
    rows = payload["pareto_pairs"]
    sched = scheduler()
    calendars = np.asarray(list(itertools.product(range(4), repeat=8)), dtype=np.uint8)

    started = time.perf_counter()
    history_cache: dict[tuple[int, float], object] = {}
    records = []
    verify_failures = []

    for row in rows:
        mode = str(row["persistence_mode"])
        kappa = MODE_TO_KAPPA[mode]
        root = int(row["history_root"])
        index = int(row["campaign_index"])
        key = (root, kappa)
        if key not in history_cache:
            history_cache[key] = build_campaign_history(
                history_root=root, campaigns=12, kappa=kappa, scheduler=sched,
                regime_persistence=0.90, dominant_share=0.90,
            )
        campaign = history_cache[key][index]
        if int(campaign.history_root) != root or int(campaign.campaign_index) != index:
            raise SystemExit(f"campaign identity mismatch for root {root} index {index}")

        frontier = simulate_full_des_frontier(
            skeleton=campaign.skeleton, scheduler=sched,
            calendars=calendars, include_q_r1_metrics=True,
        )
        objective = np.asarray(frontier[OBJECTIVE], dtype=float)

        # Self-check: replaying the frozen comparator's own calendar through the frontier
        # must reproduce the value the Pareto recorded for it.
        retained_index = int(
            np.ravel_multi_index(tuple(np.asarray(row["retained_calendar"], dtype=int)), (4,) * 8)
        )
        replayed = float(objective[retained_index])
        recorded = float(row["retained"][OBJECTIVE])
        if abs(replayed - recorded) > verify_tolerance:
            verify_failures.append(
                {"history_root": root, "campaign_index": index, "persistence_mode": mode,
                 "replayed": replayed, "recorded": recorded, "abs_delta": abs(replayed - recorded)}
            )

        records.append({
            "history_root": root,
            "campaign_index": index,
            "persistence_mode": mode,
            "frozen_retained": recorded,
            "frozen_reset": float(row["reset"][OBJECTIVE]),
            "ceiling": float(objective.max()),
            "ceiling_calendar": [int(x) for x in calendars[int(objective.argmax())]],
            "objective_by_calendar": objective,
            "guardrails": {g: np.asarray(frontier[g], dtype=float) for g in GUARDRAILS},
        })

    if verify_failures:
        raise SystemExit(
            f"frontier replay disagrees with the Pareto rows in {len(verify_failures)} campaigns; "
            f"first: {verify_failures[0]}"
        )

    # -- leave-one-root-out open-loop selection (never sees the root it is scored on) ----
    bounds: dict[str, dict] = {}
    per_mode: dict[str, dict] = {}
    for mode in ("binary_0.9", "binary_0.75", "POOLED"):
        subset = [r for r in records if mode == "POOLED" or r["persistence_mode"] == mode]
        by_root: dict[int, list[dict]] = defaultdict(list)
        for record in subset:
            by_root[int(record["history_root"])].append(record)

        d = {name: defaultdict(list) for name in
             ("ceiling_minus_frozen", "frozen_minus_open_loop", "open_loop_minus_reset",
              "ceiling_minus_open_loop", "frozen_minus_reset")}
        open_loop_choices = {}
        for held_out in sorted(by_root):
            training = np.stack(
                [rec["objective_by_calendar"] for root, recs in by_root.items()
                 if root != held_out for rec in recs]
            )
            chosen = int(training.mean(axis=0).argmax())
            open_loop_choices[held_out] = [int(x) for x in calendars[chosen]]
            for rec in by_root[held_out]:
                open_loop = float(rec["objective_by_calendar"][chosen])
                d["ceiling_minus_frozen"][held_out].append(rec["ceiling"] - rec["frozen_retained"])
                d["frozen_minus_open_loop"][held_out].append(rec["frozen_retained"] - open_loop)
                d["open_loop_minus_reset"][held_out].append(open_loop - rec["frozen_reset"])
                d["ceiling_minus_open_loop"][held_out].append(rec["ceiling"] - open_loop)
                d["frozen_minus_reset"][held_out].append(rec["frozen_retained"] - rec["frozen_reset"])

        per_mode[mode] = {
            "bounds": {name: clustered(values) for name, values in d.items()},
            "distinct_open_loop_calendars": sorted(
                {tuple(cal) for cal in open_loop_choices.values()}
            ).__len__(),
            "open_loop_calendar_by_held_out_root": {
                str(root): cal for root, cal in sorted(open_loop_choices.items())
            },
        }
    bounds = per_mode

    return {
        "schema_version": "q_r1_deployable_residual_vs_frozen_c256_v1",
        "claim_status": "BURNED_DEVELOPMENT_NO_CLAIM",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": OBJECTIVE,
        "comparator": payload["pareto"][0]["config_id"],
        "source_pareto": str(pareto_path.relative_to(ROOT)) if pareto_path.is_relative_to(ROOT) else str(pareto_path),
        "source_pareto_sha256": sha256(pareto_path),
        "campaigns": len(records),
        "calendars_enumerated_per_campaign": int(calendars.shape[0]),
        "deployable_allowlist": ["frozen_c256_retained", "frozen_c256_reset", "open_loop_fixed_calendar"],
        "excluded_from_every_deployable_bound": ["placebo_arms", "oracle_arms", "per_episode_realized_selection"],
        "clairvoyant_keys_not_deployable": ["ceiling", "ceiling_minus_frozen", "ceiling_minus_open_loop"],
        "open_loop_selection": "leave_one_root_out_mean_objective",
        "frontier_replay_verification": {
            "checked": len(records),
            "failures": len(verify_failures),
            "tolerance": verify_tolerance,
        },
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "clustering": "history_root",
        "strata": bounds,
        "elapsed_seconds": time.perf_counter() - started,
        "selection_performed": False,
        "learner_return_used": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pareto", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-tolerance", type=float, default=1e-9)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    result = run(args.pareto.resolve(), verify_tolerance=args.verify_tolerance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for mode, payload in result["strata"].items():
        print(f"=== {mode} ===")
        for name, stats in payload["bounds"].items():
            print(f"  {name:26} mean {stats['mean']:+.6f}  LCB95 {stats['lcb95_one_sided']:+.6f}  "
                  f"CI95 [{stats['ci95'][0]:+.6f}, {stats['ci95'][1]:+.6f}]  n={stats['n_roots']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
