#!/usr/bin/env python3
"""Align the Codex matched-retention lineage to the out-of-sample bar and cross-check it.

Two independent implementations of the same v2 matched-retention contract exist:

  A. this lineage   -- branch codex/q-r1-comparator-reconciliation,
                       supply_chain/oracle_curve_v2.py + scripts/run_oracle_curve_v2.py
  B. Codex lineage  -- branch codex/q-r1-oracle-v2,
                       scripts/run_q_r1_matched_retention_curve_v2.py

They are not redundant: A trains a separate policy per memory arm (isolating what a learner
can LEARN under matched rights), while B evaluates one checkpoint under both memory settings
(isolating the evaluation-time memory effect with no training confound). Both are valid; the
verdicts must agree on any campaign they share.

This script does three things, none of which modifies either lineage:

  1. Bar alignment. Codex's capture figures are recomputed against the frozen out-of-sample
     deployable bar (results/oracle_capture_v1/static_bar_out_of_sample.json), so both
     lineages are scored on the same reference instead of on the withdrawn in-sample bar.
  2. Cross-implementation consistency. For every Codex row whose campaign has an enumerated
     frontier, Codex's simulated objective is compared with this lineage's exact table lookup
     of the SAME calendar. Disagreement above 1e-9 means one of the two implementations is
     wrong about the physics, and that must be resolved before either verdict is read.
  3. Verdict comparison, to the extent the evidence allows. Codex's artifact is a smoke run,
     so its numbers are reported as an instrument check and the verdict comparison is left
     explicitly pending its full execution.

Usage:
    .venv/bin/python scripts/crosscheck_codex_matched_retention.py
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from supply_chain.oracle_capture import (  # noqa: E402
    BOOT_SEED, calendar_index, load_campaigns, pooled_capture,
)

CODEX = Path("/private/tmp/scres-q-r1-oracle-v2")
FRONTIERS = ROOT / "results/fig5_surrogate_v1/frontiers"
BAR_ARTIFACT = ROOT / "results/oracle_capture_v1/static_bar_out_of_sample.json"
OUT = ROOT / "results/oracle_curve_v2/codex_crosscheck.json"
TOLERANCE = 1e-9


def main() -> int:
    bar_spec = json.loads(BAR_ARTIFACT.read_text())
    bar_row = int(bar_spec["frontier_row"])
    campaigns = load_campaigns(FRONTIERS)
    by_key = {c.key: c for c in campaigns}
    bar = {c.key: float(c.labels[bar_row]) for c in campaigns}

    smoke = json.loads(
        (CODEX / "results/q_r1/matched_retention_curve_v2/smoke/result.json").read_text())
    rows = json.loads(
        (CODEX / "results/q_r1/matched_retention_curve_v2/smoke/rows.json").read_text())
    if isinstance(rows, dict):
        rows = rows.get("rows", [])

    # ---- 2. cross-implementation consistency on shared campaigns -----------
    checks, mismatches = [], []
    per_arm: dict[str, dict[tuple, list[int]]] = {}
    for row in rows:
        mode = "binary_0.75" if float(row["kappa"]) < 0.85 else "binary_0.9"
        key = (int(row["history_root"]), int(row["campaign_index"]), mode)
        per_arm.setdefault(str(row["arm"]), {})[key] = list(row["calendar"])
        if key not in by_key:
            continue  # campaign outside the enumerated set
        lookup = float(by_key[key].labels[calendar_index(row["calendar"])])
        codex_value = float(row["early_ret_complete_cohort"])
        delta = abs(lookup - codex_value)
        checks.append({"key": [key[0], key[1], key[2]], "arm": row["arm"],
                       "codex": codex_value, "lookup": lookup, "abs_delta": delta})
        if delta > TOLERANCE:
            mismatches.append(checks[-1])

    # ---- root-cause probe: are the two lineages even building the same campaign? -----
    from scripts.c6_perbatch_ceiling import _count_scheduler  # noqa: PLC0415
    from supply_chain.retained_context_discovery import (  # noqa: PLC0415
        build_campaign_history,
    )
    sched = _count_scheduler()
    probe_row = next((r for r in rows if (int(r["history_root"]), int(r["campaign_index"]),
                                          "binary_0.9" if float(r["kappa"]) >= 0.85
                                          else "binary_0.75") in by_key), None)
    construction = {"probe": None}
    if probe_row is not None:
        root, idx = int(probe_row["history_root"]), int(probe_row["campaign_index"])
        trials = {}
        for rho in (0.90, 0.75):
            history = build_campaign_history(
                history_root=root, campaigns=12, kappa=float(probe_row["kappa"]),
                scheduler=sched, regime_persistence=rho, dominant_share=0.90)
            trials[rho] = history[idx].skeleton.skeleton_sha256
        codex_sha = str(probe_row["skeleton_sha256"])
        matched_rho = next((rho for rho, sha in trials.items() if sha == codex_sha), None)
        construction = {
            "probe": {"history_root": root, "campaign_index": idx,
                      "kappa": float(probe_row["kappa"])},
            "codex_skeleton_sha256": codex_sha,
            "sha_under_rho_0_90": trials[0.90],
            "sha_under_rho_0_75": trials[0.75],
            "codex_regime_persistence_implied": matched_rho,
            "enumerated_frontier_regime_persistence": 0.90,
            "same_campaign": matched_rho == 0.90,
            "diagnosis": (
                "IDENTICAL CONSTRUCTION" if matched_rho == 0.90 else
                "DEFECT IN THE CODEX LINEAGE: its campaigns are built with within-campaign "
                "regime persistence rho=0.75, while every enumerated frontier, the frozen "
                "Pareto and the C6 gate use rho=0.90. rho (regime persistence INSIDE a "
                "campaign) has been conflated with kappa (knowledge persistence BETWEEN "
                "campaigns, whose strata are 0.75 and 0.90). Its campaigns are therefore "
                "different physical objects: they cannot be graded against our exact "
                "ceilings, and they are not comparable to the frozen comparator either."),
        }

    # ---- 1. bar alignment: Codex arms rescored on the frozen deployable bar -
    aligned = {}
    for arm, calendars in sorted(per_arm.items()):
        shared = {k: v for k, v in calendars.items() if k in by_key}
        if not shared:
            aligned[arm] = {"n_graded": 0,
                            "note": "no Codex campaign overlaps the enumerated frontier set"}
            continue
        present = [by_key[k] for k in shared]
        pooled = pooled_capture(present, shared, bar, np.random.default_rng(BOOT_SEED))
        aligned[arm] = {
            "VALID": construction.get("same_campaign", False),
            "invalid_reason": (None if construction.get("same_campaign")
                              else "graded against frontiers of DIFFERENT campaigns; these "
                                   "numbers are not results"),
            **pooled,
            "n_graded": len(shared),
            "distinct_calendars": len({tuple(v) for v in shared.values()}),
            "mean_label": float(np.mean([by_key[k].labels[calendar_index(v)]
                                         for k, v in shared.items()])),
            "exact_optimum_hits": int(sum(
                1 for k, v in shared.items()
                if by_key[k].ceiling - float(by_key[k].labels[calendar_index(v)])
                <= TOLERANCE)),
        }

    consistent = not mismatches and bool(checks)
    payload = {
        "schema": "codex_matched_retention_crosscheck_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "claim_status": "INSTRUMENT_CROSSCHECK_NO_OUTCOME_CLAIM",
        "lineages": {
            "this": {"branch": "codex/q-r1-comparator-reconciliation",
                     "design": "one trained policy PER memory arm; isolates what a learner "
                               "can learn under matched rights"},
            "codex": {"branch": "codex/q-r1-oracle-v2",
                      "design": "one checkpoint evaluated under BOTH memory settings; "
                                "isolates the evaluation-time memory effect",
                      "artifact_mode": smoke.get("mode"),
                      "claim_status": smoke.get("claim_status"),
                      "timesteps": smoke.get("timesteps"),
                      "same_checkpoint_both_memory_arms": smoke.get(
                          "same_checkpoint_both_memory_arms")},
        },
        "bar_alignment": {
            "bar": bar_spec["calendar"], "frontier_row": bar_row,
            "selected_on": bar_spec["calibration_block"],
            "replaces": "the withdrawn in-sample bar [0,0,3,3,3,3,3,3]",
            "codex_arms_rescored": aligned,
        },
        "campaign_construction": construction,
        "cross_implementation_consistency": {
            "tolerance": TOLERANCE,
            "n_shared_campaign_rows": len(checks),
            "n_mismatches": len(mismatches),
            "max_abs_delta": max((c["abs_delta"] for c in checks), default=None),
            "consistent": consistent,
            "mismatches": mismatches[:10],
            "interpretation": (
                "Codex's simulated objective and this lineage's exact frontier lookup agree "
                "on every shared campaign, so the two implementations describe the same "
                "physics and their verdicts are comparable"
                if consistent else
                "the values disagree because the campaigns are not the same object -- see "
                "campaign_construction; the rescored Codex arms below are therefore INVALID "
                "and are retained only to document the attempt"),
        },
        "verdict_comparison": {
            "status": ("PENDING_CODEX_FULL_RUN" if consistent
                       else "BLOCKED_CODEX_CAMPAIGN_CONSTRUCTION_DEFECT"),
            "reason": "the Codex artifact is an instrument smoke (192 timesteps, one root, "
                      "claim_status BURNED_INSTRUMENT_SMOKE_NO_CLAIM), AND, more seriously, "
                      "its campaigns are built with rho=0.75 instead of the programme-wide "
                      "rho=0.90, so no amount of further running would make its numbers "
                      "comparable until that constant is corrected",
            "what_is_comparable_now": "the physics agreement above, and Codex's smoke arms "
                                      "rescored on the aligned bar",
            "to_complete": "1) correct REGIME_PERSISTENCE to 0.90 in the Codex runner and "
                           "re-freeze its contract; 2) re-run its full mode; 3) then compare "
                           "sign and magnitude of (retained - reset), the retained-vs-static "
                           "verdict, and the exact-optimum counts against this lineage",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")

    print(f"bar aligned to {bar_spec['calendar']} (row {bar_row})")
    print(f"cross-implementation: {len(checks)} shared rows, {len(mismatches)} mismatches, "
          f"max |delta| = {payload['cross_implementation_consistency']['max_abs_delta']}")
    for arm, stats in aligned.items():
        if stats.get("n_graded"):
            tag = "" if stats.get("VALID") else "  [INVALID: different campaigns]"
            print(f"  {arm:34} n={stats['n_graded']:2d} capture "
                  f"{stats['pooled_ratio']:+.4f} mean {stats['mean_label']:.4f} "
                  f"cal {stats['distinct_calendars']} opt {stats['exact_optimum_hits']}"
                  f"{tag}")
        else:
            print(f"  {arm:34} {stats['note']}")
    print(f"verdict comparison: {payload['verdict_comparison']['status']}")
    return 0 if payload["cross_implementation_consistency"]["consistent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
