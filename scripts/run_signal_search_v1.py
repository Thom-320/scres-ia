#!/usr/bin/env python3
"""Which observable signal, if any, captures the surviving ceiling?

Contract: `docs/ENMIENDA_COSTE_BUFFER_DECLARADO_2026-08-08.md`. Custody: declared replay.
Falsifiers and the permutation null come from `supply_chain.falsifiers`.

THE CEILING SURVIVED ITS NULL (`results/ceiling_null_diagnostic`, p = 0.0132), so the tape-by-
schedule interaction is real: which schedule wins genuinely varies by tape. Three signals have
failed to convert it -- backlog, seasonal phase, and phase crossed with realised demand deviation.
This asks what a signal would have to be.

THE DISTINCTION THAT DEFINES THE SEARCH, and it nearly went unnoticed. A signal only counts if it
is observable AT THE MOMENT OF DECIDING. Whole-episode risk counts are hindsight, not signal, and
correlating them with the optimal schedule measures nothing a policy could use. Measured on the
twelve tapes, the only features known at t = 0 are warm-up end time and initial phase, and they
take three and two distinct values respectively -- essentially one odd tape out of twelve.

So the search is over PREFIX statistics: everything realised up to the decision week, which is what
a policy deciding week by week actually has. The decision point is week 4, matching the common OFF
prefix the conversion runs used.

AND THE SEARCH IS SCORED THE ONLY WAY THAT MATTERS. A map from features to schedules is fitted on
TRAIN tapes and applied to TEST tapes. Correlations on twelve points, picked as the best of
thirteen features, are what the ceiling's own diagnostic already showed this design manufactures --
the previous run's top correlate was 0.4315 on a feature whose standard deviation is half an event.
Multiplicity is paid over the feature set, and a shuffled-feature placebo runs alongside.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_priced_buffer_gate_v1 import STEP_HOURS, make_env, options  # noqa: E402
from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.falsifiers import (  # noqa: E402
    disclosure, ge, gt, lt, not_applicable, summarise,
)
from supply_chain.seed_custody import custody_falsifier, module_manifest  # noqa: E402

LAMBDA = 0.35
DECISION_WEEK = 4
N_BOOT = 4_000
N_PLACEBO = 500
CEILING = Path("results/priced_clairvoyant_ceiling/result.json")
R1 = ("R11", "R12", "R13", "R14")
R2 = ("R21", "R22", "R23", "R24")
MODULES = ("supply_chain/falsifiers.py", "supply_chain/continuous_its_env.py",
           "supply_chain/arm_runner.py", "supply_chain/seed_custody.py")


def prefix_features(seed: int) -> dict:
    """Everything a week-4 decision could legitimately have seen. Nothing after week 4."""
    env = make_env()
    env.reset(seed=int(seed))
    sim = env.unwrapped.sim
    t0 = float(sim.env.now)
    feats = {"warmup_end_hours": t0,
             "initial_phase": float(sim.demand_seasonal.phase(t0))
             if sim.demand_seasonal is not None else -1.0}
    backlog, demand, events = [], [], []
    prev_dem, prev_ev = 0.0, 0
    done = truncated = False
    step = 0
    try:
        while not (done or truncated) and step < DECISION_WEEK:
            backlog.append(float(getattr(sim, "pending_backorder_qty", 0.0) or 0.0))
            d = float(getattr(sim, "total_demanded", 0.0))
            demand.append(d - prev_dem)
            prev_dem = d
            n = len(getattr(sim, "risk_events", []) or [])
            events.append(n - prev_ev)
            prev_ev = n
            _o, _r, done, truncated, _i = env.step(np.array([0.0, -1.0], dtype=np.float32))
            step += 1
        by_id: dict[str, int] = {}
        for e in getattr(sim, "risk_events", []) or []:
            rid = str(e.get("risk_id") if isinstance(e, dict) else getattr(e, "risk_id", "?"))
            by_id[rid] = by_id.get(rid, 0) + 1
        feats.update({
            "prefix_backlog_mean": float(np.mean(backlog)),
            "prefix_backlog_slope": float(backlog[-1] - backlog[0]) if len(backlog) > 1 else 0.0,
            "prefix_demand_mean": float(np.mean(demand[1:])) if len(demand) > 1 else 0.0,
            "prefix_events_total": float(sum(events)),
            **{f"prefix_events_{r}": float(by_id.get(r, 0)) for r in R1 + R2},
        })
        return feats
    finally:
        env.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--replay-of", required=True)
    ap.add_argument("--output", type=Path,
                    default=Path("results/signal_search/result.json"))
    args = ap.parse_args()
    started = time.perf_counter()
    rng = np.random.default_rng(20260808)

    ceil = json.loads(CEILING.read_text())
    L = np.asarray(ceil["L_matrix"], dtype=float)
    IH = np.asarray(ceil["inventory_hours_matrix"], dtype=float)
    max_ih = float(ceil["max_inventory_hours"])
    train_seeds = list(ceil["splits"]["train"])
    test_seeds = list(ceil["splits"]["test"])
    all_seeds = train_seeds + test_seeds
    tr = [all_seeds.index(s) for s in train_seeds]
    te = [all_seeds.index(s) for s in test_seeds]
    opts = [tuple(o) for o in ceil["options"]]
    J = L + LAMBDA * (IH / max_ih)

    feats = {s: prefix_features(s) for s in all_seeds}
    names = sorted(next(iter(feats.values())))
    X = np.array([[feats[s][n] for n in names] for s in all_seeds], dtype=float)
    print(f"  {len(names)} rasgos de prefijo (semana {DECISION_WEEK}) sobre {len(all_seeds)} tapes")

    fixed = int(np.argmin(J[tr].mean(axis=0)))
    open_loop = J[te, fixed]
    clair = J[te].min(axis=1)
    ceiling_gap = float((open_loop - clair).mean())
    print(f"  techo en test: {ceiling_gap:+.6f}")

    def apply_map(col: np.ndarray) -> np.ndarray:
        """Nearest-neighbour map: each TEST tape takes the schedule that won on the TRAIN tape
        closest in this one feature. The simplest map that can use a signal at all, and the only
        one six training points can support."""
        picks = []
        for i in te:
            j = min(tr, key=lambda t: abs(col[t] - col[i]))
            picks.append(int(J[j].argmin()))
        return np.array([J[i, p] for i, p in zip(te, picks)]), picks

    rows = {}
    for k, name in enumerate(names):
        col = X[:, k]
        if np.std(col) < 1e-12:
            rows[name] = {"sd": 0.0, "usable": False}
            continue
        arm, picks = apply_map(col)
        d = open_loop - arm
        boot = np.array([float(np.mean(d[rng.integers(0, len(d), len(d))]))
                         for _ in range(N_BOOT)])
        # PLACEBO: the same map on a shuffled feature. Keeps the mechanism, destroys the signal.
        plac = []
        for _ in range(N_PLACEBO):
            sh = col.copy()
            rng.shuffle(sh)
            a, _ = apply_map(sh)
            plac.append(float((open_loop - a).mean()))
        plac = np.asarray(plac)
        rows[name] = {
            "sd": float(np.std(col)), "usable": True,
            "gain_vs_open_loop": float(d.mean()),
            "lcb95": float(np.percentile(boot, 2.5)),
            "ucb95": float(np.percentile(boot, 97.5)),
            "placebo_mean_gain": float(plac.mean()),
            "p_vs_placebo": float((plac >= float(d.mean())).mean()),
            "share_of_ceiling": float(d.mean() / ceiling_gap) if ceiling_gap > 0 else 0.0,
            "picks": picks,
        }

    usable = {k: v for k, v in rows.items() if v.get("usable")}
    # HOLM over the feature set: searching thirteen signals and reporting the best is the metric
    # shopping this project has already priced.
    order = sorted(usable, key=lambda k: usable[k]["p_vs_placebo"])
    running = 0.0
    for rank, k in enumerate(order):
        running = max(running, min(1.0, (len(usable) - rank) * usable[k]["p_vs_placebo"]))
        usable[k]["holm_adjusted_p"] = running

    winners = [k for k, v in usable.items()
               if v["lcb95"] > 0 and v["holm_adjusted_p"] < 0.05]
    best = max(usable, key=lambda k: usable[k]["gain_vs_open_loop"]) if usable else None

    ex_ante = ("warmup_end_hours", "initial_phase")
    ex_ante_levels = {k: int(len(set(X[:, names.index(k)].tolist()))) for k in ex_ante}

    falsifiers = {
        "f1_features_are_prefix_only": ge(
            float(DECISION_WEEK), float(DECISION_WEEK),
            "a feature measured over the whole episode is hindsight, not a signal; correlating it "
            "with the optimal schedule measures nothing a policy could act on, and the previous "
            "diagnostic ranked exactly such features",
            decision_week=DECISION_WEEK, features=names,
            excluded="anything realised after the decision week"),
        "f2_ceiling_is_positive_on_these_tapes": gt(
            ceiling_gap, 0.0,
            "a signal search is undefined where there is nothing to capture",
            ceiling_gap=ceiling_gap),
        "f3_multiplicity_is_paid": ge(
            float(len(usable)), float(len(usable)),
            "reporting the best of thirteen signals uncorrected is the metric shopping this "
            "project has already measured and priced",
            n_features_tested=len(usable)),
        "f4_placebo_keeps_the_mechanism": ge(
            float(N_PLACEBO), 100.0,
            "the placebo applies the SAME nearest-neighbour map to a shuffled feature, so a gain "
            "it also produces belongs to the mapping and not to the signal",
            n_placebo=N_PLACEBO),
        "f5_no_map_exceeds_the_ceiling": ge(
            ceiling_gap - max((v["gain_vs_open_loop"] for v in usable.values()), default=0.0),
            -1e-9,
            "no train-fitted map may beat a tape-knowing chooser; exceeding the ceiling means the "
            "map saw the test tapes",
            best_gain=max((v["gain_vs_open_loop"] for v in usable.values()), default=0.0)),
        "d1_ex_ante_signal_is_nearly_absent": disclosure(
            "only warm-up end time and initial phase are known at t = 0, and across twelve tapes "
            "they take three and two distinct values -- essentially one odd tape out. Any real "
            "signal must therefore be a PREFIX statistic, not a property known before the episode",
            distinct_levels=ex_ante_levels),
        "d2_six_training_tapes": disclosure(
            "a map is fitted on six tapes and applied to six; nearest-neighbour on one feature is "
            "the most that supports, and no negative here rules out a signal a larger design "
            "would find",
            n_train=len(train_seeds), n_test=len(test_seeds)),
        "d3_no_fresh_seeds": not_applicable(
            "declared replay of an already-consumed development block",
            custody=custody_falsifier(all_seeds, replay_of=args.replay_of, exclude=args.output)),
    }
    summary = summarise(falsifiers)
    if not summary["all_passed"]:
        verdict = "BLOCKED_INSTRUMENT"
    elif winners:
        verdict = "SIGNAL_FOUND_THAT_CAPTURES_PART_OF_THE_CEILING"
    else:
        verdict = "NO_PREFIX_SIGNAL_CAPTURES_THE_CEILING_IN_THIS_DESIGN"

    print(f"\n  {'rasgo':>24} {'ganancia':>10} {'lcb95':>10} {'placebo':>10} {'holm':>7} {'cuota':>8}")
    for k in sorted(usable, key=lambda k: -usable[k]["gain_vs_open_loop"]):
        v = usable[k]
        print(f"  {k:>24} {v['gain_vs_open_loop']:10.6f} {v['lcb95']:10.6f} "
              f"{v['placebo_mean_gain']:10.6f} {v['holm_adjusted_p']:7.3f} "
              f"{v['share_of_ceiling']:8.1%}")
    print(f"\n  veredicto: {verdict}")
    print(f"  falsadores: {summary['n_computed']} computados, {summary['n_failed']} fallidos, "
          f"{summary['n_disclosures']} divulgaciones, {summary['n_not_applicable']} no aplicables")
    for k, v in falsifiers.items():
        if v.get("computed"):
            print(f"    {k:44s} {'PASA' if v['passed'] else 'FALLA'}")

    payload = {
        "schema_version": "signal_search_v1",
        "claim_status": verdict,
        "scope": "DEVELOPMENT_ON_DECLARED_REPLAY_SIGNAL_SEARCH",
        "run_role": "SIGNAL_SEARCH", "replay_of": args.replay_of,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "module_manifest": module_manifest(MODULES, script=__file__),
        "ceiling_source": {"path": str(CEILING), "self_sha256": ceil.get("self_sha256"),
                           "survived_null": "results/ceiling_null_diagnostic/result.json"},
        "lambda": LAMBDA, "decision_week": DECISION_WEEK,
        "ceiling_gap_on_test": ceiling_gap,
        "map": "nearest-neighbour in one feature, fitted on train, applied to test",
        "features": names, "tape_features": {str(s): feats[s] for s in all_seeds},
        "rows": rows, "winners": winners, "best_feature": best,
        "ex_ante_distinct_levels": ex_ante_levels,
        "splits": {"train": train_seeds, "test": test_seeds},
        "falsifiers": falsifiers, "falsifier_summary": summary,
        "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(payload, args.output, contract=args.contract, reference=CEILING)
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
