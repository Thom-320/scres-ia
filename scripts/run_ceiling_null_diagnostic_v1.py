#!/usr/bin/env python3
"""Is the priced ceiling real signal, or a maximum taken over 27 noisy draws?

Contract: `docs/ENMIENDA_COSTE_BUFFER_DECLARADO_2026-08-08.md`. Reads the sealed matrices; NO new
episodes for the null test, twelve for the tape-feature probe. Falsifiers from
`supply_chain.falsifiers`.

THIS TESTS MY OWN POSITIVE CLAIM, and it has to. `results/lambda_refinement` reported
HEADROOM_ESTABLISHED_IN_A_PRICE_BAND at 0.045103 [LCB95 +0.028482]. That statistic is

    E_tape[ min over 27 schedules ]   minus   min over schedules of E_tape[ . ]

and by Jensen the first term sits below the second even when every schedule has the SAME true
mean and only noise separates them. A minimum over 27 draws is biased downward, so the gap is
positive under a pure-noise null. Reporting it as headroom without testing that is exactly the
winner's-curse this project has already been bitten by.

THE NULL. Permute the schedule labels independently within each tape. That preserves every tape's
marginal spread of J -- the noise -- and destroys any systematic association between a schedule and
a tape. The whole pipeline is then rerun on the permuted data, comparator selected on permuted
train and scored on permuted test, so the null gap carries the same selection bias the observed one
does. If the observed gap sits inside that distribution, the ceiling is an artifact of taking a
minimum, and three failed conversion attempts stop being surprising.

AND IF IT SURVIVES, the second half asks what actually distinguishes the tapes: warm-up end time,
initial seasonal phase, realised risk events by id, realised demand. Twelve probe episodes.
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
    disclosure, ge, gt, lt, not_applicable, permutation_null, selection_gap, summarise,
)
from supply_chain.seed_custody import custody_falsifier, module_manifest  # noqa: E402

LAMBDA = 0.35
N_NULL = 20_000
CEILING = Path("results/priced_clairvoyant_ceiling/result.json")
R1 = ("R11", "R12", "R13", "R14")
R2 = ("R21", "R22", "R23", "R24")
MODULES = ("supply_chain/falsifiers.py", "supply_chain/arm_runner.py",
           "supply_chain/seed_custody.py")


def pipeline_gap(J: np.ndarray, tr, te) -> float:
    """The exact statistic the ceiling reported: train-selected fixed column minus per-tape min."""
    fixed = int(np.argmin(J[tr].mean(axis=0)))
    return float((J[te, fixed] - J[te].min(axis=1)).mean())


def probe_tape(seed: int) -> dict:
    """One do-nothing episode, for tape-level features only."""
    env = make_env()
    env.reset(seed=int(seed))
    sim = env.unwrapped.sim
    t0 = float(sim.env.now)
    phase0 = int(sim.demand_seasonal.phase(t0)) if sim.demand_seasonal is not None else -1
    done = truncated = False
    demanded0 = float(getattr(sim, "total_demanded", 0.0))
    backlog = []
    try:
        while not (done or truncated):
            backlog.append(float(getattr(sim, "pending_backorder_qty", 0.0) or 0.0))
            _o, _r, done, truncated, _i = env.step(np.array([0.0, -1.0], dtype=np.float32))
        by_id: dict[str, int] = {}
        for e in getattr(sim, "risk_events", []) or []:
            rid = str(e.get("risk_id") if isinstance(e, dict) else getattr(e, "risk_id", "?"))
            by_id[rid] = by_id.get(rid, 0) + 1
        return {"warmup_end_hours": t0, "initial_phase": phase0,
                "demand_total": float(getattr(sim, "total_demanded", 0.0)) - demanded0,
                "backlog_mean": float(np.mean(backlog)),
                "backlog_max": float(np.max(backlog)),
                "events_total": int(sum(by_id.values())),
                **{f"events_{r}": int(by_id.get(r, 0)) for r in R1 + R2}}
    finally:
        env.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--replay-of", required=True)
    ap.add_argument("--output", type=Path,
                    default=Path("results/ceiling_null_diagnostic/result.json"))
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
    observed = pipeline_gap(J, tr, te)
    print(f"  hueco observado en lambda {LAMBDA}: {observed:+.6f}")

    # ---- THE NULL, from the shared module. The first hand-rolled version permuted labels
    # WITHIN each row, which cannot move that row's minimum, so it never tested the interaction it
    # claimed to; under it a gap SMALLER than null meant the train-selected schedule beats a random
    # one, the opposite of how it was read. The module now retains the additive mu + a_i + b_j and
    # permutes only the residuals, destroying exactly "this schedule suits this tape".
    nul = permutation_null(J, tr, te, n_draws=N_NULL, rng=rng, statistic=selection_gap)
    p_value = nul["p_value"]
    print(f"  nulo de interaccion ({N_NULL} sorteos): media {nul['null_mean']:+.6f} "
          f"p95 {nul['null_p95']:+.6f}")
    print(f"  p = P(nulo >= observado) = {p_value:.4f}")

    # ---- what distinguishes the tapes ------------------------------------------------------
    feats = {s: probe_tape(s) for s in all_seeds}
    argmin_per_tape = {s: int(J[all_seeds.index(s)].argmin()) for s in all_seeds}
    best_opt = {s: list(opts[argmin_per_tape[s]]) for s in all_seeds}
    keys = [k for k in next(iter(feats.values())) if isinstance(
        next(iter(feats.values()))[k], (int, float))]
    corr = {}
    ks = np.array([opts[argmin_per_tape[s]][1] for s in all_seeds], dtype=float)   # optimal K
    starts = np.array([opts[argmin_per_tape[s]][0] for s in all_seeds], dtype=float)
    for k in keys:
        x = np.array([feats[s][k] for s in all_seeds], dtype=float)
        if np.std(x) < 1e-12:
            corr[k] = {"sd": 0.0, "corr_with_optimal_K": None, "corr_with_optimal_start": None}
            continue
        corr[k] = {"sd": float(np.std(x)),
                   "corr_with_optimal_K": (float(np.corrcoef(x, ks)[0, 1])
                                           if np.std(ks) > 1e-12 else None),
                   "corr_with_optimal_start": (float(np.corrcoef(x, starts)[0, 1])
                                               if np.std(starts) > 1e-12 else None)}
    n_distinct_optima = len(set(argmin_per_tape.values()))

    falsifiers = {
        "f1_null_preserves_the_selection_bias": ge(
            float(nul["null_mean"]), 0.0,
            "the permutation null must itself produce a POSITIVE gap, because a minimum over 27 "
            "draws is biased downward whatever the truth; a null centred at zero would mean the "
            "permutation destroyed the very bias it exists to measure",
            null_mean=float(nul["null_mean"]), n_draws=N_NULL,
            null_model=nul["null_model"]),
        "f2_observed_gap_exceeds_the_null": lt(
            p_value, 0.05,
            "THE test of my own positive claim. If the observed gap sits inside the distribution "
            "produced by shuffling schedule labels, the ceiling is an artifact of taking a "
            "minimum over 27 noisy options and HEADROOM_ESTABLISHED must be withdrawn",
            observed_gap=observed, p_value=p_value,
            null_mean=float(nul["null_mean"]),
            null_p95=float(nul["null_p95"])),
        "f3_optima_actually_differ_across_tapes": ge(
            float(n_distinct_optima), 2.0,
            "if one schedule were optimal on every tape the gap would be zero by construction and "
            "there would be nothing to diagnose",
            n_distinct=n_distinct_optima, best_by_tape=best_opt),
        "f4_probe_features_vary": gt(
            max(v["sd"] for v in corr.values()), 0.0,
            "tape features that are constant cannot explain which schedule wins; a fully "
            "degenerate probe would leave the diagnostic empty",
            n_features=len(keys)),
        "d1_no_new_episodes_for_the_null": disclosure(
            "the null re-prices the sealed matrix; only the 12 feature probes touch the simulator",
            source=str(CEILING), source_sha=ceil.get("self_sha256")),
        "d2_no_fresh_seeds": not_applicable(
            "declared replay of an already-consumed development block",
            custody=custody_falsifier(all_seeds, replay_of=args.replay_of, exclude=args.output)),
    }
    summary = summarise(falsifiers)
    if not summary["all_passed"] and not falsifiers["f2_observed_gap_exceeds_the_null"]["passed"]:
        verdict = "CEILING_IS_A_MINIMUM_OVER_NOISE_HEADROOM_WITHDRAWN"
    elif not summary["all_passed"]:
        verdict = "BLOCKED_INSTRUMENT"
    else:
        verdict = "CEILING_SURVIVES_THE_PERMUTATION_NULL"

    ranked = sorted((k for k in corr if corr[k]["corr_with_optimal_K"] is not None),
                    key=lambda k: -abs(corr[k]["corr_with_optimal_K"]))
    print(f"\n  optimos distintos entre tapes: {n_distinct_optima}")
    print(f"  {'rasgo':>22} {'sd':>12} {'corr con K optimo':>18}")
    for k in ranked[:8]:
        print(f"  {k:>22} {corr[k]['sd']:12.1f} {corr[k]['corr_with_optimal_K']:18.4f}")
    print(f"\n  veredicto: {verdict}")
    print(f"  falsadores: {summary['n_computed']} computados, {summary['n_failed']} fallidos, "
          f"{summary['n_disclosures']} divulgaciones, {summary['n_not_applicable']} no aplicables")
    for k, v in falsifiers.items():
        if v.get("computed"):
            print(f"    {k:48s} {'PASA' if v['passed'] else 'FALLA'}")

    payload = {
        "schema_version": "ceiling_null_diagnostic_v1",
        "claim_status": verdict,
        "scope": "DEVELOPMENT_DIAGNOSTIC_TESTS_OUR_OWN_POSITIVE_CLAIM",
        "run_role": "PERMUTATION_NULL_AND_TAPE_FEATURES", "replay_of": args.replay_of,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "module_manifest": module_manifest(MODULES, script=__file__),
        "tests": {"path": "results/lambda_refinement/result.json",
                  "claim": "HEADROOM_ESTABLISHED_IN_A_PRICE_BAND"},
        "lambda": LAMBDA, "observed_gap": observed,
        "null": dict(nul),
        "superseded_null": {"method": "labels permuted within row",
                            "why_withdrawn": ("a within-row permutation cannot move that row's "
                                              "minimum, so it never tested the interaction; it "
                                              "only randomised the fixed column")},
        "per_tape_optimum": {str(s): best_opt[s] for s in all_seeds},
        "n_distinct_optima": n_distinct_optima,
        "tape_features": {str(s): feats[s] for s in all_seeds},
        "feature_correlations": corr, "ranked_features": ranked,
        "falsifiers": falsifiers, "falsifier_summary": summary,
        "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(payload, args.output, contract=args.contract, reference=CEILING)
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
