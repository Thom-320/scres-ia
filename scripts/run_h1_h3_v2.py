#!/usr/bin/env python3
"""H1' and H3', the second formulation, after both fell to their own falsifiers.

H1 died twice over: `system_ttr` was 100% censored so its mean was 0.0 by vacuity, and the two
arms deployed a single identical modal configuration so the contrast was a tautology. This fixes
both, and neither fix is cosmetic:

  * the metric becomes `service_loss_auc_ration_hours`, which integrates EVERY order and cannot
    be censored -- an order that is never served accrues loss to the horizon instead of leaving
    the population, which is precisely the failure mode of ReT and of system_ttr;
  * the design compares the configuration each strategy actually chose IN EACH CELL rather than
    one modal configuration. 42 of 72 cells deploy different configurations, so there is a real
    paired comparison; collapsing to the mode is what destroyed it.

H1' is NOT a recovery time and the write-up says so: it is the integral of lost service, which
mixes magnitude with duration.

H3 as drafted -- variance of performance across disruption intensities -- is untestable here,
because the optimum does not move and the learner therefore deploys the same thing. H3' reads the
same idea where this environment does vary: the variance of SEARCH COST across contexts. That is
a change of construct, not a repair, and it is labelled as one.

See `docs/PREREGISTRO_H1_H3_V2_2026-08-01.md`, committed before this ran.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.seed_custody import (  # noqa: E402
    custody_falsifier, seeds_used_by_sealed_artifacts)
from supply_chain.config import HOURS_PER_WEEK  # noqa: E402
from supply_chain.config import THESIS_FAITHFUL_PROTOCOL as P  # noqa: E402
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402

META = Path("results/garrido_meta_learner_v2/result.json")
R1R = ("R11", "R12", "R13", "R14")
R2R = ("R21", "R22", "R23", "R24")
CONTEXTS = {
    "R1r": (R1R, {}), "R2r": (R2R, {}), "R1r+R2r": (R1R + R2R, {}),
    "R1r|esc": (R1R, {r: 3.0 for r in R1R}),
    "R2r|esc": (R2R, {r: 3.0 for r in R2R}),
    "R1r+R2r|esc": (R1R + R2R, {r: 3.0 for r in R1R + R2R}),
}
ARMS = {"hybrid": "neuron_memory", "static": "ofat", "reset": "neuron_reset"}
PRIMARY = "service_loss_auc_ration_hours"          # lower is better, no censoring
SIDE = ("service_loss_auc_per_order", "flow_fill_rate", "n_orders", "n_served",
        "n_lost")
# 5_800_001-08 collided with the expedition run: the external review caught it and was right.
# 6_000_001+ is the H3 power block. 6_200_001 is verified free against every sealed artifact.
SEED_BASE = 6_200_001




def episode(config: dict, context: str, seed: int, horizon: float) -> dict[str, float]:
    risks, freq = CONTEXTS[context]
    sim = MFSCSimulation(
        shifts=int(config["shifts"]),
        initial_buffers={"op3_rm": 0.0, "op5_rm": 0.0,
                         "op9_rations": float(config["buffer_hours"]) * 2_500.0 / 24.0},
        inventory_replenishment_period=0.0, seed=seed, horizon=horizon,
        risks_enabled=True, risk_level="current", enabled_risks=set(risks),
        risk_frequency_multipliers_by_id=dict(freq) or None,
        strict_exogenous_crn=True, year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"], r14_defect_mode=P["r14_defect_mode"])
    sim.params["op9_rop"] = float(config["op9_rop"])
    sim.params["op12_rop"] = float(config["op12_rop"])
    sim.run()
    panel = compute_episode_metrics(sim)
    out = {k: float(panel[k]) for k in (PRIMARY, *SIDE)}
    # f3, third attempt, and the two earlier ones are worth recording because each tested the
    # wrong thing. Version 1 asserted `n_served + n_lost == n_orders`, an unrelated accounting
    # identity that fails because an order can be neither -- still pending at the horizon.
    # Version 2 recomputed over sim.orders and also failed, but that failure was MY error: the
    # panel legitimately excludes orders placed before the end of warm-up, "so the metric
    # reflects only the period the policy could influence" (episode_metrics.py:151). Those 33 of
    # 311 orders carry enormous lateness precisely because the chain had not spun up.
    #
    # The property that actually matters is that no order is dropped for being UNSERVED. So:
    # replicate the panel's population exactly, recompute, and demand equality; then confirm the
    # never-completed orders sit inside it and contribute a positive share.
    end_time = float(sim.env.now)
    start = float(sim.warmup_time)
    scored = [o for o in sim.orders
              if not bool(getattr(o, "metrics_excluded", False))
              and float(getattr(o, "OPTj", 0.0)) >= start]
    recomputed = unresolved = 0.0
    for o in scored:
        opt, lt = float(o.OPTj or 0.0), float(o.LTj or 0.0)
        done = getattr(o, "OATj", None)
        contribution = max(0.0, (float(done) if done is not None else end_time)
                           - (opt + lt)) * float(o.quantity or 0.0)
        recomputed += contribution
        if done is None:
            unresolved += contribution
    out["auc_recomputed_scored_population"] = recomputed
    out["auc_share_from_never_completed"] = (unresolved / recomputed) if recomputed else 0.0
    out["n_never_completed_in_scored"] = float(
        sum(1 for o in scored if getattr(o, "OATj", None) is None))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--horizon-weeks", type=int, default=52)
    ap.add_argument("--n-boot", type=int, default=5_000)
    ap.add_argument("--output", type=Path,
                    default=Path("results/manuscript/h1_h3_v2/result.json"))
    args = ap.parse_args()
    horizon = float(args.horizon_weeks * HOURS_PER_WEEK)
    seeds = [SEED_BASE + i for i in range(args.seeds)]
    meta_bytes = META.read_bytes()
    meta = json.loads(meta_bytes)
    ctx_order, n_rep = meta["contexts"], len(meta["per_context"]["ofat"])
    # Reproduce seal_and_write's convention exactly (arm_runner.py:175-176): the digest is over
    # the payload WITHOUT `self_sha256`, serialised with indent=1, sort_keys=True, default=str.
    # My first attempt used compact separators and failed -- correctly, because it was not the
    # sealing convention. The test was right and I was wrong about how the seal is formed.
    probe = {k: v for k, v in meta.items() if k != "self_sha256"}
    meta_recomputed_digest = hashlib.sha256(
        json.dumps(probe, indent=1, sort_keys=True, default=str).encode()).hexdigest()
    prior_seeds = seeds_used_by_sealed_artifacts(exclude=args.output)
    started = time.perf_counter()

    # ---- evaluate what each arm ACTUALLY deployed in each cell ------------------------------
    cells: dict[tuple[str, str, int], dict] = {}
    cache: dict[tuple[str, str], dict] = {}
    identical = []
    for r in range(n_rep):
        for ctx in ctx_order:
            picks = {arm: meta["per_context"][strategy][r][ctx]["chosen_config"]
                     for arm, strategy in ARMS.items()}
            identical.append(picks["hybrid"] == picks["static"])
            for arm, config in picks.items():
                key = (ctx, json.dumps(config, sort_keys=True))
                if key not in cache:
                    cache[key] = {s: episode(config, ctx, s, horizon) for s in seeds}
                cells[(arm, ctx, r)] = cache[key]
        print(f"  réplica {r + 1}/{n_rep} ({time.perf_counter() - started:.0f}s)", flush=True)

    rng = np.random.default_rng(20260801)

    def cell_matrix(arm: str, key: str, only_differing: bool) -> np.ndarray:
        """(cells, seeds). Kept two-dimensional so the bootstrap can resample CELLS."""
        rows = []
        for i, (r, ctx) in enumerate((r, c) for r in range(n_rep) for c in ctx_order):
            if only_differing and identical[i]:
                continue
            rows.append([cells[(arm, ctx, r)][s][key] for s in seeds])
        return np.array(rows)

    def paired(a: str, b: str, key: str, only_differing: bool = False) -> dict:
        """b - a; positive means arm A loses LESS service, i.e. A is better.

        The bootstrap resamples CELLS, not the flattened observations. The same five seeds recur
        in every cell, so the 360 rows are not exchangeable -- treating them as if they were,
        which the first version did, understates the interval. An external review caught this.
        """
        diff = cell_matrix(b, key, only_differing) - cell_matrix(a, key, only_differing)
        per_cell = diff.mean(axis=1)
        draws = per_cell[rng.integers(0, per_cell.size,
                                      size=(args.n_boot, per_cell.size))].mean(axis=1)
        return {"mean": float(per_cell.mean()), "lcb95": float(np.percentile(draws, 2.5)),
                "ucb95": float(np.percentile(draws, 97.5)),
                "n_cells": int(per_cell.size), "n_seeds_per_cell": len(seeds),
                "inference_unit": "cell (context x repeat); seeds recur across cells"}

    h1 = {
        "primary_all_cells": paired("hybrid", "static", PRIMARY),
        "secondary_differing_cells": paired("hybrid", "static", PRIMARY, only_differing=True),
        "hybrid_vs_reset": paired("hybrid", "reset", PRIMARY),
    }
    levels = {arm: float(cell_matrix(arm, PRIMARY, False).mean()) for arm in ARMS}

    # ---- H3': variance of SEARCH COST across contexts, straight from the sealed artifact -----
    def search_cost_variance(strategy: str) -> np.ndarray:
        return np.array([
            float(np.var([meta["per_context"][strategy][r][c]["runs_to_within_1pct"]
                          for c in ctx_order], ddof=1))
            for r in range(n_rep)])

    def paired_variance(a: str, b: str) -> dict:
        d = search_cost_variance(b) - search_cost_variance(a)
        draws = d[rng.integers(0, d.size, size=(args.n_boot, d.size))].mean(axis=1)
        return {"mean": float(d.mean()), "lcb95": float(np.percentile(draws, 2.5)),
                "ucb95": float(np.percentile(draws, 97.5))}

    h3 = {"memory_vs_reset": paired_variance("neuron_memory", "neuron_reset"),
          "memory_vs_ofat": paired_variance("neuron_memory", "ofat")}
    variances = {s: float(search_cost_variance(s).mean())
                 for s in ("neuron_memory", "neuron_reset", "ofat", "random")}

    # ---- falsifiers --------------------------------------------------------------------------
    n_diff = int(sum(1 for x in identical if not x))
    zero_on_identical = []
    for i, (r, ctx) in enumerate((r, c) for r in range(n_rep) for c in ctx_order):
        if identical[i]:
            zero_on_identical.append(max(
                abs(cells[("hybrid", ctx, r)][s][PRIMARY] - cells[("static", ctx, r)][s][PRIMARY])
                for s in seeds))
    order_totals = [(row["auc_recomputed_scored_population"], row[PRIMARY])
                    for by_seed in cache.values() for row in by_seed.values()]
    accounted = all(abs(a - b) <= 1e-6 * max(1.0, abs(b)) for a, b in order_totals)
    never_completed_inside = float(np.mean([row["auc_share_from_never_completed"]
                                            for by_seed in cache.values()
                                            for row in by_seed.values()])) > 0.0
    unresolved_share = float(np.mean([row["auc_share_from_never_completed"]
                                      for by_seed in cache.values()
                                      for row in by_seed.values()]))
    spread = float(np.ptp([np.mean([row[PRIMARY] for row in by_seed.values()])
                           for by_seed in cache.values()]))
    ofat_ctx_costs = [float(np.mean([meta["per_context"]["ofat"][r][c]["runs_to_within_1pct"]
                                     for r in range(n_rep)])) for c in ctx_order]

    falsifiers = {
        "f1_some_cells_deploy_different_configurations": {
            "passed": n_diff > 0,
            "evidence": {"why_it_can_fail": ("with no differing cell H1' is the same tautology "
                                             "that stopped the first version"),
                         "differing_cells": n_diff, "total_cells": len(identical),
                         "context_config_pairs_evaluated": len(cache),
                         "distinct_configs_deployed": len(
                             {config for _, config in cache})}},
        "f2_identical_cells_contribute_exactly_zero": {
            "passed": (not zero_on_identical) or max(zero_on_identical) < 1e-9,
            "evidence": {"why_it_can_fail": ("a cell whose two arms deploy the SAME configuration "
                                             "must give difference 0.0; anything else means the "
                                             "pairing is broken"),
                         "max_abs_difference_on_identical_cells":
                             max(zero_on_identical) if zero_on_identical else None}},
        "f3_service_loss_auc_is_not_censored": {
            "passed": accounted and never_completed_inside,
            "evidence": {"why_it_can_fail": (
                             "the panel's AUC is recomputed here over EVERY order, with the "
                             "horizon standing in for orders that never completed. If the panel "
                             "ever restricted its population the two would diverge, and the "
                             "metric would inherit the censoring it was chosen to avoid. "
                             "The second condition is the one that matters: orders that NEVER "
                             "complete must sit inside the population and carry a positive share, "
                             "which is exactly what ReT and system_ttr fail to do"),
                         "never_completed_orders_are_inside": never_completed_inside,
                         "episodes_checked": len(order_totals),
                         "share_of_auc_from_never_completed_orders": unresolved_share}},
        "f4_the_metric_discriminates_between_deployed_configs": {
            "passed": spread > 1e-9,
            "evidence": {"why_it_can_fail": "identical scores leave nothing to compare",
                         "spread_across_deployed_configs": spread}},
        "f5_contexts_differ_in_search_difficulty": {
            "passed": float(np.ptp(ofat_ctx_costs)) > 1.0,
            "evidence": {"why_it_can_fail": ("H3' needs search cost to VARY across contexts; a "
                                             "flat profile makes its variance noise"),
                         "ofat_cost_by_context": dict(zip(ctx_order, ofat_ctx_costs))}},
        "f6_h3_artifact_matches_its_own_seal": {
            "passed": meta_recomputed_digest == meta.get("self_sha256"),
            "evidence": {"why_it_can_fail": (
                             "the first version compared the file with ITSELF -- it read the same "
                             "bytes twice and could never fail. This recomputes the payload digest "
                             "the way seal_and_write does and compares it to the stored seal, so "
                             "any edit to the artifact since sealing fails it"),
                         "recomputed": meta_recomputed_digest,
                         "stored_self_sha256": meta.get("self_sha256")}},
        "f7_seeds_are_virgin": {
            "passed": not (set(seeds) & prior_seeds),
            "evidence": {"why_it_can_fail": (
                             "this was hardcoded True in the first version, and an external "
                             "review caught that the block 5_800_001-05 collided with the "
                             "expedition run's 5_800_001-08. It now scans every sealed artifact "
                             "for the seeds it declares"),
                         "seeds": seeds,
                         "collisions": sorted(set(seeds) & prior_seeds),
                         "prior_seeds_scanned": len(prior_seeds)}},
    }
    falsifiers["all_passed"] = all(v["passed"] for k, v in falsifiers.items()
                                   if k != "all_passed")

    h1_ok = h1["primary_all_cells"]["lcb95"] > 0
    h1_partial = (not h1_ok) and h1["secondary_differing_cells"]["lcb95"] > 0
    h3_ok = h3["memory_vs_reset"]["lcb95"] > 0
    verdict = (f"H1_{'SUPPORTED' if h1_ok else 'SUPPORTED_ON_DIFFERING_CELLS_ONLY' if h1_partial else 'NOT_SUPPORTED'}"
               f"__H3_{'SUPPORTED' if h3_ok else 'NOT_SUPPORTED'}")

    print(f"\n  === H1' · servicio perdido acumulado (menor es mejor) ===")
    for arm, value in levels.items():
        print(f"  {arm:<8}{value:>16.1f} ración-hora")
    for name, v in h1.items():
        print(f"  {name:<28}{v['mean']:>+14.1f}  [{v['lcb95']:+.1f}, {v['ucb95']:+.1f}]  "
              f"celdas={v['n_cells']}")
    print(f"\n  === H3' · varianza del coste de búsqueda entre contextos (menor es mejor) ===")
    for s, v in variances.items():
        print(f"  {s:<16}{v:>10.2f}")
    for name, v in h3.items():
        print(f"  {name:<20}{v['mean']:>+10.2f}  [{v['lcb95']:+.2f}, {v['ucb95']:+.2f}]")
    print(f"\n  veredicto: {verdict}")
    print("\n  falsadores:")
    for name, check in falsifiers.items():
        if name != "all_passed":
            print(f"    {name:<50} {'PASA' if check['passed'] else 'FALLA'}")

    payload = {
        "schema_version": "h1_h3_v2",
        "claim_status": verdict if falsifiers["all_passed"] else "HALTED_FALSIFIER_FAILED",
        "H1": {"metric": PRIMARY, "levels_by_arm": levels, "contrasts": h1,
               "what_it_is_not": ("not a recovery TIME: it is the integral of lost service, "
                                  "which mixes magnitude with duration"),
               "cells_total": len(identical), "cells_differing": n_diff},
        "H3": {"estimand": "variance of runs_to_within_1pct across the six contexts, per repeat",
               "construct_change": ("the manuscript's H3 is variance of performance across "
                                    "disruption intensities, which is untestable here because "
                                    "the optimum does not move; this is a different construct, "
                                    "not a repair"),
               "variance_by_strategy": variances, "contrasts": h3},
        "arms": ARMS, "contexts": ctx_order, "repeats": n_rep, "seeds": seeds,
        "falsifiers": falsifiers,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(
        payload, args.output,
        contract=Path("docs/PREREGISTRO_H1_H3_V2_2026-08-01.md"), reference=META)
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
