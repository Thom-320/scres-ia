#!/usr/bin/env python3
"""E*-C gate G1: does a shared storage budget create a REGIME-DEPENDENT decision?

Finite storage removes the second assumption Garrido declares verbatim (WRAP 2017 Section 6.5.5,
"storage capacities ... are assumed to be unlimited"), and a shared TOTAL is what turns it from a
constraint into a decision -- independent caps only make the chain worse, with nothing traded off.

The endpoint is `flow_fill_rate`, and the parent preregistration explains why: a liveness probe on
burned tapes showed that blocking 1.3 MILLION rations left `worst_claimant_fill` at 0.6791 in that
tested horizon. Both are terminal delivered/demanded ratios; `flow_fill_rate` is not a pure timing
metric and is sensitive to delay only through what remains undelivered at the horizon.

Contract: docs/ENMIENDA_ESTAR_CAPACIDAD_BARRIDO_V2_REPLAY_2026-08-05.md
Seeds: burned block 5_200_001-16, declared replay. No fresh roots. This is an instrument repair,
not a new confirmation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.config import HOURS_PER_WEEK  # noqa: E402
from supply_chain.config import THESIS_FAITHFUL_PROTOCOL as P  # noqa: E402
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.seed_custody import custody_falsifier, module_manifest  # noqa: E402
from supply_chain.service_first_metric import claimant_fills  # noqa: E402
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402

R1R = ("R11", "R12", "R13", "R14")
R2R = ("R21", "R22", "R23", "R24")
REGIMES = {
    "R1r+R2r|base": (R1R + R2R, {}, {}),
    "R1r+R2r|freq3_imp2": (R1R + R2R, {"R23": 3.0}, {"R23": 2.0}),
}
SHARES = tuple(round(0.1 + 0.1 * i, 2) for i in range(9))
#: 600 binds hard and 3000 does not bind at all, so both of these sit inside the live region.
#: Two budgets, because one cannot distinguish "no headroom" from "wrong level".
BUDGETS = (600.0, 1200.0)
PRIMARY = "flow_fill_rate"
SESOI = 0.010
MARGINS = {"worst_claimant_fill": 0.010, "lost_orders": 0.50, "flow_fill_rate": 0.005}
F6_METRICS = ("worst_claimant_fill", "lost_orders")
SEED_BASE = 5_200_001
WEEKS = 26
E_STAR_MODULES = (
    "supply_chain/supply_chain.py",
    "supply_chain/node_capacity.py",
    "supply_chain/config.py",
    "supply_chain/episode_metrics.py",
    "supply_chain/arm_runner.py",
    "supply_chain/cssu_allocation.py",
    "supply_chain/service_first_metric.py",
    "supply_chain/seed_custody.py",
)


def episode(seed, risks, freq, impact, *, share_a, budget):
    """One rollout. `budget=None` is the no-capacity null arm on the same tape."""
    from supply_chain.node_capacity import budgeted_ledger

    sim_kwargs = dict(
        shifts=1, initial_buffers={n: 0.0 for n in ("op3_rm", "op5_rm", "op9_rations")},
        inventory_replenishment_period=0.0, seed=seed, horizon=float(WEEKS * HOURS_PER_WEEK),
        risks_enabled=True, risk_level="current", enabled_risks=set(risks),
        risk_frequency_multipliers_by_id=dict(freq) or None,
        risk_impact_multipliers_by_id=dict(impact) or None,
        cssu_topology_mode="split_v1", cssu_allocation_a=0.5,
        cssu_service_rule="FIFO_PARTIAL", cssu_reallocate_unused=False,
        cssu_storage_capacity=None,
        order_fulfillment_mode="op9_linked", op9_dispatch_policy="fixed_clock_daily",
        strict_exogenous_crn=True, year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"], r14_defect_mode=P["r14_defect_mode"])
    budget_error = None
    if budget is not None:
        # The contract requires the total to be enforced by the LEDGER, not by a bare split: its
        # __post_init__ refuses to exist unless the capacities sum to the declared budget. The
        # three upstream nodes receive share 0 and are NOT wired into the DES -- this sweep
        # constrains CSSU storage only, which is stated in the payload scope.
        ledger = budgeted_ledger(float(budget), {"cssu_a": share_a, "cssu_b": 1.0 - share_a})
        caps = ledger.capacities
        budget_error = float(sum(caps.values()) - float(budget))
        sim_kwargs["cssu_storage_capacity"] = {"A": caps["cssu_a"], "B": caps["cssu_b"]}
    sim = MFSCSimulation(**sim_kwargs)
    sim.run()
    panel = compute_episode_metrics(sim)
    fills = claimant_fills(sim)
    ledger = sim._cssu_capacity_ledger
    blocked_total = 0.0 if ledger is None else float(sum(ledger.blocked_qty.values()))
    admitted_total = None if ledger is None else float(sum(ledger.admitted_qty.values()))
    binding_fraction = 0.0 if ledger is None else float(ledger.binding_fraction())
    return {PRIMARY: float(panel[PRIMARY]),
            "worst_claimant_fill": float(min(fills.values())) if fills else float("nan"),
            "lost_orders": float(panel["lost_orders"]),
            "demanded_total": float(sum(sim.cssu_demanded.values())),
            "demanded_by_claimant": {k: float(v) for k, v in sim.cssu_demanded.items()},
            "delivered_total": float(sum(sim.cssu_delivered.values())),
            "blocked_total": blocked_total,
            "admitted_total": admitted_total,
            "binding_fraction": binding_fraction,
            "budget_error": budget_error}


def demand_identity_evidence(cells, baselines, key_to_regime, *, tolerance=1e-9):
    """Compare every finite-capacity arm with a same-tape no-capacity baseline.

    A positive count is a real failure.  Merely checking that a set of observed values is
    non-empty cannot establish exogenous-demand invariance.
    """
    checks = 0
    mismatches = 0
    max_total_abs = 0.0
    max_claimant_abs = 0.0
    examples = []
    for key, by_share in cells.items():
        regime = key_to_regime[key]
        for share, rows in by_share.items():
            for seed_index, row in enumerate(rows):
                baseline = baselines[regime][seed_index]
                total_delta = abs(row["demanded_total"] - baseline["demanded_total"])
                claimant_delta = max(
                    abs(row["demanded_by_claimant"].get(claimant, 0.0)
                        - baseline["demanded_by_claimant"].get(claimant, 0.0))
                    for claimant in ("A", "B")
                )
                checks += 1
                max_total_abs = max(max_total_abs, total_delta)
                max_claimant_abs = max(max_claimant_abs, claimant_delta)
                if total_delta > tolerance or claimant_delta > tolerance:
                    mismatches += 1
                    if len(examples) < 5:
                        examples.append({"cell": key, "share": share, "seed_index": seed_index,
                                         "total_abs_delta": total_delta,
                                         "claimant_abs_delta": claimant_delta})
    return {"passed": mismatches == 0, "checks": checks, "mismatches": mismatches,
            "tolerance": tolerance, "max_total_abs_delta": max_total_abs,
            "max_claimant_abs_delta": max_claimant_abs, "examples": examples,
            "baseline": "same seed and regime with cssu_storage_capacity=None"}


def bootstrap_mean_ci(values, n_boot, rng):
    """Two-sided percentile bootstrap for a paired per-seed quantity."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("cannot bootstrap an empty vector")
    draws = rng.integers(0, values.size, size=(int(n_boot), values.size))
    boot = values[draws].mean(axis=1)
    return {"mean": float(values.mean()),
            "lcb95": float(np.percentile(boot, 2.5)),
            "ucb95": float(np.percentile(boot, 97.5)),
            "n": int(values.size), "n_boot": int(n_boot)}


def selected_guardrail_evidence(cells, baselines, argmax_by_regime, key_to_regime,
                                n_boot, rng):
    """Evaluate safety on the selected finite-capacity action against the same-tape null.

    The primary is allowed to move because capacity is the mechanism.  The safety question is
    narrower: did the selected capacity arm buy that movement by abandoning a claimant or
    increasing lost orders beyond the signed margins?
    """
    out = {}
    for key, share in argmax_by_regime.items():
        regime = key_to_regime[key]
        candidates = cells[key][share]
        references = baselines[regime]
        harm = {
            "worst_claimant_fill": np.asarray(
                [references[i]["worst_claimant_fill"] - candidates[i]["worst_claimant_fill"]
                 for i in range(len(candidates))], dtype=float),
            "lost_orders": np.asarray(
                [candidates[i]["lost_orders"] - references[i]["lost_orders"]
                 for i in range(len(candidates))], dtype=float),
        }
        cell = {"selected_share": float(share), "reference": "same-tape no-capacity arm",
                "metrics": {}}
        for metric in F6_METRICS:
            stat = bootstrap_mean_ci(harm[metric], n_boot, rng)
            delta = float(MARGINS[metric])
            stat.update({"delta": delta, "passes": bool(stat["ucb95"] <= delta)})
            cell["metrics"][metric] = stat
        cell["passed"] = all(v["passes"] for v in cell["metrics"].values())
        out[key] = cell
    return {"passed": all(v["passed"] for v in out.values()), "by_cell": out}


def boot_h_regime(cube: np.ndarray, n_boot: int, rng) -> dict:
    """H_regime over (regimes, actions, seeds), bootstrapped on the seed axis."""
    def stat(idx):
        # Average over SEEDS FIRST, then take the best action within each regime. Taking the max
        # per (regime, seed) instead lets the optimum vary with the seed, which is per-seed
        # clairvoyance and not H_regime at all -- it inflated the estimate roughly tenfold here.
        sub = cube[:, :, idx].mean(axis=2)          # (regimes, actions)
        return float(sub.max(axis=1).mean() - sub.mean(axis=0).max())
    n = cube.shape[2]
    draws = np.array([stat(rng.integers(0, n, n)) for _ in range(n_boot)])
    return {"H_regime": stat(np.arange(n)),
            "lcb95": float(np.percentile(draws, 2.5)),
            "ucb95": float(np.percentile(draws, 97.5))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--n-boot", type=int, default=5_000)
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--replay-of", required=True)
    ap.add_argument("--run-role", choices=("REPLAY",), required=True)
    ap.add_argument("--output", type=Path,
                    default=Path("results/headroom/estar_capacity_sweep/result.json"))
    args = ap.parse_args()
    seeds = [SEED_BASE + i for i in range(args.seeds)]
    started = time.perf_counter()

    # The no-capacity null arm, on the SAME tapes. f2 and f6 are undecidable without it: demand
    # invariance and abandonment harm are both statements ABOUT a comparator, and the first
    # version of this runner had neither -- f2 compared a set with itself and f6 was hardcoded.
    cells: dict[str, dict] = {}
    key_to_regime: dict[str, str] = {}
    baselines: dict[str, list[dict]] = {}
    for rname, (risks, freq, impact) in REGIMES.items():
        # Same-seed no-capacity arms are the reference for the instrument checks, not a new
        # scientific arm.
        baselines[rname] = [episode(s, risks, freq, impact, share_a=0.5, budget=None)
                            for s in seeds]
        print(f"  baseline {rname}: {len(seeds)} semillas sin capacidad")
    for budget in BUDGETS:
        for rname, (risks, freq, impact) in REGIMES.items():
            key = f"budget={budget:.0f}|{rname}"
            key_to_regime[key] = rname
            cells[key] = {sh: [episode(s, risks, freq, impact, share_a=sh, budget=budget)
                               for s in seeds] for sh in SHARES}
            print(f"  {key}: {len(SHARES)} repartos x {len(seeds)} semillas")

    rng = np.random.default_rng(20260803)
    results: dict[str, dict] = {}
    for budget in BUDGETS:
        keys = [k for k in cells if k.startswith(f"budget={budget:.0f}")]
        cube = np.array([[[cells[k][sh][i][PRIMARY] for i in range(len(seeds))]
                          for sh in SHARES] for k in keys])
        hr = boot_h_regime(cube, args.n_boot, rng)
        by_regime = {k: {sh: float(np.mean([cells[k][sh][i][PRIMARY] for i in range(len(seeds))]))
                         for sh in SHARES} for k in keys}
        argmax = {k: max(v, key=lambda sh: v[sh]) for k, v in by_regime.items()}
        results[f"budget={budget:.0f}"] = {
            "h_regime": hr, "argmax_by_regime": argmax, "mean_by_share": by_regime,
            "binding_fraction": {k: float(np.mean(
                [cells[k][sh][i]["binding_fraction"] for sh in SHARES
                 for i in range(len(seeds))])) for k in keys},
            "binding_fraction_by_share": {
                k: {sh: float(np.mean([cells[k][sh][i]["binding_fraction"]
                                       for i in range(len(seeds))])) for sh in SHARES}
                for k in keys},
            "spread": {k: float(max(v.values()) - min(v.values())) for k, v in by_regime.items()},
            "guardrails": {m: {k: float(np.mean([cells[k][sh][i][m] for sh in SHARES
                                                 for i in range(len(seeds))])) for k in keys}
                           for m in ("worst_claimant_fill", "lost_orders")},
            "mass": {k: {"demanded": float(np.mean([cells[k][sh][i]["demanded_total"]
                                                    for sh in SHARES for i in range(len(seeds))])),
                         "delivered": float(np.mean([cells[k][sh][i]["delivered_total"]
                                                     for sh in SHARES for i in range(len(seeds))])),
                         "blocked": float(np.mean([cells[k][sh][i]["blocked_total"]
                                                   for sh in SHARES for i in range(len(seeds))])),
                         "admitted": float(np.mean([cells[k][sh][i]["admitted_total"]
                                                    for sh in SHARES for i in range(len(seeds))])),
                         "budget_error_max": float(max(
                             abs(cells[k][sh][i]["budget_error"] or 0.0)
                             for sh in SHARES for i in range(len(seeds))))}
                     for k in keys}}

    argmax_all = {k: sh for r in results.values() for k, sh in r["argmax_by_regime"].items()}
    demand_ev = demand_identity_evidence(cells, baselines, key_to_regime)
    guardrail_ev = selected_guardrail_evidence(cells, baselines, argmax_all, key_to_regime,
                                               args.n_boot, rng)
    # The contract's liveness cell is (budget, regime). Keep per-share evidence and require at
    # least one live tested allocation in every such cell.
    binds = all(any(v > 0.0 for v in r["binding_fraction_by_share"][key].values())
                for r in results.values() for key in r["binding_fraction_by_share"])
    endpoint_moves = all(v > 1e-9 for r in results.values() for v in r["spread"].values())
    argmax_moves = any(len(set(r["argmax_by_regime"].values())) > 1 for r in results.values())
    material = all(r["h_regime"]["lcb95"] >= SESOI for r in results.values())

    falsifiers = {
        "f1_capacity_actually_binds": {
            "passed": bool(binds),
            "evidence": {"why_it_can_fail": "a cap that never fills makes everything downstream "
                                            "vacuous; the probe showed 3000 binds nothing at all",
                         "binding_fraction": {k: r["binding_fraction"]
                                              for k, r in results.items()},
                         "binding_fraction_by_share": {
                             k: r["binding_fraction_by_share"] for k, r in results.items()}}},
        "f2_mass_and_demand_are_untouched": {
            "passed": bool(demand_ev["passed"]),
            "evidence": {"why_it_can_fail": "a capacity that reduced DEMAND would be deleting the "
                                            "problem instead of constraining the solution; every "
                                            "capped arm is compared with its OWN no-capacity tape, "
                                            "total and per claimant, so a single drifting cell "
                                            "fails it",
                         **demand_ev,
                         "mass": {k: r["mass"] for k, r in results.items()}}},
        "f3_the_endpoint_responds_to_the_lever": {
            "passed": bool(endpoint_moves),
            "evidence": {"why_it_can_fail": "worst_claimant_fill is MEASURED blind to this "
                                            "mechanism (0.6791 at every budget); if flow_fill_rate "
                                            "is blind too, the sweep measures nothing",
                         "spread": {k: r["spread"] for k, r in results.items()}}},
        "f4_argmax_moves_with_regime": {
            "passed": bool(argmax_moves),
            "terminal_diagnostic": True,
            "evidence": {"why_it_can_fail": "a fixed optimum means constraint without decision",
                         "argmax": {k: r["argmax_by_regime"] for k, r in results.items()}}},
        "f5_no_fresh_seeds": custody_falsifier(seeds, replay_of=args.replay_of,
                                               exclude=args.output),
        "f6_no_gain_by_abandonment": {
            "passed": bool(guardrail_ev["passed"]),
            "evidence": {"why_it_can_fail": "a primary gain bought with lost orders is the "
                                            "measured failure mode of ret_excel in a new "
                                            "coordinate; harm at the SELECTED share is "
                                            "bootstrapped against the same-tape no-capacity arm "
                                            "and must satisfy UCB95(harm) <= delta",
                         **guardrail_ev,
                         "margins": MARGINS,
                         "guardrails": {k: r["guardrails"] for k, r in results.items()}}},
    }
    required_falsifiers = {
        "f1_capacity_actually_binds", "f2_mass_and_demand_are_untouched",
        "f3_the_endpoint_responds_to_the_lever", "f5_no_fresh_seeds",
        "f6_no_gain_by_abandonment",
    }
    # f4 is the scientific branch, not a software falsifier: false means the predeclared
    # terminal state CAPACITY_CONSTRAINS_WITHOUT_DECIDING, not an invalid instrument.
    falsifiers["all_passed"] = all(
        v["passed"] for k, v in falsifiers.items()
        if k in required_falsifiers and not v.get("not_applicable"))
    falsifiers["not_applicable"] = sorted(
        k for k, v in falsifiers.items()
        if k != "all_passed" and isinstance(v, dict) and v.get("not_applicable"))

    verdict = ("CAPACITY_OPENS_REGIME_DEPENDENT_HEADROOM" if material and argmax_moves
               else "ARGMAX_MOVES_WITHOUT_VALUE" if argmax_moves
               else "CAPACITY_CONSTRAINS_WITHOUT_DECIDING")

    for key, r in results.items():
        print(f"\n  === {key} ===")
        h = r["h_regime"]
        print(f"  H_regime {h['H_regime']:+.5f} [LCB95 {h['lcb95']:+.5f}]  (SESOI {SESOI})")
        for k, a in r["argmax_by_regime"].items():
            print(f"    argmax {k:<24} {a}   spread {r['spread'][k]:.5f}"
                  f"   binding {r['binding_fraction'][k]:.3f}")
    print(f"\n  veredicto: {verdict}\n")
    for name, f in falsifiers.items():
        if name in ("all_passed", "not_applicable") or not isinstance(f, dict):
            continue
        label = "NO APLICA" if f.get("not_applicable") else ("PASA" if f["passed"] else "FALLA")
        print(f"    {name:<42} {label}")

    payload = {
        "schema_version": "estar_capacity_sweep_v1",
        "claim_status": verdict if falsifiers["all_passed"] else "HALTED_FALSIFIER_FAILED",
        "scope": "DEVELOPMENT_SCREEN_ON_BURNED_TAPES_G1_NOT_GRANTED",
        "primary_metric": PRIMARY,
        "primary_choice_rationale": (
            "worst_claimant_fill did not respond in the burned-tape liveness probe: blocking "
            "1,306,164 rations left it at 0.6791 in that tested horizon. It is a cumulative "
            "delivered/demanded ratio, so this is an empirical contract result rather than a "
            "universal structural theorem. The endpoint was chosen against the mechanism before "
            "the sweep, not against a result."),
        "run_role": args.run_role,
        "replay_of": args.replay_of,
        "module_manifest": module_manifest(E_STAR_MODULES, script=__file__),
        "wired_capacity_nodes": ["cssu_a", "cssu_b"],
        "scope_caveat": (
            "The helper admits wdc/al/sb, but only CSSU A/B are connected to the DES, so this is "
            "a CSSU storage screen and not yet the per-node expansion Garrido asked for. And "
            "flow_fill_rate is a censored terminal ratio, not a pure timing metric: it is "
            "sensitive to delay only through what is still undelivered at the horizon."),
        "sesoi": SESOI, "margins": MARGINS, "budgets": list(BUDGETS), "shares": list(SHARES),
        "regimes": list(REGIMES), "seeds": seeds, "weeks": WEEKS,
        "results": results, "falsifiers": falsifiers,
        "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(payload, args.output, contract=args.contract,
                            reference=Path("results/sensitivity/contention_headroom_v1_1/result.json"),
                            stamp_extra={"run_role": args.run_role, "replay_of": args.replay_of,
                                         "module_manifest": payload["module_manifest"]})
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
