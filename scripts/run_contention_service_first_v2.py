#!/usr/bin/env python3
"""Seal `service_first_resilience_v2` as an endpoint by re-measuring the contention sweep with it.

The contention sweep is where `ret_excel` was measured preferring the split that delivers 50% of
rations over the one that delivers 80%. Re-running it under `v2` is the direct test of whether
the endpoint corrects that, and of whether any headroom survives the correction.

A lexicographic key admits no mean, so this reports two estimands and keeps them separate:

  1. the argmax per regime under the FULL key -- well defined, and it is the policy question;
  2. `H_regime` on `worst_claimant_fill`, the leading component, which is scalar.

Averaging the tuple would invent an exchange rate between components, which is exactly what the
lexicographic order exists to avoid. `f6` asserts it is never done.

See `docs/PREREGISTRO_METRICA_SERVICE_FIRST_V2_2026-08-01.md`, committed before this ran.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any
import warnings

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.config import HOURS_PER_WEEK  # noqa: E402
from supply_chain.config import THESIS_FAITHFUL_PROTOCOL as P  # noqa: E402
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.service_first_metric import (  # noqa: E402
    SERVICE_FIRST_V2_COMPONENTS, claimant_fills, service_first_key_v2)
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402

R1R = ("R11", "R12", "R13", "R14")
R2R = ("R21", "R22", "R23", "R24")
ESCALATIONS = {"base": {}, "freq_x3": {"R23": 3.0}, "freq3_imp2": {"R23": 3.0}}
IMPACTS = {"base": {}, "freq_x3": {}, "freq3_imp2": {"R23": 2.0}}
FAMILIES = {"R2r": R2R, "R1r+R2r": R1R + R2R}
SHARES = tuple(round(0.1 + 0.1 * i, 2) for i in range(9))
LEADING = "worst_claimant_fill"
SEED_BASE = 6_400_001


def seeds_used_by_sealed_artifacts(root: Path = Path("results"),
                                   exclude: Path | None = None) -> set[int]:
    used: set[int] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in {"seeds", "crn_seeds", "seed_block"} and isinstance(value, list):
                    used.update(int(x) for x in value if isinstance(x, (int, float)))
                else:
                    walk(value)
        elif isinstance(node, list):
            for value in node[:50]:
                walk(value)

    for path in root.glob("**/result.json"):
        if exclude is not None and path.resolve() == Path(exclude).resolve():
            continue
        try:
            walk(json.loads(path.read_text()))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            continue
    return used


def regimes() -> dict[str, tuple]:
    return {f"{fam}|{esc}": (risks, ESCALATIONS[esc], IMPACTS[esc])
            for fam, risks in FAMILIES.items() for esc in ESCALATIONS}


def episode(risks, freq, impact, share: float, seed: int, horizon: float) -> dict:
    sim = MFSCSimulation(
        shifts=1, initial_buffers={n: 0.0 for n in ("op3_rm", "op5_rm", "op9_rations")},
        inventory_replenishment_period=0.0, seed=seed, horizon=horizon,
        risks_enabled=True, risk_level="current", enabled_risks=set(risks),
        risk_frequency_multipliers_by_id=dict(freq) or None,
        risk_impact_multipliers_by_id=dict(impact) or None,
        cssu_topology_mode="split_v1", cssu_allocation_a=float(share),
        cssu_service_rule="FIFO_PARTIAL", cssu_reallocate_unused=False,
        order_fulfillment_mode="op9_linked", op9_dispatch_policy="fixed_clock_daily",
        strict_exogenous_crn=True, year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"], r14_defect_mode=P["r14_defect_mode"])
    sim.run()
    panel = compute_episode_metrics(sim)
    fills = claimant_fills(sim)
    key = service_first_key_v2(panel, fills)
    return {"key": [float(x) for x in key],
            LEADING: float(key[0]),
            "flow_fill_rate": float(panel["flow_fill_rate"]),
            "ret_excel_visible_clipped_0_1": float(panel["ret_excel_visible_clipped_0_1"]),
            "n_claimants": len(fills)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--horizon-weeks", type=int, default=52)
    ap.add_argument("--n-boot", type=int, default=5_000)
    ap.add_argument("--output", type=Path,
                    default=Path("results/metric_audit/contention_service_first_v2/result.json"))
    args = ap.parse_args()
    horizon = float(args.horizon_weeks * HOURS_PER_WEEK)
    seeds = [SEED_BASE + i for i in range(args.seeds)]
    reg = regimes()
    started = time.perf_counter()

    rows: dict[tuple[str, float], list[dict]] = {}
    for name, (risks, freq, impact) in reg.items():
        for share in SHARES:
            rows[(name, share)] = [episode(risks, freq, impact, share, s, horizon)
                                   for s in seeds]
        print(f"  {name} listo ({time.perf_counter() - started:.0f}s)", flush=True)

    def mean(name: str, share: float, key: str) -> float:
        return float(np.mean([r[key] for r in rows[(name, share)]]))

    def mean_key(name: str, share: float) -> tuple:
        """Component-wise mean across seeds, THEN compared lexicographically. The tuple itself
        is never collapsed into a scalar -- that is what f6 guards."""
        return tuple(np.mean([r["key"] for r in rows[(name, share)]], axis=0))

    argmax_v2 = {name: max(SHARES, key=lambda s: mean_key(name, s)) for name in reg}
    argmax_ret = {name: max(SHARES, key=lambda s: mean(name, s, "ret_excel_visible_clipped_0_1"))
                  for name in reg}
    argmax_fill = {name: max(SHARES, key=lambda s: mean(name, s, "flow_fill_rate"))
                   for name in reg}

    # H_regime on the LEADING component only, bootstrapped over seeds.
    rng = np.random.default_rng(20260801)
    cube = np.array([[[r[LEADING] for r in rows[(name, share)]] for share in SHARES]
                     for name in reg])                                   # (R, A, S)

    def stat(idx: np.ndarray) -> float:
        sub = cube[:, :, idx].mean(axis=2)
        return float(sub.max(axis=1).mean() - sub.mean(axis=0).max())

    h_point = stat(np.arange(len(seeds)))
    draws = np.array([stat(rng.integers(0, len(seeds), len(seeds))) for _ in range(args.n_boot)])
    h_regime = {"H_regime": h_point,
                "lcb95": float(np.percentile(draws, 2.5)),
                "ucb95": float(np.percentile(draws, 97.5)),
                "component": LEADING}

    argmax_moves = len(set(argmax_v2.values())) > 1

    # f6, done properly. The first version asserted `len(COMPONENTS) == 4`, which is a constant:
    # it would have passed even if the ranking were done by scalar sum. Fifth falsifier in three
    # days that checked a correlate. Two real checks instead:
    #   (a) POSITIVE CONTROL -- an independent reimplementation of lexicographic comparison must
    #       reproduce the production argmax exactly;
    #   (b) INJECTED DEFECT -- ranking by the scalar sum of the four components, which is what
    #       collapsing the tuple would amount to, and reporting whether it changes the answer.
    def argmax_independent_lex(name: str) -> float:
        best, best_key = None, None
        for share in SHARES:
            key = [float(x) for x in mean_key(name, share)]
            if best_key is None:
                best, best_key = share, key
                continue
            for mine, theirs in zip(key, best_key):        # explicit component-by-component
                if mine != theirs:
                    if mine > theirs:
                        best, best_key = share, key
                    break
        return best

    def argmax_scalar_sum(name: str) -> float:
        return max(SHARES, key=lambda s: float(sum(mean_key(name, s))))

    lex_matches = {n: argmax_independent_lex(n) == argmax_v2[n] for n in reg}
    scalar_defect = {n: argmax_scalar_sum(n) for n in reg}
    defect_changes_answer = any(scalar_defect[n] != argmax_v2[n] for n in reg)
    leading_spread = float(np.mean([np.ptp([mean(n, s, LEADING) for s in SHARES]) for n in reg]))
    prior_seeds = seeds_used_by_sealed_artifacts(exclude=args.output)
    claimants = {r["n_claimants"] for lst in rows.values() for r in lst}

    falsifiers = {
        "f1_v2_and_ret_disagree_in_every_regime": {
            "passed": all(argmax_v2[n] != argmax_ret[n] for n in reg),
            "evidence": {"why_it_can_fail": ("if v2 chose the same split as ret_excel it would be "
                                             "correcting nothing and this endpoint is pointless"),
                         "argmax_v2": argmax_v2, "argmax_ret": argmax_ret,
                         "regimes_disagreeing": sum(1 for n in reg
                                                    if argmax_v2[n] != argmax_ret[n]),
                         "regimes_total": len(reg)}},
        "f2_leading_component_binds": {
            "passed": leading_spread > 1e-6,
            "evidence": {"why_it_can_fail": ("a constant worst_claimant_fill would make v2 "
                                             "degenerate to its lower components and the leading "
                                             "one decorative"),
                         "mean_spread_across_shares": leading_spread}},
        "f3_claimant_partition_exists": {
            "passed": claimants == {2},
            "evidence": {"why_it_can_fail": ("with one claimant worst_claimant_fill IS "
                                             "flow_fill_rate and the experiment does not test "
                                             "what it claims"),
                         "claimant_counts_observed": sorted(claimants)}},
        "f4_H_regime_is_non_negative": {
            "passed": h_point >= -1e-12,
            "evidence": {"why_it_can_fail": "mean[max] >= max[mean] by construction"}},
        "f5_seeds_are_virgin": {
            "passed": not (set(seeds) & prior_seeds),
            "evidence": {"why_it_can_fail": ("hardcoding this to True is what let a real seed "
                                             "collision ship three days ago"),
                         "seeds": seeds, "collisions": sorted(set(seeds) & prior_seeds),
                         "prior_seeds_scanned": len(prior_seeds)}},
        "f6_ranking_is_actually_lexicographic": {
            "passed": all(lex_matches.values()),
            "evidence": {"why_it_can_fail": (
                             "the first version asserted len(COMPONENTS) == 4, a constant that "
                             "would pass even if the ranking were a scalar sum. This reproduces "
                             "the argmax with an INDEPENDENT component-by-component comparison; "
                             "any accidental collapse to a scalar makes them diverge"),
                         "independent_lex_matches_production": lex_matches,
                         "injected_defect_argmax_by_scalar_sum": scalar_defect,
                         "scalar_sum_defect_changes_the_answer": defect_changes_answer,
                         "note": ("if the injected defect does NOT change the answer, the "
                                  "ranking happens to be robust here and the lexicographic "
                                  "order is unfalsifiable FROM THE OUTPUT ALONE -- disclosed "
                                  "either way rather than hidden"),
                         "components": list(SERVICE_FIRST_V2_COMPONENTS)}},
    }
    falsifiers["all_passed"] = all(v["passed"] for k, v in falsifiers.items()
                                   if k != "all_passed")

    if argmax_moves and h_point >= 0.01 and h_regime["lcb95"] > 0:
        verdict = "SERVICE_FIRST_HEADROOM_FOUND"
    elif argmax_moves:
        verdict = "REGIME_DEPENDENT_BUT_BELOW_THE_BAR"
    else:
        verdict = "NEGATIVE_EXTENDS_TO_THE_SOUND_ENDPOINT"

    print(f"\n  === argmax por régimen ===")
    print(f"  {'régimen':<20}{'v2':>6}{'ReT':>8}{'fill':>8}")
    for name in reg:
        print(f"  {name:<20}{argmax_v2[name]:>6}{argmax_ret[name]:>8}{argmax_fill[name]:>8}")
    print(f"\n  H_regime sobre `{LEADING}`: {h_point:.6f} "
          f"[{h_regime['lcb95']:.6f}, {h_regime['ucb95']:.6f}]")
    print(f"  ¿el argmax de v2 se mueve entre regímenes? {argmax_moves}")
    print(f"\n  veredicto: {verdict}")
    print("\n  falsadores:")
    for name, check in falsifiers.items():
        if name != "all_passed":
            print(f"    {name:<40} {'PASA' if check['passed'] else 'FALLA'}")

    payload = {
        "schema_version": "contention_service_first_v2",
        "claim_status": verdict if falsifiers["all_passed"] else "HALTED_FALSIFIER_FAILED",
        "endpoint": "service_first_resilience_v2",
        "endpoint_components": list(SERVICE_FIRST_V2_COMPONENTS),
        "endpoint_status": ("CONTRACTED AND USED: this run is what makes v2 no longer "
                            "prospective. It remains a STIPULATED normative endpoint, never "
                            "evidence that abandonment is bad"),
        "shares": list(SHARES), "regimes": list(reg), "seeds": seeds,
        "argmax_by_regime": {"v2": argmax_v2, "ret_excel_clipped": argmax_ret,
                             "flow_fill_rate": argmax_fill},
        "argmax_moves_across_regimes": argmax_moves,
        "H_regime_leading_component": h_regime,
        "by_cell": {f"{n}|{s}": {"key_mean": list(mean_key(n, s)),
                                 LEADING: mean(n, s, LEADING),
                                 "flow_fill_rate": mean(n, s, "flow_fill_rate"),
                                 "ret_excel_visible_clipped_0_1":
                                     mean(n, s, "ret_excel_visible_clipped_0_1")}
                    for n in reg for s in SHARES},
        "falsifiers": falsifiers,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(
        payload, args.output,
        contract=Path("docs/PREREGISTRO_METRICA_SERVICE_FIRST_V2_2026-08-01.md"),
        reference=Path("results/metric_audit/service_first_v2/result.json"))
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
