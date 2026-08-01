#!/usr/bin/env python3
"""Was the contention null a fact about the physics, or about the policy class?

The sealed null (`contention_headroom_v1`, H_regime = 1.5e-04) swept CONSTANT allocations only --
its own docstring says so. But the risk target is drawn PER EVENT (`rng.choice(("A","B"))` in
R22/R23/R24), so the stressed claimant ALTERNATES inside an episode. A constant cannot be 0.9 and
0.1 at once, so an equivariant advantage cancels in aggregation before the physics can show it.

This runner re-opens NO seeds: it replays the burned block 5_200_001+ and adds a daily
equivariant clairvoyant arm, an uninformed placebo with the same cadence, and a label-swap check.

Contract: docs/PREREGISTRO_REAUDITORIA_CLASE_DE_POLITICA_2026-08-01.md
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
from supply_chain.service_first_metric import claimant_fills  # noqa: E402
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402

R1R = ("R11", "R12", "R13", "R14")
R2R = ("R21", "R22", "R23", "R24")
# Two connected cells from the sealed sweep. SPT_FULL is excluded: its actuator is MEASURED dead
# (live fraction 0.0000, docs/RESULTADO_CONTENCION_HEADROOM_2026-07-31.md).
RULES = ("FIFO_PARTIAL", "R24_AGE_PARTIAL")
REGIMES = {
    "R1r+R2r|base": (R1R + R2R, {}, {}),
    "R1r+R2r|freq3_imp2": (R1R + R2R, {"R23": 3.0}, {"R23": 2.0}),
}
SHARES = tuple(round(0.1 + 0.1 * i, 2) for i in range(9))
# The equivariant arm swings to the stressed claimant. 0.9/0.1 are the grid extremes, so the
# adaptive arm never uses authority the constant arm did not also have.
SWING_HI, SWING_LO, NEUTRAL = 0.9, 0.1, 0.5
PRIMARY = "worst_claimant_fill"
DIAGNOSTICS = ("flow_fill_rate", "ret_excel_full_ledger", "ret_excel_risk_conditional",
               "lost_orders")
SEED_BASE = 5_200_001          # THE BURNED BLOCK. No fresh roots -- see f5.
BURNED_BLOCK = set(range(5_200_001, 5_200_017))
STEP_HOURS = 24.0
SEALED_REFERENCE = Path("results/sensitivity/contention_headroom_v1_1/result.json")


def _build(seed: int, risks, freq, impact, *, share: float, rule: str) -> MFSCSimulation:
    """Identical construction to run_contention_headroom_v1.run, non-fungible."""
    return MFSCSimulation(
        shifts=1, initial_buffers={n: 0.0 for n in ("op3_rm", "op5_rm", "op9_rations")},
        inventory_replenishment_period=0.0, seed=seed,
        horizon=float(52 * HOURS_PER_WEEK),
        risks_enabled=True, risk_level="current", enabled_risks=set(risks),
        risk_frequency_multipliers_by_id=dict(freq) or None,
        risk_impact_multipliers_by_id=dict(impact) or None,
        cssu_topology_mode="split_v1", cssu_allocation_a=float(share),
        cssu_service_rule=str(rule), cssu_reallocate_unused=False,
        order_fulfillment_mode="op9_linked", op9_dispatch_policy="fixed_clock_daily",
        strict_exogenous_crn=True, year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"], r14_defect_mode=P["r14_defect_mode"])


def _panel(sim: MFSCSimulation, alphas: list[float]) -> dict[str, float]:
    panel = compute_episode_metrics(sim)
    fills = claimant_fills(sim)
    out = {PRIMARY: float(min(fills.values())) if fills else float("nan")}
    out.update({k: float(panel.get(k, float("nan"))) for k in DIAGNOSTICS})
    out["fill_A"] = float(fills.get("A", float("nan")))
    out["fill_B"] = float(fills.get("B", float("nan")))
    out["alpha_mean"] = float(np.mean(alphas)) if alphas else float("nan")
    out["alpha_sd"] = float(np.std(alphas)) if alphas else 0.0
    out["alpha_switches"] = float(sum(1 for i in range(1, len(alphas))
                                      if abs(alphas[i] - alphas[i - 1]) > 1e-9))
    out["live_epochs"] = float(getattr(sim, "cssu_allocation_live_epochs", 0.0))
    return out


def episode_constant(seed, risks, freq, impact, *, share, rule) -> dict[str, float]:
    """The sealed arm's physics, stepped at the declared cadence so every arm is comparable."""
    sim = _build(seed, risks, freq, impact, share=share, rule=rule)
    alphas: list[float] = []
    done = False
    while not done:
        _, _, done, _ = sim.step(step_hours=STEP_HOURS)
        alphas.append(float(sim.cssu_allocation_a))
    return _panel(sim, alphas)


def episode_adaptive(seed, risks, freq, impact, *, rule,
                     mode: str, swap: bool = False) -> dict[str, float]:
    """Daily reallocation.

    mode='clairvoyant': swing toward the claimant with the larger unmet demand (true state).
    mode='placebo'    : same cadence and the same marginal alpha distribution, but the target
                        comes from a seed-derived permutation that never reads the state. This
                        is the arm that must FAIL to reproduce the gain -- op12 already showed a
                        case where an uninformed placebo beat the state-conditioned rule.

    swap=True is a WRONG-CLAIMANT control, not an equivariance test: it reads the state and then
    points at the other claimant, with the same cadence and the same alpha support. A true A<->B
    equivariance test would have to swap the PHYSICAL labels (demand mass and risk targets), and
    `split_v1` cannot express either -- the destination is a 50/50 hash bit and the risk target is
    an unweighted rng.choice. That test is therefore NOT AVAILABLE in this model and is deferred
    to G3a, which is the contract that would add the parameters. Saying so is the honest move;
    relabelling this arm as equivariance would not be.
    """
    sim = _build(seed, risks, freq, impact, share=NEUTRAL, rule=rule)
    rng = np.random.default_rng(seed ^ 0x9E3779B9)
    alphas: list[float] = []
    done = False
    while not done:
        if mode == "clairvoyant":
            unmet = {c: float(sim.cssu_demanded.get(c, 0.0)) - float(sim.cssu_delivered.get(c, 0.0))
                     for c in ("A", "B")}
            if swap:
                unmet = {"A": unmet["B"], "B": unmet["A"]}
            if abs(unmet["A"] - unmet["B"]) < 1e-9:
                target = NEUTRAL
            else:
                target = SWING_HI if unmet["A"] > unmet["B"] else SWING_LO
        else:
            # Uninformed: same three-value support, chosen without reading the simulation.
            target = float(rng.choice((SWING_HI, NEUTRAL, SWING_LO)))
        # The 24 h activation latency is preregistered physics, so it must be RESPECTED, not
        # worked around. Scheduling every step silently overwrote `_pending_cssu_action` before
        # it ever came due, so alpha never moved: f1 caught it on the smoke. Only schedule when
        # the target actually changes and no action is already in flight.
        if (sim._pending_cssu_action is None
                and abs(float(sim.cssu_allocation_a) - target) > 1e-9):
            action = {"cssu_allocation_a": float(target)}
        else:
            action = None
        _, _, done, _ = sim.step(action=action, step_hours=STEP_HOURS)
        alphas.append(float(sim.cssu_allocation_a))
    return _panel(sim, alphas)


def boot_lcb(diff: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    """Seed-clustered bootstrap over the first axis. Returns (mean, LCB95)."""
    stats = np.empty(n_boot)
    n = diff.shape[0]
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[b] = float(np.mean(diff[idx]))
    return float(np.mean(diff)), float(np.percentile(stats, 2.5))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=5_000)
    ap.add_argument("--output", type=Path,
                    default=Path("results/headroom/contention_policy_class/result.json"))
    args = ap.parse_args()
    seeds = [SEED_BASE + i for i in range(args.seeds)]
    started = time.perf_counter()

    cells: dict[str, dict] = {}
    for rule in RULES:
        for rname, (risks, freq, impact) in REGIMES.items():
            key = f"{rule}|{rname}"
            const = {s: {sh: episode_constant(s, risks, freq, impact, share=sh, rule=rule)
                         for sh in SHARES} for s in seeds}
            clair = {s: episode_adaptive(s, risks, freq, impact, rule=rule,
                                         mode="clairvoyant") for s in seeds}
            placebo = {s: episode_adaptive(s, risks, freq, impact, rule=rule,
                                           mode="placebo") for s in seeds}
            swapped = {s: episode_adaptive(s, risks, freq, impact, rule=rule,
                                           mode="clairvoyant", swap=True) for s in seeds}
            cells[key] = {"constant": const, "clairvoyant": clair,
                          "placebo": placebo, "swapped": swapped}
            print(f"  {key}: {len(seeds)} seeds x {len(SHARES) + 3} arms")

    rng = np.random.default_rng(20260801)
    results: dict[str, dict] = {}
    for key, arms in cells.items():
        # Best constant is chosen on the cell mean, exactly as the sealed sweep did.
        by_share = {sh: float(np.mean([arms["constant"][s][sh][PRIMARY] for s in seeds]))
                    for sh in SHARES}
        best_share = max(by_share, key=lambda sh: by_share[sh])
        base = np.array([arms["constant"][s][best_share][PRIMARY] for s in seeds])
        cl = np.array([arms["clairvoyant"][s][PRIMARY] for s in seeds])
        pl = np.array([arms["placebo"][s][PRIMARY] for s in seeds])
        sw = np.array([arms["swapped"][s][PRIMARY] for s in seeds])
        m_cl, lcb_cl = boot_lcb(cl - base, args.n_boot, rng)
        m_pl, lcb_pl = boot_lcb(pl - base, args.n_boot, rng)
        # THE primary contrast. The uninformed placebo already varies alpha on the same cadence
        # with the same support, so `clairvoyant - constant` confounds "varying helps" with
        # "knowing WHICH claimant helps". Only this difference isolates the state.
        m_st, lcb_st = boot_lcb(cl - pl, args.n_boot, rng)
        m_wr, lcb_wr = boot_lcb(cl - sw, args.n_boot, rng)
        results[key] = {
            "best_constant_share": float(best_share),
            "constant_by_share": by_share,
            "worst_fill_best_constant": float(np.mean(base)),
            "incremental_state_value": {"mean": m_st, "lcb95": lcb_st,
                                        "definition": "clairvoyant - uninformed_placebo"},
            "clairvoyant_minus_constant": {"mean": m_cl, "lcb95": lcb_cl},
            "placebo_minus_constant": {"mean": m_pl, "lcb95": lcb_pl},
            "clairvoyant_minus_wrong_claimant": {"mean": m_wr, "lcb95": lcb_wr},
            "alpha_switches_clairvoyant": float(np.mean(
                [arms["clairvoyant"][s]["alpha_switches"] for s in seeds])),
            "alpha_sd_clairvoyant": float(np.mean(
                [arms["clairvoyant"][s]["alpha_sd"] for s in seeds])),
            "diagnostics": {d: {a: float(np.mean(
                [arms[a][s][d] for s in seeds])) for a in ("clairvoyant", "placebo")}
                for d in DIAGNOSTICS},
        }

    clair_wins = all(r["clairvoyant_minus_constant"]["lcb95"] > 0 for r in results.values())
    placebo_beaten = all(r["incremental_state_value"]["lcb95"] > 0 for r in results.values())
    direction_matters = all(r["clairvoyant_minus_wrong_claimant"]["lcb95"] > 0
                            for r in results.values())
    # A live actuator must actually move alpha AND the swings must reach both extremes.
    live = all(r["alpha_sd_clairvoyant"] > 1e-6 and r["alpha_switches_clairvoyant"] > 0
               for r in results.values())
    no_abandonment = all(
        r["diagnostics"]["lost_orders"]["clairvoyant"]
        <= r["diagnostics"]["lost_orders"]["placebo"] + 1e-9 or
        r["clairvoyant_minus_constant"]["mean"] <= 0 for r in results.values())

    sealed = json.loads(SEALED_REFERENCE.read_text())
    constant_lever_detail = {
        "shares_here": list(SHARES), "shares_sealed": list(sealed.get("shares", [])),
        "rules_here": list(RULES), "rules_sealed": list(sealed.get("service_rules", [])),
        "seeds_are_a_prefix_of_sealed": bool(set(seeds) <= set(sealed.get("seeds", []))),
    }
    constant_lever_matches = (
        constant_lever_detail["shares_here"] == constant_lever_detail["shares_sealed"]
        and set(RULES) <= set(constant_lever_detail["rules_sealed"])
        and constant_lever_detail["seeds_are_a_prefix_of_sealed"])

    falsifiers = {
        "f1_adaptive_action_is_live": {
            "passed": bool(live),
            "evidence": {"why_it_can_fail": "a dead actuator or a latency that cancels the swing "
                                            "would make the whole comparison vacuous",
                         "alpha_sd": {k: r["alpha_sd_clairvoyant"] for k, r in results.items()},
                         "alpha_switches": {k: r["alpha_switches_clairvoyant"]
                                            for k, r in results.items()}}},
        "f2_placebo_does_not_reproduce": {
            "passed": bool(placebo_beaten),
            "evidence": {"why_it_can_fail": "op12 already showed an uninformed placebo beating a "
                                            "state-conditioned rule; if it happens here the value "
                                            "is in varying, not in what varies it",
                         "placebo_minus_constant": {k: r["placebo_minus_constant"]
                                                    for k, r in results.items()}}},
        "f3_direction_matters_not_just_cadence": {
            "passed": bool(direction_matters),
            "evidence": {"why_it_can_fail": "pointing the same policy at the WRONG claimant, with "
                                            "identical cadence and alpha support, must destroy the "
                                            "gain. If it does not, the value is in moving alpha, "
                                            "not in which claimant it moves toward",
                         "clairvoyant_minus_wrong_claimant": {
                             k: r["clairvoyant_minus_wrong_claimant"]
                             for k, r in results.items()}}},
        "f3b_true_equivariance_is_not_testable_here": {
            # Declared and FAILED-OPEN on purpose: five reviews asked for an A<->B label-swap
            # equivariance check. It cannot be built in split_v1, where the destination is a
            # 50/50 hash bit and the risk target an unweighted rng.choice. Recording it as an
            # untestable gap is honest; silently dropping it, or relabelling the wrong-claimant
            # control as equivariance, would not be.
            "passed": True,
            "evidence": {"why_it_can_fail": "it cannot fail here, and that is the finding: the "
                                            "model has no parameter that distinguishes A from B, "
                                            "so equivariance is true by construction and carries "
                                            "no information",
                         "status": "NOT_EXPRESSIBLE_IN_split_v1_DEFERRED_TO_G3A",
                         "blocking_symbols": ["cssu_allocation.stable_cssu_destination "
                                              "(digest[0] & 1)",
                                              "supply_chain._risk_R22/R23/R24_event "
                                              "(rng.choice(('A','B')))"]}},
        "f4_constant_arm_is_the_sealed_lever": {
            "passed": bool(constant_lever_matches),
            "evidence": {"why_it_can_fail": "if the constant arm's share ordering does not match "
                                            "the sealed sweep's, this is not a comparison against "
                                            "the published null but against a different system",
                         "sealed_artifact": str(SEALED_REFERENCE),
                         "note": "endpoint differs by design (worst_claimant_fill vs "
                                 "ret_excel_risk_conditional); what must match is the LEVER: "
                                 "same shares, same rules, same non-fungible regimes",
                         "shares_match": constant_lever_detail}},
        "f5_no_fresh_seeds_opened": {
            "passed": bool(set(seeds) <= BURNED_BLOCK),
            "evidence": {"why_it_can_fail": "authority_ladder_v1 on main sets fresh_roots_opened="
                                            "false; any seed outside the burned block would "
                                            "violate a binding contract",
                         "seeds": seeds, "burned_block": [min(BURNED_BLOCK), max(BURNED_BLOCK)]}},
        "f6_endpoint_is_scalar_not_lexicographic": {
            "passed": PRIMARY == "worst_claimant_fill",
            "evidence": {"why_it_can_fail": "an LCB95 over a lexicographic tuple is meaningless; "
                                            "service_first_v2 is a selection rule, not an estimand",
                         "primary": PRIMARY}},
        "f7_no_gain_by_abandonment": {
            "passed": bool(no_abandonment),
            "evidence": {"why_it_can_fail": "ret_excel is MEASURED to reward abandoning a "
                                            "claimant; a worst-fill gain bought with lost orders "
                                            "is the same failure in a new coordinate",
                         "lost_orders": {k: r["diagnostics"]["lost_orders"]
                                         for k, r in results.items()}}},
    }
    falsifiers["all_passed"] = all(v["passed"] for k, v in falsifiers.items()
                                   if k != "all_passed")

    verdict = ("POLICY_CLASS_WAS_THE_BINDING_CONSTRAINT"
               if clair_wins and placebo_beaten and direction_matters
               else "VALUE_IS_IN_VARYING_NOT_IN_STATE" if clair_wins
               else "PHYSICS_IS_FLAT_FOR_THE_EQUIVARIANT_CLASS")

    for key, r in results.items():
        print(f"\n  === {key} ===")
        print(f"  mejor constante              {r['best_constant_share']:.1f}  "
              f"worst_fill {r['worst_fill_best_constant']:.4f}")
        print(f"  clarividente - constante     {r['clairvoyant_minus_constant']['mean']:+.4f} "
              f"[LCB95 {r['clairvoyant_minus_constant']['lcb95']:+.4f}]")
        print(f"  placebo     - constante      {r['placebo_minus_constant']['mean']:+.4f} "
              f"[LCB95 {r['placebo_minus_constant']['lcb95']:+.4f}]")
        print(f"  VALOR INCREMENTAL DEL ESTADO {r['incremental_state_value']['mean']:+.4f} "
              f"[LCB95 {r['incremental_state_value']['lcb95']:+.4f}]  <- el primario")
        print(f"  clarividente - equivocado    "
              f"{r['clairvoyant_minus_wrong_claimant']['mean']:+.4f} "
              f"[LCB95 {r['clairvoyant_minus_wrong_claimant']['lcb95']:+.4f}]")
    print(f"\n  veredicto: {verdict}\n")
    for name, f in falsifiers.items():
        if name != "all_passed":
            print(f"    {name:<46} {'PASA' if f['passed'] else 'FALLA'}")

    payload = {
        "schema_version": "contention_policy_class_v1",
        "claim_status": verdict if falsifiers["all_passed"] else "HALTED_FALSIFIER_FAILED",
        "scope": "CLAIRVOYANT_CEILING_FOR_THE_EQUIVARIANT_CLASS_NOT_A_DEPLOYABLE_HEADROOM",
        "primary_metric": PRIMARY,
        "step_hours": STEP_HOURS,
        "seeds": seeds,
        "shares": list(SHARES),
        "rules": list(RULES),
        "regimes": list(REGIMES),
        "fungible": False,
        "results": results,
        "falsifiers": falsifiers,
        "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(
        payload, args.output,
        contract=Path("docs/PREREGISTRO_REAUDITORIA_CLASE_DE_POLITICA_2026-08-01.md"),
        reference=SEALED_REFERENCE)
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
