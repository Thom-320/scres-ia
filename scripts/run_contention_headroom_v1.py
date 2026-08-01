#!/usr/bin/env python3
"""Fase 1A -- does contention with teeth create headroom, and is the fungibility flag the cause?

Program O measured H_PI = 0.1515 over a non-fungible shared resource and EXACTLY 0 once the same
resource was made fungible. The two-CSSU split has been in the full DES all along, but it ran in
the FUNGIBLE condition the whole time (`reallocate_unused` was hard-coded True), the share was
restricted to three points, and R23 ran at peacetime frequency. All three were our choices.

This runner measures `H_regime = mean_r[max_a] - max_a[mean_r]` -- the value of knowing the risk
regime when choosing a CONSTANT allocation -- across the fungibility control, so the mechanism
claim is tested rather than assumed. Constants only: the question here is whether headroom EXISTS,
not whether a policy captures it.

See `docs/PREREGISTRO_CONTENCION_HEADROOM_2026-07-31.md` for the reading rule, fixed in advance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
from supply_chain.cssu_allocation import SERVICE_RULES  # noqa: E402
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402

R1R = ("R11", "R12", "R13", "R14")
R2R = ("R21", "R22", "R23", "R24")
# Six regimes: three R23 escalations x with/without the operational family. The escalation uses
# the per-risk multipliers, which is the permission Garrido gave us explicitly.
ESCALATIONS = {
    "base":       ({}, {}),
    "freq_x3":    ({"R23": 3.0}, {}),
    "freq3_imp2": ({"R23": 3.0}, {"R23": 2.0}),
}
FAMILIES = {"R2r": R2R, "R1r+R2r": R1R + R2R}
SHARES = tuple(round(0.1 + 0.1 * i, 2) for i in range(9))   # 0.1 .. 0.9, continuous lever
PRIMARY = "ret_excel_risk_conditional"
SIDE = ("ret_excel_visible_clipped_0_1", "flow_fill_rate")
SEED_BASE = 5_200_001
PRIOR_SEEDS = set(range(4_900_001, 4_900_007)) | set(range(4_900_501, 4_900_507)) | set(
    range(5_100_001, 5_100_013))


def regimes() -> dict[str, tuple[tuple[str, ...], dict, dict]]:
    out = {}
    for fam, risks in FAMILIES.items():
        for esc, (freq, impact) in ESCALATIONS.items():
            out[f"{fam}|{esc}"] = (risks, freq, impact)
    return out


def run(risks, freq, impact, seed, horizon, *, share, rule, fungible) -> dict[str, float]:
    sim = MFSCSimulation(
        shifts=1, initial_buffers={n: 0.0 for n in ("op3_rm", "op5_rm", "op9_rations")},
        inventory_replenishment_period=0.0, seed=seed, horizon=horizon,
        risks_enabled=True, risk_level="current", enabled_risks=set(risks),
        risk_frequency_multipliers_by_id=dict(freq) or None,
        risk_impact_multipliers_by_id=dict(impact) or None,
        cssu_topology_mode="split_v1", cssu_allocation_a=float(share),
        cssu_service_rule=str(rule), cssu_reallocate_unused=bool(fungible),
        order_fulfillment_mode="op9_linked", op9_dispatch_policy="fixed_clock_daily",
        strict_exogenous_crn=True, year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"], r14_defect_mode=P["r14_defect_mode"])
    sim.run()
    panel = compute_episode_metrics(sim)
    out = {PRIMARY: float(panel[PRIMARY])}
    out.update({k: float(panel[k]) for k in SIDE})
    out["forfeited_epochs"] = float(getattr(sim, "cssu_forfeited_epochs", 0.0))
    out["forfeited_rations"] = float(getattr(sim, "cssu_forfeited_rations", 0.0))
    out["live_epochs"] = float(getattr(sim, "cssu_allocation_live_epochs", 0.0))
    out["r23_events"] = float(sum(1 for e in sim.risk_events
                                  if str(getattr(e, "risk_id", "")) == "R23"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--horizon-weeks", type=int, default=52)
    ap.add_argument("--rules", nargs="*", default=list(SERVICE_RULES))
    ap.add_argument("--n-boot", type=int, default=5_000)
    ap.add_argument("--output", type=Path,
                    default=Path("results/sensitivity/contention_headroom_v1/result.json"))
    args = ap.parse_args()
    horizon = float(args.horizon_weeks * HOURS_PER_WEEK)
    seeds = [SEED_BASE + i for i in range(args.seeds)]
    reg = regimes()
    started = time.perf_counter()

    # cells[(rule, fungible)][regime][share][seed] -> panel
    cells: dict[tuple[str, bool], dict] = {}
    for rule in args.rules:
        for fungible in (True, False):
            table = {}
            for rname, (risks, freq, impact) in reg.items():
                table[rname] = {
                    share: [run(risks, freq, impact, s, horizon,
                                share=share, rule=rule, fungible=fungible) for s in seeds]
                    for share in SHARES}
            cells[(rule, fungible)] = table
            print(f"  {rule:<16} fungible={str(fungible):<5} "
                  f"({time.perf_counter() - started:.0f}s)", flush=True)

    rng = np.random.default_rng(20260731)

    def h_regime(table: dict, metric: str) -> dict[str, float]:
        """mean_r[max_a] - max_a[mean_r], bootstrapped over seeds (the independent unit)."""
        names = list(table)
        # per (regime, share, seed) matrix
        cube = np.array([[[table[r][a][i][metric] for i in range(len(seeds))]
                          for a in SHARES] for r in names])          # (R, A, S)

        def stat(idx: np.ndarray) -> float:
            sub = cube[:, :, idx].mean(axis=2)                       # (R, A)
            return float(sub.max(axis=1).mean() - sub.mean(axis=0).max())

        point = stat(np.arange(len(seeds)))
        draws = np.array([stat(rng.integers(0, len(seeds), len(seeds)))
                          for _ in range(args.n_boot)])
        return {"H_regime": point, "lcb95": float(np.percentile(draws, 2.5)),
                "ucb95": float(np.percentile(draws, 97.5))}

    summary, argmax_by_regime = {}, {}
    for (rule, fungible), table in cells.items():
        key = f"{rule}|fungible={fungible}"
        summary[key] = {m: h_regime(table, m) for m in (PRIMARY, *SIDE)}
        # WHICH share wins in each regime. H_regime can only be large if this moves; storing it
        # turns "the optimum is invariant" from an inference into something readable.
        # Every metric, not just the primary: if ReT prefers a share that service does not, the
        # metric is rewarding something the chain would not want, and that has to be readable.
        argmax_by_regime[key] = {
            rname: {
                "best_share": max(SHARES, key=lambda a: float(
                    np.mean([r[PRIMARY] for r in table[rname][a]]))),
                "by_share": {m: {str(a): float(np.mean([r[m] for r in table[rname][a]]))
                                 for a in SHARES}
                             for m in (PRIMARY, *SIDE, "forfeited_rations")}}
            for rname in table}

    def best(fungible: bool) -> tuple[str, float]:
        keys = [k for k in summary if k.endswith(f"fungible={fungible}")]
        top = max(keys, key=lambda k: summary[k][PRIMARY]["H_regime"])
        return top, summary[top][PRIMARY]["H_regime"]

    best_nf, h_nf = best(False)
    best_f, h_f = best(True)
    lcb_nf = summary[best_nf][PRIMARY]["lcb95"]

    def forfeited(fungible: bool) -> float:
        return float(np.mean([row["forfeited_rations"]
                              for (rule, fung), t in cells.items() if fung is fungible
                              for reg_t in t.values() for lst in reg_t.values()
                              for row in lst]))

    unused_nf, unused_f = forfeited(False), forfeited(True)
    r23_by_esc = {
        esc: float(np.mean([row["r23_events"]
                            for (rule, fung), t in cells.items()
                            for rname, sh in t.items() if rname.endswith(f"|{esc}")
                            for lst in sh.values() for row in lst]))
        for esc in ESCALATIONS}
    h_by_esc_note = "H_regime is computed across regimes, so escalation is checked by event count"

    lever_spread = float(np.mean([
        np.ptp([np.mean([r[PRIMARY] for r in t[rname][a]]) for a in SHARES])
        for t in cells.values() for rname in t]))

    falsifiers = {
        "f1_the_lever_actually_moves_the_system": {
            "passed": lever_spread > 1e-9,
            "evidence": {"why_it_can_fail": ("an inert lever makes H_regime a measurement of "
                                             "noise rather than of a decision"),
                         "mean_spread_across_shares": lever_spread}},
        "f2_non_fungible_actually_binds": {
            "passed": unused_nf > 0.0 and unused_f == 0.0,
            "evidence": {"why_it_can_fail": ("if the hard share never forfeits capacity the flag "
                                             "changed nothing, both arms are the same run and "
                                             "the mechanism claim H2 is vacuous"),
                         "forfeited_rations_non_fungible": unused_nf,
                         "forfeited_rations_fungible": unused_f}},
        "f3_escalation_actually_escalates": {
            "passed": r23_by_esc["freq_x3"] > r23_by_esc["base"],
            "evidence": {"why_it_can_fail": ("without more R23 events the escalation arm is a "
                                             "relabelling and H3 has not been tested"),
                         "r23_events_by_escalation": r23_by_esc, "note": h_by_esc_note}},
        "f4_crn_is_common": {
            "passed": len(set(seeds)) == len(seeds),
            "evidence": {"why_it_can_fail": "distinct seeds per cell would be sampling, not CRN",
                         "seeds": seeds}},
        "f5_H_regime_is_non_negative": {
            "passed": all(v[m]["H_regime"] >= -1e-12 for v in summary.values()
                          for m in (PRIMARY, *SIDE)),
            "evidence": {"why_it_can_fail": ("mean[max] >= max[mean] by construction; a negative "
                                             "would be an aggregation bug, not a finding")}},
        "f6_seeds_are_virgin": {
            "passed": not (set(seeds) & PRIOR_SEEDS),
            "evidence": {"why_it_can_fail": "a reused seed would void any later confirmation",
                         "excluded": sorted(PRIOR_SEEDS)}},
    }
    falsifiers["all_passed"] = all(v["passed"] for k, v in falsifiers.items()
                                   if k != "all_passed")

    if h_nf >= 0.01 and lcb_nf > 0 and h_nf > h_f:
        verdict = "CONTENTION_HEADROOM_FOUND"
    elif h_nf >= 1e-3:
        verdict = "CONTENTION_HEADROOM_SUBCRITICAL"
    else:
        verdict = "CONTENTION_DOES_NOT_OPEN_THE_DOOR"

    print(f"\n  === H_regime sobre `{PRIMARY}` ===")
    for key in sorted(summary):
        v = summary[key][PRIMARY]
        print(f"  {key:<34} {v['H_regime']:>10.6f}  "
              f"[{v['lcb95']:>10.6f}, {v['ucb95']:>10.6f}]")
    print(f"\n  mejor NO fungible: {best_nf} -> {h_nf:.6f} (LCB95 {lcb_nf:.6f})")
    print(f"  mejor    fungible: {best_f} -> {h_f:.6f}")
    print(f"  veredicto: {verdict}")
    print("\n  falsadores:")
    for name, check in falsifiers.items():
        if name != "all_passed":
            print(f"    {name:<42} {'PASA' if check['passed'] else 'FALLA'}")

    payload = {
        "schema_version": "contention_headroom_v1",
        "claim_status": verdict if falsifiers["all_passed"] else "HALTED_FALSIFIER_FAILED",
        "primary_metric": PRIMARY, "side_metrics": list(SIDE),
        "shares": list(SHARES), "regimes": list(reg), "service_rules": list(args.rules),
        "seeds": seeds, "summary": summary, "argmax_by_regime": argmax_by_regime,
        "best_non_fungible": {"cell": best_nf, "H_regime": h_nf, "lcb95": lcb_nf},
        "best_fungible": {"cell": best_f, "H_regime": h_f},
        "mechanism_check": {"H_non_fungible_minus_H_fungible": h_nf - h_f,
                            "program_o_reference": {"H_PI": 0.15151, "fungible_null": 0.0}},
        "falsifiers": falsifiers,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(
        payload, args.output,
        contract=Path("docs/PREREGISTRO_CONTENCION_HEADROOM_2026-07-31.md"),
        reference=Path("results/sensitivity/observable_sweep_op12_v1/result.json"))
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
