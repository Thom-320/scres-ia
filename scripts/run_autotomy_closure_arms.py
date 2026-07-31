#!/usr/bin/env python3
"""Executes `docs/PREREGISTRO_CIERRE_AUTOTOMIA_2026-07-31.md`: A / F / FD / FDB.

    A    constant 54 h        predicate `le`             (status quo)
    F    freight waves        predicate `le`
    FD   freight waves + d    predicate `le`
    FDB  freight waves + d    predicate `band`, tol 0.05 h

`Re(APj)` is the one driver of Garrido's Fig. 4 that does not exist in our 90-configuration
table. The freight-wave arm alone does not close it -- measured, it overshoots 147x -- because
our lattice lands 60.7% of orders on `CTj = 48.0` exactly while his floor is RARE (98 rows over
~26,000). The problem is the floor's INCIDENCE, so `delta ~ U(0, HOURS_PER_SHIFT)` enters as the
only thing that can make it rare, and the band predicate as the only thing that can then let it
fire at all.

The scoring rule, the prediction and the single declared parameter are in the contract. Every
falsifier below states why it can fail, and f2 is the one that matters: it measures the floor
incidence on BOTH sides with the same rule, so the mechanism I claim is testable rather than
asserted.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from supply_chain.arm_runner import (  # noqa: E402
    aggregate, build_reference, episode_moments, run_falsifiers, scored_orders,
    seal_and_write, verdict,
)
from supply_chain.config import HOURS_PER_WEEK, LEAD_TIME_PROMISE  # noqa: E402
from supply_chain.config import GARRIDO_FULFILLMENT_DELAY_HOURS as DELAY  # noqa: E402
from supply_chain.config import THESIS_FAITHFUL_PROTOCOL as P  # noqa: E402
from supply_chain.fidelity_moments import EPSILON, MOMENT_NAMES  # noqa: E402
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402

FAMILIES = {"R1r": ("R11", "R12", "R13", "R14"), "R2r": ("R21", "R22", "R23", "R24")}
ROOTS = tuple(3_700_001 + i for i in range(12))
BAND_TOLERANCE = 0.05     # read off HIS autotomy rows: CTj - LT in [0.0074, 0.048]
ARMS = {
    "A_constant_le":  {"transit": "constant", "delta": "off", "pred": "le", "tol": 0.0},
    "F_waves_le":     {"transit": "freight_waves", "delta": "off", "pred": "le", "tol": 0.0},
    "FD_waves_delta_le": {"transit": "freight_waves", "delta": "shift_uniform",
                          "pred": "le", "tol": 0.0},
    "FDB_waves_delta_band": {"transit": "freight_waves", "delta": "shift_uniform",
                             "pred": "band", "tol": BAND_TOLERANCE},
}
PRIMARY, PROTECTED = "autotomy_share", "ret_mean"
REFERENCE = Path("results/metric_audit/fidelity_reference_v4/result.json")


def run_episode(*, family: str, seed: int, horizon: float, arm: dict) -> MFSCSimulation:
    sim = MFSCSimulation(
        shifts=1,
        initial_buffers={n: 0.0 for n in ("op3_rm", "op5_rm", "op9_rations")},
        inventory_replenishment_period=0.0, seed=seed, horizon=horizon,
        risks_enabled=True, risk_level="current", enabled_risks=set(FAMILIES[family]),
        risk_overrides={r: "increased" for r in FAMILIES[family]},
        fulfillment_transit_mode=str(arm["transit"]),
        fulfillment_delta_mode=str(arm["delta"]),
        autotomy_predicate=str(arm["pred"]),
        autotomy_tolerance_hours=float(arm["tol"]),
        strict_exogenous_crn=True, year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"], r14_defect_mode=P["r14_defect_mode"])
    sim.step(action=None, step_hours=horizon)
    return sim


def garrido_floor_incidence(workbook_dir: Path) -> dict:
    """His floor incidence, under the SAME band rule applied to our orders.

    Without this the mechanism claim ("his floor is rare, ours is modal") is an assertion. It
    reuses `build_fidelity_reference_v4.read_sheet`, the only reader that finds CF2.
    """
    from build_fidelity_reference_v4 import WORKBOOKS, read_sheet

    lt = float(LEAD_TIME_PROMISE)
    out: dict[str, dict] = {}
    for family, sheets in (("R1r", range(1, 11)), ("R2r", range(11, 21))):
        n_tot = n_floor = n_auto = 0
        minimum = float("inf")
        for i in sheets:
            for wb in WORKBOOKS:
                path = workbook_dir / wb
                if not path.exists():
                    continue
                d = read_sheet(path, f"CF{i}")
                if d is None or not len(d):
                    continue
                ct = [float(v) for v in d["CTj"].tolist() if v == v]
                ap = [float(v) for v in d["APj"].fillna(0.0).tolist()]
                n_tot += len(ct)
                minimum = min([minimum] + ct)
                for c, a in zip(ct, ap):
                    if lt <= c <= lt + BAND_TOLERANCE:
                        n_floor += 1
                        if a > 0.0:
                            n_auto += 1
                break
        out[family] = {"n_rows": n_tot, "n_in_floor_band": n_floor,
                       "n_in_band_with_autotomy": n_auto,
                       "floor_share": (n_floor / n_tot) if n_tot else 0.0,
                       "min_ctj": minimum,
                       "rule": f"CTj in [{lt}, {lt + BAND_TOLERANCE}]"}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contract", type=Path,
                    default=Path("docs/PREREGISTRO_CIERRE_AUTOTOMIA_2026-07-31.md"))
    ap.add_argument("--reference", type=Path, default=REFERENCE)
    ap.add_argument("--workbook-dir", type=Path, default=Path.home() / "Downloads")
    ap.add_argument("--roots", nargs="+", type=int, default=list(ROOTS))
    ap.add_argument("--horizon-weeks", type=int, default=52)
    ap.add_argument("--output", type=Path,
                    default=Path("results/metric_audit/autotomy_closure_arms_v1/result.json"))
    args = ap.parse_args()

    horizon = float(args.horizon_weeks * HOURS_PER_WEEK)
    ref_blob = json.loads(args.reference.read_text())
    lt = float(LEAD_TIME_PROMISE)
    started = time.perf_counter()

    per_arm: dict[str, dict[str, list]] = {}
    floor: dict[str, dict] = {}
    below_lt: dict[str, int] = {}
    band_violation: dict[str, int] = {}
    min_ctj: dict[str, float] = {}
    for arm, spec in ARMS.items():
        per_arm[arm] = {}
        n_tot = n_floor = 0
        below = viol = 0
        lowest = float("inf")
        for family in FAMILIES:
            rows = []
            for seed in args.roots:
                sim = run_episode(family=family, seed=seed, horizon=horizon, arm=spec)
                rows.append(episode_moments(sim))
                for o in scored_orders(sim):
                    if o.CTj is None:
                        continue
                    ct = float(o.CTj)
                    n_tot += 1
                    lowest = min(lowest, ct)
                    if ct < lt - 1e-9:
                        below += 1
                    if lt <= ct <= lt + BAND_TOLERANCE:
                        n_floor += 1
                    if (float(getattr(o, "APj", 0.0) or 0.0) > 0.0
                            and ct - lt > float(spec["tol"]) + 1e-9):
                        viol += 1
            per_arm[arm][family] = rows
        floor[arm] = {"n_orders": n_tot, "n_in_floor_band": n_floor,
                      "floor_share": (n_floor / n_tot) if n_tot else 0.0}
        below_lt[arm], band_violation[arm], min_ctj[arm] = below, viol, lowest
        print(f"  {arm} ({time.perf_counter() - started:.0f}s)", flush=True)

    his = garrido_floor_incidence(args.workbook_dir)
    reference = {f: build_reference(ref_blob, f) for f in FAMILIES}
    cells = {f: {a: aggregate(per_arm[a][f], reference[f]) for a in ARMS} for f in FAMILIES}
    verdicts = {f: verdict(cells[f]) for f in FAMILIES}

    def moment(arm: str, family: str, name: str) -> float:
        return cells[family][arm]["moments"][name]

    def dk(arm: str, family: str, name: str) -> float:
        return cells[family][arm]["discrepancies"][name]

    checks = {
        "f1_status_quo_block_is_intact": lambda: (
            all(moment("A_constant_le", f, "autotomy_share") == 0.0 for f in FAMILIES)
            and abs(min_ctj["A_constant_le"] - float(DELAY)) < 1e-9,
            {"why_it_can_fail": ("if the shipped default moved, every other arm is being "
                                 "compared against something else"),
             "autotomy_share": {f: moment("A_constant_le", f, "autotomy_share")
                                for f in FAMILIES},
             "min_ctj": min_ctj["A_constant_le"], "shipped_constant": float(DELAY)}),
        "f2_his_floor_is_rare_and_ours_is_modal": lambda: (
            all(his[f]["floor_share"] < 0.05 for f in FAMILIES)
            and floor["F_waves_le"]["floor_share"] > 0.5,
            {"why_it_can_fail": ("this is the MECHANISM claim. If his floor were also modal, "
                                 "the diagnosis that the freight-wave arm fails on incidence "
                                 "rather than offset is simply wrong"),
             "his": his, "ours": floor, "same_rule": f"CTj in [{lt}, {lt + BAND_TOLERANCE}]"}),
        "f3_no_order_completes_below_the_lead_time": lambda: (
            all(v == 0 for v in below_lt.values()),
            {"why_it_can_fail": "the delta draw could subtract instead of add",
             "violations": below_lt, "min_ctj": min_ctj}),
        "f4_band_predicate_fires_only_inside_its_band": lambda: (
            all(v == 0 for v in band_violation.values()),
            {"why_it_can_fail": ("if an order outside the tolerance carries APj > 0 the "
                                 "predicate is reading something other than CTj - LT"),
             "violations": band_violation, "tolerance_by_arm": {a: ARMS[a]["tol"]
                                                                for a in ARMS}}),
        "f5_epsilon_stable": lambda: (
            all(verdicts[f]["set_is_epsilon_stable"] for f in FAMILIES),
            {"why_it_can_fail": "a non-dominated set that moves with epsilon is not a result",
             "stability": {f: verdicts[f]["epsilon_stability"] for f in FAMILIES}}),
    }
    fals = run_falsifiers(checks)

    # The contract's acceptance rule, applied literally.
    acceptance = {}
    for arm in ARMS:
        if arm == "A_constant_le":
            continue
        improves = all(dk(arm, f, PRIMARY) < dk("A_constant_le", f, PRIMARY)
                       for f in FAMILIES)
        protected = all(dk(arm, f, PROTECTED) - dk("A_constant_le", f, PROTECTED) <= EPSILON
                        for f in FAMILIES)
        worse = {f"{f}.{m}": float(dk(arm, f, m) - dk("A_constant_le", f, m))
                 for f in FAMILIES for m in MOMENT_NAMES
                 if dk(arm, f, m) - dk("A_constant_le", f, m) > EPSILON}
        acceptance[arm] = {
            "improves_autotomy_dk": bool(improves),
            "ret_mean_within_epsilon": bool(protected),
            "moments_worse_beyond_epsilon": worse,
            "qualifies": bool(improves and protected and not worse and fals["all_passed"]),
        }
    adoptable = [a for a, row in acceptance.items() if row["qualifies"]]

    print(f"\n  === autotomy_share (Garrido R1r {reference['R1r'][PRIMARY].mean:.6f}, "
          f"R2r {reference['R2r'][PRIMARY].mean:.6f}) ===")
    print(f"  {'brazo':<24}{'R1r':>12}{'d_k':>8}{'R2r':>12}{'d_k':>8}"
          f"{'ret_mean R1r':>14}{'d_k':>8}")
    for arm in ARMS:
        print(f"  {arm:<24}{moment(arm, 'R1r', PRIMARY):>12.6f}{dk(arm, 'R1r', PRIMARY):>8.2f}"
              f"{moment(arm, 'R2r', PRIMARY):>12.6f}{dk(arm, 'R2r', PRIMARY):>8.2f}"
              f"{moment(arm, 'R1r', PROTECTED):>14.6f}{dk(arm, 'R1r', PROTECTED):>8.2f}")
    print("\n  falsadores:")
    for name, check in fals.items():
        if name != "all_passed":
            print(f"    {name:<44} {'PASA' if check['passed'] else 'FALLA'}")
    print(f"\n  conjuntos no dominados: "
          f"{ {f: verdicts[f]['non_dominated_set'] for f in FAMILIES} }")
    print(f"  adoptables: {adoptable or 'ninguno'}")

    payload = {
        "schema_version": "autotomy_closure_arms_v1",
        "claim_status": ("DEVELOPMENT_PREREGISTERED_AUTOTOMY_CLOSURE"
                         if fals["all_passed"] else "HALTED_FALSIFIER_FAILED"),
        "arms": ARMS, "roots": list(args.roots),
        "band_tolerance_hours": BAND_TOLERANCE,
        "band_tolerance_provenance": ("read off HIS autotomy rows (CTj - LT in "
                                      "[0.0074, 0.048]); one declared parameter, never fitted "
                                      "to our output"),
        "epsilon": EPSILON,
        "results": cells if fals["all_passed"] else None,
        "verdicts": verdicts if fals["all_passed"] else None,
        "results_withheld_note": (None if fals["all_passed"] else
                                  "moments computed but NOT reported: a falsifier failed"),
        "acceptance": {"per_arm": acceptance, "adoptable": adoptable},
        "floor_incidence_ours": floor, "floor_incidence_his": his,
        "min_ctj_by_arm": min_ctj, "falsifiers": fals,
        "per_episode": per_arm, "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(payload, args.output, contract=args.contract,
                            reference=args.reference, stamp_extra={"arms": sorted(ARMS)})
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if fals["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
