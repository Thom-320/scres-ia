#!/usr/bin/env python3
"""Measure the dwell inertia of the CSSU action, so G3c levels are derived and not assumed.

The frozen grid {1, 3, 7} was chosen before anything was measured, and the burned preflight found
that it has a DEAD MIDDLE: at 24 h activation latency and daily cadence, a dwell of 2, 3 or 4 days
holds zero actions even under maximal switching pressure. A factor with one null, one inert cell
and a single real treatment cannot support a three-level power calculation.

This script measures the quantity the levels should have been derived from in the first place: the
distribution of realised inter-switch spacing, and the smallest dwell that actually holds. It is an
INSTRUMENT CHARACTERISATION, not an experiment -- it reports no contrast, no policy comparison and
no headroom, and its output authorises nothing except the writing of an amendment.

Seeds: burned block 5_200_001-16, declared replay. No fresh roots.
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

from supply_chain.config import HOURS_PER_WEEK  # noqa: E402
from supply_chain.config import THESIS_FAITHFUL_PROTOCOL as P  # noqa: E402
from supply_chain.seed_custody import custody_falsifier, module_manifest  # noqa: E402
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402

R1R = ("R11", "R12", "R13", "R14")
R2R = ("R21", "R22", "R23", "R24")
REGIMES = {
    "R1r+R2r|base": (R1R + R2R, {}, {}),
    "R1r+R2r|freq3_imp2": (R1R + R2R, {"R23": 3.0}, {"R23": 2.0}),
}
#: Probed one day at a time: the threshold is what we are looking for, so it cannot be a level of
#: the grid we are trying to justify.
PROBE_DWELLS = tuple(range(1, 22))
SEED_BASE = 5_200_001
WEEKS, STEP_HOURS = 52, 24.0
HI, LO, NEUTRAL = 0.9, 0.1, 0.5
#: A level is a real treatment only if it holds actions on EVERY tape in EVERY regime. One tape
#: holding once is a coincidence, not a mechanism.
ALL_TAPES = 1.0
#: And holding is not enough. `dwell=4` holds on 8/8 tapes yet moves realised switching from 43.8
#: to 42.5 -- it registers the constraint without imposing it, which would put a level in the grid
#: that cannot produce a different policy problem. A treatment level must SUPPRESS switching.
MIN_SWITCH_SUPPRESSION = 0.10
MODULES = ("supply_chain/supply_chain.py", "supply_chain/g3c_temporal.py",
           "supply_chain/config.py", "supply_chain/seed_custody.py")


def _build(seed, risks, freq, impact, *, dwell):
    return MFSCSimulation(
        shifts=1, initial_buffers={n: 0.0 for n in ("op3_rm", "op5_rm", "op9_rations")},
        inventory_replenishment_period=0.0, seed=seed, horizon=float(WEEKS * HOURS_PER_WEEK),
        risks_enabled=True, risk_level="current", enabled_risks=set(risks),
        risk_frequency_multipliers_by_id=dict(freq) or None,
        risk_impact_multipliers_by_id=dict(impact) or None,
        cssu_topology_mode="split_v1", cssu_allocation_a=NEUTRAL,
        cssu_service_rule="FIFO_PARTIAL", cssu_reallocate_unused=False,
        order_fulfillment_mode="op9_linked", op9_dispatch_policy="fixed_clock_daily",
        strict_exogenous_crn=True, year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"], r14_defect_mode=P["r14_defect_mode"],
        cssu_min_dwell_days=float(dwell))


def probe(seed, risks, freq, impact, *, dwell, pressure) -> dict:
    """`pressure='maximal'` alternates every step -- the hardest any policy can ask. `'myopic'` is
    the incumbent policy class, which is what the levels will actually be used with."""
    sim = _build(seed, risks, freq, impact, dwell=dwell)
    switch_times: list[float] = []
    prev_alpha = float(sim.cssu_allocation_a)
    step, done = 0, False
    while not done:
        if pressure == "maximal":
            target = HI if step % 2 == 0 else LO
        else:
            unmet = {c: float(sim.cssu_demanded.get(c, 0.0))
                     - float(sim.cssu_delivered.get(c, 0.0)) for c in ("A", "B")}
            d = unmet["A"] - unmet["B"]
            target = HI if d > 0 else (LO if d < 0 else NEUTRAL)
        wants = abs(float(sim.cssu_allocation_a) - target) > 1e-9
        action = ({"cssu_allocation_a": float(target)}
                  if wants and sim._pending_cssu_action is None else None)
        _, _, done, _ = sim.step(action=action, step_hours=STEP_HOURS)
        now_alpha = float(sim.cssu_allocation_a)
        if abs(now_alpha - prev_alpha) > 1e-9:
            switch_times.append(float(sim.env.now))
            prev_alpha = now_alpha
        step += 1
    gaps = np.diff(np.asarray(switch_times)) / 24.0 if len(switch_times) > 1 else np.array([])
    return {"held": int(sim.cssu_blocked_by_dwell_count),
            "switches": int(sim.cssu_switch_count),
            "gap_days_median": float(np.median(gaps)) if gaps.size else float("nan"),
            "gap_days_p10": float(np.percentile(gaps, 10)) if gaps.size else float("nan"),
            "gap_days_min": float(gaps.min()) if gaps.size else float("nan")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--replay-of", required=True)
    ap.add_argument("--output", type=Path,
                    default=Path("results/headroom/g3c_dwell_inertia/result.json"))
    args = ap.parse_args()
    seeds = [SEED_BASE + i for i in range(args.seeds)]
    started = time.perf_counter()

    table: dict[str, dict] = {}
    for pressure in ("maximal", "myopic"):
        for rname, (risks, freq, impact) in REGIMES.items():
            for dwell in PROBE_DWELLS:
                rows = [probe(s, risks, freq, impact, dwell=dwell, pressure=pressure)
                        for s in seeds]
                held = [r["held"] for r in rows]
                table[f"{pressure}|{rname}|dwell={dwell}"] = {
                    "pressure": pressure, "regime": rname, "dwell": dwell,
                    "tapes_holding": int(sum(1 for h in held if h > 0)), "n_tapes": len(held),
                    "held_mean": float(np.mean(held)),
                    "switches_mean": float(np.mean([r["switches"] for r in rows])),
                    "gap_days_median": float(np.nanmedian([r["gap_days_median"] for r in rows])),
                    "gap_days_p10": float(np.nanmedian([r["gap_days_p10"] for r in rows]))}
            print(f"  {pressure} {rname}: {len(PROBE_DWELLS)} dwells x {len(seeds)} semillas")

    def cells_at(pressure: str, dwell: int) -> list[dict]:
        return [v for v in table.values()
                if v["pressure"] == pressure and v["dwell"] == dwell]

    def first_binding(pressure: str) -> int | None:
        """Smallest dwell that holds on EVERY tape in EVERY regime. Requiring every regime
        matters: a level that binds only under escalation is confounded with the regime."""
        for dwell in PROBE_DWELLS:
            cells = cells_at(pressure, dwell)
            if cells and all(c["tapes_holding"] >= ALL_TAPES * c["n_tapes"] for c in cells):
                return dwell
        return None

    def first_material(pressure: str) -> int | None:
        """Smallest dwell that both binds everywhere AND suppresses realised switching.

        Holding is necessary and not sufficient: a level can register the constraint without
        imposing it, and such a level adds a row to the grid without adding a decision problem.
        """
        base = {c["regime"]: c["switches_mean"] for c in cells_at(pressure, 1)}
        for dwell in PROBE_DWELLS:
            cells = cells_at(pressure, dwell)
            if not cells or not all(c["tapes_holding"] >= ALL_TAPES * c["n_tapes"]
                                    for c in cells):
                continue
            if all(c["switches_mean"] <= (1.0 - MIN_SWITCH_SUPPRESSION) * base[c["regime"]]
                   for c in cells):
                return dwell
        return None

    bind_max, bind_myopic = first_binding("maximal"), first_binding("myopic")
    material_myopic = first_material("myopic")
    suppression = {
        v["dwell"]: round(1.0 - v["switches_mean"] / cells_at("myopic", 1)[0]["switches_mean"], 4)
        for v in table.values() if v["pressure"] == "myopic" and v["regime"] == "R1r+R2r|base"}
    gap_median = float(np.nanmedian(
        [v["gap_days_median"] for v in table.values() if v["pressure"] == "myopic"]))

    # Levels derived from the measurement: the null, the first level that genuinely binds under
    # the policy class the contract will use, and its double as a separated second treatment.
    if material_myopic is None:
        proposed = None
        verdict = "NO_MATERIAL_LEVEL_WITHIN_PROBE_RANGE"
    else:
        proposed = [1, int(material_myopic), int(2 * material_myopic)]
        verdict = "LEVELS_DERIVED"

    falsifiers = {
        "f1_the_null_level_is_inert": {
            "passed": all(v["held_mean"] == 0.0 for v in table.values() if v["dwell"] == 1),
            "evidence": {"why_it_can_fail": "a null level that holds an action is not a "
                                            "regression null, and every G3c contrast would be "
                                            "measured against a treated baseline",
                         "dwell_1": {k: v["held_mean"] for k, v in table.items()
                                     if v["dwell"] == 1}}},
        "f2_some_level_in_range_binds": {
            "passed": bind_myopic is not None,
            "evidence": {"why_it_can_fail": "if nothing up to 21 days holds under the incumbent "
                                            "policy class, minimum dwell is not a usable factor "
                                            "at this cadence and G3c has no mechanism at all",
                         "first_binding_maximal_pressure": bind_max,
                         "first_binding_myopic_policy": bind_myopic}},
        "f3_the_dead_middle_is_confirmed_not_assumed": {
            "passed": bool(material_myopic is not None and material_myopic > 3),
            "evidence": {"why_it_can_fail": "if dwell=3 does bind here, the preflight's f2 failure "
                                            "was an artefact of one policy or one tape and the "
                                            "frozen grid needs no amendment at all",
                         "dwell_3": {k: v["tapes_holding"] for k, v in table.items()
                                     if v["dwell"] == 3}}},
        "f4_no_fresh_seeds": custody_falsifier(seeds, replay_of=args.replay_of,
                                               exclude=args.output),
    }
    falsifiers["all_passed"] = all(
        v["passed"] for k, v in falsifiers.items()
        if k != "all_passed" and isinstance(v, dict) and not v.get("not_applicable"))

    print(f"\n  primer dwell que ata (presion maxima): {bind_max}")
    print(f"  primer dwell que ata (politica miope):  {bind_myopic}")
    print(f"  primer dwell MATERIAL (suprime >=10% conmutaciones): {material_myopic}")
    print(f"  supresion de conmutaciones por nivel: {suppression}")
    print(f"  espaciado mediano entre conmutaciones:  {gap_median:.2f} dias")
    print(f"  niveles propuestos: {proposed}   ({verdict})\n")
    for name, f in falsifiers.items():
        if name == "all_passed" or not isinstance(f, dict):
            continue
        label = "NO APLICA" if f.get("not_applicable") else ("PASA" if f["passed"] else "FALLA")
        print(f"    {name:<44} {label}")

    payload = {
        "schema_version": "g3c_dwell_inertia_v1",
        "claim_status": verdict if falsifiers["all_passed"] else "HALTED_FALSIFIER_FAILED",
        "scope": "INSTRUMENT_CHARACTERISATION_ONLY_NO_CONTRAST_NO_ADJUDICATION",
        "run_role": "BURNED_INSTRUMENT_PROBE", "replay_of": args.replay_of,
        "module_manifest": module_manifest(MODULES, script=__file__),
        "probe_dwells": list(PROBE_DWELLS), "seeds": seeds, "weeks": WEEKS,
        "first_binding_maximal_pressure": bind_max,
        "first_binding_myopic_policy": bind_myopic,
        "first_material_myopic_policy": material_myopic,
        "switch_suppression_by_dwell": suppression,
        "min_switch_suppression": MIN_SWITCH_SUPPRESSION,
        "median_switch_gap_days": gap_median,
        "proposed_levels": proposed,
        "table": table, "falsifiers": falsifiers,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload["contract_path"] = str(args.contract)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"\n  -> {args.output}")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
