#!/usr/bin/env python3
"""Is the 3.75x procurement over-order ours, or Garrido's?

The Program V port and the scarcity sweep both closed at exactly zero because raw material is never
the binding constraint: Op2 contracts far more than the chain consumes, so which supplier you pick
cannot matter. Before proposing to shrink the contracted volume -- which would be outcome
engineering if the volume is a source fact -- the number has to be attributed.

TWO READINGS OF ONE SENTENCE. The thesis says Op2 handles 190,000 units of each raw material
monthly. `docs/THESIS_INTERPRETATION_DECISIONS_2026-06-24.md` D5 records that we read this as PER
raw material and marks the choice CHOSEN-AMBIGUOUS. The alternative -- 190,000 as the TOTAL across
all twelve -- is equally consistent with the sentence and is tested here by dividing by twelve and
changing nothing else.

Development verification. No seeds beyond one already-burned tape, no learner, no claim about
headroom.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.arm_runner import seal_and_write                              # noqa: E402
from supply_chain import falsifiers as F                                        # noqa: E402
from supply_chain.config import NUM_RAW_MATERIALS, OPERATIONS                   # noqa: E402
from supply_chain.supply_chain import MFSCSimulation                            # noqa: E402

SEED, WEEKS, HOURS_PER_WEEK = 8600001, 32, 168.0
OUT = Path("results/procurement_overorder_source/result.json")
CONTRACT = Path("docs/THESIS_INTERPRETATION_DECISIONS_2026-06-24.md")
#: Thesis demand: discrete uniform 2,400-2,600 rations/day, six days a week.
RATIONS_PER_DAY, DAYS_PER_WEEK = 2500.0, 6


def episode(op2_q: float) -> dict:
    sim = MFSCSimulation(seed=SEED, horizon=WEEKS * HOURS_PER_WEEK,
                         risks_enabled=True, risk_level="current")
    sim.params["op2_q"] = float(op2_q)
    sim.run()
    return {"service": float(sim.total_delivered) / max(float(sim.total_demanded), 1.0),
            "delivered_raw": float(sim.total_external_raw_material),
            "consumed_raw": float(sim.total_raw_material_consumed),
            "on_hand_close": float(sim.raw_material_wdc.level)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=Path(__file__).resolve().parent.parent / OUT)
    args = ap.parse_args()

    # Arithmetic straight from the published parameters, before any simulation.
    op2_q, op2_rop = float(OPERATIONS[2]["q"]), float(OPERATIONS[2]["rop"])
    op3_q = float(OPERATIONS[3]["q"])
    demand_raw_week = RATIONS_PER_DAY * DAYS_PER_WEEK * NUM_RAW_MATERIALS
    supply_raw_week = op2_q * NUM_RAW_MATERIALS / (op2_rop / HOURS_PER_WEEK)
    op3_raw_week = op3_q * NUM_RAW_MATERIALS
    supply_ratio = supply_raw_week / demand_raw_week
    op3_ratio = op3_raw_week / demand_raw_week

    in_use = episode(op2_q)
    alternative = episode(op2_q / NUM_RAW_MATERIALS)

    checks = {
        "f1_overorder_is_implied_by_published_parameters": F.ge(
            supply_ratio, 3.0,
            "if Garrido's own Op2 quantity and reorder period implied supply near demand, the "
            "over-order would be an artifact of our reconstruction and shrinking it would be a "
            "repair rather than an intervention"),
        "f2_distribution_is_sized_to_demand": F.lt(
            abs(op3_ratio - 1.0), 0.10,
            "if Op3 were also oversized, the ratio would be a global scaling error of ours rather "
            "than a deliberate asymmetry between procurement and distribution"),
        "f3_alternative_reading_starves_the_chain": F.lt(
            alternative["on_hand_close"], 1.0,
            "if reading 190,000 as the total across twelve raw materials ALSO left stock on hand, "
            "both readings would be viable and the attribution would be undecidable here"),
        "f4_alternative_reading_loses_service": F.gt(
            in_use["service"] - alternative["service"], 0.10,
            "if the alternative served as well, it could not be rejected on physical grounds and "
            "the choice would rest on the ambiguous sentence alone"),
        "f5_reading_in_use_never_binds": F.gt(
            in_use["on_hand_close"], 0.0,
            "this is the fact the Program V port ran into; if raw material did bind under the "
            "reading in use, the zero contrasts would need a different explanation"),
    }
    checks["d1_ambiguity_is_declared"] = F.disclosure(
        "D5 marks the per-raw-material reading CHOSEN-AMBIGUOUS. This verification does not remove "
        "the ambiguity from the sentence; it shows the alternative is refuted by the physics, "
        "which is a weaker and more honest claim",
        evidence={"decision": "D5", "contract": str(CONTRACT)})
    summary = F.summarise(checks)

    attributed = all(checks[k]["passed"] for k in
                     ("f1_overorder_is_implied_by_published_parameters",
                      "f3_alternative_reading_starves_the_chain",
                      "f4_alternative_reading_loses_service"))
    status = ("OVERORDER_IS_SOURCE_IMPLIED_NOT_A_RECONSTRUCTION_ARTIFACT" if attributed
              else "OVERORDER_ATTRIBUTION_UNRESOLVED")

    payload = {
        "schema_version": "procurement_overorder_source_v1", "claim_status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_role": "DEVELOPMENT", "scope": "SOURCE_ATTRIBUTION_ONE_BURNED_TAPE_NO_LEARNER",
        "endpoint": "not applicable -- attributes a parameter, measures no headroom",
        "seeds": [SEED],
        "published_parameters": {"op2_q_per_rm": op2_q, "op2_rop_hours": op2_rop,
                                 "op3_q_per_rm": op3_q, "n_raw_materials": NUM_RAW_MATERIALS,
                                 "rations_per_day": RATIONS_PER_DAY,
                                 "days_per_week": DAYS_PER_WEEK},
        "derived": {"demand_raw_per_week": demand_raw_week,
                    "op2_supply_raw_per_week": supply_raw_week,
                    "op3_raw_per_week": op3_raw_week,
                    "procurement_over_demand": supply_ratio,
                    "distribution_over_demand": op3_ratio},
        "readings": {"per_raw_material_D5": in_use, "total_across_twelve": alternative},
        "falsifiers": checks, "falsifier_summary": summary,
    }
    seal_and_write(payload, args.output, contract=CONTRACT,
                   reference=Path("results/raw_scarcity_boundary/result.json"))

    print(f"veredicto: {status}\n")
    print(f"  demanda            {demand_raw_week:>12,.0f} unidades crudas/sem")
    print(f"  Op2 (publicado)    {supply_raw_week:>12,.0f}  = {supply_ratio:.2f}x la demanda")
    print(f"  Op3 (publicado)    {op3_raw_week:>12,.0f}  = {op3_ratio:.2f}x la demanda")
    for name, row in payload["readings"].items():
        print(f"  {name:24} servicio {row['service']:.4f}  en mano al cierre "
              f"{row['on_hand_close']:>12,.0f}")
    print(f"\n  falsadores: {summary['n_computed']} computados, {summary['n_failed']} fallidos")
    for name, c in checks.items():
        if c.get("computed"):
            print(f"    {name:48} {'PASA' if c['passed'] else 'FALLA'}")
    print(f"\n  -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
