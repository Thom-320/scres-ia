#!/usr/bin/env python3
"""Run the CSSU interface-liveness gate without claiming finite Op11 physics.

This is intentionally an executable contract for Gate A only.  Gate B remains a sealed HOLD
until a new preregistration specifies the physical semantics of Op11.
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.arm_runner import run_falsifiers, seal_and_write  # noqa: E402
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
CONTRACT = ROOT / "docs/PREREGISTRO_CSSU_LIVENESS_2026-08-01.md"
REFERENCE = ROOT / "results/metric_audit/contention_service_first_v2/result.json"
OUTPUT = ROOT / "results/garrido_cssu_liveness_gate_v1/result.json"


def _base_kwargs() -> dict:
    return {
        "seed": 321,
        "horizon": 5_000.0,
        "initial_buffers": {"op9_rations": 1_000_000.0},
        "order_fulfillment_mode": "op9_linked",
        "op9_dispatch_policy": "fixed_clock_daily",
        "downstream_transport_capacity_mode": "parallel",
        "cssu_topology_mode": "split_v1",
        "cssu_service_rule": "FIFO_PARTIAL",
        "cssu_daily_capacity": 2_500.0,
        "demand_mean_multiplier": 2.0,
        "risks_enabled": False,
    }


def main() -> int:
    sim = MFSCSimulation(**_base_kwargs(), cssu_allocation_a=0.25)
    before = sim.get_cssu_observation()
    sim.step(
        action={"cssu_allocation_a": 0.75, "cssu_service_rule": "FIFO_PARTIAL"},
        step_hours=1.0,
    )
    scheduled = list(sim.cssu_action_events)
    pre_activation_share = float(sim.cssu_allocation_a)
    sim.env.run(until=24.0)
    sim._activate_due_cssu_action()
    post_activation_share = float(sim.cssu_allocation_a)

    dynamic = MFSCSimulation(**_base_kwargs(), cssu_allocation_a=0.25)
    static = MFSCSimulation(**_base_kwargs(), cssu_allocation_a=0.25)
    for day in range(100):
        action = (
            {"cssu_allocation_a": 0.75, "cssu_service_rule": "FIFO_PARTIAL"}
            if day == 5
            else None
        )
        dynamic.step(action=action, step_hours=24.0)
        static.step(step_hours=24.0)

    def action_changes_ledgers() -> tuple[bool, dict]:
        return (
            dynamic.cssu_dispatched["A"] > static.cssu_dispatched["A"]
            and dynamic.cssu_dispatched["B"] < static.cssu_dispatched["B"],
            {
                "dynamic_dispatched": dict(dynamic.cssu_dispatched),
                "static_dispatched": dict(static.cssu_dispatched),
            },
        )

    def aggregate_rejects() -> tuple[bool, dict]:
        aggregate = MFSCSimulation(cssu_topology_mode="aggregate")
        try:
            aggregate.step(action={"cssu_allocation_a": 0.75}, step_hours=1.0)
        except ValueError as exc:
            return True, {"error": str(exc)}
        return False, {"error": None}

    def observation_has_no_future() -> tuple[bool, dict]:
        forbidden = ("future", "next_risk", "repair_duration", "regime")
        bad = [key for key in before if any(fragment in key for fragment in forbidden)]
        return not bad, {"forbidden_keys": bad}

    def mass_is_conserved() -> tuple[bool, dict]:
        ok = (
            abs(sum(dynamic.cssu_demanded.values()) - dynamic.total_demanded) < 1e-9
            and abs(sum(dynamic.cssu_delivered.values()) - dynamic.total_delivered) < 1e-9
        )
        return ok, {
            "cssu_demanded": dict(dynamic.cssu_demanded),
            "cssu_delivered": dict(dynamic.cssu_delivered),
            "total_demanded": dynamic.total_demanded,
            "total_delivered": dynamic.total_delivered,
        }

    falsifiers = run_falsifiers(
        {
            "f1_action_is_scheduled": lambda: (
                len(scheduled) == 1 and scheduled[0]["status"] == "scheduled",
                {"events": scheduled},
            ),
            "f2_latency_is_respected": lambda: (
                pre_activation_share == 0.25 and post_activation_share == 0.75,
                {
                    "pre_activation_share": pre_activation_share,
                    "post_activation_share": post_activation_share,
                },
            ),
            "f3_action_changes_destination_ledgers": action_changes_ledgers,
            "f4_mass_is_conserved": mass_is_conserved,
            "f5_aggregate_mode_rejects_action": aggregate_rejects,
            "f6_observation_has_no_future_truth": observation_has_no_future,
        }
    )
    payload = {
        "schema_version": "garrido_cssu_liveness_gate_v1",
        "claim_status": (
            "GATE_A_PASS_GATE_B_HOLD" if falsifiers["all_passed"]
            else "HALTED_FALSIFIER_FAILED"
        ),
        "gate_a": {
            "status": "PASS" if falsifiers["all_passed"] else "FAIL",
            "scope": "split CSSU allocation action only",
            "activation_latency_hours": 24.0,
        },
        "gate_b": {
            "status": "HOLD_OP11_PHYSICS_UNSPECIFIED",
            "scope": "finite Op11 handling is not implemented or claimed",
            "required_before": "physical contention claims or neural training",
        },
        "physics_boundary": {
            "thesis_native_unchanged": True,
            "op11_handling_hours": 0.0,
            "op11_value_is_not_evidence": True,
        },
        "falsifiers": falsifiers,
    }
    digest = seal_and_write(payload, OUTPUT, contract=CONTRACT, reference=REFERENCE)
    print(f"Saved: {OUTPUT}")
    print(f"self_sha256: {digest}")
    print(f"Gate A: {payload['gate_a']['status']}; Gate B: {payload['gate_b']['status']}")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
