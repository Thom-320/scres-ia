"""Flags-off bridge helpers for the E* contract.

The bridge deliberately uses the existing replay factory and emits a small,
canonical payload.  A caller may promote it to ``PASS`` only when the payload
matches the frozen golden digest in the E* contract.  Running the same code
twice is not an identity proof; the golden digest and conservation checks are
the independent anchor.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

from supply_chain.episode_metrics import compute_episode_metrics
from supply_chain.supply_chain import MFSCSimulation


FAMILY_RISKS: dict[str, tuple[str, ...]] = {
    "R1r": ("R11", "R12", "R13", "R14"),
    "R2r": ("R21", "R22", "R23", "R24"),
}


def _canonical_sha(value: Any) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def make_flags_off_sim(tape: Mapping[str, Any]) -> MFSCSimulation:
    family = str(tape["family"])
    if family not in FAMILY_RISKS:
        raise ValueError(f"unsupported bridge family {family!r}")
    return MFSCSimulation(
        shifts=1,
        initial_buffers={"op3_rm": 0.0, "op5_rm": 0.0, "op9_rations": 0.0},
        inventory_replenishment_period=168.0,
        seed=int(tape["seed"]),
        horizon=float(tape["horizon"]),
        risks_enabled=True,
        risk_level="current",
        enabled_risks=set(FAMILY_RISKS[family]),
        risk_overrides={risk: "increased" for risk in FAMILY_RISKS[family]},
        strict_exogenous_crn=True,
        demand_source="excel_order_tape",
        excel_order_tape=list(tape["orders"]),
        risk_event_tape=list(tape["risks"]),
    )


def flags_off_payload(tape: Mapping[str, Any]) -> dict[str, Any]:
    sim = make_flags_off_sim(tape)
    sim.step(action=None, step_hours=float(tape["horizon"]))
    metric = compute_episode_metrics(sim)
    flow = sim.flow_ledger()
    return {
        "family": str(tape["family"]),
        "seed": int(tape["seed"]),
        "horizon": float(tape["horizon"]),
        "orders": len(sim.orders),
        "risk_events": len(sim.risk_events),
        "metrics": {
            key: float(metric[key])
            for key in (
                "ret_excel",
                "ret_excel_full_ledger",
                "ret_thesis",
                "flow_fill_rate",
                "lost_orders",
                "unresolved_orders",
            )
            if key in metric
        },
        "flow_ledger": {
            key: float(value) for key, value in sorted(flow.items())
        },
        "inventory": {
            key: float(value) for key, value in sorted(sim._inventory_detail().items())
        },
        "strategic_injected": float(
            sim.total_strategic_raw_injected + sim.total_strategic_rations_injected
        ),
    }


def flags_off_digest(tape: Mapping[str, Any]) -> str:
    return _canonical_sha(flags_off_payload(tape))


def check_flags_off_golden(
    tape: Mapping[str, Any], expected_digest: str | None
) -> dict[str, Any]:
    observed = flags_off_digest(tape)
    if not expected_digest:
        return {
            "status": "DES_BRIDGE_GOLDEN_MISSING",
            "passed": False,
            "observed_digest": observed,
        }
    return {
        "status": "PASS" if observed == expected_digest else "FAIL",
        "passed": observed == expected_digest,
        "observed_digest": observed,
        "expected_digest": expected_digest,
    }


def load_tape(path: Path | str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError("bridge fixture must contain exactly one tape")
        value = value[0]
    if not isinstance(value, dict):
        raise ValueError("bridge fixture must be a JSON object or singleton list")
    for key in ("family", "seed", "horizon", "orders", "risks"):
        if key not in value:
            raise ValueError(f"bridge fixture missing {key!r}")
    return value
