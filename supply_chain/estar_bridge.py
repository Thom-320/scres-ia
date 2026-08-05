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
from supply_chain.estar_des_adapter import EStarDESAdapter
from supply_chain.estar_kernel import EStarAction, MASKS, SUPPLIER_LANES
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


def make_expanded_sim(tape: Mapping[str, Any], mask_id: str) -> EStarDESAdapter:
    """Build the E* adapter on the same burned exogenous tape.

    The adapter is deliberately distinct from ``make_flags_off_sim``.  M000
    continues to use the historical constructor; P/U/D masks use explicit
    action-driven procurement and dispatch queues.
    """
    family = str(tape["family"])
    if family not in FAMILY_RISKS:
        raise ValueError(f"unsupported bridge family {family!r}")
    return EStarDESAdapter(
        mask_id=str(mask_id),
        expanded=True,
        shifts=1,
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


def _expanded_smoke_action(mask_id: str) -> EStarAction:
    mask = MASKS[str(mask_id)]
    suppliers = tuple(SUPPLIER_LANES) if mask["P"] else ()
    procurement = {lane: 100.0 for lane in suppliers}
    targets = {
        node: 500.0 if mask["U"] else 0.0
        for node in ("wdc", "al", "sb", "cssu_a", "cssu_b")
    }
    return EStarAction(
        procurement_qty=procurement,
        buffer_targets=targets,
        active_supplier_lanes=suppliers,
    )


def expanded_bridge_smoke(
    tape: Mapping[str, Any], *, m000_expected_digest: str | None = None
) -> dict[str, Any]:
    """Exercise every mask once without opening a seed or making a claim.

    This is a source-conservation/liveness receipt, not a scientific result.
    The M000 golden check remains an independent anchor; the expanded masks
    must close their ledgers and must not invoke the historical strategic-topup
    shortcut.
    """
    rows: list[dict[str, Any]] = []
    m000_adapter = check_flags_off_adapter_golden(tape, m000_expected_digest)
    for mask_id in MASKS:
        sim = make_expanded_sim(tape, mask_id)
        action = _expanded_smoke_action(mask_id)
        sim.step_e_star(action, step_hours=168.0)
        evidence = sim.e_star_evidence()
        flow = evidence["flow_ledger"]
        rows.append(
            {
                "mask_id": mask_id,
                "action": action.canonical(),
                "flow_ledger": flow,
                "strategic_raw_injected": float(sim.total_strategic_raw_injected),
                "strategic_rations_injected": float(sim.total_strategic_rations_injected),
                "payload_sha256": sim.e_star_payload_sha256(),
            }
        )
    return {
        "schema_version": "estar_expanded_bridge_smoke_v1",
        "seed": int(tape["seed"]),
        "family": str(tape["family"]),
        "horizon": float(tape["horizon"]),
        "m000_adapter_bridge": m000_adapter,
        "masks": rows,
    }


def expanded_bridge_smoke_digest(
    tape: Mapping[str, Any], *, m000_expected_digest: str | None = None
) -> str:
    return _canonical_sha(
        expanded_bridge_smoke(
            tape, m000_expected_digest=m000_expected_digest
        )
    )


def check_expanded_bridge_smoke(
    tape: Mapping[str, Any],
    expected_digest: str | None,
    *,
    m000_expected_digest: str | None = None,
) -> dict[str, Any]:
    payload = expanded_bridge_smoke(
        tape, m000_expected_digest=m000_expected_digest
    )
    residuals = [
        {
            "mask_id": row["mask_id"],
            "raw_residual": float(row["flow_ledger"].get("raw_residual", 0.0)),
            "ration_residual": float(row["flow_ledger"].get("ration_residual", 0.0)),
            "strategic_raw_injected": row["strategic_raw_injected"],
            "strategic_rations_injected": row["strategic_rations_injected"],
        }
        for row in payload["masks"]
    ]
    invariants_pass = all(
        abs(row["raw_residual"]) <= 1e-6
        and abs(row["ration_residual"]) <= 1e-6
        and row["strategic_raw_injected"] == 0.0
        and row["strategic_rations_injected"] == 0.0
        for row in residuals
    )
    m000_pass = bool(payload["m000_adapter_bridge"]["passed"])
    observed = _canonical_sha(payload)
    digest_pass = expected_digest is not None and observed == expected_digest
    return {
        "status": "PASS"
        if invariants_pass
        and m000_pass
        and (expected_digest is None or digest_pass)
        else "FAIL",
        "passed": bool(
            invariants_pass
            and m000_pass
            and (expected_digest is None or digest_pass)
        ),
        "invariants_pass": bool(invariants_pass and m000_pass),
        "m000_adapter_pass": m000_pass,
        "expected_digest": expected_digest,
        "observed_digest": observed,
        "residuals": residuals,
        "payload": payload,
    }


def flags_off_payload(tape: Mapping[str, Any]) -> dict[str, Any]:
    sim = make_flags_off_sim(tape)
    sim.step(action=None, step_hours=float(tape["horizon"]))
    return flags_off_payload_from_sim(
        sim,
        family=str(tape["family"]),
        seed=int(tape["seed"]),
        horizon=float(tape["horizon"]),
    )


def flags_off_payload_from_sim(
    sim: MFSCSimulation, *, family: str, seed: int, horizon: float
) -> dict[str, Any]:
    metric = compute_episode_metrics(sim)
    flow = sim.flow_ledger()
    return {
        "family": str(family),
        "seed": int(seed),
        "horizon": float(horizon),
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


def check_flags_off_adapter_golden(
    tape: Mapping[str, Any], expected_digest: str | None
) -> dict[str, Any]:
    """Check that the adapter's M000 mode reaches the historical golden payload."""
    family = str(tape["family"])
    sim = EStarDESAdapter(
        mask_id="M000",
        expanded=False,
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
    sim.step(action=None, step_hours=float(tape["horizon"]))
    observed = _canonical_sha(
        flags_off_payload_from_sim(
            sim,
            family=family,
            seed=int(tape["seed"]),
            horizon=float(tape["horizon"]),
        )
    )
    return {
        "status": "PASS" if expected_digest and observed == expected_digest else "FAIL",
        "passed": bool(expected_digest and observed == expected_digest),
        "expected_digest": expected_digest,
        "observed_digest": observed,
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
