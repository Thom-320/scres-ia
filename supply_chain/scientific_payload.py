"""Canonical scientific payloads for arm identity and physical nulls.

The envelope written by :func:`supply_chain.arm_runner.seal_and_write` contains
timestamps, provenance and a self-hash.  Those fields are intentionally not part
of the scientific identity of a simulation.  G3c therefore hashes this separate
payload, which contains only realized physical state, trajectories and metrics.
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping


SCIENTIFIC_PAYLOAD_SCHEMA = "scientific_payload_v1"


def _normalise(value: Any) -> Any:
    """Convert scientific values to deterministic JSON-safe values."""
    if is_dataclass(value):
        return _normalise({field.name: getattr(value, field.name) for field in fields(value)})
    if isinstance(value, Mapping):
        return {
            str(key): _normalise(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalise(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_normalise(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True,
                                                         separators=(",", ":")))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {"__bytes_hex__": value.hex()}
    # numpy scalar values expose item() but are not guaranteed to be importable in
    # every lightweight test environment.
    if hasattr(value, "item") and not isinstance(value, (str, bytes, bytearray)):
        try:
            return _normalise(value.item())
        except (AttributeError, ValueError, TypeError):
            pass
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return {"__nonfinite_float__": repr(value)}
    if value is None or isinstance(value, (bool, int, str)):
        return value
    raise TypeError(f"unsupported scientific payload value: {type(value)!r}")


def canonical_json_bytes(value: Any) -> bytes:
    """Return the canonical byte representation used for scientific hashes."""
    return json.dumps(
        _normalise(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def scientific_payload_sha256(payload: Mapping[str, Any]) -> str:
    """Hash a scientific payload, never an artifact envelope."""
    return sha256(canonical_json_bytes(payload)).hexdigest()


def _record_payload(record: Any) -> dict[str, Any]:
    if is_dataclass(record):
        return _normalise({field.name: getattr(record, field.name) for field in fields(record)})
    if hasattr(record, "__dict__"):
        return _normalise({key: value for key, value in vars(record).items()
                           if not key.startswith("_")})
    return _normalise(record)


def canonical_scientific_payload(sim: Any, metrics: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Extract realized DES science from a completed simulation.

    The function deliberately names the included traces and ledgers.  It does
    not hash the SimPy environment, Python object identities, timestamps from
    the artifact envelope, or calibration provenance.
    """
    orders = sorted(getattr(sim, "orders", []),
                    key=lambda order: (int(getattr(order, "j", 0)),
                                       float(getattr(order, "OPTj", 0.0) or 0.0)))
    risk_events = [_record_payload(event) for event in getattr(sim, "risk_events", [])]
    trace_names = (
        "cssu_action_events",
        "cssu_demand_events",
        "cssu_delivery_events",
        "cssu_local_risk_events",
        "expedite_events",
        "backorder_priority_rule_events",
    )
    traces = {name: _normalise(getattr(sim, name, [])) for name in trace_names}

    ledger_names = (
        "cssu_dispatched",
        "cssu_demanded",
        "cssu_delivered",
        "cssu_inventory",
        "cssu_in_transit",
        "cssu_inbound_in_transit",
        "cssu_outbound_in_transit",
        "cssu_forfeited_epochs",
        "cssu_forfeited_rations",
        "cssu_switch_count",
        "cssu_blocked_by_dwell_count",
        "cssu_switch_cost_paid",
        "cssu_switch_cost_unpaid",
        "total_delivered",
        "total_backorders",
        "cumulative_backorder_qty",
        "pending_backorder_qty",
    )
    ledgers = {name: _normalise(getattr(sim, name, None)) for name in ledger_names}
    container_names = (
        "raw_material_wdc", "raw_material_al", "rework_op6", "wip_op5_op6",
        "wip_op6_op7", "rations_al", "rations_sb", "rations_sb_dispatch",
        "rations_cssu", "rations_theatre",
    )
    ledgers["container_levels"] = {
        name: float(getattr(getattr(sim, name, None), "level"))
        for name in container_names
        if getattr(sim, name, None) is not None
        and getattr(getattr(sim, name, None), "level", None) is not None
    }

    config_names = (
        "seed", "horizon", "warmup_time", "cssu_topology_mode",
        "cssu_allocation_a", "cssu_service_rule", "cssu_reallocate_unused",
        "cssu_min_dwell_days", "cssu_switch_cost_rations",
    )
    configuration = {name: _normalise(getattr(sim, name, None)) for name in config_names}
    return {
        "schema": SCIENTIFIC_PAYLOAD_SCHEMA,
        "configuration": configuration,
        "orders": [_record_payload(order) for order in orders],
        "risk_events": risk_events,
        "traces": traces,
        "ledgers": ledgers,
        "metrics": _normalise(dict(metrics or {})),
    }


def scientific_payloads_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Compare scientific payloads while ignoring all envelope metadata."""
    return canonical_json_bytes(left) == canonical_json_bytes(right)
