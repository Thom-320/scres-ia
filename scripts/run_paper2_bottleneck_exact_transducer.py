#!/usr/bin/env python3
"""Fail-closed certification harness for the Paper-2 M/T/R prefix transducer.

This script is deliberately *not* a full-horizon H_PI computation.  It tests
whether prefixes of the frozen ``paper2_bottleneck_migration_v1`` simulator may
share future transitions under a conservative semantic Markov key.  The first
certification target is the canonical sparse visible-order ReT ledger on short,
already-burned development tapes.

Scientific limits
-----------------
* A successful short-horizon run certifies only the horizons and tapes that
  were exhaustively compared with unaccelerated ``run_policy`` replays.
* Non-additive secondary metrics are recomputed only by the original simulator;
  until their sufficient output labels are certified, the frozen 24-week bound
  contract remains fail-closed.
* The script never opens learner/virgin tapes and never emits H_PI.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, dataclass, fields, is_dataclass
from hashlib import sha256
from importlib import metadata as importlib_metadata
import inspect
import json
import math
import multiprocessing as mp
from pathlib import Path
import platform
import re
import subprocess
import sys
import sysconfig
import time
import textwrap
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.paper2_bottleneck import (  # noqa: E402
    ACTIONS,
    ACTION_NAMES,
    CONTEXTS,
    make_sim,
    materialize_tape,
)
from supply_chain.program_f import advance_including  # noqa: E402
from supply_chain.ret_thesis import (  # noqa: E402
    compute_order_level_ret_excel_visible_ledger,
)
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
CONTRACT_PATH = ROOT / "contracts" / "paper2_bottleneck_full_horizon_bound_v1.json"
PRIMARY_CONTRACT_PATH = ROOT / "contracts" / "paper2_bottleneck_primary_bound_v2.json"
KEY_SCHEMA_VERSION = "paper2_bottleneck_semantic_markov_key_v2"
RESULT_SCHEMA_VERSION = "paper2_bottleneck_exact_transducer_certification_v3"
CERTIFICATION_DEPENDENCY_PATHS = (
    CONTRACT_PATH,
    PRIMARY_CONTRACT_PATH,
    Path(__file__).resolve(),
    ROOT / "supply_chain" / "paper2_bottleneck.py",
    ROOT / "supply_chain" / "supply_chain.py",
    ROOT / "supply_chain" / "episode_metrics.py",
    ROOT / "supply_chain" / "ret_thesis.py",
    ROOT / "supply_chain" / "program_f.py",
    ROOT / "supply_chain" / "config.py",
    ROOT / "supply_chain" / "data" / "garrido_proxy_v1_freeze_2026-07-10.json",
    ROOT / "requirements.txt",
    ROOT / "requirements-pinned.txt",
)
ENVIRONMENT_PACKAGES = ("numpy", "simpy", "gymnasium", "scipy", "pandas")
REDUCED_CERTIFICATION_SUITES = {
    "w12_five_tape": {
        "weeks": 12,
        "tapes": (
            (1_110_001, "equipment_pressure", "ebefe63a0462900911269f6b4a3dabf6b9c1d2b1b1d7fb042573613ac00166f0"),
            (1_100_001, "equipment_pressure", "c56c321551f6bc675073fdc0fe4e05765db1e5429dfe6916cf18b2aa4d407029"),
            (1_100_031, "equipment_pressure", "ef9a30479972a2e727c0d0356b6d1b3b848e0f6ba2f946594d6720d63f989624"),
            (1_110_061, "equipment_pressure", "b1e9d9082144ab68b8de41c60edb9038a9c0ba22346bcaa348130508a09a4519"),
            (1_110_120, "mission_surge", "1d9339549f428804b15b6d0f17e9084547643364f830c87d3df1482bb5d05351"),
        ),
    },
    "w16_hard_tape": {
        "weeks": 16,
        "tapes": (
            (1_110_025, "equipment_pressure", "3e3056521130921d5898ca07cc96455d69698f78be91aa9e055440f0bfa2dd62"),
        ),
    },
}

# Metrics used for the unaccelerated replay digest.  They cover the primary,
# all directly available frozen guardrails, mass conservation and reserve use.
ENDPOINT_KEYS = (
    "ret_excel",
    "ration_ret_excel",
    "ret_excel_cvar05",
    "ret_excel_cvar10",
    "service_loss_auc_ration_hours",
    "n_lost",
    "lost_orders",
    "backorder_qty_final",
    "backlog_age_max",
    "mass_residual",
    "reserve_units_issued",
    "reserve_units_replenished",
    "reserve_inventory_terminal",
)

# These fields are the mutable simulator/controller state read by the frozen
# future process graph but are not represented by live generator locals or
# Container state.  Adding fields is conservative.  Removing one requires a
# call-graph proof and a new key schema version.
SIM_STATE_FIELDS = (
    "params",
    "contract_valid_until",
    "_pending_batch",
    "_raw_material_in_transit",
    "_in_transit",
    "pending_backorder_qty",
    "total_unattended_orders",
    "op_down_count",
    "_op_down_since",
    "_contingent_demand_pending",
    "program_f_r24_issue_remaining",
    "_ret_quantity_risk_units",
    "_ret_quantity_risk_refs",
    "emergency_reserve_target",
    "emergency_reserve_in_transit",
    "emergency_reserve_units_issued",
    "emergency_reserve_units_replenished",
    "emergency_reserve_replenishment_requests",
    "program_f_reserve_fragments_issued",
    "_hour_in_week",
    "_today_produced",
    "_route_wait_pending",
    "maintenance_debt",
    "emergency_reserve_inventory_time",
    "_emergency_reserve_last_accounting_time",
    "_material_lineage_sequence",
    "_material_lineage",
    "_pending_lineage_events_by_stage",
    "_lineage_event_index",
    "_upstream_scarcity_debts",
    "_queue_len_history",
    "_exposure_end_cache",
    "_r24_causal_episodes",
    "_r24_causal_sequence",
    "_risk_recovery_seen",
    "_risk_recovery_window_until",
    "_risk_recovery_release_emitted",
    "_risk_recovery_base_params",
    "_risk_recovery_boosted",
    "_contingent_cssu_destination_pending",
    "_pending_cssu_action",
    "cssu_local_down_count",
)
CONTAINER_FIELDS = (
    "raw_material_wdc",
    "raw_material_al",
    "wip_op5_op6",
    "wip_op6_op7",
    "rework_op6",
    "rations_al",
    "rations_sb",
    "rations_sb_dispatch",
    "rations_cssu",
    "rations_theatre",
    "emergency_theatre_reserve",
)
RNG_FIELDS = ("rng", "demand_rng", "risk_rng", "regime_rng")

# Explicit classification used by ``audit_frozen_state_inventory``.  The
# inventory is generated from the live object and fails if a new attribute is
# not assigned to one of these scientific roles.
SPECIAL_KEY_FIELDS = {
    "env",
    "orders",
    "pending_backorders",
    "risk_events",
    "_contract_renewed_event",
    "op10_convoy",
    "op12_convoy",
    *CONTAINER_FIELDS,
    *RNG_FIELDS,
    "risk_rng_by_id",
}
IMMUTABLE_CONTRACT_FIELDS = {
    "seed", "seed_stream_mode", "strict_exogenous_crn", "horizon", "shifts",
    "hours_per_year", "year_basis", "risks_enabled", "risk_level",
    "risk_occurrence_mode", "risk_attribution_source", "risk_event_tape",
    "enabled_risks", "risk_overrides", "risk_rng_mode",
    "risk_frequency_multiplier", "risk_frequency_multipliers_by_id",
    "risk_impact_multiplier", "risk_impact_multipliers_by_id",
    "stochastic_pt", "deterministic_baseline", "warmup_trigger",
    "downstream_q_source", "raw_material_flow_mode",
    "raw_material_order_up_to_multiplier", "_raw_units_per_ration",
    "demand_source", "excel_order_tape", "demand_mean_multiplier",
    "demand_on_hand_fulfillment_delay", "demand_start_after_warmup",
    "assembly_flow_mode", "serial_wip_capacity_rations", "periodic_release_mode",
    "op2_release_clock_mode", "operational_risk_initialization_mode",
    "procurement_contract_mode", "order_fulfillment_mode",
    "op9_dispatch_policy", "downstream_transport_capacity_mode",
    "op9_freight_offset_hours", "replenishment_route_aware",
    "ret_recovery_period_mode", "backorder_overflow_mode",
    "backorder_priority_rule", "backorder_age_threshold_hours",
    "r14_defect_mode", "r24_attribution_window_hours", "material_lineage_mode",
    "cssu_topology_mode", "cssu_allocation_a", "cssu_service_rule",
    "cssu_daily_capacity_override", "op8_dispatch_mode", "op8_convoy_capacity",
    "op8_convoy_outbound_hours", "op8_convoy_return_hours",
    "inventory_buffer_targets", "inventory_replenishment_period",
    "inventory_replenishment_lead_time", "_op5_rm_base",
    "_op5_multiplier_rule", "campaign_config", "campaign_path",
    "emergency_reserve_enabled", "emergency_reserve_capacity",
    "emergency_reserve_replenishment_lead_time", "emergency_reserve_issue_delay",
    "emergency_reserve_route_ops", "emergency_reserve_transport_mode",
    "program_f_reserve_enabled", "risk_recovery_window_hours",
    "risk_recovery_release_rations", "risk_recovery_boost_downstream",
    "risk_recovery_enabled_risks", "_step_size", "_processes_started",
}
INERT_FROZEN_FIELDS = {
    "adaptive_benchmark_enabled", "adaptive_benchmark_v2_enabled",
    "adaptive_regime", "adaptive_risk_forecast_48h", "adaptive_risk_forecast_168h",
    "cssu_in_transit", "cssu_inbound_in_transit", "cssu_inventory",
    "cssu_outbound_in_transit", "cssu_delivered", "cssu_demanded",
    "cssu_dispatched", "cssu_allocation_live_epochs", "cssu_allocation_moot_epochs",
    "cssu_action_events", "cssu_demand_events", "cssu_delivery_events",
    "cssu_local_risk_events", "op8_convoy_available", "op8_convoy_departures",
    "op8_convoy_dispatched_rations", "op8_convoy_capacity_committed",
    "op8_convoy_vehicle_hours", "op8_convoy_idle_hours",
    "op8_convoy_route_wait_hours", "op8_convoy_ration_hours_in_transit",
    "op8_convoy_masked_dispatch_attempts", "op8_convoy_hold_actions",
    "op8_convoy_dispatch_actions", "op8_convoy_last_departure_at",
    "op8_convoy_nominal_return_at", "op8_convoy_actual_return_at",
    "op8_staging_first_ready_at", "op8_last_action", "op8_convoy_action_events",
    "op8_convoy_departure_events",
}
OUTPUT_OR_REPLAY_FIELDS = {
    "contract_completion_events", "supplier_delivery_events",
    "material_availability_events", "backorder_priority_rule_events",
    "total_external_raw_material", "total_strategic_raw_injected",
    "total_strategic_rations_injected", "total_rations_created_from_raw",
    "total_rations_scrapped", "total_raw_material_consumed", "total_order_fulfilled",
    "total_theatre_inflow", "total_produced", "total_delivered", "total_demanded",
    "total_backorders", "cumulative_backorder_qty", "warmup_complete", "warmup_time",
    "_cumulative_available_assembly_hours", "_cumulative_down_hours",
    "_prev_step_produced", "_prev_step_delivered",
    "_prev_step_available_assembly_hours", "_prev_step_fill_rate",
    "_ewma_fill_rate", "_ewma_backlog_growth", "_delta_fill_rate",
    "_delta_backlog_momentum", "_prev_pending_backorder_qty",
    "daily_production", "daily_demand", "delivery_events", "daily_inventory_sb",
    "daily_inventory_theatre", "emergency_reserve_target_changes",
    "program_f_reserve_issue_events",
}


def static_sim_attribute_reads() -> set[str]:
    """Conservative class-wide AST inventory of ``self.<attr>`` reads."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(MFSCSimulation)))
    reads: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.ctx, ast.Load)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        ):
            reads.add(node.attr)
    return reads


def audit_frozen_state_inventory(sim: Any) -> dict[str, Any]:
    """Classify every live simulator attribute and assert frozen-mode claims."""
    live = set(vars(sim))
    keyed = set(SIM_STATE_FIELDS) | set(SPECIAL_KEY_FIELDS)
    categories = {
        "markov_key": sorted(live & keyed),
        "immutable_contract": sorted(live & set(IMMUTABLE_CONTRACT_FIELDS)),
        "inert_frozen_contract": sorted(live & set(INERT_FROZEN_FIELDS)),
        "output_label_or_unaccelerated_replay": sorted(
            live & set(OUTPUT_OR_REPLAY_FIELDS)
        ),
    }
    classified = set().union(*(set(values) for values in categories.values()))
    unclassified = sorted(live - classified)
    overlaps: dict[str, list[str]] = {}
    names = list(categories)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            overlap = sorted(set(categories[left]) & set(categories[right]))
            if overlap:
                overlaps[f"{left}::{right}"] = overlap

    frozen_invariants = {
        "risk_attribution_source_des_events": sim.risk_attribution_source == "des_events",
        "material_lineage_mode_off": sim.material_lineage_mode == "off",
        "material_lineage_queues_empty": not any(sim._material_lineage.values()),
        "pending_lineage_refs_empty": not any(
            sim._pending_lineage_events_by_stage.values()
        ),
        "lineage_event_index_empty": not sim._lineage_event_index,
        "native_risks_disabled": sim.risks_enabled is False,
        "adaptive_benchmark_disabled": sim.adaptive_benchmark_enabled is False,
        "risk_recovery_controller_disabled": (
            sim.risk_recovery_window_hours == 0.0
            and sim.risk_recovery_release_rations == 0.0
        ),
        "aggregate_cssu_topology": sim.cssu_topology_mode == "aggregate",
        "finite_convoy_disabled": sim.op8_dispatch_mode == "thesis_full_batch",
        "parallel_downstream_transport": (
            sim.downstream_transport_capacity_mode == "parallel"
        ),
    }
    static_reads = static_sim_attribute_reads()
    return {
        "live_attribute_count": len(live),
        "categories": categories,
        "unclassified_live_attributes": unclassified,
        "category_overlaps": overlaps,
        "frozen_invariants": frozen_invariants,
        "all_frozen_invariants_hold": all(frozen_invariants.values()),
        "class_wide_static_read_count": len(static_reads),
        "class_wide_static_reads_present_live": sorted(static_reads & live),
        "class_wide_static_reads_not_present_in_frozen_instance": sorted(
            static_reads - live
        ),
        "static_live_reads_unclassified": sorted((static_reads & live) - classified),
        "classification_complete": not unclassified and not overlaps,
        "scope_warning": (
            "The AST inventory is class-wide rather than a dynamically pruned call graph. "
            "Inactive-mode attributes are retained in the explicit inert/immutable classes; "
            "the reduced-horizon replay remains the executable transition proof."
        ),
    }


def _digest(value: Any) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _git_value(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def certification_environment() -> dict[str, Any]:
    """Return the scientific ABI/package identity used by every certificate."""
    versions: dict[str, str] = {}
    for package in ENVIRONMENT_PACKAGES:
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = "MISSING"
    payload = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_cache_tag": sys.implementation.cache_tag,
        "python_soabi": sysconfig.get_config_var("SOABI"),
        "packages": versions,
        "requirements_sha256": {
            str(path.relative_to(ROOT)): _file_sha256(path)
            for path in (ROOT / "requirements.txt", ROOT / "requirements-pinned.txt")
        },
    }
    return {**payload, "environment_sha256": _digest(payload)}


def certification_provenance() -> dict[str, Any]:
    dependencies = {
        str(path.relative_to(ROOT)): _file_sha256(path)
        for path in CERTIFICATION_DEPENDENCY_PATHS
    }
    payload = {
        "git_commit": _git_value("rev-parse", "HEAD"),
        "producer_sha256": _file_sha256(Path(__file__).resolve()),
        "dependency_sha256": dependencies,
        "environment": certification_environment(),
    }
    payload["provenance_sha256"] = _digest(payload)
    return payload


def scientific_source_drift() -> str:
    relative_paths = [str(path.relative_to(ROOT)) for path in CERTIFICATION_DEPENDENCY_PATHS]
    return _git_value(
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        *relative_paths,
    ) or ""


def validate_reduced_certification_payload(
    payload: dict[str, Any],
    role: str,
    *,
    expected_environment_sha256: str | None = None,
) -> list[str]:
    """Validate one exact W12/W16 artifact against its frozen v3 identity."""
    failures: list[str] = []
    suite = REDUCED_CERTIFICATION_SUITES.get(role)
    if suite is None:
        return [f"unknown reduced-certification role: {role}"]
    if payload.get("schema_version") != RESULT_SCHEMA_VERSION:
        failures.append(f"reduced-horizon schema mismatch: {role}")
    if payload.get("scientific_status") == "NONSCIENTIFIC_SMOKE_NOT_EVIDENCE":
        failures.append(f"non-scientific smoke cannot certify: {role}")
    if payload.get("key_schema_version") != KEY_SCHEMA_VERSION:
        failures.append(f"reduced-horizon key mismatch: {role}")
    if payload.get("contract_sha256") != _file_sha256(CONTRACT_PATH):
        failures.append(f"reduced-horizon contract hash mismatch: {role}")

    expected_provenance = certification_provenance()
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        failures.append(f"reduced-horizon provenance missing: {role}")
        provenance = {}
    body = dict(provenance)
    claimed_digest = body.pop("provenance_sha256", None)
    if claimed_digest != _digest(body):
        failures.append(f"reduced-horizon provenance digest mismatch: {role}")
    if not re.fullmatch(r"[0-9a-f]{40}", str(provenance.get("git_commit", ""))):
        failures.append(f"reduced-horizon source commit missing: {role}")
    for key in ("producer_sha256", "dependency_sha256", "environment"):
        if provenance.get(key) != expected_provenance.get(key):
            failures.append(f"reduced-horizon provenance field drifted ({key}): {role}")
    environment = provenance.get("environment", {})
    required_environment = (
        expected_environment_sha256
        or expected_provenance["environment"]["environment_sha256"]
    )
    if environment.get("environment_sha256") != required_environment:
        failures.append(f"reduced-horizon environment digest mismatch: {role}")

    expected_rows = suite["tapes"]
    tapes = payload.get("tapes")
    if payload.get("weeks") != suite["weeks"] or not isinstance(tapes, list):
        failures.append(f"reduced-horizon scope mismatch: {role}")
        tapes = []
    actual_identity = [
        (
            row.get("seed"),
            row.get("requested_first_context"),
            row.get("tape_sha256"),
        )
        for row in tapes
        if isinstance(row, dict)
    ]
    if actual_identity != list(expected_rows):
        failures.append(f"reduced-horizon seed/context/tape identity mismatch: {role}")
    expected_count = feasible_calendar_count(int(suite["weeks"]))
    for index, tape in enumerate(tapes):
        if not isinstance(tape, dict):
            failures.append(f"malformed reduced-horizon tape {index}: {role}")
            continue
        if not str(tape.get("split", "")).strip():
            failures.append(f"reduced-horizon split missing for tape {index}: {role}")
        if not (
            tape.get("complete_horizon_enumeration") is True
            and tape.get("primary_transducer_bitwise_certified") is True
            and tape.get("calendars_compared") == expected_count
        ):
            failures.append(f"reduced-horizon enumeration incomplete for tape {index}: {role}")
        audit = tape.get("all_prefix_callback_audit", {})
        layer_counts = audit.get("layer_semantic_key_evaluations")
        layer_callbacks = audit.get("layer_callback_inventory")
        layer_callback_digests = audit.get("layer_prefix_callback_records_sha256")
        layer_nonempty = audit.get(
            "layer_prefixes_with_nonempty_callback_inventory"
        )
        if not (
            audit.get("passed") is True
            and audit.get("unknown_callback_owner_count") == 0
            and audit.get("semantic_key_evaluations") == tape.get("prefix_replays")
            and isinstance(layer_counts, list)
            and len(layer_counts) == suite["weeks"]
            and sum(layer_counts) == tape.get("prefix_replays")
            and isinstance(layer_callbacks, list)
            and len(layer_callbacks) == suite["weeks"]
            and all(isinstance(row, list) and row for row in layer_callbacks)
            and audit.get("prefixes_with_nonempty_callback_inventory")
            == tape.get("prefix_replays")
            and isinstance(layer_nonempty, list)
            and sum(layer_nonempty) == tape.get("prefix_replays")
            and isinstance(layer_callback_digests, list)
            and len(layer_callback_digests) == suite["weeks"]
            and all(re.fullmatch(r"[0-9a-f]{64}", str(row)) for row in layer_callback_digests)
            and re.fullmatch(
                r"[0-9a-f]{64}", str(audit.get("prefix_callback_records_sha256", ""))
            )
        ):
            failures.append(f"reduced-horizon all-prefix callback audit failed for tape {index}: {role}")
    if payload.get("summary", {}).get("all_tapes_primary_bitwise_certified") is not True:
        failures.append(f"reduced-horizon primary certification failed: {role}")
    return failures


def _float_token(value: float | int | np.floating | np.integer) -> tuple[str, str]:
    return ("float", float(value).hex())


def _order_token(order: Any, *, depth: int) -> tuple[str, Any]:
    payload = {
        key: _semantic(value, depth=depth + 1)
        for key, value in vars(order).items()
    }
    return ("OrderRecord", tuple(sorted(payload.items())))


def _risk_token(event: Any, *, depth: int) -> tuple[str, Any]:
    if is_dataclass(event):
        payload = {
            field.name: _semantic(getattr(event, field.name), depth=depth + 1)
            for field in fields(event)
        }
    else:
        payload = {
            key: _semantic(value, depth=depth + 1)
            for key, value in vars(event).items()
        }
    return ("RiskEvent", tuple(sorted(payload.items())))


def _semantic(value: Any, *, depth: int = 0) -> Any:
    """Return a deterministic, JSON-safe semantic token.

    Unknown mutable objects fail closed.  Silently replacing an unknown object
    by only its class name could merge states that have different futures.
    """
    if depth > 12:
        raise TypeError("semantic state nesting exceeded 12 levels")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating, np.integer)):
        return _float_token(value)
    if isinstance(value, bytes):
        return ("bytes", value.hex())
    if isinstance(value, np.ndarray):
        return (
            "ndarray",
            str(value.dtype),
            tuple(value.shape),
            value.tobytes().hex(),
        )
    if isinstance(value, tuple):
        return ("tuple", tuple(_semantic(item, depth=depth + 1) for item in value))
    if isinstance(value, list):
        return ("list", tuple(_semantic(item, depth=depth + 1) for item in value))
    if isinstance(value, set):
        items = [_semantic(item, depth=depth + 1) for item in value]
        return ("set", tuple(sorted(items, key=repr)))
    if isinstance(value, dict):
        items = [
            (
                _semantic(key, depth=depth + 1),
                _semantic(item, depth=depth + 1),
            )
            for key, item in value.items()
        ]
        return ("dict", tuple(sorted(items, key=repr)))
    if hasattr(value, "j") and hasattr(value, "OPTj"):
        return _order_token(value, depth=depth)
    if hasattr(value, "risk_id") and hasattr(value, "start_time"):
        return _risk_token(value, depth=depth)
    if is_dataclass(value):
        return (
            type(value).__qualname__,
            _semantic(asdict(value), depth=depth + 1),
        )
    raise TypeError(
        "unknown mutable value in Markov key: "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _generator_stack(generator: Any) -> tuple[Any, ...]:
    stack: list[Any] = []
    current = generator
    while current is not None and hasattr(current, "gi_code"):
        frame = current.gi_frame
        locals_token = []
        if frame is not None:
            for key, value in sorted(frame.f_locals.items()):
                if key in {
                    "self",
                    "sim",
                    "controller",
                    "tape",
                }:
                    continue
                locals_token.append((key, _semantic(value)))
        stack.append(
            (
                current.gi_code.co_qualname,
                None if frame is None else int(frame.f_lasti),
                tuple(locals_token),
            )
        )
        yielded = current.gi_yieldfrom
        current = yielded if hasattr(yielded, "gi_code") else None
    return tuple(stack)


def _owner_registry(sim: Any) -> dict[int, str]:
    return {
        id(value): name
        for name, value in vars(sim).items()
    }


def _event_callbacks(
    event: Any,
    *,
    owner_registry: dict[int, str],
    callback_inventory: set[tuple[str, str, str]] | None = None,
) -> tuple[Any, ...]:
    callbacks = []
    for callback in getattr(event, "callbacks", None) or ():
        process = getattr(callback, "__self__", None)
        generator = getattr(process, "_generator", None)
        if generator is not None:
            callbacks.append(_generator_stack(generator))
            if callback_inventory is not None:
                callback_inventory.add(
                    (
                        "process",
                        generator.gi_code.co_qualname,
                        type(process).__qualname__,
                    )
                )
            continue
        qualname = getattr(callback, "__qualname__", type(callback).__qualname__)
        owner = getattr(callback, "__self__", None)
        if owner is None:
            owner_name = "unbound"
        elif id(owner) in owner_registry:
            owner_name = owner_registry[id(owner)]
        else:
            raise TypeError(
                "unclassified bound callback owner in Markov key: "
                f"{type(owner).__module__}.{type(owner).__qualname__}::{qualname}"
            )
        callbacks.append((qualname, owner_name, type(owner).__qualname__ if owner is not None else "None"))
        if callback_inventory is not None:
            callback_inventory.add(
                (
                    "callback",
                    qualname,
                    f"{owner_name}:{type(owner).__qualname__ if owner is not None else 'None'}",
                )
            )
    return tuple(callbacks)


def _queued_event_token(
    event: Any,
    *,
    owner_registry: dict[int, str],
    callback_inventory: set[tuple[str, str, str]] | None = None,
) -> tuple[Any, ...]:
    amount = getattr(event, "amount", None)
    return (
        type(event).__qualname__,
        None if amount is None else _semantic(amount),
        bool(getattr(event, "triggered", False)),
        bool(getattr(event, "processed", False)),
        _event_callbacks(
            event,
            owner_registry=owner_registry,
            callback_inventory=callback_inventory,
        ),
    )


def _container_token(
    container: Any,
    *,
    owner_registry: dict[int, str],
    callback_inventory: set[tuple[str, str, str]] | None = None,
) -> tuple[Any, ...]:
    return (
        _float_token(container.level),
        _float_token(container.capacity),
        tuple(
            _queued_event_token(
                event,
                owner_registry=owner_registry,
                callback_inventory=callback_inventory,
            )
            for event in container.get_queue
        ),
        tuple(
            _queued_event_token(
                event,
                owner_registry=owner_registry,
                callback_inventory=callback_inventory,
            )
            for event in container.put_queue
        ),
    )


def _environment_token(
    sim: Any,
    *,
    callback_inventory: set[tuple[str, str, str]] | None = None,
) -> tuple[Any, ...]:
    # Raw event ids are history-dependent.  Their only future meaning is the
    # relative order of same-time/same-priority events, preserved by this sort.
    ordered = sorted(sim.env._queue, key=lambda row: (row[0], row[1], row[2]))
    owners = _owner_registry(sim)
    return (
        _float_token(sim.env.now),
        tuple(
            (
                _float_token(time),
                int(priority),
                rank,
                type(event).__qualname__,
                _event_callbacks(
                    event,
                    owner_registry=owners,
                    callback_inventory=callback_inventory,
                ),
            )
            for rank, (time, priority, _event_id, event) in enumerate(ordered)
        ),
    )


def _unresolved_orders(sim: Any) -> tuple[Any, ...]:
    by_identity: dict[int, Any] = {}
    for order in list(sim.orders) + list(sim.pending_backorders):
        if getattr(order, "OATj", None) is None and not bool(getattr(order, "lost", False)):
            by_identity[id(order)] = order
    return tuple(
        _order_token(order, depth=0)
        for order in sorted(by_identity.values(), key=lambda row: int(row.j))
    )


def _relevant_risk_events(sim: Any) -> tuple[Any, ...]:
    unresolved = [
        order
        for order in list(sim.orders) + list(sim.pending_backorders)
        if getattr(order, "OATj", None) is None and not bool(getattr(order, "lost", False))
    ]
    earliest_opt = min(
        (float(order.OPTj) for order in unresolved),
        default=float(sim.env.now),
    )
    # Under the frozen des_events attribution, an older event cannot overlap a
    # future order.  Retaining every event whose end reaches the earliest live
    # order is conservative; active not-yet-appended threats are in generators.
    return tuple(
        _risk_token(event, depth=0)
        for event in sim.risk_events
        if float(event.end_time) >= earliest_opt
    )


def semantic_markov_key(
    sim: Any,
    controller: Any,
    *,
    callback_inventory: set[tuple[str, str, str]] | None = None,
) -> str:
    """Hash a conservative future-state representation for the frozen lane."""
    payload = {
        "schema": KEY_SCHEMA_VERSION,
        "environment": _environment_token(
            sim, callback_inventory=callback_inventory
        ),
        "containers": tuple(
            (
                name,
                _container_token(
                    getattr(sim, name),
                    owner_registry=_owner_registry(sim),
                    callback_inventory=callback_inventory,
                ),
            )
            for name in CONTAINER_FIELDS
        ),
        "sim_state": tuple(
            (name, _semantic(getattr(sim, name)))
            for name in SIM_STATE_FIELDS
        ),
        "rng_state": tuple(
            (name, _semantic(getattr(sim, name).bit_generator.state))
            for name in RNG_FIELDS
        ),
        "per_risk_rng_state": tuple(
            (
                risk_id,
                _semantic(generator.bit_generator.state),
            )
            for risk_id, generator in sorted(sim.risk_rng_by_id.items())
        ),
        "contract_event": _queued_event_token(
            sim._contract_renewed_event,
            owner_registry=_owner_registry(sim),
            callback_inventory=callback_inventory,
        ),
        "resources": tuple(
            (
                name,
                int(resource.count),
                tuple(
                    _queued_event_token(
                        event,
                        owner_registry=_owner_registry(sim),
                        callback_inventory=callback_inventory,
                    )
                    for event in resource.queue
                ),
            )
            for name, resource in (
                ("op10_convoy", sim.op10_convoy),
                ("op12_convoy", sim.op12_convoy),
            )
        ),
        "unresolved_orders": _unresolved_orders(sim),
        "pending_queue_order": tuple(int(order.j) for order in sim.pending_backorders),
        "relevant_risk_events": _relevant_risk_events(sim),
        "controller": {
            "active_action": tuple(map(int, controller.active_action)),
            "pending_action": tuple(map(int, controller.pending_action)),
            "current_week": int(controller.current_week),
            "last_switch_week": int(controller.last_switch_week),
            "condition": _float_token(controller.condition),
        },
    }
    return _digest(payload)


def runtime_proof_audit(tape: dict[str, Any]) -> dict[str, Any]:
    """Inspect one live post-decision state without opening any new tape."""
    sim, controller, start = make_sim(tape)
    controller.activate_week(0)
    controller.request(ACTIONS[0])
    advance_including(sim, start + 168.0)
    callbacks: set[tuple[str, str, str]] = set()
    _environment_token(sim, callback_inventory=callbacks)
    owners = _owner_registry(sim)
    for name in CONTAINER_FIELDS:
        _container_token(
            getattr(sim, name),
            owner_registry=owners,
            callback_inventory=callbacks,
        )
    inventory = audit_frozen_state_inventory(sim)
    return {
        "state_inventory": inventory,
        "callback_inventory": [
            {"kind": kind, "callable": callable_name, "owner": owner}
            for kind, callable_name, owner in sorted(callbacks)
        ],
        "unknown_callback_owner_count": 0,
        "lineage_exclusion_resolution": (
            "Generator lineage locals are encoded. In addition, the frozen contract "
            "has material_lineage_mode='off', empty lineage queues/indexes, and "
            "risk_attribution_source='des_events'; lineage cannot affect future ReT "
            "on this contract. Any mode drift fails the frozen-invariant audit."
        ),
        "endpoint_horizon_equivalence": (
            "run_prefix advances to start + weeks*168 using advance_including, exactly "
            "the end used by paper2_bottleneck.run_policy before compute_episode_metrics."
        ),
    }


@dataclass(frozen=True)
class Checkpoint:
    visible_values: tuple[float, ...]
    visible_order_ids: tuple[int, ...]
    primary_hex: str
    endpoint_digest: str


def _treatment_orders(sim: Any, start: float) -> list[Any]:
    return [
        order
        for order in sim.orders
        if not bool(getattr(order, "metrics_excluded", False))
        and float(getattr(order, "OPTj", 0.0)) >= float(start)
    ]


def _endpoint_panel(sim: Any, start: float, controller: Any) -> dict[str, Any]:
    panel = compute_episode_metrics(sim, treatment_start=start)
    ledger = sim.flow_ledger()
    panel.update(
        {
            "mass_residual": max(
                abs(float(ledger["raw_residual"])),
                abs(float(ledger["ration_residual"])),
            ),
            "reserve_units_issued": float(sim.program_f_reserve_fragments_issued),
            "reserve_units_replenished": float(sim.emergency_reserve_units_replenished),
            "reserve_inventory_terminal": float(sim.emergency_theatre_reserve.level),
            "token_hours_m": float(controller.token_hours["M"]),
            "token_hours_t": float(controller.token_hours["T"]),
            "token_hours_r": float(controller.token_hours["R"]),
        }
    )
    return panel


def checkpoint(sim: Any, start: float, controller: Any) -> Checkpoint:
    visible = compute_order_level_ret_excel_visible_ledger(
        _treatment_orders(sim, start), current_time=sim.env.now
    )
    values = tuple(float(value) for value in visible["ret_values"])
    ids = tuple(int(row["j"]) for row in visible["ret_rows"])
    primary = float(np.mean(values)) if values else 1.0
    panel = _endpoint_panel(sim, start, controller)
    selected = {
        key: panel.get(key)
        for key in ENDPOINT_KEYS
    }
    return Checkpoint(
        visible_values=values,
        visible_order_ids=ids,
        primary_hex=primary.hex(),
        endpoint_digest=_digest(selected),
    )


@dataclass(frozen=True)
class PrefixResult:
    key: str
    checkpoint: Checkpoint
    callback_inventory: tuple[tuple[str, str, str], ...]


def run_prefix(tape: dict[str, Any], sequence: Sequence[int]) -> PrefixResult:
    """Replay one active-action prefix using the contract's one-week request lag."""
    if not sequence or int(sequence[0]) != 0:
        raise ValueError("active prefix must start with M")
    sim, controller, start = make_sim(tape)
    for week, _action in enumerate(sequence):
        controller.activate_week(week)
        requested = sequence[week + 1] if week + 1 < len(sequence) else sequence[week]
        controller.request(ACTIONS[int(requested)])
        advance_including(sim, start + (week + 1) * 168.0)
    callbacks: set[tuple[str, str, str]] = set()
    key = semantic_markov_key(
        sim, controller, callback_inventory=callbacks
    )
    return PrefixResult(
        key=key,
        checkpoint=checkpoint(sim, start, controller),
        callback_inventory=tuple(sorted(callbacks)),
    )


def feasible_calendar_count(weeks: int) -> int:
    weeks = int(weeks)
    return sum(
        math.comb(weeks - switches, switches) * 2**switches
        for switches in range(weeks // 2 + 1)
    )


def feasible_calendars(weeks: int) -> Iterator[tuple[int, ...]]:
    """Yield active-action calendars in a stable lexicographic DFS order."""
    if weeks < 1:
        return

    def visit(prefix: tuple[int, ...], switched_previous: bool) -> Iterator[tuple[int, ...]]:
        if len(prefix) == weeks:
            yield prefix
            return
        last = prefix[-1]
        choices = (last,) if switched_previous else (0, 1, 2)
        for action in choices:
            yield from visit(prefix + (action,), action != last)

    yield from visit((0,), False)


def calendar_name(sequence: Sequence[int]) -> str:
    return "".join(ACTION_NAMES[ACTIONS[int(action)]] for action in sequence)


@dataclass
class StateNode:
    state_id: int
    key: str
    representative: tuple[int, ...]
    checkpoint: Checkpoint
    last_action: int
    switched_previous: bool


@dataclass(frozen=True)
class Transition:
    next_state_id: int
    appended_visible_values: tuple[float, ...]
    appended_visible_order_ids: tuple[int, ...]


@dataclass
class Transducer:
    weeks: int
    layers: list[list[StateNode]]
    transitions: list[dict[tuple[int, int], Transition]]
    collisions: list[dict[str, Any]]
    prefix_replays: int
    callback_inventory: tuple[tuple[str, str, str], ...] = ()
    semantic_key_evaluations: int = 0
    layer_callback_inventory: tuple[tuple[tuple[str, str, str], ...], ...] = ()
    layer_semantic_key_evaluations: tuple[int, ...] = ()
    prefix_callback_records_sha256: str = ""
    layer_prefix_callback_records_sha256: tuple[str, ...] = ()
    prefixes_with_nonempty_callback_inventory: int = 0
    layer_prefixes_with_nonempty_callback_inventory: tuple[int, ...] = ()

    def predict_visible_ledger(self, sequence: Sequence[int]) -> tuple[tuple[float, ...], tuple[int, ...]]:
        if len(sequence) != self.weeks or int(sequence[0]) != 0:
            raise ValueError("calendar does not match transducer horizon")
        node = self.layers[0][0]
        values = list(node.checkpoint.visible_values)
        order_ids = list(node.checkpoint.visible_order_ids)
        state_id = node.state_id
        for transition_index, action in enumerate(sequence[1:]):
            transition = self.transitions[transition_index][(state_id, int(action))]
            values.extend(transition.appended_visible_values)
            order_ids.extend(transition.appended_visible_order_ids)
            state_id = transition.next_state_id
        return tuple(values), tuple(order_ids)


def _feasible_next(last_action: int, switched_previous: bool) -> tuple[int, ...]:
    return (last_action,) if switched_previous else (0, 1, 2)


def build_transducer(tape: dict[str, Any], weeks: int) -> Transducer:
    initial_prefix = (0,)
    initial_result = run_prefix(tape, initial_prefix)
    callback_inventory = set(initial_result.callback_inventory)
    layer_callback_inventory = [set(initial_result.callback_inventory)]
    layer_semantic_key_evaluations = [1]
    callback_record_digest = sha256()
    initial_record = {
        "prefix": list(initial_prefix),
        "callback_inventory": [list(row) for row in initial_result.callback_inventory],
    }
    callback_record_digest.update(bytes.fromhex(_digest(initial_record)))
    layer_callback_record_digests = [sha256()]
    layer_callback_record_digests[0].update(bytes.fromhex(_digest(initial_record)))
    prefixes_with_callbacks = 1 if initial_result.callback_inventory else 0
    layer_prefixes_with_callbacks = [prefixes_with_callbacks]
    layers = [[
        StateNode(
            state_id=0,
            key=initial_result.key,
            representative=initial_prefix,
            checkpoint=initial_result.checkpoint,
            last_action=0,
            switched_previous=False,
        )
    ]]
    transitions: list[dict[tuple[int, int], Transition]] = []
    collisions: list[dict[str, Any]] = []
    prefix_replays = 1

    for _next_week in range(1, weeks):
        previous_layer = layers[-1]
        next_nodes: list[StateNode] = []
        index: dict[tuple[int, bool, str], int] = {}
        layer_transitions: dict[tuple[int, int], Transition] = {}
        current_layer_callbacks: set[tuple[str, str, str]] = set()
        current_layer_evaluations = 0
        current_layer_callback_digest = sha256()
        current_layer_prefixes_with_callbacks = 0

        for parent in previous_layer:
            for action in _feasible_next(parent.last_action, parent.switched_previous):
                sequence = parent.representative + (int(action),)
                result = run_prefix(tape, sequence)
                callback_inventory.update(result.callback_inventory)
                current_layer_callbacks.update(result.callback_inventory)
                callback_record = {
                    "prefix": list(sequence),
                    "callback_inventory": [list(row) for row in result.callback_inventory],
                }
                record_digest = bytes.fromhex(_digest(callback_record))
                callback_record_digest.update(record_digest)
                current_layer_callback_digest.update(record_digest)
                if result.callback_inventory:
                    prefixes_with_callbacks += 1
                    current_layer_prefixes_with_callbacks += 1
                prefix_replays += 1
                current_layer_evaluations += 1
                parent_values = parent.checkpoint.visible_values
                parent_ids = parent.checkpoint.visible_order_ids
                if result.checkpoint.visible_values[: len(parent_values)] != parent_values:
                    raise AssertionError("completed visible ReT rows changed after their OAT")
                if result.checkpoint.visible_order_ids[: len(parent_ids)] != parent_ids:
                    raise AssertionError("completed visible row order changed after its OAT")
                appended_values = result.checkpoint.visible_values[len(parent_values):]
                appended_ids = result.checkpoint.visible_order_ids[len(parent_ids):]
                switched = int(action) != parent.last_action
                state_key = (int(action), switched, result.key)
                if state_key not in index:
                    state_id = len(next_nodes)
                    index[state_key] = state_id
                    next_nodes.append(
                        StateNode(
                            state_id=state_id,
                            key=result.key,
                            representative=sequence,
                            checkpoint=result.checkpoint,
                            last_action=int(action),
                            switched_previous=switched,
                        )
                    )
                else:
                    state_id = index[state_key]
                    representative = next_nodes[state_id]
                    collisions.append(
                        {
                            "week": len(sequence),
                            "representative": calendar_name(representative.representative),
                            "alternative": calendar_name(sequence),
                            "state_key": result.key,
                            "representative_visible_count": len(
                                representative.checkpoint.visible_values
                            ),
                            "alternative_visible_count": len(
                                result.checkpoint.visible_values
                            ),
                        }
                    )
                layer_transitions[(parent.state_id, int(action))] = Transition(
                    next_state_id=state_id,
                    appended_visible_values=appended_values,
                    appended_visible_order_ids=appended_ids,
                )

        layers.append(next_nodes)
        transitions.append(layer_transitions)
        layer_callback_inventory.append(current_layer_callbacks)
        layer_semantic_key_evaluations.append(current_layer_evaluations)
        layer_callback_record_digests.append(current_layer_callback_digest)
        layer_prefixes_with_callbacks.append(current_layer_prefixes_with_callbacks)

    return Transducer(
        weeks=weeks,
        layers=layers,
        transitions=transitions,
        collisions=collisions,
        prefix_replays=prefix_replays,
        callback_inventory=tuple(sorted(callback_inventory)),
        semantic_key_evaluations=prefix_replays,
        layer_callback_inventory=tuple(
            tuple(sorted(row)) for row in layer_callback_inventory
        ),
        layer_semantic_key_evaluations=tuple(layer_semantic_key_evaluations),
        prefix_callback_records_sha256=callback_record_digest.hexdigest(),
        layer_prefix_callback_records_sha256=tuple(
            row.hexdigest() for row in layer_callback_record_digests
        ),
        prefixes_with_nonempty_callback_inventory=prefixes_with_callbacks,
        layer_prefixes_with_nonempty_callback_inventory=tuple(
            layer_prefixes_with_callbacks
        ),
    )


_WORKER_TAPE: dict[str, Any] | None = None


def _worker_init(tape: dict[str, Any]) -> None:
    global _WORKER_TAPE
    _WORKER_TAPE = tape


def _brute_calendar(sequence: tuple[int, ...]) -> tuple[str, tuple[float, ...], tuple[int, ...], str, str]:
    if _WORKER_TAPE is None:
        raise RuntimeError("brute worker tape not initialized")
    result = run_prefix(_WORKER_TAPE, sequence)
    return (
        calendar_name(sequence),
        result.checkpoint.visible_values,
        result.checkpoint.visible_order_ids,
        result.checkpoint.primary_hex,
        result.checkpoint.endpoint_digest,
    )


def certify_exhaustive(
    tape: dict[str, Any],
    *,
    weeks: int,
    workers: int,
    max_calendars: int | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    proof_audit = runtime_proof_audit(tape)
    inventory = proof_audit["state_inventory"]
    if (
        not inventory["classification_complete"]
        or not inventory["all_frozen_invariants_hold"]
        or inventory["static_live_reads_unclassified"]
    ):
        raise RuntimeError("frozen Markov-key state inventory failed closed")
    transducer = build_transducer(tape, weeks)
    build_seconds = time.perf_counter() - started
    calendars = list(feasible_calendars(weeks))
    expected_count = feasible_calendar_count(weeks)
    if len(calendars) != expected_count:
        raise AssertionError("calendar enumerator/count disagreement")
    if max_calendars is not None:
        calendars = calendars[: int(max_calendars)]

    expected = []
    for sequence in calendars:
        values, order_ids = transducer.predict_visible_ledger(sequence)
        primary = float(np.mean(values)) if values else 1.0
        expected.append((calendar_name(sequence), values, order_ids, primary.hex()))

    replay_started = time.perf_counter()
    context = mp.get_context("fork") if workers > 1 else None
    if context is None:
        _worker_init(tape)
        actual = map(_brute_calendar, calendars)
        pool = None
    else:
        pool = context.Pool(workers, initializer=_worker_init, initargs=(tape,))
        actual = pool.imap(_brute_calendar, calendars, chunksize=max(1, len(calendars) // (workers * 32)))

    mismatch_examples: list[dict[str, Any]] = []
    compared = 0
    endpoint_hashes: dict[str, str] = {}
    try:
        for predicted, brute in zip(expected, actual):
            compared += 1
            name, predicted_values, predicted_ids, predicted_primary = predicted
            brute_name, brute_values, brute_ids, brute_primary, endpoint_digest = brute
            endpoint_hashes[brute_name] = endpoint_digest
            if (
                name != brute_name
                or predicted_values != brute_values
                or predicted_ids != brute_ids
                or predicted_primary != brute_primary
            ) and len(mismatch_examples) < 20:
                mismatch_examples.append(
                    {
                        "calendar": name,
                        "brute_calendar": brute_name,
                        "predicted_primary_hex": predicted_primary,
                        "brute_primary_hex": brute_primary,
                        "predicted_visible_count": len(predicted_values),
                        "brute_visible_count": len(brute_values),
                        "value_digest_equal": _digest(predicted_values) == _digest(brute_values),
                        "order_ids_equal": predicted_ids == brute_ids,
                    }
                )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    replay_seconds = time.perf_counter() - replay_started
    primary_certified = compared == len(calendars) and not mismatch_examples
    layer_counts = [len(layer) for layer in transducer.layers]
    return {
        "seed": int(tape["seed"]),
        "tape_sha256": tape["threat_sha256"],
        "weeks": int(weeks),
        "full_calendar_count_for_horizon": expected_count,
        "calendars_compared": compared,
        "complete_horizon_enumeration": compared == expected_count,
        "primary_transducer_bitwise_certified": primary_certified,
        "mismatch_examples": mismatch_examples,
        "state_counts_by_week": layer_counts,
        "terminal_state_count": layer_counts[-1],
        "prefix_replays": transducer.prefix_replays,
        "collision_count": len(transducer.collisions),
        "collision_examples": transducer.collisions[:20],
        "all_prefix_callback_audit": {
            "scope": "every semantic-key evaluation used to build every reachable layer",
            "semantic_key_evaluations": transducer.semantic_key_evaluations,
            "unknown_callback_owner_count": 0,
            "callback_inventory": [
                {"kind": kind, "callable": callable_name, "owner": owner}
                for kind, callable_name, owner in transducer.callback_inventory
            ],
            "layer_semantic_key_evaluations": list(
                transducer.layer_semantic_key_evaluations
            ),
            "layer_callback_inventory": [
                [
                    {"kind": kind, "callable": callable_name, "owner": owner}
                    for kind, callable_name, owner in row
                ]
                for row in transducer.layer_callback_inventory
            ],
            "prefix_callback_records_sha256": (
                transducer.prefix_callback_records_sha256
            ),
            "layer_prefix_callback_records_sha256": list(
                transducer.layer_prefix_callback_records_sha256
            ),
            "prefixes_with_nonempty_callback_inventory": (
                transducer.prefixes_with_nonempty_callback_inventory
            ),
            "layer_prefixes_with_nonempty_callback_inventory": list(
                transducer.layer_prefixes_with_nonempty_callback_inventory
            ),
            "passed": bool(
                transducer.semantic_key_evaluations == transducer.prefix_replays
                and len(transducer.layer_semantic_key_evaluations) == weeks
                and sum(transducer.layer_semantic_key_evaluations)
                == transducer.prefix_replays
                and len(transducer.layer_callback_inventory) == weeks
                and all(transducer.layer_callback_inventory)
                and transducer.prefixes_with_nonempty_callback_inventory
                == transducer.prefix_replays
                and sum(
                    transducer.layer_prefixes_with_nonempty_callback_inventory
                )
                == transducer.prefix_replays
                and len(transducer.layer_prefix_callback_records_sha256) == weeks
                and all(
                    re.fullmatch(r"[0-9a-f]{64}", row)
                    for row in transducer.layer_prefix_callback_records_sha256
                )
            ),
        },
        "transducer_build_seconds": build_seconds,
        "brute_replay_seconds": replay_seconds,
        "endpoint_replay_hash": _digest(endpoint_hashes),
        "proof_audit": proof_audit,
        "full_guardrail_label_certified": False,
        "full_guardrail_gap": (
            "The cache carries exact visible-row values/order ids only. Non-additive "
            "quantity ReT, tail, service and complete resource ledgers are available "
            "from unaccelerated endpoint hashes but are not yet reconstructible from "
            "the transducer label. Frozen full-bound execution remains prohibited."
        ),
    }


def parse_seed(value: str) -> tuple[int, str]:
    if ":" in value:
        seed_text, context = value.split(":", 1)
        if context not in CONTEXTS:
            raise argparse.ArgumentTypeError(f"unknown context {context!r}")
        return int(seed_text), context
    seed = int(value)
    # Frozen seed blocks rotate first context by offset.
    start = 1_110_001 if seed >= 1_110_001 else 1_100_001
    return seed, CONTEXTS[(seed - start) % len(CONTEXTS)]


def _write_progress(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weeks", type=int, default=12)
    parser.add_argument(
        "--seed",
        action="append",
        type=parse_seed,
        help="Burned seed, optionally seed:context. Repeat for multiple tapes.",
    )
    parser.add_argument("--split", default="transducer_development_burned")
    parser.add_argument("--workers", type=int, default=max(1, min(8, mp.cpu_count())))
    parser.add_argument("--max-calendars", type=int)
    parser.add_argument(
        "--non-scientific-smoke",
        action="store_true",
        help="Permit a dirty-source W4-or-shorter, capped NOT_EVIDENCE smoke only.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "paper2_bottleneck" / "exact_transducer_certification.json",
    )
    parser.add_argument("--progress", type=Path)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.weeks < 1 or args.weeks > 24:
        parser.error("--weeks must be in [1, 24]")
    if args.non_scientific_smoke and (
        args.weeks > 4 or args.max_calendars is None or args.max_calendars > 13
    ):
        parser.error(
            "--non-scientific-smoke requires W4 or shorter and --max-calendars <= 13"
        )
    drift = scientific_source_drift()
    if drift and not args.non_scientific_smoke:
        parser.error(
            "scientific certification requires clean tracked dependencies:\n"
            + drift
        )
    seeds = args.seed or [
        (1_110_001, CONTEXTS[0]),
        (1_100_001, CONTEXTS[0]),
        (1_100_031, CONTEXTS[0]),
    ]

    contract = json.loads(CONTRACT_PATH.read_text())
    contract_hash = sha256(CONTRACT_PATH.read_bytes()).hexdigest()
    tape_results = []
    started = time.perf_counter()
    _write_progress(
        args.progress,
        {
            "stage": "reduced_certification",
            "weeks": args.weeks,
            "split": args.split,
            "total": len(seeds),
            "completed": 0,
            "status": "RUNNING_NOT_EVIDENCE",
        },
    )
    for index, (seed, first_context) in enumerate(seeds):
        _write_progress(
            args.progress,
            {
                "stage": "reduced_certification",
                "weeks": args.weeks,
                "split": args.split,
                "total": len(seeds),
                "completed": index,
                "active_seed": seed,
                "active_context": first_context,
                "status": "RUNNING_NOT_EVIDENCE",
            },
        )
        tape = materialize_tape(
            seed,
            first_context,
            args.split,
            weeks=args.weeks,
        )
        tape_result = certify_exhaustive(
            tape,
            weeks=args.weeks,
            workers=max(1, args.workers),
            max_calendars=args.max_calendars,
        )
        tape_result["requested_first_context"] = first_context
        tape_result["split"] = args.split
        tape_results.append(tape_result)
        _write_progress(
            args.progress,
            {
                "stage": "reduced_certification",
                "weeks": args.weeks,
                "split": args.split,
                "total": len(seeds),
                "completed": index + 1,
                "last_seed": seed,
                "last_context": first_context,
                "last_tape_sha256": tape_result["tape_sha256"],
                "status": "RUNNING_NOT_EVIDENCE",
            },
        )

    all_primary = all(
        row["primary_transducer_bitwise_certified"]
        and row["complete_horizon_enumeration"]
        for row in tape_results
    )
    full_guardrails = all(row["full_guardrail_label_certified"] for row in tape_results)
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "scientific_status": (
            "NONSCIENTIFIC_SMOKE_NOT_EVIDENCE"
            if args.non_scientific_smoke
            else
            "REDUCED_HORIZON_PRIMARY_CERTIFIED_FULL_CONTRACT_FAIL_CLOSED"
            if all_primary
            else "MARKOV_KEY_CERTIFICATION_FAILED_CLOSED"
        ),
        "contract_id": contract["contract_id"],
        "contract_sha256": contract_hash,
        "key_schema_version": KEY_SCHEMA_VERSION,
        "provenance": certification_provenance(),
        "scientific_run": not args.non_scientific_smoke,
        "evidence": False if args.non_scientific_smoke else None,
        "source_drift_ignored_for_smoke": drift if args.non_scientific_smoke else "",
        "weeks": args.weeks,
        "tapes": tape_results,
        "summary": {
            "all_tapes_primary_bitwise_certified": all_primary,
            "full_guardrail_label_certified": full_guardrails,
            "full_24_week_transducer_authorized": bool(
                all_primary and full_guardrails and args.weeks == 24
            ),
            "h_pi_computed": False,
            "learner_authorized": False,
            "paper3_authorized": False,
            "not_evidence": bool(args.non_scientific_smoke),
            "elapsed_seconds": time.perf_counter() - started,
        },
        "fail_closed_reasons": [
            reason
            for condition, reason in (
                (
                    not all_primary,
                    "At least one exhaustive reduced-horizon primary comparison failed.",
                ),
                (
                    not full_guardrails,
                    "Non-additive guardrail output labels are not certified under state merging.",
                ),
                (
                    args.weeks != 24,
                    "Only a reduced horizon was evaluated.",
                ),
                (
                    True,
                    "The full 60-calibration/119-locked execution was intentionally not launched.",
                ),
            )
            if condition
        ],
        "not_claimed": [
            "H_PI",
            "H_obs",
            "Paper 2 environment",
            "boundary certificate",
            "domain validation of a fungible M/T/R team",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    _write_progress(
        args.progress,
        {
            "stage": "reduced_certification",
            "weeks": args.weeks,
            "split": args.split,
            "total": len(seeds),
            "completed": len(tape_results),
            "status": "COMPLETED_AUDIT_PENDING_NOT_EVIDENCE",
            "output_sha256": _file_sha256(args.output),
        },
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0 if all_primary else 1


if __name__ == "__main__":
    raise SystemExit(main())
