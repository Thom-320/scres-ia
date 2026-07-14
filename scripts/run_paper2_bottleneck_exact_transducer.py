#!/usr/bin/env python3
"""Fail-closed certification harness for the Paper-2 M/T/R prefix transducer.

This script is deliberately *not* a full-horizon H_PI computation.  It tests
whether prefixes of the frozen ``paper2_bottleneck_migration_v1`` simulator may
share future transitions under a conservative semantic Markov key.  The current
target is the request-snapshot-v2 order-level ReT ledger on short, already-burned
development tapes. Historical key-v4/visible-v1 certificates remain quarantined.

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
import builtins
from collections.abc import Mapping
import copy
from dataclasses import asdict, dataclass, fields, is_dataclass
import dis
from functools import lru_cache
from hashlib import sha256, sha512
from importlib import metadata as importlib_metadata
import inspect
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import re
import secrets
import shutil
import signal
import subprocess
import sys
import sysconfig
import time
import textwrap
import tempfile
import types
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import simpy.core as simpy_core
import simpy.events as simpy_events
import simpy.resources.base as simpy_resources_base
import simpy.resources.container as simpy_resources_container
import simpy.resources.resource as simpy_resources_resource
from simpy.events import PENDING

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.paper2_bottleneck import (  # noqa: E402
    ACTIONS,
    ACTION_NAMES,
    BottleneckController,
    CONTEXTS,
    make_sim,
    materialize_tape,
)
from supply_chain.program_f import ProgramFController, advance_including  # noqa: E402
from supply_chain.ret_thesis import (  # noqa: E402
    compute_order_level_ret_excel_request_snapshot_ledger,
    compute_order_level_ret_excel_visible_ledger,
)
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CONTRACT_PATH = ROOT / "contracts" / "paper2_bottleneck_full_horizon_bound_v1.json"
PRIMARY_CONTRACT_PATH = ROOT / "contracts" / "paper2_bottleneck_primary_bound_v2.json"
KEY_SCHEMA_VERSION = "paper2_bottleneck_semantic_markov_key_v5"
RESULT_SCHEMA_VERSION = "paper2_bottleneck_exact_transducer_certification_v6"
MARKOV_COMPLETENESS_SCHEMA_VERSION = "paper2_markov_completeness_v1"
PROOF_CALLABLE_BINDING_SCHEMA_VERSION = "paper2_loaded_callable_binding_v1"
REDUCED_EXECUTION_VERIFICATION_SCHEMA_VERSION = (
    "paper2_reduced_independent_execution_verification_v2"
)
REDUCED_EXECUTION_AUTHORIZATION_SCHEMA_VERSION = (
    "paper2_reduced_execution_launch_authorization_v2"
)
REDUCED_EXECUTION_RECEIPT_SCHEMA_VERSION = "paper2_reduced_execution_receipt_v2"
SCIENTIFIC_CHILD_ENVIRONMENT_KEYS = frozenset(
    {
        "PATH",
        "HOME",
        "TMPDIR",
        "LANG",
        "LC_ALL",
        "TZ",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "SCRES_SCIENTIFIC_CHILD",
    }
)
_TRUSTED_PARENT_PROCESS_NONCE = secrets.token_hex(32)
REDUCED_EXECUTION_WITNESS_EXCLUSIONS = types.MappingProxyType({
    "top_level_identity_fields": ("execution_identity",),
    "summary_timing_fields": ("elapsed_seconds",),
    "per_tape_timing_fields": (
        "transducer_build_seconds",
        "brute_replay_seconds",
    ),
})
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
SIMPY_SOURCE_MODULES = (
    simpy_core,
    simpy_events,
    simpy_resources_base,
    simpy_resources_container,
    simpy_resources_resource,
)
EVENT_FIELD_ALLOWLIST = {
    "simpy.events.Event": frozenset({"env", "callbacks", "_ok", "_defused", "_value"}),
    "simpy.events.Timeout": frozenset(
        {"env", "callbacks", "_ok", "_defused", "_value", "_delay"}
    ),
    "simpy.resources.container.ContainerGet": frozenset(
        {
            "env",
            "callbacks",
            "_ok",
            "_defused",
            "_value",
            "resource",
            "proc",
            "amount",
        }
    ),
    "simpy.resources.container.ContainerPut": frozenset(
        {
            "env",
            "callbacks",
            "_ok",
            "_defused",
            "_value",
            "resource",
            "proc",
            "amount",
        }
    ),
    "simpy.resources.resource.Request": frozenset(
        {
            "env",
            "callbacks",
            "_ok",
            "_defused",
            "_value",
            "resource",
            "proc",
            "usage_since",
        }
    ),
    "simpy.resources.resource.PriorityRequest": frozenset(
        {
            "env",
            "callbacks",
            "_ok",
            "_defused",
            "_value",
            "resource",
            "proc",
            "usage_since",
            "priority",
            "preempt",
            "time",
            "key",
        }
    ),
}
PROCESS_FIELD_ALLOWLIST = frozenset(
    {"env", "callbacks", "_generator", "_target", "_ok", "_defused", "_value"}
)
REDUCED_CERTIFICATION_SUITES = {
    "w12_five_tape": {
        "weeks": 12,
        "split": "transducer_collision_suite_burned",
        "tapes": (
            (
                1_110_001,
                "equipment_pressure",
                "ebefe74394a04ee08e122e99452a4b5c1ef23c4515c8666e47f5a737f2c39d2c",
            ),
            (
                1_100_001,
                "equipment_pressure",
                "c56c36a09a04eb7d677e4c42a56e75570e98564e0ee744a47592901144c1df7f",
            ),
            (
                1_100_031,
                "equipment_pressure",
                "ef9bdbbcdc0096eee9f1c0ddffcda02a07230ef1253a703469e8c921ae7acea3",
            ),
            (
                1_110_061,
                "equipment_pressure",
                "b1e8b96c1346263f8e09ae3bfa659765ee60f9ddca96a910b2d0274256a1c31e",
            ),
            (
                1_110_120,
                "mission_surge",
                "1d9331409bf4fc6f8842029bcb8d57b52de505a75be66d7fe6ee96963af2399f",
            ),
        ),
    },
    "w16_hard_tape": {
        "weeks": 16,
        "split": "transducer_hard_state_burned",
        "tapes": (
            (
                1_110_025,
                "equipment_pressure",
                "3e3056f28c8afcea663388652d6aab3a904bfaa1883f8b6349976e6affb8c51d",
            ),
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
    "_ret_ledger_snapshot_sequence",
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
    "seed",
    "seed_stream_mode",
    "strict_exogenous_crn",
    "horizon",
    "shifts",
    "hours_per_year",
    "year_basis",
    "risks_enabled",
    "risk_level",
    "risk_occurrence_mode",
    "risk_attribution_source",
    "risk_event_tape",
    "enabled_risks",
    "risk_overrides",
    "risk_rng_mode",
    "risk_frequency_multiplier",
    "risk_frequency_multipliers_by_id",
    "risk_impact_multiplier",
    "risk_impact_multipliers_by_id",
    "stochastic_pt",
    "deterministic_baseline",
    "warmup_trigger",
    "downstream_q_source",
    "raw_material_flow_mode",
    "raw_material_order_up_to_multiplier",
    "_raw_units_per_ration",
    "demand_source",
    "excel_order_tape",
    "demand_mean_multiplier",
    "demand_on_hand_fulfillment_delay",
    "demand_start_after_warmup",
    "assembly_flow_mode",
    "serial_wip_capacity_rations",
    "periodic_release_mode",
    "op2_release_clock_mode",
    "operational_risk_initialization_mode",
    "procurement_contract_mode",
    "order_fulfillment_mode",
    "op9_dispatch_policy",
    "downstream_transport_capacity_mode",
    "op9_freight_offset_hours",
    "replenishment_route_aware",
    "ret_recovery_period_mode",
    "backorder_overflow_mode",
    "backorder_priority_rule",
    "backorder_age_threshold_hours",
    "r14_defect_mode",
    "r24_attribution_window_hours",
    "material_lineage_mode",
    "cssu_topology_mode",
    "cssu_allocation_a",
    "cssu_service_rule",
    "cssu_daily_capacity_override",
    "op8_dispatch_mode",
    "op8_convoy_capacity",
    "op8_convoy_outbound_hours",
    "op8_convoy_return_hours",
    "inventory_buffer_targets",
    "inventory_replenishment_period",
    "inventory_replenishment_lead_time",
    "_op5_rm_base",
    "_op5_multiplier_rule",
    "campaign_config",
    "campaign_path",
    "emergency_reserve_enabled",
    "emergency_reserve_capacity",
    "emergency_reserve_replenishment_lead_time",
    "emergency_reserve_issue_delay",
    "emergency_reserve_route_ops",
    "emergency_reserve_transport_mode",
    "program_f_reserve_enabled",
    "risk_recovery_window_hours",
    "risk_recovery_release_rations",
    "risk_recovery_boost_downstream",
    "risk_recovery_enabled_risks",
    "_step_size",
    "_processes_started",
}
INERT_FROZEN_FIELDS = {
    "adaptive_benchmark_enabled",
    "adaptive_benchmark_v2_enabled",
    "adaptive_regime",
    "adaptive_risk_forecast_48h",
    "adaptive_risk_forecast_168h",
    "cssu_in_transit",
    "cssu_inbound_in_transit",
    "cssu_inventory",
    "cssu_outbound_in_transit",
    "cssu_delivered",
    "cssu_demanded",
    "cssu_dispatched",
    "cssu_allocation_live_epochs",
    "cssu_allocation_moot_epochs",
    "cssu_action_events",
    "cssu_demand_events",
    "cssu_delivery_events",
    "cssu_local_risk_events",
    "op8_convoy_available",
    "op8_convoy_departures",
    "op8_convoy_dispatched_rations",
    "op8_convoy_capacity_committed",
    "op8_convoy_vehicle_hours",
    "op8_convoy_idle_hours",
    "op8_convoy_route_wait_hours",
    "op8_convoy_ration_hours_in_transit",
    "op8_convoy_masked_dispatch_attempts",
    "op8_convoy_hold_actions",
    "op8_convoy_dispatch_actions",
    "op8_convoy_last_departure_at",
    "op8_convoy_nominal_return_at",
    "op8_convoy_actual_return_at",
    "op8_staging_first_ready_at",
    "op8_last_action",
    "op8_convoy_action_events",
    "op8_convoy_departure_events",
}
OUTPUT_OR_REPLAY_FIELDS = {
    "contract_completion_events",
    "supplier_delivery_events",
    "material_availability_events",
    "backorder_priority_rule_events",
    "total_external_raw_material",
    "total_strategic_raw_injected",
    "total_strategic_rations_injected",
    "total_rations_created_from_raw",
    "total_rations_scrapped",
    "total_raw_material_consumed",
    "total_order_fulfilled",
    "total_theatre_inflow",
    "total_produced",
    "total_delivered",
    "total_demanded",
    "total_backorders",
    "cumulative_backorder_qty",
    "warmup_complete",
    "warmup_time",
    "_cumulative_available_assembly_hours",
    "_cumulative_down_hours",
    "_prev_step_produced",
    "_prev_step_delivered",
    "_prev_step_available_assembly_hours",
    "_prev_step_fill_rate",
    "_ewma_fill_rate",
    "_ewma_backlog_growth",
    "_delta_fill_rate",
    "_delta_backlog_momentum",
    "_prev_pending_backorder_qty",
    "daily_production",
    "daily_demand",
    "delivery_events",
    "daily_inventory_sb",
    "daily_inventory_theatre",
    "emergency_reserve_target_changes",
    "program_f_reserve_issue_events",
}

# The quotient theorem is deliberately scoped to the primary visible-order ReT
# transition system.  These controller fields are the complete live schema of
# ``BottleneckController``.  The first group can affect future physical state;
# the second is bound immutably for a tape; the third is append/accounting state
# that is excluded from the primary Markov projection.  The exclusion is bound
# to the exact controller/source hashes and reachable-method audit in the
# Markov-completeness certificate below.
CONTROLLER_TRANSITION_FIELDS = frozenset(
    {
        "active_action",
        "pending_action",
        "current_week",
        "last_switch_week",
        "condition",
    }
)
CONTROLLER_IMMUTABLE_ROOT_FIELDS = frozenset({"sim", "tape", "profile", "start"})
CONTROLLER_PRIMARY_OUTPUT_FIELDS = frozenset(
    {
        "action_events",
        "damage_events",
        "consumed_base_events",
        "maintenance_downtime_hours",
        "token_hours",
        "rejected_switches",
    }
)
# Exact source-level uses of controller fields projected out of the primary
# transition key.  A new read, a duplicated read, or a different access form is
# a new proof obligation: the certificate fails until the access is reviewed
# and this inventory is deliberately updated.  In particular, the only
# semantic read of damage history is in ``observation()``, which the open-loop
# exact runner never invokes.
CONTROLLER_PRIMARY_OUTPUT_ACCESS_ALLOWLIST = (
    ("BottleneckController", "activate_week", "action_events", "Load", "Attribute", 1),
    ("BottleneckController", "activate_week", "token_hours", "Load", "Subscript", 1),
    ("ProgramFController", "__init__", "action_events", "Store", "AnnAssign", 1),
    ("ProgramFController", "__init__", "consumed_base_events", "Store", "AnnAssign", 1),
    ("ProgramFController", "__init__", "damage_events", "Store", "AnnAssign", 1),
    (
        "ProgramFController",
        "__init__",
        "maintenance_downtime_hours",
        "Store",
        "Assign",
        1,
    ),
    ("ProgramFController", "__init__", "rejected_switches", "Store", "Assign", 1),
    ("ProgramFController", "__init__", "token_hours", "Store", "Assign", 1),
    (
        "ProgramFController",
        "_planned_maintenance",
        "maintenance_downtime_hours",
        "Store",
        "AugAssign",
        1,
    ),
    (
        "ProgramFController",
        "_threat_process",
        "consumed_base_events",
        "Load",
        "Attribute",
        1,
    ),
    ("ProgramFController", "_threat_process", "damage_events", "Load", "Attribute", 1),
    ("ProgramFController", "activate_week", "action_events", "Load", "Attribute", 1),
    ("ProgramFController", "activate_week", "token_hours", "Load", "Subscript", 1),
    ("ProgramFController", "observation", "damage_events", "Load", "comprehension", 1),
    ("ProgramFController", "request", "rejected_switches", "Store", "AugAssign", 1),
)
CONTROLLER_FIELD_ALLOWLIST = frozenset(
    CONTROLLER_TRANSITION_FIELDS
    | CONTROLLER_IMMUTABLE_ROOT_FIELDS
    | CONTROLLER_PRIMARY_OUTPUT_FIELDS
)
ENVIRONMENT_FIELD_ALLOWLIST = frozenset(
    {
        "_active_proc",
        "_eid",
        "_now",
        "_queue",
        "all_of",
        "any_of",
        "event",
        "process",
        "timeout",
    }
)
RESOURCE_FIELD_ALLOWLIST = frozenset(
    {
        "_capacity",
        "_env",
        "get_queue",
        "put_queue",
        "queue",
        "release",
        "request",
        "users",
    }
)
CONTAINER_FIELD_ALLOWLIST = frozenset(
    {"_capacity", "_env", "_level", "get", "get_queue", "put", "put_queue"}
)

_PROOF_CALLABLE_ROOT_NAMES: tuple[str, ...] = ()
_SCIENTIFIC_PROOF_ENTRYPOINT_NAMES = (
    "certify_exhaustive",
    "build_transducer",
    "run_prefix",
    "run_prefix_boundary_extension",
    "_brute_calendar",
    "semantic_markov_payload",
    "semantic_markov_fingerprint",
    "runtime_proof_audit",
    "markov_completeness_certificate",
    "validate_markov_completeness_certificate",
    "validate_collision_bisimulation_certificate",
)
_PROOF_CALLABLE_BASELINE: dict[str, Any] | None = None
_PROOF_METHOD_BASELINE: dict[str, Any] | None = None
_PROOF_CLASS_GLOBAL_BASELINE: dict[str, Any] | None = None
_PROOF_BUILTIN_SPECS: tuple[tuple[Any, ...], ...] = ()
_PROOF_BUILTIN_BASELINE: dict[str, Any] | None = None
_PROOF_CLASS_ROOTS: tuple[type[Any], ...] = ()
_PROOF_CLASS_SOURCE_MODULES: tuple[tuple[type[Any], Any], ...] = ()
_PROOF_CLASS_MRO_ALLOWLIST: frozenset[type[Any]] = frozenset()
_PROOF_CLASS_TOPOLOGY_BASELINE: dict[str, Any] | None = None
_PROOF_SOURCE_CLASS_ATTESTATION_BASELINE: dict[str, Any] | None = None
_PROOF_FAST_CALLABLE_SPECS: tuple[tuple[Any, tuple[str, ...]], ...] = ()
_PROOF_FAST_UNIQUE_FUNCTIONS: tuple[Any, ...] = ()
_PROOF_MODULE_ATTRIBUTE_SPECS: tuple[tuple[Any, ...], ...] = ()
_PROOF_MODULE_ATTRIBUTE_BASELINE: dict[str, Any] | None = None
_PROOF_SOURCE_CODE_ATTESTATION_BASELINE: dict[str, Any] | None = None
_PROOF_FAST_BASELINE: tuple[Any, ...] | None = None
_PROOF_DYNAMIC_GLOBAL_EXCLUSIONS = frozenset(
    {
        "_PROOF_CALLABLE_BASELINE",
        "_PROOF_METHOD_BASELINE",
        "_PROOF_CLASS_GLOBAL_BASELINE",
        "_PROOF_BUILTIN_SPECS",
        "_PROOF_BUILTIN_BASELINE",
        "_PROOF_CLASS_ROOTS",
        "_PROOF_CLASS_SOURCE_MODULES",
        "_PROOF_CLASS_MRO_ALLOWLIST",
        "_PROOF_CLASS_TOPOLOGY_BASELINE",
        "_PROOF_SOURCE_CLASS_ATTESTATION_BASELINE",
        "_PROOF_FAST_CALLABLE_SPECS",
        "_PROOF_FAST_UNIQUE_FUNCTIONS",
        "_PROOF_MODULE_ATTRIBUTE_SPECS",
        "_PROOF_MODULE_ATTRIBUTE_BASELINE",
        "_PROOF_SOURCE_CODE_ATTESTATION_BASELINE",
        "_PROOF_FAST_BASELINE",
        "_WORKER_TAPE",
    }
)
_EXTERNAL_PROOF_CALLABLE_ROOTS = frozenset(
    {
        "advance_including",
        "compute_episode_metrics",
        "compute_order_level_ret_excel_request_snapshot_ledger",
        "compute_order_level_ret_excel_visible_ledger",
        "make_sim",
        "materialize_tape",
    }
)
_PROOF_CALLABLE_MODULE_PREFIXES = (
    "scripts.run_paper2_bottleneck_exact_transducer",
    "simpy.",
    "supply_chain.",
)

# Key-v3 deliberately serializes every live mutable diagnostic/history field as
# well as the narrowly transition-facing fields above.  Some are probably
# output-only under this frozen contract, but omitting them would require a
# theorem-strength dynamic call-graph exclusion.  Keeping them in the bytes is
# conservative: it can reduce compression, never create a false merge.
SIM_STATE_FIELDS = tuple(
    dict.fromkeys(
        SIM_STATE_FIELDS
        + tuple(sorted(INERT_FROZEN_FIELDS))
        + tuple(sorted(OUTPUT_OR_REPLAY_FIELDS))
    )
)


def _collect_runtime_proof_root_classes() -> tuple[type[Any], ...]:
    roots: list[type[Any]] = [
        MFSCSimulation,
        ProgramFController,
        BottleneckController,
    ]
    for module in SIMPY_SOURCE_MODULES:
        roots.extend(
            value
            for value in vars(module).values()
            if inspect.isclass(value) and value.__module__ == module.__name__
        )
    return tuple(dict.fromkeys(roots))


def _runtime_proof_classes() -> tuple[type[Any], ...]:
    """Return target and SimPy runtime classes, including their SimPy MROs."""
    roots = _PROOF_CLASS_ROOTS or _collect_runtime_proof_root_classes()
    rows: list[type[Any]] = []
    for root in roots:
        for cls in root.__mro__:
            if cls is object:
                continue
            if cls in {MFSCSimulation, ProgramFController, BottleneckController} or str(
                cls.__module__
            ).startswith("simpy."):
                rows.append(cls)
    return tuple(dict.fromkeys(rows))


def static_class_attribute_reads(cls: type[Any]) -> frozenset[str]:
    """Conservative inventory over the exact currently loaded method bodies."""
    reads: set[str] = set()
    for descriptor in vars(cls).values():
        function = (
            descriptor.__func__
            if isinstance(descriptor, (staticmethod, classmethod))
            else descriptor
        )
        if not inspect.isfunction(function):
            continue
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.ctx, ast.Load)
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"
            ):
                reads.add(node.attr)
    return frozenset(reads)


def static_sim_attribute_reads() -> set[str]:
    return set(static_class_attribute_reads(MFSCSimulation))


def static_class_method_attribute_reads(cls: type[Any]) -> dict[str, tuple[str, ...]]:
    """Map exact loaded methods to direct ``self.<attr>`` reads."""
    rows: dict[str, tuple[str, ...]] = {}
    for method_name, descriptor in vars(cls).items():
        function = (
            descriptor.__func__
            if isinstance(descriptor, (staticmethod, classmethod))
            else descriptor
        )
        if not inspect.isfunction(function):
            continue
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        reads = {
            item.attr
            for item in ast.walk(tree)
            if (
                isinstance(item, ast.Attribute)
                and isinstance(item.ctx, ast.Load)
                and isinstance(item.value, ast.Name)
                and item.value.id == "self"
            )
        }
        rows[method_name] = tuple(sorted(reads))
    return rows


def _controller_primary_output_access_inventory() -> tuple[tuple[Any, ...], ...]:
    """Return counted, syntax-sensitive accesses to projected controller state."""
    counts: dict[tuple[str, str, str, str, str], int] = {}
    for cls in (BottleneckController, ProgramFController):
        for method_name, descriptor in vars(cls).items():
            function = (
                descriptor.__func__
                if isinstance(descriptor, (staticmethod, classmethod))
                else descriptor
            )
            if not inspect.isfunction(function):
                continue
            tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
            parents = {
                child: parent
                for parent in ast.walk(tree)
                for child in ast.iter_child_nodes(parent)
            }
            for node in ast.walk(tree):
                if not (
                    isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "self"
                    and node.attr in CONTROLLER_PRIMARY_OUTPUT_FIELDS
                ):
                    continue
                row = (
                    cls.__name__,
                    method_name,
                    node.attr,
                    type(node.ctx).__name__,
                    type(parents.get(node)).__name__,
                )
                counts[row] = counts.get(row, 0) + 1
    return tuple(sorted((*row, count) for row, count in counts.items()))


def _controller_reflection_inventory() -> tuple[tuple[str, str, int], ...]:
    """Reject reflective field access that could bypass the AST allowlist."""
    reflective_calls = {
        "getattr",
        "setattr",
        "delattr",
        "vars",
        "eval",
        "exec",
        "hasattr",
    }
    rows: list[tuple[str, str, int]] = []
    for cls in (BottleneckController, ProgramFController):
        for descriptor in vars(cls).values():
            function = (
                descriptor.__func__
                if isinstance(descriptor, (staticmethod, classmethod))
                else descriptor
            )
            if not inspect.isfunction(function):
                continue
            tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id in reflective_calls
                ):
                    rows.append((cls.__name__, node.func.id, int(node.lineno)))
                elif isinstance(node, ast.Attribute) and node.attr == "__dict__":
                    rows.append((cls.__name__, "__dict__", int(node.lineno)))
    return tuple(sorted(rows))


def _nested_code_objects(code: types.CodeType) -> Iterator[types.CodeType]:
    yield code
    for constant in code.co_consts:
        if isinstance(constant, types.CodeType):
            yield from _nested_code_objects(constant)


def _code_constant_binding_token(value: Any) -> Any:
    """Return a deterministic token for immutable code-object constants."""
    if isinstance(value, types.CodeType):
        return ("code", _code_binding_token(value))
    if value is Ellipsis:
        return ("ellipsis",)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return ("float", value.hex())
    if isinstance(value, complex):
        return ("complex", value.real.hex(), value.imag.hex())
    if isinstance(value, bytes):
        return ("bytes", value.hex())
    if isinstance(value, tuple):
        return (
            "tuple",
            tuple(_code_constant_binding_token(item) for item in value),
        )
    if isinstance(value, frozenset):
        return (
            "frozenset",
            tuple(
                sorted(
                    (_code_constant_binding_token(item) for item in value),
                    key=repr,
                )
            ),
        )
    raise TypeError(
        "unsupported immutable code constant: "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _code_binding_token(code: types.CodeType) -> tuple[Any, ...]:
    """Bind all semantics-bearing fields of a Python code object recursively."""
    return (
        code.co_name,
        code.co_qualname,
        int(code.co_argcount),
        int(code.co_posonlyargcount),
        int(code.co_kwonlyargcount),
        int(code.co_nlocals),
        int(code.co_stacksize),
        int(code.co_flags),
        sha256(code.co_code).hexdigest(),
        tuple(_code_constant_binding_token(item) for item in code.co_consts),
        tuple(code.co_names),
        tuple(code.co_varnames),
        tuple(code.co_freevars),
        tuple(code.co_cellvars),
        int(code.co_firstlineno),
        sha256(code.co_linetable).hexdigest(),
        sha256(code.co_exceptiontable).hexdigest(),
    )


@lru_cache(maxsize=None)
def _code_binding_sha256(code: types.CodeType) -> str:
    """Cache immutable code manifests without weakening replacement detection."""
    return _digest(_code_binding_token(code))


def _global_value_binding(
    value: Any,
    *,
    _active: frozenset[int] = frozenset(),
) -> Any:
    """Return a stable token for one bytecode-loaded module global."""
    tracked = isinstance(value, (tuple, set, frozenset, dict)) or inspect.isfunction(
        value
    ) or inspect.isclass(value) or inspect.isfunction(getattr(value, "__wrapped__", None))
    if tracked and id(value) in _active:
        return (
            "recursive_ref",
            str(getattr(value, "__module__", type(value).__module__)),
            str(getattr(value, "__qualname__", type(value).__qualname__)),
        )
    if tracked:
        _active = _active | frozenset({id(value)})
    if value is PENDING:
        return ("sentinel", "simpy.events.PENDING")
    if value is Ellipsis:
        return ("sentinel", "builtins.Ellipsis")
    if value is NotImplemented:
        return ("sentinel", "builtins.NotImplemented")
    if isinstance(value, Path):
        return ("path", str(value))
    if isinstance(value, tuple):
        return (
            "tuple",
            tuple(_global_value_binding(item, _active=_active) for item in value),
        )
    if isinstance(value, (set, frozenset)):
        return (
            type(value).__name__,
            tuple(
                sorted(
                    (_global_value_binding(item, _active=_active) for item in value),
                    key=repr,
                )
            ),
        )
    if isinstance(value, dict):
        return (
            "dict",
            tuple(
                sorted(
                    (
                        (
                            _global_value_binding(key, _active=_active),
                            _global_value_binding(item, _active=_active),
                        )
                        for key, item in value.items()
                    ),
                    key=repr,
                )
            ),
        )
    if isinstance(value, Mapping):
        item_tokens = tuple(
            sorted(
                (
                    (
                        _global_value_binding(key, _active=_active),
                        _global_value_binding(item, _active=_active),
                    )
                    for key, item in value.items()
                ),
                key=repr,
            )
        )
        return (
            "mapping_digest",
            f"{type(value).__module__}.{type(value).__qualname__}",
            _digest(item_tokens),
        )
    if isinstance(value, types.SimpleNamespace):
        return (
            "simple_namespace",
            _global_value_binding(vars(value), _active=_active),
        )
    if (
        type(value).__module__ == "numpy._globals"
        and type(value).__qualname__ == "_NoValueType"
    ):
        return ("sentinel", "numpy._globals._NoValue")
    if (
        type(value).__module__ == "dataclasses"
        and type(value).__qualname__
        in {"_HAS_DEFAULT_FACTORY_CLASS", "_KW_ONLY_TYPE", "_MISSING_TYPE"}
    ):
        return ("sentinel", f"dataclasses.{type(value).__qualname__}")
    try:
        return ("value", _semantic(value))
    except TypeError:
        pass
    if inspect.ismodule(value):
        return (
            "module",
            value.__name__,
            str(getattr(value, "__version__", "UNVERSIONED")),
            str(getattr(value, "__file__", "BUILTIN")),
        )
    if inspect.isfunction(value):
        return ("function", _loaded_callable_token(value, _active=_active))
    wrapped = getattr(value, "__wrapped__", None)
    if inspect.isfunction(wrapped):
        return (
            "wrapped_function",
            f"{type(value).__module__}.{type(value).__qualname__}",
            _loaded_callable_token(wrapped, _active=_active),
        )
    if inspect.isclass(value):
        method_bindings = []
        for name, descriptor in sorted(vars(value).items()):
            member = (
                descriptor.__func__
                if isinstance(descriptor, (staticmethod, classmethod))
                else descriptor
            )
            if inspect.isfunction(member):
                method_bindings.append(
                    (name, _loaded_callable_token(member, _active=_active))
                )
            elif isinstance(descriptor, property):
                accessors = tuple(
                    (kind, _loaded_callable_token(function, _active=_active))
                    for kind, function in (
                        ("get", descriptor.fget),
                        ("set", descriptor.fset),
                        ("delete", descriptor.fdel),
                    )
                    if inspect.isfunction(function)
                )
                if accessors:
                    method_bindings.append((name, "property", accessors))
        annotation_value = vars(value).get("__annotations__", {})
        annotations = (
            tuple(
                (name, str(annotation))
                for name, annotation in sorted(annotation_value.items())
            )
            if isinstance(annotation_value, Mapping)
            else (
                (
                    "descriptor",
                    f"{type(annotation_value).__module__}."
                    f"{type(annotation_value).__qualname__}",
                ),
            )
        )
        return (
            "class",
            value.__module__,
            value.__qualname__,
            tuple(method_bindings),
            annotations,
        )
    if callable(value):
        try:
            signature = str(inspect.signature(value))
        except (TypeError, ValueError):
            signature = "SIGNATURE_UNAVAILABLE"
        doc = inspect.getdoc(value) or ""
        return (
            "native_callable",
            str(getattr(value, "__module__", type(value).__module__)),
            str(getattr(value, "__qualname__", type(value).__qualname__)),
            f"{type(value).__module__}.{type(value).__qualname__}",
            signature,
            sha256(doc.encode()).hexdigest(),
        )
    raise TypeError(
        "unsupported runtime global binding: "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _class_member_function_rows(cls: type[Any]) -> tuple[tuple[str, str, Any], ...]:
    """Return every Python function with its exact class descriptor kind."""
    rows: list[tuple[str, str, Any]] = []
    for name, descriptor in sorted(vars(cls).items()):
        if inspect.isfunction(descriptor):
            rows.append((name, "plain_function", descriptor))
        elif isinstance(descriptor, staticmethod) and inspect.isfunction(
            descriptor.__func__
        ):
            rows.append((name, "staticmethod", descriptor.__func__))
        elif isinstance(descriptor, classmethod) and inspect.isfunction(
            descriptor.__func__
        ):
            rows.append((name, "classmethod", descriptor.__func__))
        elif isinstance(descriptor, property):
            rows.extend(
                (f"{name}.{accessor}", f"property_{accessor}", function)
                for accessor, function in (
                    ("fget", descriptor.fget),
                    ("fset", descriptor.fset),
                    ("fdel", descriptor.fdel),
                )
                if inspect.isfunction(function)
            )
    return tuple(rows)


def _class_member_descriptor_kind(descriptor: Any) -> str:
    if inspect.isfunction(descriptor):
        return "plain_function"
    if isinstance(descriptor, staticmethod):
        return "staticmethod"
    if isinstance(descriptor, classmethod):
        return "classmethod"
    if isinstance(descriptor, property):
        return "property"
    return f"data:{type(descriptor).__module__}.{type(descriptor).__qualname__}"


def _class_data_member_binding(
    value: Any,
    *,
    _active: frozenset[int] = frozenset(),
) -> Any:
    """Bind non-method class members without recursively expanding mappingproxy."""
    if id(value) in _active:
        return (
            "recursive_descriptor_ref",
            f"{type(value).__module__}.{type(value).__qualname__}",
        )
    if inspect.isclass(value) or inspect.isfunction(value) or inspect.ismodule(value):
        return _global_value_binding(value)
    try:
        fields = vars(value)
    except TypeError:
        fields = None
    if fields is not None:
        active = _active | frozenset({id(value)})
        return (
            "mutable_descriptor_state",
            f"{type(value).__module__}.{type(value).__qualname__}",
            tuple(
                sorted(
                    (
                        name,
                        _class_data_member_binding(item, _active=active),
                    )
                    for name, item in fields.items()
                )
            ),
        )
    try:
        return _global_value_binding(value)
    except (TypeError, ValueError):
        return (
            "opaque_class_member",
            f"{type(value).__module__}.{type(value).__qualname__}",
            str(getattr(value, "__name__", "UNNAMED")),
            str(
                getattr(
                    getattr(value, "__objclass__", None),
                    "__qualname__",
                    "NO_OBJCLASS",
                )
            ),
        )


def _class_member_descriptor_binding(descriptor: Any) -> tuple[Any, ...]:
    kind = _class_member_descriptor_kind(descriptor)
    if inspect.isfunction(descriptor):
        return (kind, _loaded_callable_token(descriptor))
    if isinstance(descriptor, (staticmethod, classmethod)):
        return (kind, _loaded_callable_token(descriptor.__func__))
    if isinstance(descriptor, property):
        return (
            kind,
            tuple(
                (
                    accessor,
                    _loaded_callable_token(function)
                    if inspect.isfunction(function)
                    else None,
                )
                for accessor, function in (
                    ("fget", descriptor.fget),
                    ("fset", descriptor.fset),
                    ("fdel", descriptor.fdel),
                )
            ),
        )
    return (kind, _class_data_member_binding(descriptor))


@lru_cache(maxsize=1)
def _runtime_global_binding_specs() -> tuple[tuple[type[Any], str, str], ...]:
    specs: list[tuple[type[Any], str, str]] = []
    for cls in _runtime_proof_classes():
        for method_name, descriptor in sorted(vars(cls).items()):
            function = (
                descriptor.__func__
                if isinstance(descriptor, (staticmethod, classmethod))
                else descriptor
            )
            if not inspect.isfunction(function):
                continue
            loaded_names = {
                str(instruction.argval)
                for code in _nested_code_objects(function.__code__)
                for instruction in dis.get_instructions(code)
                if instruction.opname in {"LOAD_GLOBAL", "LOAD_NAME"}
                and str(instruction.argval) in function.__globals__
            }
            for global_name in sorted(loaded_names):
                specs.append((cls, method_name, global_name))
    return tuple(specs)


def _runtime_global_binding_inventory() -> dict[str, Any]:
    """Bind globals actually loaded by reachable simulator/controller methods."""
    rows: list[tuple[Any, ...]] = []
    unsupported: list[tuple[str, str, str]] = []
    token_by_identity: dict[int, Any] = {}
    for cls, method_name, global_name in _runtime_global_binding_specs():
        descriptor = vars(cls).get(method_name)
        function = (
            descriptor.__func__
            if isinstance(descriptor, (staticmethod, classmethod))
            else descriptor
        )
        if not inspect.isfunction(function) or global_name not in function.__globals__:
            unsupported.append((cls.__name__, method_name, global_name))
            continue
        value = function.__globals__[global_name]
        try:
            token = token_by_identity.get(id(value))
            if token is None:
                token = _global_value_binding(value)
                token_by_identity[id(value)] = token
        except TypeError:
            unsupported.append(
                (
                    cls.__name__,
                    method_name,
                    global_name,
                )
            )
            continue
        rows.append((cls.__name__, method_name, global_name, token))
    payload = {
        "rows": tuple(rows),
        "unsupported": tuple(sorted(unsupported)),
    }
    payload["binding_sha256"] = _digest(payload)
    return payload


def _runtime_method_binding_inventory() -> dict[str, Any]:
    """Bind every class member, including exact callable descriptor kind."""
    rows = []
    for cls in _runtime_proof_classes():
        for member_name, descriptor in sorted(vars(cls).items()):
            rows.append(
                (
                    cls.__module__,
                    cls.__qualname__,
                    member_name,
                    _class_member_descriptor_binding(descriptor),
                )
            )
    payload = {"rows": tuple(rows)}
    payload["binding_sha256"] = _digest(payload)
    return payload


def _loaded_callable_token(
    function: Any,
    *,
    _active: frozenset[int] = frozenset(),
) -> tuple[Any, ...]:
    """Bind one exact loaded Python callable to both source and code."""
    if not inspect.isfunction(function):
        raise TypeError("proof callable is not a loaded Python function")
    try:
        source = textwrap.dedent(inspect.getsource(function))
        source_sha256 = sha256(source.encode()).hexdigest()
    except (OSError, TypeError):
        source_sha256 = "SOURCE_UNAVAILABLE"
    return (
        function.__module__,
        function.__qualname__,
        source_sha256,
        _code_binding_token(function.__code__),
        _global_value_binding(
            getattr(function, "__defaults__", None), _active=_active
        ),
        _global_value_binding(
            getattr(function, "__kwdefaults__", None), _active=_active
        ),
        _closure_binding_token(function, _active=_active),
    )


@lru_cache(maxsize=None)
def _loaded_callable_source_sha256(function: Any) -> str:
    """Cache immutable loaded-source text for the per-key fast signature."""
    try:
        source = textwrap.dedent(inspect.getsource(function))
    except (OSError, TypeError):
        return "SOURCE_UNAVAILABLE"
    return sha256(source.encode()).hexdigest()


def _as_loaded_function(value: Any) -> Any | None:
    if inspect.isfunction(value):
        return value
    wrapped = getattr(value, "__wrapped__", None)
    return wrapped if inspect.isfunction(wrapped) else None


def _loaded_callable_global_rows(function: Any) -> tuple[tuple[str, Any], ...]:
    loaded_names = {
        str(instruction.argval)
        for code in _nested_code_objects(function.__code__)
        for instruction in dis.get_instructions(code)
        if instruction.opname in {"LOAD_GLOBAL", "LOAD_NAME"}
        and str(instruction.argval) in function.__globals__
        and str(instruction.argval) not in _PROOF_DYNAMIC_GLOBAL_EXCLUSIONS
    }
    return tuple(
        (
            name,
            _global_value_binding(function.__globals__[name]),
        )
        for name in sorted(loaded_names)
    )


def _callable_dependency_allowed(function: Any) -> bool:
    module = str(getattr(function, "__module__", ""))
    return module == __name__ or module.startswith(_PROOF_CALLABLE_MODULE_PREFIXES)


def _runtime_proof_callable_inventory() -> dict[str, Any]:
    """Inventory the loaded transition/certificate call graph fail-closed."""
    rows: list[tuple[Any, ...]] = []
    unsupported: list[tuple[str, str]] = []
    queue: list[tuple[str, Any]] = []
    for name in _PROOF_CALLABLE_ROOT_NAMES:
        value = globals().get(name)
        function = _as_loaded_function(value)
        if function is not None:
            queue.append((f"root:{name}", function))
        else:
            unsupported.append((f"root:{name}", "not_a_loaded_function"))
    seen: set[int] = set()
    reachable_callable_bindings: list[tuple[str, str, str, str]] = []
    while queue:
        origin, function = queue.pop(0)
        if id(function) in seen:
            rows.append((origin, "alias", function.__module__, function.__qualname__))
            continue
        seen.add(id(function))
        reachable_callable_bindings.append(
            (
                function.__module__,
                function.__qualname__,
                _loaded_callable_source_sha256(function),
                _digest(_code_binding_token(function.__code__)),
            )
        )
        try:
            global_rows = _loaded_callable_global_rows(function)
            token = _loaded_callable_token(function)
        except TypeError as exc:
            unsupported.append((origin, str(exc)))
            continue
        rows.append((origin, "callable", token, global_rows))
        for global_name in sorted(
            {
                str(instruction.argval)
                for code in _nested_code_objects(function.__code__)
                for instruction in dis.get_instructions(code)
                if instruction.opname in {"LOAD_GLOBAL", "LOAD_NAME"}
                and str(instruction.argval) in function.__globals__
            }
        ):
            dependency = function.__globals__[global_name]
            dependency_function = _as_loaded_function(dependency)
            if dependency_function is not None and _callable_dependency_allowed(
                dependency_function
            ):
                queue.append(
                    (
                        f"dependency:{function.__module__}.{function.__qualname__}"
                        f"->{global_name}",
                        dependency_function,
                    )
                )
    payload = {
        "schema_version": PROOF_CALLABLE_BINDING_SCHEMA_VERSION,
        "root_names": _PROOF_CALLABLE_ROOT_NAMES,
        "reachable_callable_bindings": tuple(sorted(reachable_callable_bindings)),
        "rows": tuple(rows),
        "unsupported": tuple(sorted(unsupported)),
    }
    payload["binding_sha256"] = _digest(payload)
    return payload


def _collect_fast_callable_specs() -> tuple[tuple[Any, tuple[str, ...]], ...]:
    queue = [
        function
        for name in _PROOF_CALLABLE_ROOT_NAMES
        if (function := _as_loaded_function(globals().get(name))) is not None
    ]
    queue.extend(function for _origin, function in _runtime_method_functions())
    specs: list[tuple[Any, tuple[str, ...]]] = []
    seen: set[int] = set()
    while queue:
        function = queue.pop(0)
        if id(function) in seen:
            continue
        seen.add(id(function))
        loaded_names = tuple(
            sorted(
                {
                    str(instruction.argval)
                    for code in _nested_code_objects(function.__code__)
                    for instruction in dis.get_instructions(code)
                    if instruction.opname in {"LOAD_GLOBAL", "LOAD_NAME"}
                    and str(instruction.argval) in function.__globals__
                    and str(instruction.argval)
                    not in _PROOF_DYNAMIC_GLOBAL_EXCLUSIONS
                }
            )
        )
        specs.append((function, loaded_names))
        for name in loaded_names:
            dependency = _as_loaded_function(function.__globals__[name])
            if dependency is not None and _callable_dependency_allowed(dependency):
                queue.append(dependency)
    return tuple(specs)


def _runtime_method_functions() -> tuple[tuple[str, Any], ...]:
    rows = []
    for cls in _runtime_proof_classes():
        for name, kind, function in _class_member_function_rows(cls):
            rows.append(
                (
                    f"method:{cls.__module__}.{cls.__qualname__}.{name}:{kind}",
                    function,
                )
            )
    return tuple(rows)


def _proof_runtime_functions() -> tuple[tuple[str, Any], ...]:
    """Return every callable covered by the proof or simulator method theorem."""
    rows = [
        (f"callable:{function.__module__}.{function.__qualname__}", function)
        for function, _loaded_names in _PROOF_FAST_CALLABLE_SPECS
    ]
    rows.extend(_runtime_method_functions())
    return tuple(rows)


def _function_builtin_namespace(function: Any) -> dict[str, Any]:
    namespace = getattr(function, "__builtins__", None)
    if inspect.ismodule(namespace):
        return vars(namespace)
    if isinstance(namespace, dict):
        return namespace
    fallback = function.__globals__.get("__builtins__", builtins.__dict__)
    return vars(fallback) if inspect.ismodule(fallback) else fallback


def _collect_builtin_specs() -> tuple[tuple[Any, ...], ...]:
    """Freeze every builtin resolved by reachable loaded bytecode."""
    by_key: dict[tuple[Any, ...], tuple[Any, ...]] = {}
    for origin, function in _proof_runtime_functions():
        namespace = _function_builtin_namespace(function)
        for code in _nested_code_objects(function.__code__):
            for instruction in dis.get_instructions(code):
                if instruction.opname not in {"LOAD_GLOBAL", "LOAD_NAME"}:
                    continue
                name = str(instruction.argval)
                if name in function.__globals__ or name not in namespace:
                    continue
                key = (origin, code.co_qualname, name)
                by_key[key] = (
                    origin,
                    function,
                    code.co_qualname,
                    name,
                    namespace[name],
                )
    return tuple(by_key[key] for key in sorted(by_key, key=repr))


def _runtime_builtin_binding_inventory() -> dict[str, Any]:
    """Bind builtin identity and semantics for the full reachable call graph."""
    rows: list[tuple[Any, ...]] = []
    unsupported: list[tuple[Any, ...]] = []
    for origin, function, code_qualname, name, frozen_value in _PROOF_BUILTIN_SPECS:
        namespace = _function_builtin_namespace(function)
        if name not in namespace:
            unsupported.append((origin, code_qualname, name, "missing"))
            continue
        value = namespace[name]
        try:
            semantic_token = _global_value_binding(value)
        except (TypeError, ValueError) as exc:
            unsupported.append(
                (origin, code_qualname, name, type(exc).__name__, str(exc))
            )
            continue
        rows.append(
            (
                origin,
                code_qualname,
                name,
                value is frozen_value,
                semantic_token,
            )
        )
    payload = {
        "rows": tuple(rows),
        "unsupported": tuple(sorted(unsupported, key=repr)),
        "all_identities_match_import_freeze": all(row[3] for row in rows),
    }
    payload["binding_sha256"] = _digest(payload)
    return payload


def _collect_module_attribute_specs() -> tuple[tuple[Any, ...], ...]:
    """Collect every statically loaded attribute prefix rooted at a module."""
    by_key: dict[tuple[Any, ...], tuple[Any, ...]] = {}
    for origin, function in _proof_runtime_functions():
        for code in _nested_code_objects(function.__code__):
            instructions = list(dis.get_instructions(code))
            for index, instruction in enumerate(instructions):
                if instruction.opname not in {"LOAD_GLOBAL", "LOAD_NAME"}:
                    continue
                global_name = str(instruction.argval)
                root = function.__globals__.get(global_name)
                if not inspect.ismodule(root):
                    continue
                attributes: list[str] = []
                for successor in instructions[index + 1 :]:
                    if successor.opname in {"CACHE", "EXTENDED_ARG"}:
                        continue
                    if successor.opname not in {"LOAD_ATTR", "LOAD_METHOD"}:
                        break
                    attributes.append(str(successor.argval))
                    prefix = tuple(attributes)
                    key = (
                        origin,
                        code.co_qualname,
                        global_name,
                        prefix,
                    )
                    by_key[key] = (origin, function, code.co_qualname, global_name, prefix)
    return tuple(by_key[key] for key in sorted(by_key, key=repr))


def _dynamic_module_access_inventory() -> tuple[tuple[Any, ...], ...]:
    """Fail closed when a module root is used without a static attr chain."""
    rows: list[tuple[Any, ...]] = []
    for origin, function in _proof_runtime_functions():
        for code in _nested_code_objects(function.__code__):
            instructions = list(dis.get_instructions(code))
            for index, instruction in enumerate(instructions):
                if instruction.opname not in {"LOAD_GLOBAL", "LOAD_NAME"}:
                    continue
                global_name = str(instruction.argval)
                root = function.__globals__.get(global_name)
                if not inspect.ismodule(root):
                    continue
                successor_name = "END"
                for successor in instructions[index + 1 :]:
                    if successor.opname in {"CACHE", "EXTENDED_ARG"}:
                        continue
                    successor_name = successor.opname
                    break
                if successor_name not in {"LOAD_ATTR", "LOAD_METHOD"}:
                    rows.append(
                        (
                            origin,
                            code.co_qualname,
                            global_name,
                            int(instruction.offset),
                            successor_name,
                        )
                    )
    return tuple(sorted(rows))


def _resolve_module_attribute_spec(spec: tuple[Any, ...]) -> Any:
    _origin, function, _code_qualname, global_name, attributes = spec
    value = function.__globals__[global_name]
    for attribute in attributes:
        value = getattr(value, attribute)
    return value


def _python_functions_on_class(cls: type[Any]) -> tuple[Any, ...]:
    return tuple(function for _name, _kind, function in _class_member_function_rows(cls))


def _collect_fast_unique_functions() -> tuple[Any, ...]:
    """Deduplicate callable-state work while covering methods and attr values."""
    functions = [function for function, _names in _PROOF_FAST_CALLABLE_SPECS]
    functions.extend(function for _origin, function in _runtime_method_functions())
    for spec in _PROOF_MODULE_ATTRIBUTE_SPECS:
        value = _resolve_module_attribute_spec(spec)
        function = _as_loaded_function(value)
        if function is not None:
            functions.append(function)
        elif inspect.isclass(value):
            functions.extend(_python_functions_on_class(value))
    return tuple(dict.fromkeys(functions))


def _runtime_module_attribute_inventory() -> dict[str, Any]:
    """Bind all static module attribute chains used by reachable loaded code."""
    rows: list[tuple[Any, ...]] = []
    unsupported: list[tuple[Any, ...]] = []
    for spec in _PROOF_MODULE_ATTRIBUTE_SPECS:
        origin, _function, code_qualname, global_name, attributes = spec
        try:
            value = _resolve_module_attribute_spec(spec)
            token = (
                (
                    "mapping_owner",
                    f"{type(value).__module__}.{type(value).__qualname__}",
                )
                if isinstance(value, Mapping)
                else _global_value_binding(value)
            )
        except (AttributeError, TypeError, ValueError) as exc:
            unsupported.append(
                (
                    origin,
                    code_qualname,
                    global_name,
                    attributes,
                    type(exc).__name__,
                    str(exc),
                )
            )
            continue
        rows.append((origin, code_qualname, global_name, attributes, token))
    dynamic_access = _dynamic_module_access_inventory()
    payload = {
        "rows": tuple(rows),
        "unsupported": tuple(sorted(unsupported, key=repr)),
        "dynamic_module_access": dynamic_access,
    }
    payload["binding_sha256"] = _digest(payload)
    return payload


def _compiled_code_index(module_code: types.CodeType) -> dict[tuple[Any, ...], Any]:
    rows: dict[tuple[Any, ...], list[types.CodeType]] = {}
    for code in _nested_code_objects(module_code):
        key = (code.co_qualname, int(code.co_firstlineno), code.co_name)
        rows.setdefault(key, []).append(code)
    return {key: tuple(values) for key, values in rows.items()}


def _source_to_loaded_code_attestation() -> dict[str, Any]:
    """Compile disk bytes independently and compare every loaded code manifest."""
    functions = _proof_runtime_functions()
    source_paths: dict[Path, bytes] = {}
    compiled_indexes: dict[Path, dict[tuple[Any, ...], Any]] = {}
    failures: list[tuple[Any, ...]] = []
    rows: list[tuple[Any, ...]] = []
    for origin, function in functions:
        path = Path(function.__code__.co_filename).resolve()
        if not path.is_file() or path.suffix != ".py":
            failures.append((origin, str(path), "source_path_not_a_python_file"))
            continue
        try:
            source_bytes = source_paths.setdefault(path, path.read_bytes())
            if path not in compiled_indexes:
                compiled = compile(
                    source_bytes,
                    str(path),
                    "exec",
                    flags=0,
                    dont_inherit=True,
                    optimize=int(sys.flags.optimize),
                )
                compiled_indexes[path] = _compiled_code_index(compiled)
        except (OSError, SyntaxError, ValueError) as exc:
            failures.append((origin, str(path), f"compile_failed:{type(exc).__name__}"))
            continue
        key = (
            function.__code__.co_qualname,
            int(function.__code__.co_firstlineno),
            function.__code__.co_name,
        )
        candidates = compiled_indexes[path].get(key, ())
        loaded_token = _code_binding_token(function.__code__)
        matches = [
            candidate
            for candidate in candidates
            if _code_binding_token(candidate) == loaded_token
        ]
        row = (
            origin,
            str(path),
            key,
            sha256(source_bytes).hexdigest(),
            _digest(loaded_token),
            tuple(_digest(_code_binding_token(candidate)) for candidate in candidates),
            len(matches) == 1,
        )
        rows.append(row)
        if len(matches) != 1:
            failures.append(
                (
                    origin,
                    str(path),
                    key,
                    "loaded_code_does_not_match_unique_disk_compilation",
                    len(candidates),
                    len(matches),
                )
            )
    dependency_hashes = tuple(
        (str(path), sha256(source).hexdigest())
        for path, source in sorted(source_paths.items(), key=lambda item: str(item[0]))
    )
    payload = {
        "schema_version": "paper2_source_loaded_code_attestation_v1",
        "compile_optimize": int(sys.flags.optimize),
        "dependency_sha256": dependency_hashes,
        "rows": tuple(rows),
        "failures": tuple(failures),
        "passed": not failures and len(rows) == len(functions),
    }
    payload["attestation_sha256"] = _digest(payload)
    return payload


def _class_identity_token(cls: type[Any]) -> tuple[str, str]:
    return (str(cls.__module__), str(cls.__qualname__))


def _runtime_class_topology_inventory() -> dict[str, Any]:
    """Bind exact bases/MRO and reject every member outside the frozen allowlist."""
    rows = []
    for cls in _PROOF_CLASS_ROOTS:
        bases = tuple(
            (
                _class_identity_token(base),
                base in _PROOF_CLASS_MRO_ALLOWLIST,
            )
            for base in cls.__bases__
        )
        mro = tuple(
            (
                _class_identity_token(member),
                member in _PROOF_CLASS_MRO_ALLOWLIST,
            )
            for member in cls.__mro__
        )
        rows.append((_class_identity_token(cls), bases, mro))
    payload = {
        "rows": tuple(rows),
        "all_bases_and_mro_members_allowlisted": all(
            allowed
            for _cls, bases, mro in rows
            for _identity, allowed in (*bases, *mro)
        ),
    }
    payload["binding_sha256"] = _digest(payload)
    return payload


def _source_class_topology_attestation() -> dict[str, Any]:
    """Resolve bases from disk AST and compare them to each loaded root class."""
    rows = []
    failures: list[tuple[Any, ...]] = []
    source_cache: dict[Path, tuple[bytes, ast.Module]] = {}
    for cls in _PROOF_CLASS_ROOTS:
        module = next(
            (
                candidate
                for candidate_cls, candidate in _PROOF_CLASS_SOURCE_MODULES
                if candidate_cls is cls
            ),
            None,
        )
        path_value = getattr(module, "__file__", None) if module is not None else None
        if not path_value:
            failures.append((_class_identity_token(cls), "source_file_missing"))
            continue
        path = Path(path_value).resolve()
        try:
            if path not in source_cache:
                source_bytes = path.read_bytes()
                source_cache[path] = (source_bytes, ast.parse(source_bytes))
            source_bytes, tree = source_cache[path]
        except (OSError, SyntaxError, ValueError) as exc:
            failures.append(
                (_class_identity_token(cls), "source_parse_failed", type(exc).__name__)
            )
            continue
        definitions = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and node.name == cls.__name__
        ]
        if len(definitions) != 1:
            failures.append(
                (
                    _class_identity_token(cls),
                    "class_definition_not_unique",
                    len(definitions),
                )
            )
            continue
        definition = definitions[0]
        try:
            declared_bases = tuple(
                eval(  # noqa: S307 - frozen trusted dependency source only
                    compile(
                        ast.Expression(base),
                        str(path),
                        "eval",
                        flags=0,
                        dont_inherit=True,
                        optimize=int(sys.flags.optimize),
                    ),
                    vars(module),
                )
                for base in definition.bases
            )
            expected_bases = (
                types.resolve_bases(declared_bases) if declared_bases else (object,)
            )
        except Exception as exc:  # noqa: BLE001 - attestation records fail closed
            failures.append(
                (
                    _class_identity_token(cls),
                    "source_base_resolution_failed",
                    type(exc).__name__,
                )
            )
            continue
        loaded_bases = tuple(cls.__bases__)
        exact = len(expected_bases) == len(loaded_bases) and all(
            expected is loaded
            for expected, loaded in zip(expected_bases, loaded_bases, strict=True)
        )
        row = (
            _class_identity_token(cls),
            str(path),
            sha256(source_bytes).hexdigest(),
            tuple(ast.unparse(base) for base in definition.bases),
            tuple(_class_identity_token(base) for base in expected_bases),
            tuple(_class_identity_token(base) for base in loaded_bases),
            exact,
        )
        rows.append(row)
        if not exact:
            failures.append(
                (_class_identity_token(cls), "loaded_bases_differ_from_disk_source")
            )
    payload = {
        "schema_version": "paper2_source_class_topology_attestation_v1",
        "rows": tuple(rows),
        "failures": tuple(failures),
        "passed": not failures and len(rows) == len(_PROOF_CLASS_ROOTS),
    }
    payload["attestation_sha256"] = _digest(payload)
    return payload


def _closure_binding_token(
    function: Any,
    *,
    _active: frozenset[int] = frozenset(),
) -> tuple[Any, ...]:
    rows = []
    for cell in getattr(function, "__closure__", None) or ():
        try:
            rows.append(_global_value_binding(cell.cell_contents, _active=_active))
        except ValueError:
            rows.append(("empty_closure_cell",))
    return tuple(rows)


def _fast_callable_binding_token(
    function: Any,
    active: frozenset[int] = frozenset(),
) -> tuple[Any, ...]:
    """Cheap content binding for mutable callable state checked on every key."""
    if id(function) in active:
        return ("recursive_callable", function.__module__, function.__qualname__)
    active = active | frozenset({id(function)})
    return (
        id(function),
        function.__module__,
        function.__qualname__,
        _loaded_callable_source_sha256(function),
        str(function.__code__.co_filename),
        int(function.__code__.co_firstlineno),
        id(function.__code__),
        _code_binding_sha256(function.__code__),
        _fast_value_binding(getattr(function, "__defaults__", None), active),
        _fast_value_binding(getattr(function, "__kwdefaults__", None), active),
        tuple(
            _fast_value_binding(cell.cell_contents, active)
            if _closure_cell_is_bound(cell)
            else ("empty_closure_cell",)
            for cell in (getattr(function, "__closure__", None) or ())
        ),
    )


def _closure_cell_is_bound(cell: Any) -> bool:
    try:
        cell.cell_contents
    except ValueError:
        return False
    return True


def _fast_value_binding(value: Any, active: frozenset[int] = frozenset()) -> Any:
    """Cycle-safe content token for mutable defaults, closures, and globals."""
    if id(value) in active:
        return (
            "recursive_ref",
            str(getattr(value, "__module__", type(value).__module__)),
            str(getattr(value, "__qualname__", type(value).__qualname__)),
        )
    function = _as_loaded_function(value)
    if function is not None:
        return ("callable", _fast_callable_binding_token(function, active))
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return ("float", value.hex())
    if isinstance(value, Path):
        return ("path", str(value))
    if isinstance(value, (bytes, np.ndarray, np.floating, np.integer)):
        return _semantic(value)
    if isinstance(value, (tuple, list, set, frozenset, dict)):
        next_active = active | frozenset({id(value)})
        if isinstance(value, tuple):
            return (
                "tuple",
                tuple(_fast_value_binding(item, next_active) for item in value),
            )
        if isinstance(value, list):
            return (
                "list",
                tuple(_fast_value_binding(item, next_active) for item in value),
            )
        if isinstance(value, (set, frozenset)):
            return (
                type(value).__name__,
                tuple(
                    sorted(
                        (_fast_value_binding(item, next_active) for item in value),
                        key=repr,
                    )
                ),
            )
        return (
            "dict",
            tuple(
                sorted(
                    (
                        (
                            _fast_value_binding(key, next_active),
                            _fast_value_binding(item, next_active),
                        )
                        for key, item in value.items()
                    ),
                    key=repr,
                )
            ),
        )
    if inspect.ismodule(value):
        return (
            "module",
            id(value),
            value.__name__,
            str(getattr(value, "__version__", "UNVERSIONED")),
            str(getattr(value, "__file__", "BUILTIN")),
        )
    if inspect.isclass(value):
        next_active = active | frozenset({id(value)})
        methods = []
        for name, descriptor in sorted(vars(value).items()):
            member = (
                descriptor.__func__
                if isinstance(descriptor, (staticmethod, classmethod))
                else descriptor
            )
            if inspect.isfunction(member):
                methods.append(
                    (name, _fast_callable_binding_token(member, next_active))
                )
        return (
            "class",
            id(value),
            value.__module__,
            value.__qualname__,
            tuple(methods),
        )
    identity = None if getattr(value, "__self__", None) is not None else id(value)
    return (
        "opaque",
        identity,
        str(getattr(value, "__module__", type(value).__module__)),
        str(getattr(value, "__qualname__", type(value).__qualname__)),
        f"{type(value).__module__}.{type(value).__qualname__}",
    )


def _fast_attribute_binding_token(value: Any) -> tuple[Any, ...]:
    function = _as_loaded_function(value)
    if function is not None:
        return ("callable_ref", id(value), id(function), id(function.__code__))
    if inspect.isclass(value):
        method_refs = tuple(
            (member.__name__, id(member), id(member.__code__))
            for member in _python_functions_on_class(value)
        )
        return (
            "class_ref",
            id(value),
            value.__module__,
            value.__qualname__,
            method_refs,
        )
    if isinstance(value, (tuple, list, set, frozenset, dict)):
        return ("container", id(value), _fast_value_binding(value))
    if value is None or isinstance(value, (str, bool, int, float, bytes)):
        return ("literal", _fast_value_binding(value))
    identity = None if getattr(value, "__self__", None) is not None else id(value)
    return (
        "object_ref",
        identity,
        str(getattr(value, "__module__", type(value).__module__)),
        str(getattr(value, "__qualname__", type(value).__qualname__)),
        f"{type(value).__module__}.{type(value).__qualname__}",
    )


def _fast_global_reference_token(value: Any) -> tuple[Any, ...]:
    """Bind global identity plus mutable container contents without graph replay."""
    function = _as_loaded_function(value)
    if function is not None:
        return ("callable_ref", id(value), id(function), id(function.__code__))
    if isinstance(value, (tuple, list, set, frozenset, dict)):
        return ("container", id(value), _fast_value_binding(value))
    if value is None or isinstance(value, (str, bool, int, float, bytes)):
        return ("literal", _fast_value_binding(value))
    return (
        "object_ref",
        id(value),
        f"{type(value).__module__}.{type(value).__qualname__}",
    )


def _fast_class_member_descriptor_token(descriptor: Any) -> tuple[Any, ...]:
    kind = _class_member_descriptor_kind(descriptor)
    if inspect.isfunction(descriptor):
        return (kind, id(descriptor), id(descriptor.__code__))
    if isinstance(descriptor, (staticmethod, classmethod)):
        return (
            kind,
            id(descriptor),
            id(descriptor.__func__),
            id(descriptor.__func__.__code__),
        )
    if isinstance(descriptor, property):
        return (
            kind,
            id(descriptor),
            tuple(
                (
                    accessor,
                    id(function) if inspect.isfunction(function) else None,
                    id(function.__code__) if inspect.isfunction(function) else None,
                )
                for accessor, function in (
                    ("fget", descriptor.fget),
                    ("fset", descriptor.fset),
                    ("fdel", descriptor.fdel),
                )
            ),
        )
    try:
        fields = vars(descriptor)
    except TypeError:
        fields = None
    if fields is not None:
        return (
            kind,
            id(descriptor),
            tuple(
                sorted(
                    (name, _fast_global_reference_token(item))
                    for name, item in fields.items()
                )
            ),
        )
    return (kind, _fast_global_reference_token(descriptor))


def _runtime_fast_builtin_signature() -> tuple[Any, ...]:
    references: list[tuple[Any, ...]] = []
    unique_values: dict[tuple[int, str], tuple[Any, ...]] = {}
    for origin, function, code_qualname, name, frozen_value in _PROOF_BUILTIN_SPECS:
        namespace = _function_builtin_namespace(function)
        value = namespace.get(name)
        key = (id(namespace), name)
        references.append(
            (
                origin,
                code_qualname,
                name,
                key,
                value is frozen_value,
                id(value),
            )
        )
        if key not in unique_values:
            unique_values[key] = (
                key,
                _fast_attribute_binding_token(value),
            )
    return (tuple(references), tuple(unique_values.values()))


def _runtime_fast_class_topology_signature() -> tuple[Any, ...]:
    return tuple(
        (
            id(cls),
            _class_identity_token(cls),
            tuple(
                (id(base), _class_identity_token(base), base in _PROOF_CLASS_MRO_ALLOWLIST)
                for base in cls.__bases__
            ),
            tuple(
                (
                    id(member),
                    _class_identity_token(member),
                    member in _PROOF_CLASS_MRO_ALLOWLIST,
                )
                for member in cls.__mro__
            ),
        )
        for cls in _PROOF_CLASS_ROOTS
    )


def _runtime_fast_binding_signature() -> tuple[Any, ...]:
    root_rows = tuple(
        (
            name,
            id(globals().get(name)),
            id(function) if (function := _as_loaded_function(globals().get(name))) else None,
            id(function.__code__) if function is not None else None,
        )
        for name in _PROOF_CALLABLE_ROOT_NAMES
    )
    global_rows = tuple(
        (
            id(function),
            id(function.__code__),
            tuple(
                (
                    name,
                    _fast_global_reference_token(function.__globals__.get(name)),
                )
                for name in loaded_names
            ),
        )
        for function, loaded_names in _PROOF_FAST_CALLABLE_SPECS
    )
    method_rows = tuple(
        (
            cls.__module__,
            cls.__qualname__,
            member_name,
            _fast_class_member_descriptor_token(descriptor),
        )
        for cls in _runtime_proof_classes()
        for member_name, descriptor in sorted(vars(cls).items())
    )
    callable_state_rows = tuple(
        _fast_callable_binding_token(function)
        for function in _PROOF_FAST_UNIQUE_FUNCTIONS
    )
    class_global_rows = tuple(
        (
            cls.__name__,
            method_name,
            global_name,
            _fast_global_reference_token(function.__globals__.get(global_name)),
        )
        for cls, method_name, global_name in _runtime_global_binding_specs()
        for descriptor in (vars(cls).get(method_name),)
        for function in (
            descriptor.__func__
            if isinstance(descriptor, (staticmethod, classmethod))
            else descriptor,
        )
        if inspect.isfunction(function)
    )
    builtin_rows = _runtime_fast_builtin_signature()
    class_topology_rows = _runtime_fast_class_topology_signature()
    module_attribute_rows = tuple(
        (
            origin,
            code_qualname,
            global_name,
            attributes,
            _fast_attribute_binding_token(_resolve_module_attribute_spec(spec)),
        )
        for spec in _PROOF_MODULE_ATTRIBUTE_SPECS
        for origin, _function, code_qualname, global_name, attributes in (spec,)
    )
    return (
        root_rows,
        global_rows,
        method_rows,
        callable_state_rows,
        class_global_rows,
        builtin_rows,
        class_topology_rows,
        module_attribute_rows,
    )


def _assert_loaded_proof_bindings(*, full: bool = False) -> None:
    """Fail before a build if any frozen loaded callable/global was rebound."""
    if (
        _PROOF_CALLABLE_BASELINE is None
        or _PROOF_METHOD_BASELINE is None
        or _PROOF_CLASS_GLOBAL_BASELINE is None
        or _PROOF_BUILTIN_BASELINE is None
        or _PROOF_CLASS_TOPOLOGY_BASELINE is None
        or _PROOF_SOURCE_CLASS_ATTESTATION_BASELINE is None
        or _PROOF_MODULE_ATTRIBUTE_BASELINE is None
        or _PROOF_SOURCE_CODE_ATTESTATION_BASELINE is None
        or _PROOF_FAST_BASELINE is None
    ):
        raise RuntimeError("loaded proof binding baseline was not frozen")
    current_fast = _runtime_fast_binding_signature()
    if current_fast != _PROOF_FAST_BASELINE:
        labels = (
            "module-level proof callable bindings drifted",
            "proof callable global bindings drifted",
            "runtime method bindings drifted",
            "callable source/code/default/closure bindings drifted",
            "runtime global bindings drifted",
            "runtime builtin bindings drifted",
            "runtime class bases/MRO bindings drifted",
            "static module attribute bindings drifted",
        )
        # Do not use a builtin here: this is the fail-closed path for detecting
        # tampering with builtins themselves (including ``zip``).
        drift = []
        for index in (0, 1, 2, 3, 4, 5, 6, 7):
            if current_fast[index] != _PROOF_FAST_BASELINE[index]:
                drift.append(labels[index])
        raise RuntimeError("; ".join(drift))
    if not full:
        return
    current_callable = _runtime_proof_callable_inventory()
    current_methods = _runtime_method_binding_inventory()
    current_globals = _runtime_global_binding_inventory()
    current_builtins = _runtime_builtin_binding_inventory()
    current_class_topology = _runtime_class_topology_inventory()
    current_source_class_attestation = _source_class_topology_attestation()
    current_module_attributes = _runtime_module_attribute_inventory()
    current_source_attestation = _source_to_loaded_code_attestation()
    failures = []
    if current_callable != _PROOF_CALLABLE_BASELINE:
        failures.append("module-level proof callable/global bindings drifted")
    if current_methods != _PROOF_METHOD_BASELINE:
        failures.append("runtime method bindings drifted")
    if current_globals != _PROOF_CLASS_GLOBAL_BASELINE:
        failures.append("class-method global bindings drifted")
    if current_builtins != _PROOF_BUILTIN_BASELINE:
        failures.append("runtime builtin bindings drifted")
    if current_class_topology != _PROOF_CLASS_TOPOLOGY_BASELINE:
        failures.append("runtime class bases/MRO bindings drifted")
    if current_source_class_attestation != _PROOF_SOURCE_CLASS_ATTESTATION_BASELINE:
        failures.append("source-to-loaded-class topology attestation drifted")
    if not current_source_class_attestation["passed"]:
        failures.append("source-to-loaded-class topology attestation failed")
    if current_module_attributes != _PROOF_MODULE_ATTRIBUTE_BASELINE:
        failures.append("static module attribute bindings drifted")
    if current_source_attestation != _PROOF_SOURCE_CODE_ATTESTATION_BASELINE:
        failures.append("source-to-loaded-code attestation drifted")
    if not current_source_attestation["passed"]:
        failures.append("source-to-loaded-code attestation failed")
    if failures:
        raise RuntimeError("; ".join(failures))


def _freeze_loaded_proof_bindings() -> None:
    """Freeze original loaded bindings at module completion, before callers run."""
    global _PROOF_CALLABLE_ROOT_NAMES
    global _PROOF_CALLABLE_BASELINE
    global _PROOF_METHOD_BASELINE
    global _PROOF_CLASS_GLOBAL_BASELINE
    global _PROOF_BUILTIN_SPECS
    global _PROOF_BUILTIN_BASELINE
    global _PROOF_CLASS_ROOTS
    global _PROOF_CLASS_SOURCE_MODULES
    global _PROOF_CLASS_MRO_ALLOWLIST
    global _PROOF_CLASS_TOPOLOGY_BASELINE
    global _PROOF_SOURCE_CLASS_ATTESTATION_BASELINE
    global _PROOF_FAST_CALLABLE_SPECS
    global _PROOF_FAST_UNIQUE_FUNCTIONS
    global _PROOF_MODULE_ATTRIBUTE_SPECS
    global _PROOF_MODULE_ATTRIBUTE_BASELINE
    global _PROOF_SOURCE_CODE_ATTESTATION_BASELINE
    global _PROOF_FAST_BASELINE
    if _PROOF_CALLABLE_BASELINE is not None:
        return
    missing_entrypoints = sorted(
        name
        for name in _SCIENTIFIC_PROOF_ENTRYPOINT_NAMES
        if _as_loaded_function(globals().get(name)) is None
    )
    if missing_entrypoints:
        raise RuntimeError(f"scientific proof entrypoints missing: {missing_entrypoints}")
    _PROOF_CALLABLE_ROOT_NAMES = tuple(
        sorted(
            set(_SCIENTIFIC_PROOF_ENTRYPOINT_NAMES)
            | {
                name
                for name in _EXTERNAL_PROOF_CALLABLE_ROOTS
                if _as_loaded_function(globals().get(name)) is not None
            }
        )
    )
    _PROOF_CLASS_ROOTS = _collect_runtime_proof_root_classes()
    _PROOF_CLASS_SOURCE_MODULES = tuple(
        (cls, __import__(cls.__module__, fromlist=["*"]))
        for cls in _PROOF_CLASS_ROOTS
    )
    _PROOF_CLASS_MRO_ALLOWLIST = frozenset(
        member for cls in _PROOF_CLASS_ROOTS for member in cls.__mro__
    )
    source_class_attestation = _source_class_topology_attestation()
    if not source_class_attestation["passed"]:
        raise RuntimeError(
            "source-to-loaded-class topology attestation failed before import freeze: "
            f"{source_class_attestation['failures'][:3]}"
        )
    _PROOF_SOURCE_CLASS_ATTESTATION_BASELINE = source_class_attestation
    class_topology_baseline = _runtime_class_topology_inventory()
    if not class_topology_baseline["all_bases_and_mro_members_allowlisted"]:
        raise RuntimeError("runtime class topology contains a nonallowlisted member")
    _PROOF_CLASS_TOPOLOGY_BASELINE = class_topology_baseline
    _PROOF_FAST_CALLABLE_SPECS = _collect_fast_callable_specs()
    _PROOF_MODULE_ATTRIBUTE_SPECS = _collect_module_attribute_specs()
    _PROOF_BUILTIN_SPECS = _collect_builtin_specs()
    _PROOF_FAST_UNIQUE_FUNCTIONS = _collect_fast_unique_functions()
    source_attestation = _source_to_loaded_code_attestation()
    if not source_attestation["passed"]:
        raise RuntimeError(
            "source-to-loaded-code attestation failed before import freeze: "
            f"{source_attestation['failures'][:3]}"
        )
    _PROOF_SOURCE_CODE_ATTESTATION_BASELINE = source_attestation
    _PROOF_CALLABLE_BASELINE = _runtime_proof_callable_inventory()
    _PROOF_METHOD_BASELINE = _runtime_method_binding_inventory()
    _PROOF_CLASS_GLOBAL_BASELINE = _runtime_global_binding_inventory()
    builtin_baseline = _runtime_builtin_binding_inventory()
    if builtin_baseline["unsupported"] or not builtin_baseline[
        "all_identities_match_import_freeze"
    ]:
        raise RuntimeError(
            "builtin proof failed before import freeze: "
            f"unsupported={builtin_baseline['unsupported'][:3]}"
        )
    _PROOF_BUILTIN_BASELINE = builtin_baseline
    module_attribute_baseline = _runtime_module_attribute_inventory()
    if (
        module_attribute_baseline["unsupported"]
        or module_attribute_baseline["dynamic_module_access"]
    ):
        raise RuntimeError(
            "static module attribute proof failed before import freeze: "
            f"unsupported={module_attribute_baseline['unsupported'][:3]}, "
            f"dynamic={module_attribute_baseline['dynamic_module_access'][:3]}"
        )
    _PROOF_MODULE_ATTRIBUTE_BASELINE = module_attribute_baseline
    _PROOF_FAST_BASELINE = _runtime_fast_binding_signature()


def audit_frozen_state_inventory(sim: Any) -> dict[str, Any]:
    """Classify every live simulator attribute and assert frozen-mode claims."""
    live = set(vars(sim))
    keyed = set(SIM_STATE_FIELDS) | set(SPECIAL_KEY_FIELDS)
    categories = {
        "markov_key": sorted(live & keyed),
        "immutable_contract": sorted(live & set(IMMUTABLE_CONTRACT_FIELDS)),
        "inert_frozen_contract": sorted((live & set(INERT_FROZEN_FIELDS)) - keyed),
        "output_label_or_unaccelerated_replay": sorted(
            (live & set(OUTPUT_OR_REPLAY_FIELDS)) - keyed
        ),
    }
    classified = set().union(*(set(values) for values in categories.values()))
    unclassified = sorted(live - classified)
    overlaps: dict[str, list[str]] = {}
    names = list(categories)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap = sorted(set(categories[left]) & set(categories[right]))
            if overlap:
                overlaps[f"{left}::{right}"] = overlap

    frozen_invariants = {
        "risk_attribution_source_des_events": sim.risk_attribution_source
        == "des_events",
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


def _absolute_lexical_path(path: Path) -> Path:
    """Make a path absolute without resolving its final symlink component."""
    return Path(os.path.abspath(os.fspath(path)))


def _scientific_child_environment_failures(
    environment: Mapping[str, Any],
) -> list[str]:
    failures: list[str] = []
    if set(environment) != set(SCIENTIFIC_CHILD_ENVIRONMENT_KEYS):
        failures.append("scientific child environment key allowlist mismatch")
    if any(
        not isinstance(key, str)
        or not isinstance(value, str)
        or "\x00" in key
        or "\x00" in value
        or "=" in key
        for key, value in environment.items()
    ):
        failures.append("scientific child environment contains malformed entries")
    expected_fixed = {
        "PATH": os.defpath,
        "LANG": "C",
        "LC_ALL": "C",
        "TZ": "UTC",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "SCRES_SCIENTIFIC_CHILD": "1",
    }
    for key, value in expected_fixed.items():
        if environment.get(key) != value:
            failures.append(f"scientific child environment fixed value mismatch: {key}")
    for key in ("HOME", "TMPDIR"):
        value = environment.get(key)
        if not isinstance(value, str) or not Path(value).is_absolute():
            failures.append(f"scientific child environment path is not absolute: {key}")
    if any(
        key.startswith(("PYTHON", "LD_", "DYLD_"))
        or key in {"BASH_ENV", "ENV", "SHELLOPTS", "CDPATH", "GCONV_PATH"}
        for key in environment
    ):
        failures.append("scientific child environment contains injection variables")
    return failures


def _live_scientific_environment_failures(
    actual: Mapping[str, str], expected: Mapping[str, str]
) -> list[str]:
    normalized_actual = dict(actual)
    cf_encoding = normalized_actual.pop("__CF_USER_TEXT_ENCODING", None)
    if cf_encoding is not None and not re.fullmatch(
        r"0x[0-9A-Fa-f]+:0x[0-9A-Fa-f]+:0x[0-9A-Fa-f]+", cf_encoding
    ):
        return ["macOS-injected __CF_USER_TEXT_ENCODING is malformed"]
    pygame_prompt = normalized_actual.pop("PYGAME_HIDE_SUPPORT_PROMPT", None)
    if pygame_prompt not in {None, "hide"}:
        return ["module-controlled PYGAME_HIDE_SUPPORT_PROMPT is malformed"]
    if normalized_actual != dict(expected):
        actual_keys = set(normalized_actual)
        expected_keys = set(expected)
        changed = sorted(
            key
            for key in actual_keys & expected_keys
            if normalized_actual[key] != expected[key]
        )
        return [
            "live isolated child environment differs from exact allowlist: "
            f"extra={sorted(actual_keys - expected_keys)}; "
            f"missing={sorted(expected_keys - actual_keys)}; changed={changed}"
        ]
    return []


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _openssl_identity() -> dict[str, Any]:
    lexical_value = shutil.which("openssl")
    if not lexical_value:
        raise RuntimeError("OpenSSL executable is unavailable for Ed25519 receipts")
    lexical = Path(lexical_value).absolute()
    resolved = lexical.resolve(strict=True)
    version = subprocess.run(
        [str(resolved), "version", "-a"],
        capture_output=True,
        text=True,
        check=True,
        close_fds=True,
    )
    return {
        "lexical_path": str(lexical),
        "resolved_path": str(resolved),
        "executable_sha256": _file_sha256(resolved),
        "version_stdout": version.stdout,
        "version_stdout_sha256": sha256(version.stdout.encode()).hexdigest(),
    }


class EphemeralOpenSSLEd25519Signer:
    """Parent-memory Ed25519 key backed by a pinned OpenSSL executable.

    The PEM private key is never written to a file or environment variable and
    is never passed to the target simulation child.  It is provided only on
    stdin to the separately pinned OpenSSL signing process.
    """

    __slots__ = (
        "__private_pem",
        "__output_descriptor",
        "__output_path",
        "openssl_identity",
        "public_key_der_hex",
        "fingerprint",
    )

    def __init__(self) -> None:
        self.openssl_identity = _openssl_identity()
        executable = self.openssl_identity["resolved_path"]
        generated = subprocess.run(
            [executable, "genpkey", "-algorithm", "ED25519"],
            capture_output=True,
            check=True,
            close_fds=True,
        )
        self.__private_pem = bytearray(generated.stdout)
        self.__output_descriptor: int | None = None
        self.__output_path: str | None = None
        public = subprocess.run(
            [executable, "pkey", "-pubout", "-outform", "DER"],
            input=bytes(self.__private_pem),
            capture_output=True,
            check=True,
            close_fds=True,
        ).stdout
        self.public_key_der_hex = public.hex()
        self.fingerprint = sha256(public).hexdigest()

    def sign(self, message: bytes) -> bytes:
        if not self.__private_pem:
            raise RuntimeError("ephemeral receipt signer is closed")
        executable = self.openssl_identity["resolved_path"]
        with tempfile.NamedTemporaryFile() as message_file:
            message_file.write(message)
            message_file.flush()
            result = subprocess.run(
                [
                    executable,
                    "pkeyutl",
                    "-sign",
                    "-rawin",
                    "-inkey",
                    "/dev/stdin",
                    "-in",
                    message_file.name,
                ],
                input=bytes(self.__private_pem),
                capture_output=True,
                check=True,
                close_fds=True,
            )
        if len(result.stdout) != 64:
            raise RuntimeError("OpenSSL returned a malformed Ed25519 signature")
        return result.stdout

    def bind_exclusive_output_descriptor(self, descriptor: int, path: Path) -> None:
        if self.__output_descriptor is not None:
            raise RuntimeError("ephemeral signer already binds an output descriptor")
        self.__output_descriptor = int(descriptor)
        self.__output_path = str(path.resolve())

    def bound_output_descriptor(self, path: Path) -> int:
        if (
            self.__output_descriptor is None
            or self.__output_path != str(path.resolve())
        ):
            raise RuntimeError("ephemeral signer does not bind this output path")
        return self.__output_descriptor

    def close(self) -> None:
        if self.__output_descriptor is not None:
            try:
                os.close(self.__output_descriptor)
            except OSError:
                pass
            self.__output_descriptor = None
            self.__output_path = None
        for index in range(len(self.__private_pem)):
            self.__private_pem[index] = 0
        self.__private_pem.clear()

    def __reduce__(self) -> Any:
        raise TypeError("ephemeral receipt signers cannot be serialized")


def _verify_openssl_ed25519_signature(
    public_key_der_hex: str,
    message: bytes,
    signature_hex: str,
) -> bool:
    try:
        public_key = bytes.fromhex(public_key_der_hex)
        signature_bytes = bytes.fromhex(signature_hex)
    except ValueError:
        return False
    if len(signature_bytes) != 64:
        return False
    current_openssl = _openssl_identity()
    executable = current_openssl["resolved_path"]
    with tempfile.NamedTemporaryFile() as message_file, tempfile.NamedTemporaryFile() as signature_file:
        message_file.write(message)
        message_file.flush()
        signature_file.write(signature_bytes)
        signature_file.flush()
        result = subprocess.run(
            [
                executable,
                "pkeyutl",
                "-verify",
                "-rawin",
                "-pubin",
                "-keyform",
                "DER",
                "-inkey",
                "/dev/stdin",
                "-in",
                message_file.name,
                "-sigfile",
                signature_file.name,
            ],
            input=public_key,
            capture_output=True,
            check=False,
            close_fds=True,
        )
    return result.returncode == 0


def sign_reduced_execution_receipt(
    receipt: Mapping[str, Any],
    signer: EphemeralOpenSSLEd25519Signer,
) -> dict[str, Any]:
    unsigned = dict(receipt)
    for name in (
        "receipt_signed_body_sha256",
        "receipt_signature_ed25519",
    ):
        if name in unsigned:
            raise ValueError(f"receipt already contains signature field: {name}")
    unsigned.update(
        {
            "receipt_signature_scheme": "OpenSSL-Ed25519",
            "receipt_signing_public_key_der_hex": signer.public_key_der_hex,
            "receipt_signing_public_key_fingerprint": signer.fingerprint,
            "receipt_signing_openssl_identity": signer.openssl_identity,
        }
    )
    message = _canonical_json_bytes(unsigned)
    return {
        **unsigned,
        "receipt_signed_body_sha256": sha256(message).hexdigest(),
        "receipt_signature_ed25519": signer.sign(message).hex(),
    }


def verify_reduced_execution_receipt_signature(
    receipt: Mapping[str, Any], expected_public_key_fingerprint: str
) -> list[str]:
    failures: list[str] = []
    body = dict(receipt)
    signature = body.pop("receipt_signature_ed25519", None)
    claimed_body_sha256 = body.pop("receipt_signed_body_sha256", None)
    message = _canonical_json_bytes(body)
    if claimed_body_sha256 != sha256(message).hexdigest():
        failures.append("execution receipt signed-body digest mismatch")
    public_hex = body.get("receipt_signing_public_key_der_hex")
    fingerprint = (
        sha256(bytes.fromhex(public_hex)).hexdigest()
        if isinstance(public_hex, str)
        and re.fullmatch(r"[0-9a-f]+", public_hex)
        and len(public_hex) % 2 == 0
        else None
    )
    if fingerprint != expected_public_key_fingerprint:
        failures.append("caller-retained signing-key fingerprint mismatch")
    if body.get("receipt_signing_public_key_fingerprint") != fingerprint:
        failures.append("execution receipt public-key fingerprint mismatch")
    if body.get("receipt_signature_scheme") != "OpenSSL-Ed25519":
        failures.append("execution receipt signature scheme mismatch")
    openssl_identity = body.get("receipt_signing_openssl_identity")
    if not (
        isinstance(openssl_identity, dict)
        and isinstance(openssl_identity.get("resolved_path"), str)
        and Path(openssl_identity["resolved_path"]).is_absolute()
        and re.fullmatch(
            r"[0-9a-f]{64}", str(openssl_identity.get("executable_sha256", ""))
        )
        and re.fullmatch(
            r"[0-9a-f]{64}",
            str(openssl_identity.get("version_stdout_sha256", "")),
        )
    ):
        failures.append("execution receipt signing OpenSSL identity is malformed")
    if not isinstance(signature, str) or not isinstance(
        openssl_identity, dict
    ) or not _verify_openssl_ed25519_signature(
        str(public_hex),
        message,
        signature,
    ):
        failures.append("execution receipt Ed25519 signature is invalid")
    return failures


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
        "simpy_source_sha256": {
            module.__name__: _file_sha256(Path(module.__file__).resolve())
            for module in SIMPY_SOURCE_MODULES
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


def _ast_call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _ast_call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return None


@lru_cache(maxsize=1)
def _transition_entropy_inventory() -> dict[str, Any]:
    """Inventory external entropy calls in the frozen supply-chain sources.

    The runner's wall-clock profiling is deliberately outside this scan.  RNG
    constructors in the model are allowed only because their complete states are
    serialized and their seed/tape inputs are bound below.
    """
    forbidden_prefixes = (
        "datetime.",
        "os.urandom",
        "random.",
        "secrets.",
        "time.",
        "uuid.",
    )
    model_paths = tuple(
        path
        for path in CERTIFICATION_DEPENDENCY_PATHS
        if path.suffix == ".py" and path.parent.name == "supply_chain"
    )
    forbidden: list[dict[str, Any]] = []
    seeded_rng: list[dict[str, Any]] = []
    for path in model_paths:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _ast_call_name(node.func)
            if not name:
                continue
            row = {
                "path": str(path.relative_to(ROOT)),
                "line": int(getattr(node, "lineno", -1)),
                "call": name,
            }
            if name in {"np.random.default_rng", "np.random.SeedSequence"}:
                seeded_rng.append(row)
            if name.startswith(forbidden_prefixes):
                forbidden.append(row)
    return {
        "forbidden_external_entropy_calls": forbidden,
        "seeded_numpy_rng_calls": seeded_rng,
    }


@lru_cache(maxsize=1)
def _markov_completeness_certificate_cached() -> dict[str, Any]:
    """Return the source/runtime theorem that licenses the finite quotient.

    This is not a theorem about arbitrary Python programs.  It is a machine-bound
    proof obligation for the exact frozen code and ABI named in the certificate.
    Every live key evaluation separately enforces its runtime schema.
    """
    entropy = _transition_entropy_inventory()
    global_bindings = _runtime_global_binding_inventory()
    method_bindings = _runtime_method_binding_inventory()
    callable_bindings = _runtime_proof_callable_inventory()
    builtin_bindings = _runtime_builtin_binding_inventory()
    class_topology = _runtime_class_topology_inventory()
    source_class_attestation = _source_class_topology_attestation()
    module_attribute_bindings = _runtime_module_attribute_inventory()
    source_code_attestation = _source_to_loaded_code_attestation()
    callable_bindings_match = bool(
        _PROOF_CALLABLE_BASELINE is not None
        and callable_bindings == _PROOF_CALLABLE_BASELINE
    )
    method_bindings_match = bool(
        _PROOF_METHOD_BASELINE is not None
        and method_bindings == _PROOF_METHOD_BASELINE
    )
    global_bindings_match = bool(
        _PROOF_CLASS_GLOBAL_BASELINE is not None
        and global_bindings == _PROOF_CLASS_GLOBAL_BASELINE
    )
    builtin_bindings_match = bool(
        _PROOF_BUILTIN_BASELINE is not None
        and builtin_bindings == _PROOF_BUILTIN_BASELINE
    )
    class_topology_matches = bool(
        _PROOF_CLASS_TOPOLOGY_BASELINE is not None
        and class_topology == _PROOF_CLASS_TOPOLOGY_BASELINE
    )
    source_class_attestation_matches = bool(
        _PROOF_SOURCE_CLASS_ATTESTATION_BASELINE is not None
        and source_class_attestation == _PROOF_SOURCE_CLASS_ATTESTATION_BASELINE
    )
    module_attribute_bindings_match = bool(
        _PROOF_MODULE_ATTRIBUTE_BASELINE is not None
        and module_attribute_bindings == _PROOF_MODULE_ATTRIBUTE_BASELINE
    )
    source_code_attestation_matches = bool(
        _PROOF_SOURCE_CODE_ATTESTATION_BASELINE is not None
        and source_code_attestation == _PROOF_SOURCE_CODE_ATTESTATION_BASELINE
    )
    controller_reads = {
        "BottleneckController": static_class_method_attribute_reads(
            BottleneckController
        ),
        "ProgramFController": static_class_method_attribute_reads(ProgramFController),
    }
    run_prefix_source = (
        textwrap.dedent(inspect.getsource(run_prefix))
        if "run_prefix" in globals()
        else ""
    )
    output_access_inventory = _controller_primary_output_access_inventory()
    output_accesses_match = (
        output_access_inventory == CONTROLLER_PRIMARY_OUTPUT_ACCESS_ALLOWLIST
    )
    controller_reflection_inventory = _controller_reflection_inventory()
    partition = (
        CONTROLLER_TRANSITION_FIELDS
        | CONTROLLER_IMMUTABLE_ROOT_FIELDS
        | CONTROLLER_PRIMARY_OUTPUT_FIELDS
    )
    body = {
        "schema_version": MARKOV_COMPLETENESS_SCHEMA_VERSION,
        "key_schema_version": KEY_SCHEMA_VERSION,
        "theorem_scope": (
            "frozen W24 open-loop M/T/R primary visible-order ReT transition "
            "system only; guardrails remain unaccelerated replay outputs"
        ),
        "source_runtime_binding": certification_provenance(),
        "loaded_proof_callable_bindings": callable_bindings,
        "loaded_proof_callable_bindings_match_import_freeze": (
            callable_bindings_match
        ),
        "runtime_global_bindings": global_bindings,
        "runtime_global_bindings_match_import_freeze": global_bindings_match,
        "runtime_method_bindings": method_bindings,
        "runtime_method_bindings_match_import_freeze": method_bindings_match,
        "runtime_builtin_bindings": builtin_bindings,
        "runtime_builtin_bindings_match_import_freeze": builtin_bindings_match,
        "runtime_class_topology": class_topology,
        "runtime_class_topology_matches_import_freeze": class_topology_matches,
        "source_to_loaded_class_topology_attestation": source_class_attestation,
        "source_to_loaded_class_topology_attestation_matches_import_freeze": (
            source_class_attestation_matches
        ),
        "runtime_module_attribute_bindings": module_attribute_bindings,
        "runtime_module_attribute_bindings_match_import_freeze": (
            module_attribute_bindings_match
        ),
        "source_to_loaded_code_attestation": source_code_attestation,
        "source_to_loaded_code_attestation_matches_import_freeze": (
            source_code_attestation_matches
        ),
        "canonical_encoding": {
            "exact_float_rule": "float.hex",
            "typed_container_rule": True,
            "dictionary_and_set_order_canonicalized": True,
            "merge_uses_canonical_bytes_not_digest_only": True,
            "unknown_mutable_values_fail_closed": True,
        },
        "field_schemas": {
            "controller_allowlist": sorted(CONTROLLER_FIELD_ALLOWLIST),
            "controller_transition_fields": sorted(CONTROLLER_TRANSITION_FIELDS),
            "controller_immutable_roots": sorted(CONTROLLER_IMMUTABLE_ROOT_FIELDS),
            "controller_primary_output_projection": sorted(
                CONTROLLER_PRIMARY_OUTPUT_FIELDS
            ),
            "environment_allowlist": sorted(ENVIRONMENT_FIELD_ALLOWLIST),
            "resource_allowlist": sorted(RESOURCE_FIELD_ALLOWLIST),
            "container_allowlist": sorted(CONTAINER_FIELD_ALLOWLIST),
            "event_allowlists": {
                key: sorted(value)
                for key, value in sorted(EVENT_FIELD_ALLOWLIST.items())
            },
            "process_allowlist": sorted(PROCESS_FIELD_ALLOWLIST),
            "sim_serialized_fields": list(SIM_STATE_FIELDS),
            "sim_special_roots": sorted(SPECIAL_KEY_FIELDS),
            "sim_immutable_fields": sorted(IMMUTABLE_CONTRACT_FIELDS),
        },
        "reachable_read_inventory": {
            "MFSCSimulation": sorted(static_sim_attribute_reads()),
            "controller_methods": controller_reads,
            "future_controller_entrypoints": [
                "BottleneckController.activate_week",
                "ProgramFController.request",
                "ProgramFController._threat_process",
            ],
            "observation_excluded_from_open_loop_runner": (
                bool(run_prefix_source) and ".observation(" not in run_prefix_source
            ),
        },
        "primary_output_projection": {
            "fields": sorted(CONTROLLER_PRIMARY_OUTPUT_FIELDS),
            "rule": (
                "These fields may be appended/accounted during a prefix but are "
                "not inputs to the frozen open-loop physical transition or primary "
                "visible-order ReT label. Observation is not called by run_prefix; "
                "all guardrails are recomputed by unaccelerated winner replay."
            ),
            "exact_access_allowlist": CONTROLLER_PRIMARY_OUTPUT_ACCESS_ALLOWLIST,
            "observed_access_inventory": output_access_inventory,
            "access_inventory_matches": output_accesses_match,
            "reflective_access_inventory": controller_reflection_inventory,
            "reflective_access_absent": not controller_reflection_inventory,
            "source_hash_bound": True,
        },
        "runtime_graph_rules": {
            "per_key_live_schema_enforced": True,
            "root_locals_bound_by_identity_not_variable_name": True,
            "event_payload_and_callback_order_serialized": True,
            "process_target_and_generator_frames_serialized": True,
            "awaited_processes_fail_closed": True,
            "container_resource_queues_serialized": True,
            "domain_order_alias_invariant_checked": True,
            "completed_lost_and_live_orders_serialized": True,
            "all_realized_risk_events_serialized": True,
            "event_process_resource_alias_classes_serialized": True,
            "scheduler_event_ids_normalized_to_queue_rank": True,
            "scheduler_absolute_eid_proved_irrelevant": (
                "Future SimPy ids are monotone and affect only relative tie order; "
                "the complete existing queue order is serialized."
            ),
        },
        "determinism": {
            **entropy,
            "all_rng_states_serialized": True,
            "tape_profile_start_bound_per_key": True,
            "immutable_sim_state_bound_per_key": True,
            "loaded_module_globals_bound_per_key": True,
            "loaded_python_builtins_bound_per_key": True,
            "runtime_class_bases_and_mro_bound_per_key": True,
            "static_module_attribute_chains_bound_per_key": True,
            "dynamic_module_access_fails_closed": not module_attribute_bindings[
                "dynamic_module_access"
            ],
            "executable_method_bodies_bound_per_key": True,
            "callable_defaults_kwdefaults_and_closures_bound_per_key": True,
            "disk_source_independently_compiled_and_attested": (
                source_code_attestation["passed"]
            ),
            "python_hash_randomization": int(sys.flags.hash_randomization),
            "pythonhashseed_environment": os.environ.get("PYTHONHASHSEED", "UNSET"),
            "frozen_modes_exclude_set_order_driven_risk_and_lineage_scheduling": True,
        },
        "controller_partition_complete": partition == CONTROLLER_FIELD_ALLOWLIST,
        "passed": bool(
            partition == CONTROLLER_FIELD_ALLOWLIST
            and output_accesses_match
            and not controller_reflection_inventory
            and callable_bindings_match
            and method_bindings_match
            and global_bindings_match
            and builtin_bindings_match
            and class_topology_matches
            and source_class_attestation_matches
            and class_topology["all_bases_and_mro_members_allowlisted"]
            and source_class_attestation["passed"]
            and module_attribute_bindings_match
            and source_code_attestation_matches
            and source_code_attestation["passed"]
            and not callable_bindings["unsupported"]
            and not global_bindings["unsupported"]
            and not builtin_bindings["unsupported"]
            and builtin_bindings["all_identities_match_import_freeze"]
            and not module_attribute_bindings["unsupported"]
            and not module_attribute_bindings["dynamic_module_access"]
            and not entropy["forbidden_external_entropy_calls"]
            and bool(run_prefix_source)
            and ".observation(" not in run_prefix_source
        ),
    }
    body["certificate_sha256"] = _digest(body)
    return body


def markov_completeness_certificate(
    _binding_guard: Any = _assert_loaded_proof_bindings,
) -> dict[str, Any]:
    """Return a mutation-safe copy of the current machine-bound theorem."""
    _binding_guard(full=True)
    return json.loads(json.dumps(_markov_completeness_certificate_cached()))


def validate_markov_completeness_certificate(
    certificate: dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    if not isinstance(certificate, dict):
        return ["Markov-completeness certificate is not an object"]
    body = dict(certificate)
    claimed = body.pop("certificate_sha256", None)
    if claimed != _digest(body):
        failures.append("Markov-completeness certificate digest mismatch")
    if certificate.get("schema_version") != MARKOV_COMPLETENESS_SCHEMA_VERSION:
        failures.append("Markov-completeness schema mismatch")
    if certificate.get("key_schema_version") != KEY_SCHEMA_VERSION:
        failures.append("Markov-completeness key schema mismatch")
    if certificate.get("passed") is not True:
        failures.append("Markov-completeness theorem is not passed")
    try:
        _assert_loaded_proof_bindings(full=True)
    except RuntimeError as exc:
        failures.append(f"loaded proof runtime binding failure: {exc}")
    expected = json.loads(json.dumps(_markov_completeness_certificate_cached()))
    if certificate != expected:
        failures.append(
            "Markov-completeness certificate differs from current source/runtime theorem"
        )
    return failures


def _bound_method_invariants(owner: Any, names: Iterable[str]) -> bool:
    return all(
        getattr(getattr(owner, name, None), "__self__", None) is owner for name in names
    )


def _runtime_markov_completeness_audit(sim: Any, controller: Any) -> dict[str, Any]:
    """Fail-closed schema/root/alias audit executed for every semantic key."""
    failures: list[str] = []
    try:
        _assert_loaded_proof_bindings()
    except RuntimeError as exc:
        failures.append(f"loaded proof runtime binding failure: {exc}")
    inventory = audit_frozen_state_inventory(sim)
    if not inventory.get("classification_complete"):
        failures.append("simulator live-field classification is incomplete")
    if not inventory.get("all_frozen_invariants_hold"):
        failures.append("frozen simulator invariant failed")
    if inventory.get("static_live_reads_unclassified"):
        failures.append("live simulator read is unclassified")

    # The full inventories are frozen and re-derived at build/certificate entry.
    # Every key executes the content-sensitive fast signature above; using the
    # frozen full hashes here avoids recompiling source and re-walking hundreds
    # of methods for every quotient state.
    if (
        _PROOF_CLASS_GLOBAL_BASELINE is None
        or _PROOF_METHOD_BASELINE is None
        or _PROOF_BUILTIN_BASELINE is None
        or _PROOF_CLASS_TOPOLOGY_BASELINE is None
        or _PROOF_MODULE_ATTRIBUTE_BASELINE is None
    ):
        failures.append("full runtime binding inventories are not frozen")
        runtime_global_bindings = {"binding_sha256": "UNFROZEN"}
        runtime_method_bindings = {"binding_sha256": "UNFROZEN"}
        runtime_builtin_bindings = {"binding_sha256": "UNFROZEN"}
        runtime_class_topology = {"binding_sha256": "UNFROZEN"}
        runtime_module_attribute_bindings = {"binding_sha256": "UNFROZEN"}
    else:
        runtime_global_bindings = _PROOF_CLASS_GLOBAL_BASELINE
        runtime_method_bindings = _PROOF_METHOD_BASELINE
        runtime_builtin_bindings = _PROOF_BUILTIN_BASELINE
        runtime_class_topology = _PROOF_CLASS_TOPOLOGY_BASELINE
        runtime_module_attribute_bindings = _PROOF_MODULE_ATTRIBUTE_BASELINE

    controller_fields = set(vars(controller))
    if controller_fields != set(CONTROLLER_FIELD_ALLOWLIST):
        failures.append(
            "controller live fields differ from exact allowlist: "
            f"{sorted(controller_fields ^ set(CONTROLLER_FIELD_ALLOWLIST))}"
        )
    if type(controller) is not BottleneckController:
        failures.append("controller concrete type is not BottleneckController")
    if getattr(controller, "sim", None) is not sim:
        failures.append("controller.sim is not the serialized simulator root")
    if not isinstance(getattr(controller, "tape", None), dict):
        failures.append("controller tape root is not a dictionary")
    if not isinstance(getattr(controller, "profile", None), dict):
        failures.append("controller profile root is not a dictionary")
    elif _digest(controller.profile) != _digest(controller.tape.get("profile", {})):
        failures.append("controller profile does not match its bound tape profile")
    if tuple(getattr(controller, "active_action", ())) not in ACTIONS:
        failures.append("controller active action is outside the frozen action set")
    if tuple(getattr(controller, "pending_action", ())) not in ACTIONS:
        failures.append("controller pending action is outside the frozen action set")
    for name in ("action_events", "damage_events", "consumed_base_events"):
        if not isinstance(getattr(controller, name, None), list):
            failures.append(f"controller output field {name} is not a list")
    if not isinstance(getattr(controller, "token_hours", None), dict):
        failures.append("controller token-hours ledger is not a dictionary")

    env_fields = set(vars(sim.env))
    if env_fields != set(ENVIRONMENT_FIELD_ALLOWLIST):
        failures.append(
            "Environment live fields differ from exact allowlist: "
            f"{sorted(env_fields ^ set(ENVIRONMENT_FIELD_ALLOWLIST))}"
        )
    if getattr(sim.env, "_active_proc", None) is not None:
        failures.append("Environment checkpoint has an active process")
    if not _bound_method_invariants(
        sim.env, ("process", "timeout", "event", "all_of", "any_of")
    ):
        failures.append("Environment factory bindings drifted")
    try:
        eid_args = sim.env._eid.__reduce__()[1]
        if not (
            isinstance(eid_args, tuple)
            and 1 <= len(eid_args) <= 2
            and isinstance(eid_args[0], int)
            and (len(eid_args) == 1 or eid_args[1] == 1)
        ):
            failures.append("Environment event-id counter is not monotone unit-step")
    except (AttributeError, TypeError, ValueError):
        failures.append("Environment event-id counter cannot be audited")

    resource_rows = []
    for name, resource in (
        ("op10_convoy", sim.op10_convoy),
        ("op12_convoy", sim.op12_convoy),
    ):
        fields_now = set(vars(resource))
        if fields_now != set(RESOURCE_FIELD_ALLOWLIST):
            failures.append(
                f"Resource {name} live fields differ from exact allowlist: "
                f"{sorted(fields_now ^ set(RESOURCE_FIELD_ALLOWLIST))}"
            )
        if getattr(resource, "_env", None) is not sim.env:
            failures.append(f"Resource {name} is bound to another Environment")
        if not _bound_method_invariants(resource, ("request", "release")):
            failures.append(f"Resource {name} method bindings drifted")
        resource_rows.append(
            (
                name,
                f"{type(resource).__module__}.{type(resource).__qualname__}",
                sorted(fields_now),
            )
        )

    container_rows = []
    for name in CONTAINER_FIELDS:
        container = getattr(sim, name)
        fields_now = set(vars(container))
        if fields_now != set(CONTAINER_FIELD_ALLOWLIST):
            failures.append(
                f"Container {name} live fields differ from exact allowlist: "
                f"{sorted(fields_now ^ set(CONTAINER_FIELD_ALLOWLIST))}"
            )
        if getattr(container, "_env", None) is not sim.env:
            failures.append(f"Container {name} is bound to another Environment")
        if not _bound_method_invariants(container, ("put", "get")):
            failures.append(f"Container {name} method bindings drifted")
        container_rows.append(
            (
                name,
                f"{type(container).__module__}.{type(container).__qualname__}",
                sorted(fields_now),
            )
        )

    orders_by_j: dict[int, Any] = {}
    duplicate_order_ids: list[int] = []
    for order in sim.orders:
        j = int(order.j)
        if j in orders_by_j and orders_by_j[j] is not order:
            duplicate_order_ids.append(j)
        orders_by_j[j] = order
    for order in sim.pending_backorders:
        if orders_by_j.get(int(order.j)) is not order:
            failures.append(
                f"pending order {int(order.j)} is not the authoritative orders-list object"
            )
    if duplicate_order_ids:
        failures.append(
            f"duplicate order identities by j: {sorted(set(duplicate_order_ids))}"
        )
    if len({id(event) for event in sim.risk_events}) != len(sim.risk_events):
        failures.append("risk-events list repeats an object identity")

    immutable_sim_state = tuple(
        (name, _semantic(getattr(sim, name)))
        for name in sorted(set(vars(sim)) & set(IMMUTABLE_CONTRACT_FIELDS))
    )
    tape_binding = {
        "tape_sha256": _digest(controller.tape),
        "declared_threat_sha256": controller.tape.get("threat_sha256"),
        "profile_sha256": _digest(controller.profile),
        "start_hex": float(controller.start).hex(),
        "seed": int(controller.tape.get("seed")),
        "weeks": int(controller.tape.get("weeks")),
        "split": str(controller.tape.get("split")),
        "immutable_sim_state_sha256": _digest(immutable_sim_state),
    }
    tape_binding["binding_sha256"] = _digest(tape_binding)
    runtime_schema = {
        "sim_type": f"{type(sim).__module__}.{type(sim).__qualname__}",
        "sim_live_fields": sorted(vars(sim)),
        "sim_immutable_fields_present": [name for name, _ in immutable_sim_state],
        "runtime_global_binding_sha256": runtime_global_bindings["binding_sha256"],
        "runtime_method_binding_sha256": runtime_method_bindings["binding_sha256"],
        "runtime_builtin_binding_sha256": runtime_builtin_bindings["binding_sha256"],
        "runtime_class_topology_sha256": runtime_class_topology["binding_sha256"],
        "runtime_module_attribute_binding_sha256": (
            runtime_module_attribute_bindings["binding_sha256"]
        ),
        "source_to_loaded_code_attestation_sha256": (
            _PROOF_SOURCE_CODE_ATTESTATION_BASELINE["attestation_sha256"]
            if _PROOF_SOURCE_CODE_ATTESTATION_BASELINE is not None
            else "UNFROZEN"
        ),
        "controller_type": f"{type(controller).__module__}.{type(controller).__qualname__}",
        "controller_live_fields": sorted(controller_fields),
        "environment_type": f"{type(sim.env).__module__}.{type(sim.env).__qualname__}",
        "environment_live_fields": sorted(env_fields),
        "resources": resource_rows,
        "containers": container_rows,
        "domain_alias_invariants": [
            "order_j_unique",
            "pending_order_is_authoritative_orders_object",
            "risk_event_object_not_repeated",
        ],
    }
    result = {
        "schema_version": "paper2_runtime_markov_schema_v1",
        "markov_completeness_certificate_sha256": (
            _markov_completeness_certificate_cached()["certificate_sha256"]
        ),
        "runtime_schema": runtime_schema,
        "runtime_schema_sha256": _digest(runtime_schema),
        "tape_binding": tape_binding,
        "immutable_sim_state": immutable_sim_state,
        "failures": failures,
        "passed": not failures,
    }
    result["audit_sha256"] = _digest(result)
    if failures:
        raise TypeError(
            "Markov-completeness runtime audit failed: " + "; ".join(failures)
        )
    return result


def scientific_source_drift() -> str:
    relative_paths = [
        str(path.relative_to(ROOT)) for path in CERTIFICATION_DEPENDENCY_PATHS
    ]
    return (
        _git_value(
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            *relative_paths,
        )
        or ""
    )


def validate_reduced_certification_structure(
    payload: dict[str, Any],
    role: str,
    *,
    expected_environment_sha256: str | None = None,
) -> list[str]:
    """Validate structure/digests only; this does not prove code was executed."""
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
    for key in ("producer_sha256", "dependency_sha256"):
        if provenance.get(key) != expected_provenance.get(key):
            failures.append(f"reduced-horizon provenance field drifted ({key}): {role}")
    environment = provenance.get("environment", {})
    if not isinstance(environment, dict):
        failures.append(f"reduced-horizon environment payload missing: {role}")
        environment = {}
    environment_body = dict(environment)
    claimed_environment_sha256 = environment_body.pop("environment_sha256", None)
    if claimed_environment_sha256 != _digest(environment_body):
        failures.append(f"reduced-horizon environment payload digest mismatch: {role}")
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
        if tape.get("split") != suite["split"]:
            failures.append(f"reduced-horizon split mismatch for tape {index}: {role}")
        if not (
            tape.get("complete_horizon_enumeration") is True
            and tape.get("primary_transducer_bitwise_certified") is True
            and tape.get("calendars_compared") == expected_count
        ):
            failures.append(
                f"reduced-horizon enumeration incomplete for tape {index}: {role}"
            )
        audit = tape.get("all_prefix_callback_audit", {})
        bisimulation = tape.get("collision_bisimulation", {})
        layer_counts = audit.get("layer_semantic_key_evaluations")
        layer_callbacks = audit.get("layer_callback_inventory")
        layer_callback_digests = audit.get("layer_prefix_callback_records_sha256")
        layer_nonempty = audit.get("layer_prefixes_with_nonempty_callback_inventory")
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
            and all(
                re.fullmatch(r"[0-9a-f]{64}", str(row))
                for row in layer_callback_digests
            )
            and re.fullmatch(
                r"[0-9a-f]{64}", str(audit.get("prefix_callback_records_sha256", ""))
            )
        ):
            failures.append(
                f"reduced-horizon all-prefix callback audit failed for tape {index}: {role}"
            )
        if not (
            bisimulation.get("passed") is True
            and bisimulation.get("key_schema_version") == KEY_SCHEMA_VERSION
            and bisimulation.get("complete_state_serialization") is True
            and bisimulation.get("event_payload_serialized") is True
            and bisimulation.get("resource_users_serialized") is True
            and bisimulation.get("callback_closure_state_serialized") is True
            and bisimulation.get("process_target_state_serialized_or_fail_closed")
            is True
            and bisimulation.get("runtime_alias_graph_serialized") is True
            and bisimulation.get("collision_payload_checks")
            == tape.get("collision_count")
            and bisimulation.get("collision_root_count") == tape.get("collision_count")
            and bisimulation.get("unresolved_node_obligation_count") == 0
            and bisimulation.get("unresolved_collision_root_count") == 0
            and bisimulation.get("all_actions_covered") is True
            and bisimulation.get("backward_induction_complete") is True
            and not bisimulation.get("mismatch_examples")
            and re.fullmatch(
                r"[0-9a-f]{64}",
                str(bisimulation.get("transition_record_sha256", "")),
            )
        ):
            failures.append(
                f"reduced-horizon collision bisimulation failed for tape {index}: {role}"
            )
        for failure in validate_collision_bisimulation_certificate(
            bisimulation,
            expected_collision_count=int(tape.get("collision_count", -1)),
            weeks=int(suite["weeks"]),
        ):
            failures.append(
                f"reduced-horizon collision certificate invalid for tape {index}: "
                f"{role}: {failure}"
            )
    if (
        payload.get("summary", {}).get("all_tapes_primary_bitwise_certified")
        is not True
    ):
        failures.append(f"reduced-horizon primary certification failed: {role}")
    return failures


def _reduced_execution_witness(
    payload: dict[str, Any],
    _exclusions: Mapping[str, tuple[str, ...]] = REDUCED_EXECUTION_WITNESS_EXCLUSIONS,
) -> dict[str, Any]:
    """Extract deterministic outputs under an explicit exclusion schema.

    Only wall-clock fields and the custody identity are excluded.  Every
    calendar-output and endpoint commitment remains in the witness.
    """
    witness = copy.deepcopy(payload)
    for name in _exclusions["top_level_identity_fields"]:
        witness.pop(name, None)
    summary = witness.get("summary")
    if isinstance(summary, dict):
        for name in _exclusions["summary_timing_fields"]:
            summary.pop(name, None)
    tapes = witness.get("tapes")
    if isinstance(tapes, list):
        for row in tapes:
            if isinstance(row, dict):
                for name in _exclusions[
                    "per_tape_timing_fields"
                ]:
                    row.pop(name, None)
    return witness


def _interpreter_identity(executable: Path | None = None) -> dict[str, Any]:
    lexical_path = Path(executable or sys.executable).absolute()
    resolved_path = lexical_path.resolve()
    symlink_target = os.readlink(lexical_path) if lexical_path.is_symlink() else None
    return {
        "executable": str(lexical_path),
        "executable_sha256": _file_sha256(lexical_path),
        "resolved_executable": str(resolved_path),
        "resolved_executable_sha256": _file_sha256(resolved_path),
        "lexical_path_is_symlink": lexical_path.is_symlink(),
        "lexical_symlink_target": symlink_target,
        "lexical_symlink_target_sha256": (
            sha256(symlink_target.encode()).hexdigest()
            if symlink_target is not None
            else None
        ),
        "implementation": sys.implementation.name,
        "cache_tag": sys.implementation.cache_tag,
        "version": platform.python_version(),
    }


def _execution_seed_identity(
    seeds: Sequence[tuple[int, str]],
    *,
    weeks: int,
    split: str,
) -> list[dict[str, Any]]:
    return [
        {
            "seed": int(seed),
            "context": str(context),
            "split": str(split),
            "weeks": int(weeks),
            "tape_sha256": materialize_tape(
                int(seed), str(context), str(split), weeks=int(weeks)
            )["threat_sha256"],
        }
        for seed, context in seeds
    ]


def _execution_identity_from_authorization(
    authorization: Mapping[str, Any], authorization_sha256: str
) -> dict[str, Any]:
    return {
        key: authorization.get(key)
        for key in (
            "authorization_id",
            "receipt_id",
            "run_id",
            "launch_nonce",
            "role",
            "execution_role",
            "replay_pair_id",
            "scientific_run",
            "weeks",
            "split",
            "workers",
            "max_calendars",
            "launch_mode",
            "cwd",
            "custody_root",
            "output_initial_device",
            "output_initial_inode",
            "output_initial_sha256",
            "parent_launcher_path",
            "parent_launcher_sha256",
            "isolated_bootstrap_path",
            "isolated_bootstrap_sha256",
            "runtime_attestation_path",
            "host_runtime_sha256",
            "portable_runtime_sha256",
            "scientific_child_environment_sha256",
            "harness_execution_nonce",
            "transport_archive_plan",
            "receipt_signature_scheme",
            "receipt_signing_public_key_der_hex",
            "receipt_signing_public_key_fingerprint",
            "receipt_signing_openssl_identity",
            "source_commit",
            "runner_path",
            "runner_sha256",
            "interpreter",
            "environment_sha256",
            "contract_sha256",
            "seed_identity",
            "authorized_argv_prefix_sha256",
        )
    } | {"authorization_sha256": authorization_sha256}


def create_reduced_execution_launch_authorization(
    authorization_path: Path,
    output_path: Path,
    receipt_path: Path,
    *,
    role: str,
    execution_role: str,
    replay_pair_id: str,
    weeks: int,
    seeds: Sequence[tuple[int, str]],
    split: str,
    workers: int = 1,
    non_scientific_smoke: bool = False,
    max_calendars: int | None = None,
    run_id: str | None = None,
    receipt_signer: EphemeralOpenSSLEd25519Signer | None = None,
    launch_mode: str = "direct",
    custody_root: Path | None = None,
    isolated_bootstrap_path: Path | None = None,
    runtime_attestation_path: Path | None = None,
    host_runtime_sha256: str | None = None,
    portable_runtime_sha256: str | None = None,
    scientific_child_environment: Mapping[str, str] | None = None,
    harness_execution_nonce: str | None = None,
    parent_launcher_path: Path | None = None,
) -> dict[str, Any]:
    """Write a pre-launch authorization whose digest must be retained by caller."""
    authorization_path = _absolute_lexical_path(authorization_path)
    output_path = output_path.resolve()
    receipt_path = receipt_path.resolve()
    if len({authorization_path, output_path, receipt_path}) != 3:
        raise ValueError("authorization, output, and receipt paths must be distinct")
    if any(path.exists() for path in (authorization_path, output_path, receipt_path)):
        raise FileExistsError("execution authorization/output/receipt path already exists")
    if workers < 1:
        raise ValueError("workers must be positive")
    if execution_role not in {"producer", "independent_replay"}:
        raise ValueError("execution_role must be producer or independent_replay")
    if not re.fullmatch(r"[0-9a-f]{64}", replay_pair_id):
        raise ValueError("replay_pair_id must be a caller-generated 256-bit hex token")
    if run_id is not None and not re.fullmatch(r"[0-9a-f]{64}", run_id):
        raise ValueError("run_id must be a 256-bit hex token")
    if receipt_signer is None:
        raise ValueError("a parent-memory Ed25519 receipt signer is required")
    if launch_mode not in {"direct", "isolated_bootstrap"}:
        raise ValueError("launch_mode must be direct or isolated_bootstrap")
    if not non_scientific_smoke and launch_mode != "isolated_bootstrap":
        raise ValueError("scientific execution requires isolated_bootstrap launch mode")
    custody_root = (custody_root or authorization_path.parent).resolve()
    custody_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    custody_stat = custody_root.stat()
    if custody_stat.st_uid != os.geteuid() or custody_stat.st_mode & 0o077:
        raise PermissionError(
            "execution custody_root must be owned by this user and mode 0700-equivalent"
        )
    for path in (authorization_path, output_path, receipt_path):
        try:
            path.relative_to(custody_root)
        except ValueError as exc:
            raise ValueError("execution custody paths must remain inside custody_root") from exc
    runtime_path_value: Path | None = None
    bootstrap_value: Path | None = None
    child_environment = dict(scientific_child_environment or {})
    if launch_mode == "isolated_bootstrap":
        if isolated_bootstrap_path is None or runtime_attestation_path is None:
            raise ValueError("isolated bootstrap and runtime-attestation paths are required")
        bootstrap_value = isolated_bootstrap_path.resolve(strict=True)
        runtime_path_value = runtime_attestation_path.resolve()
        try:
            runtime_path_value.relative_to(custody_root)
        except ValueError as exc:
            raise ValueError("runtime attestation must remain inside custody_root") from exc
        if runtime_path_value.exists():
            raise FileExistsError("runtime-attestation path already exists")
        for name, value in (
            ("host_runtime_sha256", host_runtime_sha256),
            ("portable_runtime_sha256", portable_runtime_sha256),
            ("harness_execution_nonce", harness_execution_nonce),
        ):
            if not re.fullmatch(r"[0-9a-f]{64}", str(value or "")):
                raise ValueError(f"{name} must be a 256-bit hex digest/token")
        environment_failures = _scientific_child_environment_failures(
            child_environment
        )
        if environment_failures:
            raise ValueError(
                "isolated launch child environment failed closed: "
                + "; ".join(environment_failures)
            )
    elif any(
        value is not None
        for value in (
            isolated_bootstrap_path,
            runtime_attestation_path,
            host_runtime_sha256,
            portable_runtime_sha256,
            harness_execution_nonce,
        )
    ):
        raise ValueError("direct launch cannot carry isolated-bootstrap fields")
    if non_scientific_smoke:
        if weeks > 4 or max_calendars is None or max_calendars > 13:
            raise ValueError("non-scientific replay must use W4 or shorter and <=13")
    elif role not in REDUCED_CERTIFICATION_SUITES:
        raise ValueError("scientific reduced execution role is not preregistered")
    if role in REDUCED_CERTIFICATION_SUITES:
        suite = REDUCED_CERTIFICATION_SUITES[role]
        if int(weeks) != int(suite["weeks"]):
            raise ValueError("execution weeks do not match reduced role")
        expected_seeds = tuple((row[0], row[1]) for row in suite["tapes"])
        if tuple(seeds) != expected_seeds:
            raise ValueError("execution seeds do not match reduced role")
        if str(split) != suite["split"]:
            raise ValueError("execution split does not match reduced role")

    provenance = certification_provenance()
    interpreter = _interpreter_identity()
    runner = Path(__file__).resolve()
    parent_launcher = (parent_launcher_path or runner).resolve(strict=True)
    output_descriptor = os.open(
        output_path,
        os.O_CREAT | os.O_EXCL | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        output_stat = os.fstat(output_descriptor)
        os.fsync(output_descriptor)
        receipt_signer.bind_exclusive_output_descriptor(
            output_descriptor, output_path
        )
    except Exception:
        os.close(output_descriptor)
        raise
    runner_arguments = [
        "--weeks",
        str(int(weeks)),
    ]
    for seed, context in seeds:
        runner_arguments.extend(("--seed", f"{int(seed)}:{context}"))
    runner_arguments.extend(
        (
            "--split",
            str(split),
            "--workers",
            str(int(workers)),
            "--output",
            str(output_path),
        )
    )
    if non_scientific_smoke:
        runner_arguments.append("--non-scientific-smoke")
        runner_arguments.extend(("--max-calendars", str(int(max_calendars))))
    runner_arguments.extend(
        (
            "--execution-authorization",
            str(authorization_path),
            "--execution-receipt",
            str(receipt_path),
            "--execution-authorization-sha256",
        )
    )
    if launch_mode == "isolated_bootstrap":
        assert bootstrap_value is not None
        assert runtime_path_value is not None
        argv_prefix = [
            interpreter["executable"],
            "-I",
            "-B",
            "-S",
            str(bootstrap_value),
            "--repo-root",
            str(ROOT),
            "--runner",
            str(runner),
            "--attestation-output",
            str(runtime_path_value),
            "--expected-runtime-sha256",
            str(host_runtime_sha256),
            "--execution-nonce",
            str(harness_execution_nonce),
            "--execution-role",
            execution_role,
            "--",
            *runner_arguments,
        ]
    else:
        argv_prefix = [
            interpreter["executable"],
            "-I",
            "-B",
            str(runner),
            *runner_arguments,
        ]
    payload = {
        "schema_version": REDUCED_EXECUTION_AUTHORIZATION_SCHEMA_VERSION,
        "authorization_id": secrets.token_hex(32),
        "receipt_id": secrets.token_hex(32),
        "run_id": run_id or secrets.token_hex(32),
        "launch_nonce": secrets.token_hex(32),
        "role": str(role),
        "execution_role": execution_role,
        "replay_pair_id": replay_pair_id,
        "scientific_run": not non_scientific_smoke,
        "weeks": int(weeks),
        "split": str(split),
        "workers": int(workers),
        "max_calendars": int(max_calendars) if max_calendars is not None else None,
        "launch_mode": launch_mode,
        "cwd": str(ROOT),
        "custody_root": str(custody_root),
        "output_initial_device": int(output_stat.st_dev),
        "output_initial_inode": int(output_stat.st_ino),
        "output_initial_sha256": sha256(b"").hexdigest(),
        "parent_launcher_path": str(parent_launcher),
        "parent_launcher_sha256": _file_sha256(parent_launcher),
        "isolated_bootstrap_path": (
            str(bootstrap_value) if bootstrap_value is not None else None
        ),
        "isolated_bootstrap_sha256": (
            _file_sha256(bootstrap_value) if bootstrap_value is not None else None
        ),
        "runtime_attestation_path": (
            str(runtime_path_value) if runtime_path_value is not None else None
        ),
        "host_runtime_sha256": host_runtime_sha256,
        "portable_runtime_sha256": portable_runtime_sha256,
        "scientific_child_environment": child_environment,
        "scientific_child_environment_sha256": _digest(child_environment),
        "harness_execution_nonce": harness_execution_nonce,
        "transport_archive_plan": {
            "schema_version": "paper2_reduced_transport_archive_plan_v1",
            "custody_root": str(custody_root),
            "relative_artifacts": {
                "authorization": str(authorization_path.relative_to(custody_root)),
                "output": str(output_path.relative_to(custody_root)),
                "runtime_attestation": (
                    str(runtime_path_value.relative_to(custody_root))
                    if runtime_path_value is not None
                    else None
                ),
                "exact_receipt": str(receipt_path.relative_to(custody_root)),
                "stdout_log": str(
                    receipt_path.with_suffix(receipt_path.suffix + ".stdout.log")
                    .relative_to(custody_root)
                ),
                "stderr_log": str(
                    receipt_path.with_suffix(receipt_path.suffix + ".stderr.log")
                    .relative_to(custody_root)
                ),
            },
            "relocation_requires_signed_receipt_and_content_hashes": True,
            "launch_host_inode_claims_not_portable": True,
        },
        "receipt_signature_scheme": "OpenSSL-Ed25519",
        "receipt_signing_public_key_der_hex": receipt_signer.public_key_der_hex,
        "receipt_signing_public_key_fingerprint": receipt_signer.fingerprint,
        "receipt_signing_openssl_identity": receipt_signer.openssl_identity,
        "source_commit": provenance["git_commit"],
        "runner_path": str(runner),
        "runner_sha256": _file_sha256(runner),
        "interpreter": interpreter,
        "environment_sha256": provenance["environment"]["environment_sha256"],
        "contract_sha256": _file_sha256(CONTRACT_PATH),
        "seed_identity": _execution_seed_identity(
            seeds, weeks=int(weeks), split=str(split)
        ),
        "authorization_path": str(authorization_path),
        "output_path": str(output_path),
        "receipt_path": str(receipt_path),
        "authorized_argv_prefix": argv_prefix,
        "authorized_argv_prefix_sha256": _digest(argv_prefix),
        "created_before_child_launch": True,
    }
    payload["authorization_body_sha256"] = _digest(payload)
    authorization_path.parent.mkdir(parents=True, exist_ok=True)
    authorization_descriptor = os.open(
        authorization_path,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    with os.fdopen(authorization_descriptor, "w") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    authorization_sha256 = _file_sha256(authorization_path)
    return {
        "authorization_path": str(authorization_path),
        "authorization_sha256": authorization_sha256,
        "prelaunch_signing_public_key_fingerprint": receipt_signer.fingerprint,
        "materialized_argv": argv_prefix + [authorization_sha256],
        "execution_identity": _execution_identity_from_authorization(
            payload, authorization_sha256
        ),
    }


def _execution_authorization_failures(
    authorization: Mapping[str, Any],
    authorization_sha256: str,
    *,
    actual_argv: Sequence[str] | None = None,
    enforce_current_interpreter: bool = True,
    enforce_current_openssl: bool = True,
    expected_environment_sha256: str | None = None,
) -> list[str]:
    failures: list[str] = []
    body = dict(authorization)
    claimed_body_sha256 = body.pop("authorization_body_sha256", None)
    if claimed_body_sha256 != _digest(body):
        failures.append("execution authorization body digest mismatch")
    if authorization.get("schema_version") != REDUCED_EXECUTION_AUTHORIZATION_SCHEMA_VERSION:
        failures.append("execution authorization schema mismatch")
        if authorization.get("schema_version") == (
            "paper2_reduced_execution_launch_authorization_v1"
        ):
            failures.append("legacy unsigned-custody authorization v1 is rejected")
    for name in ("authorization_id", "receipt_id", "run_id", "launch_nonce"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(authorization.get(name, ""))):
            failures.append(f"execution authorization {name} is malformed")
    if authorization.get("execution_role") not in {"producer", "independent_replay"}:
        failures.append("execution authorization execution_role is malformed")
    if not re.fullmatch(
        r"[0-9a-f]{64}", str(authorization.get("replay_pair_id", ""))
    ):
        failures.append("execution authorization replay_pair_id is malformed")
    if not isinstance(authorization.get("role"), str) or not authorization.get("role"):
        failures.append("execution authorization role is malformed")
    if not isinstance(authorization.get("scientific_run"), bool):
        failures.append("execution authorization scientific_run is malformed")
    custody_paths = []
    for name in ("authorization_path", "output_path", "receipt_path"):
        value = authorization.get(name)
        if not isinstance(value, str) or not Path(value).is_absolute():
            failures.append(f"execution authorization {name} is not absolute")
        else:
            custody_paths.append(Path(value))
    if len(custody_paths) != 3 or len(set(custody_paths)) != 3:
        failures.append("execution authorization custody paths are not distinct")
    prefix = authorization.get("authorized_argv_prefix")
    if (
        not isinstance(prefix, list)
        or any(not isinstance(value, str) for value in prefix)
        or authorization.get("authorized_argv_prefix_sha256") != _digest(prefix)
    ):
        failures.append("execution authorization argv prefix is malformed")
        prefix = []
    interpreter = authorization.get("interpreter")
    if not isinstance(interpreter, dict):
        failures.append("execution authorization interpreter identity is malformed")
        interpreter = {}
    for name in (
        "executable",
        "executable_sha256",
        "resolved_executable",
        "resolved_executable_sha256",
        "implementation",
        "cache_tag",
        "version",
    ):
        if not isinstance(interpreter.get(name), str) or not interpreter.get(name):
            failures.append(
                f"execution authorization interpreter identity is malformed: {name}"
            )
    for name in ("executable_sha256", "resolved_executable_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(interpreter.get(name, ""))):
            failures.append(
                f"execution authorization interpreter digest is malformed: {name}"
            )
    launch_mode = authorization.get("launch_mode")
    if launch_mode not in {"direct", "isolated_bootstrap"}:
        failures.append("execution authorization launch_mode is malformed")
    if authorization.get("scientific_run") is True and launch_mode != "isolated_bootstrap":
        failures.append("scientific execution authorization is not isolated-bootstrap")
    public_hex = authorization.get("receipt_signing_public_key_der_hex")
    public_fingerprint = (
        sha256(bytes.fromhex(public_hex)).hexdigest()
        if isinstance(public_hex, str)
        and re.fullmatch(r"[0-9a-f]+", public_hex)
        and len(public_hex) % 2 == 0
        else None
    )
    if authorization.get("receipt_signature_scheme") != "OpenSSL-Ed25519":
        failures.append("execution authorization receipt signature scheme mismatch")
    if authorization.get("receipt_signing_public_key_fingerprint") != public_fingerprint:
        failures.append("execution authorization signing-key fingerprint mismatch")
    signing_openssl = authorization.get("receipt_signing_openssl_identity")
    if not isinstance(signing_openssl, dict) or not re.fullmatch(
        r"[0-9a-f]{64}", str(signing_openssl.get("executable_sha256", ""))
    ):
        failures.append("execution authorization signing OpenSSL identity is malformed")
    elif enforce_current_openssl and signing_openssl != _openssl_identity():
        failures.append("execution authorization signing OpenSSL identity drifted")
    expected_argv = prefix + [authorization_sha256]
    if actual_argv is not None and list(actual_argv) != expected_argv:
        failures.append("executed argv differs from pre-launch authorization")
    provenance = certification_provenance()
    expected = {
        "source_commit": provenance["git_commit"],
        "runner_sha256": _file_sha256(Path(__file__).resolve()),
        "environment_sha256": (
            expected_environment_sha256
            or provenance["environment"]["environment_sha256"]
        ),
        "contract_sha256": _file_sha256(CONTRACT_PATH),
        "created_before_child_launch": True,
    }
    if enforce_current_interpreter:
        expected["interpreter"] = _interpreter_identity()
        expected["runner_path"] = str(Path(__file__).resolve())
    elif not (
        isinstance(authorization.get("runner_path"), str)
        and Path(authorization["runner_path"]).is_absolute()
    ):
        failures.append("execution authorization runner_path is not absolute")
    for key, value in expected.items():
        if authorization.get(key) != value:
            failures.append(f"execution authorization current binding mismatch: {key}")
    cwd = authorization.get("cwd")
    custody_root = authorization.get("custody_root")
    if not isinstance(cwd, str) or not Path(cwd).is_absolute():
        failures.append("execution authorization cwd is malformed")
    if not isinstance(custody_root, str) or not Path(custody_root).is_absolute():
        failures.append("execution authorization custody_root is malformed")
    elif any(
        not isinstance(authorization.get(name), str)
        or not Path(str(authorization.get(name))).is_relative_to(Path(custody_root))
        for name in ("authorization_path", "output_path", "receipt_path")
    ):
        failures.append("execution authorization path escapes custody_root")
    if authorization.get("output_initial_sha256") != sha256(b"").hexdigest():
        failures.append("execution authorization output was not initially empty")
    for name in ("output_initial_device", "output_initial_inode"):
        if not isinstance(authorization.get(name), int) or authorization.get(name) <= 0:
            failures.append(f"execution authorization {name} is malformed")
    parent_launcher_path = authorization.get("parent_launcher_path")
    if not isinstance(parent_launcher_path, str) or not Path(
        parent_launcher_path
    ).is_absolute():
        failures.append("execution authorization parent launcher path is malformed")
    if not re.fullmatch(
        r"[0-9a-f]{64}", str(authorization.get("parent_launcher_sha256", ""))
    ):
        failures.append("execution authorization parent launcher hash is malformed")
    elif enforce_current_interpreter:
        parent_path = Path(str(parent_launcher_path))
        if (
            not parent_path.is_file()
            or _file_sha256(parent_path)
            != authorization.get("parent_launcher_sha256")
        ):
            failures.append("execution authorization parent launcher bytes drifted")
    child_environment = authorization.get("scientific_child_environment")
    if not isinstance(child_environment, dict):
        failures.append("execution authorization child environment is malformed")
        child_environment = {}
    if launch_mode == "isolated_bootstrap":
        failures.extend(_scientific_child_environment_failures(child_environment))
    elif child_environment:
        failures.append("direct execution carries a nonempty child environment")
    if authorization.get("scientific_child_environment_sha256") != _digest(
        child_environment
    ):
        failures.append("execution authorization child environment digest mismatch")
    if launch_mode == "isolated_bootstrap":
        for name in (
            "isolated_bootstrap_sha256",
            "host_runtime_sha256",
            "portable_runtime_sha256",
            "harness_execution_nonce",
        ):
            if not re.fullmatch(r"[0-9a-f]{64}", str(authorization.get(name, ""))):
                failures.append(f"execution authorization {name} is malformed")
        for name in ("isolated_bootstrap_path", "runtime_attestation_path"):
            if not isinstance(authorization.get(name), str) or not Path(
                str(authorization.get(name))
            ).is_absolute():
                failures.append(f"execution authorization {name} is malformed")
        if (
            isinstance(custody_root, str)
            and isinstance(authorization.get("runtime_attestation_path"), str)
            and not Path(authorization["runtime_attestation_path"]).is_relative_to(
                Path(custody_root)
            )
        ):
            failures.append("runtime attestation escapes custody_root")
        if enforce_current_interpreter:
            bootstrap_path = Path(str(authorization.get("isolated_bootstrap_path", "")))
            if (
                not bootstrap_path.is_file()
                or _file_sha256(bootstrap_path)
                != authorization.get("isolated_bootstrap_sha256")
            ):
                failures.append("isolated bootstrap bytes drifted")
    else:
        for name in (
            "isolated_bootstrap_path",
            "isolated_bootstrap_sha256",
            "runtime_attestation_path",
            "host_runtime_sha256",
            "portable_runtime_sha256",
            "harness_execution_nonce",
        ):
            if authorization.get(name) is not None:
                failures.append(f"direct execution carries isolated field: {name}")
    if isinstance(custody_root, str) and Path(custody_root).is_absolute():
        custody_root_path = Path(custody_root)
        try:
            authorized_receipt_path = Path(str(authorization.get("receipt_path", "")))
            expected_transport_plan = {
                "schema_version": "paper2_reduced_transport_archive_plan_v1",
                "custody_root": custody_root,
                "relative_artifacts": {
                    "authorization": str(
                        Path(str(authorization.get("authorization_path", "")))
                        .relative_to(custody_root_path)
                    ),
                    "output": str(
                        Path(str(authorization.get("output_path", ""))).relative_to(
                            custody_root_path
                        )
                    ),
                    "runtime_attestation": (
                        str(
                            Path(str(authorization.get("runtime_attestation_path")))
                            .relative_to(custody_root_path)
                        )
                        if authorization.get("runtime_attestation_path") is not None
                        else None
                    ),
                    "exact_receipt": str(
                        authorized_receipt_path.relative_to(custody_root_path)
                    ),
                    "stdout_log": str(
                        authorized_receipt_path.with_suffix(
                            authorized_receipt_path.suffix + ".stdout.log"
                        ).relative_to(custody_root_path)
                    ),
                    "stderr_log": str(
                        authorized_receipt_path.with_suffix(
                            authorized_receipt_path.suffix + ".stderr.log"
                        ).relative_to(custody_root_path)
                    ),
                },
                "relocation_requires_signed_receipt_and_content_hashes": True,
                "launch_host_inode_claims_not_portable": True,
            }
        except ValueError:
            failures.append("execution authorization transport paths escape custody root")
        else:
            if authorization.get("transport_archive_plan") != expected_transport_plan:
                failures.append("execution authorization transport archive plan mismatch")
    if not re.fullmatch(r"[0-9a-f]{40}", str(authorization.get("source_commit", ""))):
        failures.append("execution authorization source_commit is malformed")
    for name in ("runner_sha256", "environment_sha256", "contract_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(authorization.get(name, ""))):
            failures.append(f"execution authorization {name} is malformed")
    seed_identity = authorization.get("seed_identity")
    if not isinstance(seed_identity, list) or not seed_identity:
        failures.append("execution authorization seed identity is missing")
    else:
        for row in seed_identity:
            if not isinstance(row, dict) or not re.fullmatch(
                r"[0-9a-f]{64}", str(row.get("tape_sha256", ""))
            ):
                failures.append("execution authorization seed identity is malformed")
                break
    weeks = authorization.get("weeks")
    split = authorization.get("split")
    workers = authorization.get("workers")
    max_calendars = authorization.get("max_calendars")
    if not isinstance(weeks, int) or not 1 <= weeks <= 24:
        failures.append("execution authorization weeks is malformed")
    if not isinstance(split, str) or not split:
        failures.append("execution authorization split is malformed")
    if not isinstance(workers, int) or workers < 1:
        failures.append("execution authorization workers is malformed")
    if max_calendars is not None and (
        not isinstance(max_calendars, int) or max_calendars < 1
    ):
        failures.append("execution authorization max_calendars is malformed")
    if authorization.get("scientific_run") is False and not (
        isinstance(weeks, int)
        and weeks <= 4
        and isinstance(max_calendars, int)
        and max_calendars <= 13
    ):
        failures.append("non-scientific execution authorization smoke cap is invalid")
    if (
        isinstance(seed_identity, list)
        and seed_identity
        and all(isinstance(row, dict) for row in seed_identity)
        and isinstance(weeks, int)
        and isinstance(split, str)
        and isinstance(workers, int)
    ):
        canonical_runner_arguments = [
            "--weeks",
            str(weeks),
        ]
        for row in seed_identity:
            canonical_runner_arguments.extend(
                ("--seed", f"{row.get('seed')}:{row.get('context')}")
            )
        canonical_runner_arguments.extend(
            (
                "--split",
                split,
                "--workers",
                str(workers),
                "--output",
                str(authorization.get("output_path", "")),
            )
        )
        if authorization.get("scientific_run") is False:
            canonical_runner_arguments.append("--non-scientific-smoke")
            canonical_runner_arguments.extend(
                ("--max-calendars", str(max_calendars))
            )
        elif max_calendars is not None:
            failures.append(
                "scientific execution authorization cannot cap calendars"
            )
        canonical_runner_arguments.extend(
            (
                "--execution-authorization",
                str(authorization.get("authorization_path", "")),
                "--execution-receipt",
                str(authorization.get("receipt_path", "")),
                "--execution-authorization-sha256",
            )
        )
        if launch_mode == "isolated_bootstrap":
            canonical_prefix = [
                str(interpreter.get("executable", "")),
                "-I",
                "-B",
                "-S",
                str(authorization.get("isolated_bootstrap_path", "")),
                "--repo-root",
                str(ROOT),
                "--runner",
                str(authorization.get("runner_path", "")),
                "--attestation-output",
                str(authorization.get("runtime_attestation_path", "")),
                "--expected-runtime-sha256",
                str(authorization.get("host_runtime_sha256", "")),
                "--execution-nonce",
                str(authorization.get("harness_execution_nonce", "")),
                "--execution-role",
                str(authorization.get("execution_role", "")),
                "--",
                *canonical_runner_arguments,
            ]
        else:
            canonical_prefix = [
                str(interpreter.get("executable", "")),
                "-I",
                "-B",
                str(authorization.get("runner_path", "")),
                *canonical_runner_arguments,
            ]
        if prefix != canonical_prefix:
            failures.append(
                "execution authorization argv is not canonical for its scientific scope"
            )
    role = authorization.get("role")
    if authorization.get("scientific_run") is True:
        suite = REDUCED_CERTIFICATION_SUITES.get(str(role))
        if suite is None:
            failures.append("scientific execution authorization role is unknown")
        elif weeks != int(suite["weeks"]) or split != suite["split"]:
            failures.append("scientific execution authorization scope mismatch")
        elif not (
            isinstance(seed_identity, list)
            and len(seed_identity) == len(suite["tapes"])
            and all(isinstance(row, dict) for row in seed_identity)
        ):
            failures.append("scientific execution authorization scope mismatch")
        else:
            expected_scope = [
                {
                    "seed": seed,
                    "context": context,
                    "split": suite["split"],
                    "weeks": int(suite["weeks"]),
                    "tape_sha256": tape_sha,
                }
                for index, (seed, context, tape_sha) in enumerate(suite["tapes"])
            ]
            if seed_identity != expected_scope:
                failures.append("scientific execution authorization scope mismatch")
    return failures


def _read_nofollow(path: Path) -> tuple[bytes, os.stat_result]:
    path = _absolute_lexical_path(path)
    before = path.lstat()
    if path.is_symlink():
        raise OSError("custody path is a symbolic link")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        stat_result = os.fstat(descriptor)
        if (
            stat_result.st_dev != before.st_dev
            or stat_result.st_ino != before.st_ino
        ):
            raise OSError("custody path inode changed while opening")
        blocks = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            blocks.append(block)
        after = path.lstat()
        if (
            path.is_symlink()
            or after.st_dev != stat_result.st_dev
            or after.st_ino != stat_result.st_ino
        ):
            raise OSError("custody path inode changed while reading")
        return b"".join(blocks), stat_result
    finally:
        os.close(descriptor)


def _read_bound_descriptor(descriptor: int) -> tuple[bytes, os.stat_result]:
    stat_result = os.fstat(descriptor)
    os.lseek(descriptor, 0, os.SEEK_SET)
    blocks = []
    while True:
        block = os.read(descriptor, 1024 * 1024)
        if not block:
            break
        blocks.append(block)
    return b"".join(blocks), stat_result


def _open_file_descriptors() -> set[int]:
    for root in (Path("/dev/fd"), Path("/proc/self/fd")):
        if root.is_dir():
            candidates = {
                int(path.name) for path in root.iterdir() if path.name.isdigit()
            }
            live: set[int] = set()
            for descriptor in candidates:
                try:
                    os.fstat(descriptor)
                except OSError:
                    continue
                live.add(descriptor)
            return live
    live = set()
    for descriptor in range(3, 256):
        try:
            os.fstat(descriptor)
        except OSError:
            continue
        live.add(descriptor)
    return live


def _posix_spawn_and_wait(
    argv: Sequence[str],
    environment: Mapping[str, str],
    *,
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: float | None,
) -> tuple[
    int,
    int,
    bytes,
    bytes,
    os.stat_result,
    os.stat_result,
    tuple[int, ...],
]:
    if not hasattr(os, "posix_spawn"):
        raise RuntimeError("trusted receipt launcher requires os.posix_spawn")
    log_flags = os.O_CREAT | os.O_EXCL | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    stdout_descriptor = os.open(stdout_path, log_flags, 0o600)
    try:
        stderr_descriptor = os.open(stderr_path, log_flags, 0o600)
    except Exception:
        os.close(stdout_descriptor)
        raise
    log_descriptors = {stdout_descriptor, stderr_descriptor}
    keep_for_child = {0, 1, 2, *log_descriptors}
    closed_descriptors = tuple(
        sorted(
            fd
            for fd in _open_file_descriptors()
            if fd > 2 and fd not in keep_for_child
        )
    )
    file_actions = [
        (os.POSIX_SPAWN_DUP2, stdout_descriptor, 1),
        (os.POSIX_SPAWN_DUP2, stderr_descriptor, 2),
        (os.POSIX_SPAWN_CLOSE, stdout_descriptor),
        (os.POSIX_SPAWN_CLOSE, stderr_descriptor),
        *(
            (os.POSIX_SPAWN_CLOSE, descriptor)
            for descriptor in closed_descriptors
        ),
    ]
    try:
        pid = os.posix_spawn(
            argv[0],
            list(argv),
            dict(environment),
            file_actions=file_actions,
        )
        deadline = (
            time.monotonic() + timeout_seconds
            if timeout_seconds is not None
            else None
        )
        returncode: int | None = None
        while returncode is None:
            waited_pid, status = os.waitpid(pid, os.WNOHANG)
            if waited_pid == pid:
                returncode = os.waitstatus_to_exitcode(status)
                break
            if deadline is not None and time.monotonic() >= deadline:
                os.kill(pid, signal.SIGKILL)
                _waited_pid, status = os.waitpid(pid, 0)
                returncode = os.waitstatus_to_exitcode(status)
                break
            time.sleep(0.01)
        stdout, stdout_stat = _read_bound_descriptor(stdout_descriptor)
        stderr, stderr_stat = _read_bound_descriptor(stderr_descriptor)
        for path, stat_result in (
            (stdout_path, stdout_stat),
            (stderr_path, stderr_stat),
        ):
            path_stat = path.lstat()
            if (
                path.is_symlink()
                or path_stat.st_dev != stat_result.st_dev
                or path_stat.st_ino != stat_result.st_ino
            ):
                raise RuntimeError("child log inode changed during execution")
        return (
            pid,
            int(returncode),
            stdout,
            stderr,
            stdout_stat,
            stderr_stat,
            closed_descriptors,
        )
    finally:
        os.close(stdout_descriptor)
        os.close(stderr_descriptor)


def launch_reduced_execution_fresh_process(
    authorization_path: Path,
    *,
    expected_authorization_sha256: str,
    receipt_signer: EphemeralOpenSSLEd25519Signer,
    expected_public_key_fingerprint: str,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Launch via POSIX spawn and sign a post-exit receipt in the trusted parent."""
    authorization_path = _absolute_lexical_path(authorization_path)
    failures: list[str] = []
    try:
        authorization_bytes, authorization_stat = _read_nofollow(authorization_path)
        authorization = json.loads(authorization_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        receipt_signer.close()
        return {"passed": False, "failures": [f"authorization read failed: {exc}"]}
    actual_authorization_sha256 = sha256(authorization_bytes).hexdigest()
    if actual_authorization_sha256 != expected_authorization_sha256:
        failures.append("caller-retained execution authorization digest mismatch")
    if receipt_signer.fingerprint != expected_public_key_fingerprint:
        failures.append("caller-retained prelaunch signing-key fingerprint mismatch")
    if (
        authorization.get("receipt_signing_public_key_fingerprint")
        != expected_public_key_fingerprint
        or authorization.get("receipt_signing_public_key_der_hex")
        != receipt_signer.public_key_der_hex
    ):
        failures.append("authorization does not bind the parent-memory signing key")
    argv = list(authorization.get("authorized_argv_prefix", [])) + [
        expected_authorization_sha256
    ]
    failures.extend(
        _execution_authorization_failures(
            authorization,
            expected_authorization_sha256,
            actual_argv=argv,
        )
    )
    output_path = _absolute_lexical_path(
        Path(str(authorization.get("output_path", "")))
    )
    receipt_path = _absolute_lexical_path(
        Path(str(authorization.get("receipt_path", "")))
    )
    stdout_path = receipt_path.with_suffix(receipt_path.suffix + ".stdout.log")
    stderr_path = receipt_path.with_suffix(receipt_path.suffix + ".stderr.log")
    runtime_path_value = authorization.get("runtime_attestation_path")
    runtime_path = (
        _absolute_lexical_path(Path(runtime_path_value))
        if isinstance(runtime_path_value, str)
        else None
    )
    try:
        output_descriptor = receipt_signer.bound_output_descriptor(output_path)
        output_before = os.fstat(output_descriptor)
        output_lstat = output_path.lstat()
    except (OSError, RuntimeError) as exc:
        failures.append(f"parent-bound output descriptor failed: {exc}")
        output_descriptor = -1
        output_before = None
        output_lstat = None
    if output_before is not None and (
        output_before.st_dev != authorization.get("output_initial_device")
        or output_before.st_ino != authorization.get("output_initial_inode")
        or output_before.st_size != 0
        or output_lstat is None
        or output_lstat.st_dev != output_before.st_dev
        or output_lstat.st_ino != output_before.st_ino
        or output_path.is_symlink()
    ):
        failures.append("exclusive output inode changed before child launch")
    if (
        receipt_path.exists()
        or stdout_path.exists()
        or stderr_path.exists()
        or (runtime_path is not None and runtime_path.exists())
    ):
        failures.append("receipt, log, or runtime-attestation path exists before launch")
    if Path.cwd().resolve() != Path(str(authorization.get("cwd", ""))).resolve():
        failures.append("trusted parent cwd differs from authorization")
    if failures:
        receipt_signer.close()
        return {"passed": False, "failures": failures}
    try:
        (
            child_pid,
            returncode,
            stdout,
            stderr,
            stdout_stat,
            stderr_stat,
            closed_descriptors,
        ) = (
            _posix_spawn_and_wait(
                argv,
                authorization["scientific_child_environment"],
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                timeout_seconds=timeout_seconds,
            )
        )
    except (OSError, RuntimeError) as exc:
        receipt_signer.close()
        return {
            "passed": False,
            "failures": [f"trusted POSIX parent launch failed: {exc}"],
        }
    if returncode != 0:
        receipt_signer.close()
        return {
            "passed": False,
            "child_pid": child_pid,
            "returncode": returncode,
            "failures": ["independent execution child failed"],
            "stdout": stdout.decode(errors="replace"),
            "stderr": stderr.decode(errors="replace"),
        }
    try:
        output_bytes, output_after = _read_bound_descriptor(output_descriptor)
        output_lstat_after = output_path.lstat()
        output = json.loads(output_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        receipt_signer.close()
        return {
            "passed": False,
            "child_pid": child_pid,
            "returncode": returncode,
            "failures": [f"child output validation failed: {exc}"],
        }
    if (
        output_after.st_dev != authorization["output_initial_device"]
        or output_after.st_ino != authorization["output_initial_inode"]
        or output_after.st_size != len(output_bytes)
        or output_lstat_after.st_dev != output_after.st_dev
        or output_lstat_after.st_ino != output_after.st_ino
        or output_path.is_symlink()
    ):
        receipt_signer.close()
        return {
            "passed": False,
            "child_pid": child_pid,
            "returncode": returncode,
            "failures": ["exclusive output inode changed during child execution"],
        }
    output_sha256 = sha256(output_bytes).hexdigest()
    execution_identity = _execution_identity_from_authorization(
        authorization, expected_authorization_sha256
    )
    if output.get("execution_identity") != execution_identity:
        receipt_signer.close()
        return {
            "passed": False,
            "child_pid": child_pid,
            "returncode": returncode,
            "failures": ["child output execution identity mismatch"],
        }
    runtime_attestation = None
    runtime_attestation_file_sha256 = None
    runtime_stat = None
    if authorization.get("launch_mode") == "isolated_bootstrap":
        if runtime_path is None:
            receipt_signer.close()
            return {"passed": False, "failures": ["runtime attestation path missing"]}
        try:
            runtime_bytes, runtime_stat = _read_nofollow(runtime_path)
            runtime_attestation = json.loads(runtime_bytes)
        except (OSError, json.JSONDecodeError) as exc:
            receipt_signer.close()
            return {
                "passed": False,
                "failures": [f"runtime attestation validation failed: {exc}"],
            }
        runtime_attestation_file_sha256 = sha256(runtime_bytes).hexdigest()
        runtime_body = dict(runtime_attestation)
        runtime_sha256 = runtime_body.pop("runtime_sha256", None)
        if not (
            runtime_attestation.get("schema_version")
            == "paper2_isolated_runtime_attestation_v2"
            and runtime_sha256 == _digest(runtime_body)
            and runtime_sha256 == authorization.get("host_runtime_sha256")
            and runtime_attestation.get("portable_sha256")
            == authorization.get("portable_runtime_sha256")
            and runtime_attestation.get("isolation_checks_passed") is True
        ):
            receipt_signer.close()
            return {
                "passed": False,
                "failures": ["runtime attestation content failed closed"],
            }
    receipt = {
        "schema_version": REDUCED_EXECUTION_RECEIPT_SCHEMA_VERSION,
        **execution_identity,
        "authorization_path": str(authorization_path),
        "authorization_device": int(authorization_stat.st_dev),
        "authorization_inode": int(authorization_stat.st_ino),
        "materialized_argv": argv,
        "materialized_argv_sha256": _digest(argv),
        "fresh_child_process": True,
        "trusted_parent_pid": os.getpid(),
        "trusted_parent_ppid": os.getppid(),
        "trusted_parent_hostname": platform.node(),
        "trusted_parent_process_nonce": _TRUSTED_PARENT_PROCESS_NONCE,
        "parent_launch_primitive": "os.posix_spawn",
        "parent_preexec_fn_used": False,
        "child_closed_file_descriptors": list(closed_descriptors),
        "child_pid": int(child_pid),
        "returncode": int(returncode),
        "stdout_sha256": sha256(stdout).hexdigest(),
        "stderr_sha256": sha256(stderr).hexdigest(),
        "stdout_path": str(stdout_path),
        "stdout_device": int(stdout_stat.st_dev),
        "stdout_inode": int(stdout_stat.st_ino),
        "stdout_bytes": len(stdout),
        "stderr_path": str(stderr_path),
        "stderr_device": int(stderr_stat.st_dev),
        "stderr_inode": int(stderr_stat.st_ino),
        "stderr_bytes": len(stderr),
        "output_path": str(output_path),
        "output_device": int(output_after.st_dev),
        "output_inode": int(output_after.st_ino),
        "output_bytes": len(output_bytes),
        "output_sha256": output_sha256,
        "runtime_attestation_path": str(runtime_path) if runtime_path else None,
        "runtime_attestation_device": (
            int(runtime_stat.st_dev) if runtime_stat is not None else None
        ),
        "runtime_attestation_inode": (
            int(runtime_stat.st_ino) if runtime_stat is not None else None
        ),
        "runtime_attestation_file_sha256": runtime_attestation_file_sha256,
        "output_execution_identity_sha256": _digest(execution_identity),
        "written_after_child_exit_and_output_validation": True,
        "cross_tree_read_isolation": (
            "NOT_ENFORCED_TRUSTED_PINNED_RUNNER_RESIDUAL"
        ),
        "tee_or_proof_of_computation_claimed": False,
    }
    receipt["trusted_parent_process_identity_sha256"] = _digest(
        {
            key: receipt[key]
            for key in (
                "trusted_parent_pid",
                "trusted_parent_ppid",
                "trusted_parent_hostname",
                "trusted_parent_process_nonce",
                "parent_launcher_path",
                "parent_launcher_sha256",
            )
        }
    )
    receipt = sign_reduced_execution_receipt(receipt, receipt_signer)
    receipt_signer.close()
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        receipt_descriptor = os.open(
            receipt_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        with os.fdopen(receipt_descriptor, "w") as stream:
            stream.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        return {
            "passed": False,
            "child_pid": child_pid,
            "returncode": returncode,
            "failures": ["execution receipt path appeared before exclusive write"],
        }
    receipt_sha256 = _file_sha256(receipt_path)
    return {
        "passed": True,
        "failures": [],
        "child_pid": child_pid,
        "returncode": returncode,
        "authorization_sha256": expected_authorization_sha256,
        "execution_receipt_path": str(receipt_path),
        "execution_receipt_sha256": receipt_sha256,
        "output_path": str(output_path),
        "output_sha256": output_sha256,
        "signing_public_key_fingerprint": expected_public_key_fingerprint,
        "runtime_attestation_file_sha256": runtime_attestation_file_sha256,
        "stdout": stdout.decode(errors="replace"),
        "stderr": stderr.decode(errors="replace"),
    }


def _execution_field_failures(payload: Mapping[str, Any], label: str) -> list[str]:
    failures: list[str] = []
    weeks = payload.get("weeks")
    tapes = payload.get("tapes")
    if not isinstance(weeks, int) or not isinstance(tapes, list):
        return [f"{label}: execution scope is malformed"]
    for index, tape in enumerate(tapes):
        prefix = f"{label}: tape {index}"
        if not isinstance(tape, dict):
            failures.append(f"{prefix} is malformed")
            continue
        state_counts = tape.get("state_counts_by_week")
        proof_audit = tape.get("proof_audit")
        state_inventory = (
            proof_audit.get("state_inventory", {})
            if isinstance(proof_audit, dict)
            else {}
        )
        if not (
            tape.get("complete_horizon_enumeration") is True
            and tape.get("primary_transducer_bitwise_certified") is True
            and tape.get("mismatch_examples") == []
            and isinstance(state_counts, list)
            and len(state_counts) == weeks
            and all(isinstance(value, int) and value > 0 for value in state_counts)
            and tape.get("terminal_state_count") == state_counts[-1]
            and isinstance(tape.get("collision_examples"), list)
            and re.fullmatch(r"[0-9a-f]{64}", str(tape.get("endpoint_replay_hash", "")))
            and isinstance(proof_audit, dict)
            and proof_audit.get("unknown_callback_owner_count") == 0
            and state_inventory.get("classification_complete") is True
            and state_inventory.get("all_frozen_invariants_hold") is True
            and state_inventory.get("static_live_reads_unclassified") == []
            and proof_audit.get("markov_completeness_certificate", {}).get("passed")
            is True
            and isinstance(tape.get("policy_output_count"), int)
            and tape.get("policy_output_count") == tape.get("calendars_compared")
            and re.fullmatch(
                r"[0-9a-f]{64}", str(tape.get("policy_output_records_sha256", ""))
            )
        ):
            failures.append(f"{prefix} execution-field schema/content invalid")
    return failures


def _load_bound_execution(
    output_path: Path,
    receipt_path: Path,
    *,
    expected_output_sha256: str,
    expected_receipt_sha256: str,
    expected_authorization_sha256: str,
    expected_public_key_fingerprint: str,
    runtime_attestation_path: Path | None,
    expected_runtime_attestation_sha256: str | None,
    role: str,
    label: str,
    expected_environment_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    failures: list[str] = []
    try:
        output_path = _absolute_lexical_path(output_path)
        receipt_path = _absolute_lexical_path(receipt_path)
        output_bytes, output_stat = _read_nofollow(output_path)
        receipt_bytes, _receipt_stat = _read_nofollow(receipt_path)
        output = json.loads(output_bytes)
        custody_receipt = json.loads(receipt_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        return {}, {}, [f"{label}: custody artifact read failed: {type(exc).__name__}"]
    output_sha256 = sha256(output_bytes).hexdigest()
    receipt_sha256 = sha256(receipt_bytes).hexdigest()
    if output_sha256 != expected_output_sha256:
        failures.append(f"{label}: caller-retained output SHA-256 mismatch")
    if receipt_sha256 != expected_receipt_sha256:
        failures.append(f"{label}: caller-retained execution receipt digest mismatch")
    harness_receipt: dict[str, Any] | None = None
    exact_receipt_path = receipt_path
    receipt = custody_receipt
    if custody_receipt.get("schema_version") == "paper2_reduced_signed_harness_receipt_v1":
        harness_receipt = custody_receipt
        harness_body = dict(harness_receipt)
        claimed_harness_digest = harness_body.pop("harness_receipt_body_sha256", None)
        if claimed_harness_digest != _digest(harness_body):
            failures.append(f"{label}: signed harness receipt body digest mismatch")
        exact_receipt_path = _absolute_lexical_path(
            Path(str(harness_receipt.get("exact_execution_receipt_path", "")))
        )
        try:
            exact_receipt_bytes, _exact_receipt_stat = _read_nofollow(
                exact_receipt_path
            )
            receipt = json.loads(exact_receipt_bytes)
        except (OSError, json.JSONDecodeError) as exc:
            receipt = {}
            failures.append(
                f"{label}: exact signed receipt chain read failed: {type(exc).__name__}"
            )
        exact_receipt_sha256 = (
            sha256(exact_receipt_bytes).hexdigest() if receipt else ""
        )
        if harness_receipt.get("exact_execution_receipt_sha256") != exact_receipt_sha256:
            failures.append(f"{label}: harness-to-exact receipt digest mismatch")
        harness_required = {
            "role": role,
            "exact_authorization_sha256": expected_authorization_sha256,
            "exact_output_path": str(output_path),
            "exact_output_sha256": output_sha256,
            "runtime_attestation_path": (
                str(_absolute_lexical_path(runtime_attestation_path))
                if runtime_attestation_path is not None
                else None
            ),
            "runtime_attestation_file_sha256": (
                expected_runtime_attestation_sha256
            ),
            "fresh_child_process": True,
            "returncode": 0,
            "external_ack_received_before_child_launch": True,
            "exact_signed_receipt_verified_before_harness_receipt": True,
            "written_after_child_exit": True,
        }
        for key, value in harness_required.items():
            if harness_receipt.get(key) != value:
                failures.append(f"{label}: signed harness receipt field mismatch: {key}")
        prelaunch = harness_receipt.get("prelaunch_record")
        acknowledgement = harness_receipt.get("prelaunch_acknowledgement")
        if not isinstance(prelaunch, dict):
            prelaunch = {}
            failures.append(f"{label}: signed harness prelaunch record is missing")
        if not isinstance(acknowledgement, dict):
            acknowledgement = {}
            failures.append(f"{label}: signed harness acknowledgement is missing")
        prelaunch_body = dict(prelaunch)
        claimed_prelaunch_digest = prelaunch_body.pop(
            "prelaunch_record_sha256", None
        )
        if claimed_prelaunch_digest != _digest(prelaunch_body):
            failures.append(f"{label}: signed harness prelaunch digest mismatch")
        expected_ack = {
            "schema_version": "paper2_reduced_signed_prelaunch_ack_v1",
            "prelaunch_record_sha256": prelaunch.get("prelaunch_record_sha256"),
            "public_key_fingerprint": expected_public_key_fingerprint,
            "authorization_sha256": expected_authorization_sha256,
            "host_runtime_sha256": harness_receipt.get("host_runtime_sha256"),
            "acknowledged_before_child_launch": True,
        }
        if acknowledgement != expected_ack or harness_receipt.get(
            "prelaunch_acknowledgement_sha256"
        ) != _digest(acknowledgement):
            failures.append(f"{label}: signed harness acknowledgement mismatch")
        if not (
            prelaunch.get("public_key_fingerprint")
            == expected_public_key_fingerprint
            and prelaunch.get("authorization_sha256")
            == expected_authorization_sha256
            and prelaunch.get("host_runtime_sha256")
            == harness_receipt.get("host_runtime_sha256")
            and prelaunch.get("portable_runtime_sha256")
            == harness_receipt.get("portable_runtime_sha256")
            and prelaunch.get("harness_execution_nonce")
            == harness_receipt.get("harness_execution_nonce")
            and prelaunch.get("caller_must_retain_before_ack") is True
            and prelaunch.get("child_launch_has_not_occurred") is True
        ):
            failures.append(f"{label}: signed harness prelaunch binding mismatch")
    failures.extend(
        f"{label}: {failure}"
        for failure in verify_reduced_execution_receipt_signature(
            receipt, expected_public_key_fingerprint
        )
    )
    if receipt.get("schema_version") != REDUCED_EXECUTION_RECEIPT_SCHEMA_VERSION:
        failures.append(f"{label}: execution receipt schema mismatch")
        if receipt.get("schema_version") == "paper2_reduced_execution_receipt_v1":
            failures.append(f"{label}: legacy execution receipt v1 is rejected")
    authorization_path = _absolute_lexical_path(
        Path(str(receipt.get("authorization_path", "")))
    )
    try:
        authorization_bytes, authorization_stat = _read_nofollow(authorization_path)
        authorization = json.loads(authorization_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        authorization = {}
        failures.append(f"{label}: authorization chain read failed: {type(exc).__name__}")
    authorization_sha256 = sha256(authorization_bytes).hexdigest() if authorization else ""
    expected_custody_paths = {
        "authorization_path": str(authorization_path),
        "output_path": str(output_path),
        "receipt_path": str(exact_receipt_path),
    }
    for key, value in expected_custody_paths.items():
        if authorization.get(key) != value:
            failures.append(f"{label}: authorization custody path mismatch: {key}")
    if authorization_sha256 != expected_authorization_sha256:
        failures.append(
            f"{label}: caller-retained pre-launch authorization digest mismatch"
        )
    if authorization_sha256 != receipt.get("authorization_sha256"):
        failures.append(f"{label}: execution receipt authorization digest mismatch")
    if (
        authorization
        and receipt.get("authorization_device") != int(authorization_stat.st_dev)
    ):
        failures.append(f"{label}: execution receipt authorization device mismatch")
    if (
        authorization
        and receipt.get("authorization_inode") != int(authorization_stat.st_ino)
    ):
        failures.append(f"{label}: execution receipt authorization inode mismatch")
    failures.extend(
        f"{label}: {failure}"
        for failure in _execution_authorization_failures(
            authorization,
            authorization_sha256,
            enforce_current_interpreter=False,
            enforce_current_openssl=False,
            expected_environment_sha256=expected_environment_sha256,
        )
    )
    identity = _execution_identity_from_authorization(
        authorization, authorization_sha256
    )
    for key, value in identity.items():
        if receipt.get(key) != value:
            failures.append(f"{label}: execution receipt identity mismatch: {key}")
    if (
        receipt.get("receipt_signing_openssl_identity")
        != authorization.get("receipt_signing_openssl_identity")
    ):
        failures.append(f"{label}: signing OpenSSL identity is not authorization-bound")
    expected_argv = list(authorization.get("authorized_argv_prefix", [])) + [
        authorization_sha256
    ]
    if (
        receipt.get("materialized_argv") != expected_argv
        or receipt.get("materialized_argv_sha256") != _digest(expected_argv)
    ):
        failures.append(f"{label}: execution receipt exact argv mismatch")
    if harness_receipt is not None:
        harness_exact_required = {
            "execution_role": authorization.get("execution_role"),
            "replay_pair_id": authorization.get("replay_pair_id"),
            "parent_launcher_path": authorization.get("parent_launcher_path"),
            "parent_launcher_sha256": authorization.get("parent_launcher_sha256"),
            "isolated_bootstrap_path": authorization.get(
                "isolated_bootstrap_path"
            ),
            "isolated_bootstrap_sha256": authorization.get(
                "isolated_bootstrap_sha256"
            ),
            "scientific_child_environment": authorization.get(
                "scientific_child_environment"
            ),
            "scientific_child_environment_sha256": authorization.get(
                "scientific_child_environment_sha256"
            ),
            "harness_execution_nonce": authorization.get(
                "harness_execution_nonce"
            ),
            "host_runtime_sha256": authorization.get("host_runtime_sha256"),
            "portable_runtime_sha256": authorization.get(
                "portable_runtime_sha256"
            ),
            "exact_authorization_path": str(authorization_path),
            "exact_materialized_argv": expected_argv,
            "exact_materialized_argv_sha256": _digest(expected_argv),
            "exact_execution_identity": identity,
            "child_pid": receipt.get("child_pid"),
        }
        for key, value in harness_exact_required.items():
            if harness_receipt.get(key) != value:
                failures.append(
                    f"{label}: signed harness exact-chain mismatch: {key}"
                )
    required_receipt = {
        "fresh_child_process": True,
        "returncode": 0,
        "written_after_child_exit_and_output_validation": True,
        "role": role,
        "output_path": str(output_path),
        "output_sha256": output_sha256,
        "output_execution_identity_sha256": _digest(identity),
    }
    for key, value in required_receipt.items():
        if receipt.get(key) != value:
            failures.append(f"{label}: execution receipt field mismatch: {key}")
    if not (
        receipt.get("output_device") == int(output_stat.st_dev)
        and receipt.get("output_inode") == int(output_stat.st_ino)
        and receipt.get("output_bytes") == len(output_bytes)
    ):
        failures.append(f"{label}: execution receipt output inode/size mismatch")
    if not isinstance(receipt.get("child_pid"), int) or receipt.get("child_pid") <= 0:
        failures.append(f"{label}: execution receipt child PID is invalid")
    if (
        not isinstance(receipt.get("trusted_parent_pid"), int)
        or receipt.get("trusted_parent_pid") <= 0
        or receipt.get("trusted_parent_pid") == receipt.get("child_pid")
    ):
        failures.append(f"{label}: execution receipt trusted-parent PID is invalid")
    parent_process_fields = {
        key: receipt.get(key)
        for key in (
            "trusted_parent_pid",
            "trusted_parent_ppid",
            "trusted_parent_hostname",
            "trusted_parent_process_nonce",
            "parent_launcher_path",
            "parent_launcher_sha256",
        )
    }
    if not (
        isinstance(receipt.get("trusted_parent_ppid"), int)
        and receipt.get("trusted_parent_ppid") >= 0
        and isinstance(receipt.get("trusted_parent_hostname"), str)
        and receipt.get("trusted_parent_hostname")
        and re.fullmatch(
            r"[0-9a-f]{64}",
            str(receipt.get("trusted_parent_process_nonce", "")),
        )
        and receipt.get("trusted_parent_process_identity_sha256")
        == _digest(parent_process_fields)
    ):
        failures.append(f"{label}: trusted-parent process identity is invalid")
    if output.get("execution_identity") != identity:
        failures.append(f"{label}: output execution identity mismatch")
    for stream_name in ("stdout", "stderr"):
        stream_path_value = receipt.get(f"{stream_name}_path")
        if not isinstance(stream_path_value, str):
            failures.append(f"{label}: execution receipt {stream_name} path missing")
            continue
        stream_path = _absolute_lexical_path(Path(stream_path_value))
        expected_stream_path = exact_receipt_path.with_suffix(
            exact_receipt_path.suffix + f".{stream_name}.log"
        )
        if stream_path != expected_stream_path:
            failures.append(f"{label}: {stream_name} custody log path mismatch")
        try:
            stream_bytes, stream_stat = _read_nofollow(stream_path)
        except OSError as exc:
            failures.append(
                f"{label}: {stream_name} custody log read failed: {type(exc).__name__}"
            )
            continue
        if not (
            receipt.get(f"{stream_name}_sha256")
            == sha256(stream_bytes).hexdigest()
            and receipt.get(f"{stream_name}_bytes") == len(stream_bytes)
            and receipt.get(f"{stream_name}_device") == int(stream_stat.st_dev)
            and receipt.get(f"{stream_name}_inode") == int(stream_stat.st_ino)
        ):
            failures.append(f"{label}: {stream_name} custody log binding mismatch")
    scientific_run = authorization.get("scientific_run") is True
    if scientific_run and (
        runtime_attestation_path is None
        or expected_runtime_attestation_sha256 is None
    ):
        failures.append(f"{label}: caller-retained runtime attestation is required")
    if runtime_attestation_path is not None:
        runtime_path = _absolute_lexical_path(runtime_attestation_path)
        try:
            runtime_bytes, runtime_stat = _read_nofollow(runtime_path)
            runtime = json.loads(runtime_bytes)
        except (OSError, json.JSONDecodeError) as exc:
            runtime = {}
            failures.append(
                f"{label}: runtime attestation read failed: {type(exc).__name__}"
            )
        runtime_file_sha256 = sha256(runtime_bytes).hexdigest() if runtime else ""
        if runtime_file_sha256 != expected_runtime_attestation_sha256:
            failures.append(
                f"{label}: caller-retained runtime-attestation SHA-256 mismatch"
            )
        if receipt.get("runtime_attestation_path") != str(runtime_path):
            failures.append(f"{label}: execution receipt runtime path mismatch")
        if receipt.get("runtime_attestation_file_sha256") != runtime_file_sha256:
            failures.append(f"{label}: execution receipt runtime digest mismatch")
        if runtime and (
            receipt.get("runtime_attestation_device") != int(runtime_stat.st_dev)
            or receipt.get("runtime_attestation_inode") != int(runtime_stat.st_ino)
        ):
            failures.append(f"{label}: execution receipt runtime inode mismatch")
        runtime_body = dict(runtime)
        runtime_sha256 = runtime_body.pop("runtime_sha256", None)
        if runtime:
            try:
                from scripts.paper2_bound_execution_harness import (
                    validate_runtime_attestation_payload,
                )

                validate_runtime_attestation_payload(runtime)
            except Exception as exc:
                failures.append(
                    f"{label}: runtime attestation v2 validation failed: "
                    f"{type(exc).__name__}"
                )
        if runtime and not (
            runtime.get("schema_version")
            == "paper2_isolated_runtime_attestation_v2"
            and runtime_sha256 == _digest(runtime_body)
            and runtime_sha256 == authorization.get("host_runtime_sha256")
            and runtime.get("portable_sha256")
            == authorization.get("portable_runtime_sha256")
            and runtime.get("isolation_checks_passed") is True
        ):
            failures.append(f"{label}: runtime attestation content failed closed")
    failures.extend(_execution_field_failures(output, label))
    return output, receipt, failures


def verify_independent_reduced_execution(
    producer_path: Path,
    independent_path: Path,
    role: str,
    *,
    expected_producer_sha256: str,
    expected_independent_sha256: str,
    expected_producer_authorization_sha256: str,
    expected_independent_authorization_sha256: str,
    producer_receipt_path: Path,
    expected_producer_receipt_sha256: str,
    independent_receipt_path: Path,
    expected_independent_receipt_sha256: str,
    expected_producer_public_key_fingerprint: str,
    expected_independent_public_key_fingerprint: str,
    producer_runtime_attestation_path: Path,
    expected_producer_runtime_attestation_sha256: str,
    independent_runtime_attestation_path: Path,
    expected_independent_runtime_attestation_sha256: str,
    expected_environment_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify two separately launched, custody-pinned exact executions."""
    failures: list[str] = []
    producer_path = _absolute_lexical_path(producer_path)
    independent_path = _absolute_lexical_path(independent_path)
    if producer_path == independent_path:
        failures.append("producer and independent replay paths are identical")
    producer, producer_receipt, producer_failures = _load_bound_execution(
        producer_path,
        producer_receipt_path,
        expected_output_sha256=expected_producer_sha256,
        expected_receipt_sha256=expected_producer_receipt_sha256,
        expected_authorization_sha256=expected_producer_authorization_sha256,
        expected_public_key_fingerprint=expected_producer_public_key_fingerprint,
        runtime_attestation_path=producer_runtime_attestation_path,
        expected_runtime_attestation_sha256=(
            expected_producer_runtime_attestation_sha256
        ),
        role=role,
        label="producer",
        expected_environment_sha256=expected_environment_sha256,
    )
    independent, independent_receipt, independent_failures = _load_bound_execution(
        independent_path,
        independent_receipt_path,
        expected_output_sha256=expected_independent_sha256,
        expected_receipt_sha256=expected_independent_receipt_sha256,
        expected_authorization_sha256=expected_independent_authorization_sha256,
        expected_public_key_fingerprint=expected_independent_public_key_fingerprint,
        runtime_attestation_path=independent_runtime_attestation_path,
        expected_runtime_attestation_sha256=(
            expected_independent_runtime_attestation_sha256
        ),
        role=role,
        label="independent",
        expected_environment_sha256=expected_environment_sha256,
    )
    failures.extend(producer_failures)
    failures.extend(independent_failures)
    if (
        expected_producer_public_key_fingerprint
        == expected_independent_public_key_fingerprint
    ):
        failures.append("producer and independent signing-key fingerprints collide")
    distinct_identity_fields = (
        "authorization_id",
        "receipt_id",
        "run_id",
        "launch_nonce",
        "authorization_sha256",
    )
    for key in distinct_identity_fields:
        if producer_receipt.get(key) == independent_receipt.get(key):
            failures.append(f"producer and independent receipt identities collide: {key}")
    if expected_producer_sha256 == expected_independent_sha256:
        failures.append("producer and independent output hashes are identical")
    if expected_producer_receipt_sha256 == expected_independent_receipt_sha256:
        failures.append("producer and independent receipt digests are identical")
    if producer_receipt.get("execution_role") != "producer":
        failures.append("producer receipt does not carry producer execution_role")
    if independent_receipt.get("execution_role") != "independent_replay":
        failures.append(
            "independent receipt does not carry independent_replay execution_role"
        )
    if (
        not re.fullmatch(
            r"[0-9a-f]{64}", str(producer_receipt.get("replay_pair_id", ""))
        )
        or producer_receipt.get("replay_pair_id")
        != independent_receipt.get("replay_pair_id")
    ):
        failures.append("execution receipts do not share one valid replay_pair_id")
    scientific_scope_fields = (
        "role",
        "scientific_run",
        "weeks",
        "split",
        "workers",
        "max_calendars",
        "launch_mode",
        "cwd",
        "parent_launcher_path",
        "parent_launcher_sha256",
        "isolated_bootstrap_path",
        "isolated_bootstrap_sha256",
        "host_runtime_sha256",
        "portable_runtime_sha256",
        "scientific_child_environment_sha256",
        "source_commit",
        "runner_path",
        "runner_sha256",
        "interpreter",
        "environment_sha256",
        "contract_sha256",
        "seed_identity",
    )
    for key in scientific_scope_fields:
        if producer_receipt.get(key) != independent_receipt.get(key):
            failures.append(f"producer/independent scientific scope differs: {key}")
    if producer_receipt.get(
        "trusted_parent_process_identity_sha256"
    ) == independent_receipt.get("trusted_parent_process_identity_sha256"):
        failures.append("producer and independent executions reuse one trusted parent")
    for label, candidate in (("producer", producer), ("independent", independent)):
        failures.extend(
            f"{label}: {failure}"
            for failure in validate_reduced_certification_structure(
                candidate,
                role,
                expected_environment_sha256=expected_environment_sha256,
            )
        )
        if candidate.get("scientific_run") is not True:
            failures.append(f"{label}: artifact is not a scientific execution")
    producer_witness = _reduced_execution_witness(producer)
    independent_witness = _reduced_execution_witness(independent)
    producer_witness_sha = _digest(producer_witness)
    independent_witness_sha = _digest(independent_witness)
    if producer_witness != independent_witness:
        failures.append("independent execution witness differs from producer")
    result = {
        "schema_version": REDUCED_EXECUTION_VERIFICATION_SCHEMA_VERSION,
        "role": role,
        "producer_path": str(producer_path),
        "independent_path": str(independent_path),
        "producer_custody_sha256": expected_producer_sha256,
        "independent_custody_sha256": expected_independent_sha256,
        "producer_execution_receipt_sha256": expected_producer_receipt_sha256,
        "independent_execution_receipt_sha256": (
            expected_independent_receipt_sha256
        ),
        "producer_prelaunch_authorization_sha256": (
            expected_producer_authorization_sha256
        ),
        "independent_prelaunch_authorization_sha256": (
            expected_independent_authorization_sha256
        ),
        "producer_execution_witness_sha256": producer_witness_sha,
        "independent_execution_witness_sha256": independent_witness_sha,
        "exact_execution_witness_match": producer_witness == independent_witness,
        "execution_witness_exclusion_schema": {
            key: list(value)
            for key, value in REDUCED_EXECUTION_WITNESS_EXCLUSIONS.items()
        },
        "structural_validation_is_not_execution_verification": True,
        "receipt_verifier_openssl_identity": _openssl_identity(),
        "trust_boundary": (
            "CONDITIONAL_ON_HONEST_PINNED_PARENT_AND_CONTROL_PLANE; "
            "HOSTILE_ROOT_OR_CALLER_OUT_OF_SCOPE; NO_TEE_OR_PROOF_OF_COMPUTATION"
        ),
        "failures": failures,
        "passed": not failures,
    }
    result["verification_sha256"] = _digest(result)
    return result


def validate_reduced_certification_payload(
    payload: dict[str, Any],
    role: str,
    *,
    expected_environment_sha256: str | None = None,
) -> list[str]:
    """Fail closed: one self-consistent payload cannot prove its own execution."""
    failures = validate_reduced_certification_structure(
        payload,
        role,
        expected_environment_sha256=expected_environment_sha256,
    )
    failures.append(
        "independent custody-bound execution replay verification is required; "
        "structural payload validation is not an execution witness"
    )
    return failures


def _float_token(value: float | int | np.floating | np.integer) -> tuple[str, str]:
    return ("float", float(value).hex())


def _order_token(order: Any, *, depth: int) -> tuple[str, Any]:
    payload = {
        key: _semantic(value, depth=depth + 1) for key, value in vars(order).items()
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
            key: _semantic(value, depth=depth + 1) for key, value in vars(event).items()
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
    if isinstance(value, Path):
        return ("path", str(value))
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
    if isinstance(value, frozenset):
        items = [_semantic(item, depth=depth + 1) for item in value]
        return ("frozenset", tuple(sorted(items, key=repr)))
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


def _domain_alias_registry(sim: Any) -> dict[int, tuple[str, int]]:
    registry: dict[int, tuple[str, int]] = {}
    for order in sim.orders:
        registry[id(order)] = ("order", int(order.j))
    for index, event in enumerate(sim.risk_events):
        registry[id(event)] = ("risk_event", int(index))
    return registry


def _semantic_bound(
    value: Any,
    *,
    root_registry: dict[int, str],
    domain_alias_registry: dict[int, tuple[str, int]],
) -> Any:
    if id(value) in root_registry:
        return ("runtime_root_ref", root_registry[id(value)])
    alias = domain_alias_registry.get(id(value))
    if alias is not None:
        return ("domain_alias_ref", alias, _semantic(value))
    return _semantic(value)


def _generator_stack(
    generator: Any,
    *,
    root_registry: dict[int, str] | None = None,
    domain_alias_registry: dict[int, tuple[str, int]] | None = None,
) -> tuple[Any, ...]:
    roots = root_registry or {}
    aliases = domain_alias_registry or {}
    stack: list[Any] = []
    current = generator
    while current is not None and hasattr(current, "gi_code"):
        frame = current.gi_frame
        locals_token = []
        if frame is not None:
            for key, value in sorted(frame.f_locals.items()):
                locals_token.append(
                    (
                        key,
                        _semantic_bound(
                            value,
                            root_registry=roots,
                            domain_alias_registry=aliases,
                        ),
                    )
                )
        stack.append(
            (
                current.gi_code.co_qualname,
                sha256(current.gi_code.co_code).hexdigest(),
                None if frame is None else int(frame.f_lasti),
                bool(current.gi_running),
                bool(getattr(current, "gi_suspended", True)),
                tuple(locals_token),
            )
        )
        yielded = current.gi_yieldfrom
        current = yielded if hasattr(yielded, "gi_code") else None
    return tuple(stack)


def _callable_closure_token(
    callback: Any,
    *,
    root_registry: dict[int, str] | None = None,
    domain_alias_registry: dict[int, tuple[str, int]] | None = None,
) -> tuple[Any, ...]:
    """Serialize state carried by an unbound callback or closure."""
    roots = root_registry or {}
    aliases = domain_alias_registry or {}
    closure = getattr(callback, "__closure__", None) or ()
    defaults = getattr(callback, "__defaults__", None) or ()
    kwdefaults = getattr(callback, "__kwdefaults__", None) or {}
    return (
        getattr(callback, "__module__", type(callback).__module__),
        getattr(callback, "__qualname__", type(callback).__qualname__),
        tuple(
            _semantic_bound(
                cell.cell_contents,
                root_registry=roots,
                domain_alias_registry=aliases,
            )
            for cell in closure
        ),
        _semantic_bound(defaults, root_registry=roots, domain_alias_registry=aliases),
        _semantic_bound(kwdefaults, root_registry=roots, domain_alias_registry=aliases),
    )


def _owner_registry(sim: Any) -> dict[int, str]:
    return {id(value): name for name, value in vars(sim).items()}


def _event_callbacks(
    event: Any,
    *,
    owner_registry: dict[int, str],
    root_registry: dict[int, str] | None = None,
    domain_alias_registry: dict[int, tuple[str, int]] | None = None,
    callback_inventory: set[tuple[str, str, str]] | None = None,
) -> tuple[Any, ...]:
    roots = root_registry or {}
    aliases = domain_alias_registry or {}
    callbacks = []
    for callback in getattr(event, "callbacks", None) or ():
        process = getattr(callback, "__self__", None)
        generator = getattr(process, "_generator", None)
        if generator is not None:
            actual_fields = set(vars(process))
            unexpected = sorted(actual_fields - PROCESS_FIELD_ALLOWLIST)
            if unexpected:
                raise TypeError(
                    "unclassified fields on simpy.events.Process: " f"{unexpected}"
                )
            if getattr(process, "_target", None) is not event:
                raise TypeError(
                    "reachable Process target is not the Event carrying its resume"
                )
            process_callbacks = getattr(process, "callbacks", None) or ()
            if process_callbacks:
                raise TypeError(
                    "nested/awaited Process callbacks require recursive graph "
                    "serialization and fail closed in key-v4"
                )
            callbacks.append(
                (
                    "process_resume",
                    _generator_stack(
                        generator,
                        root_registry=roots,
                        domain_alias_registry=aliases,
                    ),
                    "target_is_current_event",
                    tuple(sorted(actual_fields)),
                    None if not hasattr(process, "_ok") else bool(process._ok),
                    (
                        "simpy_pending"
                        if getattr(process, "_value", PENDING) is PENDING
                        else _semantic(getattr(process, "_value"))
                    ),
                    bool(hasattr(process, "_defused")),
                )
            )
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
            closure_token = _callable_closure_token(
                callback,
                root_registry=roots,
                domain_alias_registry=aliases,
            )
        elif id(owner) in owner_registry:
            owner_name = owner_registry[id(owner)]
            closure_token = ()
        else:
            raise TypeError(
                "unclassified bound callback owner in Markov key: "
                f"{type(owner).__module__}.{type(owner).__qualname__}::{qualname}"
            )
        callbacks.append(
            (
                qualname,
                owner_name,
                type(owner).__qualname__ if owner is not None else "None",
                closure_token,
            )
        )
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
    root_registry: dict[int, str] | None = None,
    domain_alias_registry: dict[int, tuple[str, int]] | None = None,
    callback_inventory: set[tuple[str, str, str]] | None = None,
) -> tuple[Any, ...]:
    roots = root_registry or {}
    aliases = domain_alias_registry or {}
    event_type = f"{type(event).__module__}.{type(event).__qualname__}"
    allowed_fields = EVENT_FIELD_ALLOWLIST.get(event_type)
    if allowed_fields is None:
        raise TypeError(f"unclassified SimPy event type in Markov key: {event_type}")
    actual_fields = set(vars(event))
    unexpected_fields = sorted(actual_fields - allowed_fields)
    if unexpected_fields:
        raise TypeError(f"unclassified fields on {event_type}: {unexpected_fields}")
    amount = getattr(event, "amount", None)
    event_state = []
    for name, value in sorted(vars(event).items()):
        if name in {"env", "callbacks"}:
            continue
        if value is PENDING:
            token = ("simpy_pending",)
        elif name == "resource":
            if id(value) not in owner_registry:
                raise TypeError(
                    "unclassified SimPy resource in Markov key: "
                    f"{type(value).__module__}.{type(value).__qualname__}"
                )
            token = ("sim_owner", owner_registry[id(value)])
        elif name == "proc":
            generator = getattr(value, "_generator", None)
            if generator is None:
                raise TypeError("resource user lacks a serializable process generator")
            actual_fields = set(vars(value))
            unexpected = sorted(actual_fields - PROCESS_FIELD_ALLOWLIST)
            if unexpected:
                raise TypeError(
                    "unclassified fields on simpy.events.Process: " f"{unexpected}"
                )
            if getattr(value, "_target", None) is not event:
                raise TypeError("resource user Process target is not its request Event")
            if getattr(value, "callbacks", None):
                raise TypeError(
                    "nested/awaited resource-user Process fails closed in key-v4"
                )
            token = (
                "process",
                _generator_stack(
                    generator,
                    root_registry=roots,
                    domain_alias_registry=aliases,
                ),
                "target_is_current_event",
                tuple(sorted(actual_fields)),
            )
        else:
            token = _semantic(value)
        event_state.append((name, token))
    return (
        event_type,
        None if amount is None else _semantic(amount),
        bool(getattr(event, "triggered", False)),
        bool(getattr(event, "processed", False)),
        (
            "simpy_pending"
            if getattr(event, "_value", PENDING) is PENDING
            else _semantic(getattr(event, "_value"))
        ),
        None if not hasattr(event, "_ok") else bool(event._ok),
        bool(hasattr(event, "_defused")),
        tuple(event_state),
        _event_callbacks(
            event,
            owner_registry=owner_registry,
            root_registry=roots,
            domain_alias_registry=aliases,
            callback_inventory=callback_inventory,
        ),
    )


def _resource_token(
    resource: Any,
    *,
    owner_registry: dict[int, str],
    root_registry: dict[int, str] | None = None,
    domain_alias_registry: dict[int, tuple[str, int]] | None = None,
    callback_inventory: set[tuple[str, str, str]] | None = None,
) -> tuple[Any, ...]:
    return (
        int(resource.capacity),
        int(resource.count),
        tuple(
            _queued_event_token(
                event,
                owner_registry=owner_registry,
                root_registry=root_registry,
                domain_alias_registry=domain_alias_registry,
                callback_inventory=callback_inventory,
            )
            for event in resource.users
        ),
        tuple(
            _queued_event_token(
                event,
                owner_registry=owner_registry,
                root_registry=root_registry,
                domain_alias_registry=domain_alias_registry,
                callback_inventory=callback_inventory,
            )
            for event in resource.queue
        ),
        tuple(
            _queued_event_token(
                event,
                owner_registry=owner_registry,
                root_registry=root_registry,
                domain_alias_registry=domain_alias_registry,
                callback_inventory=callback_inventory,
            )
            for event in resource.get_queue
        ),
    )


def _container_token(
    container: Any,
    *,
    owner_registry: dict[int, str],
    root_registry: dict[int, str] | None = None,
    domain_alias_registry: dict[int, tuple[str, int]] | None = None,
    callback_inventory: set[tuple[str, str, str]] | None = None,
) -> tuple[Any, ...]:
    return (
        _float_token(container.level),
        _float_token(container.capacity),
        tuple(
            _queued_event_token(
                event,
                owner_registry=owner_registry,
                root_registry=root_registry,
                domain_alias_registry=domain_alias_registry,
                callback_inventory=callback_inventory,
            )
            for event in container.get_queue
        ),
        tuple(
            _queued_event_token(
                event,
                owner_registry=owner_registry,
                root_registry=root_registry,
                domain_alias_registry=domain_alias_registry,
                callback_inventory=callback_inventory,
            )
            for event in container.put_queue
        ),
    )


def _environment_token(
    sim: Any,
    *,
    root_registry: dict[int, str] | None = None,
    domain_alias_registry: dict[int, tuple[str, int]] | None = None,
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
                _queued_event_token(
                    event,
                    owner_registry=owners,
                    root_registry=root_registry,
                    domain_alias_registry=domain_alias_registry,
                    callback_inventory=callback_inventory,
                ),
            )
            for rank, (time, priority, _event_id, event) in enumerate(ordered)
        ),
    )


def _runtime_alias_token(sim: Any) -> tuple[Any, ...]:
    """Encode identity/alias relations among every reachable frozen runtime root.

    Individual event tokens intentionally contain no Python object addresses.
    This companion section preserves whether two deterministic root paths refer
    to the same Event, Process, or Resource object, which can affect future
    release/callback behavior even when their value fields are identical.
    """
    event_paths: dict[int, list[str]] = {}
    process_paths: dict[int, list[str]] = {}
    resource_paths: dict[int, list[str]] = {}

    def visit_event(path: str, event: Any) -> None:
        event_paths.setdefault(id(event), []).append(path)
        process = getattr(event, "proc", None)
        if process is not None:
            process_paths.setdefault(id(process), []).append(f"{path}.proc")
        resource = getattr(event, "resource", None)
        if resource is not None:
            resource_paths.setdefault(id(resource), []).append(f"{path}.resource")
        for index, callback in enumerate(getattr(event, "callbacks", None) or ()):
            owner = getattr(callback, "__self__", None)
            if getattr(owner, "_generator", None) is not None:
                process_paths.setdefault(id(owner), []).append(
                    f"{path}.callbacks[{index}].process"
                )

    ordered = sorted(sim.env._queue, key=lambda row: (row[0], row[1], row[2]))
    for rank, (_time, _priority, _event_id, event) in enumerate(ordered):
        visit_event(f"env.queue[{rank}]", event)
    visit_event("sim._contract_renewed_event", sim._contract_renewed_event)
    for name in CONTAINER_FIELDS:
        container = getattr(sim, name)
        for index, event in enumerate(container.get_queue):
            visit_event(f"container.{name}.get_queue[{index}]", event)
        for index, event in enumerate(container.put_queue):
            visit_event(f"container.{name}.put_queue[{index}]", event)
    for name, resource in (
        ("op10_convoy", sim.op10_convoy),
        ("op12_convoy", sim.op12_convoy),
    ):
        resource_paths.setdefault(id(resource), []).append(f"resource.{name}")
        for queue_name in ("users", "put_queue", "get_queue"):
            for index, event in enumerate(getattr(resource, queue_name)):
                visit_event(f"resource.{name}.{queue_name}[{index}]", event)

    def classes(kind: str, mapping: dict[int, list[str]]) -> tuple[Any, ...]:
        # Object ids only group paths inside this replay; they are never emitted.
        grouped = [tuple(sorted(paths)) for paths in mapping.values()]
        return (kind, tuple(sorted(grouped)))

    return (
        classes("event_alias_classes", event_paths),
        classes("process_alias_classes", process_paths),
        classes("resource_alias_classes", resource_paths),
    )


def _all_order_history(sim: Any) -> tuple[Any, ...]:
    """Conservatively retain completed, lost, and live orders in list order."""
    return tuple(
        _order_token(order, depth=0)
        for order in sim.orders
    )


def _all_risk_event_history(sim: Any) -> tuple[Any, ...]:
    """Conservatively retain every realized risk event in exact list order."""
    return tuple(
        _risk_token(event, depth=0)
        for event in sim.risk_events
    )


def semantic_markov_payload(
    sim: Any,
    controller: Any,
    *,
    callback_inventory: set[tuple[str, str, str]] | None = None,
) -> dict[str, Any]:
    """Return the complete serialized future state for the frozen lane."""
    runtime_audit = _runtime_markov_completeness_audit(sim, controller)
    owners = _owner_registry(sim)
    roots = {
        id(sim): "sim",
        id(sim.env): "sim.env",
        id(controller): "controller",
        id(controller.tape): "controller.tape",
        id(controller.profile): "controller.profile",
    }
    domain_aliases = _domain_alias_registry(sim)
    return {
        "schema": KEY_SCHEMA_VERSION,
        "markov_completeness": {
            "certificate_sha256": runtime_audit[
                "markov_completeness_certificate_sha256"
            ],
            "runtime_schema_sha256": runtime_audit["runtime_schema_sha256"],
            "runtime_audit_sha256": runtime_audit["audit_sha256"],
            "tape_binding": runtime_audit["tape_binding"],
        },
        "sim_immutable_contract": runtime_audit["immutable_sim_state"],
        "environment": _environment_token(
            sim,
            root_registry=roots,
            domain_alias_registry=domain_aliases,
            callback_inventory=callback_inventory,
        ),
        "runtime_graph_aliases": _runtime_alias_token(sim),
        "containers": tuple(
            (
                name,
                _container_token(
                    getattr(sim, name),
                    owner_registry=owners,
                    root_registry=roots,
                    domain_alias_registry=domain_aliases,
                    callback_inventory=callback_inventory,
                ),
            )
            for name in CONTAINER_FIELDS
        ),
        "sim_state": tuple(
            (name, _semantic(getattr(sim, name))) for name in SIM_STATE_FIELDS
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
            owner_registry=owners,
            root_registry=roots,
            domain_alias_registry=domain_aliases,
            callback_inventory=callback_inventory,
        ),
        "resources": tuple(
            (
                name,
                _resource_token(
                    resource,
                    owner_registry=owners,
                    root_registry=roots,
                    domain_alias_registry=domain_aliases,
                    callback_inventory=callback_inventory,
                ),
            )
            for name, resource in (
                ("op10_convoy", sim.op10_convoy),
                ("op12_convoy", sim.op12_convoy),
            )
        ),
        "all_order_history": _all_order_history(sim),
        "pending_queue_order": tuple(int(order.j) for order in sim.pending_backorders),
        "all_risk_event_history": _all_risk_event_history(sim),
        "controller": {
            "active_action": tuple(map(int, controller.active_action)),
            "pending_action": tuple(map(int, controller.pending_action)),
            "current_week": int(controller.current_week),
            "last_switch_week": int(controller.last_switch_week),
            "condition": _float_token(controller.condition),
        },
    }


def semantic_markov_fingerprint(
    sim: Any,
    controller: Any,
    *,
    callback_inventory: set[tuple[str, str, str]] | None = None,
) -> tuple[str, str, int, bytes]:
    payload = semantic_markov_payload(
        sim,
        controller,
        callback_inventory=callback_inventory,
    )
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return (
        sha256(encoded).hexdigest(),
        sha512(encoded).hexdigest(),
        len(encoded),
        encoded,
    )


def semantic_markov_key(
    sim: Any,
    controller: Any,
    *,
    callback_inventory: set[tuple[str, str, str]] | None = None,
) -> str:
    """Hash the complete v3 future-state serialization."""
    return semantic_markov_fingerprint(
        sim,
        controller,
        callback_inventory=callback_inventory,
    )[0]


def runtime_proof_audit(tape: dict[str, Any]) -> dict[str, Any]:
    """Inspect one live post-decision state without opening any new tape."""
    sim, controller, start = make_sim(tape)
    controller.activate_week(0)
    controller.request(ACTIONS[0])
    advance_including(sim, start + 168.0)
    callbacks: set[tuple[str, str, str]] = set()
    payload = semantic_markov_payload(sim, controller, callback_inventory=callbacks)
    inventory = audit_frozen_state_inventory(sim)
    return {
        "state_inventory": inventory,
        "callback_inventory": [
            {"kind": kind, "callable": callable_name, "owner": owner}
            for kind, callable_name, owner in sorted(callbacks)
        ],
        "unknown_callback_owner_count": 0,
        "markov_completeness": payload["markov_completeness"],
        "markov_completeness_certificate": markov_completeness_certificate(),
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
    treatment_orders = _treatment_orders(sim, start)
    visible = compute_order_level_ret_excel_request_snapshot_ledger(
        treatment_orders, current_time=sim.env.now
    )
    ret_by_j = {int(row["j"]): float(row["ret"]) for row in visible["ret_rows"]}
    completed = sorted(
        (
            order
            for order in treatment_orders
            if getattr(order, "OATj", None) is not None
            and int(order.j) in ret_by_j
        ),
        key=lambda order: (float(order.OATj), int(order.j)),
    )
    ids = tuple(int(order.j) for order in completed)
    values = tuple(ret_by_j[order_id] for order_id in ids)
    if set(ids) != set(ret_by_j):
        raise RuntimeError("request-snapshot visible population/order mapping drifted")
    primary = float(np.mean(values)) if values else 1.0
    panel = _endpoint_panel(sim, start, controller)
    selected = {key: panel.get(key) for key in ENDPOINT_KEYS}
    return Checkpoint(
        visible_values=values,
        visible_order_ids=ids,
        primary_hex=primary.hex(),
        endpoint_digest=_digest(selected),
    )


@dataclass(frozen=True)
class PrefixResult:
    key: str
    payload_sha512: str
    payload_bytes: int
    canonical_state_bytes: bytes
    checkpoint: Checkpoint
    callback_inventory: tuple[tuple[str, str, str], ...]
    markov_completeness_certificate_sha256: str = ""
    runtime_schema_sha256: str = ""
    tape_binding_sha256: str = ""


def _prefix_result_from_state(sim: Any, controller: Any, start: float) -> PrefixResult:
    callbacks: set[tuple[str, str, str]] = set()
    key, payload_sha512, payload_bytes, canonical_state_bytes = (
        semantic_markov_fingerprint(sim, controller, callback_inventory=callbacks)
    )
    markov_binding = json.loads(canonical_state_bytes)["markov_completeness"]
    return PrefixResult(
        key=key,
        payload_sha512=payload_sha512,
        payload_bytes=payload_bytes,
        canonical_state_bytes=canonical_state_bytes,
        checkpoint=checkpoint(sim, start, controller),
        callback_inventory=tuple(sorted(callbacks)),
        markov_completeness_certificate_sha256=markov_binding["certificate_sha256"],
        runtime_schema_sha256=markov_binding["runtime_schema_sha256"],
        tape_binding_sha256=markov_binding["tape_binding"]["binding_sha256"],
    )


def run_prefix(
    tape: dict[str, Any],
    sequence: Sequence[int],
    _binding_guard: Any = _assert_loaded_proof_bindings,
) -> PrefixResult:
    """Replay one active-action prefix using the contract's one-week request lag."""
    _binding_guard()
    if not sequence or int(sequence[0]) != 0:
        raise ValueError("active prefix must start with M")
    sim, controller, start = make_sim(tape)
    for week, _action in enumerate(sequence):
        controller.activate_week(week)
        requested = sequence[week + 1] if week + 1 < len(sequence) else sequence[week]
        controller.request(ACTIONS[int(requested)])
        advance_including(sim, start + (week + 1) * 168.0)
    return _prefix_result_from_state(sim, controller, start)


def run_prefix_boundary_extension(
    tape: dict[str, Any],
    prefix: Sequence[int],
    action: int,
    _binding_guard: Any = _assert_loaded_proof_bindings,
) -> PrefixResult:
    """Execute an appended action by requesting it at the stored boundary.

    This is intentionally independent of ``run_prefix(prefix + action)``.  It
    first executes the stored prefix with its terminal self-request, then issues
    the next request at the boundary before activation.  Equality of the two
    executions is the executable request-lag proof obligation.
    """
    _binding_guard()
    if not prefix or int(prefix[0]) != 0:
        raise ValueError("active prefix must start with M")
    action = int(action)
    last_action = int(prefix[-1])
    switched_previous = len(prefix) > 1 and last_action != int(prefix[-2])
    if action not in _feasible_next(last_action, switched_previous):
        raise ValueError("boundary extension violates the dwell rule")
    sim, controller, start = make_sim(tape)
    for week, _active in enumerate(prefix):
        controller.activate_week(week)
        requested = prefix[week + 1] if week + 1 < len(prefix) else prefix[week]
        controller.request(ACTIONS[int(requested)])
        advance_including(sim, start + (week + 1) * 168.0)
    controller.request(ACTIONS[action])
    next_week = len(prefix)
    controller.activate_week(next_week)
    controller.request(ACTIONS[action])
    advance_including(sim, start + (next_week + 1) * 168.0)
    return _prefix_result_from_state(sim, controller, start)


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

    def visit(
        prefix: tuple[int, ...], switched_previous: bool
    ) -> Iterator[tuple[int, ...]]:
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
    represented_prefix_count: int = 1
    payload_sha512: str = ""
    payload_bytes: int = 0
    canonical_state_bytes: bytes = b""
    markov_completeness_certificate_sha256: str = ""
    runtime_schema_sha256: str = ""
    tape_binding_sha256: str = ""


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
    collision_bisimulation: dict[str, Any] | None = None
    request_lag_equivalence_records: tuple[dict[str, Any], ...] = ()

    def predict_visible_ledger(
        self, sequence: Sequence[int]
    ) -> tuple[tuple[float, ...], tuple[int, ...]]:
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


@dataclass(frozen=True)
class CollisionWitness:
    representative: tuple[int, ...]
    alternative: tuple[int, ...]
    representative_checkpoint: Checkpoint
    alternative_checkpoint: Checkpoint
    payload_sha512: str
    payload_bytes: int
    representative_state_id: int
    canonical_bytes_equal_at_merge: bool
    discarded_transition_id: str = ""
    discarded_parent_state_id: int = -1
    discarded_action: int = -1
    represented_prefix_count: int = 0
    markov_completeness_certificate_sha256: str = ""
    runtime_schema_sha256: str = ""
    tape_binding_sha256: str = ""


def _appended_after(
    earlier: Checkpoint,
    later: Checkpoint,
) -> tuple[tuple[float, ...], tuple[int, ...]]:
    values = earlier.visible_values
    order_ids = earlier.visible_order_ids
    if later.visible_values[: len(values)] != values:
        raise AssertionError("completed visible ReT rows changed after their OAT")
    if later.visible_order_ids[: len(order_ids)] != order_ids:
        raise AssertionError("completed visible row order changed after its OAT")
    return (
        later.visible_values[len(values) :],
        later.visible_order_ids[len(order_ids) :],
    )


def audit_collision_bisimulation(
    tape: dict[str, Any],
    witnesses: Sequence[CollisionWitness],
    *,
    weeks: int,
    layers: Sequence[Sequence[StateNode]],
    transitions: Sequence[dict[tuple[int, int], Transition]],
    request_lag_records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Build a complete backward-inductive bisimulation certificate.

    Every quotient node is an obligation whose outgoing action edges point to
    already-complete obligations in the next layer.  Every discarded collision
    prefix is then replayed byte-for-byte and all of its next actions must land
    on those same child obligations with identical incremental primary labels.
    This closes delayed divergence through the terminal layer rather than
    treating a one-step comparison as a proof.
    """
    mismatches: list[dict[str, Any]] = []
    transition_checks = 0
    payload_checks = 0
    node_obligations: list[dict[str, Any]] = []
    node_complete: dict[tuple[int, int], bool] = {}
    completeness_certificate = markov_completeness_certificate()
    completeness_failures = validate_markov_completeness_certificate(
        completeness_certificate
    )
    expected_completeness_sha = completeness_certificate.get("certificate_sha256")
    node_completeness_shas = {
        node.markov_completeness_certificate_sha256
        for layer in layers
        for node in layer
    }
    runtime_schema_shas = {
        node.runtime_schema_sha256 for layer in layers for node in layer
    }
    tape_binding_shas = {node.tape_binding_sha256 for layer in layers for node in layer}
    if completeness_failures:
        mismatches.extend(
            {"reason": "markov_completeness_invalid", "detail": failure}
            for failure in completeness_failures
        )
    if node_completeness_shas != {expected_completeness_sha}:
        mismatches.append({"reason": "per_key_markov_completeness_binding_drift"})
    if len(runtime_schema_shas) != 1 or not next(iter(runtime_schema_shas), ""):
        mismatches.append({"reason": "per_key_runtime_schema_binding_drift"})
    if len(tape_binding_shas) != 1 or not next(iter(tape_binding_shas), ""):
        mismatches.append({"reason": "per_key_tape_binding_drift"})

    if len(layers) != weeks or len(transitions) != max(0, weeks - 1):
        mismatches.append({"reason": "quotient_graph_horizon_mismatch"})

    layer_multiplicity_records: list[dict[str, Any]] = []
    discarded_transition_count = 0
    for layer_index, layer in enumerate(layers):
        represented = sum(int(node.represented_prefix_count) for node in layer)
        expected = feasible_calendar_count(layer_index + 1)
        outgoing_quotient_transitions = (
            0
            if layer_index == weeks - 1
            else sum(
                len(_feasible_next(node.last_action, node.switched_previous))
                for node in layer
            )
        )
        represented_successors = (
            0
            if layer_index == weeks - 1
            else sum(
                int(node.represented_prefix_count)
                * len(_feasible_next(node.last_action, node.switched_previous))
                for node in layer
            )
        )
        kept_successor_states = (
            0 if layer_index == weeks - 1 else len(layers[layer_index + 1])
        )
        discarded = (
            0
            if layer_index == weeks - 1
            else outgoing_quotient_transitions - kept_successor_states
        )
        discarded_transition_count += discarded
        complete = bool(
            represented == expected
            and all(int(node.represented_prefix_count) > 0 for node in layer)
            and (
                layer_index == weeks - 1
                or represented_successors == feasible_calendar_count(layer_index + 2)
            )
            and discarded >= 0
        )
        if not complete:
            mismatches.append(
                {
                    "reason": "quotient_multiplicity_mismatch",
                    "week": layer_index + 1,
                    "represented": represented,
                    "expected": expected,
                }
            )
        layer_multiplicity_records.append(
            {
                "week": layer_index + 1,
                "node_count": len(layer),
                "represented_prefix_count": represented,
                "closed_form_prefix_count": expected,
                "outgoing_quotient_transition_count": outgoing_quotient_transitions,
                "represented_successor_count": represented_successors,
                "kept_successor_state_count": kept_successor_states,
                "discarded_transition_count": discarded,
                "status": "COMPLETE" if complete else "INCOMPLETE",
            }
        )

    expected_lag_obligations = {
        f"request-lag:w{layer_index + 2}:p{node.state_id}:a{int(action)}": (
            layer_index + 1,
            int(node.state_id),
            int(action),
        )
        for layer_index, layer in enumerate(layers[:-1])
        for node in layer
        for action in _feasible_next(node.last_action, node.switched_previous)
    }
    lag_obligation_ids = [
        row.get("obligation_id")
        for row in request_lag_records
        if isinstance(row, dict)
    ]
    request_lag_complete = bool(
        len(lag_obligation_ids) == len(set(lag_obligation_ids))
        and set(lag_obligation_ids) == set(expected_lag_obligations)
        and all(
            row.get("status") == "COMPLETE"
            and row.get("state_bytes_equal") is True
            and row.get("checkpoint_equal") is True
            and row.get("callback_inventory_equal") is True
            and row.get("markov_binding_equal") is True
            and (
                row.get("parent_week"),
                row.get("parent_state_id"),
                row.get("action"),
            )
            == expected_lag_obligations.get(row.get("obligation_id"))
            for row in request_lag_records
        )
    )
    if not request_lag_complete:
        mismatches.append({"reason": "request_lag_equivalence_incomplete"})

    # Terminal-to-root worklist.  Because every edge increases the layer by one,
    # reverse layer order is a complete topological proof, not a sampled walk.
    for layer_index in reversed(range(len(layers))):
        for node in layers[layer_index]:
            obligation_id = f"node:w{layer_index + 1}:n{node.state_id}"
            expected_actions = (
                ()
                if layer_index == weeks - 1
                else _feasible_next(node.last_action, node.switched_previous)
            )
            edges: list[dict[str, Any]] = []
            complete = True
            actual_actions = (
                ()
                if layer_index == weeks - 1
                else tuple(
                    action
                    for state_id, action in transitions[layer_index]
                    if state_id == node.state_id
                )
            )
            if actual_actions != expected_actions:
                complete = False
                mismatches.append(
                    {
                        "reason": "representative_action_set_mismatch",
                        "obligation_id": obligation_id,
                        "expected_actions": list(expected_actions),
                        "actual_actions": list(actual_actions),
                    }
                )
            for action in expected_actions:
                transition = transitions[layer_index].get((node.state_id, action))
                if transition is None:
                    complete = False
                    mismatches.append(
                        {
                            "reason": "representative_transition_missing",
                            "obligation_id": obligation_id,
                            "action": int(action),
                        }
                    )
                    continue
                child_key = (layer_index + 1, transition.next_state_id)
                child_complete = node_complete.get(child_key, False)
                complete = complete and child_complete
                edges.append(
                    {
                        "action": int(action),
                        "represented_prefix_count": int(
                            node.represented_prefix_count
                        ),
                        "incremental_label_sha256": _digest(
                            (
                                transition.appended_visible_values,
                                transition.appended_visible_order_ids,
                            )
                        ),
                        "child_obligation_id": (
                            f"node:w{layer_index + 2}:n{transition.next_state_id}"
                        ),
                        "child_complete": child_complete,
                    }
                )
            if layer_index == weeks - 1:
                complete = True
            node_complete[(layer_index, node.state_id)] = complete
            node_obligations.append(
                {
                    "obligation_id": obligation_id,
                    "week": layer_index + 1,
                    "state_id": node.state_id,
                    "state_sha256": node.key,
                    "state_sha512": node.payload_sha512,
                    "state_bytes": node.payload_bytes,
                    "last_action": int(node.last_action),
                    "switched_previous": bool(node.switched_previous),
                    "represented_prefix_count": int(
                        node.represented_prefix_count
                    ),
                    "expected_actions": [int(action) for action in expected_actions],
                    "edges": edges,
                    "status": "COMPLETE" if complete else "INCOMPLETE",
                }
            )

    collision_roots: list[dict[str, Any]] = []
    for collision_id, witness in enumerate(witnesses):
        payload_checks += 1
        root_id = f"collision:{collision_id:08d}"
        if len(witness.representative) != len(witness.alternative):
            mismatches.append(
                {"reason": "collision_horizon_mismatch", "root_id": root_id}
            )
            continue
        layer_index = len(witness.representative) - 1
        rep_last = witness.representative[-1]
        alt_last = witness.alternative[-1]
        rep_switched = (
            len(witness.representative) > 1 and rep_last != witness.representative[-2]
        )
        alt_switched = (
            len(witness.alternative) > 1 and alt_last != witness.alternative[-2]
        )
        if (
            rep_last != alt_last
            or rep_switched != alt_switched
            or not witness.payload_sha512
            or witness.payload_bytes <= 0
            or not witness.canonical_bytes_equal_at_merge
        ):
            mismatches.append(
                {"reason": "collision_control_state_mismatch", "root_id": root_id}
            )
            continue

        representative_root = run_prefix(tape, witness.representative)
        alternative_root = run_prefix(tape, witness.alternative)
        root_markov_binding_equal = bool(
            representative_root.markov_completeness_certificate_sha256
            == alternative_root.markov_completeness_certificate_sha256
            == expected_completeness_sha
            == witness.markov_completeness_certificate_sha256
            and representative_root.runtime_schema_sha256
            == alternative_root.runtime_schema_sha256
            == witness.runtime_schema_sha256
            == next(iter(runtime_schema_shas), None)
            and representative_root.tape_binding_sha256
            == alternative_root.tape_binding_sha256
            == witness.tape_binding_sha256
            == next(iter(tape_binding_shas), None)
        )
        root_bytes_equal = (
            representative_root.canonical_state_bytes
            == alternative_root.canonical_state_bytes
        )
        root_callbacks_equal = (
            representative_root.callback_inventory
            == alternative_root.callback_inventory
        )
        if (
            not root_bytes_equal
            or not root_callbacks_equal
            or not root_markov_binding_equal
        ):
            mismatches.append(
                {
                    "reason": "collision_root_state_not_byte_equal",
                    "root_id": root_id,
                    "representative_sha256": representative_root.key,
                    "alternative_sha256": alternative_root.key,
                    "markov_binding_equal": root_markov_binding_equal,
                }
            )

        expected_actions = (
            ()
            if len(witness.representative) >= weeks
            else _feasible_next(rep_last, rep_switched)
        )
        edges: list[dict[str, Any]] = []
        root_complete = (
            root_bytes_equal and root_callbacks_equal and root_markov_binding_equal
        )
        for action in expected_actions:
            transition_checks += 1
            left = run_prefix(tape, witness.representative + (action,))
            right = run_prefix(tape, witness.alternative + (action,))
            left_label = _appended_after(
                witness.representative_checkpoint,
                left.checkpoint,
            )
            right_label = _appended_after(
                witness.alternative_checkpoint,
                right.checkpoint,
            )
            graph_transition = transitions[layer_index].get(
                (witness.representative_state_id, int(action))
            )
            if graph_transition is None:
                mismatches.append(
                    {
                        "reason": "collision_representative_edge_missing",
                        "root_id": root_id,
                        "action": int(action),
                    }
                )
                root_complete = False
                continue
            target = layers[layer_index + 1][graph_transition.next_state_id]
            target_replay = (
                left
                if target.representative == witness.representative + (action,)
                else run_prefix(tape, target.representative)
            )
            child_obligation_key = (layer_index + 1, target.state_id)
            child_complete = node_complete.get(child_obligation_key, False)
            byte_equal = (
                left.canonical_state_bytes == right.canonical_state_bytes
                and left.canonical_state_bytes == target_replay.canonical_state_bytes
                and target_replay.key == target.key
                and target_replay.payload_sha512 == target.payload_sha512
                and target_replay.payload_bytes == target.payload_bytes
            )
            label_equal = left_label == right_label and left_label == (
                graph_transition.appended_visible_values,
                graph_transition.appended_visible_order_ids,
            )
            callbacks_equal = left.callback_inventory == right.callback_inventory
            markov_binding_equal = bool(
                left.markov_completeness_certificate_sha256
                == right.markov_completeness_certificate_sha256
                == expected_completeness_sha
                and left.runtime_schema_sha256
                == right.runtime_schema_sha256
                == next(iter(runtime_schema_shas), None)
                and left.tape_binding_sha256
                == right.tape_binding_sha256
                == next(iter(tape_binding_shas), None)
            )
            edge_complete = (
                byte_equal
                and label_equal
                and callbacks_equal
                and markov_binding_equal
                and child_complete
            )
            root_complete = root_complete and edge_complete
            row = {
                "root_id": root_id,
                "week": len(witness.representative),
                "action": int(action),
                "state_bytes_equal": byte_equal,
                "incremental_labels_bitwise_equal": label_equal,
                "callback_inventory_equal": callbacks_equal,
                "markov_binding_equal": markov_binding_equal,
                "child_obligation_id": (f"node:w{layer_index + 2}:n{target.state_id}"),
                "child_obligation_complete": child_complete,
                "status": "COMPLETE" if edge_complete else "INCOMPLETE",
            }
            edges.append(row)
            if not edge_complete and len(mismatches) < 20:
                mismatches.append(
                    {
                        "reason": "collision_successor_obligation_incomplete",
                        **row,
                        "alternative_next_key": right.key,
                        "representative_next_key": left.key,
                    }
                )
        collision_roots.append(
            {
                "collision_id": collision_id,
                "root_id": root_id,
                "discarded_transition_id": witness.discarded_transition_id,
                "discarded_parent_state_id": witness.discarded_parent_state_id,
                "discarded_action": witness.discarded_action,
                "represented_prefix_count": witness.represented_prefix_count,
                "week": len(witness.representative),
                "representative_state_id": witness.representative_state_id,
                "last_action": int(rep_last),
                "switched_previous": bool(rep_switched),
                "representative": calendar_name(witness.representative),
                "alternative": calendar_name(witness.alternative),
                "canonical_bytes_equal": root_bytes_equal,
                "callback_inventory_equal": root_callbacks_equal,
                "markov_binding_equal": root_markov_binding_equal,
                "expected_actions": [int(action) for action in expected_actions],
                "edges": edges,
                "status": "COMPLETE" if root_complete else "INCOMPLETE",
            }
        )

    all_nodes_complete = len(node_complete) == sum(
        len(layer) for layer in layers
    ) and all(node_complete.values())
    all_roots_complete = len(collision_roots) == len(witnesses) and all(
        root["status"] == "COMPLETE" for root in collision_roots
    )
    discarded_ids = [
        witness.discarded_transition_id for witness in witnesses
    ]
    discarded_witness_bijection = bool(
        discarded_transition_count == len(witnesses)
        and len(discarded_ids) == len(set(discarded_ids))
        and all(discarded_ids)
    )
    if not discarded_witness_bijection:
        mismatches.append({"reason": "discarded_transition_witness_bijection_failed"})
    passed = (
        payload_checks == len(witnesses)
        and all_nodes_complete
        and all_roots_complete
        and discarded_witness_bijection
        and request_lag_complete
        and not mismatches
    )
    certificate_body = {
        "schema_version": "paper2_collision_bisimulation_v2",
        "key_schema_version": KEY_SCHEMA_VERSION,
        "complete_state_serialization": True,
        "event_payload_serialized": True,
        "resource_users_serialized": True,
        "callback_closure_state_serialized": True,
        "process_target_state_serialized_or_fail_closed": True,
        "runtime_alias_graph_serialized": True,
        "markov_completeness_certificate": completeness_certificate,
        "markov_completeness_certificate_sha256": expected_completeness_sha,
        "markov_completeness_validated": not completeness_failures,
        "per_key_runtime_schema_enforced": True,
        "runtime_schema_sha256": next(iter(runtime_schema_shas), None),
        "tape_binding_sha256": next(iter(tape_binding_shas), None),
        "deterministic_transition_semantics_bound": True,
        "control_state_bound_into_obligations": True,
        "exact_feasible_action_sets_validated": True,
        "quotient_multiplicity_validated": True,
        "request_lag_equivalence_validated": request_lag_complete,
        "request_lag_equivalence_check_count": len(request_lag_records),
        "request_lag_equivalence_records": list(request_lag_records),
        "request_lag_equivalence_records_sha256": _digest(request_lag_records),
        "layer_multiplicity_records": layer_multiplicity_records,
        "layer_multiplicity_records_sha256": _digest(layer_multiplicity_records),
        "terminal_represented_prefix_count": sum(
            int(node.represented_prefix_count) for node in layers[-1]
        ),
        "terminal_closed_form_prefix_count": feasible_calendar_count(weeks),
        "w24_terminal_prefix_target": 11_184_811,
        "discarded_transition_count": discarded_transition_count,
        "discarded_transition_witness_count": len(witnesses),
        "discarded_transition_witness_bijection": discarded_witness_bijection,
        "discarded_transition_ids_sha256": _digest(discarded_ids),
        "collision_payload_checks": payload_checks,
        "collision_root_count": len(collision_roots),
        "transition_congruence_checks": transition_checks,
        "node_obligation_count": len(node_obligations),
        "terminal_node_obligation_count": sum(len(layer) for layer in layers[-1:]),
        "unresolved_node_obligation_count": sum(
            status is not True for status in node_complete.values()
        ),
        "unresolved_collision_root_count": sum(
            root["status"] != "COMPLETE" for root in collision_roots
        ),
        "all_actions_covered": all_nodes_complete and all_roots_complete,
        "backward_induction_complete": all_nodes_complete and all_roots_complete,
        "node_obligations": node_obligations,
        "collision_roots": collision_roots,
        "mismatch_examples": mismatches,
        "induction_rule": (
            "Each collision successor is byte-equal to a quotient child whose "
            "complete action set points only to complete later-layer obligations; "
            "the terminal layer is the base case."
        ),
        "passed": passed,
    }
    certificate_body["node_obligation_records_sha256"] = _digest(node_obligations)
    certificate_body["collision_root_records_sha256"] = _digest(collision_roots)
    certificate_body["transition_record_sha256"] = _digest(
        {
            "nodes": certificate_body["node_obligation_records_sha256"],
            "roots": certificate_body["collision_root_records_sha256"],
        }
    )
    certificate_body["certificate_sha256"] = _digest(certificate_body)
    return certificate_body


def validate_collision_bisimulation_certificate(
    certificate: dict[str, Any],
    *,
    expected_collision_count: int,
    weeks: int,
    _binding_guard: Any = _assert_loaded_proof_bindings,
) -> list[str]:
    """Re-verify coverage, digests and terminal-to-root obligation closure."""
    _binding_guard(full=True)
    failures: list[str] = []
    if not isinstance(certificate, dict):
        return ["collision certificate is not an object"]
    body = dict(certificate)
    claimed_digest = body.pop("certificate_sha256", None)
    if claimed_digest != _digest(body):
        failures.append("collision certificate digest mismatch")
    nodes = certificate.get("node_obligations")
    roots = certificate.get("collision_roots")
    if not isinstance(nodes, list) or not isinstance(roots, list):
        return failures + ["collision certificate obligation records missing"]
    if certificate.get("node_obligation_records_sha256") != _digest(nodes):
        failures.append("node-obligation record digest mismatch")
    if certificate.get("collision_root_records_sha256") != _digest(roots):
        failures.append("collision-root record digest mismatch")
    lag_records = certificate.get("request_lag_equivalence_records")
    multiplicity_records = certificate.get("layer_multiplicity_records")
    if not isinstance(lag_records, list):
        failures.append("request-lag equivalence records missing")
        lag_records = []
    if not isinstance(multiplicity_records, list):
        failures.append("layer multiplicity records missing")
        multiplicity_records = []
    if certificate.get("request_lag_equivalence_records_sha256") != _digest(
        lag_records
    ):
        failures.append("request-lag equivalence record digest mismatch")
    if certificate.get("layer_multiplicity_records_sha256") != _digest(
        multiplicity_records
    ):
        failures.append("layer multiplicity record digest mismatch")
    completeness_certificate = certificate.get("markov_completeness_certificate")
    completeness_failures = validate_markov_completeness_certificate(
        completeness_certificate
    )
    failures.extend(completeness_failures)
    if certificate.get("markov_completeness_certificate_sha256") != (
        completeness_certificate.get("certificate_sha256")
        if isinstance(completeness_certificate, dict)
        else None
    ):
        failures.append("collision certificate Markov-completeness binding mismatch")
    for key in ("runtime_schema_sha256", "tape_binding_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(certificate.get(key, ""))):
            failures.append(f"collision certificate {key} is missing")
    node_by_id = {
        row.get("obligation_id"): row
        for row in nodes
        if isinstance(row, dict) and isinstance(row.get("obligation_id"), str)
    }
    if len(node_by_id) != len(nodes):
        failures.append("node-obligation identifiers are missing or duplicated")
    if len(roots) != expected_collision_count:
        failures.append("collision-root coverage count mismatch")
    root_ids = {
        row.get("root_id")
        for row in roots
        if isinstance(row, dict) and isinstance(row.get("root_id"), str)
    }
    if len(root_ids) != len(roots):
        failures.append("collision-root identifiers are missing or duplicated")

    incoming_multiplicity: dict[str, int] = {}
    expected_lag_obligations: dict[str, tuple[int, int, int]] = {}

    for row in nodes:
        if not isinstance(row, dict):
            failures.append("malformed node obligation")
            continue
        week = row.get("week")
        actions = row.get("expected_actions")
        edges = row.get("edges")
        edge_actions = (
            [edge.get("action") for edge in edges if isinstance(edge, dict)]
            if isinstance(edges, list)
            else []
        )
        last_action = row.get("last_action")
        switched_previous = row.get("switched_previous")
        represented_prefix_count = row.get("represented_prefix_count")
        exact_actions = (
            []
            if week == weeks
            else (
                list(_feasible_next(last_action, switched_previous))
                if isinstance(last_action, int)
                and not isinstance(last_action, bool)
                and last_action in (0, 1, 2)
                and isinstance(switched_previous, bool)
                else None
            )
        )
        if (
            row.get("status") != "COMPLETE"
            or not isinstance(week, int)
            or not isinstance(actions, list)
            or not isinstance(edges, list)
            or len(edge_actions) != len(edges)
            or any(not isinstance(action, int) for action in actions + edge_actions)
            or len(actions) != len(set(actions))
            or len(edge_actions) != len(set(edge_actions))
            or exact_actions is None
            or actions != exact_actions
            or edge_actions != exact_actions
            or not isinstance(represented_prefix_count, int)
            or isinstance(represented_prefix_count, bool)
            or represented_prefix_count <= 0
        ):
            failures.append(f"incomplete node obligation: {row.get('obligation_id')}")
            continue
        if week == weeks and actions:
            failures.append(f"terminal node has actions: {row.get('obligation_id')}")
        if week < weeks and not actions:
            failures.append(
                f"nonterminal node has no actions: {row.get('obligation_id')}"
            )
        for edge in edges:
            child_id = edge.get("child_obligation_id")
            child = node_by_id.get(child_id)
            if (
                edge.get("child_complete") is not True
                or child is None
                or child.get("week") != week + 1
                or child.get("status") != "COMPLETE"
                or edge.get("represented_prefix_count")
                != represented_prefix_count
            ):
                failures.append(
                    f"dangling/incomplete child obligation: {row.get('obligation_id')}"
                )
            elif isinstance(child_id, str):
                incoming_multiplicity[child_id] = (
                    incoming_multiplicity.get(child_id, 0)
                    + represented_prefix_count
                )
            if week < weeks:
                lag_id = (
                    f"request-lag:w{week + 1}:p{row.get('state_id')}"
                    f":a{edge.get('action')}"
                )
                expected_lag_obligations[lag_id] = (
                    week,
                    row.get("state_id"),
                    edge.get("action"),
                )

    for row in nodes:
        if not isinstance(row, dict) or not isinstance(row.get("week"), int):
            continue
        obligation_id = row.get("obligation_id")
        represented = row.get("represented_prefix_count")
        if row["week"] == 1:
            if represented != 1 or row.get("last_action") != 0 or row.get(
                "switched_previous"
            ) is not False:
                failures.append("root quotient control/multiplicity is invalid")
        elif incoming_multiplicity.get(obligation_id) != represented:
            failures.append(
                f"incoming multiplicity mismatch: {obligation_id}"
            )

    for root in roots:
        if not isinstance(root, dict):
            failures.append("malformed collision root")
            continue
        actions = root.get("expected_actions")
        edges = root.get("edges")
        edge_actions = (
            [edge.get("action") for edge in edges if isinstance(edge, dict)]
            if isinstance(edges, list)
            else []
        )
        last_action = root.get("last_action")
        switched_previous = root.get("switched_previous")
        root_week = root.get("week")
        exact_actions = (
            []
            if root_week == weeks
            else (
                list(_feasible_next(last_action, switched_previous))
                if isinstance(last_action, int)
                and not isinstance(last_action, bool)
                and last_action in (0, 1, 2)
                and isinstance(switched_previous, bool)
                else None
            )
        )
        if (
            root.get("status") != "COMPLETE"
            or root.get("canonical_bytes_equal") is not True
            or root.get("callback_inventory_equal") is not True
            or root.get("markov_binding_equal") is not True
            or not isinstance(actions, list)
            or not isinstance(edges, list)
            or len(edge_actions) != len(edges)
            or any(not isinstance(action, int) for action in actions + edge_actions)
            or len(actions) != len(set(actions))
            or len(edge_actions) != len(set(edge_actions))
            or exact_actions is None
            or actions != exact_actions
            or edge_actions != exact_actions
        ):
            failures.append(f"incomplete collision root: {root.get('root_id')}")
            continue
        for edge in edges:
            child = node_by_id.get(edge.get("child_obligation_id"))
            if not (
                edge.get("status") == "COMPLETE"
                and edge.get("state_bytes_equal") is True
                and edge.get("incremental_labels_bitwise_equal") is True
                and edge.get("callback_inventory_equal") is True
                and edge.get("markov_binding_equal") is True
                and edge.get("child_obligation_complete") is True
                and child is not None
                and child.get("status") == "COMPLETE"
                and child.get("week") == root.get("week") + 1
            ):
                failures.append(f"incomplete collision edge: {root.get('root_id')}")

        parent_id = (
            f"node:w{root_week - 1}:n{root.get('discarded_parent_state_id')}"
            if isinstance(root_week, int) and root_week > 1
            else None
        )
        parent = node_by_id.get(parent_id)
        child = node_by_id.get(
            f"node:w{root_week}:n{root.get('representative_state_id')}"
            if isinstance(root_week, int)
            else None
        )
        discarded_action = root.get("discarded_action")
        if (
            parent is None
            or child is None
            or discarded_action
            not in _feasible_next(
                parent.get("last_action"), parent.get("switched_previous")
            )
            or discarded_action != last_action
            or child.get("last_action") != last_action
            or child.get("switched_previous") != switched_previous
            or root.get("represented_prefix_count")
            != parent.get("represented_prefix_count")
        ):
            failures.append(
                f"collision discarded-transition binding invalid: {root.get('root_id')}"
            )

    lag_ids = [
        row.get("obligation_id") for row in lag_records if isinstance(row, dict)
    ]
    if (
        len(lag_ids) != len(lag_records)
        or len(lag_ids) != len(set(lag_ids))
        or set(lag_ids) != set(expected_lag_obligations)
        or any(
            row.get("status") != "COMPLETE"
            or row.get("state_bytes_equal") is not True
            or row.get("checkpoint_equal") is not True
            or row.get("callback_inventory_equal") is not True
            or row.get("markov_binding_equal") is not True
            or (
                row.get("parent_week"),
                row.get("parent_state_id"),
                row.get("action"),
            )
            != expected_lag_obligations.get(row.get("obligation_id"))
            for row in lag_records
            if isinstance(row, dict)
        )
    ):
        failures.append("request-lag equivalence obligations are incomplete")

    nodes_by_week: dict[int, list[dict[str, Any]]] = {}
    for row in nodes:
        if isinstance(row, dict) and isinstance(row.get("week"), int):
            nodes_by_week.setdefault(row["week"], []).append(row)
    if len(multiplicity_records) != weeks:
        failures.append("layer multiplicity record count mismatch")
    for week in range(1, weeks + 1):
        layer_nodes = nodes_by_week.get(week, [])
        represented = sum(
            int(row.get("represented_prefix_count", 0)) for row in layer_nodes
        )
        record = next(
            (
                row
                for row in multiplicity_records
                if isinstance(row, dict) and row.get("week") == week
            ),
            None,
        )
        outgoing = sum(len(row.get("expected_actions", [])) for row in layer_nodes)
        kept = len(nodes_by_week.get(week + 1, [])) if week < weeks else 0
        discarded = outgoing - kept if week < weeks else 0
        represented_successors = (
            sum(
                int(row.get("represented_prefix_count", 0))
                * len(row.get("expected_actions", []))
                for row in layer_nodes
            )
            if week < weeks
            else 0
        )
        if (
            record is None
            or record.get("status") != "COMPLETE"
            or record.get("node_count") != len(layer_nodes)
            or record.get("represented_prefix_count") != represented
            or record.get("closed_form_prefix_count")
            != feasible_calendar_count(week)
            or represented != feasible_calendar_count(week)
            or record.get("outgoing_quotient_transition_count") != outgoing
            or record.get("represented_successor_count") != represented_successors
            or record.get("kept_successor_state_count") != kept
            or record.get("discarded_transition_count") != discarded
        ):
            failures.append(f"layer multiplicity invalid at week {week}")

    discarded_ids = [
        root.get("discarded_transition_id")
        for root in roots
        if isinstance(root, dict)
    ]
    discarded_count = sum(
        int(row.get("discarded_transition_count", 0))
        for row in multiplicity_records
        if isinstance(row, dict)
    )
    if (
        len(discarded_ids) != len(set(discarded_ids))
        or any(not isinstance(value, str) or not value for value in discarded_ids)
        or certificate.get("discarded_transition_count") != discarded_count
        or certificate.get("discarded_transition_witness_count") != len(roots)
        or discarded_count != len(roots)
        or certificate.get("discarded_transition_ids_sha256")
        != _digest(discarded_ids)
    ):
        failures.append("discarded-transition witness bijection invalid")
    terminal_represented = sum(
        int(row.get("represented_prefix_count", 0))
        for row in nodes_by_week.get(weeks, [])
    )
    if (
        certificate.get("terminal_represented_prefix_count")
        != terminal_represented
        or certificate.get("terminal_closed_form_prefix_count")
        != feasible_calendar_count(weeks)
        or terminal_represented != feasible_calendar_count(weeks)
        or certificate.get("w24_terminal_prefix_target") != 11_184_811
    ):
        failures.append("terminal quotient multiplicity is invalid")

    required_true = (
        "passed",
        "complete_state_serialization",
        "event_payload_serialized",
        "resource_users_serialized",
        "callback_closure_state_serialized",
        "process_target_state_serialized_or_fail_closed",
        "runtime_alias_graph_serialized",
        "markov_completeness_validated",
        "per_key_runtime_schema_enforced",
        "deterministic_transition_semantics_bound",
        "control_state_bound_into_obligations",
        "exact_feasible_action_sets_validated",
        "quotient_multiplicity_validated",
        "request_lag_equivalence_validated",
        "discarded_transition_witness_bijection",
        "all_actions_covered",
        "backward_induction_complete",
    )
    if certificate.get("schema_version") != "paper2_collision_bisimulation_v2":
        failures.append("collision certificate schema mismatch")
    if certificate.get("key_schema_version") != KEY_SCHEMA_VERSION:
        failures.append("collision certificate key schema mismatch")
    for key in required_true:
        if certificate.get(key) is not True:
            failures.append(f"collision certificate field is not true: {key}")
    if certificate.get("collision_payload_checks") != expected_collision_count:
        failures.append("collision payload-check count mismatch")
    if certificate.get("collision_root_count") != expected_collision_count:
        failures.append("collision root-count field mismatch")
    if certificate.get("request_lag_equivalence_check_count") != len(lag_records):
        failures.append("request-lag equivalence count mismatch")
    if certificate.get("node_obligation_count") != len(nodes):
        failures.append("node obligation-count field mismatch")
    if certificate.get("unresolved_node_obligation_count") != 0:
        failures.append("unresolved node obligations remain")
    if certificate.get("unresolved_collision_root_count") != 0:
        failures.append("unresolved collision roots remain")
    if certificate.get("mismatch_examples"):
        failures.append("collision certificate contains mismatches")
    return failures


def build_transducer(
    tape: dict[str, Any],
    weeks: int,
    _binding_guard: Any = _assert_loaded_proof_bindings,
) -> Transducer:
    _binding_guard(full=True)
    initial_prefix = (0,)
    initial_result = run_prefix(tape, initial_prefix)
    expected_markov_binding = (
        initial_result.markov_completeness_certificate_sha256,
        initial_result.runtime_schema_sha256,
        initial_result.tape_binding_sha256,
    )
    if not all(expected_markov_binding):
        raise RuntimeError("initial semantic key lacks Markov-completeness binding")

    def assert_markov_binding(result: PrefixResult, sequence: Sequence[int]) -> None:
        actual = (
            result.markov_completeness_certificate_sha256,
            result.runtime_schema_sha256,
            result.tape_binding_sha256,
        )
        if actual != expected_markov_binding:
            raise RuntimeError(
                "per-key Markov-completeness binding drifted at prefix "
                + calendar_name(sequence)
            )

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
    layers = [
        [
            StateNode(
                state_id=0,
                key=initial_result.key,
                payload_sha512=initial_result.payload_sha512,
                payload_bytes=initial_result.payload_bytes,
                canonical_state_bytes=initial_result.canonical_state_bytes,
                representative=initial_prefix,
                checkpoint=initial_result.checkpoint,
                last_action=0,
                switched_previous=False,
                represented_prefix_count=1,
                markov_completeness_certificate_sha256=(
                    initial_result.markov_completeness_certificate_sha256
                ),
                runtime_schema_sha256=initial_result.runtime_schema_sha256,
                tape_binding_sha256=initial_result.tape_binding_sha256,
            )
        ]
    ]
    transitions: list[dict[tuple[int, int], Transition]] = []
    collisions: list[dict[str, Any]] = []
    collision_witnesses: list[CollisionWitness] = []
    request_lag_records: list[dict[str, Any]] = []
    prefix_replays = 1

    for _next_week in range(1, weeks):
        previous_layer = layers[-1]
        next_nodes: list[StateNode] = []
        index: dict[tuple[int, bool, bytes], int] = {}
        layer_transitions: dict[tuple[int, int], Transition] = {}
        current_layer_callbacks: set[tuple[str, str, str]] = set()
        current_layer_evaluations = 0
        current_layer_callback_digest = sha256()
        current_layer_prefixes_with_callbacks = 0

        for parent in previous_layer:
            for action in _feasible_next(parent.last_action, parent.switched_previous):
                sequence = parent.representative + (int(action),)
                result = run_prefix(tape, sequence)
                assert_markov_binding(result, sequence)
                boundary_result = run_prefix_boundary_extension(
                    tape,
                    parent.representative,
                    int(action),
                )
                assert_markov_binding(boundary_result, sequence)
                lag_state_equal = bool(
                    result.canonical_state_bytes
                    == boundary_result.canonical_state_bytes
                    and result.key == boundary_result.key
                    and result.payload_sha512 == boundary_result.payload_sha512
                    and result.payload_bytes == boundary_result.payload_bytes
                )
                lag_checkpoint_equal = result.checkpoint == boundary_result.checkpoint
                lag_callbacks_equal = (
                    result.callback_inventory == boundary_result.callback_inventory
                )
                lag_binding_equal = bool(
                    result.markov_completeness_certificate_sha256
                    == boundary_result.markov_completeness_certificate_sha256
                    and result.runtime_schema_sha256
                    == boundary_result.runtime_schema_sha256
                    and result.tape_binding_sha256
                    == boundary_result.tape_binding_sha256
                )
                lag_complete = bool(
                    lag_state_equal
                    and lag_checkpoint_equal
                    and lag_callbacks_equal
                    and lag_binding_equal
                )
                lag_record = {
                    "obligation_id": (
                        f"request-lag:w{len(sequence)}:p{parent.state_id}:a{int(action)}"
                    ),
                    "parent_week": len(parent.representative),
                    "parent_state_id": int(parent.state_id),
                    "action": int(action),
                    "rewrite_state_sha256": result.key,
                    "boundary_state_sha256": boundary_result.key,
                    "state_bytes_equal": lag_state_equal,
                    "checkpoint_equal": lag_checkpoint_equal,
                    "callback_inventory_equal": lag_callbacks_equal,
                    "markov_binding_equal": lag_binding_equal,
                    "status": "COMPLETE" if lag_complete else "INCOMPLETE",
                }
                request_lag_records.append(lag_record)
                if not lag_complete:
                    raise AssertionError(
                        "request-lag boundary execution differs from replay rewrite: "
                        + lag_record["obligation_id"]
                    )
                for replay_kind, replay_result in (
                    ("rewrite", result),
                    ("boundary", boundary_result),
                ):
                    callback_inventory.update(replay_result.callback_inventory)
                    current_layer_callbacks.update(replay_result.callback_inventory)
                    callback_record = {
                        "replay_kind": replay_kind,
                        "prefix": list(sequence),
                        "callback_inventory": [
                            list(row) for row in replay_result.callback_inventory
                        ],
                    }
                    record_digest = bytes.fromhex(_digest(callback_record))
                    callback_record_digest.update(record_digest)
                    current_layer_callback_digest.update(record_digest)
                    if replay_result.callback_inventory:
                        prefixes_with_callbacks += 1
                        current_layer_prefixes_with_callbacks += 1
                prefix_replays += 2
                current_layer_evaluations += 2
                parent_values = parent.checkpoint.visible_values
                parent_ids = parent.checkpoint.visible_order_ids
                if (
                    result.checkpoint.visible_values[: len(parent_values)]
                    != parent_values
                ):
                    raise AssertionError(
                        "completed visible ReT rows changed after their OAT"
                    )
                if result.checkpoint.visible_order_ids[: len(parent_ids)] != parent_ids:
                    raise AssertionError(
                        "completed visible row order changed after its OAT"
                    )
                appended_values = result.checkpoint.visible_values[len(parent_values) :]
                appended_ids = result.checkpoint.visible_order_ids[len(parent_ids) :]
                switched = int(action) != parent.last_action
                state_key = (
                    int(action),
                    switched,
                    result.canonical_state_bytes,
                )
                if state_key not in index:
                    state_id = len(next_nodes)
                    index[state_key] = state_id
                    next_nodes.append(
                        StateNode(
                            state_id=state_id,
                            key=result.key,
                            payload_sha512=result.payload_sha512,
                            payload_bytes=result.payload_bytes,
                            canonical_state_bytes=result.canonical_state_bytes,
                            representative=sequence,
                            checkpoint=result.checkpoint,
                            last_action=int(action),
                            switched_previous=switched,
                            represented_prefix_count=0,
                            markov_completeness_certificate_sha256=(
                                result.markov_completeness_certificate_sha256
                            ),
                            runtime_schema_sha256=result.runtime_schema_sha256,
                            tape_binding_sha256=result.tape_binding_sha256,
                        )
                    )
                else:
                    state_id = index[state_key]
                    representative = next_nodes[state_id]
                    if not (
                        representative.payload_sha512 == result.payload_sha512
                        and representative.payload_bytes == result.payload_bytes
                        and representative.canonical_state_bytes
                        == result.canonical_state_bytes
                    ):
                        raise AssertionError(
                            "byte-complete state mismatch on proposed collision"
                        )
                    collision_witnesses.append(
                        CollisionWitness(
                            representative=representative.representative,
                            alternative=sequence,
                            representative_checkpoint=representative.checkpoint,
                            alternative_checkpoint=result.checkpoint,
                            payload_sha512=result.payload_sha512,
                            payload_bytes=result.payload_bytes,
                            representative_state_id=state_id,
                            canonical_bytes_equal_at_merge=(
                                representative.canonical_state_bytes
                                == result.canonical_state_bytes
                            ),
                            discarded_transition_id=(
                                f"discarded:w{len(sequence)}:p{parent.state_id}"
                                f":a{int(action)}"
                            ),
                            discarded_parent_state_id=int(parent.state_id),
                            discarded_action=int(action),
                            represented_prefix_count=(
                                parent.represented_prefix_count
                            ),
                            markov_completeness_certificate_sha256=(
                                result.markov_completeness_certificate_sha256
                            ),
                            runtime_schema_sha256=result.runtime_schema_sha256,
                            tape_binding_sha256=result.tape_binding_sha256,
                        )
                    )
                    collisions.append(
                        {
                            "week": len(sequence),
                            "representative": calendar_name(
                                representative.representative
                            ),
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
                next_nodes[state_id].represented_prefix_count += (
                    parent.represented_prefix_count
                )
                layer_transitions[(parent.state_id, int(action))] = Transition(
                    next_state_id=state_id,
                    appended_visible_values=appended_values,
                    appended_visible_order_ids=appended_ids,
                )

        layers.append(next_nodes)
        transitions.append(layer_transitions)
        # Exact byte comparisons for this layer are complete.  Retain hashes and
        # representative prefixes; collision audit replays target representatives
        # when needed, avoiding O(nodes * state-bytes) resident memory at W24.
        for node in previous_layer:
            node.canonical_state_bytes = b""
        layer_callback_inventory.append(current_layer_callbacks)
        layer_semantic_key_evaluations.append(current_layer_evaluations)
        layer_callback_record_digests.append(current_layer_callback_digest)
        layer_prefixes_with_callbacks.append(current_layer_prefixes_with_callbacks)

    collision_bisimulation = audit_collision_bisimulation(
        tape,
        collision_witnesses,
        weeks=weeks,
        layers=layers,
        transitions=transitions,
        request_lag_records=request_lag_records,
    )
    collision_failures = validate_collision_bisimulation_certificate(
        collision_bisimulation,
        expected_collision_count=len(collisions),
        weeks=weeks,
    )
    if collision_failures:
        raise RuntimeError(
            "collision bisimulation failed closed: " + "; ".join(collision_failures)
        )
    for layer in layers:
        for node in layer:
            node.canonical_state_bytes = b""
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
        collision_bisimulation=collision_bisimulation,
        request_lag_equivalence_records=tuple(request_lag_records),
    )


_WORKER_TAPE: dict[str, Any] | None = None


def _worker_init(tape: dict[str, Any]) -> None:
    global _WORKER_TAPE
    _WORKER_TAPE = tape


def _brute_calendar(
    sequence: tuple[int, ...],
    _binding_guard: Any = _assert_loaded_proof_bindings,
) -> tuple[str, tuple[float, ...], tuple[int, ...], str, str]:
    _binding_guard()
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
    _binding_guard: Any = _assert_loaded_proof_bindings,
) -> dict[str, Any]:
    _binding_guard(full=True)
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
        actual = pool.imap(
            _brute_calendar,
            calendars,
            chunksize=max(1, len(calendars) // (workers * 32)),
        )

    mismatch_examples: list[dict[str, Any]] = []
    compared = 0
    endpoint_hashes: dict[str, str] = {}
    policy_output_records: list[dict[str, Any]] = []
    try:
        for predicted, brute in zip(expected, actual):
            compared += 1
            name, predicted_values, predicted_ids, predicted_primary = predicted
            brute_name, brute_values, brute_ids, brute_primary, endpoint_digest = brute
            endpoint_hashes[brute_name] = endpoint_digest
            policy_output_records.append(
                {
                    "calendar": brute_name,
                    "visible_values_sha256": _digest(brute_values),
                    "visible_order_ids_sha256": _digest(brute_ids),
                    "primary_hex": brute_primary,
                    "endpoint_digest": endpoint_digest,
                }
            )
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
                        "value_digest_equal": _digest(predicted_values)
                        == _digest(brute_values),
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
        "policy_output_count": len(policy_output_records),
        "policy_output_records_sha256": _digest(policy_output_records),
        "collision_bisimulation": transducer.collision_bisimulation,
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
                and sum(transducer.layer_prefixes_with_nonempty_callback_inventory)
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
        default=ROOT
        / "results"
        / "paper2_bottleneck"
        / "exact_transducer_certification.json",
    )
    parser.add_argument("--progress", type=Path)
    parser.add_argument("--execution-authorization", type=Path)
    parser.add_argument("--execution-receipt", type=Path)
    parser.add_argument("--execution-authorization-sha256")
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
            "scientific certification requires clean tracked dependencies:\n" + drift
        )
    seeds = args.seed or [
        (1_110_001, CONTEXTS[0]),
        (1_100_001, CONTEXTS[0]),
        (1_100_031, CONTEXTS[0]),
    ]
    execution_identity = None
    execution_options = (
        args.execution_authorization,
        args.execution_receipt,
        args.execution_authorization_sha256,
    )
    if any(value is not None for value in execution_options):
        if any(value is None for value in execution_options):
            parser.error("all execution authorization/receipt options are required")
        try:
            authorization_bytes, _authorization_stat = _read_nofollow(
                _absolute_lexical_path(args.execution_authorization)
            )
            authorization = json.loads(authorization_bytes)
        except (OSError, json.JSONDecodeError) as exc:
            parser.error(f"execution authorization cannot be read: {exc}")
        authorization_sha256 = sha256(authorization_bytes).hexdigest()
        authorization_failures = []
        if authorization_sha256 != args.execution_authorization_sha256:
            authorization_failures.append(
                "caller-supplied execution authorization digest mismatch"
            )
        authorization_failures.extend(
            _execution_authorization_failures(
                authorization,
                args.execution_authorization_sha256,
                actual_argv=sys.orig_argv,
                enforce_current_openssl=False,
            )
        )
        if authorization.get("output_path") != str(
            _absolute_lexical_path(args.output)
        ):
            authorization_failures.append("authorized output path mismatch")
        if authorization.get("receipt_path") != str(
            _absolute_lexical_path(args.execution_receipt)
        ):
            authorization_failures.append("authorized receipt path mismatch")
        if authorization.get("scientific_run") is not (not args.non_scientific_smoke):
            authorization_failures.append("authorized scientific-run mode mismatch")
        if authorization.get("launch_mode") == "isolated_bootstrap":
            expected_live_environment = {
                **authorization["scientific_child_environment"],
                "SCRES_EXECUTION_NONCE": authorization["harness_execution_nonce"],
                "SCRES_EXECUTION_ROLE": authorization["execution_role"],
                "SCRES_HOST_RUNTIME_SHA256": authorization["host_runtime_sha256"],
                "SCRES_PORTABLE_RUNTIME_SHA256": authorization[
                    "portable_runtime_sha256"
                ],
            }
            authorization_failures.extend(
                _live_scientific_environment_failures(
                    dict(os.environ), expected_live_environment
                )
            )
        actual_seed_identity = _execution_seed_identity(
            seeds, weeks=args.weeks, split=args.split
        )
        if authorization.get("seed_identity") != actual_seed_identity:
            authorization_failures.append("authorized seed/tape scope mismatch")
        if authorization_failures:
            parser.error(
                "execution authorization failed closed:\n"
                + "\n".join(authorization_failures)
            )
        execution_identity = _execution_identity_from_authorization(
            authorization, authorization_sha256
        )

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
            else (
                "REDUCED_HORIZON_PRIMARY_CERTIFIED_FULL_CONTRACT_FAIL_CLOSED"
                if all_primary
                else "MARKOV_KEY_CERTIFICATION_FAILED_CLOSED"
            )
        ),
        "contract_id": contract["contract_id"],
        "contract_sha256": contract_hash,
        "key_schema_version": KEY_SCHEMA_VERSION,
        "execution_identity": execution_identity,
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
    output_text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if execution_identity is None:
        args.output.write_text(output_text)
    else:
        try:
            descriptor = os.open(
                args.output,
                os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                output_stat = os.fstat(descriptor)
                if (
                    output_stat.st_dev != authorization["output_initial_device"]
                    or output_stat.st_ino != authorization["output_initial_inode"]
                    or output_stat.st_size != 0
                ):
                    parser.error("authorized execution output inode is not reserved-empty")
                encoded_output = output_text.encode()
                offset = 0
                while offset < len(encoded_output):
                    written = os.write(descriptor, encoded_output[offset:])
                    if written <= 0:
                        raise OSError("short write to reserved authorized output")
                    offset += written
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError as exc:
            parser.error(f"authorized execution output write failed: {exc}")
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


_freeze_loaded_proof_bindings()


if __name__ == "__main__":
    raise SystemExit(main())
