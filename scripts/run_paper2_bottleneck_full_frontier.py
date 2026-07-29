#!/usr/bin/env python3
"""Exact full-calendar frontier evaluator for the frozen Paper-2 M/T/R lane.

The expensive DES state reduction lives in
``run_paper2_bottleneck_exact_transducer.py``.  This module consumes those
transducers without modifying the certification harness.  It enumerates the
stable 0-based calendar index in bounded batches, never materializes the full
calendar family, and uses a two-pass interval screen:

1. outward-rounded intervals bound the canonical floating ReT for every
   calendar and tape;
2. every calendar whose upper bound can attain the best lower bound is
   rescored by an unaccelerated canonical prefix replay;
3. the emitted binary floats are compared as exact fractions.

Consequently a reported maximum is exact for the emitted canonical metric,
not merely the winner of a rounded table lookup.  Any missing acceleration
certificate, incomplete enumeration, contender overflow, or replay mismatch
fails closed.  The CLI is an implementation harness; it does not launch the
60-calibration/119-locked execution unless invoked explicitly.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from functools import lru_cache
from hashlib import sha256
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import platform
import re
import subprocess
import sys
import time
from typing import Any, Callable, Iterator, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_paper2_bottleneck_exact_transducer import (  # noqa: E402
    KEY_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
    Transducer,
    build_transducer,
    certification_environment,
    feasible_calendar_count,
    feasible_calendars,
    run_prefix,
    runtime_proof_audit,
    validate_collision_bisimulation_certificate,
    validate_reduced_certification_structure,
)
from scripts.paper2_bound_execution_harness import (  # noqa: E402
    ISOLATED_BOOTSTRAP,
    HarnessError,
    _confined_checksum_path,
    reverify_authorized_reduced_pair_archive,
)
from supply_chain.paper2_bottleneck import (  # noqa: E402
    ACTIONS,
    ACTION_NAMES,
    CONTEXTS,
    materialize_tape,
    run_policy,
)


ROOT = Path(__file__).resolve().parent.parent
PRIMARY_CONTRACT_PATH = (
    ROOT / "contracts" / "paper2_bottleneck_primary_bound_v2.json"
)
TRANSDUCER_RUNNER_PATH = (
    ROOT / "scripts" / "run_paper2_bottleneck_exact_transducer.py"
)
CHECKPOINT_DEPENDENCY_PATHS = (
    ROOT / "contracts" / "paper2_bottleneck_full_horizon_bound_v1.json",
    PRIMARY_CONTRACT_PATH,
    TRANSDUCER_RUNNER_PATH,
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
SCHEMA_VERSION = "paper2_bottleneck_full_frontier_v2"
MANIFEST_SCHEMA_VERSION = "paper2_bottleneck_full_frontier_manifest_v2"
AUTHORIZATION_SCHEMA_VERSION = "paper2_bound_execution_authorization_v9"
W24_AUDIT_SCHEMA_VERSION = "paper2_bottleneck_w24_profile_state_audit_v3"
CALENDAR_INDEX_SCHEMA_VERSION = "mtr_no_adjacent_switch_lexicographic_v1"
NEG_INF = float("-inf")
POS_INF = float("inf")
UNIT_ROUNDOFF = 2.0**-53

REPLAY_KEYS = (
    "ret_excel",
    "ret_excel_visible",
    "ret_excel_visible_n",
    "ration_ret_excel",
    "ret_excel_cvar05",
    "ret_excel_cvar10",
    "service_loss_auc_ration_hours",
    "n_lost",
    "lost_orders",
    "backorder_qty_final",
    "backlog_age_max",
    "fill_rate",
    "fill_rate_on_time",
    "mass_residual",
    "reserve_units_issued",
    "reserve_units_replenished",
    "reserve_inventory_initial",
    "reserve_inventory_terminal",
    "reserve_committed_pending_terminal",
    "reserve_in_transit_terminal",
    "reserve_capacity",
    "reserve_target_terminal",
    "reserve_replenishment_lead_time",
    "reserve_issue_delay",
    "reserve_replenishment_requests",
    "reserve_stock_balance_residual",
    "token_hours_m",
    "token_hours_t",
    "token_hours_r",
    "total_token_hours",
    "consumed_base_threat_sha256",
    "realized_demand_sha256",
)


def _json_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _fraction_payload(value: Fraction) -> dict[str, Any]:
    return {
        "numerator": str(value.numerator),
        "denominator": str(value.denominator),
        "float": float(value),
        "float_hex": float(value).hex(),
        "numerator_bits": value.numerator.bit_length(),
        "denominator_bits": value.denominator.bit_length(),
    }


def _action_names(sequence: Sequence[int]) -> str:
    return "".join(ACTION_NAMES[ACTIONS[int(action)]] for action in sequence)


@lru_cache(maxsize=None)
def _suffix_count(remaining: int, last_action: int, switched_previous: bool) -> int:
    """Number of suffixes from one automaton state, including all choices."""
    if remaining == 0:
        return 1
    choices = (last_action,) if switched_previous else (0, 1, 2)
    return sum(
        _suffix_count(remaining - 1, action, action != last_action)
        for action in choices
    )


def calendar_at_index(index: int, weeks: int) -> tuple[int, ...]:
    """Unrank the stable lexicographic DFS calendar index without enumeration."""
    total = feasible_calendar_count(weeks)
    if index < 0 or index >= total:
        raise IndexError(f"calendar index {index} outside [0, {total})")
    remaining_index = int(index)
    sequence = [0]
    last = 0
    switched_previous = False
    for position in range(1, weeks):
        choices = (last,) if switched_previous else (0, 1, 2)
        remaining = weeks - position - 1
        for action in choices:
            block = _suffix_count(remaining, action, action != last)
            if remaining_index < block:
                switched_previous = action != last
                last = action
                sequence.append(action)
                break
            remaining_index -= block
        else:  # pragma: no cover - arithmetic invariant
            raise AssertionError("calendar unranking exhausted every branch")
    return tuple(sequence)


def calendar_index(sequence: Sequence[int]) -> int:
    """Rank a feasible calendar in the same order as ``feasible_calendars``."""
    weeks = len(sequence)
    if weeks < 1 or int(sequence[0]) != 0:
        raise ValueError("calendar must be non-empty and start with M")
    rank = 0
    last = 0
    switched_previous = False
    for position in range(1, weeks):
        selected = int(sequence[position])
        choices = (last,) if switched_previous else (0, 1, 2)
        if selected not in choices:
            raise ValueError("calendar violates the no-adjacent-switch rule")
        remaining = weeks - position - 1
        for action in choices:
            if action == selected:
                break
            rank += _suffix_count(remaining, action, action != last)
        switched_previous = selected != last
        last = selected
    return rank


def iter_calendar_batches(weeks: int, batch_size: int) -> Iterator[tuple[int, np.ndarray]]:
    """Yield bounded action matrices; never retain the complete tuple family."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    matrix = np.empty((batch_size, weeks), dtype=np.int8)
    start_index = 0
    used = 0
    for index, sequence in enumerate(feasible_calendars(weeks)):
        if used == 0:
            start_index = index
        matrix[used, :] = sequence
        used += 1
        if used == batch_size:
            yield start_index, matrix.copy()
            used = 0
    if used:
        yield start_index, matrix[:used].copy()


def _exact_binary_sum(values: Sequence[float]) -> Fraction:
    return sum((Fraction.from_float(float(value)) for value in values), Fraction(0))


def _fraction_interval(value: Fraction) -> tuple[float, float]:
    """Tight adjacent-float enclosure of an exact rational."""
    rounded = float(value)
    rounded_exact = Fraction.from_float(rounded)
    if rounded_exact == value:
        return rounded, rounded
    if rounded_exact < value:
        return rounded, float(np.nextafter(rounded, POS_INF))
    return float(np.nextafter(rounded, NEG_INF)), rounded


def _values_label(values: Sequence[float]) -> tuple[float, float, float, int, Fraction]:
    exact = _exact_binary_sum(values)
    lower, upper = _fraction_interval(exact)
    abs_exact = sum(
        (abs(Fraction.from_float(float(value))) for value in values),
        Fraction(0),
    )
    _abs_lower, abs_upper = _fraction_interval(abs_exact)
    return lower, upper, abs_upper, len(values), exact


@dataclass(frozen=True)
class ExactEdge:
    total: Fraction
    count: int


@dataclass
class ScoreTransducer:
    """Compact score-only projection of a certified semantic transducer."""

    weeks: int
    seed: int
    tape_sha256: str
    initial_lower: float
    initial_upper: float
    initial_abs_upper: float
    initial_edge: ExactEdge
    next_state: list[np.ndarray]
    delta_lower: list[np.ndarray]
    delta_upper: list[np.ndarray]
    delta_abs_upper: list[np.ndarray]
    delta_count: list[np.ndarray]
    exact_edges: list[list[ExactEdge | None]]
    state_counts: list[int]
    prefix_replays: int
    collision_count: int
    table_sha256: str
    callback_inventory: tuple[tuple[str, str, str], ...] = ()
    semantic_key_evaluations: int = 0
    layer_callback_inventory: tuple[tuple[tuple[str, str, str], ...], ...] = ()
    layer_semantic_key_evaluations: tuple[int, ...] = ()
    transition_counts_by_layer: tuple[int, ...] = ()
    prefix_callback_records_sha256: str = ""
    layer_prefix_callback_records_sha256: tuple[str, ...] = ()
    prefixes_with_nonempty_callback_inventory: int = 0
    layer_prefixes_with_nonempty_callback_inventory: tuple[int, ...] = ()
    collision_bisimulation: dict[str, Any] | None = None

    def exact_primary(self, sequence: Sequence[int]) -> Fraction:
        if len(sequence) != self.weeks or int(sequence[0]) != 0:
            raise ValueError("calendar does not match score transducer")
        total = self.initial_edge.total
        count = self.initial_edge.count
        state = 0
        for layer, action in enumerate(sequence[1:]):
            action = int(action)
            flat = state * 3 + action
            edge = self.exact_edges[layer][flat]
            if edge is None:
                raise ValueError("infeasible calendar transition")
            total += edge.total
            count += edge.count
            state = int(self.next_state[layer][state, action])
        return Fraction(1) if count == 0 else total / count

    def score_batch(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Rigorous interval for the canonical per-tape floating mean."""
        if actions.ndim != 2 or actions.shape[1] != self.weeks:
            raise ValueError("action batch has the wrong horizon")
        if np.any(actions[:, 0] != 0):
            raise ValueError("every calendar must start with M")
        n = actions.shape[0]
        state = np.zeros(n, dtype=np.int32)
        lower = np.full(n, self.initial_lower, dtype=np.float64)
        upper = np.full(n, self.initial_upper, dtype=np.float64)
        abs_upper = np.full(n, self.initial_abs_upper, dtype=np.float64)
        count = np.full(n, self.initial_edge.count, dtype=np.int32)
        for layer, action_column in enumerate(actions[:, 1:].T):
            action = action_column.astype(np.intp, copy=False)
            next_ids = self.next_state[layer][state, action]
            if np.any(next_ids < 0):
                raise AssertionError("calendar batch reached an infeasible transition")
            lower = np.nextafter(
                lower + self.delta_lower[layer][state, action], NEG_INF
            )
            upper = np.nextafter(
                upper + self.delta_upper[layer][state, action], POS_INF
            )
            abs_upper = np.nextafter(
                abs_upper + self.delta_abs_upper[layer][state, action], POS_INF
            )
            count += self.delta_count[layer][state, action]
            state = next_ids

        zero = count == 0
        denominator = np.where(zero, 1, count).astype(np.float64)
        mean_lower = np.nextafter(lower / denominator, NEG_INF)
        mean_upper = np.nextafter(upper / denominator, POS_INF)
        mean_lower[zero] = 1.0
        mean_upper[zero] = 1.0

        # Conservative IEEE-754 bound for any summation tree used by np.mean.
        # gamma_(n-1) is looser than pairwise summation and therefore safe.
        operations = np.maximum(count - 1, 0).astype(np.float64)
        gamma = (operations * UNIT_ROUNDOFF) / (
            1.0 - operations * UNIT_ROUNDOFF
        )
        sum_error = gamma * abs_upper
        magnitude = np.maximum(np.abs(mean_lower), np.abs(mean_upper))
        division_error = UNIT_ROUNDOFF * (
            magnitude + sum_error / denominator
        ) / (1.0 - UNIT_ROUNDOFF)
        padding = sum_error / denominator + division_error + np.nextafter(0.0, 1.0)
        mean_lower = np.nextafter(mean_lower - padding, NEG_INF)
        mean_upper = np.nextafter(mean_upper + padding, POS_INF)
        mean_lower[zero] = 1.0
        mean_upper[zero] = 1.0
        if not (np.all(np.isfinite(mean_lower)) and np.all(np.isfinite(mean_upper))):
            raise FloatingPointError("non-finite score interval")
        return mean_lower, mean_upper


def compile_score_transducer(
    transducer: Transducer,
    *,
    seed: int,
    tape_sha256: str,
) -> ScoreTransducer:
    collision_failures = validate_collision_bisimulation_certificate(
        transducer.collision_bisimulation or {},
        expected_collision_count=len(transducer.collisions),
        weeks=transducer.weeks,
    )
    if collision_failures:
        raise ValueError(
            "cannot compile transducer without complete collision certificate: "
            + "; ".join(collision_failures)
        )
    initial_values = transducer.layers[0][0].checkpoint.visible_values
    initial_lower, initial_upper, initial_abs, initial_count, initial_total = (
        _values_label(initial_values)
    )
    next_tables: list[np.ndarray] = []
    lower_tables: list[np.ndarray] = []
    upper_tables: list[np.ndarray] = []
    abs_tables: list[np.ndarray] = []
    count_tables: list[np.ndarray] = []
    exact_tables: list[list[ExactEdge | None]] = []
    digest = sha256()
    digest.update(
        repr(
            (
                initial_total.numerator,
                initial_total.denominator,
                initial_count,
                initial_lower.hex(),
                initial_upper.hex(),
                initial_abs.hex(),
            )
        ).encode()
    )

    for layer_index, transitions in enumerate(transducer.transitions):
        n_states = len(transducer.layers[layer_index])
        next_state = np.full((n_states, 3), -1, dtype=np.int32)
        lower = np.full((n_states, 3), np.nan, dtype=np.float64)
        upper = np.full((n_states, 3), np.nan, dtype=np.float64)
        abs_upper = np.full((n_states, 3), np.nan, dtype=np.float64)
        count = np.zeros((n_states, 3), dtype=np.int32)
        exact: list[ExactEdge | None] = [None] * (n_states * 3)
        for (state_id, action), transition in transitions.items():
            label = _values_label(transition.appended_visible_values)
            next_state[state_id, action] = transition.next_state_id
            lower[state_id, action] = label[0]
            upper[state_id, action] = label[1]
            abs_upper[state_id, action] = label[2]
            count[state_id, action] = label[3]
            exact[state_id * 3 + action] = ExactEdge(label[4], label[3])
        feasible = next_state >= 0
        if np.any(~np.isfinite(lower[feasible])) or np.any(~np.isfinite(upper[feasible])):
            raise FloatingPointError("compiled transition has a non-finite label")
        for array in (next_state, lower, upper, abs_upper, count):
            digest.update(array.tobytes())
        digest.update(
            repr(
                [
                    None
                    if edge is None
                    else (edge.total.numerator, edge.total.denominator, edge.count)
                    for edge in exact
                ]
            ).encode()
        )
        next_tables.append(next_state)
        lower_tables.append(lower)
        upper_tables.append(upper)
        abs_tables.append(abs_upper)
        count_tables.append(count)
        exact_tables.append(exact)

    return ScoreTransducer(
        weeks=transducer.weeks,
        seed=int(seed),
        tape_sha256=str(tape_sha256),
        initial_lower=initial_lower,
        initial_upper=initial_upper,
        initial_abs_upper=initial_abs,
        initial_edge=ExactEdge(initial_total, initial_count),
        next_state=next_tables,
        delta_lower=lower_tables,
        delta_upper=upper_tables,
        delta_abs_upper=abs_tables,
        delta_count=count_tables,
        exact_edges=exact_tables,
        state_counts=[len(layer) for layer in transducer.layers],
        prefix_replays=transducer.prefix_replays,
        collision_count=len(transducer.collisions),
        table_sha256=digest.hexdigest(),
        callback_inventory=tuple(transducer.callback_inventory),
        semantic_key_evaluations=int(transducer.semantic_key_evaluations),
        layer_callback_inventory=tuple(transducer.layer_callback_inventory),
        layer_semantic_key_evaluations=tuple(
            transducer.layer_semantic_key_evaluations
        ),
        transition_counts_by_layer=tuple(
            len(row) for row in transducer.transitions
        ),
        prefix_callback_records_sha256=transducer.prefix_callback_records_sha256,
        layer_prefix_callback_records_sha256=tuple(
            transducer.layer_prefix_callback_records_sha256
        ),
        prefixes_with_nonempty_callback_inventory=(
            transducer.prefixes_with_nonempty_callback_inventory
        ),
        layer_prefixes_with_nonempty_callback_inventory=tuple(
            transducer.layer_prefixes_with_nonempty_callback_inventory
        ),
        collision_bisimulation=dict(transducer.collision_bisimulation or {}),
    )


def collision_certificate_coverage(
    compiled: Sequence[ScoreTransducer],
) -> dict[str, Any]:
    """Fail-closed aggregate coverage for every scientific tape transducer."""
    rows: list[dict[str, Any]] = []
    identities: set[tuple[int, str]] = set()
    failures: list[str] = []
    for index, transducer in enumerate(compiled):
        certificate = transducer.collision_bisimulation or {}
        identity = (int(transducer.seed), str(transducer.tape_sha256))
        if identity in identities:
            failures.append(f"duplicate certificate identity at index {index}")
        identities.add(identity)
        certificate_failures = validate_collision_bisimulation_certificate(
            certificate,
            expected_collision_count=transducer.collision_count,
            weeks=transducer.weeks,
        )
        failures.extend(
            f"index {index}: {failure}" for failure in certificate_failures
        )
        rows.append(
            {
                "index": index,
                "seed": transducer.seed,
                "tape_sha256": transducer.tape_sha256,
                "collision_count": transducer.collision_count,
                "collision_root_count": certificate.get("collision_root_count"),
                "node_obligation_count": certificate.get(
                    "node_obligation_count"
                ),
                "certificate_sha256": certificate.get("certificate_sha256"),
                "complete": not certificate_failures,
            }
        )
    payload = {
        "schema_version": "paper2_collision_certificate_coverage_v1",
        "required_count": len(compiled),
        "complete_count": sum(row["complete"] is True for row in rows),
        "unique_identity_count": len(identities),
        "rows": rows,
        "rows_sha256": _json_digest(rows),
        "failures": failures,
        "passed": (
            bool(compiled)
            and len(identities) == len(compiled)
            and all(row["complete"] is True for row in rows)
            and not failures
        ),
    }
    payload["coverage_sha256"] = _json_digest(payload)
    return payload


def _collision_certificate_summary(certificate: dict[str, Any]) -> dict[str, Any]:
    """Small provenance view; complete records remain inside the hashed pickle."""
    omitted = {"node_obligations", "collision_roots"}
    return {
        **{key: value for key, value in certificate.items() if key not in omitted},
        "records_embedded": False,
        "records_location": "hashed_score_checkpoint_pickle",
    }


@dataclass(frozen=True)
class BuildSpec:
    index: int
    seed: int
    context: str
    split: str
    weeks: int


def _build_fingerprint(spec: BuildSpec) -> dict[str, Any]:
    return {
        "schema_version": "paper2_bottleneck_score_checkpoint_v2",
        "index": spec.index,
        "seed": spec.seed,
        "context": spec.context,
        "split": spec.split,
        "weeks": spec.weeks,
        "key_schema_version": KEY_SCHEMA_VERSION,
        "git_head": _git_value("rev-parse", "HEAD"),
        "frontier_runner_sha256": _file_sha256(Path(__file__)),
        "dependency_sha256": {
            str(path.relative_to(ROOT)): _file_sha256(path)
            for path in CHECKPOINT_DEPENDENCY_PATHS
        },
        "environment": certification_environment(),
    }


def _compiled_transducer_proof(compiled: ScoreTransducer) -> dict[str, Any]:
    payload = {
        "schema_version": "paper2_bottleneck_compiled_transducer_proof_v2",
        "builder": "build_transducer_then_compile_score_transducer",
        "weeks": compiled.weeks,
        "seed": compiled.seed,
        "tape_sha256": compiled.tape_sha256,
        "score_table_sha256": compiled.table_sha256,
        "state_counts": compiled.state_counts,
        "transition_counts_by_layer": list(compiled.transition_counts_by_layer),
        "prefix_replays": compiled.prefix_replays,
        "semantic_key_evaluations": compiled.semantic_key_evaluations,
        "layer_semantic_key_evaluations": list(
            compiled.layer_semantic_key_evaluations
        ),
        "callback_inventory": [list(row) for row in compiled.callback_inventory],
        "layer_callback_inventory": [
            [list(item) for item in row]
            for row in compiled.layer_callback_inventory
        ],
        "prefix_callback_records_sha256": compiled.prefix_callback_records_sha256,
        "layer_prefix_callback_records_sha256": list(
            compiled.layer_prefix_callback_records_sha256
        ),
        "prefixes_with_nonempty_callback_inventory": (
            compiled.prefixes_with_nonempty_callback_inventory
        ),
        "layer_prefixes_with_nonempty_callback_inventory": list(
            compiled.layer_prefixes_with_nonempty_callback_inventory
        ),
        "collision_bisimulation": _collision_certificate_summary(
            compiled.collision_bisimulation or {}
        ),
    }
    return {**payload, "proof_sha256": _json_digest(payload)}


def _checkpoint_paths(directory: Path, spec: BuildSpec) -> tuple[Path, Path]:
    stem = f"{spec.index:03d}_seed_{spec.seed}_{spec.context}"
    return directory / f"{stem}.pickle", directory / f"{stem}.json"


def _write_pickle_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _build_score_checkpoint(spec: BuildSpec, directory: Path) -> dict[str, Any]:
    """Spawn-safe worker: build one tape and persist it before returning."""
    data_path, metadata_path = _checkpoint_paths(directory, spec)
    fingerprint = _build_fingerprint(spec)
    tape = materialize_tape(
        spec.seed,
        spec.context,
        spec.split,
        weeks=spec.weeks,
    )
    compiled = compile_score_transducer(
        build_transducer(tape, spec.weeks),
        seed=spec.seed,
        tape_sha256=tape["threat_sha256"],
    )
    _write_pickle_atomic(data_path, compiled)
    source_proof = _compiled_transducer_proof(compiled)
    metadata = {
        **fingerprint,
        "fingerprint_sha256": _json_digest(fingerprint),
        "data_path": str(data_path.resolve()),
        "data_sha256": _file_sha256(data_path),
        "data_bytes": data_path.stat().st_size,
        "tape_sha256": compiled.tape_sha256,
        "score_table_sha256": compiled.table_sha256,
        "state_counts": compiled.state_counts,
        "prefix_replays": compiled.prefix_replays,
        "collision_count": compiled.collision_count,
        "source_transducer_proof": source_proof,
    }
    _atomic_json(metadata_path, metadata)
    return metadata


def _load_score_checkpoint(
    spec: BuildSpec,
    directory: Path,
) -> tuple[ScoreTransducer, dict[str, Any]] | None:
    data_path, metadata_path = _checkpoint_paths(directory, spec)
    if not (data_path.is_file() and metadata_path.is_file()):
        return None
    try:
        metadata = json.loads(metadata_path.read_text())
        fingerprint = _build_fingerprint(spec)
        if metadata.get("fingerprint_sha256") != _json_digest(fingerprint):
            return None
        if metadata.get("data_sha256") != _file_sha256(data_path):
            return None
        with data_path.open("rb") as handle:
            compiled = pickle.load(handle)
    except (
        OSError,
        EOFError,
        AttributeError,
        ImportError,
        ValueError,
        json.JSONDecodeError,
        pickle.PickleError,
    ):
        return None
    if not isinstance(compiled, ScoreTransducer):
        return None
    if (
        compiled.seed != spec.seed
        or compiled.weeks != spec.weeks
        or compiled.table_sha256 != metadata.get("score_table_sha256")
        or compiled.tape_sha256 != metadata.get("tape_sha256")
        or metadata.get("source_transducer_proof")
        != _compiled_transducer_proof(compiled)
        or validate_collision_bisimulation_certificate(
            compiled.collision_bisimulation or {},
            expected_collision_count=compiled.collision_count,
            weeks=compiled.weeks,
        )
    ):
        return None
    return compiled, metadata


def build_score_transducers(
    specs: Sequence[BuildSpec],
    *,
    workers: int,
    checkpoint_dir: Path,
    progress_path: Path,
    started: float,
) -> tuple[list[ScoreTransducer], list[dict[str, Any]]]:
    if workers < 1:
        raise ValueError("build workers must be positive")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    compiled_by_index: dict[int, ScoreTransducer] = {}
    metadata_by_index: dict[int, dict[str, Any]] = {}
    pending: list[BuildSpec] = []
    for spec in specs:
        cached = _load_score_checkpoint(spec, checkpoint_dir)
        if cached is None:
            pending.append(spec)
        else:
            compiled_by_index[spec.index], metadata_by_index[spec.index] = cached

    def record_progress() -> None:
        _atomic_json(
            progress_path,
            {
                "stage": "build_transducers",
                "completed": len(compiled_by_index),
                "total": len(specs),
                "resumed": len(specs) - len(pending),
                "completed_indices": sorted(compiled_by_index),
                "checkpoint_dir": str(checkpoint_dir.resolve()),
                "checkpoint_sha256": {
                    str(index): metadata_by_index[index]["data_sha256"]
                    for index in sorted(metadata_by_index)
                },
                "elapsed_seconds": time.perf_counter() - started,
            },
        )

    record_progress()
    if workers == 1:
        for spec in pending:
            _build_score_checkpoint(spec, checkpoint_dir)
            loaded = _load_score_checkpoint(spec, checkpoint_dir)
            if loaded is None:
                raise RuntimeError(f"new checkpoint failed validation: seed {spec.seed}")
            compiled_by_index[spec.index], metadata_by_index[spec.index] = loaded
            record_progress()
    elif pending:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            futures = {
                pool.submit(_build_score_checkpoint, spec, checkpoint_dir): spec
                for spec in pending
            }
            for future in as_completed(futures):
                spec = futures[future]
                future.result()
                loaded = _load_score_checkpoint(spec, checkpoint_dir)
                if loaded is None:
                    raise RuntimeError(
                        f"new checkpoint failed validation: seed {spec.seed}"
                    )
                compiled_by_index[spec.index], metadata_by_index[spec.index] = loaded
                record_progress()
    if set(compiled_by_index) != {spec.index for spec in specs}:
        raise RuntimeError("not every transducer checkpoint was completed")
    return (
        [compiled_by_index[spec.index] for spec in specs],
        [metadata_by_index[spec.index] for spec in specs],
    )


def _aggregate_intervals(
    per_tape: Sequence[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    n = len(per_tape)
    lower = np.zeros_like(per_tape[0][0])
    upper = np.zeros_like(per_tape[0][1])
    for tape_lower, tape_upper in per_tape:
        lower = np.nextafter(lower + tape_lower, NEG_INF)
        upper = np.nextafter(upper + tape_upper, POS_INF)
    return (
        np.nextafter(lower / n, NEG_INF),
        np.nextafter(upper / n, POS_INF),
    )


@dataclass
class ScreeningResult:
    objective_scope: str
    calendar_count: int
    pass1_count: int
    pass2_count: int
    aggregate_best_lower: float
    per_tape_best_lower: list[float]
    aggregate_contenders: list[int]
    per_tape_contenders: list[list[int]]
    contender_overflow: dict[str, Any]
    pass1_stream_sha256: str
    pass2_stream_sha256: str


def _update_screen_digest(
    digest: Any,
    *,
    start: int,
    actions: np.ndarray,
    intervals: Sequence[tuple[np.ndarray, np.ndarray]],
    aggregate: tuple[np.ndarray, np.ndarray] | None,
    objective_scope: str,
) -> None:
    """Hash index stream and numeric screen output for pass reproducibility."""
    digest.update(objective_scope.encode("ascii"))
    digest.update(int(start).to_bytes(8, "little", signed=False))
    digest.update(np.asarray(actions.shape, dtype=np.int64).tobytes())
    digest.update(actions.tobytes(order="C"))
    for lower, upper in intervals:
        digest.update(lower.tobytes(order="C"))
        digest.update(upper.tobytes(order="C"))
    if aggregate is not None:
        digest.update(aggregate[0].tobytes(order="C"))
        digest.update(aggregate[1].tobytes(order="C"))


def screen_frontier(
    transducers: Sequence[ScoreTransducer],
    *,
    batch_size: int,
    max_contenders: int,
    objective_scope: str = "aggregate_and_per_tape",
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> ScreeningResult:
    if not transducers:
        raise ValueError("at least one score transducer is required")
    weeks = transducers[0].weeks
    if any(row.weeks != weeks for row in transducers):
        raise ValueError("all transducers must share a horizon")
    if objective_scope not in {
        "aggregate_only",
        "per_tape_only",
        "aggregate_and_per_tape",
    }:
        raise ValueError(f"unsupported frontier objective scope: {objective_scope}")
    need_aggregate = objective_scope != "per_tape_only"
    need_per_tape = objective_scope != "aggregate_only"
    expected = feasible_calendar_count(weeks)
    aggregate_best_lower = NEG_INF
    per_tape_best_lower = (
        np.full(len(transducers), NEG_INF, dtype=np.float64)
        if need_per_tape
        else np.asarray([], dtype=np.float64)
    )
    pass1_count = 0
    pass1_digest = sha256()
    for start, actions in iter_calendar_batches(weeks, batch_size):
        intervals = [row.score_batch(actions) for row in transducers]
        aggregate = _aggregate_intervals(intervals) if need_aggregate else None
        _update_screen_digest(
            pass1_digest,
            start=start,
            actions=actions,
            intervals=intervals,
            aggregate=aggregate,
            objective_scope=objective_scope,
        )
        if aggregate is not None:
            aggregate_best_lower = max(
                aggregate_best_lower, float(np.max(aggregate[0]))
            )
        if need_per_tape:
            for tape_index, (lower, _upper) in enumerate(intervals):
                per_tape_best_lower[tape_index] = max(
                    per_tape_best_lower[tape_index], float(np.max(lower))
                )
        pass1_count += len(actions)
        if progress is not None:
            progress({"pass": 1, "completed": pass1_count, "total": expected, "batch_start": start})
    if pass1_count != expected:
        raise AssertionError("pass 1 did not enumerate the complete calendar family")

    aggregate_contenders: list[int] = []
    per_tape_contenders: list[list[int]] = (
        [[] for _ in transducers] if need_per_tape else []
    )
    overflow: dict[str, Any] = {"aggregate": False, "per_tape": []}
    overflow_per_tape = [False] * len(transducers) if need_per_tape else []
    pass2_count = 0
    pass2_digest = sha256()
    for start, actions in iter_calendar_batches(weeks, batch_size):
        intervals = [row.score_batch(actions) for row in transducers]
        aggregate = _aggregate_intervals(intervals) if need_aggregate else None
        _update_screen_digest(
            pass2_digest,
            start=start,
            actions=actions,
            intervals=intervals,
            aggregate=aggregate,
            objective_scope=objective_scope,
        )
        if aggregate is not None:
            aggregate_hits = np.flatnonzero(aggregate[1] >= aggregate_best_lower)
            for offset in aggregate_hits:
                if len(aggregate_contenders) < max_contenders:
                    aggregate_contenders.append(start + int(offset))
                else:
                    overflow["aggregate"] = True
        if need_per_tape:
            for tape_index, (_lower, upper) in enumerate(intervals):
                hits = np.flatnonzero(upper >= per_tape_best_lower[tape_index])
                for offset in hits:
                    if len(per_tape_contenders[tape_index]) < max_contenders:
                        per_tape_contenders[tape_index].append(start + int(offset))
                    else:
                        overflow_per_tape[tape_index] = True
        pass2_count += len(actions)
        if progress is not None:
            progress({"pass": 2, "completed": pass2_count, "total": expected, "batch_start": start})
    overflow["per_tape"] = [
        transducers[index].seed
        for index, value in enumerate(overflow_per_tape)
        if value
    ]
    if pass2_count != expected:
        raise AssertionError("pass 2 did not enumerate the complete calendar family")
    if pass1_digest.digest() != pass2_digest.digest():
        raise AssertionError("the two complete frontier passes were not identical")
    return ScreeningResult(
        objective_scope=objective_scope,
        calendar_count=expected,
        pass1_count=pass1_count,
        pass2_count=pass2_count,
        aggregate_best_lower=aggregate_best_lower,
        per_tape_best_lower=[float(value) for value in per_tape_best_lower],
        aggregate_contenders=aggregate_contenders,
        per_tape_contenders=per_tape_contenders,
        contender_overflow=overflow,
        pass1_stream_sha256=pass1_digest.hexdigest(),
        pass2_stream_sha256=pass2_digest.hexdigest(),
    )


ScoreProvider = Callable[[int, int], Fraction]


def _select_exact(
    contenders: Sequence[int],
    score: Callable[[int], Fraction],
) -> tuple[Fraction, list[int]]:
    if not contenders:
        raise ValueError("exact selection received no contenders")
    maximum: Fraction | None = None
    winners: list[int] = []
    for index in sorted(set(map(int, contenders))):
        value = score(index)
        if maximum is None or value > maximum:
            maximum = value
            winners = [index]
        elif value == maximum:
            winners.append(index)
    assert maximum is not None
    return maximum, winners


def resolve_frontier(
    transducers: Sequence[ScoreTransducer],
    screening: ScreeningResult,
    *,
    score_provider: ScoreProvider | None = None,
    acceleration_certified: bool,
) -> dict[str, Any]:
    """Resolve every interval overlap with exact binary scores."""
    need_aggregate = screening.objective_scope != "per_tape_only"
    need_per_tape = screening.objective_scope != "aggregate_only"
    overflow = bool(
        need_aggregate and screening.contender_overflow["aggregate"]
    ) or bool(need_per_tape and screening.contender_overflow["per_tape"])
    complete = (
        screening.pass1_count == screening.calendar_count
        and screening.pass2_count == screening.calendar_count
    )
    provider = score_provider or (
        lambda tape_index, index: transducers[tape_index].exact_primary(
            calendar_at_index(index, transducers[tape_index].weeks)
        )
    )
    if overflow or not complete:
        return {
            "objective_scope": screening.objective_scope,
            "exact_maximum_certified": False,
            "fail_closed_reason": "contender_overflow_or_incomplete_enumeration",
            "overflow_is_terminal": bool(overflow),
            "partial_selection_performed": False,
            "aggregate": None,
            "per_tape": [],
        }

    aggregate_row = None
    if need_aggregate:
        aggregate_value, aggregate_winners = _select_exact(
            screening.aggregate_contenders,
            lambda index: sum(
                (
                    provider(tape_index, index)
                    for tape_index in range(len(transducers))
                ),
                Fraction(0),
            )
            / len(transducers),
        )
        aggregate_row = {
            "score": _fraction_payload(aggregate_value),
            "winner_indices": aggregate_winners,
            "primary_tie_count": len(aggregate_winners),
            "primary_tie_indices_sha256": _json_digest(aggregate_winners),
            "winner_calendars": [
                _action_names(calendar_at_index(index, transducers[0].weeks))
                for index in aggregate_winners
            ],
            "unique_winner": len(aggregate_winners) == 1,
            "interval_contender_count": len(screening.aggregate_contenders),
        }
    per_tape_rows = []
    if need_per_tape:
        for tape_index, row in enumerate(transducers):
            value, winners = _select_exact(
                screening.per_tape_contenders[tape_index],
                lambda index, tape_index=tape_index: provider(tape_index, index),
            )
            per_tape_rows.append(
                {
                    "seed": row.seed,
                    "tape_sha256": row.tape_sha256,
                    "oracle_score": _fraction_payload(value),
                    "winner_indices": winners,
                    "oracle_tie_count": len(winners),
                    "oracle_tie_indices_sha256": _json_digest(winners),
                    "oracle_tie_selection_rule": "none_all_exact_ties_retained",
                    "display_representative_index": min(winners),
                    "representative_semantics": (
                        "display_only_not_used_for_h_pi_or_guardrail_feasibility"
                    ),
                    "winner_calendars": [
                        _action_names(calendar_at_index(index, row.weeks))
                        for index in winners
                    ],
                    "unique_winner": len(winners) == 1,
                    "interval_contender_count": len(
                        screening.per_tape_contenders[tape_index]
                    ),
                }
            )
    certified = bool(acceleration_certified and complete and not overflow)
    return {
        "objective_scope": screening.objective_scope,
        "exact_maximum_certified": certified,
        "fail_closed_reason": None if certified else "acceleration_not_certified",
        "aggregate": aggregate_row,
        "per_tape": per_tape_rows,
    }


def _week_from_observation(observation: dict[str, float], weeks: int) -> int:
    return int(round(float(observation["week_phase"]) * (weeks - 1)))


def active_calendar_policy(sequence: Sequence[int]):
    sequence = tuple(map(int, sequence))
    weeks = len(sequence)
    if not sequence or sequence[0] != 0:
        raise ValueError("active calendar must start with M")

    def policy(observation: dict[str, float]):
        week = _week_from_observation(observation, weeks)
        return ACTIONS[sequence[min(week + 1, weeks - 1)]]

    return policy


def _selected_replay_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row.get(key) for key in REPLAY_KEYS}


def replay_calendar(
    tape: dict[str, Any],
    sequence: Sequence[int],
    expected_score: Fraction,
) -> dict[str, Any]:
    row = run_policy(tape, active_calendar_policy(sequence))
    active = "".join(
        ACTION_NAMES[tuple(event["action"])] for event in row["action_events"]
    )
    emitted = Fraction.from_float(float(row["ret_excel"]))
    expected_team_hours = 168.0 * len(sequence)
    token_sum = sum(
        float(row[key]) for key in ("token_hours_m", "token_hours_t", "token_hours_r")
    )
    resource_semantics_match = bool(
        float(row["total_token_hours"]) == expected_team_hours
        and token_sum == expected_team_hours
        and float(row["mass_residual"]) == 0.0
        and float(row["reserve_inventory_initial"]) == 10_000.0
        and float(row["reserve_capacity"]) == 10_000.0
        and float(row["reserve_target_terminal"]) == 10_000.0
        and float(row["reserve_replenishment_lead_time"]) == 168.0
        and float(row["reserve_issue_delay"]) == 24.0
        and abs(float(row["reserve_stock_balance_residual"])) <= 1e-9
        and all(
            float(row[key]) >= 0.0
            for key in (
                "reserve_units_issued",
                "reserve_units_replenished",
                "reserve_inventory_terminal",
                "reserve_committed_pending_terminal",
                "reserve_in_transit_terminal",
                "reserve_replenishment_requests",
            )
        )
    )
    crn_hashes_present = bool(
        _is_sha256(row.get("consumed_base_threat_sha256"))
        and _is_sha256(row.get("realized_demand_sha256"))
    )
    return {
        "seed": int(tape["seed"]),
        "tape_sha256": tape["threat_sha256"],
        "calendar": _action_names(sequence),
        "calendar_index": calendar_index(sequence),
        "active_sequence_matches": active == _action_names(sequence),
        "expected_score": _fraction_payload(expected_score),
        "emitted_score": _fraction_payload(emitted),
        "primary_exact_match": emitted == expected_score,
        "resource_semantics_match": resource_semantics_match,
        "crn_hashes_present": crn_hashes_present,
        "expected_team_hours": expected_team_hours,
        "guardrails": _selected_replay_row(row),
        "guardrail_digest": _json_digest(_selected_replay_row(row)),
    }


def resolve_calibration_tie_break(
    resolved: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    *,
    contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply the frozen guardrail ordering only among exact aggregate ties."""
    contract = contract or json.loads(PRIMARY_CONTRACT_PATH.read_text())
    tie_spec = contract.get("calibration_tie_break", {})
    priorities = tie_spec.get("lexicographic_priorities")
    failures: list[str] = []
    aggregate = resolved.get("aggregate") or {}
    winners = sorted(set(map(int, aggregate.get("winner_indices", []))))
    expected_seeds = [
        row["seed"] for row in _contract_seed_rows(contract, "calibration")
    ]
    if not winners:
        failures.append("no exact aggregate primary tie set")
    if not isinstance(priorities, list) or not priorities:
        failures.append("calibration tie priorities are missing")
        priorities = []
    fields = [row.get("field") for row in priorities if isinstance(row, dict)]
    directions = [row.get("direction") for row in priorities if isinstance(row, dict)]
    priorities_are_valid = not (
        len(fields) != len(priorities)
        or len(set(fields)) != len(fields)
        or fields[-1:] != ["calendar_index"]
        or directions[-1:] != ["minimize"]
        or any(direction not in {"minimize", "maximize"} for direction in directions)
        or any(field not in REPLAY_KEYS for field in fields[:-1])
    )
    if not priorities_are_valid:
        failures.append("calibration tie priorities are malformed or unsupported")
        # Do not continue through partially understood priorities.  An empty key
        # is harmless because any accumulated failure prevents selection below.
        priorities = []

    tie_rows = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("role") == "calibration_aggregate_winner"
    ]
    expected_pairs = {
        (winner, seed) for winner in winners for seed in expected_seeds
    }
    actual_pairs = [
        (int(row.get("calendar_index", -1)), int(row.get("seed", -1)))
        for row in tie_rows
    ]
    if len(actual_pairs) != len(expected_pairs) or set(actual_pairs) != expected_pairs:
        failures.append("calibration tie replay coverage is not ties x 60")

    candidate_rows: list[dict[str, Any]] = []
    comparable: list[tuple[tuple[Fraction | int, ...], int]] = []
    for winner in winners:
        replay_group = [
            row for row in tie_rows if int(row.get("calendar_index", -1)) == winner
        ]
        aggregate_guardrails: dict[str, dict[str, Any]] = {}
        key: list[Fraction | int] = []
        for priority in priorities:
            field = priority.get("field")
            direction = priority.get("direction")
            if field == "calendar_index":
                key.append(winner)
                continue
            values: list[Fraction] = []
            for replay in replay_group:
                value = replay.get("guardrails", {}).get(field)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                ):
                    failures.append(
                        f"missing or nonfinite calibration tie field {field} for {winner}"
                    )
                    continue
                values.append(
                    Fraction(int(value), 1)
                    if isinstance(value, int)
                    else Fraction.from_float(float(value))
                )
            if len(values) != len(expected_seeds):
                failures.append(
                    f"calibration tie field coverage mismatch {field} for {winner}"
                )
                mean = Fraction(0)
            else:
                mean = sum(values, Fraction(0)) / len(values)
            aggregate_guardrails[field] = _fraction_payload(mean)
            key.append(mean if direction == "minimize" else -mean)
        candidate_rows.append(
            {
                "calendar_index": winner,
                "aggregate_guardrails": aggregate_guardrails,
                "comparison_key_exact": [
                    _fraction_payload(value)
                    if isinstance(value, Fraction)
                    else {"integer": value}
                    for value in key
                ],
            }
        )
        comparable.append((tuple(key), winner))

    selected = min(comparable)[1] if comparable and not failures else None
    payload = {
        "schema_version": "paper2_calibration_tie_break_audit_v1",
        "contract_section_sha256": _json_digest(tie_spec),
        "primary_tie_indices": winners,
        "primary_tie_count": len(winners),
        "primary_tie_indices_sha256": _json_digest(winners),
        "required_replay_count": len(winners) * len(expected_seeds),
        "observed_replay_count": len(tie_rows),
        "aggregate_rule": tie_spec.get("aggregate_rule"),
        "lexicographic_priorities": priorities,
        "candidate_aggregates": candidate_rows,
        "selected_calendar_index": selected,
        "guardrail_constrained_frontier_certified": False,
        "passed": selected is not None and not failures,
        "failures": sorted(set(failures)),
    }
    payload["audit_sha256"] = _json_digest(payload)
    return payload


def validate_selected_replay_set(
    rows: Sequence[dict[str, Any]],
    *,
    phase: str | None = None,
    aggregate_winner_indices: Sequence[int] | None = None,
    per_tape_winner_indices: dict[int, Sequence[int]] | None = None,
    expected_exogenous_by_seed: dict[int, dict[str, str]] | None = None,
    contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Require action-invariant exogenous hashes and the complete resource ledger."""
    failures: list[str] = []
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in rows:
        tape_sha = str(row.get("tape_sha256", ""))
        grouped.setdefault((int(row.get("seed", -1)), tape_sha), []).append(row)
        if not _is_sha256(tape_sha):
            failures.append("selected replay tape hash missing or malformed")
        if row.get("primary_exact_match") is not True:
            failures.append("selected replay primary mismatch")
        if row.get("active_sequence_matches") is not True:
            failures.append("selected replay action trajectory mismatch")
        if row.get("resource_semantics_match") is not True:
            failures.append("selected replay resource ledger mismatch")
        if row.get("crn_hashes_present") is not True:
            failures.append("selected replay CRN hashes missing")
    crn_groups: list[dict[str, Any]] = []
    contract = contract or json.loads(PRIMARY_CONTRACT_PATH.read_text())
    expected_block = (
        _contract_seed_rows(contract, "calibration")
        if phase == "calibration"
        else _contract_seed_rows(contract, "locked_bound")
        if phase == "locked"
        else []
    )
    expected_seeds = {row["seed"] for row in expected_block}
    if phase in {"calibration", "locked"} and {
        seed for seed, _tape in grouped
    } != expected_seeds:
        failures.append(f"{phase} replay seed coverage mismatch")
    if phase in {"calibration", "locked"} and len(grouped) != len(expected_seeds):
        failures.append(f"{phase} replay tape multiplicity mismatch")
    if phase in {"calibration", "locked"} and (
        not isinstance(expected_exogenous_by_seed, dict)
        or set(expected_exogenous_by_seed) != expected_seeds
    ):
        failures.append(f"{phase} independent exogenous-hash coverage mismatch")
    for (seed, tape_sha), group in sorted(grouped.items()):
        threat = {row.get("guardrails", {}).get("consumed_base_threat_sha256") for row in group}
        demand = {row.get("guardrails", {}).get("realized_demand_sha256") for row in group}
        expected_indices: set[int] | None = None
        expected_roles: dict[str, int] | None = None
        if phase == "calibration":
            expected_indices = set(map(int, aggregate_winner_indices or []))
            expected_roles = {"calibration_aggregate_winner": len(expected_indices)}
        elif phase == "locked":
            oracle_indices = set(
                map(int, (per_tape_winner_indices or {}).get(seed, []))
            )
            expected_indices = oracle_indices
            expected_roles = {
                "fixed_calibration_comparator": 1,
                "per_tape_oracle_winner": len(oracle_indices),
            }
        actual_roles = {
            role: sum(row.get("role") == role for row in group)
            for role in {str(row.get("role")) for row in group}
        }
        index_ok = True
        if phase == "calibration":
            index_ok = {
                int(row.get("calendar_index", -1)) for row in group
            } == expected_indices
        elif phase == "locked":
            index_ok = {
                int(row.get("calendar_index", -1))
                for row in group
                if row.get("role") == "per_tape_oracle_winner"
            } == expected_indices
        crn_ok = (
            len(threat) == 1
            and len(demand) == 1
            and _is_sha256(next(iter(threat)))
            and _is_sha256(next(iter(demand)))
        )
        expected_exogenous = (
            (expected_exogenous_by_seed or {}).get(seed, {})
        )
        expected_hashes_match = bool(
            expected_exogenous
            and next(iter(threat), None)
            == expected_exogenous.get("consumed_base_threat_sha256")
            and next(iter(demand), None)
            == expected_exogenous.get("realized_demand_sha256")
            and tape_sha == expected_exogenous.get("tape_sha256")
        ) if phase in {"calibration", "locked"} else True
        passed = bool(group) and crn_ok and expected_hashes_match and index_ok
        if expected_roles is not None:
            passed = passed and actual_roles == expected_roles
        elif phase is None and len(group) < 2:
            passed = False
        if not passed:
            failures.append(f"CRN replay identity failed for seed {seed}")
        crn_groups.append({
            "seed": seed,
            "tape_sha256": tape_sha,
            "replay_count": len(group),
            "consumed_base_threat_sha256": next(iter(threat)) if len(threat) == 1 else None,
            "realized_demand_sha256": next(iter(demand)) if len(demand) == 1 else None,
            "independent_expected_exogenous": expected_exogenous or None,
            "expected_hashes_match": expected_hashes_match,
            "passed": passed,
        })
    payload = {
        "schema_version": "paper2_selected_replay_crn_resource_audit_v1",
        "groups": crn_groups,
        "replay_count": len(rows),
        "passed": bool(rows) and not failures,
        "failures": sorted(set(failures)),
    }
    payload["audit_sha256"] = _json_digest(payload)
    return payload


def _git_value(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _scientific_source_drift() -> str:
    paths = (Path(__file__),) + CHECKPOINT_DEPENDENCY_PATHS
    return _git_value(
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        *(str(path.relative_to(ROOT)) for path in paths),
    ) or ""


def _repo_artifact(relative: Any) -> Path:
    try:
        rel, candidate = _confined_checksum_path(ROOT, relative)
    except HarnessError as exc:
        raise ValueError("authorization artifact escapes the repository") from exc
    if not candidate.is_file():
        raise ValueError(f"authorization artifact is missing: {relative}")
    if not _git_value("ls-files", "--error-unmatch", "--", rel):
        raise ValueError(f"authorization artifact is not tracked: {relative}")
    shown = subprocess.run(
        ["git", "show", f"HEAD:{rel}"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    if shown.returncode != 0 or sha256(shown.stdout).hexdigest() != _file_sha256(candidate):
        raise ValueError(f"authorization artifact differs from HEAD: {relative}")
    return candidate


def validate_w24_profile_state_audit_payload(
    audit: dict[str, Any],
    *,
    expected_environment_sha256: str | None = None,
) -> list[str]:
    """Validate the generated W24 cache/profile proof without reopening its tape."""
    failures: list[str] = []
    expected_spec = BuildSpec(
        index=0,
        seed=1_110_001,
        context=CONTEXTS[0],
        split="w24_profile_state_audit_burned",
        weeks=24,
    )
    expected_fingerprint = _build_fingerprint(expected_spec)
    expected = {
        "schema_version": W24_AUDIT_SCHEMA_VERSION,
        "generated_not_authorization": True,
        "weeks": 24,
        "seed": 1_110_001,
        "split": expected_spec.split,
        "key_schema_version": KEY_SCHEMA_VERSION,
        "primary_contract_sha256": _file_sha256(PRIMARY_CONTRACT_PATH),
        "generated_by_frontier_runner_sha256": _file_sha256(Path(__file__)),
        "transducer_runner_sha256": _file_sha256(TRANSDUCER_RUNNER_PATH),
        "profile_audit_passed": True,
        "semantic_state_inventory_complete": True,
        "unknown_mutable_state_count": 0,
        "unclassified_callback_owner_count": 0,
        "collision_bisimulation_passed": True,
        "dependency_sha256": expected_fingerprint["dependency_sha256"],
    }
    for key, value in expected.items():
        if audit.get(key) != value:
            failures.append(f"W24 profile/state audit field mismatch: {key}")
    environment = audit.get("environment", {})
    required_environment = (
        expected_environment_sha256
        or expected_fingerprint["environment"]["environment_sha256"]
    )
    if environment != expected_fingerprint["environment"]:
        failures.append("W24 profile/state audit environment identity drifted")
    if environment.get("environment_sha256") != required_environment:
        failures.append("W24 profile/state audit environment digest mismatch")

    summary = audit.get("checkpoint_summary", {})
    summary_body = dict(summary) if isinstance(summary, dict) else {}
    summary_digest = summary_body.pop("summary_sha256", None)
    if summary_digest != _json_digest(summary_body):
        failures.append("W24 checkpoint-summary digest mismatch")
    source_proof = summary.get("source_transducer_proof", {}) if isinstance(summary, dict) else {}
    proof_body = dict(source_proof) if isinstance(source_proof, dict) else {}
    proof_digest = proof_body.pop("proof_sha256", None)
    if proof_digest != _json_digest(proof_body):
        failures.append("W24 source-transducer proof digest mismatch")
    layer_evaluations = source_proof.get("layer_semantic_key_evaluations", [])
    layer_callbacks = source_proof.get("layer_callback_inventory", [])
    layer_callback_digests = source_proof.get(
        "layer_prefix_callback_records_sha256", []
    )
    layer_nonempty = source_proof.get(
        "layer_prefixes_with_nonempty_callback_inventory", []
    )
    collision_bisimulation = audit.get("collision_bisimulation", {})
    if not (
        source_proof.get("builder") == "build_transducer_then_compile_score_transducer"
        and source_proof.get("weeks") == 24
        and source_proof.get("seed") == 1_110_001
        and source_proof.get("tape_sha256") == audit.get("tape_sha256")
        and source_proof.get("score_table_sha256") == summary.get("score_table_sha256")
        and source_proof.get("state_counts") == summary.get("state_counts")
        and source_proof.get("prefix_replays") == summary.get("prefix_replays")
        and source_proof.get("semantic_key_evaluations") == summary.get("prefix_replays")
        and isinstance(layer_evaluations, list)
        and len(layer_evaluations) == 24
        and sum(layer_evaluations) == summary.get("prefix_replays")
        and isinstance(layer_callbacks, list)
        and len(layer_callbacks) == 24
        and all(isinstance(row, list) and row for row in layer_callbacks)
        and len(source_proof.get("transition_counts_by_layer", [])) == 23
        and source_proof.get("prefixes_with_nonempty_callback_inventory")
        == summary.get("prefix_replays")
        and isinstance(layer_nonempty, list)
        and sum(layer_nonempty) == summary.get("prefix_replays")
        and _is_sha256(source_proof.get("prefix_callback_records_sha256"))
        and isinstance(layer_callback_digests, list)
        and len(layer_callback_digests) == 24
        and all(_is_sha256(row) for row in layer_callback_digests)
        and source_proof.get("collision_bisimulation")
        == _collision_certificate_summary(collision_bisimulation)
    ):
        failures.append("W24 layer-wide callback/evaluation proof failed")
    if not (
        collision_bisimulation.get("passed") is True
        and collision_bisimulation.get("key_schema_version")
        == KEY_SCHEMA_VERSION
        and collision_bisimulation.get("complete_state_serialization") is True
        and collision_bisimulation.get("event_payload_serialized") is True
        and collision_bisimulation.get("resource_users_serialized") is True
        and collision_bisimulation.get("callback_closure_state_serialized") is True
        and collision_bisimulation.get(
            "process_target_state_serialized_or_fail_closed"
        ) is True
        and collision_bisimulation.get("runtime_alias_graph_serialized") is True
        and collision_bisimulation.get("collision_payload_checks")
        == summary.get("collision_count")
        and collision_bisimulation.get("collision_root_count")
        == summary.get("collision_count")
        and collision_bisimulation.get("unresolved_node_obligation_count") == 0
        and collision_bisimulation.get("unresolved_collision_root_count") == 0
        and collision_bisimulation.get("all_actions_covered") is True
        and collision_bisimulation.get("backward_induction_complete") is True
        and not collision_bisimulation.get("mismatch_examples")
        and _is_sha256(collision_bisimulation.get("transition_record_sha256"))
    ):
        failures.append("W24 collision bisimulation is incomplete")
    for failure in validate_collision_bisimulation_certificate(
        collision_bisimulation,
        expected_collision_count=int(summary.get("collision_count", -1)),
        weeks=24,
    ):
        failures.append(f"W24 collision certificate invalid: {failure}")
    if not all(
        _is_sha256(summary.get(key))
        for key in (
            "raw_cache_sha256",
            "checkpoint_metadata_sha256",
            "score_table_sha256",
        )
    ):
        failures.append("W24 raw checkpoint provenance is incomplete")
    if summary.get("raw_cache_required_in_git") is not False:
        failures.append("W24 raw checkpoint must not be a hidden Git dependency")
    content = dict(audit)
    content_digest = content.pop("audit_content_sha256", None)
    if content_digest != _json_digest(content):
        failures.append("W24 profile/state audit content digest mismatch")
    return failures


def validate_acceleration_authorization(
    path: Path,
    weeks: int,
    *,
    expected_git_commit: str | None = None,
) -> dict[str, Any]:
    """Validate the dedicated pre-run bundle; no certificate self-authorizes W24."""
    payload = json.loads(path.read_text())
    failures: list[str] = []
    dependencies: list[dict[str, Any]] = []
    if weeks != 24:
        failures.append("scientific full-frontier authorization is W24 only")
    if payload.get("schema_version") != AUTHORIZATION_SCHEMA_VERSION:
        failures.append("authorization schema mismatch")
    if payload.get("authorization_scope") not in {
        "primary_bound_only",
        "full_guardrail_frontier",
    }:
        failures.append("authorization scope mismatch")
    for key in (
        "execution_authorized",
        "primary_bound_batch_authorized",
        "reduced_horizon_key_v4_certified",
        "full_horizon_primary_acceleration_authorized",
        "primary_frontier_exactness_required",
        "original_runner_replay_required",
        "resource_semantics_frozen",
    ):
        if payload.get(key) is not True:
            failures.append(f"authorization field {key} is not true")
    for key in (
        "material_hpi_promotion_authorized",
        "learner_authorized",
        "paper3_authorized",
    ):
        if payload.get(key) is not False:
            failures.append(f"authorization field {key} is not false")
    if payload.get("key_schema_version") != KEY_SCHEMA_VERSION:
        failures.append("authorization key schema mismatch")
    required_commit = expected_git_commit or _git_value("rev-parse", "HEAD")
    if payload.get("git_commit") != required_commit:
        failures.append("authorization git commit mismatch")
    if payload.get("contract_sha256") != _file_sha256(PRIMARY_CONTRACT_PATH):
        failures.append("authorization primary-bound contract hash mismatch")
    if payload.get("runner_sha256") != _file_sha256(Path(__file__)):
        failures.append("authorization frontier-runner hash mismatch")
    if payload.get("isolated_bootstrap_sha256") != _file_sha256(
        ISOLATED_BOOTSTRAP
    ):
        failures.append("authorization isolated-bootstrap hash mismatch")
    for key in (
        "host_runtime_sha256",
        "portable_runtime_sha256",
        "scientific_child_environment_sha256",
        "execution_nonce",
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", str(payload.get(key, ""))):
            failures.append(f"authorization runtime identity is malformed: {key}")
    if payload.get("environment_sha256") != certification_environment().get(
        "environment_sha256"
    ):
        failures.append("authorization environment digest mismatch")

    certifications = payload.get("reduced_horizon_certification_artifacts")
    if not isinstance(certifications, list):
        failures.append("reduced-horizon artifact list is missing")
        certifications = []
    roles = {
        row.get("role") for row in certifications if isinstance(row, dict)
    }
    expected_roles = {"w12_five_tape", "w16_hard_tape"}
    if roles != expected_roles or len(certifications) != 2:
        failures.append("reduced-horizon roles must be exactly W12 and W16")
    for row in certifications:
        if not isinstance(row, dict):
            failures.append("malformed reduced-horizon artifact row")
            continue
        role = row.get("role")
        try:
            artifact = _repo_artifact(row.get("path"))
            digest = _file_sha256(artifact)
            reduced = json.loads(artifact.read_text())
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            failures.append(f"cannot validate {role}: {exc}")
            continue
        dependencies.append(
            {"role": role, "path": str(artifact), "sha256": digest}
        )
        if digest != row.get("sha256"):
            failures.append(f"reduced-horizon hash mismatch: {role}")
        failures.extend(
            validate_reduced_certification_structure(
                reduced,
                str(role),
                expected_environment_sha256=payload.get("environment_sha256"),
            )
        )
        pair_row = row.get("independent_execution_verification")
        if not isinstance(pair_row, dict):
            failures.append(
                f"independent execution verification is missing: {role}"
            )
        else:
            try:
                pair_artifact = _repo_artifact(pair_row.get("path"))
                pair_digest = _file_sha256(pair_artifact)
                json.loads(pair_artifact.read_text())
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                failures.append(
                    f"cannot validate independent execution verification {role}: {exc}"
                )
            else:
                dependencies.append(
                    {
                        "role": f"{role}_independent_execution_verification",
                        "path": str(pair_artifact),
                        "sha256": pair_digest,
                    }
                )
                if pair_digest != pair_row.get("sha256"):
                    failures.append(
                        f"independent execution verification hash mismatch: {role}"
                    )
                producer_fingerprint = pair_row.get(
                    "producer_public_key_fingerprint"
                )
                independent_fingerprint = pair_row.get(
                    "independent_public_key_fingerprint"
                )
                if not re.fullmatch(r"[0-9a-f]{64}", str(producer_fingerprint)) or not re.fullmatch(
                    r"[0-9a-f]{64}", str(independent_fingerprint)
                ):
                    failures.append(
                        f"caller-retained pair fingerprints are missing: {role}"
                    )
                else:
                    archive_reverification = reverify_authorized_reduced_pair_archive(
                        archive_root=ROOT,
                        pair_verification_path=pair_artifact,
                        expected_pair_verification_sha256=str(
                            pair_row.get("sha256")
                        ),
                        verification_mode=str(
                            pair_row.get("verification_mode", "")
                        ),
                        role=str(role),
                        certified_artifact_sha256=str(row.get("sha256")),
                        expected_producer_public_key_fingerprint=str(
                            producer_fingerprint
                        ),
                        expected_independent_public_key_fingerprint=str(
                            independent_fingerprint
                        ),
                        expected_producer_manifest_sha256=pair_row.get(
                            "producer_manifest_sha256"
                        ),
                        expected_independent_manifest_sha256=pair_row.get(
                            "independent_manifest_sha256"
                        ),
                    )
                    failures.extend(
                        f"archived pair reverification: {failure}"
                        for failure in archive_reverification.get("failures", [])
                    )
                    if archive_reverification.get("passed") is not True:
                        failures.append(
                            f"archived pair reverification failed: {role}"
                        )
        if row.get("source_git_commit") != reduced.get("provenance", {}).get(
            "git_commit"
        ):
            failures.append(f"authorization does not pin reduced source commit: {role}")

    audit_row = payload.get("w24_profile_state_audit")
    if not isinstance(audit_row, dict):
        failures.append("W24 profile/state audit is missing")
    else:
        try:
            artifact = _repo_artifact(audit_row.get("path"))
            digest = _file_sha256(artifact)
            audit = json.loads(artifact.read_text())
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            failures.append(f"cannot validate W24 profile/state audit: {exc}")
        else:
            dependencies.append(
                {"role": "w24_profile_state_audit", "path": str(artifact), "sha256": digest}
            )
            if digest != audit_row.get("sha256"):
                failures.append("W24 profile/state audit hash mismatch")
            failures.extend(validate_w24_profile_state_audit_payload(audit))
            if not str(audit.get("git_head", "")).strip():
                failures.append("W24 profile/state audit source commit is missing")
            if audit_row.get("source_git_commit") != audit.get("git_head"):
                failures.append("authorization does not pin the W24 audit source commit")

    return {
        "path": str(path.resolve()),
        "sha256": _file_sha256(path),
        "schema_version": payload.get("schema_version"),
        "key_schema_version": payload.get("key_schema_version"),
        "contract_sha256": payload.get("contract_sha256"),
        "environment_sha256": payload.get("environment_sha256"),
        "dependencies": dependencies,
        "authorized": not failures,
        "failures": failures,
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _fraction_from_payload(payload: dict[str, Any]) -> Fraction:
    return Fraction(int(payload["numerator"]), int(payload["denominator"]))


def _is_sha256(value: Any) -> bool:
    try:
        return len(str(value)) == 64 and int(str(value), 16) >= 0
    except (TypeError, ValueError):
        return False


def _tracked_head_json(path: Path) -> tuple[dict[str, Any], str, str]:
    """Read JSON only when its working bytes are tracked and identical to HEAD."""
    resolved = path.resolve()
    try:
        relative = str(resolved.relative_to(ROOT.resolve()))
    except ValueError as exc:
        raise ValueError("calibration provenance artifact escapes repository") from exc
    if not resolved.is_file():
        raise ValueError(f"calibration provenance artifact is missing: {relative}")
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )
    if tracked.returncode != 0:
        raise ValueError(f"calibration provenance artifact is not tracked: {relative}")
    head = subprocess.run(
        ["git", "show", f"HEAD:{relative}"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    working = resolved.read_bytes()
    if head.returncode != 0 or head.stdout != working:
        raise ValueError(f"calibration provenance artifact differs from HEAD: {relative}")
    try:
        payload = json.loads(working)
    except json.JSONDecodeError as exc:
        raise ValueError(f"calibration provenance artifact is invalid JSON: {relative}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"calibration provenance artifact is not an object: {relative}")
    return payload, sha256(working).hexdigest(), relative


def _resolve_tracked_json(
    recorded_path: Any,
    expected_sha256: Any,
    *,
    result_path: Path,
    role: str,
) -> tuple[dict[str, Any], str, str]:
    """Resolve transported absolute paths by content, then require tracked HEAD bytes."""
    recorded = Path(str(recorded_path))
    candidates: list[Path] = []
    if recorded.is_absolute():
        candidates.append(recorded)
    else:
        candidates.append(ROOT / recorded)
    if role == "runner_manifest":
        candidates.append(result_path.parent / recorded.name)
    elif role == "authorization":
        for parent in result_path.parents:
            candidates.append(parent / recorded.name)
            if parent == ROOT:
                break
    parts = recorded.parts
    for marker in ("contracts", "results", "scripts", "supply_chain"):
        if marker in parts:
            candidates.append(ROOT.joinpath(*parts[parts.index(marker) :]))
            break
    seen: set[Path] = set()
    errors: list[str] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if not resolved.is_file() or (
            expected_sha256 is not None
            and _file_sha256(resolved) != expected_sha256
        ):
            continue
        try:
            return _tracked_head_json(resolved)
        except ValueError as exc:
            errors.append(str(exc))
    detail = "; ".join(errors) if errors else "no matching tracked file"
    raise ValueError(f"cannot resolve tracked {role}: {detail}")


def _contract_seed_rows(contract: dict[str, Any], block_name: str) -> list[dict[str, Any]]:
    block = contract["seed_blocks"][block_name]
    start, end, count = int(block["start"]), int(block["end"]), int(block["n"])
    if end - start + 1 != count:
        raise ValueError(f"contract seed block {block_name} is internally inconsistent")
    offset = start if block_name == "calibration" else start - 1
    split = "calibration" if block_name == "calibration" else "locked"
    return [
        {
            "seed": seed,
            "context_0": CONTEXTS[(seed - offset) % len(CONTEXTS)],
            "split": split,
        }
        for seed in range(start, end + 1)
    ]


def _dependency_identities(rows: Any) -> list[tuple[str, str, str]]:
    """Compare transported dependency records without binding to machine roots."""
    identities: list[tuple[str, str, str]] = []
    if not isinstance(rows, list):
        return identities
    for row in rows:
        if not isinstance(row, dict):
            continue
        parts = Path(str(row.get("path", ""))).parts
        relative = ""
        for marker in ("contracts", "results", "scripts", "supply_chain"):
            if marker in parts:
                relative = str(Path(*parts[parts.index(marker) :]))
                break
        identities.append(
            (str(row.get("role", "")), relative, str(row.get("sha256", "")))
        )
    return sorted(identities)


def _load_calibration_winner(
    path: Path,
    weeks: int,
) -> tuple[tuple[int, ...], dict[str, Any]]:
    payload, result_sha256, result_relative = _tracked_head_json(path)
    contract = json.loads(PRIMARY_CONTRACT_PATH.read_text())
    expected_rows = _contract_seed_rows(contract, "calibration")
    failures: list[str] = []
    resolved = payload.get("resolved_frontier", {})
    aggregate = resolved.get("aggregate") or {}
    winners = aggregate.get("winner_indices") or []
    raw_tie_audit = payload.get("calibration_tie_break_audit")
    tie_audit = raw_tie_audit if isinstance(raw_tie_audit, dict) else {}
    selected_index = tie_audit.get("selected_calendar_index")
    audit_digest = tie_audit.get("audit_sha256")
    audit_body = dict(tie_audit) if isinstance(tie_audit, dict) else {}
    audit_body.pop("audit_sha256", None)
    expected_contract_sha = _file_sha256(PRIMARY_CONTRACT_PATH)
    checks = {
        "scientific_status": payload.get("scientific_status")
        == "EXACT_PRIMARY_FRONTIER_CERTIFIED_SELECTED_REPLAYS_PASSED",
        "phase": payload.get("phase") == "calibration",
        "weeks": payload.get("weeks") == weeks,
        "contract_id": payload.get("primary_contract_id")
        == contract.get("contract_id"),
        "contract_hash": payload.get("primary_contract_sha256") == expected_contract_sha,
        "phase_complete": payload.get("phase_execution_complete") is True,
        "exact_maximum": payload.get("exact_maximum_certified") is True,
        "resolved_exact": resolved.get("exact_maximum_certified") is True,
        "resolved_objective_scope": resolved.get("objective_scope")
        == "aggregate_only",
        "primary_winners_present": bool(winners),
        "tie_break_passed": tie_audit.get("passed") is True,
        "tie_break_digest": audit_digest == _json_digest(audit_body),
        "tie_break_contract": tie_audit.get("contract_section_sha256")
        == _json_digest(contract.get("calibration_tie_break", {})),
        "selected_tie": selected_index in set(map(int, winners)),
        "selected_index_matches": payload.get("selected_calibration_index")
        == selected_index,
        "frontier_scope": payload.get("frontier_scope")
        == "unconstrained_primary_metric",
        "top_level_not_guardrail_frontier": payload.get(
            "guardrail_constrained_frontier_certified"
        )
        is False,
        "not_guardrail_frontier": tie_audit.get(
            "guardrail_constrained_frontier_certified"
        )
        is False,
        "selected_replay_complete": payload.get("selected_replay_complete") is True,
        "scientific_run": payload.get("scientific_run") is True,
    }
    failures.extend(key for key, passed in checks.items() if not passed)
    calendar_count = int(contract["physics"]["effective_calendar_count"])
    screening = payload.get("screening", {})
    if (
        screening.get("objective_scope") != "aggregate_only"
        or
        payload.get("calendar_index", {}).get("calendar_count") != calendar_count
        or screening.get("pass1_count") != calendar_count
        or screening.get("pass2_count") != calendar_count
        or screening.get("passes_identical") is not True
        or not isinstance(screening.get("aggregate_contender_count"), int)
        or screening.get("aggregate_contender_count", 0) < 1
        or screening.get("per_tape_contender_counts") != []
        or screening.get("contender_overflow", {}).get("aggregate")
    ):
        failures.append("calibration complete-frontier screen is missing or incomplete")

    transducers = payload.get("transducers")
    if not isinstance(transducers, list) or len(transducers) != len(expected_rows):
        failures.append("calibration transducer block is incomplete")
        transducers = []
    transducer_by_seed = {
        int(row.get("seed", -1)): row for row in transducers if isinstance(row, dict)
    }
    if (
        len(transducer_by_seed) != len(expected_rows)
        or [row.get("seed") for row in transducers]
        != [row["seed"] for row in expected_rows]
    ):
        failures.append("calibration transducer seeds are duplicated or missing")
    if resolved.get("per_tape") != []:
        failures.append("calibration unexpectedly contains per-tape oracle selection")

    manifest_recorded = payload.get("runner_manifest")
    if not manifest_recorded:
        failures.append("runner manifest path is missing")
        manifest = {}
        manifest_sha256 = None
        manifest_relative = None
    else:
        try:
            manifest, manifest_sha256, manifest_relative = _resolve_tracked_json(
                manifest_recorded,
                None,
                result_path=path,
                role="runner_manifest",
            )
        except ValueError:
            manifest = {}
            manifest_sha256 = None
            manifest_relative = None
            failures.append("tracked runner manifest is missing or modified")

    seed_manifest = manifest.get("seed_manifest") if isinstance(manifest, dict) else None
    if not isinstance(seed_manifest, list) or len(seed_manifest) != len(expected_rows):
        failures.append("runner manifest does not contain exactly 60 calibration tapes")
        seed_manifest = []
    for expected, actual in zip(expected_rows, seed_manifest):
        if not isinstance(actual, dict):
            failures.append("malformed calibration seed row")
            continue
        seed = expected["seed"]
        tape_hash = actual.get("tape_sha256")
        if (
            actual.get("seed") != seed
            or actual.get("context_0") != expected["context_0"]
            or actual.get("split") != expected["split"]
            or not _is_sha256(tape_hash)
        ):
            failures.append(f"calibration seed/context/tape row mismatch: {seed}")
            continue
        transducer = transducer_by_seed.get(seed, {})
        if transducer.get("tape_sha256") != tape_hash:
            failures.append(f"calibration transducer tape hash mismatch: {seed}")
    tape_by_seed = {
        int(row.get("seed", -1)): row.get("tape_sha256")
        for row in seed_manifest
        if isinstance(row, dict)
    }
    if len(set(tape_by_seed.values())) != len(expected_rows):
        failures.append("calibration tape hashes are duplicated")

    aggregate_replays = [
        row
        for row in payload.get("selected_replays", [])
        if isinstance(row, dict) and row.get("role") == "calibration_aggregate_winner"
    ]
    expected_replay_pairs = {
        (int(winner), row["seed"])
        for winner in winners
        for row in expected_rows
    }
    actual_replay_pairs = [
        (int(row.get("calendar_index", -1)), int(row.get("seed", -1)))
        for row in aggregate_replays
    ]
    if (
        len(actual_replay_pairs) != len(expected_replay_pairs)
        or set(actual_replay_pairs) != expected_replay_pairs
        or tie_audit.get("required_replay_count") != len(expected_replay_pairs)
        or tie_audit.get("observed_replay_count") != len(actual_replay_pairs)
        or tie_audit.get("primary_tie_indices") != sorted(map(int, winners))
        or tie_audit.get("primary_tie_indices_sha256")
        != _json_digest(sorted(map(int, winners)))
    ):
        failures.append("calibration aggregate-winner replay block is incomplete")
    all_replays = payload.get("selected_replays", [])
    if not isinstance(all_replays, list) or any(
        not isinstance(row, dict)
        or row.get("role") != "calibration_aggregate_winner"
        or row.get("primary_exact_match") is not True
        or row.get("active_sequence_matches") is not True
        or row.get("resource_semantics_match") is not True
        for row in all_replays
    ):
        failures.append("calibration selected replay set contains an invalid row")
    for row in all_replays:
        if not isinstance(row, dict):
            continue
        seed = int(row.get("seed", -1))
        if row.get("tape_sha256") != tape_by_seed.get(seed):
            failures.append(f"calibration replay tape hash mismatch: {seed}")
    if isinstance(all_replays, list):
        recomputed_tie_audit = resolve_calibration_tie_break(
            resolved,
            all_replays,
            contract=contract,
        )
        if (
            recomputed_tie_audit.get("passed") is not True
            or recomputed_tie_audit != tie_audit
        ):
            failures.append("calibration tie-break audit does not revalidate")

    input_artifacts = manifest.get("input_artifacts", {}) if isinstance(manifest, dict) else {}
    authorization_row = input_artifacts.get("authorization") or {}
    authorization_sha = authorization_row.get("sha256")
    try:
        authorization_payload, actual_auth_sha, authorization_relative = _resolve_tracked_json(
            authorization_row.get("path"),
            authorization_sha,
            result_path=path,
            role="authorization",
        )
    except (TypeError, ValueError) as exc:
        failures.append(f"tracked calibration authorization is invalid: {exc}")
        authorization_payload = {}
        actual_auth_sha = None
        authorization_relative = None

    source_commit = manifest.get("git_head") if isinstance(manifest, dict) else None
    if not source_commit or authorization_payload.get("git_commit") != source_commit:
        failures.append("calibration manifest/authorization source commit mismatch")
    commit_exists = (
        subprocess.run(
            ["git", "cat-file", "-e", f"{source_commit}^{{commit}}"],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
        if source_commit
        else False
    )
    if not commit_exists:
        failures.append("calibration source commit is unavailable")
    authorization_validation = (
        validate_acceleration_authorization(
            ROOT / authorization_relative,
            weeks,
            expected_git_commit=source_commit,
        )
        if authorization_relative and source_commit
        else {"authorized": False, "failures": ["authorization unavailable"], "dependencies": []}
    )
    if authorization_validation.get("authorized") is not True or authorization_validation.get("failures"):
        failures.append("calibration authorization does not revalidate")
    result_authorization = payload.get("acceleration_authorization", {})
    if (
        result_authorization.get("authorized") is not True
        or result_authorization.get("failures")
        or result_authorization.get("sha256") != actual_auth_sha
        or authorization_sha != actual_auth_sha
        or result_authorization.get("schema_version")
        != authorization_validation.get("schema_version")
        or result_authorization.get("key_schema_version") != KEY_SCHEMA_VERSION
        or result_authorization.get("contract_sha256") != expected_contract_sha
        or _dependency_identities(result_authorization.get("dependencies", []))
        != _dependency_identities(authorization_validation.get("dependencies", []))
    ):
        failures.append("calibration result authorization digest/status mismatch")
    if input_artifacts.get("primary_contract", {}).get("sha256") != expected_contract_sha:
        failures.append("runner manifest primary-contract hash mismatch")
    if manifest.get("result_sha256") != result_sha256:
        failures.append("runner manifest result hash mismatch")
    manifest_checks = {
        "manifest_schema": manifest.get("schema_version") == MANIFEST_SCHEMA_VERSION,
        "manifest_clean_source": manifest.get("git_status_sha256") == _json_digest(""),
        "manifest_phase_complete": manifest.get("phase_execution_complete") is True,
        "manifest_exact": manifest.get("exact_maximum_certified") is True,
        "manifest_key": manifest.get("key_schema_version") == KEY_SCHEMA_VERSION,
        "manifest_invoked": manifest.get("full_execution_was_explicitly_invoked") is True,
    }
    failures.extend(key for key, passed in manifest_checks.items() if not passed)
    command = manifest.get("command")
    if (
        not isinstance(command, list)
        or "--phase" not in command
        or command[command.index("--phase") + 1 : command.index("--phase") + 2]
        != ["calibration"]
        or "--authorization" not in command
        or "--weeks" not in command
        or command[command.index("--weeks") + 1 : command.index("--weeks") + 2]
        != [str(weeks)]
    ):
        failures.append("calibration runner command manifest is incomplete")

    expected_code_paths = (
        PRIMARY_CONTRACT_PATH,
        Path(__file__),
        TRANSDUCER_RUNNER_PATH,
        ROOT / "supply_chain" / "paper2_bottleneck.py",
        ROOT / "supply_chain" / "episode_metrics.py",
    )
    expected_code = {
        str(code_path.relative_to(ROOT)): _file_sha256(code_path)
        for code_path in expected_code_paths
    }
    if manifest.get("code_sha256") != expected_code:
        failures.append("calibration code/dependency hashes drifted")
    if commit_exists:
        for relative, digest in expected_code.items():
            blob = subprocess.run(
                ["git", "show", f"{source_commit}:{relative}"],
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if blob.returncode != 0 or sha256(blob.stdout).hexdigest() != digest:
                failures.append(f"calibration source-commit blob mismatch: {relative}")
    manifest_dependencies = input_artifacts.get("authorization_dependencies", [])
    if _dependency_identities(manifest_dependencies) != _dependency_identities(
        authorization_validation.get("dependencies", [])
    ):
        failures.append("calibration authorization dependency manifest mismatch")

    result_checkpoints = payload.get("build", {}).get("checkpoints", [])
    manifest_checkpoints = manifest.get("checkpoint_artifacts", [])
    def checkpoint_rows(rows: Any) -> list[tuple[Any, Any, Any]]:
        if not isinstance(rows, list):
            return []
        return [
            (row.get("seed"), row.get("data_sha256"), row.get("score_table_sha256"))
            for row in rows
            if isinstance(row, dict)
        ]
    if (
        len(checkpoint_rows(result_checkpoints)) != len(expected_rows)
        or checkpoint_rows(result_checkpoints) != checkpoint_rows(manifest_checkpoints)
        or [seed for seed, _data, _score in checkpoint_rows(result_checkpoints)]
        != [row["seed"] for row in expected_rows]
        or any(
            not _is_sha256(data_sha) or not _is_sha256(score_sha)
            for _seed, data_sha, score_sha in checkpoint_rows(result_checkpoints)
        )
    ):
        failures.append("calibration checkpoint manifest is incomplete or inconsistent")

    if failures:
        raise ValueError("calibration result failed closed: " + "; ".join(sorted(set(failures))))
    return calendar_at_index(int(selected_index), weeks), {
        "path": str(path.resolve()),
        "relative_path": result_relative,
        "sha256": result_sha256,
        "winner_index": int(selected_index),
        "primary_tie_count": len(winners),
        "tie_break_audit_sha256": audit_digest,
        "runner_manifest_relative": manifest_relative,
        "runner_manifest_sha256": manifest_sha256,
        "authorization_relative": authorization_relative,
        "authorization_sha256": actual_auth_sha,
        "source_git_commit": source_commit,
        "seed_manifest_sha256": _json_digest(seed_manifest),
    }


def _paired_h_pi_inference(
    rows: Sequence[dict[str, Any]],
    *,
    contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the frozen paired percentile bootstrap on exact locked-tape H_PI rows."""
    contract = contract or json.loads(PRIMARY_CONTRACT_PATH.read_text())
    expected = _contract_seed_rows(contract, "locked_bound")
    if len(rows) != len(expected):
        raise ValueError(
            f"paired H_PI requires exactly {len(expected)} locked rows, got {len(rows)}"
        )
    deltas: list[Fraction] = []
    seen_seeds: set[int] = set()
    seen_tapes: set[str] = set()
    for expected_row, row in zip(expected, rows):
        seed = int(row.get("seed", -1))
        tape_sha = str(row.get("tape_sha256", ""))
        if seed != expected_row["seed"]:
            raise ValueError(f"paired H_PI seed/order mismatch: {seed}")
        if seed in seen_seeds or not _is_sha256(tape_sha) or tape_sha in seen_tapes:
            raise ValueError(f"paired H_PI duplicate or invalid tape identity: {seed}")
        seen_seeds.add(seed)
        seen_tapes.add(tape_sha)
        try:
            oracle = _fraction_from_payload(row["oracle_score"])
            fixed = _fraction_from_payload(row["fixed_calibration_score"])
            delta = _fraction_from_payload(row["h_pi"])
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            raise ValueError(f"paired H_PI exact score payload is invalid: {seed}") from exc
        if delta != oracle - fixed:
            raise ValueError(f"paired H_PI arithmetic mismatch: {seed}")
        if delta < 0:
            raise ValueError(f"paired H_PI oracle delta is negative: {seed}")
        tie_indices = row.get("oracle_tie_indices")
        if (
            not isinstance(tie_indices, list)
            or not tie_indices
            or any(isinstance(index, bool) or not isinstance(index, int) for index in tie_indices)
            or tie_indices != sorted(set(tie_indices))
            or row.get("oracle_tie_count") != len(tie_indices)
            or row.get("oracle_tie_indices_sha256") != _json_digest(tie_indices)
            or row.get("display_representative_index") != min(tie_indices)
            or row.get("oracle_tie_selection_rule")
            != "none_all_exact_ties_retained"
            or row.get("representative_semantics")
            != "display_only_not_used_for_h_pi_or_guardrail_feasibility"
        ):
            raise ValueError(f"paired H_PI oracle tie metadata is invalid: {seed}")
        deltas.append(delta)

    inference = contract["inference"]
    resamples = int(inference["bootstrap_resamples"])
    bootstrap_seed = int(inference["bootstrap_seed"])
    confidence = float(inference["confidence_level"])
    if inference.get("paired_unit") != "tape" or resamples != 10_000:
        raise ValueError("contract paired-bootstrap specification is not the frozen protocol")
    values = np.asarray([float(value) for value in deltas], dtype=np.float64)
    rng = np.random.default_rng(bootstrap_seed)
    sample_indices = rng.integers(
        0, len(values), size=(resamples, len(values)), endpoint=False
    )
    bootstrap_means = values[sample_indices].mean(axis=1)
    alpha = 1.0 - confidence
    lcb, ucb = np.quantile(
        bootstrap_means,
        [alpha / 2.0, 1.0 - alpha / 2.0],
        method="linear",
    )
    mean_exact = sum(deltas, Fraction(0)) / len(deltas)
    gate = float(contract["metric"]["practical_gate"])
    if float(ucb) < gate:
        decision = "BOUNDARY_CLOSE_H_PI_UCB95_BELOW_PRACTICAL_GATE"
    elif float(lcb) >= gate:
        decision = "MATERIAL_H_PI_DIAGNOSTIC_ONLY_NO_PROMOTION"
    else:
        decision = "AMBIGUOUS_H_PI_FAMILY_REMAINS_ACTIVE"
    return {
        "inference_status": "COMPUTED_PREREGISTERED_PAIRED_PERCENTILE_BOOTSTRAP",
        "paired_unit": inference["paired_unit"],
        "n_tapes": len(rows),
        "bootstrap_resamples": resamples,
        "bootstrap_seed": bootstrap_seed,
        "confidence_level": confidence,
        "bootstrap_method": "paired_tape_percentile_with_replacement_numpy_quantile_linear",
        "mean": float(mean_exact),
        "mean_exact": _fraction_payload(mean_exact),
        "lcb95": float(lcb),
        "ucb95": float(ucb),
        "practical_gate": gate,
        "boundary_decision": decision,
        "oracle_tie_handling": "all_exact_ties_retained_no_guardrail_selection",
        "material_hpi_promotion_authorized": False,
        "learner_authorized": False,
        "paper3_authorized": False,
    }


def assemble_w24_profile_state_audit(
    compiled: ScoreTransducer,
    checkpoint_metadata: dict[str, Any],
    proof: dict[str, Any],
    tape: dict[str, Any],
) -> dict[str, Any]:
    """Assemble a generated profile audit; this artifact is not authorization."""
    inventory = proof["state_inventory"]
    unknown_mutable = sorted(
        set(inventory.get("unclassified_live_attributes", []))
        | set(inventory.get("static_live_reads_unclassified", []))
    )
    unknown_callbacks = int(proof.get("unknown_callback_owner_count", -1))
    spec = BuildSpec(
        index=0,
        seed=1_110_001,
        context=CONTEXTS[0],
        split="w24_profile_state_audit_burned",
        weeks=24,
    )
    fingerprint = _build_fingerprint(spec)
    data_path = Path(str(checkpoint_metadata.get("data_path", "")))
    raw_cache_matches = bool(
        data_path.is_file()
        and checkpoint_metadata.get("data_sha256") == _file_sha256(data_path)
        and checkpoint_metadata.get("data_bytes") == data_path.stat().st_size
    )
    source_proof = checkpoint_metadata.get("source_transducer_proof")
    source_proof_matches = source_proof == _compiled_transducer_proof(compiled)
    fingerprint_matches = bool(
        checkpoint_metadata.get("fingerprint_sha256") == _json_digest(fingerprint)
        and all(checkpoint_metadata.get(key) == value for key, value in fingerprint.items())
    )
    layer_wide_callbacks = bool(
        compiled.semantic_key_evaluations == compiled.prefix_replays
        and len(compiled.layer_semantic_key_evaluations) == 24
        and sum(compiled.layer_semantic_key_evaluations) == compiled.prefix_replays
        and len(compiled.layer_callback_inventory) == 24
        and all(compiled.layer_callback_inventory)
        and len(compiled.transition_counts_by_layer) == 23
        and compiled.prefixes_with_nonempty_callback_inventory
        == compiled.prefix_replays
        and sum(compiled.layer_prefixes_with_nonempty_callback_inventory)
        == compiled.prefix_replays
        and _is_sha256(compiled.prefix_callback_records_sha256)
        and len(compiled.layer_prefix_callback_records_sha256) == 24
        and all(_is_sha256(row) for row in compiled.layer_prefix_callback_records_sha256)
    )
    collision_bisimulation = compiled.collision_bisimulation or {}
    collision_bisimulation_passed = bool(
        collision_bisimulation.get("passed") is True
        and collision_bisimulation.get("key_schema_version") == KEY_SCHEMA_VERSION
        and collision_bisimulation.get("complete_state_serialization") is True
        and collision_bisimulation.get("event_payload_serialized") is True
        and collision_bisimulation.get("resource_users_serialized") is True
        and collision_bisimulation.get("callback_closure_state_serialized") is True
        and collision_bisimulation.get(
            "process_target_state_serialized_or_fail_closed"
        ) is True
        and collision_bisimulation.get("runtime_alias_graph_serialized") is True
        and collision_bisimulation.get("collision_payload_checks")
        == compiled.collision_count
        and collision_bisimulation.get("collision_root_count")
        == compiled.collision_count
        and collision_bisimulation.get("unresolved_node_obligation_count") == 0
        and collision_bisimulation.get("unresolved_collision_root_count") == 0
        and collision_bisimulation.get("all_actions_covered") is True
        and collision_bisimulation.get("backward_induction_complete") is True
        and not collision_bisimulation.get("mismatch_examples")
        and _is_sha256(collision_bisimulation.get("transition_record_sha256"))
        and not validate_collision_bisimulation_certificate(
            collision_bisimulation,
            expected_collision_count=compiled.collision_count,
            weeks=compiled.weeks,
        )
    )
    passed = bool(
        compiled.weeks == 24
        and compiled.seed == 1_110_001
        and compiled.tape_sha256 == tape["threat_sha256"]
        and raw_cache_matches
        and source_proof_matches
        and fingerprint_matches
        and layer_wide_callbacks
        and checkpoint_metadata.get("score_table_sha256") == compiled.table_sha256
        and inventory.get("classification_complete") is True
        and inventory.get("all_frozen_invariants_hold") is True
        and not unknown_mutable
        and unknown_callbacks == 0
        and collision_bisimulation_passed
    )
    checkpoint_summary = {
        "raw_cache_locator": "separate_execution_artifact_manifest",
        "raw_cache_sha256": checkpoint_metadata.get("data_sha256"),
        "raw_cache_bytes": checkpoint_metadata.get("data_bytes"),
        "checkpoint_metadata_sha256": _json_digest(checkpoint_metadata),
        "raw_cache_required_in_git": False,
        "raw_cache_required_for_pre_run_authorization": False,
        "score_table_sha256": compiled.table_sha256,
        "state_counts": compiled.state_counts,
        "prefix_replays": compiled.prefix_replays,
        "collision_count": compiled.collision_count,
        "source_transducer_proof": source_proof,
        "raw_cache_verified_at_audit_generation": raw_cache_matches,
        "checkpoint_fingerprint_verified_at_audit_generation": fingerprint_matches,
    }
    checkpoint_summary["summary_sha256"] = _json_digest(checkpoint_summary)
    payload = {
        "schema_version": W24_AUDIT_SCHEMA_VERSION,
        "generated_not_authorization": True,
        "weeks": 24,
        "seed": 1_110_001,
        "split": "w24_profile_state_audit_burned",
        "tape_sha256": tape["threat_sha256"],
        "key_schema_version": KEY_SCHEMA_VERSION,
        "primary_contract_sha256": _file_sha256(PRIMARY_CONTRACT_PATH),
        "generated_by_frontier_runner_sha256": _file_sha256(Path(__file__)),
        "transducer_runner_sha256": _file_sha256(TRANSDUCER_RUNNER_PATH),
        "git_head": _git_value("rev-parse", "HEAD"),
        "dependency_sha256": fingerprint["dependency_sha256"],
        "environment": fingerprint["environment"],
        "checkpoint_summary": checkpoint_summary,
        "state_inventory": inventory,
        "callback_inventory": proof.get("callback_inventory", []),
        "profile_audit_passed": passed,
        "semantic_state_inventory_complete": bool(
            inventory.get("classification_complete")
            and inventory.get("all_frozen_invariants_hold")
        ),
        "unknown_mutable_state_count": len(unknown_mutable),
        "unknown_mutable_state": unknown_mutable,
        "unclassified_callback_owner_count": unknown_callbacks,
        "layer_wide_callback_audit_passed": layer_wide_callbacks,
        "collision_bisimulation": collision_bisimulation,
        "collision_bisimulation_passed": collision_bisimulation_passed,
    }
    payload["audit_content_sha256"] = _json_digest(payload)
    return payload


def write_w24_profile_state_audit(
    *,
    output: Path,
    checkpoint_dir: Path,
    progress_path: Path,
    workers: int,
) -> dict[str, Any]:
    """Build burned seed 1110001 at W24 and emit a machine-derived audit."""
    started = time.perf_counter()
    spec = BuildSpec(
        index=0,
        seed=1_110_001,
        context=CONTEXTS[0],
        split="w24_profile_state_audit_burned",
        weeks=24,
    )
    compiled, metadata = build_score_transducers(
        [spec],
        workers=workers,
        checkpoint_dir=checkpoint_dir,
        progress_path=progress_path,
        started=started,
    )
    tape = materialize_tape(spec.seed, spec.context, spec.split, weeks=24)
    proof = runtime_proof_audit(tape)
    audit = assemble_w24_profile_state_audit(
        compiled[0], metadata[0], proof, tape
    )
    _atomic_json(output, audit)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("calibration", "locked"))
    parser.add_argument("--weeks", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=65_536)
    parser.add_argument("--max-contenders", type=int, default=100_000)
    parser.add_argument(
        "--authorization",
        "--certification",
        dest="authorization",
        type=Path,
        help="Dedicated W24 authorization bundle (legacy option spelling retained).",
    )
    parser.add_argument("--calibration-result", type=Path)
    parser.add_argument("--build-workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument(
        "--write-w24-audit",
        type=Path,
        help="Generate, but do not authorize, the burned-seed W24 profile/state audit.",
    )
    parser.add_argument(
        "--non-scientific-smoke-seed",
        type=int,
        help="One burned seed, W6 or shorter; exercises the CLI but cannot certify science.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--progress", type=Path)
    args = parser.parse_args()
    if args.weeks < 1 or args.weeks > 24:
        parser.error("--weeks must be in [1, 24]")
    if args.build_workers < 1:
        parser.error("--build-workers must be positive")

    if args.write_w24_audit is not None:
        if args.phase is not None or args.output is not None:
            parser.error("--write-w24-audit is a separate mode; omit --phase/--output")
        if args.weeks != 24:
            parser.error("W24 profile/state audit requires --weeks 24")
        if args.authorization is not None or args.calibration_result is not None:
            parser.error("W24 audit generation does not consume scientific authorization")
        audit_checkpoint_dir = args.checkpoint_dir or args.write_w24_audit.with_suffix(
            ".checkpoints"
        )
        audit_progress = args.progress or args.write_w24_audit.with_suffix(
            ".progress.json"
        )
        audit = write_w24_profile_state_audit(
            output=args.write_w24_audit,
            checkpoint_dir=audit_checkpoint_dir,
            progress_path=audit_progress,
            workers=args.build_workers,
        )
        print(
            json.dumps(
                {
                    "profile_audit_passed": audit["profile_audit_passed"],
                    "audit_path": str(args.write_w24_audit.resolve()),
                    "audit_sha256": _file_sha256(args.write_w24_audit),
                    "generated_not_authorization": True,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if audit["profile_audit_passed"] else 1

    if args.phase is None or args.output is None:
        parser.error("frontier execution requires --phase and --output")

    scientific_run = args.non_scientific_smoke_seed is None
    if scientific_run:
        if args.weeks != 24:
            parser.error("scientific full-frontier execution requires --weeks 24")
        if args.authorization is None:
            parser.error("scientific execution requires --authorization")
        source_drift = _scientific_source_drift()
        if source_drift:
            parser.error(
                "scientific execution requires committed immutable inputs; drift: "
                + source_drift.replace("\n", " | ")
            )
        if args.phase == "locked" and args.calibration_result is None:
            parser.error("--locked requires --calibration-result")
        authorization = validate_acceleration_authorization(
            args.authorization, args.weeks
        )
        if not authorization["authorized"]:
            parser.error(
                "authorization bundle failed closed: "
                + "; ".join(authorization["failures"])
            )
    else:
        if args.phase != "calibration" or args.weeks > 6:
            parser.error("non-scientific smoke is calibration-only and limited to W6")
        if args.calibration_result is not None:
            parser.error("non-scientific smoke cannot consume a calibration result")
        authorization = {
            "authorized": False,
            "non_scientific_smoke_override": True,
            "failures": ["non-scientific smoke is never an authorization"],
            "dependencies": [],
        }

    if scientific_run and args.phase == "calibration":
        seeds = list(range(1_100_001, 1_100_061))
        offset = 1_100_001
        split = "calibration"
    elif scientific_run:
        seeds = list(range(1_110_002, 1_110_121))
        offset = 1_110_001
        split = "locked"
    else:
        seeds = [int(args.non_scientific_smoke_seed)]
        offset = seeds[0]
        split = "full_frontier_cli_smoke_burned"

    specs = [
        BuildSpec(
            index=index,
            seed=seed,
            context=CONTEXTS[(seed - offset) % len(CONTEXTS)],
            split=split,
            weeks=args.weeks,
        )
        for index, seed in enumerate(seeds)
    ]
    progress_path = args.progress or args.output.with_suffix(".progress.json")
    checkpoint_dir = args.checkpoint_dir or args.output.with_suffix(".checkpoints")
    started = time.perf_counter()
    compiled, checkpoint_metadata = build_score_transducers(
        specs,
        workers=args.build_workers,
        checkpoint_dir=checkpoint_dir,
        progress_path=progress_path,
        started=started,
    )
    certificate_coverage = collision_certificate_coverage(compiled)
    if not certificate_coverage["passed"]:
        raise RuntimeError(
            "per-tape collision-certificate coverage failed closed: "
            + "; ".join(certificate_coverage["failures"])
        )
    tapes = [
        materialize_tape(
            spec.seed, spec.context, spec.split, weeks=spec.weeks
        )
        for spec in specs
    ]

    def progress(payload: dict[str, Any]) -> None:
        _atomic_json(
            progress_path,
            {
                "stage": "calendar_screen",
                **payload,
                "elapsed_seconds": time.perf_counter() - started,
            },
        )

    objective_scope = (
        "aggregate_only" if args.phase == "calibration" else "per_tape_only"
    )
    screening = screen_frontier(
        compiled,
        batch_size=args.batch_size,
        max_contenders=args.max_contenders,
        objective_scope=objective_scope,
        progress=progress,
    )
    canonical_cache: dict[tuple[int, int], Fraction] = {}

    def canonical_score(tape_index: int, index: int) -> Fraction:
        key = (tape_index, index)
        if key not in canonical_cache:
            sequence = calendar_at_index(index, args.weeks)
            checkpoint = run_prefix(tapes[tape_index], sequence).checkpoint
            canonical_cache[key] = Fraction.from_float(
                float.fromhex(checkpoint.primary_hex)
            )
        return canonical_cache[key]

    resolved = resolve_frontier(
        compiled,
        screening,
        score_provider=canonical_score,
        acceleration_certified=bool(
            authorization["authorized"] or not scientific_run
        ),
    )

    fixed_calendar: tuple[int, ...] | None = None
    calibration_provenance: dict[str, Any] | None = None
    fixed_exact_scores: list[Fraction] | None = None
    if args.phase == "locked":
        assert args.calibration_result is not None
        fixed_calendar, calibration_provenance = _load_calibration_winner(
            args.calibration_result, args.weeks
        )
        fixed_index = calendar_index(fixed_calendar)
        fixed_exact_scores = [
            canonical_score(tape_index, fixed_index)
            for tape_index in range(len(tapes))
        ]

    replay_rows: list[dict[str, Any]] = []
    if resolved["exact_maximum_certified"]:
        if args.phase == "calibration":
            for winner_index in resolved["aggregate"]["winner_indices"]:
                sequence = calendar_at_index(winner_index, args.weeks)
                for tape_index, tape in enumerate(tapes):
                    replay_rows.append({
                        **replay_calendar(
                            tape, sequence, canonical_score(tape_index, winner_index)
                        ),
                        "role": "calibration_aggregate_winner",
                    })
        else:
            assert fixed_calendar is not None and fixed_exact_scores is not None
            for tape_index, tape in enumerate(tapes):
                replay_rows.append({
                    **replay_calendar(
                        tape, fixed_calendar, fixed_exact_scores[tape_index]
                    ),
                    "role": "fixed_calibration_comparator",
                })
            for tape_index, row in enumerate(resolved["per_tape"]):
                for winner_index in row["winner_indices"]:
                    sequence = calendar_at_index(winner_index, args.weeks)
                    replay_rows.append({
                        **replay_calendar(
                            tapes[tape_index],
                            sequence,
                            canonical_score(tape_index, winner_index),
                        ),
                        "role": "per_tape_oracle_winner",
                    })

    calibration_tie_break_audit = None
    if args.phase == "calibration" and resolved["exact_maximum_certified"]:
        if scientific_run:
            calibration_tie_break_audit = resolve_calibration_tie_break(
                resolved, replay_rows
            )
        else:
            smoke_winners = sorted(
                map(int, resolved["aggregate"]["winner_indices"])
            )
            calibration_tie_break_audit = {
                "schema_version": "non_scientific_smoke_tie_selection_v1",
                "primary_tie_indices": smoke_winners,
                "selected_calendar_index": min(smoke_winners),
                "guardrail_constrained_frontier_certified": False,
                "passed": bool(smoke_winners),
                "failures": [],
            }

    paired_h_pi: dict[str, Any] | None = None
    if (
        args.phase == "locked"
        and resolved["exact_maximum_certified"]
        and fixed_exact_scores is not None
    ):
        paired_rows = []
        for tape_index, row in enumerate(resolved["per_tape"]):
            oracle = _fraction_from_payload(row["oracle_score"])
            fixed = fixed_exact_scores[tape_index]
            delta = oracle - fixed
            paired_rows.append({
                "seed": row["seed"],
                "tape_sha256": row["tape_sha256"],
                "oracle_score": _fraction_payload(oracle),
                "fixed_calibration_score": _fraction_payload(fixed),
                "h_pi": _fraction_payload(delta),
                "oracle_tie_indices": sorted(map(int, row["winner_indices"])),
                "oracle_tie_count": len(row["winner_indices"]),
                "oracle_tie_indices_sha256": _json_digest(
                    sorted(map(int, row["winner_indices"]))
                ),
                "oracle_tie_selection_rule": "none_all_exact_ties_retained",
                "display_representative_index": min(
                    map(int, row["winner_indices"])
                ),
                "representative_semantics": (
                    "display_only_not_used_for_h_pi_or_guardrail_feasibility"
                ),
            })
        paired_h_pi = {
            "definition": "exact per-tape feasible oracle minus frozen calibration calendar",
            "rows": paired_rows,
            **_paired_h_pi_inference(paired_rows),
        }

    expected_exogenous_by_seed: dict[int, dict[str, str]] = {}
    if scientific_run:
        reference_calendar = (0,) * args.weeks
        for tape in tapes:
            reference = run_policy(
                tape,
                active_calendar_policy(reference_calendar),
            )
            expected_exogenous_by_seed[int(tape["seed"])] = {
                "tape_sha256": str(tape["threat_sha256"]),
                "consumed_base_threat_sha256": str(
                    reference["consumed_base_threat_sha256"]
                ),
                "realized_demand_sha256": str(
                    reference["realized_demand_sha256"]
                ),
                "reference_calendar": "M" * args.weeks,
            }

    selected_replay_audit = validate_selected_replay_set(
        replay_rows,
        phase=args.phase if scientific_run else "smoke",
        aggregate_winner_indices=(
            resolved["aggregate"]["winner_indices"]
            if args.phase == "calibration" and resolved.get("aggregate")
            else None
        ),
        per_tape_winner_indices=(
            {
                int(row["seed"]): row["winner_indices"]
                for row in resolved["per_tape"]
            }
            if args.phase == "locked"
            else None
        ),
        expected_exogenous_by_seed=(
            expected_exogenous_by_seed if scientific_run else None
        ),
    )
    replay_ok = selected_replay_audit["passed"] is True
    selection_complete = bool(
        calibration_tie_break_audit
        and calibration_tie_break_audit.get("passed") is True
        if args.phase == "calibration"
        else fixed_calendar is not None
    )
    phase_complete = bool(
        resolved["exact_maximum_certified"]
        and certificate_coverage["passed"] is True
        and replay_ok
        and selection_complete
        and (
            args.phase != "locked"
            or (
                paired_h_pi is not None
                and paired_h_pi.get("inference_status")
                == "COMPUTED_PREREGISTERED_PAIRED_PERCENTILE_BOOTSTRAP"
            )
        )
    )
    exact_certified = bool(scientific_run and phase_complete)
    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    result = {
        "schema_version": SCHEMA_VERSION,
        "scientific_status": (
            "EXACT_PRIMARY_FRONTIER_CERTIFIED_SELECTED_REPLAYS_PASSED"
            if exact_certified
            else "NONSCIENTIFIC_SMOKE_COMPLETE_NOT_EVIDENCE"
            if not scientific_run and phase_complete
            else "FRONTIER_FAIL_CLOSED"
        ),
        "scientific_run": scientific_run,
        "primary_contract_id": "paper2_bottleneck_primary_bound_v2",
        "primary_contract_sha256": _file_sha256(PRIMARY_CONTRACT_PATH),
        "frontier_scope": "unconstrained_primary_metric",
        "guardrail_constrained_frontier_certified": False,
        "phase": args.phase,
        "weeks": args.weeks,
        "calendar_index": {
            "schema_version": CALENDAR_INDEX_SCHEMA_VERSION,
            "zero_based": True,
            "calendar_count": feasible_calendar_count(args.weeks),
            "enumeration": "streamed_lexicographic_DFS",
            "all_calendar_tuples_materialized": False,
        },
        "acceleration_authorization": authorization,
        "build": {
            "workers": args.build_workers,
            "checkpoint_dir": str(checkpoint_dir.resolve()),
            "resumable": True,
            "checkpoints": checkpoint_metadata,
            "collision_certificate_coverage": certificate_coverage,
        },
        "transducers": [
            {
                "seed": row.seed,
                "tape_sha256": row.tape_sha256,
                "state_counts": row.state_counts,
                "prefix_replays": row.prefix_replays,
                "collision_count": row.collision_count,
                "collision_certificate_sha256": (
                    (row.collision_bisimulation or {}).get("certificate_sha256")
                ),
                "score_table_sha256": row.table_sha256,
            }
            for row in compiled
        ],
        "screening": {
            "objective_scope": screening.objective_scope,
            "pass1_count": screening.pass1_count,
            "pass2_count": screening.pass2_count,
            "pass1_stream_sha256": screening.pass1_stream_sha256,
            "pass2_stream_sha256": screening.pass2_stream_sha256,
            "passes_identical": (
                screening.pass1_stream_sha256 == screening.pass2_stream_sha256
            ),
            "aggregate_best_lower_hex": (
                screening.aggregate_best_lower.hex()
                if screening.objective_scope != "per_tape_only"
                else None
            ),
            "per_tape_best_lower_hex": [
                value.hex() for value in screening.per_tape_best_lower
            ],
            "aggregate_contender_count": len(screening.aggregate_contenders),
            "per_tape_contender_counts": [
                len(row) for row in screening.per_tape_contenders
            ],
            "contender_overflow": screening.contender_overflow,
            "roundoff_certificate": {
                "input_semantics": "every emitted ReT float treated as its exact IEEE-754 binary rational",
                "edge_labels": "tight adjacent-float enclosure of exact binary sums",
                "path_arithmetic": "np.nextafter outward after every add/divide",
                "canonical_mean_padding": "gamma_(n-1) sequential-summation bound plus division rounding",
                "resolution": "every interval-overlap contender rescored by unaccelerated run_prefix and compared as Fraction.from_float",
            },
        },
        "resolved_frontier": resolved,
        "calibration_tie_break_audit": calibration_tie_break_audit,
        "selected_calibration_index": (
            calibration_tie_break_audit.get("selected_calendar_index")
            if calibration_tie_break_audit
            else None
        ),
        "fixed_calibration_calendar": (
            None
            if fixed_calendar is None or fixed_exact_scores is None
            else {
                "index": calendar_index(fixed_calendar),
                "calendar": _action_names(fixed_calendar),
                "per_tape_scores": [
                    _fraction_payload(value) for value in fixed_exact_scores
                ],
                "calibration_result": calibration_provenance,
            }
        ),
        "paired_locked_h_pi": paired_h_pi,
        "selected_replays": replay_rows,
        "selected_replay_audit": selected_replay_audit,
        "selected_replay_complete": replay_ok,
        "calibration_tie_rule": (
            "frozen_guardrail_then_calendar_index_among_exact_primary_ties"
            if args.phase == "calibration"
            else "not_applicable"
        ),
        "phase_execution_complete": phase_complete,
        "exact_maximum_certified": exact_certified,
        "runner_manifest": str(manifest_path.resolve()),
        "elapsed_seconds": time.perf_counter() - started,
        "h_pi_computed": paired_h_pi is not None,
        "h_pi_claimed": False,
        "paper2_claimed": False,
        "paper3_authorized": False,
    }
    _atomic_json(args.output, result)

    tracked_paths = [
        PRIMARY_CONTRACT_PATH,
        Path(__file__),
        TRANSDUCER_RUNNER_PATH,
        ROOT / "supply_chain" / "paper2_bottleneck.py",
        ROOT / "supply_chain" / "episode_metrics.py",
    ]
    input_artifacts = {
        "primary_contract": {
            "path": str(PRIMARY_CONTRACT_PATH.resolve()),
            "sha256": _file_sha256(PRIMARY_CONTRACT_PATH),
        },
        "authorization": (
            None
            if args.authorization is None
            else {
                "path": str(args.authorization.resolve()),
                "sha256": _file_sha256(args.authorization),
            }
        ),
        "authorization_dependencies": authorization.get("dependencies", []),
        "calibration_result": calibration_provenance,
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": list(sys.argv),
        "cwd": str(ROOT),
        "git_head": _git_value("rev-parse", "HEAD"),
        "git_status_sha256": _json_digest(
            _git_value("status", "--porcelain=v1") or ""
        ),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "environment": certification_environment(),
        "environment_sha256": certification_environment()["environment_sha256"],
        "input_artifacts": input_artifacts,
        "code_sha256": {
            str(path.relative_to(ROOT)): _file_sha256(path)
            for path in tracked_paths
        },
        "seed_manifest": [
            {
                "seed": int(tape["seed"]),
                "context_0": tape["context_schedule"][0],
                "tape_sha256": tape["threat_sha256"],
                "split": split,
            }
            for tape in tapes
        ],
        "checkpoint_artifacts": [
            {
                "seed": row["seed"],
                "data_path": row["data_path"],
                "data_sha256": row["data_sha256"],
                "score_table_sha256": row["score_table_sha256"],
                "collision_certificate_sha256": row[
                    "source_transducer_proof"
                ]["collision_bisimulation"]["certificate_sha256"],
            }
            for row in checkpoint_metadata
        ],
        "collision_certificate_coverage_sha256": certificate_coverage[
            "coverage_sha256"
        ],
        "result_path": str(args.output.resolve()),
        "result_sha256": _file_sha256(args.output),
        "progress_path": str(progress_path.resolve()),
        "full_execution_was_explicitly_invoked": scientific_run,
        "exact_maximum_certified": exact_certified,
        "phase_execution_complete": phase_complete,
        "key_schema_version": KEY_SCHEMA_VERSION,
    }
    _atomic_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "status": result["scientific_status"],
                "exact_maximum_certified": exact_certified,
                "phase_execution_complete": phase_complete,
                "calendar_count": screening.calendar_count,
                "elapsed_seconds": result["elapsed_seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if phase_complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
