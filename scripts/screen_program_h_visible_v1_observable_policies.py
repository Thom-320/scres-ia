#!/usr/bin/env python3
"""Fail-closed Program-H visible-v1 adapter audit and policy-screen harness.

The calibration and locked blocks used here are historical burned Program-H
tapes.  This script does not open a virgin block, train a learner, or make a
confirmatory claim.  Before any policy evaluation, the CLI tests whether the
governing daily adapter reacts to the contract arm and route-closure tape.  The
current adapter fails both tests.  The OAT-derived visible-v1 metric is also
globally quarantined as not source-validated.  The CLI therefore emits only an
invalidation artifact and exits nonzero.  Policy-screen functions remain
testable but their outputs must not be used while either defect exists.

The historical ``program_g.mpc_policy`` is also replayed as an audit-only
diagnostic.  It is excluded from the deployable policy set because its rollout
reads current/future route states and, for horizon > 1, future signal rows.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.config import HOURS_PER_WEEK
from supply_chain.program_g import (
    ACTIONS,
    CONVOY_LOAD,
    CYCLES_PER_WEEK,
    CSSU_CAP,
    DEMAND_DAYS,
    SB_INITIAL,
    S1_DAILY,
    cover_signal_policy,
    materialize_tape,
    metrics_all,
    mpc_policy,
    signal_hysteresis_policy,
    simulate,
)
from supply_chain.program_h import ARM, belief_rollout_actions
from supply_chain.ret_thesis import (
    compute_order_level_ret_excel_visible_ledger,
)


ROOT = Path(__file__).resolve().parent.parent
VISIBLE_V1_SEMANTICS_AUDIT = (
    ROOT
    / "research/paper2_exhaustive_search/ret_excel_visible_v1_source_semantics_audit_20260714.json"
)
REQUEST_SNAPSHOT_V2_AUDIT = (
    ROOT
    / "research/paper2_exhaustive_search/ret_excel_request_snapshot_v2_implementation_audit_20260714.json"
)
WEEKS = 4
REFERENCE = ("A", "B", "A", "B")
REGION = [
    {
        "cell_id": f"P{p}_Q{int(q * 100)}_L{lead}_S150",
        "signal_q": q,
        "lead_weeks": lead,
        "surge_mult": 1.50,
        "persistence": p,
        "r22_weekly_prob": 0.05,
    }
    for p in ("short", "long")
    for q in (0.65, 0.75, 0.85)
    for lead in (1, 2)
]
METRICS = (
    "ret_visible",
    "ret_quantity",
    "lost_orders",
    "attended_orders",
    "worst_cssu_fill",
    "unfulfilled_rations",
    "scheduled_dispatches",
    "cargo_departures",
    "dispatched_rations",
)
SIGNAL_DEPENDENT = {
    "cover_signal",
    "signal_hysteresis",
    "belief_rollout_1w",
    "belief_rollout_2w",
    "belief_rollout_3w",
    "belief_point_rollout",
}
ELIGIBLE_POLICIES = (
    "cover_no_signal",
    "cover_signal",
    "signal_hysteresis",
    "belief_rollout_1w",
    "belief_rollout_2w",
    "belief_rollout_3w",
    "belief_point_rollout",
)
INELIGIBLE_DIAGNOSTICS = ("historical_mpc_h2_privileged",)
PLACEBOS = ("block_shuffled", "delayed", "wrong_location")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def metric_governance_audit() -> dict[str, Any]:
    """Load and fail closed on the current source-semantics decisions."""
    visible = json.loads(VISIBLE_V1_SEMANTICS_AUDIT.read_text())
    request_snapshot = json.loads(REQUEST_SNAPSHOT_V2_AUDIT.read_text())
    visible_status = visible["scientific_disposition"]["current_ret_excel_visible_v1"]
    h_disposition = visible["scientific_disposition"][
        "program_h_j_visible_v1_repairs"
    ]
    superseded = request_snapshot["superseded_contract"]
    if visible_status != "HOLD_FOR_LEDGER_SEMANTICS_REPAIR":
        raise ValueError("unexpected visible-v1 source-semantics status")
    if not h_disposition.startswith("QUARANTINE_IF_SCORED_BY_THE_CURRENT_OAT_LEDGER"):
        raise ValueError("Program-H metric quarantine is absent")
    if superseded != {
        "disposition": "QUARANTINED_OAT_LEDGER_NOT_SOURCE_VALIDATED",
        "id": "ret_excel_visible_v1",
    }:
        raise ValueError("unexpected request-snapshot-v2 supersession record")
    return {
        "diagnostic_metric": "ret_excel_visible_v1",
        "diagnostic_metric_status": visible_status,
        "diagnostic_metric_disposition": superseded["disposition"],
        "program_h_disposition": h_disposition,
        "replacement_development_contract": request_snapshot[
            "canonical_development_contract"
        ],
        "replacement_status": request_snapshot["status"],
        "prior_h_results_restored": request_snapshot["scientific_authorization"][
            "prior_h_j_mtr_results_restored"
        ],
        "required_rescore": request_snapshot["scientific_authorization"][
            "required_next"
        ],
        "visible_v1_semantics_audit": {
            "path": str(VISIBLE_V1_SEMANTICS_AUDIT),
            "sha256": sha256(VISIBLE_V1_SEMANTICS_AUDIT),
            "content_sha256": visible["content_sha256"],
        },
        "request_snapshot_v2_audit": {
            "path": str(REQUEST_SNAPSHOT_V2_AUDIT),
            "sha256": sha256(REQUEST_SNAPSHOT_V2_AUDIT),
            "content_sha256": request_snapshot["content_sha256"],
        },
        "can_support_h_result": False,
    }


def make_tape(index: int, start: int):
    return materialize_tape(
        start + index,
        REGION[index % len(REGION)],
        WEEKS,
        persistent=True,
    )


def policy_actions(name: str, current_tape) -> tuple[str, ...]:
    """Call only existing repository policies; no fitted or neural policy."""
    if name == "ABAB":
        return REFERENCE
    if name == "cover_no_signal":
        return cover_signal_policy(current_tape, ARM, use_signal=False)
    if name == "cover_signal":
        return cover_signal_policy(current_tape, ARM, use_signal=True)
    if name == "signal_hysteresis":
        return signal_hysteresis_policy(current_tape)
    if name.startswith("belief_rollout_") and name.endswith("w"):
        lookahead = int(name.removeprefix("belief_rollout_").removesuffix("w"))
        return belief_rollout_actions(current_tape, lookahead=lookahead)
    if name == "belief_point_rollout":
        return belief_rollout_actions(current_tape, lookahead=None)
    if name == "historical_mpc_h2_privileged":
        return mpc_policy(current_tape, ARM, horizon=2)
    raise KeyError(name)


def signal_donors(tapes: Sequence[Any]) -> list[int]:
    """Return a deterministic within-cell derangement for block-shuffle placebo."""
    groups: dict[str, list[int]] = defaultdict(list)
    for index, current_tape in enumerate(tapes):
        groups[str(current_tape.cell["cell_id"])].append(index)
    donors = list(range(len(tapes)))
    for indices in groups.values():
        if len(indices) < 2:
            # Tiny unit-test blocks can have singleton cells.  Rotating the full
            # block remains deterministic but is reported in the donor ledger.
            continue
        for offset, index in enumerate(indices):
            donors[index] = indices[(offset + 1) % len(indices)]
    if len(tapes) > 1:
        singleton_indices = [index for index, donor in enumerate(donors) if index == donor]
        for index in singleton_indices:
            donors[index] = (index + 1) % len(tapes)
    return donors


def placebo_tape(
    current_tape,
    *,
    kind: str,
    donor_tape=None,
):
    """Change only the observation signal; physical demand and route tapes stay fixed."""
    signal = np.asarray(current_tape.signal, dtype=int).copy()
    if kind == "block_shuffled":
        if donor_tape is None:
            raise ValueError("block_shuffled placebo requires donor_tape")
        signal = np.asarray(donor_tape.signal, dtype=int).copy()
    elif kind == "delayed":
        signal[1:] = signal[:-1]
        signal[0] = 0
    elif kind == "wrong_location":
        signal = signal[:, ::-1].copy()
    else:
        raise KeyError(kind)
    return replace(current_tape, signal=signal)


def resource_ledger(current_tape, actions: Sequence[str]) -> dict[str, float]:
    """Replay the exact daily delivery loop used by ``program_g.metrics_all``.

    Scheduled dispatches are the budgeted convoy commitments. Cargo departures
    and moved rations are additionally exposed so a result cannot conceal higher
    realized transport use behind an equal action-space statement.
    """
    inv = [0.0, 0.0]
    backlog = [0.0, 0.0]
    sb = float(SB_INITIAL)
    scheduled = 0
    cargo_departures = 0
    dispatched_rations = 0.0
    for week in range(current_tape.weeks):
        action = actions[week] if week < len(actions) else "HOLD"
        priority = 0 if action == "A" else 1 if action == "B" else None
        daily = [
            current_tape.demand[week, 0] / DEMAND_DAYS,
            current_tape.demand[week, 1] / DEMAND_DAYS,
        ]
        for dow in range(DEMAND_DAYS):
            sb += S1_DAILY
            if priority is not None and dow in (1, 3, 5):
                scheduled += 1
                delivered = min(CONVOY_LOAD, sb, CSSU_CAP - inv[priority])
                inv[priority] += delivered
                sb -= delivered
                if delivered > 1e-12:
                    cargo_departures += 1
                    dispatched_rations += delivered
            for cssu in range(2):
                backlog[cssu] += daily[cssu]
                served = min(inv[cssu], backlog[cssu])
                inv[cssu] -= served
                backlog[cssu] -= served
    return {
        "scheduled_dispatches": float(scheduled),
        "cargo_departures": float(cargo_departures),
        "dispatched_rations": float(dispatched_rations),
    }


def evaluate(current_tape, actions: Sequence[str]) -> dict[str, float]:
    metrics = metrics_all(current_tape, actions, ARM)
    visible = compute_order_level_ret_excel_visible_ledger(
        metrics["orders"], current_time=WEEKS * HOURS_PER_WEEK
    )
    resources = resource_ledger(current_tape, actions)
    return {
        "ret_visible": float(visible["mean_ret_excel"]),
        "ret_quantity": float(metrics["ret_quantity"]),
        "lost_orders": float(metrics["lost_orders"]),
        "attended_orders": float(metrics["attended_orders"]),
        "worst_cssu_fill": float(metrics["worst_cssu_fill"]),
        "unfulfilled_rations": float(metrics["unfulfilled_rations_at_horizon"]),
        **resources,
    }


def governing_endpoint_snapshot(
    current_tape,
    actions: Sequence[str],
    *,
    arm: str,
) -> dict[str, float]:
    """Only endpoint fields, for exact physical-liveness comparisons."""
    metrics = metrics_all(current_tape, actions, arm)
    visible = compute_order_level_ret_excel_visible_ledger(
        metrics["orders"], current_time=WEEKS * HOURS_PER_WEEK
    )
    return {
        "ret_visible": float(visible["mean_ret_excel"]),
        "ret_quantity": float(metrics["ret_quantity"]),
        "lost_orders": float(metrics["lost_orders"]),
        "attended_orders": float(metrics["attended_orders"]),
        "worst_cssu_fill": float(metrics["worst_cssu_fill"]),
        "unfulfilled_rations": float(metrics["unfulfilled_rations_at_horizon"]),
    }


def adapter_liveness_audit(current_tape) -> dict[str, Any]:
    """Prove whether r22 and arm can affect the governing endpoint."""
    zero_r22 = replace(current_tape, r22=np.zeros_like(current_tape.r22))
    one_r22 = replace(current_tape, r22=np.ones_like(current_tape.r22))
    r22_zero = governing_endpoint_snapshot(zero_r22, REFERENCE, arm="TRS")
    r22_one = governing_endpoint_snapshot(one_r22, REFERENCE, arm="TRS")
    arm_t = governing_endpoint_snapshot(current_tape, REFERENCE, arm="T")
    arm_trs = governing_endpoint_snapshot(current_tape, REFERENCE, arm="TRS")
    weekly_zero = simulate(zero_r22, REFERENCE, arm="TRS")
    weekly_one = simulate(one_r22, REFERENCE, arm="TRS")
    return {
        "probe_seed": int(current_tape.seed),
        "probe_actions": list(REFERENCE),
        "r22_all_zero": r22_zero,
        "r22_all_one": r22_one,
        "r22_bit_identical": json_sha256(r22_zero) == json_sha256(r22_one),
        "arm_T": arm_t,
        "arm_TRS": arm_trs,
        "arm_bit_identical": json_sha256(arm_t) == json_sha256(arm_trs),
        "source_mechanism_check": {
            "metrics_all_reads_arm": False,
            "metrics_all_reads_tape_r22": False,
            "evidence": "supply_chain/program_g.py metrics_all accepts arm but never references it and its daily delivery loop never references tape.r22.",
        },
        "weekly_reference_physics": {
            "r22_zero_service_loss": float(weekly_zero.service_loss),
            "r22_one_service_loss": float(weekly_one.service_loss),
            "service_loss_bit_identical": bool(
                weekly_zero.service_loss == weekly_one.service_loss
            ),
            "r22_zero_convoy_missions": int(weekly_zero.convoy_missions),
            "r22_one_convoy_missions": int(weekly_one.convoy_missions),
            "mission_ledger_changes": bool(
                weekly_zero.convoy_missions != weekly_one.convoy_missions
            ),
            "structural_delivery_ceiling": {
                "normal_cycles_times_load": CYCLES_PER_WEEK * CONVOY_LOAD,
                "closed_cycles_times_load": (CYCLES_PER_WEEK - 1) * CONVOY_LOAD,
                "cssu_capacity": CSSU_CAP,
                "proof": "A route closure reduces three cycles to two, but two 5,000-ration loads already equal the 10,000-ration CSSU capacity. Since free capacity is at most 10,000, the closure cannot reduce weekly delivered quantity in the frozen weekly kernel; it changes only the mission ledger.",
            },
        },
    }


def evaluate_split(tapes: Sequence[Any]) -> dict[str, Any]:
    donors = signal_donors(tapes)
    names = ("ABAB",) + ELIGIBLE_POLICIES + INELIGIBLE_DIAGNOSTICS
    outcomes = {
        name: {metric: [] for metric in METRICS}
        for name in names
    }
    actions: dict[str, list[list[str]]] = {name: [] for name in names}
    for base in SIGNAL_DEPENDENT:
        for placebo in PLACEBOS:
            name = f"{base}__placebo_{placebo}"
            outcomes[name] = {metric: [] for metric in METRICS}
            actions[name] = []

    for index, current_tape in enumerate(tapes):
        for name in names:
            sequence = policy_actions(name, current_tape)
            row = evaluate(current_tape, sequence)
            actions[name].append(list(sequence))
            for metric in METRICS:
                outcomes[name][metric].append(row[metric])
        for base in SIGNAL_DEPENDENT:
            for placebo in PLACEBOS:
                name = f"{base}__placebo_{placebo}"
                observed_tape = placebo_tape(
                    current_tape,
                    kind=placebo,
                    donor_tape=tapes[donors[index]],
                )
                sequence = policy_actions(base, observed_tape)
                # Physical outcomes always use the original tape.
                row = evaluate(current_tape, sequence)
                actions[name].append(list(sequence))
                for metric in METRICS:
                    outcomes[name][metric].append(row[metric])
    arrays = {
        name: {metric: np.asarray(values, dtype=float) for metric, values in rows.items()}
        for name, rows in outcomes.items()
    }
    return {
        "outcomes": arrays,
        "actions": actions,
        "donor_indices": donors,
    }


def modal_sequence(sequences: Sequence[Sequence[str]]) -> tuple[str, ...]:
    counts = Counter(tuple(sequence) for sequence in sequences)
    return sorted(counts, key=lambda seq: (-counts[seq], seq))[0]


def phase_mode_sequence(sequences: Sequence[Sequence[str]]) -> tuple[str, ...]:
    return tuple(
        sorted(
            Counter(sequence[week] for sequence in sequences).items(),
            key=lambda item: (-item[1], item[0]),
        )[0][0]
        for week in range(WEEKS)
    )


def bootstrap_indices(n_tapes: int, n_resamples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_tapes, size=(n_resamples, n_tapes))


def paired_ci(
    candidate: np.ndarray,
    baseline: np.ndarray,
    indices: np.ndarray,
) -> dict[str, float]:
    delta = np.asarray(candidate, dtype=float) - np.asarray(baseline, dtype=float)
    means = delta[indices].mean(axis=1)
    return {
        "mean": float(delta.mean()),
        "lcb95": float(np.quantile(means, 0.025)),
        "ucb95": float(np.quantile(means, 0.975)),
    }


def means(rows: dict[str, np.ndarray]) -> dict[str, float]:
    return {metric: float(values.mean()) for metric, values in rows.items()}


def calibration_eligible(
    rows: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    *,
    worst_fill_margin: float,
) -> bool:
    """Retrospective selector; exact confirmatory gates are deliberately not claimed."""
    return bool(
        rows["lost_orders"].mean() <= reference["lost_orders"].mean() + 1e-12
        and rows["ret_quantity"].mean() >= reference["ret_quantity"].mean() - 1e-12
        and rows["worst_cssu_fill"].mean()
        >= reference["worst_cssu_fill"].mean() - worst_fill_margin - 1e-12
        and rows["scheduled_dispatches"].mean()
        <= reference["scheduled_dispatches"].mean() + 1e-12
    )


def action_audit(
    name: str,
    calibration: dict[str, Any],
    locked: dict[str, Any],
    tapes: Sequence[Any],
    indices: np.ndarray,
) -> dict[str, Any]:
    cal_sequences = calibration["actions"][name]
    locked_sequences = locked["actions"][name]
    modal = modal_sequence(cal_sequences)
    phase = phase_mode_sequence(cal_sequences)
    fixed_sequences = {"calibration_modal": modal, "phase_mode": phase, "ABAB": REFERENCE}
    fixed_rows: dict[str, dict[str, np.ndarray]] = {}
    for label, sequence in fixed_sequences.items():
        rows = [evaluate(current_tape, sequence) for current_tape in tapes]
        fixed_rows[label] = {
            metric: np.asarray([row[metric] for row in rows], dtype=float)
            for metric in METRICS
        }
    candidate_rows = locked["outcomes"][name]
    counts = Counter(tuple(sequence) for sequence in locked_sequences)
    locked_modal = modal_sequence(locked_sequences)
    return {
        "calibration_modal_sequence": list(modal),
        "calibration_phase_mode_sequence": list(phase),
        "locked_unique_sequence_count": len(counts),
        "locked_modal_sequence": list(locked_modal),
        "locked_modal_count": int(counts[locked_modal]),
        "locked_modal_fraction": float(counts[locked_modal] / len(locked_sequences)),
        "per_phase_unique_actions_locked": [
            sorted({sequence[week] for sequence in locked_sequences})
            for week in range(WEEKS)
        ],
        "replacement_comparisons": {
            label: {
                "fixed_means": means(rows),
                "candidate_minus_fixed": {
                    metric: paired_ci(candidate_rows[metric], rows[metric], indices)
                    for metric in METRICS
                },
            }
            for label, rows in fixed_rows.items()
        },
        "state_feedback_certificate_against_best_fixed": bool(
            len(counts) > 1
            and paired_ci(
                candidate_rows["ret_visible"],
                fixed_rows["ABAB"]["ret_visible"],
                indices,
            )["lcb95"]
            > 0.0
        ),
        "interpretation": "The certificate requires both multiple locked action sequences and a positive paired lower bound versus the calibration-selected full-horizon fixed comparator ABAB.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-start", type=int, default=1_060_001)
    parser.add_argument("--calibration-tapes", type=int, default=200)
    parser.add_argument("--locked-start", type=int, default=1_070_001)
    parser.add_argument("--locked-tapes", type=int, default=400)
    parser.add_argument("--bootstrap-resamples", type=int, default=4000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260714)
    parser.add_argument("--worst-fill-margin", type=float, default=0.02)
    parser.add_argument(
        "--frontier-source",
        type=Path,
        default=ROOT / "results/program_h/visible_v1_repair/verdict.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/program_h/visible_v1_repair/observable_policy_screen.json",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    source = json.loads(args.frontier_source.read_text())
    if source.get("schema_version") != "program_h_visible_v1_frontier_repair_v1":
        raise ValueError("unexpected Program-H frontier source schema")
    if source["governing_metric"] != "ret_excel_visible_v1":
        raise ValueError("frontier source does not use the governing visible-v1 metric")
    support = source["guardrailed_comparator"]["support"]
    if (
        len(support) != 1
        or support[0]["sequence"] != "ABAB"
        or not np.isclose(float(support[0]["weight"]), 1.0)
    ):
        raise ValueError("observable screen requires the calibrated pure ABAB comparator")

    # Mandatory metric-governance and physical-liveness gates. No policy screen
    # may be emitted under the quarantined OAT ledger or when the named TRS
    # mechanism is absent from the governing adapter.
    metric_governance = metric_governance_audit()
    liveness = adapter_liveness_audit(make_tape(0, args.calibration_start))
    if (
        not metric_governance["can_support_h_result"]
        or liveness["r22_bit_identical"]
        or liveness["arm_bit_identical"]
    ):
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        invalid = {
            "schema_version": "program_h_visible_v1_observable_policy_screen_invalid_v1",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "repository_head": head,
            "scientific_status": "INVALID_PROGRAM_H_DEAD_ROUTE_AND_QUARANTINED_VISIBLE_V1",
            "governing_metric": None,
            "diagnostic_metric": "ret_excel_visible_v1",
            "frontier_source": {
                "path": str(args.frontier_source),
                "sha256": sha256(args.frontier_source),
            },
            "metric_governance": metric_governance,
            "adapter_liveness_audit": liveness,
            "policies_evaluated": False,
            "retrospective_screen_pass": False,
            "valid_for_H_obs": False,
            "valid_for_H_PI": False,
            "valid_for_paper2": False,
            "tapes": {
                "probe_seed": args.calibration_start,
                "probe_status": "historical burned",
                "virgin_opened": False,
            },
            "reopening_requirements": [
                "Obtain domain authorization for materially new R22 physics; the frozen weekly kernel makes a one-cycle closure service-inert when two remaining 5,000-ration cycles already fill 10,000-ration CSSU capacity.",
                "Implement and independently validate the authorized R22 mechanism in the Program-G/H daily governing adapter, including causal arm and route-closure endpoint-liveness tests.",
                "Rescore the identical burned/calibration tapes with ret_excel_request_snapshot_v2 and rebuild every fixed and observable comparator under that contract.",
                "Retain the request-snapshot-v2 provisional boundary, including Garrido confirmation of same-timestamp request-ledger event order, before any virgin confirmation.",
            ],
            "required_resolution": "Program H can reopen only after both an authorized, causally live R22 adapter and a complete request-snapshot-v2 rescore on identical burned/calibration tapes. Fixing only physics or only the metric is insufficient.",
            "claim_limit": "The quarantined visible-v1 metric cannot support any Program-H H_PI, H_obs, ceiling, null, or positive. Demand-only outputs from the dead-route adapter are independently invalid. The previously generated full screen was deleted and replaced by this fail-closed invalidation artifact.",
        }
        invalid["content_sha256"] = json_sha256(invalid)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(invalid, indent=2, sort_keys=True) + "\n")
        print(json.dumps(invalid, indent=2, sort_keys=True))
        return 2

    started = time.perf_counter()
    calibration_tapes = [
        make_tape(index, args.calibration_start)
        for index in range(args.calibration_tapes)
    ]
    locked_tapes = [
        make_tape(index, args.locked_start) for index in range(args.locked_tapes)
    ]
    calibration = evaluate_split(calibration_tapes)
    locked = evaluate_split(locked_tapes)
    reference_cal = calibration["outcomes"]["ABAB"]
    eligible_on_calibration = [
        name
        for name in ELIGIBLE_POLICIES
        if calibration_eligible(
            calibration["outcomes"][name],
            reference_cal,
            worst_fill_margin=args.worst_fill_margin,
        )
    ]
    selected = (
        sorted(
            eligible_on_calibration,
            key=lambda name: (
                -calibration["outcomes"][name]["ret_visible"].mean(),
                name,
            ),
        )[0]
        if eligible_on_calibration
        else None
    )
    indices = bootstrap_indices(
        args.locked_tapes, args.bootstrap_resamples, args.bootstrap_seed
    )
    locked_reference = locked["outcomes"]["ABAB"]

    real_policy_rows = {}
    action_audits = {}
    for name in ELIGIBLE_POLICIES + INELIGIBLE_DIAGNOSTICS:
        rows = locked["outcomes"][name]
        comparison = {
            metric: paired_ci(rows[metric], locked_reference[metric], indices)
            for metric in METRICS
        }
        real_policy_rows[name] = {
            "eligible_nonanticipative": name in ELIGIBLE_POLICIES,
            "calibration_means": means(calibration["outcomes"][name]),
            "calibration_guardrails_satisfied": name in eligible_on_calibration,
            "locked_means": means(rows),
            "candidate_minus_ABAB_paired_bootstrap": comparison,
            "favorable_tape_fraction_ret_visible": float(
                np.mean(rows["ret_visible"] > locked_reference["ret_visible"])
            ),
            "screen_gates": {
                "ret_visible_point_at_least_0_01": comparison["ret_visible"]["mean"]
                >= 0.01,
                "ret_visible_lcb95_positive": comparison["ret_visible"]["lcb95"]
                > 0.0,
                "lost_orders_noninferior": comparison["lost_orders"]["ucb95"]
                <= 0.0,
                "quantity_ret_noninferior": comparison["ret_quantity"]["lcb95"]
                >= 0.0,
                "worst_cssu_within_margin": comparison["worst_cssu_fill"]["lcb95"]
                >= -args.worst_fill_margin,
                "scheduled_dispatches_non_superior": comparison[
                    "scheduled_dispatches"
                ]["ucb95"]
                <= 0.0,
                "cargo_departures_non_superior": comparison["cargo_departures"][
                    "ucb95"
                ]
                <= 0.0,
                "favorable_tapes_at_least_0_70": float(
                    np.mean(rows["ret_visible"] > locked_reference["ret_visible"])
                )
                >= 0.70,
            },
        }
        action_audits[name] = action_audit(
            name, calibration, locked, locked_tapes, indices
        )
        real_policy_rows[name]["screen_gates"]["state_feedback_certificate"] = (
            action_audits[name]["state_feedback_certificate_against_best_fixed"]
        )
        real_policy_rows[name]["passes_all_retrospective_screen_gates"] = bool(
            name in ELIGIBLE_POLICIES
            and all(real_policy_rows[name]["screen_gates"].values())
        )

    placebo_audit: dict[str, Any] = {}
    for base in SIGNAL_DEPENDENT:
        placebo_audit[base] = {}
        for placebo in PLACEBOS:
            placebo_name = f"{base}__placebo_{placebo}"
            real_rows = locked["outcomes"][base]
            placebo_rows = locked["outcomes"][placebo_name]
            placebo_audit[base][placebo] = {
                "placebo_means": means(placebo_rows),
                "real_minus_placebo_paired_bootstrap": {
                    metric: paired_ci(real_rows[metric], placebo_rows[metric], indices)
                    for metric in METRICS
                },
                "action_sequence_disagreement_fraction": float(
                    np.mean(
                        [
                            real != placebo_actions
                            for real, placebo_actions in zip(
                                locked["actions"][base],
                                locked["actions"][placebo_name],
                            )
                        ]
                    )
                ),
            }

    selected_placebo_pass = False
    if selected in SIGNAL_DEPENDENT:
        selected_placebo_pass = all(
            placebo_audit[selected][placebo]["real_minus_placebo_paired_bootstrap"][
                "ret_visible"
            ]["lcb95"]
            > 0.0
            for placebo in PLACEBOS
        )
    selected_pass = bool(
        selected is not None
        and real_policy_rows[selected]["passes_all_retrospective_screen_gates"]
        and selected_placebo_pass
    )

    trajectory_rows = {}
    for name, sequences in locked["actions"].items():
        trajectory_rows[name] = [
            {
                "seed": int(current_tape.seed),
                "cell_id": str(current_tape.cell["cell_id"]),
                "actions": sequence,
            }
            for current_tape, sequence in zip(locked_tapes, sequences)
        ]
    donor_rows = [
        {
            "recipient_seed": int(current_tape.seed),
            "donor_seed": int(locked_tapes[locked["donor_indices"][index]].seed),
            "recipient_cell": str(current_tape.cell["cell_id"]),
            "donor_cell": str(
                locked_tapes[locked["donor_indices"][index]].cell["cell_id"]
            ),
        }
        for index, current_tape in enumerate(locked_tapes)
    ]
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    result = {
        "schema_version": "program_h_visible_v1_observable_policy_screen_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_head": head,
        "scientific_status": "RETROSPECTIVE_BURNED_TAPE_OBSERVABLE_LOWER_BOUND_SCREEN_NOT_CONFIRMATORY",
        "governing_metric": "ret_excel_visible_v1",
        "frontier_source": {
            "path": str(args.frontier_source),
            "sha256": sha256(args.frontier_source),
        },
        "contract": {
            "weeks": WEEKS,
            "arm": ARM,
            "actions": list(ACTIONS),
            "fixed_comparator": list(REFERENCE),
            "worst_cssu_margin": args.worst_fill_margin,
            "eligible_policy_names": list(ELIGIBLE_POLICIES),
            "ineligible_diagnostics": {
                "historical_mpc_h2_privileged": "Excluded: program_g.mpc_policy reads actual current/future r22 rows and future signal rows inside its horizon, which are absent from the O0 observation whitelist.",
            },
            "observation_boundary": "Eligible policies use current/past signal rows, public cell parameters, inventory implied by past realized demand, and the week phase. They do not read latent z or future demand/r22/signal rows.",
            "resource_rule": "All policies share the same weekly action rights. Scheduled dispatch commitments, cargo-bearing departures, and moved rations are separately reported.",
        },
        "tapes": {
            "calibration": {
                "seed_start": args.calibration_start,
                "n": args.calibration_tapes,
                "status": "historical burned",
            },
            "locked": {
                "seed_start": args.locked_start,
                "n": args.locked_tapes,
                "status": "historical burned; opened previously by Program H",
            },
            "virgin_opened": False,
        },
        "bootstrap": {
            "method": "paired nonparametric tape bootstrap with one shared resample-index matrix",
            "resamples": args.bootstrap_resamples,
            "seed": args.bootstrap_seed,
            "indices_sha256": json_sha256(indices.tolist()),
        },
        "calibration_selection": {
            "rule": "Among the named eligible non-neural policies satisfying calibration mean lost-order, quantity-ReT, worst-CSSU, and scheduled-dispatch constraints versus ABAB, maximize calibration mean visible-v1 ReT; break ties by policy name.",
            "eligible_after_guardrails": eligible_on_calibration,
            "selected_policy": selected,
            "selection_is_retrospective_not_preregistered": True,
        },
        "reference_ABAB_locked_means": means(locked_reference),
        "policy_results": real_policy_rows,
        "action_trajectory_audit": action_audits,
        "information_placebo_audit": {
            "definitions": {
                "block_shuffled": "Signal tensor is taken from a deterministic other tape; physical demand and route histories remain the recipient's.",
                "delayed": "At week w the signal is the recipient's week w-1 signal; week zero is all-clear.",
                "wrong_location": "CSSU-A and CSSU-B signal columns are swapped; physical node outcomes are unchanged.",
            },
            "block_shuffle_donor_ledger": donor_rows,
            "policies": placebo_audit,
        },
        "selected_policy_placebo_gate": {
            "applicable": selected in SIGNAL_DEPENDENT,
            "real_signal_lcb_positive_against_all_three": selected_placebo_pass,
        },
        "retrospective_screen_pass": selected_pass,
        "locked_action_trajectories": trajectory_rows,
        "locked_action_trajectories_sha256": json_sha256(trajectory_rows),
        "claim_limit": "This finite named-policy screen is only a lower bound on H_obs and uses previously opened tapes after the metric repair. A pass would justify a fresh preregistered pre-learner gate, not Paper 2. A failure does not prove information impossibility or close Program H. No learner or retained-learning claim is authorized.",
        "known_physics_limit": "Program G/H is a disclosed stylized two-CSSU adapter, not the full Garrido DES. Its daily order adapter does not apply the tape's r22 closures, so r22 is inert in the governing endpoint; this screen does not repair that physical limitation.",
        "elapsed_seconds": time.perf_counter() - started,
    }
    result["content_sha256"] = json_sha256(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "selected": selected,
                "selected_pass": selected_pass,
                "selected_placebo_pass": selected_placebo_pass,
                "selected_result": real_policy_rows.get(selected),
                "elapsed_seconds": result["elapsed_seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
