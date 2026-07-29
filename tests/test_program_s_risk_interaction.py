from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.paper2_exhaustive_search.program_s_design import (
    build_operational_alarms,
    build_program_s_risk_tape,
)
from supply_chain.program_o_full_des import run_program_o_full_des_episode
from supply_chain.program_s_risk_interaction import (
    OperationalAlarm,
    ProgramSCell,
    ProgramSRiskAwareSimulation,
    validate_program_s_risk_tape,
)


ROOT = Path(__file__).resolve().parents[1]
PARENT = json.loads(
    (ROOT / "contracts/program_o_full_des_hpi_translation_v1.json").read_text()
)
SCHEDULER = PARENT["action"]["within_week_schedulers"][
    PARENT["action"]["primary_scheduler"]
]


def cell(**overrides) -> ProgramSCell:
    values = {
        "stratum": "THESIS_NATIVE_INDEPENDENT",
        "mask": "PRODUCTION_QUALITY_SURGE",
        "coupling": "independent",
        "phi_by_risk": {"R11": 1.0, "R14": 1.0, "R24": 1.0},
        "psi_by_risk": {"R11": 1.0, "R14": 1.0, "R24": 1.0},
        "r14_probability_multiplier": 1.0,
        "baseline_capacity_multiplier": 1.0,
        "regime_persistence": 0.75,
        "dominant_share": 0.90,
        "alarm_lead_hours": 24.0,
        "alarm_balanced_accuracy": 0.70,
    }
    values.update(overrides)
    return ProgramSCell(**values)


def test_cell_rejects_cross_stratum_timing_and_foreign_risks() -> None:
    with pytest.raises(ValueError, match="thesis-native"):
        cell(coupling="coincident")
    with pytest.raises(ValueError, match="outside the mask"):
        cell(phi_by_risk={"R22": 2.0})
    wartime = cell(
        stratum="RESEARCHER_WARTIME_COUPLED",
        coupling="coincident",
    )
    assert wartime.stratum == "RESEARCHER_WARTIME_COUPLED"


def test_r3_and_duplicate_cluster_events_fail_closed() -> None:
    with pytest.raises(ValueError, match="R3"):
        validate_program_s_risk_tape(
            [{"risk_id": "R3", "start_time": 1, "affected_ops": [1]}],
            mask="PRODUCTION_QUALITY_SURGE",
        )
    duplicate = {
        "risk_id": "R11",
        "start_time": 12,
        "duration": 2,
        "end_time": 14,
        "affected_ops": [5],
    }
    with pytest.raises(ValueError, match="duplicated"):
        validate_program_s_risk_tape(
            [duplicate, duplicate], mask="PRODUCTION_QUALITY_SURGE"
        )


def test_policy_independent_risk_tape_is_reproducible_and_has_no_r3() -> None:
    first = build_program_s_risk_tape(cell(), tape_id=7510000, horizon_hours=8 * 168)
    second = build_program_s_risk_tape(cell(), tape_id=7510000, horizon_hours=8 * 168)
    assert first == second
    assert first["r3_event_count"] == 0
    assert {row["risk_id"] for row in first["events"]} <= {"R11", "R14", "R24"}


def test_alarm_channel_is_reproducible_and_never_contains_risk_id() -> None:
    tape = build_program_s_risk_tape(cell(), tape_id=7510000, horizon_hours=8 * 168)
    with pytest.raises(RuntimeError, match="forbidden in amended Program S-NATIVE"):
        build_operational_alarms(
            cell(), tape_id=7510000, events=tape["events"], horizon_hours=8 * 168
        )


def test_program_o_new_risk_arguments_preserve_riskoff_defaults() -> None:
    kwargs = dict(
        seed=7400048,
        calendar=[2] * 8,
        scheduler=SCHEDULER,
        regime_persistence=0.75,
        dominant_share=0.90,
    )
    _, historical = run_program_o_full_des_episode(**kwargs)
    _, explicit = run_program_o_full_des_episode(
        **kwargs,
        risks_enabled=False,
        risk_impact_multipliers_by_id=None,
        risk_event_tape=None,
        risk_rng_mode="shared",
    )
    assert explicit == historical


def test_program_s_replays_risk_after_neutral_prefix_and_hides_generators() -> None:
    event = {
        "risk_id": "R11",
        "start_time": 24.0,
        "duration": 2.0,
        "end_time": 26.0,
        "affected_ops": [5],
        "magnitude": 1.0,
    }
    sim = ProgramSRiskAwareSimulation(
        seed=7510000,
        calendar=[2],
        scheduler=SCHEDULER,
        cell=cell(),
        risk_event_tape=[event],
    ).run_contract()
    observed = [row for row in sim.risk_events if row.risk_id == "R11"]
    assert len(observed) == 1
    assert observed[0].start_time == pytest.approx(sim.program_o_decision_start + 24)
    observation = sim.program_s_observation()
    forbidden = {"seed", "tape", "phi", "psi", "rho", "share", "risk_id", "regime"}
    assert forbidden.isdisjoint(observation)
    assert observation["alarm"] is None


def test_native_simulation_rejects_legacy_anticipatory_alarm() -> None:
    with pytest.raises(ValueError, match="S-NATIVE forbids"):
        ProgramSRiskAwareSimulation(
            seed=7510000,
            calendar=[2],
            scheduler=SCHEDULER,
            cell=cell(),
            risk_event_tape=[],
            operational_alarms=[
                OperationalAlarm(0, (24, 26), "production", "low", 0.5, 0.7)
            ],
        )


def test_r14_rework_preserves_product_mass_and_returns_from_op6() -> None:
    event = {
        "risk_id": "R14",
        "start_time": 120.0,
        "duration": 0.0,
        "end_time": 120.0,
        "affected_ops": [7],
        "magnitude": 30.0,
        "unit": "defective_products",
    }
    sim = ProgramSRiskAwareSimulation(
        seed=7510000,
        calendar=[2, 2],
        scheduler=SCHEDULER,
        cell=cell(),
        risk_event_tape=[event],
    ).run_contract()
    panel = sim.product_outcome_panel()
    starts = [
        row for row in sim.program_o_product_events
        if row["event"] == "r14_product_rework_started"
    ]
    returns = [
        row for row in sim.program_o_product_events
        if row["event"] == "r14_rework_returned_to_pending_batch"
    ]
    assert starts and returns
    assert sum(row["quantity"] for row in starts) == pytest.approx(
        sum(row["quantity"] for row in returns)
    )
    assert panel["conservation"]["max_abs_product_residual"] <= 1e-8
    assert panel["conservation"]["max_abs_partition_residual"] <= 1e-8
