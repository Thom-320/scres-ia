from __future__ import annotations

import json
import math
import subprocess
import sys

import pytest

from scripts.run_thesis_factorial import spec_row
from scripts.run_thesis_faithful import THESIS_BACKBONE
from supply_chain.config import (
    HOURS_PER_YEAR_THESIS,
    R14_DEFECT_MODE_OPTIONS,
    RAW_MATERIAL_FLOW_MODE_OPTIONS,
    RAW_MATERIAL_COMPONENTS,
    SIMULATION_HORIZON,
    THESIS_DOWNSTREAM_Q_RANGES,
    THESIS_FAITHFUL_PROTOCOL,
    VALIDATION_TABLE_6_10,
    canonical_raw_material_flow_mode,
)
from supply_chain.external_env_interface import (
    get_thesis_aligned_training_env_spec,
    make_thesis_aligned_training_env,
)
from supply_chain.ret_thesis import compute_ret_per_order
from supply_chain.supply_chain import MFSCSimulation, OrderRecord
from supply_chain.thesis_design import design_spec_for_cfi, parse_cf_range


def test_thesis_protocol_blocks_rl_features() -> None:
    assert THESIS_BACKBONE["protocol"] == "thesis_1to1"
    assert THESIS_BACKBONE["year_basis"] == "thesis"
    assert THESIS_BACKBONE["horizon_hours"] == SIMULATION_HORIZON
    assert THESIS_BACKBONE["warmup_trigger"] == "op9_arrival"
    assert THESIS_BACKBONE["rl_enabled"] is False
    assert THESIS_BACKBONE["reward_mode"] is None
    assert THESIS_BACKBONE["priming_enabled"] is False
    assert THESIS_BACKBONE["action_multipliers_enabled"] is False


def test_extracted_thesis_constants_match_lane_contract() -> None:
    assert HOURS_PER_YEAR_THESIS == 8064
    assert VALIDATION_TABLE_6_10["RMSE"] == 87_918
    assert (
        pytest.approx(
            sum(VALIDATION_TABLE_6_10["ECS_simulated"])
            / len(VALIDATION_TABLE_6_10["ECS_simulated"])
        )
        == 767_591.5
    )
    assert THESIS_FAITHFUL_PROTOCOL["r14_defect_mode"] in R14_DEFECT_MODE_OPTIONS
    assert THESIS_FAITHFUL_PROTOCOL["r14_defect_mode"] == "thesis_strict_op6"
    assert THESIS_FAITHFUL_PROTOCOL["ret_weights"] == {
        "max": 1.0,
        "mean": 0.5,
        "min": 0.0,
    }
    assert THESIS_DOWNSTREAM_Q_RANGES["figure_6_2"]["op9"] == (2400, 2600)
    assert THESIS_DOWNSTREAM_Q_RANGES["table_6_20"]["op9"] == (2000, 2500)
    assert len(RAW_MATERIAL_COMPONENTS) == 12
    assert RAW_MATERIAL_COMPONENTS[0]["id"] == "rm1"
    assert RAW_MATERIAL_COMPONENTS[-1]["id"] == "rm12"


def test_warmup_can_be_marked_at_op9_arrival_instead_of_production() -> None:
    production = MFSCSimulation(
        shifts=1,
        risks_enabled=False,
        seed=42,
        horizon=SIMULATION_HORIZON,
        year_basis="thesis",
        deterministic_baseline=True,
        warmup_trigger="production",
    ).run()
    op9_arrival = MFSCSimulation(
        shifts=1,
        risks_enabled=False,
        seed=42,
        horizon=SIMULATION_HORIZON,
        year_basis="thesis",
        deterministic_baseline=True,
        warmup_trigger="op9_arrival",
    ).run()

    assert op9_arrival.warmup_time > production.warmup_time
    assert op9_arrival.warmup_time - production.warmup_time == pytest.approx(24.0)


def test_downstream_q_source_is_explicitly_selectable() -> None:
    figure = MFSCSimulation(downstream_q_source="figure_6_2")
    table = MFSCSimulation(downstream_q_source="table_6_20")

    assert figure.params["op9_q_min"] == 2400
    assert figure.params["op9_q_max"] == 2600
    assert table.params["op9_q_min"] == 2000
    assert table.params["op9_q_max"] == 2500


def test_invalid_thesis_lane_options_fail_fast() -> None:
    with pytest.raises(ValueError, match="warmup_trigger"):
        MFSCSimulation(warmup_trigger="fixed_838")
    with pytest.raises(ValueError, match="downstream_q_source"):
        MFSCSimulation(downstream_q_source="mixed")
    with pytest.raises(ValueError, match="r14_defect_mode"):
        MFSCSimulation(r14_defect_mode="silent_drop")
    with pytest.raises(ValueError, match="raw_material_flow_mode"):
        MFSCSimulation(raw_material_flow_mode="not_a_mode")


def test_raw_material_flow_mode_aliases_are_explicit_and_canonical() -> None:
    assert "kit_equivalent" in RAW_MATERIAL_FLOW_MODE_OPTIONS
    assert "kit_equivalent_order_up_to" in RAW_MATERIAL_FLOW_MODE_OPTIONS
    assert canonical_raw_material_flow_mode("kit_equivalent") == "bom_total_units"
    assert (
        canonical_raw_material_flow_mode("kit_equivalent_order_up_to")
        == "bom_total_units_order_up_to"
    )

    sim = MFSCSimulation(raw_material_flow_mode="kit_equivalent_order_up_to")
    assert sim.raw_material_flow_mode == "bom_total_units_order_up_to"


def test_bom_total_units_mode_scales_raw_buffer_targets_only() -> None:
    sim = MFSCSimulation(
        initial_buffers={"op3_rm": 46_080, "op5_rm": 46_080, "op9_rations": 47_250},
        raw_material_flow_mode="bom_total_units",
    )

    assert sim.inventory_buffer_targets == {
        "op3_rm": pytest.approx(46_080 * 12),
        "op5_rm": pytest.approx(46_080 * 12),
        "op9_rations": pytest.approx(47_250),
    }
    assert sim.raw_material_wdc.level == pytest.approx(46_080 * 12)
    assert sim.raw_material_al.level == pytest.approx(46_080 * 12)
    assert sim.rations_sb.level == pytest.approx(47_250)


def test_legacy_validated_mode_preserves_raw_buffer_targets() -> None:
    sim = MFSCSimulation(
        initial_buffers={"op3_rm": 46_080, "op5_rm": 46_080, "op9_rations": 47_250},
        raw_material_flow_mode="legacy_validated",
    )

    assert sim.inventory_buffer_targets == {
        "op3_rm": pytest.approx(46_080),
        "op5_rm": pytest.approx(46_080),
        "op9_rations": pytest.approx(47_250),
    }


def test_r14_thesis_strict_returns_defects_to_op6_rework_queue(monkeypatch) -> None:
    sim = MFSCSimulation(
        seed=42,
        risks_enabled=True,
        r14_defect_mode="thesis_strict_op6",
    )
    monkeypatch.setattr(sim, "_get_risk_p", lambda _risk_id: 1.0)
    sim._today_produced = 10
    sim._pending_batch = 10
    sim.total_produced = 10

    sim.env.process(sim._risk_R14())
    sim.env.run(until=25)

    assert sim._pending_batch == 0
    assert sim.total_produced == 0
    assert sim.rework_op6.level == 10
    assert sim.raw_material_al.level == 0
    assert sim.risk_events[-1].description.endswith("(returned to Op6)")


def test_order_level_ret_module_matches_thesis_cases() -> None:
    fill_rate_order = OrderRecord(j=1, OPTj=0, OATj=40, CTj=40, LTj=48)
    autotomy_order = OrderRecord(j=2, OPTj=0, OATj=40, CTj=40, LTj=48, APj=24)
    recovery_order = OrderRecord(j=3, OPTj=0, OATj=72, CTj=72, LTj=48, RPj=24)
    non_recovery_order = OrderRecord(j=4, OPTj=0, OATj=80, CTj=80, LTj=48, DPj=80)

    assert compute_ret_per_order(fill_rate_order, fill_rate=0.75) == (
        0.75,
        "fill_rate",
    )
    assert compute_ret_per_order(autotomy_order, fill_rate=0.75) == (
        pytest.approx(0.5),
        "autotomy",
    )
    assert compute_ret_per_order(recovery_order, fill_rate=0.75) == (
        pytest.approx(0.5 / 24),
        "recovery",
    )
    assert compute_ret_per_order(non_recovery_order, fill_rate=0.75) == (
        0.0,
        "non_recovery",
    )


def test_cf0_deterministic_gate_matches_table_6_10_average() -> None:
    sim = MFSCSimulation(
        shifts=1,
        risks_enabled=False,
        seed=42,
        horizon=SIMULATION_HORIZON,
        year_basis="thesis",
        deterministic_baseline=True,
        warmup_trigger="op9_arrival",
        downstream_q_source=THESIS_FAITHFUL_PROTOCOL["downstream_q_source"],
        r14_defect_mode=THESIS_FAITHFUL_PROTOCOL["r14_defect_mode"],
    ).run()
    throughput = sim.get_annual_throughput(start_time=sim.warmup_time, num_years=8)
    produced_by_year = throughput["produced_by_year"]
    ecs = VALIDATION_TABLE_6_10["ECS_simulated"]
    produced = [float(produced_by_year[index + 1]) for index in range(len(ecs))]
    rmse = math.sqrt(
        sum((actual - expected) ** 2 for actual, expected in zip(produced, ecs))
        / len(ecs)
    )
    avg_delta = (sum(produced) / len(produced) - sum(ecs) / len(ecs)) / (
        sum(ecs) / len(ecs)
    )

    assert rmse <= VALIDATION_TABLE_6_10["RMSE"] * 1.20
    assert -0.05 <= avg_delta <= 0.05
    assert throughput["avg_annual_production"] == pytest.approx(738_432, rel=0.02)


def test_bom_order_up_to_mode_passes_table_6_10_production_gate() -> None:
    sim = MFSCSimulation(
        shifts=1,
        risks_enabled=False,
        seed=42,
        horizon=SIMULATION_HORIZON,
        year_basis="thesis",
        deterministic_baseline=True,
        warmup_trigger="op9_arrival",
        downstream_q_source=THESIS_FAITHFUL_PROTOCOL["downstream_q_source"],
        r14_defect_mode=THESIS_FAITHFUL_PROTOCOL["r14_defect_mode"],
        raw_material_flow_mode="bom_total_units_order_up_to",
        raw_material_order_up_to_multiplier=2.0,
    ).run()
    throughput = sim.get_annual_throughput(start_time=sim.warmup_time, num_years=8)

    assert throughput["avg_annual_production"] == pytest.approx(738_432, rel=0.02)
    inventory = sim._inventory_detail()
    assert inventory["raw_material_al"] < 1_000_000


def test_kit_equivalent_order_up_to_alias_passes_table_6_10_production_gate() -> None:
    sim = MFSCSimulation(
        shifts=1,
        risks_enabled=False,
        seed=42,
        horizon=SIMULATION_HORIZON,
        year_basis="thesis",
        deterministic_baseline=True,
        warmup_trigger="op9_arrival",
        downstream_q_source=THESIS_FAITHFUL_PROTOCOL["downstream_q_source"],
        r14_defect_mode=THESIS_FAITHFUL_PROTOCOL["r14_defect_mode"],
        raw_material_flow_mode="kit_equivalent_order_up_to",
        raw_material_order_up_to_multiplier=2.0,
    ).run()
    throughput = sim.get_annual_throughput(start_time=sim.warmup_time, num_years=8)

    assert throughput["avg_annual_production"] == pytest.approx(738_432, rel=0.02)


def test_thesis_faithful_launcher_writes_auditable_artifacts(tmp_path) -> None:
    subprocess.run(
        [
            sys.executable,
            "scripts/run_thesis_faithful.py",
            "--label",
            "pytest_cf0",
            "--scenario",
            "cf0",
            "--output-root",
            str(tmp_path),
        ],
        check=True,
    )
    run_dir = tmp_path / "pytest_cf0"
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))

    assert status["status"] == "SUCCEEDED"
    assert manifest["protocol"]["protocol"] == "thesis_1to1"
    assert manifest["protocol"]["rl_enabled"] is False
    assert manifest["protocol"]["year_basis"] == "thesis"
    assert summary["scenario"] == "cf0"
    assert summary["runs"][0]["warmup_trigger"] == "op9_arrival"
    assert summary["runs"][0]["year_basis"] == "thesis"
    assert (run_dir / "command.txt").exists()
    assert (run_dir / "pid.txt").exists()
    assert (run_dir / "orders_seed_42.csv").exists()
    assert (run_dir / "risk_events_seed_42.csv").exists()


def test_table_6_10_reporter_writes_year_by_year_artifacts(tmp_path) -> None:
    subprocess.run(
        [
            sys.executable,
            "scripts/report_table_6_10_reproduction.py",
            "--label",
            "pytest_table_6_10",
            "--output-root",
            str(tmp_path),
            "--raw-material-flow-mode",
            "kit_equivalent_order_up_to",
        ],
        check=True,
    )
    run_dir = tmp_path / "pytest_table_6_10"
    summary = json.loads(
        (run_dir / "table_6_10_comparison.json").read_text(encoding="utf-8")
    )["summary"]

    assert (run_dir / "TABLE_6_10_REPRODUCTION.md").exists()
    assert (run_dir / "table_6_10_comparison.csv").exists()
    assert summary["avg_python_production"] == pytest.approx(738_432, rel=0.02)
    assert summary["rmse_vs_thesis_ecs"] <= VALIDATION_TABLE_6_10["RMSE"]


def test_thesis_decision_tables_reporter_writes_match_artifacts(tmp_path) -> None:
    subprocess.run(
        [
            sys.executable,
            "scripts/report_thesis_decision_tables.py",
            "--label",
            "pytest_decision_tables",
            "--output-root",
            str(tmp_path),
        ],
        check=True,
    )
    run_dir = tmp_path / "pytest_decision_tables"
    payload = json.loads(
        (run_dir / "thesis_decision_tables.json").read_text(encoding="utf-8")
    )

    assert payload["status"] == "PASS"
    assert payload["num_raw_materials"] == 12
    assert len(payload["inventory"]) == 15
    assert len(payload["capacity"]) == 24
    assert (run_dir / "THESIS_DECISION_TABLES.md").exists()
    assert (run_dir / "thesis_decision_tables.csv").exists()


def test_thesis_risk_tables_reporter_writes_match_artifacts(tmp_path) -> None:
    subprocess.run(
        [
            sys.executable,
            "scripts/report_thesis_risk_tables.py",
            "--label",
            "pytest_risk_tables",
            "--output-root",
            str(tmp_path),
        ],
        check=True,
    )
    run_dir = tmp_path / "pytest_risk_tables"
    payload = json.loads(
        (run_dir / "thesis_risk_tables.json").read_text(encoding="utf-8")
    )

    assert payload["status"] == "PASS"
    assert len(payload["rows"]) == 18
    assert {row["level"] for row in payload["rows"]} == {"current", "increased"}
    assert (run_dir / "THESIS_RISK_TABLES.md").exists()
    assert (run_dir / "thesis_risk_tables.csv").exists()


def test_thesis_operations_table_reporter_writes_match_artifacts(tmp_path) -> None:
    subprocess.run(
        [
            sys.executable,
            "scripts/report_thesis_operations_table.py",
            "--label",
            "pytest_operations_table",
            "--output-root",
            str(tmp_path),
        ],
        check=True,
    )
    run_dir = tmp_path / "pytest_operations_table"
    payload = json.loads(
        (run_dir / "thesis_operations_table.json").read_text(encoding="utf-8")
    )

    assert payload["status"] == "PASS"
    assert len(payload["rows"]) == 85
    assert {row["section"] for row in payload["rows"]} == {
        "demand",
        "downstream_q",
        "operations",
        "time_constants",
    }
    assert (run_dir / "THESIS_OPERATIONS_BACKBONE.md").exists()
    assert (run_dir / "thesis_operations_table.csv").exists()


def test_thesis_design_matrix_reporter_writes_match_artifacts(tmp_path) -> None:
    subprocess.run(
        [
            sys.executable,
            "scripts/report_thesis_design_matrix.py",
            "--label",
            "pytest_design_matrix",
            "--output-root",
            str(tmp_path),
        ],
        check=True,
    )
    run_dir = tmp_path / "pytest_design_matrix"
    payload = json.loads(
        (run_dir / "thesis_design_matrix.json").read_text(encoding="utf-8")
    )
    summary = payload["summary"]

    assert summary["status"] == "PASS"
    assert summary["row_count"] == 90
    assert summary["mismatch_count"] == 0
    assert summary["family_counts"] == {
        "capacity": 30,
        "inventory": 30,
        "risk_r1": 10,
        "risk_r2": 10,
        "risk_r3": 10,
    }
    assert summary["horizon_year_counts"] == {"10": 60, "20": 30}
    assert (run_dir / "THESIS_DESIGN_MATRIX.md").exists()
    assert (run_dir / "thesis_design_matrix.csv").exists()


def test_thesis_ret_schema_reporter_writes_match_artifacts(tmp_path) -> None:
    subprocess.run(
        [
            sys.executable,
            "scripts/report_thesis_ret_schema.py",
            "--label",
            "pytest_ret_schema",
            "--output-root",
            str(tmp_path),
        ],
        check=True,
    )
    run_dir = tmp_path / "pytest_ret_schema"
    payload = json.loads(
        (run_dir / "thesis_ret_schema.json").read_text(encoding="utf-8")
    )

    assert payload["status"] == "PASS"
    assert len(payload["rows"]) == 23
    assert {row["section"] for row in payload["rows"]} == {
        "ret_cases",
        "ret_weights",
        "sdm_schema",
    }
    assert sum(row["section"] == "sdm_schema" for row in payload["rows"]) == 12
    assert sum(row["section"] == "ret_cases" for row in payload["rows"]) == 5
    assert (run_dir / "THESIS_RET_SCHEMA.md").exists()
    assert (run_dir / "thesis_ret_schema.csv").exists()


def test_thesis_bom_semantics_reporter_writes_match_artifacts(tmp_path) -> None:
    subprocess.run(
        [
            sys.executable,
            "scripts/report_thesis_bom_semantics.py",
            "--label",
            "pytest_bom_semantics",
            "--output-root",
            str(tmp_path),
        ],
        check=True,
    )
    run_dir = tmp_path / "pytest_bom_semantics"
    payload = json.loads(
        (run_dir / "thesis_bom_semantics.json").read_text(encoding="utf-8")
    )

    assert payload["status"] == "PASS"
    assert len(payload["rows"]) == 16
    assert sum(row["section"] == "table_6_1_component" for row in payload["rows"]) == 12
    assert sum(row["section"] == "flow_semantics" for row in payload["rows"]) == 4
    assert (run_dir / "THESIS_BOM_SEMANTICS.md").exists()
    assert (run_dir / "thesis_bom_semantics.csv").exists()


def test_thesis_aligned_training_env_is_trainable_but_not_1to1() -> None:
    spec = get_thesis_aligned_training_env_spec()
    env = make_thesis_aligned_training_env(max_steps=1)
    obs, info = env.reset(seed=42)

    assert spec.env_variant == "thesis_aligned_training"
    assert spec.action_fields[-1] == "assembly_shift_signal"
    assert env.action_space.shape == (6,)
    assert obs.shape == env.observation_space.shape
    assert info["training_protocol"] == "thesis_aligned_gym"
    assert info["year_basis"] == "thesis"
    assert info["warmup_metadata"]["priming_enabled"] is False
    assert info["warmup_metadata"]["warmup_trigger"] == "op9_arrival"
    assert info["warmup_metadata"]["sim_warmup_time"] >= 943.0


def test_thesis_aligned_static_gate_writes_policy_artifacts(tmp_path) -> None:
    subprocess.run(
        [
            sys.executable,
            "scripts/run_thesis_aligned_static_gate.py",
            "--label",
            "pytest_static_gate",
            "--output-root",
            str(tmp_path),
            "--eval-steps",
            "1",
        ],
        check=True,
    )
    run_dir = tmp_path / "pytest_static_gate"
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    assert manifest["env_spec"]["env_variant"] == "thesis_aligned_training"
    assert len(summary["policy_summary"]) == 3
    assert (run_dir / "episode_metrics.csv").exists()
    assert (run_dir / "policy_summary.csv").exists()


def test_thesis_factorial_design_matrix_maps_canonical_rows() -> None:
    assert parse_cf_range("1-3,47,85") == [1, 2, 3, 47, 85]

    cf1 = design_spec_for_cfi(1)
    assert cf1.enabled_risks == ("R11", "R12", "R13", "R14")
    assert cf1.risk_overrides == {
        "R11": "current",
        "R12": "current",
        "R13": "increased",
        "R14": "increased",
    }
    assert cf1.horizon_hours == 10 * HOURS_PER_YEAR_THESIS

    cf47 = design_spec_for_cfi(47)
    assert cf47.source_cfi == 17
    assert cf47.family == "inventory"
    assert cf47.inventory_replenishment_period == 168.0
    assert cf47.risk_overrides == {
        "R21": "current",
        "R22": "increased",
        "R23": "increased",
        "R24": "current",
    }

    cf85 = design_spec_for_cfi(85)
    assert cf85.source_cfi == 25
    assert cf85.enabled_risks == ("R3",)
    assert cf85.risk_overrides == {"R3": "increased"}
    assert cf85.shifts == 2
    assert cf85.horizon_hours == 20 * HOURS_PER_YEAR_THESIS
    assert json.loads(spec_row(cf85)["risk_overrides"]) == {"R3": "increased"}


def test_thesis_factorial_launcher_dry_run_writes_design_matrix(tmp_path) -> None:
    subprocess.run(
        [
            sys.executable,
            "scripts/run_thesis_factorial.py",
            "--cfi",
            "1",
            "47",
            "85",
            "--label",
            "pytest_factorial_dry",
            "--output-root",
            str(tmp_path),
            "--dry-run",
        ],
        check=True,
    )
    run_dir = tmp_path / "pytest_factorial_dry"
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    matrix_text = (run_dir / "design_matrix.csv").read_text(encoding="utf-8")

    assert status["status"] == "SUCCEEDED"
    assert "Cf85" not in matrix_text
    assert "85,capacity,25,R3" in matrix_text
