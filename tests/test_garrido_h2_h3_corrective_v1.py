from __future__ import annotations

import json
from pathlib import Path

import pytest

from supply_chain.config import INVENTORY_BUFFERS
from supply_chain.garrido_thesis_design import DESIGN, THESIS_SEEDS
from supply_chain.supply_chain import MFSCSimulation


ROOT = Path(__file__).resolve().parent.parent
CONTRACT = json.loads(
    (ROOT / "contracts" / "garrido_h2_h3_corrective_v1.json").read_text()
)


def test_design_has_exact_90_configs_and_correct_cf2_seed() -> None:
    assert set(DESIGN) == set(range(1, 91))
    assert THESIS_SEEDS[2] == 91
    for base in range(1, 31):
        baseline = DESIGN[base]
        buffered = DESIGN[base + 30]
        shifted = DESIGN[base + 60]
        assert buffered.base_index == shifted.base_index == base
        assert buffered.risk_pattern == shifted.risk_pattern == baseline.risk_pattern
        assert buffered.risk_family == shifted.risk_family == baseline.risk_family
        assert buffered.horizon_hours == shifted.horizon_hours == baseline.horizon_hours


def test_contract_has_twelve_unique_tape_roots_and_common_start() -> None:
    roots = CONTRACT["execution"]["tape_roots"]
    assert len(roots) == len(set(roots)) == 12
    assert CONTRACT["execution"]["common_evaluation_start_hours"] == 2016
    assert CONTRACT["inference"]["orders_as_independent_replicates"] is False


def test_invalid_assembly_batch_release_mode_fails_closed() -> None:
    with pytest.raises(ValueError, match="assembly_batch_release_mode"):
        MFSCSimulation(assembly_batch_release_mode="not-a-mode")


def test_default_batch_release_mode_is_identity() -> None:
    kwargs = dict(seed=8181, horizon=2_000, risks_enabled=False)
    historical = MFSCSimulation(**kwargs)
    explicit = MFSCSimulation(
        **kwargs, assembly_batch_release_mode="threshold_immediate"
    )
    historical.step(action=None, step_hours=2_000)
    explicit.step(action=None, step_hours=2_000)
    assert historical.material_availability_events == explicit.material_availability_events
    assert historical.delivery_events == explicit.delivery_events


def test_periodic_buffer_replenishes_after_initial_injection() -> None:
    level = INVENTORY_BUFFERS[168]
    buffers = {
        "op3_rm": float(level["op3_rm"]),
        "op5_rm": float(level["op5_rm"]),
        "op9_rations": float(level["op9_rations"]),
    }
    sim = MFSCSimulation(
        initial_buffers=buffers,
        inventory_replenishment_period=168.0,
        seed=7272,
        horizon=2_500,
        risks_enabled=False,
        raw_material_flow_mode="bom_total_units",
        raw_material_order_up_to_multiplier=1.0,
    )
    initial = (
        12.0 * (buffers["op3_rm"] + buffers["op5_rm"])
        + buffers["op9_rations"]
    )
    sim.step(action=None, step_hours=2_500)
    injected = (
        sim.total_strategic_raw_injected
        + sim.total_strategic_rations_injected
    )
    assert injected > initial


def test_table_6_20_trace_preflight_passes() -> None:
    from scripts.run_garrido_h2_h3_corrective_v1 import trace_preflight

    result = trace_preflight(CONTRACT)
    assert result["status"] == "PASS_TABLE_6_20_TRACE"
    assert [row["op7_median_gap"] for row in result["rows"]] == [48.0, 24.0, 24.0]
    assert [row["op3_median_start_gap"] for row in result["rows"]] == [168.0] * 3
