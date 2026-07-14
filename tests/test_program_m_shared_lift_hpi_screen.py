from __future__ import annotations

from dataclasses import asdict
import json

import pytest

from scripts.screen_program_m_shared_lift_hpi import (
    Cell,
    adjacent,
    all_calendars,
    build_run_contract,
    candidate_regions,
    execute,
    frozen_cells,
    generate_exogenous_tape,
    summarize_cell,
    validate_resume,
)


SYNTHETIC_SEEDS = (91, 92)


def _fake_shard(cell: Cell, seed: int, *, preferred: int) -> dict:
    rows = []
    for index, calendar in enumerate(all_calendars()):
        ret = 0.80 + (0.02 if index == preferred else 0.0)
        rows.append(
            {
                "calendar_index": index,
                "calendar": list(calendar),
                "ret_request_snapshot_v2": ret,
                "quantity_ret_request_snapshot_v2": ret,
                "service_loss_auc_ration_hours": 100.0 - ret,
                "attended_orders": 10,
                "lost_orders": 0,
                "remaining_backlog_quantity": 0.0,
                "remaining_backlog_orders": 0,
                "maximum_backlog_age_hours": 0.0,
                "worst_cssu_fill": 0.9,
                "ret_cvar05": ret,
                "mass_residual": 0.0,
                "reserved_slots": 8,
                "reserved_payload_capacity_rations": 20_800.0,
                "reserved_vehicle_hours": 384.0,
                "protected_loaded_departures": 1,
                "protected_empty_departures": 7,
                "protected_actual_payload_rations": 2_500.0,
                "protected_actual_loaded_vehicle_hours": 48.0,
                "base_terrestrial_departures": 9,
                "base_terrestrial_payload_rations": 22_500.0,
                "base_terrestrial_vehicle_hours": 432.0,
                "risk_sha256": "r",
                "demand_sha256": "d",
            }
        )
    return {
        "schema_version": "program_m_shared_lift_hpi_raw_shard_v1",
        "cell": asdict(cell),
        "seed": seed,
        "warmup_time": 100.0,
        "tape": {"risk_sha256": "r"},
        "n_calendars": 256,
        "evaluations": rows,
    }


def test_frozen_portfolio_has_one_null_eighteen_positive_and_exact_frontier():
    cells = frozen_cells()
    assert len(cells) == 19
    assert sum(cell.is_null for cell in cells) == 1
    assert sum(not cell.is_null for cell in cells) == 18
    assert len(all_calendars()) == 256
    assert len(set(all_calendars())) == 256
    assert all(len(calendar) == 8 for calendar in all_calendars())


def test_tape_is_deterministic_nested_and_start_location_do_not_depend_on_cell():
    cells = {cell.cell_id: cell for cell in frozen_cells()}
    low = cells["h25_d24_s70"]
    high = cells["h75_d120_s85"]
    anchor = 321.5
    low_tape = generate_exogenous_tape(seed=91, cell=low, decision_start_time=anchor)
    repeated = generate_exogenous_tape(seed=91, cell=low, decision_start_time=anchor)
    high_tape = generate_exogenous_tape(seed=91, cell=high, decision_start_time=anchor)
    assert low_tape == repeated
    low_weeks = {row["week"] for row in low_tape["draws"] if row["hazard_u"] < low.hazard}
    high_weeks = {row["week"] for row in high_tape["draws"] if row["hazard_u"] < high.hazard}
    assert low_weeks <= high_weeks
    for left, right in zip(low_tape["draws"], high_tape["draws"]):
        assert (left["location_u"], left["start_u"], left["start_time"]) == (
            right["location_u"], right["start_u"], right["start_time"]
        )
        assert anchor + left["week"] * 168 + 24 <= left["start_time"]
        assert left["start_time"] < anchor + left["week"] * 168 + 168


def test_null_cell_has_no_events():
    null = next(cell for cell in frozen_cells() if cell.is_null)
    tape = generate_exogenous_tape(seed=92, cell=null, decision_start_time=400.0)
    assert tape["risk_events"] == []
    assert len(tape["draws"]) == 8


def test_adjacency_is_exactly_one_neighboring_factor_step():
    cells = {cell.cell_id: cell for cell in frozen_cells()}
    assert adjacent(cells["h25_d24_s70"], cells["h50_d24_s70"])
    assert adjacent(cells["h25_d24_s70"], cells["h25_d72_s70"])
    assert adjacent(cells["h25_d24_s70"], cells["h25_d24_s85"])
    assert not adjacent(cells["h25_d24_s70"], cells["h75_d24_s70"])
    assert not adjacent(cells["h25_d24_s70"], cells["h50_d72_s70"])
    assert not adjacent(next(cell for cell in frozen_cells() if cell.is_null), cells["h25_d24_s85"])


def test_summary_reselects_complete_static_frontier_and_reports_no_ci():
    cell = next(cell for cell in frozen_cells() if cell.cell_id == "h25_d24_s70")
    summary = summarize_cell(
        [_fake_shard(cell, SYNTHETIC_SEEDS[0], preferred=3), _fake_shard(cell, SYNTHETIC_SEEDS[1], preferred=5)]
    )
    assert summary["n_calendars"] == 256
    assert summary["best_static_calendar_index"] == 3
    assert summary["unique_oracle_calendars"] == 2
    assert summary["h_pi_mean"] == pytest.approx(0.01)
    assert summary["tape_tail_cvar05"]["tail_n"] == 1
    assert "paired_h_pi_lower_tail_mean" in summary["tape_tail_cvar05"]
    assert not any("ci" in key.lower() or "lcb" in key.lower() for key in summary)


def test_candidate_region_requires_size_hazard_and_duration_span():
    passing = {"h25_d24_s70", "h50_d24_s70", "h50_d72_s70"}
    rows = [
        {"cell_id": cell.cell_id, "h_pi_mean": 0.011 if cell.cell_id in passing else 0.0}
        for cell in frozen_cells()
    ]
    assert candidate_regions(rows) == [sorted(passing)]
    rows[1]["h_pi_mean"] = 0.0


def test_tiny_monkeypatched_execute_is_atomic_complete_and_resume_validates(tmp_path):
    # Synthetic seeds only: this test never opens any project screen block.
    def fake_evaluator(task):
        cell = Cell(**task["cell"])
        preferred = 3 if int(task["seed"]) == SYNTHETIC_SEEDS[0] else 5
        return _fake_shard(cell, int(task["seed"]), preferred=preferred)

    run_dir = tmp_path / "tiny"
    result = execute(
        run_dir=run_dir,
        seeds=SYNTHETIC_SEEDS,
        workers=1,
        resume=False,
        evaluator=fake_evaluator,
    )
    assert result["status"] == "SCREEN_COMPLETE_SELECTION_ONLY__NO_CI_NO_PROMOTION"
    assert result["n_des_evaluations"] == 19 * 2 * 256
    progress = json.loads((run_dir / "progress.json").read_text())
    assert progress["complete"] is True
    assert progress["completed_count"] == 38
    validate_resume(run_dir, build_run_contract(seeds=SYNTHETIC_SEEDS))
    resumed = execute(
        run_dir=run_dir,
        seeds=SYNTHETIC_SEEDS,
        workers=1,
        resume=True,
        evaluator=fake_evaluator,
    )
    assert resumed["n_des_evaluations"] == result["n_des_evaluations"]

    shard = run_dir / progress["completed_shards"][0]["path"]
    shard.write_text("{}\n")
    with pytest.raises(RuntimeError, match="custody mismatch"):
        validate_resume(run_dir, build_run_contract(seeds=SYNTHETIC_SEEDS))


def test_run_contract_freezes_metric_sources_and_absolute_warmup_rule():
    payload = build_run_contract(seeds=SYNTHETIC_SEEDS)
    assert payload["governing_metric"] == "ret_excel_request_snapshot_v2"
    assert payload["seed_role"] == "synthetic_test_or_explicit_noncanonical"
    assert "absolute" in payload["treatment_start_rule"]
    assert "supply_chain/program_m_shared_lift.py" in payload["source_sha256"]
    assert payload["content_sha256"]
