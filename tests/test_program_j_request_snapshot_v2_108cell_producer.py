import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts.produce_program_j_request_snapshot_v2_108cell_frontier import (
    ALL_SEQUENCES,
    FIELDS,
    GUARDRAIL_DIRECTIONS,
    REFERENCE,
    RESOURCE_FIELDS,
    SCHEDULED_PM_HOURS,
    all_cells,
    arrays_from_matrix,
    guardrail_report,
    json_sha256,
    load_raw_shard,
    resource_envelope,
    run_contract,
    run_sequence_with_exact_crew_ledger,
    selected_sequences,
    solve_resource_only_pi,
    solve_resource_only_static,
    write_raw_shard,
)
from scripts.watch_program_j_request_snapshot_v2_108cell import (
    session_processes,
    snapshot,
)


def synthetic_arrays() -> dict[str, np.ndarray]:
    shape = (2, 3)
    arrays = {field: np.zeros(shape, dtype=float) for field in FIELDS}
    arrays["ret_request_snapshot_v2"][:] = np.asarray(
        [[0.1, 0.9, 1.0], [0.1, 0.9, 1.0]]
    )
    arrays["scheduled_pm_hours"][:] = SCHEDULED_PM_HOURS
    arrays["executed_pm_hours"][:] = SCHEDULED_PM_HOURS
    arrays["corrective_hours"][:] = np.asarray(
        [[2.0, 1.0, 3.0], [2.0, 1.0, 3.0]]
    )
    arrays["total_crew_hours"][:] = (
        arrays["executed_pm_hours"] + arrays["corrective_hours"]
    )
    # Schedule one is resource-superior and highest among feasible schedules,
    # but intentionally awful on outcome guardrails.
    arrays["lost_orders"][:, 1] = 100.0
    arrays["service_loss_auc"][:, 1] = 1_000.0
    arrays["ret_quantity"][:, 0] = 0.8
    arrays["ret_quantity"][:, 1] = 0.1
    arrays["ret_cvar05"][:, 0] = 0.8
    arrays["ret_cvar05"][:, 1] = 0.1
    arrays["ret_cvar10"][:, 0] = 0.8
    arrays["ret_cvar10"][:, 1] = 0.1
    return arrays


def test_full_design_is_exactly_108_cells_by_all_3_to_8_schedules():
    cells = all_cells()

    assert len(cells) == 3 * 3 * 3 * 2 * 2 == 108
    assert len({row["cell_id"] for row in cells}) == 108
    assert [row["cell_index"] for row in cells] == list(range(108))
    assert len(ALL_SEQUENCES) == 3**8 == 6561
    assert REFERENCE in ALL_SEQUENCES
    assert 108 * 4 * len(ALL_SEQUENCES) == 2_834_352


def test_smoke_sequence_subset_retains_resource_reference():
    sequences = selected_sequences(smoke=True, smoke_schedules=3)

    assert len(sequences) == 3
    assert REFERENCE in sequences
    assert len(set(sequences)) == len(sequences)


def test_resource_only_optimizers_do_not_use_outcome_guardrails():
    arrays = synthetic_arrays()
    envelope = resource_envelope(arrays, reference_index=0)
    static = solve_resource_only_static(arrays, envelope)
    pi = solve_resource_only_pi(arrays, envelope)

    assert RESOURCE_FIELDS == ("total_crew_hours",)
    assert static["deterministic_index"] == 1
    assert np.argmax(static["weights"]) == 1
    # The aggregate resource-matched PI can spend the saved corrective hour on
    # one tape to use the otherwise-infeasible higher-ReT schedule on another.
    assert sorted(np.argmax(pi["weights"], axis=1).tolist()) == [1, 2]
    assert np.isclose(
        pi["expected"]["total_crew_hours"].mean(),
        envelope["B_total_crew_hours"],
    )
    assert np.isclose(pi["expected"]["ret_request_snapshot_v2"].mean(), 0.95)
    report = guardrail_report(static["expected"], {
        field: values[:, 0] for field, values in arrays.items()
    })
    assert report["optimization_use"] == "REPORT_ONLY_NOT_AN_OPTIMIZATION_CONSTRAINT"
    assert report["all_point_noninferiority"] is False
    assert set(report["rows"]) == set(GUARDRAIL_DIRECTIONS)


def test_raw_shard_is_immutable_checksummed_and_schema_verified(tmp_path):
    sequences = selected_sequences(smoke=True, smoke_schedules=3)
    matrix = np.arange(len(sequences) * len(FIELDS), dtype=float).reshape(
        len(sequences), len(FIELDS)
    )
    row = {
        "cell_index": 0,
        "cell_id": "J000",
        "cell": {
            key: value
            for key, value in all_cells()[0].items()
            if key not in {"cell_index", "cell_id"}
        },
        "seed": 1_200_001,
        "base_exogenous_sha256": "a" * 64,
        "matrix": matrix,
    }
    path = tmp_path / "raw.npz"
    record = write_raw_shard(path, row, sequences)

    assert load_raw_shard(record, sequences=sequences).tolist() == matrix.tolist()
    with pytest.raises(FileExistsError):
        write_raw_shard(path, row, sequences)
    assert set(arrays_from_matrix(matrix[np.newaxis, :, :])) == set(FIELDS)


def test_exact_crew_ledger_includes_completed_and_active_partial_work():
    from supply_chain.maintenance_control import materialize_tape

    tape = materialize_tape(1_200_001, weeks=8)
    cell = {
        key: value
        for key, value in all_cells()[0].items()
        if key not in {"cell_index", "cell_id"}
    }
    outcome = run_sequence_with_exact_crew_ledger(tape, REFERENCE, cell=cell)

    assert outcome["ret_excel_contract_version"] == "ret_excel_request_snapshot_v2"
    assert np.isclose(
        outcome["total_in_horizon_crew_hours"],
        outcome["executed_pm_hours"]
        + outcome["corrective_hours"]
        + outcome["active_crew_partial_hours"],
    )
    assert outcome["crew_queue_at_horizon"] >= 0.0


def test_cli_smoke_checkpoints_shards_and_refuses_overwrite(tmp_path):
    work = tmp_path / "work"
    progress = tmp_path / "progress.json"
    verdict = tmp_path / "verdict.json"
    command = [
        ".venv/bin/python",
        "scripts/produce_program_j_request_snapshot_v2_108cell_frontier.py",
        "--smoke",
        "--smoke-cells",
        "1",
        "--smoke-tapes",
        "1",
        "--smoke-schedules",
        "3",
        "--workers",
        "1",
        "--work-root",
        str(work),
        "--progress",
        str(progress),
        "--verdict",
        str(verdict),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)

    assert completed.returncode == 0, completed.stderr
    result = json.loads(verdict.read_text())
    checkpoint = json.loads(progress.read_text())
    assert result["scientific_status"] == "SMOKE_ONLY_NO_SCIENTIFIC_EVIDENCE"
    assert result["scientific_claim"] is None
    assert result["execution"]["virgin_tapes_opened"] is False
    assert result["run_contract"]["sequence_count"] == 3
    assert result["run_contract"]["cell_count"] == 1
    assert result["run_contract"]["tape_status"] == (
        "BURNED_DEVELOPMENT_SCREEN_TAPES_NOT_VIRGIN"
    )
    assert result["run_contract"]["guardrail_coverage"]["promotion_block"]
    assert result["screen_decision_rule"]["learner_authorized_by_this_screen"] is False
    assert result["run_contract"]["optimization_rule"].endswith(
        "Outcome guardrails never constrain or select the comparator or PI."
    )
    assert result["cell_results"][0]["outcome_guardrails_used_in_optimization"] is False
    assert checkpoint["state"] == "complete"
    assert checkpoint["completed_raw_shards"] == 1
    assert checkpoint["completed_cells"] == 1
    assert Path(checkpoint["raw_shards"][0]["path"]).exists()
    frozen_path = Path(checkpoint["frozen_run_contract"]["path"])
    frozen = json.loads(frozen_path.read_text())
    frozen_content = frozen.pop("content_sha256")
    assert frozen_content == json_sha256(frozen)
    assert frozen == result["run_contract"]
    unhashed = dict(result)
    stored = unhashed.pop("content_sha256")
    assert stored == json_sha256(unhashed)

    repeated = subprocess.run(command, check=False, capture_output=True, text=True)
    assert repeated.returncode != 0
    assert "refusing to overwrite final verdict" in repeated.stderr


def test_cli_accepts_only_matching_precreated_contract(tmp_path):
    work = tmp_path / "work"
    progress = tmp_path / "progress.json"
    verdict = tmp_path / "verdict.json"
    frozen_path = tmp_path / "frozen.json"
    cells = all_cells()[:1]
    sequences = selected_sequences(smoke=True, smoke_schedules=3)
    frozen = run_contract(
        cells=cells,
        sequences=sequences,
        seed_start=1_200_001,
        tapes_per_cell=1,
        smoke=True,
    )
    frozen["content_sha256"] = json_sha256(frozen)
    frozen_path.write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n")
    command = [
        ".venv/bin/python",
        "scripts/produce_program_j_request_snapshot_v2_108cell_frontier.py",
        "--smoke",
        "--smoke-cells",
        "1",
        "--smoke-tapes",
        "1",
        "--smoke-schedules",
        "3",
        "--workers",
        "1",
        "--work-root",
        str(work),
        "--progress",
        str(progress),
        "--run-contract",
        str(frozen_path),
        "--use-precreated-run-contract",
        "--verdict",
        str(verdict),
    ]

    completed = subprocess.run(command, check=False, capture_output=True, text=True)

    assert completed.returncode == 0, completed.stderr
    assert json.loads(progress.read_text())["state"] == "complete"


def test_watcher_covers_the_entire_session_and_attests_prestart(tmp_path):
    current_session = os.getsid(0)
    members = session_processes(current_session)

    if sys.platform.startswith("linux"):
        assert any(row["pid"] == os.getpid() for row in members)
    else:
        # The scientific watcher is a Linux/VPS contract; macOS ps does not
        # expose the POSIX session id through the same `sid` keyword.
        assert members == []
    payload = snapshot(
        tmp_path,
        watcher_started_at_utc="2026-07-14T00:00:00+00:00",
        peak_rss_bytes=0,
        minimum_memory_available_bytes=None,
    )
    assert payload["state"] == "watching_prestart"
    assert payload["watcher_scope"] == (
        "ENTIRE_SCIENTIFIC_SESSION_INCLUDING_REPARENTED_WORKERS"
    )
