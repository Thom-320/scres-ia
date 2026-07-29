from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts.screen_program_m_shared_lift_hpi import all_calendars
from scripts.validate_program_m_shared_lift_hpi import (
    BOOTSTRAP_RNG_SEED,
    SCIENTIFIC_COMMIT,
    SCREEN_RESULT_SHA256,
    SELECTED_CELL_IDS,
    bootstrap_simultaneous_lcbs,
    build_run_contract,
    execute,
    observed_hpi,
    passing_components,
    select_least_observed,
    selected_cells,
    selection_artifact,
    validate_resume,
)


ROOT = Path(__file__).resolve().parent.parent
SYNTHETIC_SEEDS = (101, 102, 103, 104)


def test_cli_imports_repo_without_pythonpath() -> None:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/validate_program_m_shared_lift_hpi.py"), "--help"],
        cwd=ROOT.parent,
        env=environment,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert completed.returncode == 0, completed.stderr


def test_selection_binds_complete_screen_commit_cells_and_sealed_block() -> None:
    payload = selection_artifact()
    assert payload["scientific_commit"] == SCIENTIFIC_COMMIT
    assert payload["screen_result"]["sha256"] == SCREEN_RESULT_SHA256
    assert payload["screen_result"]["completed_shards"] == 456
    assert payload["screen_result"]["expected_shards"] == 456
    assert tuple(payload["selected_cell_ids"]) == SELECTED_CELL_IDS
    assert payload["h_pi_validation"]["state"] == "SEALED_NOT_OPENED"
    assert tuple(cell.cell_id for cell in selected_cells()) == SELECTED_CELL_IDS


def _matrix(static_index: int, oracle_indices: list[int], gain: float) -> np.ndarray:
    matrix = np.full((len(oracle_indices), 256), 0.5, dtype=float)
    matrix[:, static_index] = 0.6
    for row, oracle_index in enumerate(oracle_indices):
        matrix[row, oracle_index] = 0.6 + gain
    return matrix


def test_observed_hpi_uses_rowwise_oracle_and_lowest_index_static_tie() -> None:
    matrix = _matrix(3, [4, 5, 4, 5], 0.02)
    matrix[:, 2] = matrix[:, 3]
    result = observed_hpi(matrix)
    assert result["best_static_calendar_index"] == 2
    assert result["observed_h_pi"] == pytest.approx(0.02)
    assert result["unique_oracle_calendars"] == 2


def test_bootstrap_is_deterministic_paired_and_reselects_static() -> None:
    matrices = {
        "a": _matrix(3, [4, 5, 4, 5], 0.02),
        "b": _matrix(7, [8, 9, 8, 9], 0.03),
    }
    first = bootstrap_simultaneous_lcbs(matrices, n_resamples=200, rng_seed=BOOTSTRAP_RNG_SEED)
    second = bootstrap_simultaneous_lcbs(matrices, n_resamples=200, rng_seed=BOOTSTRAP_RNG_SEED)
    assert first == second
    assert first["static_reselected_in_every_cell_and_resample"] is True
    assert first["shared_paired_tape_indices_across_cells"] is True
    assert set(first["simultaneous_lcb95"]) == {"a", "b"}


def test_connected_region_and_least_observed_selection_use_frozen_graph() -> None:
    passing = {
        "h50_d120_s70",
        "h75_d120_s70",
        "h75_d72_s70",
    }
    rows = [
        {
            "cell_id": cell_id,
            "simultaneous_lcb95": 0.011 if cell_id in passing else 0.009,
            "observed_h_pi": {
                "h50_d120_s70": 0.014,
                "h75_d120_s70": 0.013,
                "h75_d72_s70": 0.012,
            }.get(cell_id, 0.02),
        }
        for cell_id in SELECTED_CELL_IDS
    ]
    components = passing_components(rows)
    assert components == [sorted(passing)]
    assert select_least_observed(rows, components) == "h75_d72_s70"


def _fake_shard(cell, seed: int) -> dict:
    preferred = (seed % 3) + 1
    rows = []
    for index, calendar in enumerate(all_calendars()):
        ret = 0.7 + (0.02 if index == preferred else 0.0)
        rows.append(
            {
                "calendar_index": index,
                "calendar": list(calendar),
                "ret_request_snapshot_v2": ret,
            }
        )
    return {
        "schema_version": "synthetic_program_m_validation_shard_v1",
        "cell": asdict(cell),
        "seed": seed,
        "n_calendars": 256,
        "evaluations": rows,
    }


def test_synthetic_execute_is_atomic_complete_and_resume_custody(tmp_path: Path) -> None:
    def fake_evaluator(task):
        cell = next(cell for cell in selected_cells() if cell.cell_id == task["cell"]["cell_id"])
        return _fake_shard(cell, int(task["seed"]))

    run_dir = tmp_path / "validation"
    result = execute(
        run_dir=run_dir,
        seeds=SYNTHETIC_SEEDS,
        workers=1,
        resume=False,
        evaluator=fake_evaluator,
        n_resamples=100,
    )
    assert result["n_des_evaluations"] == 6 * 4 * 256
    assert result["authorization"] == {
        "h_obs": False,
        "learner": False,
        "paper2": False,
        "paper3": False,
        "virgin_tapes": False,
    }
    progress = json.loads((run_dir / "progress.json").read_text())
    assert progress["complete"] is True
    assert progress["completed_count"] == 24
    contract = build_run_contract(seeds=SYNTHETIC_SEEDS)
    validate_resume(run_dir, contract)
    resumed = execute(
        run_dir=run_dir,
        seeds=SYNTHETIC_SEEDS,
        workers=1,
        resume=True,
        evaluator=fake_evaluator,
        n_resamples=100,
    )
    assert resumed["cell_results"] == result["cell_results"]
    shard = run_dir / progress["completed_shards"][0]["path"]
    shard.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="custody mismatch"):
        validate_resume(run_dir, contract)


def test_run_contract_freezes_sources_inference_and_noncanonical_test_role() -> None:
    payload = build_run_contract(seeds=SYNTHETIC_SEEDS)
    assert payload["seed_role"] == "synthetic_test_or_explicit_noncanonical"
    assert payload["scientific_commit"] == SCIENTIFIC_COMMIT
    assert payload["screen_result_sha256"] == SCREEN_RESULT_SHA256
    assert payload["inference"]["bootstrap_resamples"] == 10_000
    assert "scripts/validate_program_m_shared_lift_hpi.py" in payload["source_sha256"]
    assert len(payload["cells"]) == 6
    assert len(payload["calendars"]) == 256
