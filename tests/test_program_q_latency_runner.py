from __future__ import annotations

from pathlib import Path

import pytest

from scripts import run_program_q_latency_benchmark as latency


def test_latency_runner_uses_frozen_checkpoint_and_selected_family() -> None:
    assert latency.MODEL.name == "recurrent_ppo_seed_8101.zip"
    assert set(latency.SELECTED) == {
        "rho75_share90",
        "rho90_share75",
        "rho90_share90",
    }
    assert latency.SELECTED["rho90_share90"].config_id == "max_pressure__0"
    assert all(not 7_490_001 <= seed <= 7_490_256 for seed in latency.TAPE_SEEDS)


def test_latency_output_is_new_only(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    payload = {"schema_version": "test"}
    latency.write_result(output, payload)
    assert '"schema_version": "test"' in output.read_text()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        latency.write_result(output, payload)
