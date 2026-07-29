from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.aggregate_q_r1_matched_retention_factorial_v4 import load_workers
from scripts.run_q_r1_matched_retention_factorial_v4 import (
    development_timesteps,
    validate_full_screen_selection,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "contracts/q_r1_matched_retention_factorial_v4.json"
AMENDMENT_PATH = (
    ROOT / "contracts/q_r1_factorial_v4_full_phase_runner_amendment_v1.json"
)


def test_phase_budget_uses_the_already_frozen_contract_values() -> None:
    contract = json.loads(CONTRACT_PATH.read_text())
    assert development_timesteps(contract, "screen") == 96_000
    assert development_timesteps(contract, "full") == 240_000
    with pytest.raises(ValueError, match="unknown development phase"):
        development_timesteps(contract, "confirmation")


def test_full_phase_amendment_does_not_change_scientific_settings() -> None:
    amendment = json.loads(AMENDMENT_PATH.read_text())
    assert amendment["base_contract_sha256"] == (
        "bb92a2cbfcd3691a77f7f9ab8a269d7ffab65823d37b41f70d0b13795d92e764"
    )
    assert amendment["screen_selection"]["advanced_config_ids"] == ["s07", "s06"]
    assert amendment["full_phase"]["timesteps_per_seed"] == 240_000
    assert amendment["full_phase"]["expected_checkpoint_timesteps"] == list(
        range(0, 240_001, 24_000)
    )
    assert not any(amendment["scientific_invariants"].values())


def test_full_worker_requires_a_selected_configuration(tmp_path: Path) -> None:
    contract = json.loads(CONTRACT_PATH.read_text())
    selection = {
        "phase": "screen",
        "contract_sha256": (
            "bb92a2cbfcd3691a77f7f9ab8a269d7ffab65823d37b41f70d0b13795d92e764"
        ),
        "advanced_config_ids": ["s07", "s06"],
    }
    path = tmp_path / "screen_selection.json"
    path.write_text(json.dumps(selection))
    assert validate_full_screen_selection(
        path, contract=contract, config_id="s07"
    ) == selection
    with pytest.raises(RuntimeError, match="did not advance"):
        validate_full_screen_selection(path, contract=contract, config_id="s05")


def test_full_aggregation_rejects_a_96000_timestep_screen_worker(
    tmp_path: Path,
) -> None:
    source = next(
        Path(
            "/Users/thom/Projects/research/scres-ia-runs/"
            "q_r1_factorial_v4_shared_screen_v1"
        ).glob("*/result.json")
    )
    payload = json.loads(source.read_text())
    copied = tmp_path / "result.json"
    copied.write_text(json.dumps(payload))
    assert len(load_workers([copied], expected_phase="screen")) == 1
    with pytest.raises(ValueError, match="expected full"):
        load_workers([copied], expected_phase="full")


def test_full_aggregation_accepts_only_the_exact_240000_schedule(
    tmp_path: Path,
) -> None:
    source = next(
        Path(
            "/Users/thom/Projects/research/scres-ia-runs/"
            "q_r1_factorial_v4_shared_screen_v1"
        ).glob("*/result.json")
    )
    payload = json.loads(source.read_text())
    payload["development_phase"] = "full"
    template = copy.deepcopy(payload["checkpoints"][-1])
    payload["checkpoints"] = [
        {**template, "timesteps": step} for step in range(0, 240_001, 24_000)
    ]
    copied = tmp_path / "result.json"
    copied.write_text(json.dumps(payload))
    assert len(load_workers([copied], expected_phase="full")) == 1
    payload["checkpoints"].pop()
    copied.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="checkpoint schedule"):
        load_workers([copied], expected_phase="full")
