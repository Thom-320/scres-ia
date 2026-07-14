from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.launch_program_m_shared_lift_hpi_validation import (
    MAX_WORKERS,
    VALIDATION_SEED_END,
    VALIDATION_SEED_START,
    file_sha256,
    scientific_command,
    validation_seed_manifest,
    write_prestart_manifests,
)


def test_validation_command_has_no_seed_override_and_caps_workers(tmp_path: Path) -> None:
    command = scientific_command(workers=MAX_WORKERS, scientific_dir=tmp_path / "science")
    assert command[-4:] == ["--run-dir", str(tmp_path / "science"), "--workers", "6"]
    assert "validate_program_m_shared_lift_hpi.py" in command[1]
    assert not any("seed" in token.lower() for token in command)
    with pytest.raises(ValueError, match="workers must be between"):
        scientific_command(workers=7, scientific_dir=tmp_path / "science")


def test_validation_seed_manifest_is_fixed_producer_owned_and_not_materialized() -> None:
    payload = validation_seed_manifest()
    assert payload["seed_start"] == VALIDATION_SEED_START == 7_300_025
    assert payload["seed_end"] == VALIDATION_SEED_END == 7_300_048
    assert payload["count"] == 24
    assert payload["use"] == "BURNED_H_PI_VALIDATION_ONLY"
    assert payload["state_at_prestart"] == "SEALED_UNTIL_PRODUCER_START"
    assert payload["virgin"] is False
    assert payload["launcher_override_permitted"] is False
    assert "seeds" not in payload


def test_prestart_manifests_bind_selection_sources_command_and_checksums(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "scripts.launch_program_m_shared_lift_hpi_validation.checked_output",
        lambda command: "package==1" if "freeze" in command else "",
    )
    command = scientific_command(workers=2, scientific_dir=tmp_path / "scientific_output")
    checksums = write_prestart_manifests(
        run_dir=tmp_path,
        head="b" * 40,
        status=[],
        command=command,
        workers=2,
        watch_interval_seconds=5.0,
        progress_stale_seconds=60.0,
    )
    expected = {
        "source_manifest.json",
        "contract_manifest.json",
        "selection_manifest.json",
        "environment_manifest.json",
        "seed_manifest.json",
        "command_manifest.json",
    }
    assert set(checksums["files"]) == expected
    for name in expected:
        assert checksums["files"][name] == file_sha256(tmp_path / name)

    selection = json.loads((tmp_path / "selection_manifest.json").read_text())
    assert selection["screen_result_sha256"] == (
        "54d26a8c2e8651159d33694ad56c311f7cef8e6483d9caeb1a449a14b14e8101"
    )
    assert selection["screen_completed_shards"] == 456
    assert selection["validation_seed_state"] == "SEALED_NOT_OPENED"
    assert len(selection["selected_cell_ids"]) == 6

    source = json.loads((tmp_path / "source_manifest.json").read_text())
    assert "scripts/validate_program_m_shared_lift_hpi.py" in source["files"]
    assert "scripts/watch_program_m_shared_lift_hpi.py" in source["files"]
    command_payload = json.loads((tmp_path / "command_manifest.json").read_text())
    assert command_payload["maximum_workers"] == 6
    assert command_payload["workers"] == 2
    assert not any("seed" in token.lower() for token in command_payload["command"])
