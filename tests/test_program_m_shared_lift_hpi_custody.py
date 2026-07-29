import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.launch_program_m_shared_lift_hpi import (
    MAX_WORKERS,
    SCREEN_SEED_END,
    SCREEN_SEED_START,
    file_sha256,
    scientific_command,
    screen_seed_manifest,
    write_prestart_manifests,
)
from scripts import watch_program_m_shared_lift_hpi as watcher


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_scientific_command_has_no_seed_override_and_caps_workers(tmp_path):
    command = scientific_command(workers=MAX_WORKERS, scientific_dir=tmp_path / "science")

    assert command[-4:] == ["--run-dir", str(tmp_path / "science"), "--workers", "6"]
    assert not any("seed" in token.lower() for token in command)
    with pytest.raises(ValueError, match="workers must be between"):
        scientific_command(workers=7, scientific_dir=tmp_path / "science")


def test_seed_manifest_describes_burned_producer_owned_block_without_materializing_it():
    payload = screen_seed_manifest()

    assert payload["seed_start"] == SCREEN_SEED_START == 7_300_001
    assert payload["seed_end"] == SCREEN_SEED_END == 7_300_024
    assert payload["count"] == 24
    assert payload["use"] == "BURNED_DEVELOPMENT_SCREEN_ONLY"
    assert payload["virgin"] is False
    assert payload["launcher_override_permitted"] is False
    assert "seeds" not in payload


def test_session_processes_includes_reparented_members_of_same_sid(monkeypatch):
    output = "\n".join(
        [
            "101 1 101 444 25.0 100 S python reparented-worker",
            "102 101 101 444 10.0 200 R python child-worker",
            "103 1 103 555 99.0 300 R unrelated",
        ]
    )
    monkeypatch.setattr(
        watcher.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout=output),
    )

    rows = watcher.session_processes(444)

    assert [row["pid"] for row in rows] == [101, 102]
    assert rows[0]["ppid"] == 1
    assert all(row["session_id"] == 444 for row in rows)


def test_watcher_prestart_and_live_stale_states(tmp_path, monkeypatch):
    monkeypatch.setattr(watcher, "memory_available_bytes", lambda: 1_000_000)
    monkeypatch.setattr(watcher, "session_processes", lambda _sid: [])
    prestart = watcher.snapshot(
        tmp_path,
        watcher_started_at_utc="2026-07-14T00:00:00+00:00",
        peak_rss_bytes=0,
        minimum_memory_available_bytes=None,
        progress_stale_seconds=30.0,
    )
    assert prestart["state"] == "watching_prestart"
    assert prestart["watcher_scope"] == (
        "ENTIRE_SCIENTIFIC_SESSION_INCLUDING_REPARENTED_WORKERS"
    )

    write_json(
        tmp_path / "pid.json",
        {
            "scientific_pid": 901,
            "scientific_process_group_id": 901,
            "scientific_session_id": 901,
            "result": str(tmp_path / "scientific_output/result.json"),
        },
    )
    progress = tmp_path / "scientific_output/progress.json"
    write_json(progress, {"complete": False, "completed_count": 1})
    os.utime(progress, (1, 1))
    monkeypatch.setattr(
        watcher,
        "session_processes",
        lambda _sid: [
            {
                "pid": 999,
                "ppid": 1,
                "process_group_id": 901,
                "session_id": 901,
                "cpu_percent": 50.0,
                "rss_bytes": 4096,
                "state": "R",
                "command": "synthetic worker",
            }
        ],
    )
    stale = watcher.snapshot(
        tmp_path,
        watcher_started_at_utc="2026-07-14T00:00:00+00:00",
        peak_rss_bytes=0,
        minimum_memory_available_bytes=None,
        progress_stale_seconds=30.0,
    )

    assert stale["state"] == "running_progress_stale"
    assert stale["scientific_session_process_count"] == 1
    assert stale["peak_scientific_session_rss_bytes"] == 4096


def test_terminal_requires_complete_true_and_exact_result_hash(tmp_path, monkeypatch):
    scientific = tmp_path / "scientific_output"
    result = scientific / "result.json"
    result.parent.mkdir(parents=True)
    result.write_text('{"status":"done"}\n', encoding="utf-8")
    write_json(
        tmp_path / "pid.json",
        {
            "scientific_pid": 901,
            "scientific_process_group_id": 901,
            "scientific_session_id": 901,
            "result": str(result),
        },
    )
    monkeypatch.setattr(watcher, "session_processes", lambda _sid: [])
    monkeypatch.setattr(watcher, "memory_available_bytes", lambda: None)

    write_json(
        scientific / "progress.json",
        {"complete": True, "result_sha256": file_sha256(result)},
    )
    verified = watcher.snapshot(
        tmp_path,
        watcher_started_at_utc="2026-07-14T00:00:00+00:00",
        peak_rss_bytes=0,
        minimum_memory_available_bytes=None,
    )
    assert verified["state"] == "completed_verified"
    assert verified["terminal_result_hash_matches"] is True

    write_json(
        scientific / "progress.json",
        {"complete": True, "result_sha256": "0" * 64},
    )
    mismatch = watcher.snapshot(
        tmp_path,
        watcher_started_at_utc="2026-07-14T00:00:00+00:00",
        peak_rss_bytes=0,
        minimum_memory_available_bytes=None,
    )
    assert mismatch["state"] == "failed_or_incomplete"
    assert mismatch["terminal_result_hash_matches"] is False

    write_json(
        scientific / "progress.json",
        {"complete": False, "result_sha256": file_sha256(result)},
    )
    incomplete = watcher.snapshot(
        tmp_path,
        watcher_started_at_utc="2026-07-14T00:00:00+00:00",
        peak_rss_bytes=0,
        minimum_memory_available_bytes=None,
    )
    assert incomplete["state"] == "failed_or_incomplete"


def test_prestart_manifests_are_separate_and_checksum_bound(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "scripts.launch_program_m_shared_lift_hpi.checked_output",
        lambda command: "package==1" if "freeze" in command else "",
    )
    command = scientific_command(workers=2, scientific_dir=tmp_path / "scientific_output")
    checksums = write_prestart_manifests(
        run_dir=tmp_path,
        head="a" * 40,
        status=[],
        command=command,
        workers=2,
        watch_interval_seconds=5.0,
        progress_stale_seconds=60.0,
    )

    expected = {
        "source_manifest.json",
        "contract_manifest.json",
        "environment_manifest.json",
        "seed_manifest.json",
        "command_manifest.json",
    }
    assert set(checksums["files"]) == expected
    for name in expected:
        assert checksums["files"][name] == file_sha256(tmp_path / name)
    assert json.loads((tmp_path / "contract_manifest.json").read_text())[
        "contract_id"
    ] == "program_m_shared_lift_reservation_v1"
    assert json.loads((tmp_path / "command_manifest.json").read_text())[
        "maximum_workers"
    ] == 6
