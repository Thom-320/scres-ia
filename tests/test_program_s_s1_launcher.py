from __future__ import annotations

from pathlib import Path

from scripts.run_program_s_s1_native import EXPECTED_SHARDS, pending_tasks, shard_path, tasks


def test_frozen_s1_task_manifest_is_complete_and_unique() -> None:
    rows = tasks()
    assert len(rows) == EXPECTED_SHARDS == 5_760
    assert len(set(rows)) == EXPECTED_SHARDS
    assert {row[-1] for row in rows} == set(range(7_510_001, 7_510_013))
    assert {row[3] for row in rows} == {
        "rho75_share90", "rho90_share75", "rho90_share90"
    }


def test_resume_detects_only_missing_shards(tmp_path: Path) -> None:
    rows = tasks()[:2]
    first = shard_path(tmp_path, rows[0])
    first.parent.mkdir(parents=True)
    first.write_bytes(b"valid-nonempty-smoke")
    assert pending_tasks(tmp_path, rows) == [rows[1]]
