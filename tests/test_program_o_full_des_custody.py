import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from scripts.program_o_full_des_guard import create_seed_claim, verify_seed_claim
from scripts.watch_program_o_full_des_hpi import monitor, snapshot


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_watcher_waits_fail_closed_before_control(tmp_path: Path):
    state = snapshot(tmp_path, stale_seconds=60)
    assert state["status"] == "AWAITING_PRODUCER_CONTROL"
    assert state["group_alive"] is False


def test_watcher_tracks_the_whole_process_group_and_terminal_exit(tmp_path: Path):
    custody = tmp_path / "custody"
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(0.5)"],
        start_new_session=True,
    )
    control = {
        "stage": "development",
        "producer_pid": process.pid,
        "producer_pgid": os.getpgid(process.pid),
        "producer_sid": os.getsid(process.pid),
    }
    write_json(custody / "producer_control.json", control)
    running = snapshot(tmp_path, stale_seconds=60)
    assert running["status"] == "RUNNING"
    assert running["group_alive"] is True
    assert running["group_member_count"] >= 1

    process.wait(timeout=5)
    write_json(custody / "producer_exit.json", {"returncode": 0})
    terminal = snapshot(tmp_path, stale_seconds=60)
    assert terminal["status"] == "COMPLETE_PENDING_RETRIEVAL"
    assert terminal["group_alive"] is False


def test_once_monitor_writes_atomic_heartbeat_and_journal(tmp_path: Path):
    assert monitor(tmp_path, interval_seconds=0.01, stale_seconds=60, once=True) == 0
    state = json.loads((tmp_path / "custody/watcher_state.json").read_text())
    assert state["status"] == "AWAITING_PRODUCER_CONTROL"
    assert (tmp_path / "custody/watcher_state.jsonl").read_text().count("\n") == 1
    assert (
        json.loads((tmp_path / "custody/watcher_ready.json").read_text())["watcher_pid"]
        == os.getpid()
    )


def test_watcher_preserves_custody_after_parent_exits_with_live_child(
    tmp_path: Path,
):
    custody = tmp_path / "custody"
    parent = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import subprocess,sys; "
                "subprocess.Popen([sys.executable,'-c',"
                "'import time; time.sleep(1.5)'])"
            ),
        ],
        start_new_session=True,
    )
    pgid = os.getpgid(parent.pid)
    sid = os.getsid(parent.pid)
    write_json(
        custody / "producer_control.json",
        {
            "stage": "development",
            "producer_pid": parent.pid,
            "producer_pgid": pgid,
            "producer_sid": sid,
        },
    )
    parent.wait(timeout=5)
    deadline = time.time() + 1.0
    state = snapshot(tmp_path, stale_seconds=60)
    while not state["custody_scope_alive"] and time.time() < deadline:
        time.sleep(0.02)
        state = snapshot(tmp_path, stale_seconds=60)
    assert state["status"] == "RUNNING"
    assert state["custody_scope_alive"] is True
    assert any(row["pid"] != parent.pid for row in state["session_members"])


def test_seed_claim_is_exclusive_and_bound_to_authorization(tmp_path: Path):
    authorization = {
        "scientific_commit": "a" * 40,
        "freeze_sha256": "b" * 64,
        "freeze_path": "/repo/freeze.json",
        "run_id": "program-o-test",
        "run_dir": "/remote/program-o-test",
        "stage": "development",
        "seed_range": [7400049, 7400072],
    }
    claim = create_seed_claim(
        claim_root=tmp_path,
        authorization=authorization,
        contract_sha256="c" * 64,
    )

    verified = verify_seed_claim(
        claim_path=claim,
        authorization=authorization,
        contract_sha256="c" * 64,
    )
    assert verified["run_id"] == "program-o-test"
    with pytest.raises(FileExistsError):
        create_seed_claim(
            claim_root=tmp_path,
            authorization=authorization,
            contract_sha256="c" * 64,
        )


def test_seed_claim_tampering_fails_closed(tmp_path: Path):
    authorization = {
        "scientific_commit": "a" * 40,
        "freeze_sha256": "b" * 64,
        "freeze_path": "/repo/freeze.json",
        "run_id": "program-o-test",
        "run_dir": "/remote/program-o-test",
        "stage": "development",
        "seed_range": [7400049, 7400072],
    }
    claim = create_seed_claim(
        claim_root=tmp_path,
        authorization=authorization,
        contract_sha256="c" * 64,
    )
    payload = json.loads(claim.read_text())
    payload["run_id"] = "tampered"
    claim.write_text(json.dumps(payload))

    with pytest.raises(RuntimeError, match="run_id"):
        verify_seed_claim(
            claim_path=claim,
            authorization=authorization,
            contract_sha256="c" * 64,
        )
