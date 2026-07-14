import json
from hashlib import sha256
import os
import subprocess

import numpy as np
import pytest

from scripts.launch_paper2_switch4 import (
    CONTRACT,
    PREFLIGHT_DEPENDENCIES,
    RESULT_ROOT,
    ROOT,
    SEARCH,
    VERIFIER,
    WATCHER,
    file_sha256,
    json_sha256,
    validate_preflight_evidence,
)
from scripts.run_paper2_bottleneck_full_frontier import calendar_index
from scripts.search_paper2_bottleneck_switch4 import (
    CANDIDATE_COUNT,
    CONTRACT_PATH,
    EXPECTED_COUNTS,
    candidate_calendars,
    exact_argmax,
)
from scripts.search_paper2_bottleneck_switch_complexity import switch_count
from scripts.verify_paper2_bottleneck_switch4 import VERIFICATION_SCHEMA
from scripts.watch_paper2_switch4 import snapshot


def test_complete_at_most_four_switch_family_is_exact_and_feasible():
    rows = candidate_calendars()
    contract = json.loads(CONTRACT_PATH.read_text())
    counts = {
        switches: sum(switch_count(row) == switches for row in rows)
        for switches in EXPECTED_COUNTS
    }
    assert len(rows) == contract["candidate_family"]["candidate_count"]
    assert len(rows) == CANDIDATE_COUNT == 89_131
    assert counts == EXPECTED_COUNTS
    assert len(set(rows)) == CANDIDATE_COUNT
    assert rows[0] == (0,) * 24
    assert len({calendar_index(row) for row in rows}) == CANDIDATE_COUNT
    assert all(
        not (
            row[index] != row[index - 1]
            and row[index - 1] != row[index - 2]
        )
        for row in rows
        for index in range(2, 24)
    )


def test_exact_argmax_uses_binary64_values_and_minimum_index_for_ties():
    matrix = np.zeros((3, 12), dtype=float)
    matrix[:, 7] = 0.25
    matrix[:, 9] = 0.25
    sums, selected = exact_argmax(matrix)
    assert selected == 7
    assert sums[7] == sums[9]
    assert sums[7] > sums[0]


def _preflight_files(
    tmp_path,
    *,
    peak_rss=1_000_000_000,
    include_live=True,
    worker_count=6,
    dependency_subset=False,
):
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    host = "vps-f733423b"
    command = ["python", "search_paper2_bottleneck_switch4.py", "--preflight-only"]
    dependencies = {
        str(path.relative_to(ROOT)): file_sha256(path)
        for path in PREFLIGHT_DEPENDENCIES
    }
    if dependency_subset:
        dependencies = {
            "contracts/paper2_bottleneck_switch_complexity_screen_v2.json": file_sha256(
                CONTRACT
            )
        }
    result = {
        "schema_version": "paper2_bottleneck_switch4_preflight_v1",
        "contract_id": "paper2_bottleneck_switch_complexity_screen_v2",
        "contract_sha256": file_sha256(CONTRACT),
        "preflight_only": True,
        "git_head": head,
        "launch_git_status_porcelain": [],
        "environment": {"hostname": host},
        "dependency_sha256": dependencies,
        "command": command,
        "candidate_count": 89_131,
        "scores_evaluated": 534_786,
        "seed_start": 1_100_001,
        "seed_end": 1_100_006,
        "n_tapes": 6,
        "tapes": [
            {
                "seed": seed,
                "tape_sha256": f"{seed:064x}",
                "scores_float_hex_sha256": f"{seed + 10:064x}",
            }
            for seed in range(1_100_001, 1_100_007)
        ],
        "locked_tapes_accessed": False,
        "virgin_tapes_accessed": False,
        "h_pi_computed": False,
        "h_obs_computed": False,
        "w24_authorized": False,
        "learner_authorized": False,
        "paper2_authorized": False,
        "paper3_authorized": False,
    }
    result["content_sha256"] = json_sha256(result)
    result_path = tmp_path / "preflight.json"
    result_path.write_text(json.dumps(result))
    result_sha = sha256(result_path.read_bytes()).hexdigest()
    pid_record = {
        "launched_at_utc": "2026-07-13T00:00:02+00:00",
        "scientific_pid": 222,
        "watcher_pid": 111,
        "command": command,
        "output": str(result_path),
        "progress": str(tmp_path / "progress.json"),
        "git_head": head,
    }
    pid_path = tmp_path / "pid.json"
    pid_path.write_text(json.dumps(pid_record))
    receipt = {
        "schema_version": "paper2_switch4_detached_launch_v1",
        "operation": "vps_six_tape_memory_preflight",
        "hostname": host,
        "git_head": head,
        "contract_sha256": file_sha256(CONTRACT),
        "launcher_sha256": file_sha256(
            ROOT / "scripts" / "launch_paper2_switch4.py"
        ),
        "search_sha256": file_sha256(SEARCH),
        "verifier_sha256": file_sha256(VERIFIER),
        "watcher_sha256": file_sha256(WATCHER),
        "watcher_started_before_scientific_process": True,
        "scientific_pid": 222,
        "watcher_pid": 111,
        "pid_record_sha256": sha256(pid_path.read_bytes()).hexdigest(),
        "output": str(result_path),
        "progress": str(tmp_path / "progress.json"),
        "command": command,
    }
    receipt_path = tmp_path / "launch_receipt.json"
    receipt_path.write_text(json.dumps(receipt))
    prestart = {
        "hostname": host,
        "state": "watching_prestart",
        "observed_at_utc": "2026-07-13T00:00:01+00:00",
        "watcher_pid": 111,
        "scientific_pid": None,
        "scientific_pid_alive": False,
        "scientific_process_tree": [],
        "scientific_process_tree_rss_bytes": 0,
        "memory_available_bytes": 9_000_000_000,
    }
    live = {
        "hostname": host,
        "state": "running_alive_awaiting_first_progress",
        "observed_at_utc": "2026-07-13T00:00:03+00:00",
        "watcher_pid": 111,
        "scientific_pid": 222,
        "scientific_pid_alive": True,
        "scientific_process_tree": [
            {
                "pid": 222,
                "ppid": 1,
                "rss_bytes": 1,
                "command": "python search_paper2_bottleneck_switch4.py",
            },
            {
                "pid": 223,
                "ppid": 222,
                "rss_bytes": 1,
                "command": "python -c from multiprocessing.resource_tracker import main",
            },
            *[
                {
                    "pid": 300 + index,
                    "ppid": 222,
                    "rss_bytes": 1,
                    "command": (
                        "python -c from multiprocessing.spawn import spawn_main; "
                        "spawn_main() --multiprocessing-fork"
                    ),
                }
                for index in range(worker_count)
            ],
        ],
        "scientific_process_tree_rss_bytes": peak_rss,
        "memory_available_bytes": 8_000_000_000,
    }
    watcher = {
        "hostname": host,
        "state": "completed_unverified",
        "observed_at_utc": "2026-07-13T00:10:00+00:00",
        "watcher_pid": 111,
        "scientific_pid": 222,
        "result_exists": True,
        "scientific_pid_alive": False,
        "scientific_process_tree": [],
        "scientific_process_tree_rss_bytes": 0,
        "output": str(result_path),
        "result_sha256": result_sha,
        "progress": {
            "stage": "complete",
            "completed": 6,
            "total": 6,
            "output_sha256": result_sha,
        },
        "stderr_bytes": 0,
        "peak_scientific_process_tree_rss_bytes": peak_rss,
        "memory_available_bytes": 9_000_000_000,
        "minimum_memory_available_bytes": 8_000_000_000,
    }
    watcher_path = tmp_path / "watcher_latest.json"
    watcher_path.write_text(json.dumps(watcher))
    watcher_log_path = tmp_path / "watcher.jsonl"
    watcher_rows = [prestart]
    if include_live:
        watcher_rows.append(live)
    watcher_rows.append(watcher)
    watcher_log_path.write_text(
        "".join(json.dumps(row) + "\n" for row in watcher_rows)
    )
    return (
        head,
        result_path,
        watcher_path,
        receipt_path,
        pid_path,
        watcher_log_path,
    )


def _mock_vps_and_committed_dependencies(monkeypatch):
    monkeypatch.setattr(
        "scripts.launch_paper2_switch4.socket.gethostname",
        lambda: "vps-f733423b",
    )
    monkeypatch.setattr(
        "scripts.launch_paper2_switch4.git_blob_sha256",
        lambda commit, relative: file_sha256(ROOT / relative),
    )


def test_preflight_evidence_is_bound_to_live_six_worker_custody(
    tmp_path, monkeypatch
):
    _mock_vps_and_committed_dependencies(monkeypatch)
    paths = _preflight_files(tmp_path)
    evidence = validate_preflight_evidence(
        *paths[1:], expected_head=paths[0]
    )
    assert evidence["git_head"] == paths[0]
    assert evidence["checks"] and all(evidence["checks"].values())


def test_preflight_evidence_fails_closed_above_rss_gate(tmp_path, monkeypatch):
    _mock_vps_and_committed_dependencies(monkeypatch)
    paths = _preflight_files(
        tmp_path, peak_rss=7_000_000_000
    )
    with pytest.raises(ValueError, match="rss_below_gate"):
        validate_preflight_evidence(*paths[1:], expected_head=paths[0])


def test_preflight_evidence_rejects_posthoc_watcher_without_live_pool(
    tmp_path, monkeypatch
):
    _mock_vps_and_committed_dependencies(monkeypatch)
    paths = _preflight_files(tmp_path, include_live=False)
    with pytest.raises(ValueError, match="six_worker_live_sample"):
        validate_preflight_evidence(*paths[1:], expected_head=paths[0])


def test_preflight_evidence_rejects_five_workers_plus_tracker(
    tmp_path, monkeypatch
):
    _mock_vps_and_committed_dependencies(monkeypatch)
    paths = _preflight_files(tmp_path, worker_count=5)
    with pytest.raises(ValueError, match="six_worker_live_sample"):
        validate_preflight_evidence(*paths[1:], expected_head=paths[0])


def test_preflight_evidence_rejects_dependency_subset(tmp_path, monkeypatch):
    _mock_vps_and_committed_dependencies(monkeypatch)
    paths = _preflight_files(tmp_path, dependency_subset=True)
    with pytest.raises(ValueError, match="dependency_hashes_current"):
        validate_preflight_evidence(*paths[1:], expected_head=paths[0])


def test_preflight_evidence_rejects_non_designated_runtime_host(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "scripts.launch_paper2_switch4.socket.gethostname",
        lambda: "not-the-vps",
    )
    monkeypatch.setattr(
        "scripts.launch_paper2_switch4.git_blob_sha256",
        lambda commit, relative: file_sha256(ROOT / relative),
    )
    paths = _preflight_files(tmp_path)
    with pytest.raises(ValueError, match="designated_vps_runtime"):
        validate_preflight_evidence(*paths[1:], expected_head=paths[0])


def test_watcher_tracks_prestart_peak_and_terminal_hash(tmp_path):
    prestart = snapshot(
        tmp_path,
        watcher_started="2026-07-13T00:00:00+00:00",
        peak_rss_bytes=123,
        min_memory_available_bytes=456,
    )
    assert prestart["state"] == "watching_prestart"
    assert prestart["peak_scientific_process_tree_rss_bytes"] == 123
    (tmp_path / "pid.json").write_text(json.dumps({
        "scientific_pid": os.getpid(),
        "output": str(tmp_path / "result.json"),
    }))
    awaiting = snapshot(
        tmp_path, watcher_started="2026-07-13T00:00:00+00:00"
    )
    assert awaiting["state"] == "running_alive_awaiting_first_progress"
    assert any(row["pid"] == os.getpid() for row in awaiting["scientific_process_tree"])
    result = tmp_path / "result.json"
    result.write_text('{"ok":true}\n')
    result_sha = sha256(result.read_bytes()).hexdigest()
    (tmp_path / "progress.json").write_text(json.dumps({
        "stage": "complete",
        "output_sha256": result_sha,
    }))
    (tmp_path / "pid.json").write_text(json.dumps({
        "scientific_pid": 999_999_999,
        "output": str(result),
    }))
    terminal = snapshot(
        tmp_path, watcher_started="2026-07-13T00:00:00+00:00"
    )
    assert terminal["state"] == "completed_unverified"
    assert terminal["result_sha256"] == result_sha


def test_contract_and_claim_scope_are_fail_closed():
    contract = json.loads(CONTRACT_PATH.read_text())
    assert contract["candidate_family"]["calibration_rollouts"] == 5_347_860
    assert contract["calibration"]["locked_seed_access_forbidden"] is True
    assert contract["vps_preflight"]["required_before_producer"] is True
    assert contract["vps_preflight"]["workers"] == 6
    assert contract["execution_discipline"]["target_host_alias"] == "ovh-agent-lab"
    assert contract["execution_discipline"]["expected_hostname"] == "vps-f733423b"
    assert contract["vps_preflight"]["tapes"] == 6
    assert contract["vps_preflight"]["preflight_rollouts"] == 534_786
    assert all(
        contract["decision_rules"][field] is False
        for field in (
            "h_pi_computed",
            "h_obs_computed",
            "w24_authorized",
            "learner_authorized",
            "paper2_authorized",
            "paper3_authorized",
        )
    )
    assert RESULT_ROOT.name == "switch_complexity_screen_v2"
    assert VERIFICATION_SCHEMA == "paper2_bottleneck_switch4_verification_v1"
