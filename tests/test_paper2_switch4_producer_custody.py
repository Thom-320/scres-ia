import json
from pathlib import Path
import subprocess

from scripts.launch_paper2_switch4 import file_sha256, json_sha256
from scripts.validate_paper2_switch4_producer_custody import (
    EXPECTED_PYTHON,
    PREFLIGHT_FILES,
    SCRIPT_BLOBS,
    _environment_is_frozen,
    validate_producer_custody,
)


ROOT = Path(__file__).resolve().parent.parent


def _head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _custody_fixture(tmp_path: Path, *, workers: int = 6):
    run_id = "mtr-switch4-producer-test"
    run = tmp_path / run_id
    preflight = tmp_path / "preflight"
    run.mkdir()
    preflight.mkdir()
    head = _head()
    remote = (
        Path("/source")
        / "results"
        / "paper2_bound_harness"
        / "switch_complexity_screen_v2"
        / run_id
    )
    contract_sha = file_sha256(
        ROOT / "contracts" / "paper2_bottleneck_switch_complexity_screen_v2.json"
    )
    command = [
        EXPECTED_PYTHON,
        "/source/scripts/search_paper2_bottleneck_switch4.py",
        "--output",
        str(remote / "result.json"),
        "--progress",
        str(remote / "progress.json"),
        "--workers",
        "6",
    ]
    result = {
        "schema_version": "paper2_bottleneck_switch4_screen_v1",
        "contract_sha256": contract_sha,
        "git_head": head,
        "launch_git_status_porcelain": [],
        "environment": {
            "hostname": "vps-f733423b",
            "python_executable": EXPECTED_PYTHON,
        },
        "command": command,
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
    _write(run / "result.json", result)
    result_sha = file_sha256(run / "result.json")
    pid = {
        "launched_at_utc": "2026-07-14T00:00:02+00:00",
        "scientific_pid": 222,
        "watcher_pid": 111,
        "command": command,
        "output": str(remote / "result.json"),
        "progress": str(remote / "progress.json"),
        "git_head": head,
    }
    _write(run / "pid.json", pid)
    for filename in (
        "preflight.json",
        "progress.json",
        "watcher_latest.json",
        "launch_manifest.json",
        "launch_receipt.json",
        "pid.json",
        "watcher.jsonl",
        "stdout.log",
        "stderr.log",
    ):
        (preflight / filename).write_text(filename + "\n")
    trusted = {
        "git_head": head,
        "hostname": "vps-f733423b",
        "checks": {"all": True},
        "result_sha256": file_sha256(preflight / "preflight.json"),
        "watcher_sha256": file_sha256(preflight / "watcher_latest.json"),
        "launch_receipt_sha256": file_sha256(preflight / "launch_receipt.json"),
        "pid_sha256": file_sha256(preflight / "pid.json"),
        "watcher_log_sha256": file_sha256(preflight / "watcher.jsonl"),
    }
    receipt = {
        "schema_version": "paper2_switch4_detached_launch_v1",
        "created_at_utc": "2026-07-14T00:00:00+00:00",
        "operation": "calibration_switch4_screen",
        "git_head": head,
        "launch_git_status_porcelain": [],
        "hostname": "vps-f733423b",
        "cwd": "/source",
        "contract_sha256": contract_sha,
        "command": command,
        "output": str(remote / "result.json"),
        "progress": str(remote / "progress.json"),
        "scientific_pid": 222,
        "watcher_pid": 111,
        "watcher_started_before_scientific_process": True,
        "workers": 6,
        "verified_result": None,
        "verified_result_sha256": None,
        "pid_record_sha256": file_sha256(run / "pid.json"),
        "trusted_preflight_evidence": trusted,
        **{
            key: file_sha256(ROOT / relative)
            for key, relative in SCRIPT_BLOBS.items()
        },
    }
    _write(run / "launch_receipt.json", receipt)
    manifest = {
        key: value
        for key, value in receipt.items()
        if key
        not in {
            "pid_record_sha256",
            "scientific_pid",
            "watcher_pid",
            "watcher_started_before_scientific_process",
        }
    }
    _write(run / "launch_manifest.json", manifest)
    watcher_started = "2026-07-14T00:00:00.500000+00:00"
    prestart = {
        "schema_version": "paper2_switch4_watcher_v1",
        "state": "watching_prestart",
        "observed_at_utc": "2026-07-14T00:00:01+00:00",
        "hostname": "vps-f733423b",
        "watcher_started_at_utc": watcher_started,
        "watcher_pid": 111,
        "scientific_pid": None,
        "scientific_pid_alive": False,
        "scientific_command": None,
        "output": str(remote / "result.json"),
        "scientific_process_tree_rss_bytes": 0,
        "memory_available_bytes": 9_000_000_000,
    }
    live = {
        "schema_version": "paper2_switch4_watcher_v1",
        "state": "running_alive_awaiting_first_progress",
        "observed_at_utc": "2026-07-14T00:00:03+00:00",
        "hostname": "vps-f733423b",
        "watcher_started_at_utc": watcher_started,
        "watcher_pid": 111,
        "scientific_pid": 222,
        "scientific_pid_alive": True,
        "scientific_command": " ".join(command),
        "output": str(remote / "result.json"),
        "scientific_process_tree_rss_bytes": 1_000_000_000,
        "memory_available_bytes": 8_000_000_000,
        "scientific_process_tree": [
            {
                "pid": 222,
                "ppid": 1,
                "command": " ".join(command),
                "rss_bytes": 100_000_000,
            },
            *[
                {
                    "pid": 300 + index,
                    "ppid": 222,
                    "command": (
                        f"{EXPECTED_PYTHON} -c from multiprocessing.spawn import "
                        "spawn_main; "
                        "spawn_main() --multiprocessing-fork"
                    ),
                    "rss_bytes": 150_000_000,
                }
                for index in range(workers)
            ],
        ],
    }
    final_progress = {
        "schema_version": "paper2_bottleneck_switch4_progress_v1",
        "stage": "complete",
        "completed": 120,
        "total": 120,
        "output": str(remote / "result.json"),
        "output_sha256": result_sha,
        "elapsed_seconds": 10.0,
        "updated_at_utc": "2026-07-14T00:00:03.900000+00:00",
    }
    _write(run / "progress.json", final_progress)
    terminal = {
        "schema_version": "paper2_switch4_watcher_v1",
        "state": "completed_unverified",
        "observed_at_utc": "2026-07-14T00:00:04+00:00",
        "hostname": "vps-f733423b",
        "watcher_started_at_utc": watcher_started,
        "watcher_pid": 111,
        "scientific_pid": 222,
        "scientific_pid_alive": False,
        "scientific_command": None,
        "scientific_process_tree_rss_bytes": 0,
        "peak_scientific_process_tree_rss_bytes": 1_000_000_000,
        "memory_available_bytes": 9_000_000_000,
        "minimum_memory_available_bytes": 8_000_000_000,
        "result_exists": True,
        "result_sha256": result_sha,
        "output": str(remote / "result.json"),
        "progress": final_progress,
        "progress_sha256": file_sha256(run / "progress.json"),
        "stderr_bytes": 0,
        "stdout_bytes": 3,
    }
    rows = [prestart, live, terminal]
    (run / "watcher.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    _write(run / "watcher_latest.json", terminal)
    (run / "stderr.log").write_bytes(b"")
    (run / "stdout.log").write_bytes(b"ok\n")
    watcher_lines = [json.dumps(row, sort_keys=True) + "\n" for row in rows]
    prefix = "".join(watcher_lines[:2]).encode()
    import hashlib

    anchor = {
        "schema_version": "paper2_switch4_precompletion_anchor_v1",
        "observed_at_utc": "2026-07-14T00:00:03.500000+00:00",
        "hostname": "vps-f733423b",
        "boot_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "source_git_head": head,
        "source_git_status_porcelain": [],
        "run_id": run_id,
        "result_bytes": 0,
        "progress_bytes": 0,
        "claim_limit": (
            "Pre-completion custody anchor only; no scientific result or policy "
            "ranking exists at this observation."
        ),
        "processes": {
            "scientific": {
                "pid": 222,
                "ppid": 1,
                "start_ticks": 12345,
                "cmdline": " ".join(command),
            },
            "watcher": {
                "pid": 111,
                "ppid": 1,
                "start_ticks": 12340,
                "cmdline": (
                    f"{EXPECTED_PYTHON} /source/scripts/watch_paper2_switch4.py "
                    f"--run-dir {remote} --interval-seconds 5.0"
                ),
            },
        },
        "immutable_run_files": {
            filename: {
                "bytes": (run / filename).stat().st_size,
                "sha256": file_sha256(run / filename),
            }
            for filename in (
                "launch_manifest.json",
                "launch_receipt.json",
                "pid.json",
            )
        },
        "preflight_files": {
            filename: file_sha256(preflight / filename)
            for filename in (
                "preflight.json",
                "progress.json",
                "watcher_latest.json",
                "launch_manifest.json",
                "launch_receipt.json",
                "pid.json",
                "watcher.jsonl",
                "stdout.log",
                "stderr.log",
            )
        },
        "watcher_log_prefix": {
            "bytes": len(prefix),
            "sha256": hashlib.sha256(prefix).hexdigest(),
        },
    }
    anchor_path = tmp_path / "precompletion_anchor.json"
    _write(anchor_path, anchor)
    return run, preflight, anchor_path, head


def _patch_science_and_blobs(monkeypatch, *, valid_preflight: bool = True):
    monkeypatch.setattr(
        "scripts.validate_paper2_switch4_producer_custody.validate_payload",
        lambda payload: [],
    )
    monkeypatch.setattr(
        "scripts.validate_paper2_switch4_producer_custody.git_blob_sha256",
        lambda head, relative: file_sha256(ROOT / relative),
    )
    monkeypatch.setattr(
        "scripts.validate_paper2_switch4_producer_custody._validator_provenance",
        lambda: {"passed": True, "git_head": _head()},
    )
    monkeypatch.setattr(
        "scripts.validate_paper2_switch4_producer_custody._validation_runtime_provenance",
        lambda expected_head: {"passed": True, "expected_head": expected_head},
    )
    monkeypatch.setattr(
        "scripts.validate_paper2_switch4_producer_custody._environment_is_frozen",
        lambda environment: True,
    )
    monkeypatch.setattr(
        "scripts.validate_paper2_switch4_producer_custody._tracked_file_provenance",
        lambda path: {"passed": True, "path": str(path)},
    )
    if valid_preflight:
        monkeypatch.setattr(
            "scripts.validate_paper2_switch4_producer_custody._validate_relocated_preflight",
            lambda preflight_dir, expected_head: {
                "passed": True,
                "checks": {"synthetic": True},
                "environment": json.loads(
                    (preflight_dir.parent / "mtr-switch4-producer-test" / "result.json").read_text()
                )["environment"],
                **{
                    key: file_sha256(preflight_dir / filename)
                    for key, filename in PREFLIGHT_FILES.items()
                },
            },
        )


def test_environment_validator_rejects_minimal_vps_identity():
    assert (
        _environment_is_frozen(
            {
                "hostname": "vps-f733423b",
                "python_executable": EXPECTED_PYTHON,
            }
        )
        is False
    )


def test_completed_producer_custody_binds_science_watcher_and_preflight(
    tmp_path, monkeypatch
):
    _patch_science_and_blobs(monkeypatch)
    run, preflight, anchor, head = _custody_fixture(tmp_path)
    audit = validate_producer_custody(run, preflight, anchor, expected_head=head)
    assert audit["passed"] is True
    assert audit["failed_checks"] == []
    assert all(audit["checks"].values())


def test_completed_producer_custody_rejects_five_worker_pool(
    tmp_path, monkeypatch
):
    _patch_science_and_blobs(monkeypatch)
    run, preflight, anchor, head = _custody_fixture(tmp_path, workers=5)
    audit = validate_producer_custody(run, preflight, anchor, expected_head=head)
    assert audit["passed"] is False
    assert audit["checks"]["six_worker_live_sample"] is False


def test_completed_producer_custody_rejects_tampered_preflight(
    tmp_path, monkeypatch
):
    _patch_science_and_blobs(monkeypatch, valid_preflight=False)
    run, preflight, anchor, head = _custody_fixture(tmp_path)
    (preflight / "preflight.json").write_text("tampered\n")
    audit = validate_producer_custody(run, preflight, anchor, expected_head=head)
    assert audit["passed"] is False
    assert audit["checks"]["trusted_preflight_hashes"] is False
    assert audit["checks"]["preflight_custody_valid"] is False


def test_completed_producer_custody_rejects_forged_preflight_all_true(
    tmp_path, monkeypatch
):
    _patch_science_and_blobs(monkeypatch, valid_preflight=False)
    run, preflight, anchor, head = _custody_fixture(tmp_path)
    audit = validate_producer_custody(run, preflight, anchor, expected_head=head)
    assert audit["passed"] is False
    assert audit["checks"]["trusted_preflight_checks"] is True
    assert audit["checks"]["trusted_preflight_hashes"] is True
    assert audit["checks"]["preflight_custody_valid"] is False


def test_completed_producer_custody_rejects_wrong_terminal_progress(
    tmp_path, monkeypatch
):
    _patch_science_and_blobs(monkeypatch)
    run, preflight, anchor, head = _custody_fixture(tmp_path)
    terminal = json.loads((run / "watcher_latest.json").read_text())
    terminal["progress"]["completed"] = 60
    terminal["progress"]["total"] = 60
    _write(run / "watcher_latest.json", terminal)
    rows = [json.loads(line) for line in (run / "watcher.jsonl").read_text().splitlines()]
    rows[-1] = terminal
    (run / "watcher.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    audit = validate_producer_custody(run, preflight, anchor, expected_head=head)
    assert audit["passed"] is False
    assert audit["checks"]["progress_complete"] is False


def test_completed_producer_custody_rejects_progress_artifact_tamper(
    tmp_path, monkeypatch
):
    _patch_science_and_blobs(monkeypatch)
    run, preflight, anchor, head = _custody_fixture(tmp_path)
    progress = json.loads((run / "progress.json").read_text())
    progress["completed"] = 119
    _write(run / "progress.json", progress)
    audit = validate_producer_custody(run, preflight, anchor, expected_head=head)
    assert audit["passed"] is False
    assert audit["checks"]["progress_complete"] is False


def test_completed_producer_custody_rejects_anchor_prefix_tamper(
    tmp_path, monkeypatch
):
    _patch_science_and_blobs(monkeypatch)
    run, preflight, anchor, head = _custody_fixture(tmp_path)
    payload = json.loads(anchor.read_text())
    payload["watcher_log_prefix"]["sha256"] = "0" * 64
    _write(anchor, payload)
    audit = validate_producer_custody(run, preflight, anchor, expected_head=head)
    assert audit["passed"] is False
    assert audit["checks"]["precompletion_anchor"] is False
    assert (
        audit["precompletion_anchor_validation"]["checks"]["watcher_prefix"]
        is False
    )


def test_completed_producer_custody_rejects_result_content_tamper(
    tmp_path, monkeypatch
):
    _patch_science_and_blobs(monkeypatch)
    run, preflight, anchor, head = _custody_fixture(tmp_path)
    payload = json.loads((run / "result.json").read_text())
    payload["paper2_authorized"] = True
    _write(run / "result.json", payload)
    audit = validate_producer_custody(run, preflight, anchor, expected_head=head)
    assert audit["passed"] is False
    assert audit["checks"]["result_content_hash"] is False
    assert audit["checks"]["claim_flags_closed"] is False
