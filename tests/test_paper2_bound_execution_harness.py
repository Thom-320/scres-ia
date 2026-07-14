import json
import copy
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

import scripts.paper2_bound_execution_harness as harness

from scripts.paper2_bound_execution_harness import (
    DEFAULT_CONTRACT,
    DEFAULT_RUNNER,
    DEFAULT_SMOKE_RUNNER,
    HarnessError,
    _authorization_template,
    _frozen_evidence_command_row,
    _frozen_evidence_profile,
    _frontier_command_row,
    _frozen_phase_seed_specs,
    _validate_scientific_result,
    _validate_frozen_evidence_result,
    _materialize_argv,
    _seed_rows,
    capture_runtime_attestation,
    execute_run,
    execute_reduced_signed_session,
    launch_vps,
    prepare_run,
    seal_run,
    scientific_child_environment,
    validate_scientific_environment_payload,
    validate_runtime_attestation_payload,
    verify_artifacts,
)
from scripts.run_paper2_bottleneck_exact_transducer import (
    build_parser as exact_transducer_parser,
)


ROOT = Path(__file__).resolve().parent.parent


def _fake_runtime_attestation() -> dict:
    distribution_manifests = {}
    portable_distribution_files = {}
    host_native_distribution_files = {}
    for package in harness.RUNTIME_PACKAGES:
        source_row = {
            "declared_path": f"{package}/__init__.py",
            "relative_to_site_packages": f"{package}/__init__.py",
            "bytes": 1,
            "sha256": "6" * 64,
            "classification": "python_source",
        }
        manifest_body = {
            "schema_version": harness.RUNTIME_DISTRIBUTION_MANIFEST_SCHEMA,
            "package": package,
            "distribution_name": package,
            "version": "1.0",
            "exclusion_schema": harness.RUNTIME_MANIFEST_EXCLUSIONS,
            "files": [source_row],
            "excluded": [],
        }
        distribution_manifests[package] = {
            **manifest_body,
            "manifest_sha256": harness.canonical_json_sha256(manifest_body),
        }
        portable_distribution_files[package] = {
            "file_count": 1,
            "files_sha256": harness.canonical_json_sha256([source_row]),
        }
        host_native_distribution_files[package] = {
            "file_count": 0,
            "files_sha256": harness.canonical_json_sha256([]),
        }
    portable = {
        "runner_sha256": "4" * 64,
        "bootstrap_sha256": "7" * 64,
        "distribution_manifest_schema": (
            harness.RUNTIME_DISTRIBUTION_MANIFEST_SCHEMA
        ),
        "distribution_manifest_exclusions": harness.RUNTIME_MANIFEST_EXCLUSIONS,
        "portable_distribution_files": portable_distribution_files,
    }
    checks = {
        "isolated": True,
        "no_site": True,
        "no_user_site": True,
        "safe_path": True,
        "dont_write_bytecode": True,
        "site_module_not_loaded": True,
        "customizers_absent_and_not_loaded": True,
        "pth_files_not_processed": True,
        "python_environment_absent": True,
    }
    host = {
        "flags": {
            "isolated": 1,
            "no_site": 1,
            "no_user_site": 1,
            "safe_path": True,
            "dont_write_bytecode": True,
        },
        "forbidden_python_environment": [],
        "pth_files": [],
        "customizers": [],
        "distribution_installed_files": distribution_manifests,
        "host_native_distribution_files": host_native_distribution_files,
    }
    body = {
        "schema_version": harness.RUNTIME_ATTESTATION_SCHEMA,
        "portable": portable,
        "portable_sha256": harness.canonical_json_sha256(portable),
        "host": host,
        "isolation_checks": checks,
        "isolation_checks_passed": True,
    }
    return {**body, "runtime_sha256": harness.canonical_json_sha256(body)}


def _write_fake_runtime(run_dir: Path) -> dict:
    payload = _fake_runtime_attestation()
    harness.atomic_write_json(run_dir / "environment/scientific_runtime.json", payload)
    return payload


def test_direct_cli_prepare_resolves_repository_namespace(tmp_path):
    run_dir = tmp_path / "direct-cli"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "paper2_bound_execution_harness.py"),
            "prepare",
            "--mode",
            "smoke",
            "--run-id",
            "pytest-direct-cli",
            "--run-dir",
            str(run_dir),
            "--runner",
            str(DEFAULT_SMOKE_RUNNER),
            "--seed",
            "1110001:equipment_pressure",
            "--split",
            "harness_pytest_burned",
            "--weeks",
            "4",
            "--runner-workers",
            "1",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["mode"] == "smoke"
    assert manifest["inputs"]["runner_relative"].endswith(
        "run_paper2_bottleneck_exact_transducer.py"
    )


def _prepare(tmp_path: Path, *, mode: str = "smoke") -> Path:
    run_dir = tmp_path / f"run-{mode}"
    prepare_run(
        run_dir=run_dir,
        run_id=f"pytest-{mode}",
        mode=mode,
        contract_path=DEFAULT_CONTRACT,
        runner_path=DEFAULT_SMOKE_RUNNER,
        seeds=[(1_110_001, "equipment_pressure")],
        split="harness_pytest_burned",
        weeks=4,
        runner_workers=1,
        heartbeat_interval=0.05,
    )
    return run_dir


def test_prepare_records_hashes_manifests_and_not_evidence(tmp_path):
    run_dir = _prepare(tmp_path)
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    seeds = json.loads((run_dir / "seed_manifest.json").read_text())
    commands = json.loads((run_dir / "command_manifest.json").read_text())
    status = json.loads((run_dir / "status" / "run_status.json").read_text())

    assert manifest["git"]["commit"]
    assert manifest["contract_id"] == "paper2_bottleneck_primary_bound_v2"
    assert DEFAULT_RUNNER.name == "run_paper2_bottleneck_full_frontier.py"
    assert manifest["inputs"]["runner_relative"].endswith(
        "run_paper2_bottleneck_exact_transducer.py"
    )
    assert len(manifest["inputs"]["contract_sha256"]) == 64
    assert len(manifest["inputs"]["runner_sha256"]) == 64
    assert len(manifest["inputs"]["harness_sha256"]) == 64
    assert len(manifest["inputs"]["isolated_bootstrap_sha256"]) == 64
    assert len(manifest["inputs"]["host_runtime_sha256"]) == 64
    assert len(manifest["inputs"]["portable_runtime_sha256"]) == 64
    assert len(manifest["inputs"]["execution_nonce"]) == 64
    assert seeds["seed_count"] == 1
    assert commands["shell_execution"] is False
    assert commands["commands"][0]["seed"] == 1_110_001
    assert "--non-scientific-smoke" in commands["commands"][0]["argv_template"]
    assert "--max-calendars" in commands["commands"][0]["argv_template"]
    assert commands["commands"][0]["argv_template"][1:4] == ["-I", "-B", "-S"]
    assert "paper2_isolated_bootstrap.py" in commands["commands"][0][
        "argv_template"
    ][4]
    assert status["state"] == "prepared"
    assert manifest["evidence"] is False
    assert "NOT_EVIDENCE" in manifest["evidence_status"]
    assert (run_dir / "environment" / "pip_freeze.txt").is_file()
    assert (run_dir / "environment" / "machine.json").is_file()
    runtime = validate_runtime_attestation_payload(
        json.loads((run_dir / "environment/scientific_runtime.json").read_text())
    )
    assert runtime["isolation_checks_passed"] is True
    assert runtime["host"]["native_extensions"]
    assert all(row["processed"] is False for row in runtime["host"]["pth_files"])
    assert runtime["portable"]["bootstrap_sha256"] == manifest["inputs"][
        "isolated_bootstrap_sha256"
    ]


def test_scientific_child_environment_is_minimal_and_python_free():
    environment = scientific_child_environment()
    assert not any(key.startswith("PYTHON") for key in environment)
    assert environment["LC_ALL"] == "C"
    assert environment["TZ"] == "UTC"
    assert all(
        environment[key] == "1"
        for key in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
    )


def test_runtime_attestation_fails_closed_on_flag_or_digest_tamper(tmp_path):
    payload = capture_runtime_attestation(
        python=sys.executable,
        repo_root=ROOT,
        runner_path=DEFAULT_SMOKE_RUNNER,
    )
    tampered = copy.deepcopy(payload)
    tampered["host"]["flags"]["no_site"] = 0
    body = dict(tampered)
    body.pop("runtime_sha256")
    tampered["runtime_sha256"] = harness.canonical_json_sha256(body)
    with pytest.raises(HarnessError, match="isolation checks|flags"):
        validate_runtime_attestation_payload(tampered)

    malformed = copy.deepcopy(payload)
    malformed["runtime_sha256"] = "0" * 64
    with pytest.raises(HarnessError, match="digest mismatch"):
        validate_runtime_attestation_payload(malformed)

    source_tamper = copy.deepcopy(payload)
    source_tamper["host"]["distribution_installed_files"]["simpy"]["files"][0][
        "sha256"
    ] = "0" * 64
    body = dict(source_tamper)
    body.pop("runtime_sha256")
    source_tamper["runtime_sha256"] = harness.canonical_json_sha256(body)
    with pytest.raises(HarnessError, match="manifest digest mismatch"):
        validate_runtime_attestation_payload(source_tamper)


def test_runtime_attestation_path_is_exclusive_and_never_replaced(tmp_path):
    output = tmp_path / "runtime.json"
    output.write_text("caller-retained sentinel")
    with pytest.raises(HarnessError, match="command failed"):
        capture_runtime_attestation(
            python=sys.executable,
            repo_root=ROOT,
            runner_path=DEFAULT_SMOKE_RUNNER,
            output_path=output,
        )
    assert output.read_text() == "caller-retained sentinel"


def test_signed_reduced_session_never_launches_without_exact_external_ack(
    tmp_path, monkeypatch
):
    import scripts.run_paper2_bottleneck_exact_transducer as exact

    runtime = capture_runtime_attestation(
        python=sys.executable,
        repo_root=ROOT,
        runner_path=DEFAULT_SMOKE_RUNNER,
    )
    launched = False

    def forbidden_launch(*args, **kwargs):
        nonlocal launched
        launched = True
        raise AssertionError("child launch occurred before exact ACK")

    monkeypatch.setattr(exact, "launch_reduced_execution_fresh_process", forbidden_launch)
    with pytest.raises(HarnessError, match="out-of-band prelaunch acknowledgement"):
        execute_reduced_signed_session(
            custody_root=tmp_path,
            role="pytest_w2_signed_ack_gate",
            execution_role="producer",
            replay_pair_id="a" * 64,
            weeks=2,
            seeds=((1_110_001, harness.CONTEXTS[0]),),
            split="signed_ack_gate_pytest_burned",
            workers=1,
            output_path=tmp_path / "result.json",
            authorization_path=tmp_path / "authorization.json",
            exact_receipt_path=tmp_path / "exact_receipt.json",
            runtime_attestation_path=tmp_path / "runtime.json",
            harness_receipt_path=tmp_path / "harness_receipt.json",
            host_runtime_sha256=runtime["runtime_sha256"],
            portable_runtime_sha256=runtime["portable_sha256"],
            harness_execution_nonce="b" * 64,
            acknowledgement_callback=lambda _record: {"ack": "synthetic"},
            non_scientific_smoke=True,
            max_calendars=exact.feasible_calendar_count(2),
        )
    assert launched is False
    assert (tmp_path / "authorization.json").is_file()
    assert (tmp_path / "result.json").is_file()
    assert (tmp_path / "result.json").stat().st_size == 0
    assert not (tmp_path / "exact_receipt.json").exists()
    assert not (tmp_path / "harness_receipt.json").exists()


def test_signed_reduced_w2_session_launches_only_after_retained_ack(tmp_path):
    import scripts.run_paper2_bottleneck_exact_transducer as exact

    runtime = capture_runtime_attestation(
        python=sys.executable,
        repo_root=ROOT,
        runner_path=DEFAULT_SMOKE_RUNNER,
    )
    retained = []

    def acknowledge(prelaunch):
        assert not (tmp_path / "runtime.json").exists()
        assert not (tmp_path / "exact_receipt.json").exists()
        retained.append(copy.deepcopy(prelaunch))
        return {
            "schema_version": harness.SIGNED_PRELAUNCH_ACK_SCHEMA,
            "prelaunch_record_sha256": prelaunch["prelaunch_record_sha256"],
            "public_key_fingerprint": prelaunch["public_key_fingerprint"],
            "authorization_sha256": prelaunch["authorization_sha256"],
            "host_runtime_sha256": prelaunch["host_runtime_sha256"],
            "acknowledged_before_child_launch": True,
        }

    result = execute_reduced_signed_session(
        custody_root=tmp_path,
        role="pytest_w2_signed_launch_not_evidence",
        execution_role="producer",
        replay_pair_id="c" * 64,
        weeks=2,
        seeds=((1_110_001, harness.CONTEXTS[0]),),
        split="signed_w2_pytest_burned",
        workers=1,
        output_path=tmp_path / "payload" / "result.json",
        authorization_path=tmp_path / "custody" / "authorization.json",
        exact_receipt_path=tmp_path / "custody" / "exact_receipt.json",
        runtime_attestation_path=tmp_path / "runtime" / "runtime.json",
        harness_receipt_path=tmp_path / "custody" / "harness_receipt.json",
        host_runtime_sha256=runtime["runtime_sha256"],
        portable_runtime_sha256=runtime["portable_sha256"],
        harness_execution_nonce="d" * 64,
        acknowledgement_callback=acknowledge,
        non_scientific_smoke=True,
        max_calendars=exact.feasible_calendar_count(2),
        timeout_seconds=60,
    )
    assert result["passed"] is True
    assert len(retained) == 1
    exact_receipt = json.loads(
        (tmp_path / "custody" / "exact_receipt.json").read_text()
    )
    harness_receipt = json.loads(
        (tmp_path / "custody" / "harness_receipt.json").read_text()
    )
    assert exact_receipt["fresh_child_process"] is True
    assert exact_receipt["receipt_signing_public_key_fingerprint"] == retained[0][
        "public_key_fingerprint"
    ]
    assert harness_receipt["schema_version"] == harness.SIGNED_HARNESS_RECEIPT_SCHEMA
    assert harness_receipt["external_ack_received_before_child_launch"] is True
    assert harness_receipt["exact_signed_receipt_verified_before_harness_receipt"] is True
    assert harness_receipt["evidence"] is False
    transfer = harness.create_reduced_signed_transfer_manifest(
        launch_root=tmp_path,
        harness_receipt_path=tmp_path / "custody" / "harness_receipt.json",
        expected_harness_receipt_sha256=result["harness_receipt_sha256"],
        role="pytest_w2_signed_launch_not_evidence",
        expected_public_key_fingerprint=retained[0]["public_key_fingerprint"],
        manifest_path=tmp_path / "transfer_manifest.json",
    )
    relocated = tmp_path.parent / "relocated-signed-w2"
    shutil.copytree(tmp_path, relocated)
    retrieval = harness.verify_retrieved_reduced_transfer(
        retrieved_root=relocated,
        manifest_relative="transfer_manifest.json",
        expected_manifest_sha256=transfer["manifest_sha256"],
        expected_public_key_fingerprint=retained[0]["public_key_fingerprint"],
    )
    assert retrieval["passed"] is True, retrieval
    assert retrieval["retrieved_inode_equality_not_required"] is True
    assert retrieval["lexical_no_symlink_archive_paths_verified"] is True

    transfer_payload = json.loads(
        (relocated / "transfer_manifest.json").read_text()
    )
    output_relative = transfer_payload["artifacts"]["output"]["relative_path"]
    output_path = relocated / output_relative
    output_copy = output_path.with_name("result-identical-copy.json")
    shutil.copy2(output_path, output_copy)
    output_path.unlink()
    output_path.symlink_to(output_copy.name)
    result_leaf_symlink = harness.verify_retrieved_reduced_transfer(
        retrieved_root=relocated,
        manifest_relative="transfer_manifest.json",
        expected_manifest_sha256=transfer["manifest_sha256"],
        expected_public_key_fingerprint=retained[0]["public_key_fingerprint"],
    )
    assert result_leaf_symlink["passed"] is False
    assert any(
        "contains a symlink" in failure
        for failure in result_leaf_symlink["failures"]
    )
    output_path.unlink()
    output_copy.replace(output_path)

    manifest_path = relocated / "transfer_manifest.json"
    manifest_copy = relocated / "transfer-manifest-identical-copy.json"
    shutil.copy2(manifest_path, manifest_copy)
    manifest_path.unlink()
    manifest_path.symlink_to(manifest_copy.name)
    with pytest.raises(HarnessError, match="contains a symlink"):
        harness.verify_retrieved_reduced_transfer(
            retrieved_root=relocated,
            manifest_relative="transfer_manifest.json",
            expected_manifest_sha256=transfer["manifest_sha256"],
            expected_public_key_fingerprint=retained[0]["public_key_fingerprint"],
        )
    manifest_path.unlink()
    manifest_copy.replace(manifest_path)

    payload_dir = relocated / "payload"
    payload_copy = relocated / "payload-identical-copy"
    payload_dir.rename(payload_copy)
    payload_dir.symlink_to(payload_copy.name, target_is_directory=True)
    intermediate_symlink = harness.verify_retrieved_reduced_transfer(
        retrieved_root=relocated,
        manifest_relative="transfer_manifest.json",
        expected_manifest_sha256=transfer["manifest_sha256"],
        expected_public_key_fingerprint=retained[0]["public_key_fingerprint"],
    )
    assert intermediate_symlink["passed"] is False
    assert any(
        "contains a symlink" in failure
        for failure in intermediate_symlink["failures"]
    )


def test_retrieved_signed_transfer_rejects_content_tamper(tmp_path):
    manifest_body = {
        "schema_version": harness.SIGNED_TRANSFER_MANIFEST_SCHEMA,
        "public_key_fingerprint": "a" * 64,
        "artifacts": {},
    }
    manifest = {
        **manifest_body,
        "manifest_body_sha256": harness.canonical_json_sha256(manifest_body),
    }
    harness.exclusive_write_json(tmp_path / "transfer.json", manifest)
    result = harness.verify_retrieved_reduced_transfer(
        retrieved_root=tmp_path,
        manifest_relative="transfer.json",
        expected_manifest_sha256=harness.sha256_file(tmp_path / "transfer.json"),
        expected_public_key_fingerprint="a" * 64,
    )
    assert result["passed"] is False
    assert any("artifact index" in failure for failure in result["failures"])


def test_relocated_pair_uses_verified_transfer_chains_not_remote_inodes(
    tmp_path, monkeypatch
):
    (tmp_path / "producer").mkdir()
    (tmp_path / "independent").mkdir()
    scope = {
        "role": "w16_hard_tape",
        "scientific_run": True,
        "weeks": 16,
        "portable_runtime_sha256": "9" * 64,
    }

    def fake_transfer(*, retrieved_root, **_kwargs):
        producer = retrieved_root.name == "producer"
        return {
            "passed": True,
            "failures": [],
            "role": "w16_hard_tape",
            "execution_role": "producer" if producer else "independent_replay",
            "replay_pair_id": "8" * 64,
            "scientific_run": True,
            "authorization_sha256": ("1" if producer else "2") * 64,
            "output_sha256": ("3" if producer else "4") * 64,
            "exact_receipt_sha256": ("5" if producer else "6") * 64,
            "trusted_parent_pid": 123 if producer else 456,
            "host_runtime_sha256": "e" * 64,
            "portable_scientific_scope": scope,
            "execution_witness_sha256": "7" * 64,
            "lexical_no_symlink_archive_paths_verified": True,
        }

    monkeypatch.setattr(harness, "verify_retrieved_reduced_transfer", fake_transfer)
    result = harness.verify_retrieved_reduced_pair(
        archive_root=tmp_path,
        producer_root_relative="producer",
        independent_root_relative="independent",
        producer_manifest_relative="transfer.json",
        independent_manifest_relative="transfer.json",
        expected_producer_manifest_sha256="a" * 64,
        expected_independent_manifest_sha256="b" * 64,
        expected_producer_public_key_fingerprint="c" * 64,
        expected_independent_public_key_fingerprint="d" * 64,
        role="w16_hard_tape",
    )
    assert result["passed"] is True
    assert result["relocated_pair_verification"] is True
    assert result["launch_host_inode_claims_preserved_not_reperformed"] is True


def test_relocated_pair_rejects_witness_or_scope_mismatch(tmp_path, monkeypatch):
    (tmp_path / "producer").mkdir()
    (tmp_path / "independent").mkdir()

    def fake_transfer(*, retrieved_root, **_kwargs):
        producer = retrieved_root.name == "producer"
        return {
            "passed": True,
            "failures": [],
            "role": "w16_hard_tape",
            "execution_role": "producer" if producer else "independent_replay",
            "replay_pair_id": "8" * 64,
            "scientific_run": True,
            "authorization_sha256": ("1" if producer else "2") * 64,
            "output_sha256": ("3" if producer else "4") * 64,
            "exact_receipt_sha256": ("5" if producer else "6") * 64,
            "trusted_parent_pid": 123 if producer else 456,
            "host_runtime_sha256": "e" * 64,
            "portable_scientific_scope": {"weeks": 16 if producer else 12},
            "execution_witness_sha256": ("7" if producer else "0") * 64,
            "lexical_no_symlink_archive_paths_verified": True,
        }

    monkeypatch.setattr(harness, "verify_retrieved_reduced_transfer", fake_transfer)
    result = harness.verify_retrieved_reduced_pair(
        archive_root=tmp_path,
        producer_root_relative="producer",
        independent_root_relative="independent",
        producer_manifest_relative="transfer.json",
        independent_manifest_relative="transfer.json",
        expected_producer_manifest_sha256="a" * 64,
        expected_independent_manifest_sha256="b" * 64,
        expected_producer_public_key_fingerprint="c" * 64,
        expected_independent_public_key_fingerprint="d" * 64,
        role="w16_hard_tape",
    )
    assert result["passed"] is False
    assert "retrieved pair portable scientific scopes differ" in result["failures"]
    assert "retrieved pair exact execution witnesses differ" in result["failures"]


def _portable_pair_fixture(tmp_path, monkeypatch):
    (tmp_path / "producer").mkdir()
    (tmp_path / "independent").mkdir()
    calls = []
    scope = {
        "role": "w16_hard_tape",
        "scientific_run": True,
        "weeks": 16,
        "portable_runtime_sha256": "9" * 64,
    }

    def fake_transfer(*, retrieved_root, **kwargs):
        calls.append((retrieved_root.name, kwargs))
        producer = retrieved_root.name == "producer"
        return {
            "passed": True,
            "failures": [],
            "role": "w16_hard_tape",
            "execution_role": "producer" if producer else "independent_replay",
            "replay_pair_id": "8" * 64,
            "scientific_run": True,
            "authorization_sha256": ("1" if producer else "2") * 64,
            "output_sha256": ("3" if producer else "4") * 64,
            "exact_receipt_sha256": ("5" if producer else "6") * 64,
            "trusted_parent_pid": 123 if producer else 456,
            "host_runtime_sha256": "e" * 64,
            "portable_scientific_scope": scope,
            "execution_witness_sha256": "7" * 64,
            "lexical_no_symlink_archive_paths_verified": True,
        }

    monkeypatch.setattr(harness, "verify_retrieved_reduced_transfer", fake_transfer)
    payload = harness.verify_retrieved_reduced_pair(
        archive_root=tmp_path,
        producer_root_relative="producer",
        independent_root_relative="independent",
        producer_manifest_relative="transfer.json",
        independent_manifest_relative="transfer.json",
        expected_producer_manifest_sha256="a" * 64,
        expected_independent_manifest_sha256="b" * 64,
        expected_producer_public_key_fingerprint="c" * 64,
        expected_independent_public_key_fingerprint="d" * 64,
        role="w16_hard_tape",
    )
    return payload, calls


def test_portable_pair_archive_reverification_ignores_cached_pass(tmp_path, monkeypatch):
    payload, calls = _portable_pair_fixture(tmp_path, monkeypatch)
    payload["producer_transfer_verification"] = {
        "passed": True,
        "output_sha256": "f" * 64,
    }
    payload["independent_transfer_verification"] = {
        "passed": True,
        "output_sha256": "f" * 64,
    }
    payload["verification_sha256"] = harness.canonical_json_sha256(
        {key: value for key, value in payload.items() if key != "verification_sha256"}
    )
    pair = tmp_path / "portable_pair.json"
    harness.exclusive_write_json(pair, payload)

    result = harness.reverify_retrieved_reduced_pair_archive(
        archive_root=tmp_path,
        pair_verification_path=pair,
        expected_pair_verification_sha256=harness.sha256_file(pair),
        role="w16_hard_tape",
        certified_artifact_sha256="3" * 64,
        expected_producer_manifest_sha256="a" * 64,
        expected_independent_manifest_sha256="b" * 64,
        expected_producer_public_key_fingerprint="c" * 64,
        expected_independent_public_key_fingerprint="d" * 64,
    )
    assert result["passed"] is True, result
    assert result["persisted_pass_claim_ignored"] is True
    assert result["cached_transfer_verifications_ignored"] is True
    assert len(calls) == 4
    assert result["current_verification"]["producer_transfer_verification"][
        "output_sha256"
    ] == "3" * 64


def test_portable_pair_archive_reverification_rejects_fabricated_cached_chain(
    tmp_path, monkeypatch
):
    payload, _calls = _portable_pair_fixture(tmp_path, monkeypatch)
    pair = tmp_path / "portable_pair.json"
    harness.exclusive_write_json(pair, payload)
    monkeypatch.setattr(
        harness,
        "verify_retrieved_reduced_pair",
        lambda **_kwargs: {
            "schema_version": harness.RETRIEVED_PAIR_VERIFICATION_SCHEMA,
            "passed": False,
            "failures": ["signed receipt fabrication"],
        },
    )
    result = harness.reverify_retrieved_reduced_pair_archive(
        archive_root=tmp_path,
        pair_verification_path=pair,
        expected_pair_verification_sha256=harness.sha256_file(pair),
        role="w16_hard_tape",
        certified_artifact_sha256="3" * 64,
        expected_producer_manifest_sha256="a" * 64,
        expected_independent_manifest_sha256="b" * 64,
        expected_producer_public_key_fingerprint="c" * 64,
        expected_independent_public_key_fingerprint="d" * 64,
    )
    assert result["passed"] is False
    assert any("fabrication" in failure for failure in result["failures"])
    assert any("absent" in failure for failure in result["failures"])


def test_portable_pair_archive_reverification_rejects_external_digest_and_paths(
    tmp_path, monkeypatch
):
    payload, _calls = _portable_pair_fixture(tmp_path, monkeypatch)
    pair = tmp_path / "portable_pair.json"
    harness.exclusive_write_json(pair, payload)
    digest_mismatch = harness.reverify_retrieved_reduced_pair_archive(
        archive_root=tmp_path,
        pair_verification_path=pair,
        expected_pair_verification_sha256=harness.sha256_file(pair),
        role="w16_hard_tape",
        certified_artifact_sha256="3" * 64,
        expected_producer_manifest_sha256="0" * 64,
        expected_independent_manifest_sha256="b" * 64,
        expected_producer_public_key_fingerprint="c" * 64,
        expected_independent_public_key_fingerprint="d" * 64,
    )
    assert digest_mismatch["passed"] is False
    assert any("inputs mismatch" in failure for failure in digest_mismatch["failures"])

    traversal = copy.deepcopy(payload)
    traversal["portable_archived_artifacts"]["producer"][
        "run_root_relative"
    ] = "../escape"
    traversal["verification_sha256"] = harness.canonical_json_sha256(
        {key: value for key, value in traversal.items() if key != "verification_sha256"}
    )
    traversal_path = tmp_path / "portable_pair_traversal.json"
    harness.exclusive_write_json(traversal_path, traversal)
    monkeypatch.undo()
    path_result = harness.reverify_retrieved_reduced_pair_archive(
        archive_root=tmp_path,
        pair_verification_path=traversal_path,
        expected_pair_verification_sha256=harness.sha256_file(traversal_path),
        role="w16_hard_tape",
        certified_artifact_sha256="3" * 64,
        expected_producer_manifest_sha256="a" * 64,
        expected_independent_manifest_sha256="b" * 64,
        expected_producer_public_key_fingerprint="c" * 64,
        expected_independent_public_key_fingerprint="d" * 64,
    )
    assert path_result["passed"] is False
    assert any("reconstruction failed" in failure for failure in path_result["failures"])


def test_portable_pair_archive_reverification_rejects_legacy_schema(
    tmp_path, monkeypatch
):
    payload, _calls = _portable_pair_fixture(tmp_path, monkeypatch)
    payload["schema_version"] = "paper2_reduced_retrieved_pair_verification_v1"
    payload["verification_sha256"] = harness.canonical_json_sha256(
        {key: value for key, value in payload.items() if key != "verification_sha256"}
    )
    pair = tmp_path / "legacy_pair.json"
    harness.exclusive_write_json(pair, payload)
    with pytest.raises(HarnessError, match="identity is invalid"):
        harness.reverify_retrieved_reduced_pair_archive(
            archive_root=tmp_path,
            pair_verification_path=pair,
            expected_pair_verification_sha256=harness.sha256_file(pair),
            role="w16_hard_tape",
            certified_artifact_sha256="3" * 64,
            expected_producer_manifest_sha256="a" * 64,
            expected_independent_manifest_sha256="b" * 64,
            expected_producer_public_key_fingerprint="c" * 64,
            expected_independent_public_key_fingerprint="d" * 64,
        )


def test_authorized_pair_dispatch_preserves_classic_and_rejects_legacy_mode(
    tmp_path, monkeypatch
):
    called = {}

    def fake_classic(**kwargs):
        called.update(kwargs)
        return {"passed": True, "failures": []}

    monkeypatch.setattr(harness, "reverify_reduced_pair_archive", fake_classic)
    result = harness.reverify_authorized_reduced_pair_archive(
        archive_root=tmp_path,
        pair_verification_path=tmp_path / "classic.json",
        expected_pair_verification_sha256="a" * 64,
        verification_mode=harness.CLASSIC_PAIR_VERIFICATION_MODE,
        role="w12_five_tape",
        certified_artifact_sha256="b" * 64,
        expected_producer_public_key_fingerprint="c" * 64,
        expected_independent_public_key_fingerprint="d" * 64,
    )
    assert result["passed"] is True
    assert called["role"] == "w12_five_tape"
    with pytest.raises(HarnessError, match="unknown or legacy"):
        harness.reverify_authorized_reduced_pair_archive(
            archive_root=tmp_path,
            pair_verification_path=tmp_path / "legacy.json",
            expected_pair_verification_sha256="a" * 64,
            verification_mode="",
            role="w12_five_tape",
            certified_artifact_sha256="b" * 64,
            expected_producer_public_key_fingerprint="c" * 64,
            expected_independent_public_key_fingerprint="d" * 64,
        )


def test_reduced_pair_verifier_requires_two_custody_bound_run_trees(
    tmp_path, monkeypatch
):
    import scripts.run_paper2_bottleneck_exact_transducer as exact

    pair_id = "a" * 64
    rows = []
    for label in ("producer", "independent"):
        run_dir = tmp_path / label
        run_dir.mkdir()
        output = run_dir / "result.json"
        output.write_text(json.dumps({"label": label, "pair": pair_id}))
        authorization = run_dir / "launch_authorization.json"
        authorization.write_text(json.dumps({"label": label, "pair": pair_id}))
        authorization_sha = harness.sha256_file(authorization)
        receipt = run_dir / "execution_receipt.json"
        receipt.write_text(
            json.dumps(
                {
                    "authorization_path": str(authorization.resolve()),
                    "authorization_sha256": authorization_sha,
                }
            )
        )
        runtime = run_dir / "runtime.json"
        runtime.write_text(json.dumps({"runtime": label}))
        rows.append(
            {
                "run_dir": run_dir,
                "output_sha": harness.sha256_file(output),
                "receipt_sha": harness.sha256_file(receipt),
                "authorization_sha": authorization_sha,
                "runtime_sha": harness.sha256_file(runtime),
            }
        )

    called = {}

    def fake_exact(*args, **kwargs):
        called["args"] = args
        called["kwargs"] = kwargs
        return {"passed": True, "failures": [], "verification_sha256": "f" * 64}

    monkeypatch.setattr(exact, "verify_independent_reduced_execution", fake_exact)
    output_path = tmp_path / "pair_verification.json"
    result = harness.verify_reduced_evidence_pair(
        archive_root=tmp_path,
        producer_run_dir=rows[0]["run_dir"],
        independent_run_dir=rows[1]["run_dir"],
        role="w16_hard_tape",
        producer_output_relative="result.json",
        independent_output_relative="result.json",
        expected_producer_output_sha256=rows[0]["output_sha"],
        expected_independent_output_sha256=rows[1]["output_sha"],
        producer_receipt_relative="execution_receipt.json",
        independent_receipt_relative="execution_receipt.json",
        expected_producer_receipt_sha256=rows[0]["receipt_sha"],
        expected_independent_receipt_sha256=rows[1]["receipt_sha"],
        producer_authorization_relative="launch_authorization.json",
        independent_authorization_relative="launch_authorization.json",
        expected_producer_authorization_sha256=rows[0]["authorization_sha"],
        expected_independent_authorization_sha256=rows[1]["authorization_sha"],
        expected_producer_public_key_fingerprint="1" * 64,
        expected_independent_public_key_fingerprint="2" * 64,
        producer_runtime_attestation_relative="runtime.json",
        independent_runtime_attestation_relative="runtime.json",
        expected_producer_runtime_attestation_sha256=rows[0]["runtime_sha"],
        expected_independent_runtime_attestation_sha256=rows[1]["runtime_sha"],
        output_path=output_path,
    )
    assert result["passed"] is True
    assert result["single_payload_validation_remains_fail_closed"] is True
    assert output_path.is_file()
    assert called["kwargs"]["producer_receipt_path"] == (
        rows[0]["run_dir"] / "execution_receipt.json"
    )
    assert any(
        "cannot authorize itself" in failure
        for failure in harness.validate_reduced_pair_verification_payload(
            result,
            role="w16_hard_tape",
            certified_artifact_sha256=rows[0]["output_sha"],
        )
    )
    archive_replay = harness.reverify_reduced_pair_archive(
        archive_root=tmp_path,
        pair_verification_path=output_path,
        expected_pair_verification_sha256=harness.sha256_file(output_path),
        role="w16_hard_tape",
        certified_artifact_sha256=rows[0]["output_sha"],
        expected_producer_public_key_fingerprint="1" * 64,
        expected_independent_public_key_fingerprint="2" * 64,
    )
    assert archive_replay["passed"] is True
    assert archive_replay["persisted_pass_claim_ignored"] is True
    assert called["kwargs"]["producer_runtime_attestation_path"] == (
        rows[0]["run_dir"] / "runtime.json"
    )


def test_fabricated_two_tree_pair_cannot_self_bless_without_signed_receipts(tmp_path):
    rows = []
    for index, label in enumerate(("producer", "independent")):
        run_dir = tmp_path / label
        run_dir.mkdir()
        output = run_dir / "result.json"
        output.write_text(json.dumps({"synthetic": True, "index": index}))
        authorization = run_dir / "authorization.json"
        authorization.write_text(json.dumps({"synthetic": True, "index": index}))
        authorization_sha = harness.sha256_file(authorization)
        receipt = run_dir / "harness_receipt.json"
        receipt.write_text(
            json.dumps(
                {
                    "schema_version": harness.SIGNED_HARNESS_RECEIPT_SCHEMA,
                    "exact_authorization_path": str(authorization.resolve()),
                    "exact_authorization_sha256": authorization_sha,
                    "fabricated": True,
                }
            )
        )
        runtime = run_dir / "runtime.json"
        runtime.write_text(json.dumps({"synthetic": True, "index": index}))
        rows.append(
            {
                "run_dir": run_dir,
                "output_sha": harness.sha256_file(output),
                "authorization_sha": authorization_sha,
                "receipt_sha": harness.sha256_file(receipt),
                "runtime_sha": harness.sha256_file(runtime),
            }
        )
    result = harness.verify_reduced_evidence_pair(
        archive_root=tmp_path,
        producer_run_dir=rows[0]["run_dir"],
        independent_run_dir=rows[1]["run_dir"],
        role="w16_hard_tape",
        producer_output_relative="result.json",
        independent_output_relative="result.json",
        expected_producer_output_sha256=rows[0]["output_sha"],
        expected_independent_output_sha256=rows[1]["output_sha"],
        producer_receipt_relative="harness_receipt.json",
        independent_receipt_relative="harness_receipt.json",
        expected_producer_receipt_sha256=rows[0]["receipt_sha"],
        expected_independent_receipt_sha256=rows[1]["receipt_sha"],
        producer_authorization_relative="authorization.json",
        independent_authorization_relative="authorization.json",
        expected_producer_authorization_sha256=rows[0]["authorization_sha"],
        expected_independent_authorization_sha256=rows[1]["authorization_sha"],
        expected_producer_public_key_fingerprint="4" * 64,
        expected_independent_public_key_fingerprint="5" * 64,
        producer_runtime_attestation_relative="runtime.json",
        independent_runtime_attestation_relative="runtime.json",
        expected_producer_runtime_attestation_sha256=rows[0]["runtime_sha"],
        expected_independent_runtime_attestation_sha256=rows[1]["runtime_sha"],
        output_path=tmp_path / "forged_pair.json",
    )
    assert result["passed"] is False
    assert any(
        "signature" in failure or "exact signed receipt chain" in failure
        for failure in result["failures"]
    )


def test_reduced_pair_verifier_rejects_one_tree_and_reused_custody(tmp_path):
    run_dir = tmp_path / "one"
    run_dir.mkdir()
    for name in (
        "result.json",
        "execution_receipt.json",
        "launch_authorization.json",
        "runtime.json",
    ):
        (run_dir / name).write_text("{}")
    digest = harness.sha256_file(run_dir / "result.json")
    result = harness.verify_reduced_evidence_pair(
        archive_root=tmp_path,
        producer_run_dir=run_dir,
        independent_run_dir=run_dir,
        role="w16_hard_tape",
        producer_output_relative="result.json",
        independent_output_relative="result.json",
        expected_producer_output_sha256=digest,
        expected_independent_output_sha256=digest,
        producer_receipt_relative="execution_receipt.json",
        independent_receipt_relative="execution_receipt.json",
        expected_producer_receipt_sha256=digest,
        expected_independent_receipt_sha256=digest,
        producer_authorization_relative="launch_authorization.json",
        independent_authorization_relative="launch_authorization.json",
        expected_producer_authorization_sha256=digest,
        expected_independent_authorization_sha256=digest,
        expected_producer_public_key_fingerprint="3" * 64,
        expected_independent_public_key_fingerprint="3" * 64,
        producer_runtime_attestation_relative="runtime.json",
        independent_runtime_attestation_relative="runtime.json",
        expected_producer_runtime_attestation_sha256=digest,
        expected_independent_runtime_attestation_sha256=digest,
        output_path=tmp_path / "failed_pair.json",
    )
    assert result["passed"] is False
    assert "producer and independent run trees are identical" in result["failures"]
    assert any("digests collide" in failure for failure in result["failures"])


def test_one_tape_w4_smoke_has_heartbeat_partial_status_and_checksums(tmp_path):
    run_dir = _prepare(tmp_path)
    assert execute_run(run_dir=run_dir, repo_root=ROOT, location="local") == 0

    status = json.loads((run_dir / "status" / "run_status.json").read_text())
    heartbeat = json.loads((run_dir / "status" / "heartbeat.json").read_text())
    seed_statuses = list((run_dir / "status" / "seeds").glob("*.json"))
    checksums = json.loads((run_dir / "artifact_checksums.json").read_text())

    assert status["state"] == "completed"
    assert status["completed_seed_count"] == 1
    assert status["evidence"] is False
    assert heartbeat["state"] == "completed"
    assert len(seed_statuses) == 1
    assert json.loads(seed_statuses[0].read_text())["state"] == "completed"
    assert checksums["record_count"] >= 5
    assert all(len(row["sha256"]) == 64 for row in checksums["records"])

    verification = verify_artifacts(run_dir, retrieved=False)
    assert verification["checks_passed"] is True
    assert verification["evidence"] is False
    assert "AUDIT_PENDING_NOT_EVIDENCE" in verification["evidence_status"]
    result = json.loads(next((run_dir / "artifacts").glob("*/result.json")).read_text())
    assert result["scientific_status"] == "NONSCIENTIFIC_SMOKE_NOT_EVIDENCE"
    assert result["scientific_run"] is False
    assert result["summary"]["not_evidence"] is True
    receipt = json.loads(
        next((run_dir / "status/jobs").glob("*.execution_receipt.json")).read_text()
    )
    assert receipt["execution_nonce"] == json.loads(
        (run_dir / "run_manifest.json").read_text()
    )["inputs"]["execution_nonce"]
    assert receipt["host_runtime_sha256"]
    assert receipt["isolated_bootstrap_sha256"]


def test_prepared_environment_digest_tampering_fails_before_execution(tmp_path):
    run_dir = _prepare(tmp_path)
    path = run_dir / "run_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["inputs"]["environment_sha256"] = "0" * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(HarnessError, match="environment digest changed"):
        execute_run(run_dir=run_dir, repo_root=ROOT, location="local")


def test_checksum_tampering_fails_retrieval_verification(tmp_path):
    run_dir = _prepare(tmp_path)
    assert execute_run(run_dir=run_dir, repo_root=ROOT, location="local") == 0
    artifact = next((run_dir / "artifacts").glob("*/result.json"))
    artifact.write_text("{}\n")
    verification = verify_artifacts(run_dir, retrieved=False)
    assert verification["checks_passed"] is False
    assert any("checksum mismatch" in reason for reason in verification["failures"])
    assert verification["evidence_status"] == "VERIFICATION_FAILED_NOT_EVIDENCE"


def test_checksum_manifest_is_anchored_and_covers_status_receipts(tmp_path):
    run_dir = _prepare(tmp_path)
    assert execute_run(run_dir=run_dir, repo_root=ROOT, location="local") == 0
    status = json.loads((run_dir / "status" / "run_status.json").read_text())
    checksum_path = run_dir / "artifact_checksums.json"
    checksums = json.loads(checksum_path.read_text())
    recorded = {row["path"] for row in checksums["records"]}
    seed_status = next((run_dir / "status" / "seeds").glob("*.json"))
    execution_receipt = next(
        (run_dir / "status" / "jobs").glob("*.execution_receipt.json")
    )

    assert status["checksums_sha256"] == harness.sha256_file(checksum_path)
    assert "status/run_completion_receipt.json" in recorded
    assert str(seed_status.relative_to(run_dir)) in recorded
    assert str(execution_receipt.relative_to(run_dir)) in recorded

    checksums["generated_at_utc"] = "tampered"
    checksum_path.write_text(json.dumps(checksums))
    verification = verify_artifacts(run_dir, retrieved=False)
    assert verification["checks_passed"] is False
    assert any("run status anchor" in reason for reason in verification["failures"])


def test_checksum_manifest_structure_and_confinement_fail_closed(tmp_path):
    baseline = _prepare(tmp_path)
    assert execute_run(run_dir=baseline, repo_root=ROOT, location="local") == 0

    cases = {
        "count": "record_count",
        "duplicate": "duplicate checksum record path",
        "traversal": "escapes run directory",
        "omission": "omits protected status/receipt paths",
    }
    for case, expected_failure in cases.items():
        run_dir = tmp_path / case
        shutil.copytree(baseline, run_dir)
        checksum_path = run_dir / "artifact_checksums.json"
        checksums = json.loads(checksum_path.read_text())
        if case == "count":
            checksums["record_count"] += 1
        elif case == "duplicate":
            checksums["records"].append(dict(checksums["records"][0]))
            checksums["record_count"] = len(checksums["records"])
        elif case == "traversal":
            checksums["records"][0]["path"] = "../outside.json"
        else:
            checksums["records"] = [
                row
                for row in checksums["records"]
                if not row["path"].startswith("status/seeds/")
            ]
            checksums["record_count"] = len(checksums["records"])
        checksum_path.write_text(json.dumps(checksums))
        status_path = run_dir / "status" / "run_status.json"
        status = json.loads(status_path.read_text())
        status["checksums_sha256"] = harness.sha256_file(checksum_path)
        status_path.write_text(json.dumps(status))

        verification = verify_artifacts(run_dir, retrieved=False)
        assert verification["checks_passed"] is False
        assert any(
            expected_failure in reason for reason in verification["failures"]
        )


def test_seed_status_tampering_is_checksum_protected(tmp_path):
    run_dir = _prepare(tmp_path)
    assert execute_run(run_dir=run_dir, repo_root=ROOT, location="local") == 0
    seed_status = next((run_dir / "status" / "seeds").glob("*.json"))
    payload = json.loads(seed_status.read_text())
    payload["state"] = "completed-but-tampered"
    seed_status.write_text(json.dumps(payload))

    verification = verify_artifacts(run_dir, retrieved=False)
    assert verification["checks_passed"] is False
    assert any(
        f"checksum mismatch {seed_status.relative_to(run_dir)}" in reason
        for reason in verification["failures"]
    )


def _grouped_verification_fixture(tmp_path: Path) -> Path:
    run_dir = tmp_path / "grouped-verification"
    run_id = "grouped-verification"
    seed_id = "seed_1100001_equipment_pressure"
    item_id = "frontier_calibration"
    output_relative = f"artifacts/{item_id}/result.json"
    stdout_relative = f"logs/{item_id}.stdout.log"
    stderr_relative = f"logs/{item_id}.stderr.log"
    materialized_argv = [
        "/runtime/python",
        "/repository/runner.py",
        "--output",
        "/run/out",
    ]
    materialization_context = {
        "python_executable": "/runtime/python",
        "repository_root": "/repository",
        "run_directory": "/run",
        "host_runtime_sha256": _fake_runtime_attestation()["runtime_sha256"],
        "execution_nonce": "9" * 64,
    }
    command_sha256 = harness.canonical_json_sha256(materialized_argv)
    evidence_status = "COMPLETED_HASHED_AUDIT_PENDING_NOT_EVIDENCE"

    seed_manifest = {
        "schema_version": harness.SCHEMA,
        "run_id": run_id,
        "mode": "smoke",
        "seed_count": 1,
        "seeds": [
            {
                "seed": 1_100_001,
                "context": "equipment_pressure",
                "split": "calibration",
                "weeks": 24,
                "expected_tape_sha256": harness._expected_tape_sha256(
                    1_100_001, "equipment_pressure", "calibration", 24
                ),
            }
        ],
    }
    argv_template = ["{python}", "{repo_root}/runner.py", "--output", "{run_dir}/out"]
    command_manifest = {
        "schema_version": harness.SCHEMA,
        "run_id": run_id,
        "mode": "smoke",
        "commands": [
            {
                "job_id": item_id,
                "phase": "calibration",
                "covered_seed_ids": [seed_id],
                "argv_template": argv_template,
                "argv_template_sha256": harness.canonical_json_sha256(
                    argv_template
                ),
                "output_relative": output_relative,
                "stdout_relative": stdout_relative,
                "stderr_relative": stderr_relative,
            }
        ],
    }
    harness.atomic_write_json(run_dir / "seed_manifest.json", seed_manifest)
    harness.atomic_write_json(run_dir / "command_manifest.json", command_manifest)
    (run_dir / "environment").mkdir(parents=True, exist_ok=True)
    (run_dir / "environment/pip_freeze.txt").write_text("pytest==1\n")
    harness.atomic_write_json(
        run_dir / "environment/machine.json", {"fixture": True}
    )
    runtime = _write_fake_runtime(run_dir)
    child_environment = scientific_child_environment()
    run_manifest = {
        "schema_version": harness.SCHEMA,
        "run_id": run_id,
        "mode": "smoke",
        "git": {"commit": "2" * 40},
        "inputs": {
            "contract_sha256": "3" * 64,
            "result_contract_sha256": None,
            "runner_sha256": "4" * 64,
            "harness_sha256": "5" * 64,
            "isolated_bootstrap_sha256": "7" * 64,
            "environment_sha256": "6" * 64,
            "host_runtime_attestation_relative": "environment/scientific_runtime.json",
            "host_runtime_sha256": runtime["runtime_sha256"],
            "portable_runtime_sha256": runtime["portable_sha256"],
            "scientific_child_environment": child_environment,
            "scientific_child_environment_sha256": harness.canonical_json_sha256(
                child_environment
            ),
            "execution_nonce": "9" * 64,
            "dependency_snapshot_sha256": harness.sha256_file(
                run_dir / "environment/pip_freeze.txt"
            ),
            "machine_snapshot_sha256": harness.sha256_file(
                run_dir / "environment/machine.json"
            ),
            "seed_manifest_sha256": harness.sha256_file(
                run_dir / "seed_manifest.json"
            ),
            "command_manifest_sha256": harness.sha256_file(
                run_dir / "command_manifest.json"
            ),
        },
    }
    harness.atomic_write_json(run_dir / "run_manifest.json", run_manifest)
    harness.atomic_write_json(run_dir / output_relative, {"completed": True})
    (run_dir / stdout_relative).parent.mkdir(parents=True, exist_ok=True)
    (run_dir / stdout_relative).write_text("completed\n")
    (run_dir / stderr_relative).write_text("")
    output_sha256 = harness.sha256_file(run_dir / output_relative)

    job_status = {
        "schema_version": harness.SCHEMA,
        "run_id": run_id,
        "item_id": item_id,
        "seed": None,
        "context": None,
        "phase": "calibration",
        "state": "completed",
        "returncode": 0,
        "command_sha256": command_sha256,
        "output_relative": output_relative,
        "output_valid_json": True,
        "output_sha256": output_sha256,
        "output_error": None,
        "stdout_relative": stdout_relative,
        "stderr_relative": stderr_relative,
        "evidence": False,
    }
    seed_status = {
        "schema_version": harness.SCHEMA,
        "run_id": run_id,
        "seed_id": seed_id,
        "seed": 1_100_001,
        "context": "equipment_pressure",
        "phase": "calibration",
        "state": "completed",
        "output_sha256": output_sha256,
        "returncode": 0,
        "evidence": False,
    }
    execution_receipt = {
        "schema_version": harness.SCHEMA,
        "run_id": run_id,
        "job_id": item_id,
        "mode": "smoke",
        "git_commit": run_manifest["git"]["commit"],
        "contract_sha256": run_manifest["inputs"]["contract_sha256"],
        "result_contract_sha256": None,
        "runner_sha256": run_manifest["inputs"]["runner_sha256"],
        "harness_sha256": run_manifest["inputs"]["harness_sha256"],
        "environment_sha256": run_manifest["inputs"]["environment_sha256"],
        "scientific_child_environment_sha256": run_manifest["inputs"][
            "scientific_child_environment_sha256"
        ],
        "execution_nonce": run_manifest["inputs"]["execution_nonce"],
        "execution_role": "frontier_calibration",
        "isolated_bootstrap_sha256": run_manifest["inputs"][
            "isolated_bootstrap_sha256"
        ],
        "host_runtime_sha256": run_manifest["inputs"]["host_runtime_sha256"],
        "portable_runtime_sha256": run_manifest["inputs"][
            "portable_runtime_sha256"
        ],
        "seed_manifest_sha256": run_manifest["inputs"]["seed_manifest_sha256"],
        "command_manifest_sha256": run_manifest["inputs"][
            "command_manifest_sha256"
        ],
        "materialization_context": materialization_context,
        "materialized_argv": materialized_argv,
        "materialized_command_sha256": command_sha256,
        "output_relative": output_relative,
        "output_sha256": output_sha256,
        "validated_not_independently_audited": True,
        "evidence": False,
    }
    completion_receipt = {
        "schema_version": harness.SCHEMA,
        "run_id": run_id,
        "state": "completed",
        "location": "local",
        "seed_count": 1,
        "completed_seed_count": 1,
        "failed_seed_count": 0,
        "run_manifest_sha256": harness.sha256_file(run_dir / "run_manifest.json"),
        "seed_manifest_sha256": harness.sha256_file(run_dir / "seed_manifest.json"),
        "command_manifest_sha256": harness.sha256_file(
            run_dir / "command_manifest.json"
        ),
        "immutable_status_snapshot": True,
        "evidence": False,
        "evidence_status": evidence_status,
    }
    harness.atomic_write_json(
        run_dir / "status" / "jobs" / f"{item_id}.json", job_status
    )
    harness.atomic_write_json(
        run_dir
        / "status"
        / "jobs"
        / f"{item_id}.execution_receipt.json",
        execution_receipt,
    )
    harness.atomic_write_json(
        run_dir / "status" / "seeds" / f"{seed_id}.json", seed_status
    )
    harness.atomic_write_json(
        run_dir / "status" / "run_completion_receipt.json", completion_receipt
    )
    protected = [
        "run_manifest.json",
        "seed_manifest.json",
        "command_manifest.json",
        "environment/pip_freeze.txt",
        "environment/machine.json",
        "environment/scientific_runtime.json",
        output_relative,
        stdout_relative,
        stderr_relative,
        f"status/jobs/{item_id}.json",
        f"status/jobs/{item_id}.execution_receipt.json",
        f"status/seeds/{seed_id}.json",
        "status/run_completion_receipt.json",
    ]
    checksums = {
        "schema_version": harness.SCHEMA,
        "run_id": run_id,
        "records": harness._checksum_records(run_dir, protected),
        "evidence": False,
    }
    checksums["record_count"] = len(checksums["records"])
    harness.atomic_write_json(run_dir / "artifact_checksums.json", checksums)
    harness.atomic_write_json(
        run_dir / "status" / "run_status.json",
        {
            "schema_version": harness.SCHEMA,
            "run_id": run_id,
            "state": "completed",
            "location": "local",
            "evidence": False,
            "evidence_status": evidence_status,
            "seed_count": 1,
            "completed_seed_count": 1,
            "failed_seed_count": 0,
            "checksums_sha256": harness.sha256_file(
                run_dir / "artifact_checksums.json"
            ),
            "independent_scientific_audit_required": True,
        },
    )
    return run_dir


def _rehash_verification_fixture(run_dir: Path) -> None:
    checksum_path = run_dir / "artifact_checksums.json"
    checksums = json.loads(checksum_path.read_text())
    for row in checksums["records"]:
        path = run_dir / row["path"]
        row["sha256"] = harness.sha256_file(path)
        row["bytes"] = path.stat().st_size
    harness.atomic_write_json(checksum_path, checksums)
    status_path = run_dir / "status" / "run_status.json"
    status = json.loads(status_path.read_text())
    status["checksums_sha256"] = harness.sha256_file(checksum_path)
    harness.atomic_write_json(status_path, status)


def _trusted_retrieved_copy(
    source: Path, tmp_path: Path, name: str
) -> tuple[Path, Path, str, Path, str]:
    custody_source = tmp_path / f"{name}.prepared-control"
    shutil.copytree(source, custody_source)
    (custody_source / "artifact_checksums.json").unlink(missing_ok=True)
    (custody_source / "status/run_completion_receipt.json").unlink(
        missing_ok=True
    )
    status_path = custody_source / "status/run_status.json"
    status = json.loads(status_path.read_text())
    status.update({"state": "prepared", "sealed_for_execution": True})
    harness.atomic_write_json(status_path, status)
    anchor_path = tmp_path / f"{name}.trusted-anchor.json"
    anchor_path, anchor_sha256 = harness.create_trusted_local_custody_anchor(
        custody_source, anchor_path=anchor_path
    )
    transport = tmp_path / f"{name}.transport"
    transport.mkdir()
    source_bundle = transport / "source.bundle"
    control_tar = transport / "control.tar"
    source_bundle.write_bytes(b"fixture source bundle\n")
    control_tar.write_bytes(b"fixture control archive\n")
    run_id = json.loads((custody_source / "run_manifest.json").read_text())["run_id"]
    stage_path, stage_sha256 = harness.create_trusted_stage_custody_receipt(
        custody_source,
        source_bundle=source_bundle,
        control_tar=control_tar,
        host=harness.DEFAULT_HOST,
        remote_run=f"~/paper2-bound-runs/{run_id}",
        custody_path=anchor_path,
        custody_sha256=anchor_sha256,
    )
    first_receipt = json.loads(
        next((source / "status/jobs").glob("*.execution_receipt.json")).read_text()
    )
    launch_path, launch_sha256 = harness.create_trusted_launch_custody_receipt(
        custody_source,
        stage_path=stage_path,
        stage_sha256=stage_sha256,
        materialization_context=first_receipt["materialization_context"],
        remote_python_requested="/runtime/python",
        remote_launch_shell_command="fixture-launch-command",
    )
    retrieved = tmp_path / name
    shutil.copytree(source, retrieved)
    return retrieved, anchor_path, anchor_sha256, launch_path, launch_sha256


def _refresh_control_chain_hashes(run_dir: Path) -> None:
    run_manifest_path = run_dir / "run_manifest.json"
    seed_manifest_path = run_dir / "seed_manifest.json"
    command_manifest_path = run_dir / "command_manifest.json"
    receipt_path = (
        run_dir
        / "status/jobs/frontier_calibration.execution_receipt.json"
    )
    receipt = json.loads(receipt_path.read_text())
    receipt["seed_manifest_sha256"] = harness.sha256_file(seed_manifest_path)
    receipt["command_manifest_sha256"] = harness.sha256_file(
        command_manifest_path
    )
    harness.atomic_write_json(receipt_path, receipt)
    completion_path = run_dir / "status/run_completion_receipt.json"
    completion = json.loads(completion_path.read_text())
    completion["run_manifest_sha256"] = harness.sha256_file(run_manifest_path)
    completion["seed_manifest_sha256"] = harness.sha256_file(seed_manifest_path)
    completion["command_manifest_sha256"] = harness.sha256_file(
        command_manifest_path
    )
    harness.atomic_write_json(completion_path, completion)
    _rehash_verification_fixture(run_dir)


def _coherently_replace_all_native_seed_identities(run_dir: Path) -> None:
    offset = 500_000
    seed_manifest_path = run_dir / "seed_manifest.json"
    seed_manifest = json.loads(seed_manifest_path.read_text())
    old_seed_ids = [
        f"seed_{row['seed']}_{row['context']}" for row in seed_manifest["seeds"]
    ]
    for row in seed_manifest["seeds"]:
        row["seed"] += offset
        row["expected_tape_sha256"] = harness._expected_tape_sha256(
            row["seed"], row["context"], row["split"], row["weeks"]
        )
    new_seed_ids = [
        f"seed_{row['seed']}_{row['context']}" for row in seed_manifest["seeds"]
    ]
    harness.atomic_write_json(seed_manifest_path, seed_manifest)

    command_manifest_path = run_dir / "command_manifest.json"
    command_manifest = json.loads(command_manifest_path.read_text())
    command_manifest["commands"][0]["covered_seed_ids"] = new_seed_ids
    harness.atomic_write_json(command_manifest_path, command_manifest)

    output_path = run_dir / "artifacts/frontier_calibration/result.json"
    payload = json.loads(output_path.read_text())
    for collection in (
        payload["transducers"],
        payload["build"]["checkpoints"],
        payload["build"]["collision_certificate_coverage"]["rows"],
    ):
        for row in collection:
            row["seed"] += offset
    for row in payload["selected_replays"]:
        row["seed"] += offset
    coverage = payload["build"]["collision_certificate_coverage"]
    coverage["rows_sha256"] = harness.canonical_json_sha256(coverage["rows"])
    coverage.pop("coverage_sha256")
    coverage["coverage_sha256"] = harness.canonical_json_sha256(coverage)
    harness.atomic_write_json(output_path, payload)
    output_sha256 = harness.sha256_file(output_path)

    runner_path = run_dir / "artifacts/frontier_calibration/runner_manifest.json"
    runner_manifest = json.loads(runner_path.read_text())
    for row in runner_manifest["seed_manifest"]:
        row["seed"] += offset
    for row in runner_manifest["checkpoint_artifacts"]:
        row["seed"] += offset
    runner_manifest["collision_certificate_coverage_sha256"] = coverage[
        "coverage_sha256"
    ]
    runner_manifest["result_sha256"] = output_sha256
    harness.atomic_write_json(runner_path, runner_manifest)

    run_manifest_path = run_dir / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text())
    run_manifest["inputs"]["seed_manifest_sha256"] = harness.sha256_file(
        seed_manifest_path
    )
    run_manifest["inputs"]["command_manifest_sha256"] = harness.sha256_file(
        command_manifest_path
    )
    harness.atomic_write_json(run_manifest_path, run_manifest)

    seed_status_dir = run_dir / "status/seeds"
    renamed_paths = {}
    for old_seed_id, new_seed_id, seed_row in zip(
        old_seed_ids, new_seed_ids, seed_manifest["seeds"]
    ):
        old_path = seed_status_dir / f"{old_seed_id}.json"
        document = json.loads(old_path.read_text())
        document["seed_id"] = new_seed_id
        document["seed"] = seed_row["seed"]
        document["output_sha256"] = output_sha256
        new_path = seed_status_dir / f"{new_seed_id}.json"
        harness.atomic_write_json(new_path, document)
        old_path.unlink()
        renamed_paths[
            f"status/seeds/{old_seed_id}.json"
        ] = f"status/seeds/{new_seed_id}.json"
    job_status_path = run_dir / "status/jobs/frontier_calibration.json"
    job_status = json.loads(job_status_path.read_text())
    job_status["output_sha256"] = output_sha256
    harness.atomic_write_json(job_status_path, job_status)
    receipt_path = (
        run_dir
        / "status/jobs/frontier_calibration.execution_receipt.json"
    )
    receipt = json.loads(receipt_path.read_text())
    receipt["output_sha256"] = output_sha256
    harness.atomic_write_json(receipt_path, receipt)

    checksum_path = run_dir / "artifact_checksums.json"
    checksums = json.loads(checksum_path.read_text())
    for record in checksums["records"]:
        if record["path"] in renamed_paths:
            record["path"] = renamed_paths[record["path"]]
    harness.atomic_write_json(checksum_path, checksums)
    _refresh_control_chain_hashes(run_dir)


def _coherently_replace_all_returned_proof_identities(run_dir: Path) -> None:
    """Forge every returned proof identity without touching anchored inputs."""
    output_path = run_dir / "artifacts/frontier_calibration/result.json"
    payload = json.loads(output_path.read_text())
    transducers = payload["transducers"]
    checkpoints = payload["build"]["checkpoints"]
    coverage = payload["build"]["collision_certificate_coverage"]
    by_seed = {}
    for index, (transducer, checkpoint, coverage_row) in enumerate(
        zip(transducers, checkpoints, coverage["rows"])
    ):
        tape = f"{10_000 + index:064x}"
        certificate = f"{20_000 + index:064x}"
        score = f"{30_000 + index:064x}"
        seed = transducer["seed"]
        transducer.update(
            {
                "tape_sha256": tape,
                "collision_certificate_sha256": certificate,
                "score_table_sha256": score,
            }
        )
        checkpoint["tape_sha256"] = tape
        checkpoint["score_table_sha256"] = score
        checkpoint["source_transducer_proof"]["collision_bisimulation"][
            "certificate_sha256"
        ] = certificate
        coverage_row["tape_sha256"] = tape
        coverage_row["certificate_sha256"] = certificate
        by_seed[seed] = (tape, certificate, score)
    for replay in payload["selected_replays"]:
        replay["tape_sha256"] = by_seed[replay["seed"]][0]
    payload["resolved_frontier"]["forged_delta_ret"] = 0.123456
    coverage["rows_sha256"] = harness.canonical_json_sha256(coverage["rows"])
    coverage.pop("coverage_sha256", None)
    coverage["coverage_sha256"] = harness.canonical_json_sha256(coverage)
    harness.atomic_write_json(output_path, payload)
    output_sha256 = harness.sha256_file(output_path)

    runner_path = run_dir / "artifacts/frontier_calibration/runner_manifest.json"
    runner = json.loads(runner_path.read_text())
    for row in runner["seed_manifest"]:
        row["tape_sha256"] = by_seed[row["seed"]][0]
    for row in runner["checkpoint_artifacts"]:
        _, certificate, score = by_seed[row["seed"]]
        row["collision_certificate_sha256"] = certificate
        row["score_table_sha256"] = score
    runner["collision_certificate_coverage_sha256"] = coverage[
        "coverage_sha256"
    ]
    runner["result_sha256"] = output_sha256
    harness.atomic_write_json(runner_path, runner)
    for path in [
        run_dir / "status/jobs/frontier_calibration.json",
        run_dir / "status/jobs/frontier_calibration.execution_receipt.json",
        *sorted((run_dir / "status/seeds").glob("*.json")),
    ]:
        document = json.loads(path.read_text())
        document["output_sha256"] = output_sha256
        harness.atomic_write_json(path, document)
    _rehash_verification_fixture(run_dir)


def test_retrieved_verification_requires_external_launch_custody_chain(tmp_path):
    run_dir = _grouped_verification_fixture(tmp_path)
    verification = verify_artifacts(run_dir, retrieved=True)
    assert verification["checks_passed"] is False
    assert any(
        "trusted launch custody receipt" in failure
        for failure in verification["failures"]
    )
    retrieved, anchor_path, anchor_sha256, launch_path, launch_sha256 = (
        _trusted_retrieved_copy(run_dir, tmp_path, "explicit-custody-copy")
    )
    verification = verify_artifacts(
        retrieved,
        retrieved=True,
        trusted_local_anchor=anchor_path,
        trusted_local_anchor_sha256=anchor_sha256,
        trusted_launch_receipt=launch_path,
        trusted_launch_receipt_sha256=launch_sha256,
    )
    assert verification["checks_passed"] is True
    assert "CUSTODY_AND_HASH_VERIFIED" in verification["evidence_status"]
    assert verification["evidence"] is False


def test_trusted_custody_anchor_is_external_and_never_overwritten(tmp_path):
    run_dir = _prepare(tmp_path)
    anchor_path, anchor_sha256 = harness.create_trusted_local_custody_anchor(
        run_dir
    )
    assert anchor_path.parent == run_dir.parent
    assert anchor_path != run_dir / anchor_path.name
    original_bytes = anchor_path.read_bytes()
    repeated_path, repeated_sha256 = harness.create_trusted_local_custody_anchor(
        run_dir
    )
    assert repeated_path == anchor_path
    assert repeated_sha256 == anchor_sha256

    command_path = run_dir / "command_manifest.json"
    command = json.loads(command_path.read_text())
    command["commands"][0]["argv_template"].append("--forged")
    harness.atomic_write_json(command_path, command)
    with pytest.raises(HarnessError, match="will not be overwritten"):
        harness.create_trusted_local_custody_anchor(run_dir)
    assert anchor_path.read_bytes() == original_bytes


def test_external_custody_rejects_coherent_command_template_context_forgery(
    tmp_path,
):
    source = _grouped_verification_fixture(tmp_path)
    retrieved, anchor_path, anchor_sha256, launch_path, launch_sha256 = _trusted_retrieved_copy(
        source, tmp_path, "retrieved-command-forgery"
    )
    command_manifest_path = retrieved / "command_manifest.json"
    command_manifest = json.loads(command_manifest_path.read_text())
    command = command_manifest["commands"][0]
    command["argv_template"][1] = "{repo_root}/forged_runner.py"
    command["argv_template_sha256"] = harness.canonical_json_sha256(
        command["argv_template"]
    )
    harness.atomic_write_json(command_manifest_path, command_manifest)
    run_manifest_path = retrieved / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text())
    run_manifest["inputs"]["command_manifest_sha256"] = harness.sha256_file(
        command_manifest_path
    )
    harness.atomic_write_json(run_manifest_path, run_manifest)
    receipt_path = (
        retrieved
        / "status/jobs/frontier_calibration.execution_receipt.json"
    )
    receipt = json.loads(receipt_path.read_text())
    receipt["materialization_context"]["repository_root"] = "/forged-repository"
    receipt["materialized_argv"][1] = "/forged-repository/forged_runner.py"
    forged_sha256 = harness.canonical_json_sha256(receipt["materialized_argv"])
    receipt["materialized_command_sha256"] = forged_sha256
    harness.atomic_write_json(receipt_path, receipt)
    job_status_path = retrieved / "status/jobs/frontier_calibration.json"
    job_status = json.loads(job_status_path.read_text())
    job_status["command_sha256"] = forged_sha256
    harness.atomic_write_json(job_status_path, job_status)
    _refresh_control_chain_hashes(retrieved)

    assert verify_artifacts(retrieved, retrieved=False)["checks_passed"] is True
    verification = verify_artifacts(
        retrieved,
        retrieved=True,
        trusted_local_anchor=anchor_path,
        trusted_local_anchor_sha256=anchor_sha256,
        trusted_launch_receipt=launch_path,
        trusted_launch_receipt_sha256=launch_sha256,
    )
    assert verification["checks_passed"] is False
    assert any(
        "trusted local custody byte mismatch" in failure
        for failure in verification["failures"]
    )


def test_external_custody_rejects_coherent_all_60_identity_bundle_forgery(
    tmp_path,
):
    source = _native_scientific_verification_fixture(tmp_path)
    retrieved, anchor_path, anchor_sha256, launch_path, launch_sha256 = _trusted_retrieved_copy(
        source, tmp_path, "retrieved-all-60-forgery"
    )
    _coherently_replace_all_native_seed_identities(retrieved)

    assert verify_artifacts(retrieved, retrieved=False)["checks_passed"] is False
    verification = verify_artifacts(
        retrieved,
        retrieved=True,
        trusted_local_anchor=anchor_path,
        trusted_local_anchor_sha256=anchor_sha256,
        trusted_launch_receipt=launch_path,
        trusted_launch_receipt_sha256=launch_sha256,
    )
    assert verification["checks_passed"] is False
    assert any(
        "trusted local custody byte mismatch: seed_manifest.json" == failure
        for failure in verification["failures"]
    )


def test_precommitted_tapes_reject_full_60_returned_identity_forgery(tmp_path):
    source = _native_scientific_verification_fixture(tmp_path)
    retrieved, anchor_path, anchor_sha256, launch_path, launch_sha256 = (
        _trusted_retrieved_copy(
            source, tmp_path, "retrieved-return-only-60-forgery"
        )
    )
    _coherently_replace_all_returned_proof_identities(retrieved)

    verification = verify_artifacts(
        retrieved,
        retrieved=True,
        trusted_local_anchor=anchor_path,
        trusted_local_anchor_sha256=anchor_sha256,
        trusted_launch_receipt=launch_path,
        trusted_launch_receipt_sha256=launch_sha256,
    )
    assert verification["checks_passed"] is False
    assert any(
        "frontier harness/runner seed identity mismatch" in failure
        for failure in verification["failures"]
    )


def test_launch_custody_rejects_context_only_argv_substitution(tmp_path):
    source = _grouped_verification_fixture(tmp_path)
    retrieved, anchor_path, anchor_sha256, launch_path, launch_sha256 = (
        _trusted_retrieved_copy(
            source, tmp_path, "retrieved-context-only-forgery"
        )
    )
    receipt_path = (
        retrieved
        / "status/jobs/frontier_calibration.execution_receipt.json"
    )
    receipt = json.loads(receipt_path.read_text())
    receipt["materialization_context"]["repository_root"] = "/attacker/repository"
    command = json.loads((retrieved / "command_manifest.json").read_text())[
        "commands"
    ][0]
    receipt["materialized_argv"] = harness._materialize_argv_from_context(
        command["argv_template"], receipt["materialization_context"]
    )
    forged_sha256 = harness.canonical_json_sha256(receipt["materialized_argv"])
    receipt["materialized_command_sha256"] = forged_sha256
    harness.atomic_write_json(receipt_path, receipt)
    status_path = retrieved / "status/jobs/frontier_calibration.json"
    status = json.loads(status_path.read_text())
    status["command_sha256"] = forged_sha256
    harness.atomic_write_json(status_path, status)
    _rehash_verification_fixture(retrieved)

    assert verify_artifacts(retrieved, retrieved=False)["checks_passed"] is True
    verification = verify_artifacts(
        retrieved,
        retrieved=True,
        trusted_local_anchor=anchor_path,
        trusted_local_anchor_sha256=anchor_sha256,
        trusted_launch_receipt=launch_path,
        trusted_launch_receipt_sha256=launch_sha256,
    )
    assert verification["checks_passed"] is False
    assert (
        "launch custody execution mismatch: "
        "frontier_calibration:materialization_context"
        in verification["failures"]
    )


def test_completed_bundle_cannot_mint_a_posthoc_base_anchor(tmp_path):
    completed = _grouped_verification_fixture(tmp_path)
    with pytest.raises(
        HarnessError,
        match="only be minted for a sealed pre-execution run",
    ):
        harness.create_trusted_local_custody_anchor(
            completed,
            anchor_path=tmp_path / "posthoc-anchor.json",
        )


def test_stage_receipt_binds_archive_bytes_and_rejects_digest_swap(tmp_path):
    run_dir = _prepare(tmp_path)
    anchor_path, anchor_sha256 = harness.create_trusted_local_custody_anchor(run_dir)
    source_bundle = tmp_path / "source.bundle"
    control_tar = tmp_path / "control.tar"
    source_bundle.write_bytes(b"immutable source\n")
    control_tar.write_bytes(b"immutable control\n")
    stage_path, stage_sha256 = harness.create_trusted_stage_custody_receipt(
        run_dir,
        source_bundle=source_bundle,
        control_tar=control_tar,
        host=harness.DEFAULT_HOST,
        remote_run="~/paper2-bound-runs/pytest-smoke",
        custody_path=anchor_path,
        custody_sha256=anchor_sha256,
    )
    stage = json.loads(stage_path.read_text())
    assert stage["source_bundle"]["sha256"] == harness.sha256_file(source_bundle)
    assert stage["control_tar"]["sha256"] == harness.sha256_file(control_tar)

    stage["control_tar"]["sha256"] = "f" * 64
    stage_body = dict(stage)
    stage_body.pop("stage_body_sha256")
    stage["stage_body_sha256"] = harness.canonical_json_sha256(stage_body)
    harness.atomic_write_json(stage_path, stage)
    with pytest.raises(HarnessError, match="missing or changed"):
        harness.create_trusted_launch_custody_receipt(
            run_dir,
            stage_path=stage_path,
            stage_sha256=stage_sha256,
            materialization_context={
                "python_executable": "/runtime/python",
                "repository_root": "/remote/source",
                "run_directory": "/remote/control",
                "host_runtime_sha256": "8" * 64,
                "execution_nonce": "9" * 64,
            },
            remote_python_requested="/runtime/python",
            remote_launch_shell_command="fixture",
        )


def test_retrieval_uses_external_launch_receipt_not_mutable_stage_pointer(
    monkeypatch, tmp_path
):
    remote_control = _grouped_verification_fixture(tmp_path)
    local_run = tmp_path / "local-run"
    shutil.copytree(remote_control, local_run)
    (local_run / "artifact_checksums.json").unlink()
    (local_run / "status/run_completion_receipt.json").unlink()
    local_status_path = local_run / "status/run_status.json"
    local_status = json.loads(local_status_path.read_text())
    local_status.update({"state": "prepared", "sealed_for_execution": True})
    harness.atomic_write_json(local_status_path, local_status)
    anchor_path, anchor_sha256 = harness.create_trusted_local_custody_anchor(
        local_run
    )
    source_bundle = tmp_path / "pointer-source.bundle"
    control_tar = tmp_path / "pointer-control.tar"
    source_bundle.write_bytes(b"source\n")
    control_tar.write_bytes(b"control\n")
    stage_path, stage_sha256 = harness.create_trusted_stage_custody_receipt(
        local_run,
        source_bundle=source_bundle,
        control_tar=control_tar,
        host=harness.DEFAULT_HOST,
        remote_run="~/paper2-bound-runs/trusted-run",
        custody_path=anchor_path,
        custody_sha256=anchor_sha256,
    )
    context = json.loads(
        (
            remote_control
            / "status/jobs/frontier_calibration.execution_receipt.json"
        ).read_text()
    )["materialization_context"]
    launch_path, launch_sha256 = harness.create_trusted_launch_custody_receipt(
        local_run,
        stage_path=stage_path,
        stage_sha256=stage_sha256,
        materialization_context=context,
        remote_python_requested="/runtime/python",
        remote_launch_shell_command="trusted-launch",
    )
    harness.atomic_write_json(
        local_run / "status/remote_stage.json",
        {
            "remote_run": "~/paper2-bound-runs/attacker-run",
            "trusted_local_anchor_path": "/tmp/attacker-anchor.json",
            "trusted_local_anchor_sha256": "f" * 64,
        },
    )
    captured = {}

    def fake_run_capture(argv, **_kwargs):
        captured["argv"] = argv
        destination = Path(argv[-1])
        for child in remote_control.iterdir():
            target = destination / child.name
            if child.is_dir():
                shutil.copytree(child, target)
            else:
                shutil.copy2(child, target)
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(harness, "run_capture", fake_run_capture)
    result = harness.retrieve_vps(
        run_dir=local_run,
        host=harness.DEFAULT_HOST,
        trusted_launch_receipt=launch_path,
        trusted_launch_receipt_sha256=launch_sha256,
    )
    assert result["verification_passed"] is True
    assert "trusted-run" in captured["argv"][-2]
    assert "attacker-run" not in captured["argv"][-2]


def test_retrieval_requires_caller_retained_launch_digest(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(HarnessError, match="caller-retained"):
        harness.retrieve_vps(run_dir=run_dir, host=harness.DEFAULT_HOST)


def test_nonce_less_launch_receipt_fails_even_with_newly_retained_digest(tmp_path):
    completed = _grouped_verification_fixture(tmp_path)
    retrieved, anchor_path, anchor_sha256, launch_path, _launch_sha256 = (
        _trusted_retrieved_copy(completed, tmp_path, "nonce-required")
    )
    launch = json.loads(launch_path.read_text())
    launch.pop("custody_nonce")
    launch_body = dict(launch)
    launch_body.pop("launch_body_sha256")
    launch["launch_body_sha256"] = harness.canonical_json_sha256(launch_body)
    harness.atomic_write_json(launch_path, launch)
    newly_retained_digest = harness.sha256_file(launch_path)

    verification = verify_artifacts(
        retrieved,
        retrieved=True,
        trusted_local_anchor=anchor_path,
        trusted_local_anchor_sha256=anchor_sha256,
        trusted_launch_receipt=launch_path,
        trusted_launch_receipt_sha256=newly_retained_digest,
    )
    assert verification["checks_passed"] is False
    assert (
        "trusted launch custody receipt custody nonce is missing or malformed"
        in verification["failures"]
    )


def test_posthoc_full_chain_cannot_match_original_retained_launch_digest(
    tmp_path,
):
    completed = _grouped_verification_fixture(tmp_path)
    untouched, _base, _base_sha, original_launch, original_digest = (
        _trusted_retrieved_copy(
            completed, tmp_path, "legitimate-preexecution-chain"
        )
    )

    rewound = tmp_path / "posthoc-rewound"
    shutil.copytree(completed, rewound)
    (rewound / "artifact_checksums.json").unlink()
    (rewound / "status/run_completion_receipt.json").unlink()
    status_path = rewound / "status/run_status.json"
    status = json.loads(status_path.read_text())
    status.update({"state": "prepared", "sealed_for_execution": True})
    harness.atomic_write_json(status_path, status)
    forged_base, forged_base_sha = harness.create_trusted_local_custody_anchor(
        rewound
    )
    source_bundle = tmp_path / "posthoc-source.bundle"
    control_tar = tmp_path / "posthoc-control.tar"
    source_bundle.write_bytes(b"fixture source bundle\n")
    control_tar.write_bytes(b"fixture control archive\n")
    forged_stage, forged_stage_sha = harness.create_trusted_stage_custody_receipt(
        rewound,
        source_bundle=source_bundle,
        control_tar=control_tar,
        host=harness.DEFAULT_HOST,
        remote_run="~/paper2-bound-runs/grouped-verification",
        custody_path=forged_base,
        custody_sha256=forged_base_sha,
    )
    context = json.loads(
        (
            completed
            / "status/jobs/frontier_calibration.execution_receipt.json"
        ).read_text()
    )["materialization_context"]
    forged_launch, forged_digest = harness.create_trusted_launch_custody_receipt(
        rewound,
        stage_path=forged_stage,
        stage_sha256=forged_stage_sha,
        materialization_context=context,
        remote_python_requested="/runtime/python",
        remote_launch_shell_command="fixture-launch-command",
    )
    assert forged_digest != original_digest
    assert original_launch.is_file()

    verification = verify_artifacts(
        untouched,
        retrieved=True,
        trusted_launch_receipt=forged_launch,
        trusted_launch_receipt_sha256=original_digest,
    )
    assert verification["checks_passed"] is False
    assert "trusted launch custody receipt digest mismatch" in verification[
        "failures"
    ]


def test_rewritten_stage_and_launch_fail_original_retained_digest(tmp_path):
    completed = _grouped_verification_fixture(tmp_path)
    retrieved, _base, _base_sha, launch_path, retained_digest = (
        _trusted_retrieved_copy(
            completed, tmp_path, "rewritten-legitimate-chain"
        )
    )
    launch = json.loads(launch_path.read_text())
    stage_path = Path(launch["trusted_stage_custody"]["path"])
    stage = json.loads(stage_path.read_text())
    stage["remote"]["remote_run"] = "~/paper2-bound-runs/attacker-run"
    stage["source_bundle"]["sha256"] = "a" * 64
    stage["control_tar"]["sha256"] = "b" * 64
    stage_body = dict(stage)
    stage_body.pop("stage_body_sha256")
    stage["stage_body_sha256"] = harness.canonical_json_sha256(stage_body)
    harness.atomic_write_json(stage_path, stage)
    rewritten_stage_digest = harness.sha256_file(stage_path)

    launch["trusted_stage_custody"]["sha256"] = rewritten_stage_digest
    launch["remote"] = stage["remote"]
    launch["source_bundle"] = stage["source_bundle"]
    launch["control_tar"] = stage["control_tar"]
    launch_body = dict(launch)
    launch_body.pop("launch_body_sha256")
    launch["launch_body_sha256"] = harness.canonical_json_sha256(launch_body)
    harness.atomic_write_json(launch_path, launch)
    assert harness.sha256_file(launch_path) != retained_digest

    verification = verify_artifacts(
        retrieved,
        retrieved=True,
        trusted_launch_receipt=launch_path,
        trusted_launch_receipt_sha256=retained_digest,
    )
    assert verification["checks_passed"] is False
    assert "trusted launch custody receipt digest mismatch" in verification[
        "failures"
    ]


@pytest.mark.parametrize(
    ("target", "field", "value", "expected_failure"),
    [
        (
            "status/jobs/frontier_calibration.execution_receipt.json",
            "output_sha256",
            "f" * 64,
            "execution receipt mismatch: frontier_calibration:output_sha256",
        ),
        (
            "status/jobs/frontier_calibration.execution_receipt.json",
            "output_relative",
            "artifacts/wrong/result.json",
            "execution receipt mismatch: frontier_calibration:output_relative",
        ),
        (
            "status/jobs/frontier_calibration.execution_receipt.json",
            "materialized_command_sha256",
            "f" * 64,
            "execution receipt mismatch: frontier_calibration:materialized_command_sha256",
        ),
        (
            "status/jobs/frontier_calibration.json",
            "phase",
            "locked",
            "command/job status mismatch: frontier_calibration:phase",
        ),
        (
            "status/jobs/frontier_calibration.json",
            "output_valid_json",
            False,
            "command/job status mismatch: frontier_calibration:output_valid_json",
        ),
        (
            "status/jobs/frontier_calibration.json",
            "returncode",
            1,
            "command/job status mismatch: frontier_calibration:returncode",
        ),
        (
            "status/seeds/seed_1100001_equipment_pressure.json",
            "seed",
            1_100_002,
            "per-seed status mismatch: seed_1100001_equipment_pressure:seed",
        ),
        (
            "status/seeds/seed_1100001_equipment_pressure.json",
            "context",
            "mission_surge",
            "per-seed status mismatch: seed_1100001_equipment_pressure:context",
        ),
        (
            "status/seeds/seed_1100001_equipment_pressure.json",
            "phase",
            "locked",
            "per-seed status mismatch: seed_1100001_equipment_pressure:phase",
        ),
        (
            "status/seeds/seed_1100001_equipment_pressure.json",
            "seed_id",
            "seed_1100002_equipment_pressure",
            "per-seed status mismatch: seed_1100001_equipment_pressure:seed_id",
        ),
    ],
)
def test_verify_artifacts_rejects_rehashed_semantic_receipt_and_status_tampering(
    tmp_path, target, field, value, expected_failure
):
    run_dir = _grouped_verification_fixture(tmp_path)
    assert verify_artifacts(run_dir, retrieved=False)["checks_passed"] is True
    path = run_dir / target
    payload = json.loads(path.read_text())
    payload[field] = value
    harness.atomic_write_json(path, payload)
    _rehash_verification_fixture(run_dir)

    verification = verify_artifacts(run_dir, retrieved=False)
    assert verification["checks_passed"] is False
    assert expected_failure in verification["failures"]


def test_dry_run_and_non_scientific_vps_launch_fail_closed(tmp_path):
    run_dir = _prepare(tmp_path, mode="dry-run")
    with pytest.raises(HarnessError, match="dry-run manifest cannot execute"):
        execute_run(run_dir=run_dir, repo_root=ROOT, location="local")

    (run_dir / "status" / "remote_stage.json").write_text(
        json.dumps({"remote_run": "~/paper2-bound-runs/pytest-dry-run"})
    )
    with pytest.raises(HarnessError, match="sealed evidence-execution manifest"):
        launch_vps(
            run_dir=run_dir,
            host="ovh-agent-lab",
            remote_python="~/scres-ia/.venv/bin/python",
        )


def test_vps_launcher_groups_only_detached_process(monkeypatch, tmp_path):
    run_dir = tmp_path / "sealed"
    (run_dir / "status").mkdir(parents=True)
    (run_dir / "environment").mkdir()
    (run_dir / "environment/pip_freeze.txt").write_text("pytest==1\n")
    harness.atomic_write_json(run_dir / "environment/machine.json", {"fixture": True})
    runtime = _write_fake_runtime(run_dir)
    harness.atomic_write_json(
        run_dir / "seed_manifest.json",
        {
            "schema_version": harness.SCHEMA,
            "run_id": "pytest-sealed",
            "mode": "reduced_w12",
            "seed_count": 1,
            "seeds": [],
        },
    )
    argv_template = ["{python}", "{repo_root}/runner.py", "--output", "{run_dir}/out"]
    harness.atomic_write_json(
        run_dir / "command_manifest.json",
        {
            "schema_version": harness.SCHEMA,
            "run_id": "pytest-sealed",
            "mode": "reduced_w12",
            "commands": [
                {
                    "job_id": "fixture",
                    "argv_template": argv_template,
                    "argv_template_sha256": harness.canonical_json_sha256(
                        argv_template
                    ),
                }
            ],
        },
    )
    harness.atomic_write_json(
        run_dir / "run_manifest.json",
        {
            "schema_version": harness.SCHEMA,
            "run_id": "pytest-sealed",
            "mode": "reduced_w12",
            "git": {"commit": "1" * 40},
            "execution": {"sealed_for_execution": True},
            "inputs": {
                "harness_relative": "scripts/paper2_bound_execution_harness.py",
                "runner_relative": "runner.py",
                "portable_runtime_sha256": runtime["portable_sha256"],
                "host_runtime_sha256": runtime["runtime_sha256"],
                "execution_nonce": "9" * 64,
                "isolated_bootstrap_sha256": "7" * 64,
                "scientific_child_environment_sha256": harness.canonical_json_sha256(
                    scientific_child_environment()
                ),
            },
        },
    )
    harness.atomic_write_json(
        run_dir / "status/run_status.json",
        {"state": "prepared", "sealed_for_execution": True},
    )
    anchor_path, anchor_sha256 = harness.create_trusted_local_custody_anchor(run_dir)
    source_bundle = tmp_path / "source.bundle"
    control_tar = tmp_path / "control.tar"
    source_bundle.write_bytes(b"source\n")
    control_tar.write_bytes(b"control\n")
    harness.create_trusted_stage_custody_receipt(
        run_dir,
        source_bundle=source_bundle,
        control_tar=control_tar,
        host=harness.DEFAULT_HOST,
        remote_run="~/paper2-bound-runs/pytest-sealed",
        custody_path=anchor_path,
        custody_sha256=anchor_sha256,
    )
    captured = {"calls": []}

    def fake_run_capture(argv, **_kwargs):
        captured["calls"].append(argv)
        if " -c " in argv[-1]:
            return SimpleNamespace(
                stdout=json.dumps(
                    {
                        "python_executable": "/runtime/python",
                        "repository_root": "/remote/source",
                        "run_directory": "/remote/control",
                        "host_runtime_sha256": "8" * 64,
                        "execution_nonce": "9" * 64,
                    }
                )
            )
        if "--attest-only" in argv[-1]:
            remote_runtime = _fake_runtime_attestation()
            remote_runtime["portable"] = runtime["portable"]
            remote_runtime["portable_sha256"] = runtime["portable_sha256"]
            body = dict(remote_runtime)
            body.pop("runtime_sha256")
            remote_runtime["runtime_sha256"] = harness.canonical_json_sha256(body)
            return SimpleNamespace(stdout=json.dumps(remote_runtime))
        return SimpleNamespace(stdout="12345\n")

    monkeypatch.setattr(harness, "run_capture", fake_run_capture)
    payload = launch_vps(
        run_dir=run_dir,
        host="ovh-agent-lab",
        remote_python="~/scres-ia/.venv/bin/python",
    )

    remote_command = captured["calls"][-1][-1]
    assert "&& { nohup " in remote_command
    assert " -I -B " in remote_command
    assert "< /dev/null & echo $!; }" in remote_command
    assert payload["remote_launcher_pid"] == 12345
    assert (run_dir / "status" / "remote_submission.json").is_file()


def test_primary_bound_scope_requires_exact_primary_replay_but_not_full_guardrail():
    manifest = {
        "inputs": {"contract_sha256": "c" * 64, "runner_sha256": "r" * 64},
        "execution": {"authorization_scope": "primary_bound_only"},
    }
    payload = {
        "execution_assurance": {
            "key_schema_version": "paper2_bottleneck_semantic_markov_key_v4",
            "primary_frontier_exact": True,
            "numeric_approximation_gap": 0,
            "original_runner_replay_passed": True,
            "resource_semantics_passed": True,
            "contract_sha256": "c" * 64,
            "runner_sha256": "r" * 64,
            "full_guardrail_frontier_exact": False,
            "all_mandatory_guardrails_audited": False,
        },
        "promotion_authorized": False,
        "learner_authorized": False,
        "paper3_authorized": False,
    }
    assert _validate_scientific_result(payload, manifest) == []

    payload["promotion_authorized"] = True
    assert any(
        "prohibited authorization" in failure
        for failure in _validate_scientific_result(payload, manifest)
    )

    payload["promotion_authorized"] = False
    manifest["execution"]["authorization_scope"] = "full_guardrail_frontier"
    failures = _validate_scientific_result(payload, manifest)
    assert any("exact guardrail frontier" in failure for failure in failures)
    assert any("mandatory guardrails" in failure for failure in failures)


def test_scientific_phase_command_covers_frozen_block_without_per_seed_relaunch():
    specs = _frozen_phase_seed_specs("locked")
    rows = [
        {"seed": seed, "context": context, "split": "locked", "weeks": 24}
        for seed, context in specs
    ]
    commands = _frontier_command_row(
        rows,
        runner_rel="scripts/run_paper2_bottleneck_full_frontier.py",
        phase="locked",
        weeks=24,
        calibration_result_rel="results/paper2_bottleneck/calibration.json",
        batch_size=65_536,
        max_contenders=100_000,
        build_workers=4,
    )
    assert len(specs) == 119
    assert len(commands) == 1
    assert len(commands[0]["covered_seed_ids"]) == 119
    assert commands[0]["runner_mode"] == "exact_primary_frontier_phase"
    argv = commands[0]["argv_template"]
    assert argv[argv.index("--authorization") + 1] == "{run_dir}/authorization.json"
    assert argv[argv.index("--build-workers") + 1] == "4"
    assert argv[argv.index("--checkpoint-dir") + 1].startswith("{run_dir}/")
    assert argv[argv.index("--calibration-result") + 1].startswith("{repo_root}/")
    assert commands[0]["checkpoint_relative"].endswith("/checkpoints")


def test_frozen_evidence_profiles_materialize_contractual_real_argv(tmp_path):
    contract = json.loads(DEFAULT_CONTRACT.read_text())
    w12 = _frozen_evidence_profile(contract, "reduced_w12")
    assert w12["weeks"] == 12
    assert w12["split"] == "transducer_collision_suite_burned"
    assert len(w12["seeds"]) == 5
    assert w12["output_path"] == (
        "results/paper2_bottleneck/exact_transducer_certification_w12.json"
    )
    rows = _seed_rows(w12["seeds"], w12["split"], w12["weeks"])
    command = _frozen_evidence_command_row(
        rows,
        profile=w12,
        runner_rel=w12["runner"],
        runner_workers=3,
    )[0]
    argv = _materialize_argv(command["argv_template"], repo_root=ROOT, run_dir=tmp_path)
    parsed = exact_transducer_parser().parse_args(argv[argv.index("--") + 1 :])
    assert parsed.weeks == 12
    assert parsed.seed == w12["seeds"]
    assert parsed.split == w12["split"]
    assert parsed.workers == 3
    assert parsed.progress == tmp_path / command["progress_relative"]
    assert parsed.output == tmp_path / w12["output_path"]
    assert parsed.non_scientific_smoke is False
    assert parsed.max_calendars is None

    w24 = _frozen_evidence_profile(contract, "w24_audit")
    w24_command = _frozen_evidence_command_row(
        _seed_rows(w24["seeds"], w24["split"], w24["weeks"]),
        profile=w24,
        runner_rel=w24["runner"],
        runner_workers=4,
    )[0]
    assert w24_command["output_relative"] == w24["output_path"]
    assert "--write-w24-audit" in w24_command["argv_template"]
    assert "--progress" in w24_command["argv_template"]
    assert w24_command["checkpoint_relative"].endswith("/checkpoints")


def _immutable_git_fixture(_repo_root, _critical_paths):
    return {
        "commit": "a" * 40,
        "branch": "codex/test",
        "source_status": [],
        "critical_inputs_tracked": {},
        "critical_inputs_match_head": True,
        "scientific_source_immutable": True,
    }


def test_frozen_evidence_prepare_is_sealed_and_tampering_fails_closed(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(harness, "git_snapshot", _immutable_git_fixture)
    contract = json.loads(DEFAULT_CONTRACT.read_text())
    profile = _frozen_evidence_profile(contract, "reduced_w16")
    run_dir = tmp_path / "w16"
    manifest = prepare_run(
        run_dir=run_dir,
        run_id="pytest-reduced-w16",
        mode="reduced_w16",
        contract_path=DEFAULT_CONTRACT,
        runner_path=DEFAULT_SMOKE_RUNNER,
        seeds=profile["seeds"],
        split=profile["split"],
        weeks=profile["weeks"],
        runner_workers=2,
        heartbeat_interval=1.0,
    )
    assert manifest["execution"]["sealed_for_execution"] is True
    assert manifest["execution"]["authorization_scope"] == (
        "PRE_GATE_CERTIFICATION_ONLY"
    )
    assert manifest["inputs"]["contract_sha256"] != manifest["inputs"][
        "result_contract_sha256"
    ]
    assert manifest["execution"]["frozen_evidence_profile"] == profile

    manifest["execution"]["frozen_evidence_profile"]["split"] = "tampered"
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(HarnessError, match="profile differs from contract"):
        harness._validate_prepared_inputs(run_dir, ROOT)


def test_remote_environment_snapshot_allows_only_soabi_difference(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(harness, "git_snapshot", _immutable_git_fixture)
    from scripts.run_paper2_bottleneck_exact_transducer import (
        certification_environment,
    )

    remote_environment = certification_environment()
    remote_environment["python_soabi"] = "cpython-311-x86_64-linux-gnu"
    unhashed = {
        key: value
        for key, value in remote_environment.items()
        if key != "environment_sha256"
    }
    remote_environment["environment_sha256"] = harness.canonical_json_sha256(
        unhashed
    )
    environment_path = tmp_path / "remote-environment.json"
    environment_path.write_text(json.dumps(remote_environment))
    profile = _frozen_evidence_profile(
        json.loads(DEFAULT_CONTRACT.read_text()), "reduced_w16"
    )
    run_dir = tmp_path / "w16-remote"
    manifest = prepare_run(
        run_dir=run_dir,
        run_id="pytest-reduced-w16-remote",
        mode="reduced_w16",
        contract_path=DEFAULT_CONTRACT,
        runner_path=DEFAULT_SMOKE_RUNNER,
        seeds=profile["seeds"],
        split=profile["split"],
        weeks=profile["weeks"],
        runner_workers=2,
        heartbeat_interval=1.0,
        scientific_environment_path=environment_path,
    )
    assert manifest["inputs"]["environment"] == remote_environment
    assert manifest["inputs"]["environment_source"] == (
        "provided_remote_preflight"
    )
    harness._validate_prepared_inputs(
        run_dir,
        ROOT,
        allow_preflight_platform_environment=True,
    )
    with pytest.raises(HarnessError, match="environment identity"):
        harness._validate_prepared_inputs(run_dir, ROOT)

    tampered = dict(remote_environment)
    tampered["packages"] = {**tampered["packages"], "numpy": "0.0"}
    with pytest.raises(HarnessError, match="digest mismatch"):
        validate_scientific_environment_payload(tampered)


def test_frozen_evidence_result_binds_harness_commit_runner_and_both_contracts(
    monkeypatch,
):
    import scripts.run_paper2_bottleneck_exact_transducer as exact

    monkeypatch.setattr(exact, "validate_reduced_certification_payload", lambda *_args, **_kwargs: [])
    manifest = {
        "mode": "reduced_w12",
        "git": {"commit": "a" * 40},
        "inputs": {
            "environment_sha256": "e" * 64,
            "runner_sha256": "r" * 64,
            "contract_sha256": "p" * 64,
            "result_contract_sha256": "f" * 64,
        },
    }
    payload = {
        "provenance": {"git_commit": "a" * 40, "producer_sha256": "r" * 64},
        "contract_sha256": "f" * 64,
        "scientific_run": True,
    }
    assert _validate_frozen_evidence_result(payload, manifest) == []
    payload["provenance"]["git_commit"] = "b" * 40
    assert any(
        "source commit" in failure
        for failure in _validate_frozen_evidence_result(payload, manifest)
    )

    import scripts.run_paper2_bottleneck_full_frontier as frontier

    monkeypatch.setattr(frontier, "validate_w24_profile_state_audit_payload", lambda *_args, **_kwargs: [])
    manifest["mode"] = "w24_audit"
    manifest["inputs"]["result_contract_sha256"] = "p" * 64
    w24 = {
        "git_head": "a" * 40,
        "generated_by_frontier_runner_sha256": "r" * 64,
        "primary_contract_sha256": "p" * 64,
    }
    assert _validate_frozen_evidence_result(w24, manifest) == []
    w24["generated_by_frontier_runner_sha256"] = "x" * 64
    assert any(
        "producer" in failure
        for failure in _validate_frozen_evidence_result(w24, manifest)
    )


def test_seal_adds_untracked_control_authorization_without_regenerating_manifests(
    tmp_path, monkeypatch
):
    run_dir = tmp_path / "prepared"
    (run_dir / "status").mkdir(parents=True)
    seed_manifest = {"run_id": "seal-test"}
    command_manifest = {"run_id": "seal-test"}
    manifest = {
        "mode": "scientific",
        "inputs": {},
        "execution": {"sealed_for_execution": False},
        "evidence_status": "AWAITING_AUTHORIZATION_NOT_EXECUTABLE_NOT_EVIDENCE",
    }
    (run_dir / "seed_manifest.json").write_text(json.dumps(seed_manifest))
    (run_dir / "command_manifest.json").write_text(json.dumps(command_manifest))
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest))
    (run_dir / "status" / "run_status.json").write_text(
        json.dumps({"state": "awaiting_authorization"})
    )
    (run_dir / "status" / "heartbeat.json").write_text(
        json.dumps({"state": "awaiting_authorization"})
    )
    authorization = tmp_path / "control-authorization.json"
    authorization.write_text(json.dumps({"authorization_scope": "primary_bound_only"}))
    monkeypatch.setattr(
        harness,
        "_validate_prepared_inputs",
        lambda *_args, **_kwargs: (manifest, seed_manifest, command_manifest),
    )
    monkeypatch.setattr(harness, "_validate_authorization", lambda *_args, **_kwargs: None)
    sealed = seal_run(
        run_dir=run_dir,
        authorization_path=authorization,
        repo_root=ROOT,
    )
    assert sealed["execution"]["sealed_for_execution"] is True
    assert (run_dir / "authorization.json").read_bytes() == authorization.read_bytes()
    assert sealed["inputs"]["authorization_sha256"]
    assert json.loads((run_dir / "seed_manifest.json").read_text()) == seed_manifest
    assert json.loads((run_dir / "command_manifest.json").read_text()) == command_manifest


def test_authorization_template_separates_reduced_and_w24_primary_gates():
    manifest = {
        "git": {"commit": "a" * 40},
        "inputs": {
            "contract_sha256": "c" * 64,
            "runner_sha256": "r" * 64,
            "harness_sha256": "h" * 64,
            "isolated_bootstrap_sha256": "b" * 64,
            "host_runtime_sha256": "t" * 64,
            "portable_runtime_sha256": "p" * 64,
            "scientific_child_environment_sha256": "d" * 64,
            "execution_nonce": "n" * 64,
            "calibration_result_sha256": None,
            "environment_sha256": "e" * 64,
        },
    }
    template = _authorization_template(
        manifest,
        seed_manifest_sha256="s" * 64,
        command_manifest_sha256="m" * 64,
    )
    assert template["reduced_horizon_key_v4_certified"] is False
    assert template["full_horizon_primary_acceleration_authorized"] is False
    for row in template["reduced_horizon_certification_artifacts"]:
        pair = row["independent_execution_verification"]
        assert pair["verification_mode"] == harness.PORTABLE_PAIR_VERIFICATION_MODE
        assert "producer_manifest_sha256" in pair
        assert "independent_manifest_sha256" in pair
    assert template["w24_profile_state_audit"] == {
        "path": "results/paper2_bottleneck/w24_profile_state_audit.json",
        "sha256": "FILL_AFTER_TRACKED_AUDIT_VERIFICATION",
        "source_git_commit": "FILL_FROM_AUDIT_GIT_HEAD",
    }


def _native_frontier_validation_fixture(phase="calibration"):
    count = 60 if phase == "calibration" else 119
    seed_start = 1_100_001 if phase == "calibration" else 1_110_002
    context_offset = 0 if phase == "calibration" else 1
    certificate_rows = [
        {
            "index": index,
            "seed": seed_start + index,
            "tape_sha256": harness._expected_tape_sha256(
                seed_start + index,
                harness.CONTEXTS[
                    (index + context_offset) % len(harness.CONTEXTS)
                ],
                phase,
                24,
            ),
            "collision_count": 0,
            "collision_root_count": 0,
            "node_obligation_count": 1,
            "certificate_sha256": f"{index + 101:064x}"[-64:],
            "complete": True,
        }
        for index in range(count)
    ]
    transducers = [
        {
            "seed": row["seed"],
            "tape_sha256": row["tape_sha256"],
            "collision_certificate_sha256": row["certificate_sha256"],
            "score_table_sha256": f"{index + 201:064x}"[-64:],
        }
        for index, row in enumerate(certificate_rows)
    ]
    checkpoints = [
        {
            "index": index,
            "seed": row["seed"],
            "context": harness.CONTEXTS[
                (index + context_offset) % len(harness.CONTEXTS)
            ],
            "split": phase,
            "weeks": 24,
            "tape_sha256": row["tape_sha256"],
            "score_table_sha256": row["score_table_sha256"],
            "source_transducer_proof": {
                "collision_bisimulation": {
                    "certificate_sha256": row[
                        "collision_certificate_sha256"
                    ]
                }
            },
            "environment": {"environment_sha256": "e" * 64},
        }
        for index, row in enumerate(transducers)
    ]
    coverage = {
        "schema_version": "paper2_collision_certificate_coverage_v1",
        "required_count": count,
        "complete_count": count,
        "unique_identity_count": count,
        "rows": certificate_rows,
        "rows_sha256": harness.canonical_json_sha256(certificate_rows),
        "failures": [],
        "passed": True,
    }
    coverage["coverage_sha256"] = harness.canonical_json_sha256(coverage)
    manifest = {
        "inputs": {
            "contract_sha256": "c" * 64,
            "runner_sha256": "r" * 64,
            "runner_relative": "scripts/run_paper2_bottleneck_full_frontier.py",
            "authorization_sha256": "a" * 64,
            "environment_sha256": "e" * 64,
        },
        "execution": {
            "phase": phase,
            "authorization_scope": "primary_bound_only",
        },
    }
    payload = {
        "schema_version": "paper2_bottleneck_full_frontier_v2",
        "phase": phase,
        "weeks": 24,
        "primary_contract_sha256": "c" * 64,
        "phase_execution_complete": True,
        "exact_maximum_certified": True,
        "selected_replay_complete": True,
        "selected_replay_audit": {"passed": True},
        "resolved_frontier": {"exact_maximum_certified": True},
        "calendar_index": {"calendar_count": 11_184_811},
        "screening": {
            "pass1_count": 11_184_811,
            "pass2_count": 11_184_811,
            "passes_identical": True,
            "contender_overflow": {"aggregate": False, "per_tape": []},
        },
        "acceleration_authorization": {
            "key_schema_version": "paper2_bottleneck_semantic_markov_key_v4",
            "authorized": True,
                "sha256": "a" * 64,
                "environment_sha256": "e" * 64,
            },
            "build": {
                "checkpoints": checkpoints,
                "collision_certificate_coverage": coverage,
            },
        "transducers": transducers,
        "selected_replays": [
            {
                "seed": certificate_rows[0]["seed"],
                "tape_sha256": certificate_rows[0]["tape_sha256"],
                "guardrails": {
                    "token_hours_m": 4032.0,
                    "token_hours_t": 0.0,
                    "token_hours_r": 0.0,
                    "total_token_hours": 4032.0,
                        "mass_residual": 0.0,
                        "reserve_inventory_initial": 10_000.0,
                        "reserve_capacity": 10_000.0,
                        "reserve_target_terminal": 10_000.0,
                        "reserve_replenishment_lead_time": 168.0,
                        "reserve_issue_delay": 24.0,
                        "reserve_stock_balance_residual": 0.0,
                        "consumed_base_threat_sha256": "t" * 64,
                        "realized_demand_sha256": "d" * 64,
                }
            }
        ],
        "paper3_authorized": False,
    }
    runner_manifest = {
        "schema_version": "paper2_bottleneck_full_frontier_manifest_v2",
        "phase_execution_complete": True,
        "full_execution_was_explicitly_invoked": True,
        "input_artifacts": {
            "primary_contract": {"sha256": "c" * 64},
            "authorization": {"sha256": "a" * 64},
        },
        "code_sha256": {
            "scripts/run_paper2_bottleneck_full_frontier.py": "r" * 64
        },
        "key_schema_version": "paper2_bottleneck_semantic_markov_key_v4",
        "exact_maximum_certified": True,
        "environment_sha256": "e" * 64,
        "collision_certificate_coverage_sha256": coverage["coverage_sha256"],
        "seed_manifest": [
            {
                "seed": row["seed"],
                "context_0": harness.CONTEXTS[
                    (index + context_offset) % len(harness.CONTEXTS)
                ],
                "tape_sha256": row["tape_sha256"],
                "split": phase,
            }
            for index, row in enumerate(certificate_rows)
        ],
        "checkpoint_artifacts": [
            {
                "seed": row["seed"],
                "collision_certificate_sha256": row[
                    "collision_certificate_sha256"
                ],
                "score_table_sha256": row["score_table_sha256"],
            }
            for row in transducers
        ],
    }
    seed_manifest = {
        "seeds": [
            {
                "seed": row["seed"],
                "context": harness.CONTEXTS[
                    (index + context_offset) % len(harness.CONTEXTS)
                ],
                "split": phase,
                "weeks": 24,
                "expected_tape_sha256": row["tape_sha256"],
            }
            for index, row in enumerate(certificate_rows)
        ]
    }
    return payload, manifest, runner_manifest, seed_manifest


def _native_scientific_verification_fixture(tmp_path: Path) -> Path:
    payload, manifest_stub, runner_manifest, seed_stub = copy.deepcopy(
        _native_frontier_validation_fixture("calibration")
    )
    run_dir = tmp_path / "native-scientific-verification"
    run_id = "native-scientific-verification"
    item_id = "frontier_calibration"
    output_relative = f"artifacts/{item_id}/result.json"
    runner_manifest_relative = f"artifacts/{item_id}/runner_manifest.json"
    stdout_relative = f"logs/{item_id}.stdout.log"
    stderr_relative = f"logs/{item_id}.stderr.log"
    seed_rows = seed_stub["seeds"]
    seed_ids = [
        f"seed_{row['seed']}_{row['context']}" for row in seed_rows
    ]
    seed_manifest = {
        "schema_version": harness.SCHEMA,
        "run_id": run_id,
        "mode": "scientific",
        "seed_count": len(seed_rows),
        "seeds": seed_rows,
    }
    argv_template = [
        "{python}",
        "{repo_root}/scripts/run_paper2_bottleneck_full_frontier.py",
        "--phase",
        "calibration",
        "--output",
        f"{{run_dir}}/{output_relative}",
        "--manifest",
        f"{{run_dir}}/{runner_manifest_relative}",
    ]
    materialized_argv = [
        "/runtime/python",
        "/repository/scripts/run_paper2_bottleneck_full_frontier.py",
        "--phase",
        "calibration",
        "--output",
        f"/run/{output_relative}",
        "--manifest",
        f"/run/{runner_manifest_relative}",
    ]
    materialization_context = {
        "python_executable": "/runtime/python",
        "repository_root": "/repository",
        "run_directory": "/run",
        "host_runtime_sha256": _fake_runtime_attestation()["runtime_sha256"],
        "execution_nonce": "9" * 64,
    }
    command_sha256 = harness.canonical_json_sha256(materialized_argv)
    command_manifest = {
        "schema_version": harness.SCHEMA,
        "run_id": run_id,
        "mode": "scientific",
        "commands": [
            {
                "job_id": item_id,
                "phase": "calibration",
                "covered_seed_ids": seed_ids,
                "argv_template": argv_template,
                "argv_template_sha256": harness.canonical_json_sha256(
                    argv_template
                ),
                "output_relative": output_relative,
                "runner_manifest_relative": runner_manifest_relative,
                "stdout_relative": stdout_relative,
                "stderr_relative": stderr_relative,
            }
        ],
    }
    harness.atomic_write_json(run_dir / "seed_manifest.json", seed_manifest)
    harness.atomic_write_json(run_dir / "command_manifest.json", command_manifest)
    harness.atomic_write_json(
        run_dir / "authorization.json", {"fixture_authorization": True}
    )
    authorization_sha256 = harness.sha256_file(run_dir / "authorization.json")
    manifest_stub["inputs"]["authorization_sha256"] = authorization_sha256
    payload["acceleration_authorization"]["sha256"] = authorization_sha256
    runner_manifest["input_artifacts"]["authorization"][
        "sha256"
    ] = authorization_sha256
    (run_dir / "environment").mkdir(parents=True, exist_ok=True)
    (run_dir / "environment/pip_freeze.txt").write_text("pytest==1\n")
    harness.atomic_write_json(
        run_dir / "environment/machine.json", {"fixture": True}
    )
    runtime = _write_fake_runtime(run_dir)
    child_environment = scientific_child_environment()
    inputs = {
        **manifest_stub["inputs"],
        "harness_sha256": "h" * 64,
        "isolated_bootstrap_sha256": "7" * 64,
        "result_contract_sha256": None,
        "host_runtime_attestation_relative": "environment/scientific_runtime.json",
        "host_runtime_sha256": runtime["runtime_sha256"],
        "portable_runtime_sha256": runtime["portable_sha256"],
        "scientific_child_environment": child_environment,
        "scientific_child_environment_sha256": harness.canonical_json_sha256(
            child_environment
        ),
        "execution_nonce": "9" * 64,
        "dependency_snapshot_sha256": harness.sha256_file(
            run_dir / "environment/pip_freeze.txt"
        ),
        "machine_snapshot_sha256": harness.sha256_file(
            run_dir / "environment/machine.json"
        ),
        "seed_manifest_sha256": harness.sha256_file(
            run_dir / "seed_manifest.json"
        ),
        "command_manifest_sha256": harness.sha256_file(
            run_dir / "command_manifest.json"
        ),
    }
    run_manifest = {
        "schema_version": harness.SCHEMA,
        "run_id": run_id,
        "mode": "scientific",
        "git": {"commit": "2" * 40},
        "inputs": inputs,
        "execution": manifest_stub["execution"],
    }
    harness.atomic_write_json(run_dir / "run_manifest.json", run_manifest)
    harness.atomic_write_json(run_dir / output_relative, payload)
    output_sha256 = harness.sha256_file(run_dir / output_relative)
    runner_manifest["result_sha256"] = output_sha256
    harness.atomic_write_json(run_dir / runner_manifest_relative, runner_manifest)
    (run_dir / stdout_relative).parent.mkdir(parents=True, exist_ok=True)
    (run_dir / stdout_relative).write_text("completed\n")
    (run_dir / stderr_relative).write_text("")

    job_status = {
        "schema_version": harness.SCHEMA,
        "run_id": run_id,
        "item_id": item_id,
        "seed": None,
        "context": None,
        "phase": "calibration",
        "state": "completed",
        "returncode": 0,
        "command_sha256": command_sha256,
        "output_relative": output_relative,
        "output_valid_json": True,
        "output_sha256": output_sha256,
        "output_error": None,
        "stdout_relative": stdout_relative,
        "stderr_relative": stderr_relative,
        "evidence": False,
    }
    execution_receipt = {
        "schema_version": harness.SCHEMA,
        "run_id": run_id,
        "job_id": item_id,
        "mode": "scientific",
        "git_commit": run_manifest["git"]["commit"],
        "contract_sha256": inputs["contract_sha256"],
        "result_contract_sha256": None,
        "runner_sha256": inputs["runner_sha256"],
        "harness_sha256": inputs["harness_sha256"],
        "environment_sha256": inputs["environment_sha256"],
        "scientific_child_environment_sha256": inputs[
            "scientific_child_environment_sha256"
        ],
        "execution_nonce": inputs["execution_nonce"],
        "execution_role": "frontier_calibration",
        "isolated_bootstrap_sha256": inputs["isolated_bootstrap_sha256"],
        "host_runtime_sha256": inputs["host_runtime_sha256"],
        "portable_runtime_sha256": inputs["portable_runtime_sha256"],
        "seed_manifest_sha256": inputs["seed_manifest_sha256"],
        "command_manifest_sha256": inputs["command_manifest_sha256"],
        "materialization_context": materialization_context,
        "materialized_argv": materialized_argv,
        "materialized_command_sha256": command_sha256,
        "output_relative": output_relative,
        "output_sha256": output_sha256,
        "validated_not_independently_audited": True,
        "evidence": False,
    }
    evidence_status = "COMPLETED_HASHED_AUDIT_PENDING_NOT_EVIDENCE"
    completion_receipt = {
        "schema_version": harness.SCHEMA,
        "run_id": run_id,
        "state": "completed",
        "location": "local",
        "seed_count": len(seed_rows),
        "completed_seed_count": len(seed_rows),
        "failed_seed_count": 0,
        "run_manifest_sha256": harness.sha256_file(run_dir / "run_manifest.json"),
        "seed_manifest_sha256": harness.sha256_file(run_dir / "seed_manifest.json"),
        "command_manifest_sha256": harness.sha256_file(
            run_dir / "command_manifest.json"
        ),
        "immutable_status_snapshot": True,
        "evidence": False,
        "evidence_status": evidence_status,
    }
    harness.atomic_write_json(
        run_dir / "status" / "jobs" / f"{item_id}.json", job_status
    )
    harness.atomic_write_json(
        run_dir
        / "status"
        / "jobs"
        / f"{item_id}.execution_receipt.json",
        execution_receipt,
    )
    seed_status_relatives = []
    for seed_id, row in zip(seed_ids, seed_rows):
        relative = f"status/seeds/{seed_id}.json"
        seed_status_relatives.append(relative)
        harness.atomic_write_json(
            run_dir / relative,
            {
                "schema_version": harness.SCHEMA,
                "run_id": run_id,
                "seed_id": seed_id,
                "seed": row["seed"],
                "context": row["context"],
                "phase": row["split"],
                "state": "completed",
                "output_sha256": output_sha256,
                "returncode": 0,
                "evidence": False,
            },
        )
    harness.atomic_write_json(
        run_dir / "status" / "run_completion_receipt.json", completion_receipt
    )
    protected = [
        "run_manifest.json",
        "seed_manifest.json",
        "command_manifest.json",
        "authorization.json",
        "environment/pip_freeze.txt",
        "environment/machine.json",
        "environment/scientific_runtime.json",
        output_relative,
        runner_manifest_relative,
        stdout_relative,
        stderr_relative,
        f"status/jobs/{item_id}.json",
        f"status/jobs/{item_id}.execution_receipt.json",
        *seed_status_relatives,
        "status/run_completion_receipt.json",
    ]
    checksums = {
        "schema_version": harness.SCHEMA,
        "run_id": run_id,
        "records": harness._checksum_records(run_dir, protected),
        "evidence": False,
    }
    checksums["record_count"] = len(checksums["records"])
    harness.atomic_write_json(run_dir / "artifact_checksums.json", checksums)
    harness.atomic_write_json(
        run_dir / "status" / "run_status.json",
        {
            "schema_version": harness.SCHEMA,
            "run_id": run_id,
            "state": "completed",
            "location": "local",
            "evidence": False,
            "evidence_status": evidence_status,
            "seed_count": len(seed_rows),
            "completed_seed_count": len(seed_rows),
            "failed_seed_count": 0,
            "checksums_sha256": harness.sha256_file(
                run_dir / "artifact_checksums.json"
            ),
            "independent_scientific_audit_required": True,
        },
    )
    return run_dir


@pytest.mark.parametrize("phase", ["calibration", "locked"])
def test_finalized_native_frontier_result_schema_passes_primary_only_validation(
    phase,
):
    payload, manifest, runner_manifest, seed_manifest = (
        _native_frontier_validation_fixture(phase)
    )
    assert (
        _validate_scientific_result(
            payload, manifest, runner_manifest, seed_manifest
        )
        == []
    )


def test_native_schema_cannot_be_bypassed_with_injected_execution_assurance():
    payload, manifest, runner_manifest, seed_manifest = copy.deepcopy(
        _native_frontier_validation_fixture()
    )
    payload["execution_assurance"] = {
        "key_schema_version": "paper2_bottleneck_semantic_markov_key_v4",
        "primary_frontier_exact": True,
        "numeric_approximation_gap": 0,
        "original_runner_replay_passed": True,
        "resource_semantics_passed": True,
        "contract_sha256": manifest["inputs"]["contract_sha256"],
        "runner_sha256": manifest["inputs"]["runner_sha256"],
    }

    failures = _validate_scientific_result(
        payload, manifest, runner_manifest, seed_manifest
    )
    assert "native frontier payload mixes a foreign execution assurance" in failures

    foreign = {
        "schema_version": "foreign_frontier_v1",
        "execution_assurance": payload["execution_assurance"],
    }
    failures = _validate_scientific_result(foreign, manifest)
    assert "execution-assurance payload has an unsupported foreign schema" in failures


def test_verify_artifacts_revalidates_coordinated_rehashed_scientific_output(
    tmp_path,
):
    run_dir = _native_scientific_verification_fixture(tmp_path)
    assert verify_artifacts(run_dir, retrieved=False)["checks_passed"] is True

    output_path = run_dir / "artifacts/frontier_calibration/result.json"
    payload = json.loads(output_path.read_text())
    payload["transducers"] = payload["transducers"][:-1]
    harness.atomic_write_json(output_path, payload)
    output_sha256 = harness.sha256_file(output_path)
    runner_manifest_path = (
        run_dir / "artifacts/frontier_calibration/runner_manifest.json"
    )
    runner_manifest = json.loads(runner_manifest_path.read_text())
    runner_manifest["result_sha256"] = output_sha256
    harness.atomic_write_json(runner_manifest_path, runner_manifest)
    for path in [
        run_dir / "status/jobs/frontier_calibration.json",
        run_dir
        / "status/jobs/frontier_calibration.execution_receipt.json",
        *(run_dir / "status/seeds").glob("*.json"),
    ]:
        document = json.loads(path.read_text())
        document["output_sha256"] = output_sha256
        harness.atomic_write_json(path, document)
    _rehash_verification_fixture(run_dir)

    verification = verify_artifacts(run_dir, retrieved=False)
    assert verification["checks_passed"] is False
    assert any(
        "scientific result revalidation failed: frontier ordered identity source has wrong count: result transducers"
        == failure
        for failure in verification["failures"]
    )


def test_verify_artifacts_recomputes_coordinated_materialized_command_hash(
    tmp_path,
):
    run_dir = _grouped_verification_fixture(tmp_path)
    assert verify_artifacts(run_dir, retrieved=False)["checks_passed"] is True
    forged_sha256 = "f" * 64
    for path in [
        run_dir / "status/jobs/frontier_calibration.json",
        run_dir
        / "status/jobs/frontier_calibration.execution_receipt.json",
    ]:
        document = json.loads(path.read_text())
        key = (
            "command_sha256"
            if path.name == "frontier_calibration.json"
            else "materialized_command_sha256"
        )
        document[key] = forged_sha256
        harness.atomic_write_json(path, document)
    _rehash_verification_fixture(run_dir)

    verification = verify_artifacts(run_dir, retrieved=False)
    assert verification["checks_passed"] is False
    assert (
        "command/job status command hash is not recomputable: frontier_calibration"
        in verification["failures"]
    )


def test_verify_artifacts_rejects_coordinated_argv_and_hash_replacement(
    tmp_path,
):
    run_dir = _grouped_verification_fixture(tmp_path)
    assert verify_artifacts(run_dir, retrieved=False)["checks_passed"] is True
    receipt_path = (
        run_dir
        / "status/jobs/frontier_calibration.execution_receipt.json"
    )
    receipt = json.loads(receipt_path.read_text())
    forged_argv = list(receipt["materialized_argv"])
    forged_argv[1] = "/repository/forged_runner.py"
    forged_sha256 = harness.canonical_json_sha256(forged_argv)
    receipt["materialized_argv"] = forged_argv
    receipt["materialized_command_sha256"] = forged_sha256
    harness.atomic_write_json(receipt_path, receipt)
    job_status_path = run_dir / "status/jobs/frontier_calibration.json"
    job_status = json.loads(job_status_path.read_text())
    job_status["command_sha256"] = forged_sha256
    harness.atomic_write_json(job_status_path, job_status)
    _rehash_verification_fixture(run_dir)

    verification = verify_artifacts(run_dir, retrieved=False)
    assert verification["checks_passed"] is False
    assert (
        "execution receipt argv does not match command template: frontier_calibration"
        in verification["failures"]
    )


@pytest.mark.parametrize(
    ("tamper", "expected_failure"),
    [
        ("coverage_order", "ordered tape/certificate identities mismatch: coverage"),
        ("coverage_tape", "ordered tape/certificate identities mismatch: coverage"),
        (
            "runner_certificate",
            "ordered certificate identities mismatch: runner checkpoints",
        ),
        (
            "transducer_certificate",
            "ordered tape/certificate identities mismatch: checkpoints",
        ),
        ("transducer_order", "ordered tape identities mismatch: transducers"),
        (
            "transducers_short",
            "ordered identity source has wrong count: result transducers",
        ),
        (
            "transducers_missing",
            "ordered identity source has wrong count: result transducers",
        ),
        (
            "runner_seeds_short",
            "ordered identity source has wrong count: runner seed manifest",
        ),
        (
            "runner_seeds_missing",
            "ordered identity source has wrong count: runner seed manifest",
        ),
        ("harness_order", "harness/runner seed identity mismatch"),
        ("checkpoint_index", "checkpoint scope identity mismatch"),
        ("checkpoint_context", "checkpoint scope identity mismatch"),
        ("checkpoint_split", "checkpoint scope identity mismatch"),
        ("checkpoint_weeks", "checkpoint scope identity mismatch"),
    ],
)
def test_native_frontier_ordered_identity_reconciliation_fails_closed(
    tamper, expected_failure
):
    payload, manifest, runner_manifest, seed_manifest = copy.deepcopy(
        _native_frontier_validation_fixture()
    )
    if tamper in {"coverage_order", "coverage_tape"}:
        coverage = payload["build"]["collision_certificate_coverage"]
        if tamper == "coverage_order":
            coverage["rows"] = coverage["rows"][1:] + coverage["rows"][:1]
            for index, row in enumerate(coverage["rows"]):
                row["index"] = index
        else:
            coverage["rows"][0]["tape_sha256"] = "f" * 64
        coverage["rows_sha256"] = harness.canonical_json_sha256(coverage["rows"])
        coverage.pop("coverage_sha256")
        coverage["coverage_sha256"] = harness.canonical_json_sha256(coverage)
        runner_manifest["collision_certificate_coverage_sha256"] = coverage[
            "coverage_sha256"
        ]
    elif tamper == "runner_certificate":
        rows = runner_manifest["checkpoint_artifacts"]
        rows[0]["collision_certificate_sha256"], rows[1][
            "collision_certificate_sha256"
        ] = (
            rows[1]["collision_certificate_sha256"],
            rows[0]["collision_certificate_sha256"],
        )
    elif tamper == "transducer_order":
        payload["transducers"][0], payload["transducers"][1] = (
            payload["transducers"][1],
            payload["transducers"][0],
        )
    elif tamper == "transducer_certificate":
        payload["transducers"][0]["collision_certificate_sha256"] = "f" * 64
    elif tamper == "transducers_short":
        payload["transducers"].pop()
    elif tamper == "transducers_missing":
        payload.pop("transducers")
    elif tamper == "runner_seeds_short":
        runner_manifest["seed_manifest"].pop()
    elif tamper == "runner_seeds_missing":
        runner_manifest.pop("seed_manifest")
    elif tamper == "harness_order":
        seed_manifest["seeds"][0], seed_manifest["seeds"][1] = (
            seed_manifest["seeds"][1],
            seed_manifest["seeds"][0],
        )
    else:
        field = tamper.removeprefix("checkpoint_")
        payload["build"]["checkpoints"][0][field] = {
            "index": 1,
            "context": "interdiction_campaign",
            "split": "locked",
            "weeks": 23,
        }[field]

    failures = _validate_scientific_result(
        payload, manifest, runner_manifest, seed_manifest
    )
    assert any(expected_failure in failure for failure in failures)
