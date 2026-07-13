import json
from pathlib import Path
import subprocess
import sys

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
    execute_run,
    launch_vps,
    prepare_run,
    seal_run,
    verify_artifacts,
)
from scripts.run_paper2_bottleneck_exact_transducer import (
    build_parser as exact_transducer_parser,
)


ROOT = Path(__file__).resolve().parent.parent


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
    assert seeds["seed_count"] == 1
    assert commands["shell_execution"] is False
    assert commands["commands"][0]["seed"] == 1_110_001
    assert "--non-scientific-smoke" in commands["commands"][0]["argv_template"]
    assert "--max-calendars" in commands["commands"][0]["argv_template"]
    assert status["state"] == "prepared"
    assert manifest["evidence"] is False
    assert "NOT_EVIDENCE" in manifest["evidence_status"]
    assert (run_dir / "environment" / "pip_freeze.txt").is_file()
    assert (run_dir / "environment" / "machine.json").is_file()


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
    verification = verify_artifacts(run_dir, retrieved=True)
    assert verification["checks_passed"] is False
    assert any("checksum mismatch" in reason for reason in verification["failures"])
    assert verification["evidence_status"] == "VERIFICATION_FAILED_NOT_EVIDENCE"


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


def test_primary_bound_scope_requires_exact_primary_replay_but_not_full_guardrail():
    manifest = {
        "inputs": {"contract_sha256": "c" * 64, "runner_sha256": "r" * 64},
        "execution": {"authorization_scope": "primary_bound_only"},
    }
    payload = {
        "execution_assurance": {
            "key_schema_version": "paper2_bottleneck_semantic_markov_key_v2",
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
    parsed = exact_transducer_parser().parse_args(argv[2:])
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
            "calibration_result_sha256": None,
            "environment_sha256": "e" * 64,
        },
    }
    template = _authorization_template(
        manifest,
        seed_manifest_sha256="s" * 64,
        command_manifest_sha256="m" * 64,
    )
    assert template["reduced_horizon_key_v2_certified"] is False
    assert template["full_horizon_primary_acceleration_authorized"] is False
    assert template["w24_profile_state_audit"] == {
        "path": "results/paper2_bottleneck/w24_profile_state_audit.json",
        "sha256": "FILL_AFTER_TRACKED_AUDIT_VERIFICATION",
        "source_git_commit": "FILL_FROM_AUDIT_GIT_HEAD",
    }


def test_finalized_native_frontier_result_schema_passes_primary_only_validation():
    manifest = {
        "inputs": {
            "contract_sha256": "c" * 64,
            "runner_sha256": "r" * 64,
            "runner_relative": "scripts/run_paper2_bottleneck_full_frontier.py",
            "authorization_sha256": "a" * 64,
            "environment_sha256": "e" * 64,
        },
        "execution": {
            "phase": "calibration",
            "authorization_scope": "primary_bound_only",
        },
    }
    payload = {
        "schema_version": "paper2_bottleneck_full_frontier_v1",
        "phase": "calibration",
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
            "key_schema_version": "paper2_bottleneck_semantic_markov_key_v2",
            "authorized": True,
                "sha256": "a" * 64,
                "environment_sha256": "e" * 64,
            },
            "build": {"checkpoints": [{"environment": {"environment_sha256": "e" * 64}}]},
        "selected_replays": [
            {
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
        "input_artifacts": {
            "primary_contract": {"sha256": "c" * 64},
            "authorization": {"sha256": "a" * 64},
        },
        "code_sha256": {
            "scripts/run_paper2_bottleneck_full_frontier.py": "r" * 64
        },
        "key_schema_version": "paper2_bottleneck_semantic_markov_key_v2",
        "exact_maximum_certified": True,
        "environment_sha256": "e" * 64,
    }
    assert _validate_scientific_result(payload, manifest, runner_manifest) == []
