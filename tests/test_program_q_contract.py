from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.adjudicate_program_q import CELL_IDS, adjudicate
from scripts.audit_program_q_full_des import compare_promoted_sources
from scripts.audit_program_q_power_preopen import expected_selected_N
from scripts.audit_program_q_seed_custody import scan
from scripts.benchmark_program_q_latency import benchmark_callable
from scripts.evaluate_program_q_replication import (
    simultaneous_guardrail_inference,
    simultaneous_primary_inference,
    validate_shard,
    verify_authorization,
    write_plan,
)
from scripts.launch_program_q_confirmation import archive_previous_attempt
from scripts.run_program_q_confirmation import verify_pre_reduction_shard_manifest
from scripts.power_program_q_replication import (
    _load_classical_shard,
    _write_classical_shard,
    bootstrap_effects,
    point_effects,
)
from supply_chain.program_o_full_des_transducer import MATRIX_KEYS


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads((ROOT / "contracts/program_q_frozen_policy_replication_v1.json").read_text())
FALLBACK = (
    ROOT
    / "research/paper2_exhaustive_search/program_q_historical_recurrentppo_fallback_freeze_20260717.json"
)
APPROXIMATE_POWER = ROOT / "research/paper2_exhaustive_search/program_q_power_20260718.json"


def test_historical_fallback_is_hash_frozen_without_retraining() -> None:
    payload = json.loads(FALLBACK.read_text())
    assert payload["status"] == "FROZEN_PRIMARY_CANDIDATE_NO_EXTERNAL_DEPENDENCY"
    assert payload["training"]["retraining_for_program_q"] is False
    assert payload["training"]["checkpoint_selection"] == "final only"
    assert len(payload["checkpoints_sha256"]) == 10
    assert all(len(value) == 64 for value in payload["checkpoints_sha256"].values())
    assert payload["scientific_seed_state"]["7490001_7490256"] == "UNOPENED"
    assert payload["selection_rule"]["external_challenger_can_replace_primary"] is False
    assert CONTRACT["primary_candidate"]["replacement_by_external_challenger"] == "forbidden"
    assert CONTRACT["architecture_sidecar_firewall"]["may_delay_or_replace_program_q"] is False


def result_fixture(*, h_lcb=0.02, delta_lcb=-0.005, delta_ucb=0.005):
    estimates = {}
    summaries = {}
    for cell in CELL_IDS:
        estimates[f"{cell}::H_OL"] = {"lcb95": h_lcb, "point": 0.05}
        estimates[f"{cell}::Delta_N"] = {
            "lcb95": delta_lcb,
            "ucb95": delta_ucb,
            "point": 0.0,
        }
        summaries[cell] = {
            "favorable_tapes_fraction_vs_open_loop": 0.8,
            "positive_learner_seeds_H_OL": 9,
        }
    return {
        "schema_version": "program_q_frozen_policy_replication_evaluation_v1",
        "N": 256,
        "seed_range": [7490001, 7490256],
        "shard_count": 768,
        "bootstrap_resamples": 10000,
        "inference": {
            "method": "two-way learner-seed/tape studentized max-t",
            "comparator_reselection_inside_every_resample": True,
            "H_OL_one_sided_and_Delta_N_two_sided": True,
            "estimates": estimates,
        },
        "cell_summaries": summaries,
        "integrity_gates": {
            "feedback": True,
            "replacement_controls": True,
            "scheduled_resources_exact": True,
            "mass_partition_demand": True,
            "ret_full_noninferior": True,
            "quantity_ret_full_noninferior": True,
            "worst_product_fill_noninferior": True,
        },
    }


def test_program_q_equivalence_and_premium_are_distinct() -> None:
    direct_audit = {"passed": True}
    equivalent = adjudicate(result_fixture(), CONTRACT, direct_audit)
    assert equivalent["verdict"] == "PASS_Q_LEARNED_ADAPTATION_CLASSICALLY_EQUIVALENT"
    premium = adjudicate(
        result_fixture(delta_lcb=0.011, delta_ucb=0.03), CONTRACT, direct_audit
    )
    assert premium["verdict"] == "PASS_Q_NEURAL_PREMIUM"


def test_program_q_adjudication_fails_closed_on_missing_integrity() -> None:
    result = result_fixture()
    del result["integrity_gates"]["feedback"]
    payload = adjudicate(result, CONTRACT, {"passed": True})
    assert payload["verdict"] == "STOP_Q_NO_REPLICATED_LEARNED_ADAPTATION"
    assert not payload["paper3_authorized"]


def test_program_q_adjudication_requires_direct_full_des_replay() -> None:
    payload = adjudicate(result_fixture(), CONTRACT)
    assert payload["verdict"] == "STOP_Q_NO_REPLICATED_LEARNED_ADAPTATION"
    assert payload["direct_full_des_replay"] is False


def test_program_q_adjudication_rejects_truncated_seed_design() -> None:
    result = result_fixture()
    result["N"] = 128
    result["seed_range"] = [7490001, 7490128]
    result["shard_count"] = 384
    payload = adjudicate(result, CONTRACT, {"passed": True})
    assert payload["verdict"] == "STOP_Q_NO_REPLICATED_LEARNED_ADAPTATION"
    assert payload["design_gates"]["N"] is False


def test_power_bootstrap_reselects_comparator_families() -> None:
    panels = {}
    for cell in CELL_IDS:
        learner = np.full((3, 4), 0.75)
        open_loop = np.asarray([[0.70, 0.72], [0.76, 0.69], [0.70, 0.72], [0.76, 0.69]])
        classical = np.asarray([[0.73, 0.73, 0.73, 0.73], [0.74, 0.70, 0.74, 0.70]])
        panels[cell] = {"learner": learner, "open_loop": open_loop, "classical": classical}
    points = point_effects(panels)
    assert points.shape == (6,)
    draws = bootstrap_effects(
        panels, tape_count=8, replicates=12, rng=np.random.default_rng(11), batch_size=4
    )
    assert draws.shape == (12, 6)
    assert np.isfinite(draws).all()


def test_classical_cache_shards_are_atomic_resumable_and_identity_checked(
    tmp_path: Path,
) -> None:
    indices = list(range(10))
    _write_classical_shard(tmp_path, 0, 7480001, indices)
    shard = tmp_path / f"{CELL_IDS[0]}__tape_7480001.npz"
    assert shard.is_file()
    assert not list(tmp_path.glob("*.tmp"))
    assert (
        _load_classical_shard(shard, expected_cell_index=0, expected_tape_seed=7480001) == indices
    )
    with np.testing.assert_raises_regex(RuntimeError, "identity mismatch"):
        _load_classical_shard(shard, expected_cell_index=0, expected_tape_seed=7480002)


def test_seed_custody_scan_allows_contract_declaration_only(tmp_path: Path) -> None:
    (tmp_path / "contracts").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "contracts/program_q_frozen_policy_replication_v1.json").write_text(
        '{"reserved": [7490001, 7490256]}'
    )
    (tmp_path / "scripts/audit_program_q_seed_custody.py").write_text("7_490_001")
    assert scan(tmp_path)["pass"]
    (tmp_path / "results").mkdir()
    (tmp_path / "results/result.json").write_text('{"seed": 7490001}')
    payload = scan(tmp_path)
    assert not payload["pass"]
    assert payload["status"] == "STOP_PROGRAM_Q_SEED_COLLISION"


def test_seed_custody_scan_detects_reserved_seed_in_binary_artifact_filename(
    tmp_path: Path,
) -> None:
    (tmp_path / "results/run").mkdir(parents=True)
    artifact = tmp_path / "results/run/tape_7490001.npz"
    artifact.write_bytes(b"binary")
    payload = scan(tmp_path)
    assert payload["pass"] is False
    assert payload["suspicious"][0]["seed_in_filename"] is True


def test_historical_program_q_preopen_scanner_detects_terminal_opening() -> None:
    payload = scan(ROOT)
    assert payload["pass"] is False
    assert payload["status"] == "STOP_PROGRAM_Q_SEED_COLLISION"
    assert any("confirmation_v1_20260718" in row["path"] for row in payload["suspicious"])


def test_early_power_approximation_cannot_select_program_q_N() -> None:
    payload = json.loads(APPROXIMATE_POWER.read_text())
    assert payload["status"] == "NONAUTHORITATIVE_APPROXIMATION"
    assert payload["program_q_N_authority"] is False
    assert payload["selected_N"] is None
    assert payload["verdict"] == "NO_CONTRACTUAL_VERDICT"
    assert CONTRACT["power"]["script"] == "scripts/power_program_q_replication.py"


def test_authoritative_power_selects_the_minimum_jointly_powered_N() -> None:
    rows = {
        "128": {"H_OL": 1.0, "Delta_N_equivalence": 0.81, "joint": 0.79},
        "160": {"H_OL": 1.0, "Delta_N_equivalence": 0.88, "joint": 0.82},
        "192": {"H_OL": 1.0, "Delta_N_equivalence": 0.91, "joint": 0.90},
        "256": {"H_OL": 1.0, "Delta_N_equivalence": 0.97, "joint": 0.96},
    }
    assert expected_selected_N(rows, [128, 160, 192, 256], 0.8) == 160


def test_program_q_contract_freezes_authoritative_N_256() -> None:
    power = CONTRACT["power"]["authoritative_result"]
    assert (
        CONTRACT["status"]
        == "FROZEN_N_256_CONFIRMATION_IMPLEMENTATION_AUDITED_PENDING_AUTHORIZATION"
    )
    assert CONTRACT["confirmation"]["N"] == 256
    assert power["selected_N"] == 256
    assert power["joint_power"] >= CONTRACT["power"]["minimum_joint_power"]
    assert len(power["result_sha256"]) == 64
    assert len(power["cache_sha256"]) == 64
    assert CONTRACT["confirmation"]["opened"] is False
    assert (
        CONTRACT["confirmation"]["preopening_gate_state"]["confirmatory_launcher_audit"]
        == "PASS_CONFIRMATION_IMPLEMENTATION_PREOPEN_AUDIT"
    )
    assert (
        CONTRACT["confirmation"]["preopening_gate_state"]["independent_authorization"]
        == "PENDING"
    )


def test_latency_benchmark_reports_batch_one_and_failures() -> None:
    payload = benchmark_callable(
        lambda observation: int(np.argmax(observation)),
        [np.asarray([0.0, 1.0], dtype=np.float32)],
        warmup=2,
        repeats=10,
    )
    assert payload["batch_size"] == 1
    assert payload["failures"] == 0
    assert payload["p95_ms"] >= 0.0


def test_program_q_primary_inference_reselects_and_builds_bilateral_equivalence() -> None:
    panels = {}
    for cell in CELL_IDS:
        panels[cell] = {
            "learner": np.asarray(
                [[0.80, 0.78, 0.81, 0.79], [0.79, 0.80, 0.80, 0.79]], dtype=float
            ),
            "open_loop": np.asarray(
                [[0.70, 0.75], [0.72, 0.74], [0.71, 0.73], [0.70, 0.76]],
                dtype=float,
            ),
            "classical": np.asarray(
                [[0.79, 0.80, 0.80, 0.79], [0.78, 0.79, 0.79, 0.78]], dtype=float
            ),
        }
    result = simultaneous_primary_inference(panels, resamples=100, rng_seed=123)
    assert result["comparator_reselection_inside_every_resample"] is True
    assert result["H_OL_one_sided_and_Delta_N_two_sided"] is True
    for cell in CELL_IDS:
        h = result["estimates"][f"{cell}::H_OL"]
        delta = result["estimates"][f"{cell}::Delta_N"]
        assert h["point"] > 0.0
        assert delta["lcb95"] <= delta["point"] <= delta["ucb95"]


def test_program_q_bootstrap_one_is_rejected_fail_closed() -> None:
    panels = {
        cell: {
            "learner": np.asarray([[0.8, 0.9], [0.81, 0.91]]),
            "open_loop": np.asarray([[0.7, 0.75], [0.71, 0.74]]),
            "classical": np.asarray([[0.79, 0.89], [0.78, 0.88]]),
        }
        for cell in CELL_IDS
    }
    with np.testing.assert_raises_regex(ValueError, "at least two"):
        simultaneous_primary_inference(panels, resamples=1)


def test_program_q_guardrail_inference_accepts_only_exact_zero_se_contrasts() -> None:
    panels = {}
    for cell_index, cell in enumerate(CELL_IDS):
        learner_ret = np.asarray([[0.8, 0.9, 0.85], [0.81, 0.89, 0.86]])
        open_ret = np.asarray([[0.7, 0.72], [0.74, 0.71], [0.73, 0.75]])
        classical_ret = np.asarray([[0.79, 0.88, 0.84], [0.78, 0.87, 0.83]])
        exact_learner = np.zeros_like(learner_ret)
        exact_open = np.zeros_like(open_ret)
        exact_classical = np.zeros_like(classical_ret)
        varying_learner = learner_ret - 0.1 + cell_index * 0.001
        varying_open = open_ret - 0.1
        varying_classical = classical_ret - 0.1
        panels[cell] = {
            "learner_ret": learner_ret,
            "open_ret": open_ret,
            "classical_ret": classical_ret,
            "learner__ret_full": exact_learner,
            "open__ret_full": exact_open,
            "classical__ret_full": exact_classical,
            "learner__quantity_ret_full": exact_learner,
            "open__quantity_ret_full": exact_open,
            "classical__quantity_ret_full": exact_classical,
            "learner__worst_product_fill": varying_learner,
            "open__worst_product_fill": varying_open,
            "classical__worst_product_fill": varying_classical,
        }
    result = simultaneous_guardrail_inference(panels, resamples=100, rng_seed=123)
    assert len(result["deterministic_zero_se_endpoints"]) == 12
    for name in result["deterministic_zero_se_endpoints"]:
        estimate = result["estimates"][name]
        assert estimate["point"] == estimate["lcb95"] == 0.0
        assert estimate["inference_kind"] == "exact_deterministic_contrast"
    stochastic = [
        row
        for name, row in result["estimates"].items()
        if "worst_product_fill" in name
    ]
    assert stochastic
    assert all(row["inference_kind"] == "studentized_max_t" for row in stochastic)


def test_program_q_plan_does_not_open_scientific_seeds(tmp_path: Path) -> None:
    path = tmp_path / "plan.json"
    write_plan(path)
    payload = json.loads(path.read_text())
    assert payload["N"] == 256
    assert payload["expected_shards"] == 768
    assert payload["scientific_seeds_opened_by_plan"] == 0
    assert payload["external_collaborator_dependency"] is False


def test_program_q_producer_fails_closed_without_independent_authorization(
    tmp_path: Path,
) -> None:
    authorization = tmp_path / "authorization.json"
    authorization.write_text('{"status": "NOT_AUTHORIZED"}')
    with np.testing.assert_raises_regex(RuntimeError, "not independently authorized"):
        verify_authorization(authorization)


def test_program_q_resume_rejects_incomplete_existing_shard(tmp_path: Path) -> None:
    shard = tmp_path / "tape_7490001.npz"
    np.savez_compressed(shard, cell_index=0, tape_seed=7490001)
    with np.testing.assert_raises_regex(RuntimeError, "schema mismatch"):
        validate_shard(shard, cell_index=0, tape_seed=7490001)


def test_program_q_resume_binds_exact_pre_reduction_shard_manifest(
    tmp_path: Path,
) -> None:
    from supply_chain.program_o_eval_custody import sha256, write_sha256_manifest

    shards = tmp_path / "shards"
    custody = tmp_path / "custody"
    paths = []
    for index in range(2):
        path = shards / f"cell/tape_{index}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"shard-{index}".encode())
        paths.append(path)
    manifest = custody / "pre_reduction_shards.sha256"
    write_sha256_manifest(shards, paths, manifest)
    authorization = {"pre_reduction_shard_manifest_sha256": sha256(manifest)}
    rows = verify_pre_reduction_shard_manifest(
        shards=shards, custody=custody, authorization=authorization, expected=2
    )
    assert len(rows) == 2
    paths[0].write_bytes(b"tampered")
    with np.testing.assert_raises_regex(ValueError, "SHA-256 mismatch"):
        verify_pre_reduction_shard_manifest(
            shards=shards, custody=custody, authorization=authorization, expected=2
        )


def test_direct_replay_detects_corrupt_promoted_learner_matrix() -> None:
    payload = {}
    direct = {}
    for key in MATRIX_KEYS:
        direct[key] = 1.0
        payload[f"open_loop__{key}"] = np.asarray([1.0])
        payload[f"learner__{key}"] = np.asarray([1.0])
    payload["learner__ret_visible"][0] = 9.0
    _, failures = compare_promoted_sources(
        payload,
        direct,
        calendar=(0, 0, 0, 0, 0, 0, 0, 0),
        sources=[("open_loop", 0, "open"), ("learner", 0, "learner")],
        atol=1e-8,
    )
    assert len(failures) == 1
    assert failures[0]["source"] == "learner"


def test_program_q_resume_archives_prior_live_custody(tmp_path: Path) -> None:
    custody = tmp_path / "custody"
    custody.mkdir()
    for name in ("producer_control.json", "producer_exit.json", "watcher_ready.json"):
        (custody / name).write_text("{}")
    attempt = archive_previous_attempt(custody)
    assert attempt == 1
    assert not (custody / "producer_control.json").exists()
    assert (custody / "attempts/attempt_1/producer_exit.json").is_file()
    (custody / "producer_exit.json").write_text("{}")
    attempt = archive_previous_attempt(custody)
    assert attempt == 2
    assert (custody / "attempts/attempt_2/producer_exit.json").is_file()
