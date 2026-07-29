from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from scripts.run_q_r1_matched_retention_factorial_v4 import (
    authoritative_static_bar_sha256,
    build_histories,
    estimands,
    evaluate_neural_arm,
    json_sha256,
    load_authority,
    main,
    sha256,
    static_rows,
    validate_shared_static_bar,
    validate_shared_structured_bar,
)
from scripts.salvage_q_r1_matched_retention_factorial_v4 import (
    _verify_source_hashes,
)


class ConstantModel:
    def predict(self, observation, *, state, episode_start, deterministic):
        assert observation.shape == (23,)
        assert deterministic is True
        return np.asarray(0), state


def _write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _static_bar_chain(tmp_path: Path) -> tuple[Path, Path, Path, list[int]]:
    roots = list(range(1001, 1017))
    identities = [
        {
            "history_root": root,
            "campaign_index": campaign,
            "kappa": kappa,
            "skeleton_sha256": hashlib.sha256(
                f"{root}:{campaign}:{kappa}".encode()
            ).hexdigest(),
        }
        for root in roots
        for kappa in (0.5, 0.75, 0.9)
        for campaign in range(12)
    ]
    contract_sha = "contract-sha"
    opening_path = tmp_path / "development_opening_receipt.json"
    _write(
        opening_path,
        {
            "mode": "static-bar",
            "contract_sha256": contract_sha,
        },
    )
    bar_path = tmp_path / "static_bar.json"
    bar = {
        "calendar": [0] * 8,
        "frontier_row": 0,
        "selection_campaigns": 576,
        "identities": identities,
    }
    _write(bar_path, bar)
    completion_path = tmp_path / "static_bar_completion_receipt.json"
    _write(
        completion_path,
        {
            "schema_version": (
                "q_r1_factorial_v4_static_bar_completion_receipt"
            ),
            "mode": "static-bar",
            "contract_sha256": contract_sha,
            "opening_receipt_sha256": sha256(opening_path),
            "static_bar_sha256": sha256(bar_path),
            "identities_sha256": json_sha256(identities),
            "selection_roots": roots,
            "selection_campaigns": 576,
            "calendar": [0] * 8,
            "frontier_row": 0,
        },
    )
    return bar_path, completion_path, opening_path, roots


def _structured_bar_chain(
    tmp_path: Path,
) -> tuple[Path, Path, Path, list[int]]:
    roots = list(range(1001, 1017))
    rows = [
        {
            "history_root": root,
            "campaign_index": campaign,
            "kappa": kappa,
            "arm": arm,
            "skeleton_sha256": hashlib.sha256(
                f"skeleton:{root}:{campaign}:{kappa}".encode()
            ).hexdigest(),
            "prefix_state_hash": hashlib.sha256(
                f"prefix:{root}:{campaign}:{kappa}".encode()
            ).hexdigest(),
            "early_ret_complete_cohort": 0.5,
        }
        for root in roots
        for kappa in (0.5, 0.75, 0.9)
        for campaign in (0, 1)
        for arm in ("structured_reset", "structured_retained")
    ]
    identities = [
        {
            "history_root": int(row["history_root"]),
            "kappa": float(row["kappa"]),
            "campaign_index": int(row["campaign_index"]),
            "arm": str(row["arm"]),
            "skeleton_sha256": str(row["skeleton_sha256"]),
            "prefix_state_hash": str(row["prefix_state_hash"]),
        }
        for row in rows
    ]
    opening = tmp_path / "structured_bar_opening_receipt.json"
    _write(
        opening,
        {
            "schema_version": "q_r1_shared_structured_opening_v1",
            "base_contract_sha256": "contract-sha",
            "amendment_sha256": "amendment-sha",
        },
    )
    artifact = tmp_path / "structured_rows.json"
    _write(artifact, rows)
    completion = tmp_path / "structured_bar_completion_receipt.json"
    _write(
        completion,
        {
            "schema_version": "q_r1_shared_structured_completion_v1",
            "opening_receipt_sha256": sha256(opening),
            "base_contract_sha256": "contract-sha",
            "amendment_sha256": "amendment-sha",
            "structured_rows_sha256": sha256(artifact),
            "rows_digest_sha256": json_sha256(rows),
            "identities_sha256": json_sha256(identities),
            "confirmation_roots_opened": False,
        },
    )
    return artifact, completion, opening, roots


def test_freeze_receipt_closes_preflight_and_authorizes_development_loader() -> None:
    with pytest.raises(RuntimeError, match="closed after the contract freeze"):
        load_authority("instrument-preflight")
    contract, receipt = load_authority("development-worker")
    assert contract["status"] == "DRAFT_PROSPECTIVE_UNOPENED_NOT_AUTHORITY"
    assert receipt is not None
    assert receipt["status"] == "FROZEN_PROSPECTIVE_UNOPENED"
    assert receipt["fresh_development_roots_opened"] is False


def test_neural_arm_emits_complete_campaign_rows_with_one_checkpoint_hash() -> None:
    histories = build_histories([7_570_801], (0.90,))
    rows = evaluate_neural_arm(
        ConstantModel(),
        histories=histories,
        arm="P1_H1",
        retained_prior=True,
        reset_hidden_at_boundaries=False,
        checkpoint_sha256="abc",
    )
    assert len(rows) == 12
    assert {row["checkpoint_sha256"] for row in rows} == {"abc"}
    assert {row["arm"] for row in rows} == {"P1_H1"}
    for field in (
        "early_ret_complete_cohort",
        "early_ret_visible",
        "worst_product_fill",
        "unresolved_orders",
        "unresolved_quantity",
        "lost_orders",
        "lost_quantity",
        "service_loss",
    ):
        assert all(field in row for row in rows)


def test_estimands_include_the_neural_premium_from_common_rows() -> None:
    rows = []
    values = {
        "P0_H0": 0.50,
        "P1_H0": 0.55,
        "P0_H1": 0.52,
        "P1_H1": 0.60,
        "structured_reset": 0.53,
        "structured_retained": 0.58,
    }
    for arm, value in values.items():
        rows.append(
            {
                "arm": arm,
                "history_root": 1,
                "kappa": 0.9,
                "campaign_index": 0,
                "early_ret_complete_cohort": value,
            }
        )
    result = estimands(rows)
    assert result["explicit_context_value"]["mean"] == pytest.approx(0.05)
    assert result["raw_recurrent_memory_value"]["mean"] == pytest.approx(0.02)
    assert result["structured_retained_value"]["mean"] == pytest.approx(0.05)
    assert result["neural_premium"]["mean"] == pytest.approx(0.02)


def test_static_rows_emit_the_same_mandatory_service_aliases() -> None:
    histories = build_histories([7_570_801], (0.90,))
    rows = static_rows(histories, calendar=[0] * 8)
    assert len(rows) == 12
    assert all(row["service_loss"] == row["service_loss_auc"] for row in rows)
    assert all(row["whole_campaign_ret"] == row["ret_visible"] for row in rows)


def test_static_bar_chain_accepts_the_one_authoritative_artifact(
    tmp_path: Path,
) -> None:
    bar, completion, opening, roots = _static_bar_chain(tmp_path)
    loaded = validate_shared_static_bar(
        static_bar_path=bar,
        completion_receipt_path=completion,
        opening_receipt_path=opening,
        expected_contract_sha256="contract-sha",
        expected_roots=roots,
        expected_campaigns=576,
    )
    assert loaded["calendar"] == [0] * 8


def test_static_bar_chain_rejects_an_altered_bar(tmp_path: Path) -> None:
    bar, completion, opening, roots = _static_bar_chain(tmp_path)
    payload = json.loads(bar.read_text())
    payload["calendar"] = [3] * 8
    _write(bar, payload)
    with pytest.raises(RuntimeError, match="artifact hash mismatch"):
        validate_shared_static_bar(
            static_bar_path=bar,
            completion_receipt_path=completion,
            opening_receipt_path=opening,
            expected_contract_sha256="contract-sha",
            expected_roots=roots,
            expected_campaigns=576,
        )


def test_static_bar_chain_rejects_a_wrong_completion_receipt(
    tmp_path: Path,
) -> None:
    bar, completion, opening, roots = _static_bar_chain(tmp_path)
    receipt = json.loads(completion.read_text())
    receipt["static_bar_sha256"] = "0" * 64
    _write(completion, receipt)
    with pytest.raises(RuntimeError, match="artifact hash mismatch"):
        validate_shared_static_bar(
            static_bar_path=bar,
            completion_receipt_path=completion,
            opening_receipt_path=opening,
            expected_contract_sha256="contract-sha",
            expected_roots=roots,
            expected_campaigns=576,
        )


def test_two_workers_cannot_accept_different_static_bar_hashes(
    tmp_path: Path,
) -> None:
    bar, completion, opening, roots = _static_bar_chain(tmp_path)
    divergent = tmp_path / "divergent_static_bar.json"
    payload = json.loads(bar.read_text())
    payload["frontier_row"] = 1
    _write(divergent, payload)
    validate_shared_static_bar(
        static_bar_path=bar,
        completion_receipt_path=completion,
        opening_receipt_path=opening,
        expected_contract_sha256="contract-sha",
        expected_roots=roots,
        expected_campaigns=576,
    )
    with pytest.raises(RuntimeError, match="artifact hash mismatch"):
        validate_shared_static_bar(
            static_bar_path=divergent,
            completion_receipt_path=completion,
            opening_receipt_path=opening,
            expected_contract_sha256="contract-sha",
            expected_roots=roots,
            expected_campaigns=576,
        )


def test_shared_structured_chain_accepts_exact_frozen_coverage(
    tmp_path: Path,
) -> None:
    artifact, completion, opening, roots = _structured_bar_chain(tmp_path)
    rows, _receipt = validate_shared_structured_bar(
        rows_path=artifact,
        completion_receipt_path=completion,
        opening_receipt_path=opening,
        expected_contract_sha256="contract-sha",
        expected_amendment_sha256="amendment-sha",
        expected_roots=roots,
    )
    assert len(rows) == 192


def test_shared_structured_chain_rejects_altered_rows(tmp_path: Path) -> None:
    artifact, completion, opening, roots = _structured_bar_chain(tmp_path)
    rows = json.loads(artifact.read_text())
    rows[0]["early_ret_complete_cohort"] = 0.9
    _write(artifact, rows)
    with pytest.raises(RuntimeError, match="artifact hash mismatch"):
        validate_shared_structured_bar(
            rows_path=artifact,
            completion_receipt_path=completion,
            opening_receipt_path=opening,
            expected_contract_sha256="contract-sha",
            expected_amendment_sha256="amendment-sha",
            expected_roots=roots,
        )


def test_shared_structured_chain_rejects_incomplete_coverage(
    tmp_path: Path,
) -> None:
    artifact, completion, opening, roots = _structured_bar_chain(tmp_path)
    rows = json.loads(artifact.read_text())[:-1]
    _write(artifact, rows)
    receipt = json.loads(completion.read_text())
    receipt["structured_rows_sha256"] = sha256(artifact)
    receipt["rows_digest_sha256"] = json_sha256(rows)
    _write(completion, receipt)
    with pytest.raises(RuntimeError, match="row count mismatch"):
        validate_shared_structured_bar(
            rows_path=artifact,
            completion_receipt_path=completion,
            opening_receipt_path=opening,
            expected_contract_sha256="contract-sha",
            expected_amendment_sha256="amendment-sha",
            expected_roots=roots,
        )


def test_runner_rejects_an_existing_output_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    existing = tmp_path / "already-exists"
    existing.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--mode",
            "instrument-preflight",
            "--output-dir",
            str(existing),
        ],
    )
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        main()


def test_worker_hashes_the_shared_authoritative_static_bar(tmp_path: Path) -> None:
    output = tmp_path / "worker"
    output.mkdir()
    shared = tmp_path / "static_bar.json"
    shared.write_text("{}\n")
    assert authoritative_static_bar_sha256(
        mode="development-worker",
        output_dir=output,
        static_bar_path=shared,
    ) == sha256(shared)


def test_static_bar_mode_hashes_its_own_output(tmp_path: Path) -> None:
    output = tmp_path / "bar"
    output.mkdir()
    local = output / "static_bar.json"
    local.write_text('{"calendar":[0]}\n')
    divergent = tmp_path / "other.json"
    divergent.write_text('{"calendar":[3]}\n')
    assert authoritative_static_bar_sha256(
        mode="static-bar",
        output_dir=output,
        static_bar_path=divergent,
    ) == sha256(local)


def test_salvage_source_hash_validation_is_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "s01_1"
    source.mkdir()
    tracked = source / "checkpoint_progress.json"
    tracked.write_text("{}\n")
    manifest = {
        "files": [
            {
                "name": tracked.name,
                "sha256": sha256(tracked),
            }
        ]
    }
    _verify_source_hashes(source, manifest)
    tracked.write_text('{"altered":true}\n')
    with pytest.raises(RuntimeError, match="source hash mismatch"):
        _verify_source_hashes(source, manifest)
