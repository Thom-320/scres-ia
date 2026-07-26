from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from scripts.run_q_r1_matched_retention_factorial_v4 import (
    build_histories,
    estimands,
    evaluate_neural_arm,
    json_sha256,
    load_authority,
    main,
    sha256,
    static_rows,
    validate_shared_static_bar,
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
