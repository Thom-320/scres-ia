"""Program O-R evaluator v1.2 custody and ledger fail-closed tests."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.adjudicate_program_o_ret_calibration import adjudicate  # noqa: E402
from scripts.evaluate_program_o_ret_learner import (  # noqa: E402
    demand_ledger_residuals,
    simultaneous_bootstrap,
)
from supply_chain.program_o_eval_custody import (  # noqa: E402
    sha256,
    verify_sha256_manifest,
    write_sha256_manifest,
)


def _ledger() -> dict[str, np.ndarray]:
    return {
        "generated_orders": np.asarray([10.0]),
        "visible_rows": np.asarray([7.0]),
        "omitted_rows": np.asarray([3.0]),
        "unresolved_orders": np.asarray([3.0]),
        "omitted_quantity": np.asarray([30.0]),
        "unresolved_quantity": np.asarray([30.0]),
        "remaining_quantity_P_C": np.asarray([10.0]),
        "remaining_quantity_P_H": np.asarray([20.0]),
        "lost_orders": np.asarray([0.0]),
        "lost_quantity": np.asarray([0.0]),
    }


def test_explicit_demand_identities_detect_disappearing_order():
    metrics = _ledger()
    assert all(np.max(np.abs(row)) == 0.0 for row in demand_ledger_residuals(metrics).values())
    metrics["omitted_rows"] = np.asarray([2.0])
    residuals = demand_ledger_residuals(metrics)
    assert residuals["generated_equals_visible_plus_omitted"].item() == 1.0
    assert residuals["omitted_rows_equal_unresolved_orders"].item() == -1.0


def test_manifest_roundtrip_and_tamper_detection(tmp_path: Path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"trusted")
    manifest = tmp_path / "files.sha256"
    write_sha256_manifest(tmp_path, [artifact], manifest)
    assert verify_sha256_manifest(tmp_path, manifest) == {
        "artifact.bin": sha256(artifact)
    }
    artifact.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_sha256_manifest(tmp_path, manifest)


def test_adjudication_fails_when_direct_audit_does_not_bind_result(tmp_path: Path):
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    raw = calibration / "raw.npz"
    raw.write_bytes(b"raw")
    raw_manifest = calibration / "raw_files.sha256"
    write_sha256_manifest(calibration, [raw], raw_manifest)
    result = calibration / "result.json"
    result.write_text(json.dumps({
        "schema_version": "program_o_ret_only_learner_evaluation_v1_2",
        "phase": "calibration",
        "provisional_primary_pass": True,
        "amendment_gates": {"all": True},
    }))
    write_sha256_manifest(
        calibration,
        [result, raw_manifest, raw],
        calibration / "evaluation_files.sha256",
    )
    audit_dir = tmp_path / "audit"
    audit_dir.mkdir()
    audit = audit_dir / "independent_full_des_audit.json"
    audit.write_text(json.dumps({
        "phase": "calibration",
        "passed": True,
        "evaluation_result_sha256": "0" * 64,
    }))
    write_sha256_manifest(audit_dir, [audit], audit_dir / "audit_files.sha256")
    verdict = adjudicate(result, audit)
    assert verdict["status"] == "STOP_CALIBRATION_NOT_ELIGIBLE"
    assert verdict["checks"]["direct_audit_binds_result"] is False


def test_simultaneous_bootstrap_has_one_named_row_per_estimand():
    tapes = 4
    learner_seeds = 2
    rows = {
        "c1": {
            "learner": {
                key: np.full((learner_seeds, tapes), 0.8)
                for key in ("ret_visible", "ret_full", "quantity_ret_full", "worst_product_fill")
            },
            "open_loop": {
                key: np.full((tapes, 2), 0.6)
                for key in ("ret_visible", "ret_full", "quantity_ret_full", "worst_product_fill")
            },
            "classical": {
                key: np.full((2, tapes), 0.7)
                for key in ("ret_visible", "ret_full", "quantity_ret_full", "worst_product_fill")
            },
        }
    }
    result = simultaneous_bootstrap(rows, resamples=20)
    assert len(result["estimates"]) == 8
