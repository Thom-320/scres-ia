from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

from tools.verify_sealed_payload import verify_payload


def _sealed(path: Path, payload: dict) -> None:
    body = json.dumps(payload, indent=1, sort_keys=True, default=str).encode()
    payload = dict(payload, self_sha256=sha256(body).hexdigest())
    path.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str) + "\n")


def test_independent_verifier_accepts_a_minimal_sealed_payload(tmp_path: Path) -> None:
    artifact = tmp_path / "result.json"
    _sealed(artifact, {"schema_version": "test", "claim_status": "OK"})
    assert verify_payload(artifact, tmp_path) == []


def test_independent_verifier_rejects_payload_mutation(tmp_path: Path) -> None:
    artifact = tmp_path / "result.json"
    _sealed(artifact, {"schema_version": "test", "claim_status": "OK"})
    payload = json.loads(artifact.read_text())
    payload["claim_status"] = "TAMPERED"
    artifact.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    assert any("self_sha256" in error for error in verify_payload(artifact, tmp_path))
