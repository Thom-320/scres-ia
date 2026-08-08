#!/usr/bin/env python3
"""Independent, stdlib-only verification of a sealed Paper 2 payload.

This module deliberately does not import ``supply_chain``.  It verifies the envelope produced by
``seal_and_write`` and the source hashes recorded in ``module_manifest`` from outside the module
whose drift is under adjudication.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_body(payload: dict[str, Any]) -> bytes:
    body = dict(payload)
    body.pop("self_sha256", None)
    return json.dumps(body, indent=1, sort_keys=True, default=str).encode()


def _repo_file(root: Path, rel: str) -> Path:
    candidate = (root / rel).resolve()
    root_resolved = root.resolve()
    if candidate != root_resolved and root_resolved not in candidate.parents:
        raise ValueError(f"path escapes repository root: {rel}")
    return candidate


def verify_payload(path: Path, root: Path | None = None,
                   expected_file_sha256: str | None = None) -> list[str]:
    """Return all verification errors; an empty list means the payload is internally sound."""
    root = (root or Path.cwd()).resolve()
    errors: list[str] = []
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read JSON payload: {exc}"]
    if not isinstance(payload, dict):
        return ["payload root is not an object"]

    stored_self = payload.get("self_sha256")
    if not isinstance(stored_self, str):
        errors.append("missing self_sha256")
    elif stored_self != _sha_bytes(_canonical_body(payload)):
        errors.append("self_sha256 does not match the canonical pre-seal body")

    actual_file = _sha_bytes(path.read_bytes())
    if expected_file_sha256 and actual_file != expected_file_sha256:
        errors.append("file_sha256 does not match the supplied expected digest")

    for key in ("contract_path", "reference_path"):
        rel = payload.get(key)
        expected = payload.get(key.replace("_path", "_sha256"))
        if not isinstance(rel, str) or not isinstance(expected, str):
            continue
        try:
            dep = _repo_file(root, rel)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if not dep.exists():
            errors.append(f"missing {key}: {rel}")
        elif key == "reference_path":
            # The repository's sealed-payload convention stores reference_sha256 as
            # the referenced JSON payload's *self* hash, not as the file-byte hash.
            # A future producer may add reference_file_sha256 when byte identity is
            # also required; keep that check separate and unambiguous.
            reference_file_expected = payload.get("reference_file_sha256")
            if isinstance(reference_file_expected, str):
                if _sha_bytes(dep.read_bytes()) != reference_file_expected:
                    errors.append(f"reference file_sha256 mismatch: {rel}")
            else:
                try:
                    reference_payload = json.loads(dep.read_text())
                except (OSError, json.JSONDecodeError) as exc:
                    errors.append(f"cannot read referenced payload: {rel}: {exc}")
                else:
                    reference_self = reference_payload.get("self_sha256")
                    if reference_self != expected:
                        errors.append(f"reference self_sha256 mismatch: {rel}")
        elif _sha_bytes(dep.read_bytes()) != expected:
            errors.append(f"{key} file hash mismatch: {rel}")

    manifests: list[tuple[str, dict[str, Any]]] = []
    manifest = payload.get("module_manifest")
    if isinstance(manifest, dict):
        manifests.append(("module_manifest", manifest))
    cache_manifests = payload.get("cache_module_manifests")
    if isinstance(cache_manifests, dict):
        for name, cache_manifest in cache_manifests.items():
            if isinstance(cache_manifest, dict):
                manifests.append((f"cache_module_manifests.{name}", cache_manifest))
    for manifest_name, manifest in manifests:
        for rel, expected in (manifest.get("modules") or {}).items():
            try:
                dep = _repo_file(root, str(rel))
            except ValueError as exc:
                errors.append(str(exc))
                continue
            if not dep.exists():
                errors.append(f"{manifest_name} module missing: {rel}")
            elif _sha_bytes(dep.read_bytes()) != expected:
                errors.append(f"{manifest_name} hash mismatch: {rel}")
        entry = manifest.get("entry_script")
        entry_expected = manifest.get("entry_script_sha256")
        if entry and entry_expected:
            try:
                script = _repo_file(root, str(entry))
            except ValueError as exc:
                errors.append(str(exc))
            else:
                if not script.exists():
                    errors.append(f"{manifest_name} entry script missing: {entry}")
                elif _sha_bytes(script.read_bytes()) != entry_expected:
                    errors.append(f"{manifest_name} entry hash mismatch: {entry}")

    falsifiers = payload.get("falsifiers")
    if isinstance(falsifiers, dict) and falsifiers.get("all_passed") is not True:
        errors.append("falsifiers.all_passed is not true")
    if payload.get("schema_version") == "frozen_path_equivalence_v2":
        surface = payload.get("surface", {})
        expected_slices = len(payload.get("contexts", [])) * len(payload.get("seeds", []))
        expected_cells = {
            "base": expected_slices * int(payload.get("n_base_configs", 0)),
            "ext": expected_slices * int(payload.get("n_ext_configs", 0)),
        }
        for kind in ("base", "ext"):
            row = surface.get(kind, {}) if isinstance(surface, dict) else {}
            if not row or row.get("slices", 0) == 0:
                errors.append(f"sealed equivalence surface is incomplete: {kind}")
            if row.get("slices") != expected_slices:
                errors.append(f"sealed equivalence surface has incomplete slice count: {kind}")
            if row.get("cells") != expected_cells[kind]:
                errors.append(f"sealed equivalence surface has incomplete cell count: {kind}")
            if row.get("mismatches") != 0:
                errors.append(f"sealed equivalence surface has mismatches: {kind}")
        chain = payload.get("downstream_chain", {})
        if chain.get("ran") is not True or chain.get("n_differing") != 0:
            errors.append("downstream chain was not reproduced exactly")
        if payload.get("verdict_a_historical_identity") != \
                "HISTORICAL_SOURCE_RECOVERED_AND_REPRODUCES":
            errors.append("historical source certificate A is not green")
        m4 = payload.get("mutation_controls", {}).get(
            "m4_seal_only_must_not_move_science", {})
        if not (m4.get("applicable") is True and m4.get("manifest_moved") is True
                and m4.get("science_unchanged") is True and m4.get("detected") is True):
            errors.append("seal-only mutation control did not prove manifest drift with identical science")
        allowed_classes = {
            "SOURCE_HASH_MATCH",
            "SOURCE_DRIFT__NO_SCIENTIFIC_PATH_EFFECT",
            "SOURCE_DRIFT__OBSERVATIONALLY_EQUIVALENT",
            "SOURCE_DRIFT__SCIENTIFICALLY_MATERIAL",
        }
        classifications = payload.get("mutation_controls", {}).get(
            "source_drift_classification")
        if not isinstance(classifications, dict) or not classifications:
            errors.append("source drift classification is missing")
        else:
            for rel, row in classifications.items():
                if not isinstance(row, dict) or row.get("classification") not in allowed_classes:
                    errors.append(f"invalid source drift classification: {rel}")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("artifact", type=Path)
    ap.add_argument("--root", type=Path, default=Path.cwd())
    ap.add_argument("--expected-file-sha256")
    args = ap.parse_args()
    errors = verify_payload(args.artifact, args.root, args.expected_file_sha256)
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1
    print(f"OK: {args.artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
