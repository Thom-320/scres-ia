"""A sealed surface-cache slice must never be overwritten by the audit (incident 2026-11-26).

A run in an environment without torch saw a module-manifest mismatch, recomputed the surface and
replaced results/surface_cache/wrap288_v1/* with unsealed slices, which broke every later reader.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_meta_learner_normaliser_audit_v1 import load_or_compute_slice


def _sealed(path: Path) -> str:
    blob = {"schema_version": "garrido_surface_cache_v1", "scientific_payload_sha256": "x",
            "self_sha256": "y", "module_manifest": {"modules": ["a"], "missing": []},
            "cells": [{"value": 1.0}]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(blob))
    return path.read_text()


def test_incompatible_sealed_slice_fails_closed_and_is_untouched(tmp_path: Path) -> None:
    p = tmp_path / "R1r" / "5300001.json"
    before = _sealed(p)
    manifest = {"modules": ["a"], "missing": ["torch"]}
    with pytest.raises(ValueError, match="sealed"):
        load_or_compute_slice(p, manifest, lambda: [{"value": 2.0}], "R1r", 5300001, 1.0)
    assert p.read_text() == before


def test_missing_slice_is_computed_and_written(tmp_path: Path) -> None:
    p = tmp_path / "R1r" / "5300001.json"
    cells = load_or_compute_slice(p, {"modules": [], "missing": []}, lambda: [{"value": 2.0}],
                                  "R1r", 5300001, 1.0)
    assert cells == [{"value": 2.0}]
    assert json.loads(p.read_text())["cells"] == cells


def test_unsealed_stale_slice_is_recomputed(tmp_path: Path) -> None:
    p = tmp_path / "R1r" / "5300001.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"module_manifest": {"modules": ["old"]}, "cells": [{"value": 0.0}]}))
    cells = load_or_compute_slice(p, {"modules": ["new"], "missing": []},
                                  lambda: [{"value": 3.0}], "R1r", 5300001, 1.0)
    assert cells == [{"value": 3.0}]
