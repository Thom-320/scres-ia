from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.seal_garrido_surface_cache_v1 import (
    PANEL_KEYS,
    observable_key,
    validate_slice,
)
from scripts.run_meta_learner_normaliser_audit_v1 import (
    CONFIGS,
    CONTEXTS,
    twin_surface_falsifier,
)


def _cell() -> dict:
    panel = {
        "delivered_rations": 10.0,
        "demanded_rations": 12.0,
        "flow_fill_rate": 0.8,
        "lost_orders": 2.0,
        "ret_excel": 0.1,
        "ret_excel_full_ledger": 0.09,
        "ret_excel_risk_conditional": 0.1,
    }
    return {"value": 0.1, "drivers": [0.0, 1.0, 0.0, 0.0], "panel": panel}


def _slice() -> dict:
    return {
        "grid_id": "wrap288_v1",
        "context": "R1r",
        "seed": 5_300_001,
        "horizon_hours": 8736.0,
        "metric": "ret_excel_risk_conditional",
        "module_manifest": {"schema": "module_manifest_v2"},
        "cells": [_cell() for _ in range(288)],
    }


def test_validate_slice_requires_full_panel() -> None:
    payload = _slice()
    validate_slice(payload)
    assert observable_key(payload["cells"][0])["components"] == list(PANEL_KEYS)

    broken = copy.deepcopy(payload)
    del broken["cells"][0]["panel"]["flow_fill_rate"]
    with pytest.raises(ValueError, match="missing"):
        validate_slice(broken)


def test_validate_slice_rejects_scalar_only_cache() -> None:
    payload = _slice()
    payload["cells"][0].pop("panel")
    with pytest.raises(ValueError, match="full value/panel"):
        validate_slice(payload)


def test_twin_surface_falsifier_detects_oracle_tail_dependency() -> None:
    """The affine-scale test must be supplemented by a hidden-tail twin test."""
    seed = 5_300_001
    surface = {
        (context, seed): [
            {"value": float(index), "drivers": [0.0], "panel": {}}
            for index in range(len(CONFIGS))
        ]
        for context in CONTEXTS
    }

    result = twin_surface_falsifier(surface, seed, budget=24)

    assert result["passed"] is True
    assert result["prefix_passed"] is True
    assert result["oracle_reacted"] is True
