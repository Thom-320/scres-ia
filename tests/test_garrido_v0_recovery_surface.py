from __future__ import annotations

import copy
import json

import pytest

from scripts.build_garrido_v0_recovery_surface_v1 import (
    CONTEXT_ORDER,
    GRID_ID,
    scientific_hash,
    verify_surface,
)
from supply_chain.expanded_contract_controllers_v2 import ALL_POSTURES


def minimal_surface() -> dict:
    panel = {
        "temporal": {"system_ttr_n_clusters": 1},
        "order_tape_sha256": "same",
    }
    cell = {
        "posture": None,
        "utility": 1.0,
        "recovery": {"restricted_ttr_hours": 0.0},
        "risk": panel,
        "placebo": panel,
    }
    payload = {
        "grid_id": GRID_ID,
        "context_order": list(CONTEXT_ORDER),
        "placebo_cells": [
            {"posture": list(posture), "panel": panel} for posture in ALL_POSTURES
        ],
        "contexts": {
            context: [dict(cell, posture=list(posture)) for posture in ALL_POSTURES]
            for context in CONTEXT_ORDER
        },
    }
    payload["scientific_payload_sha256"] = scientific_hash(payload)
    return payload


def test_surface_verifier_accepts_complete_scientific_payload():
    verify_surface(minimal_surface())


def test_surface_verifier_rejects_missing_posture():
    payload = minimal_surface()
    payload["contexts"][CONTEXT_ORDER[0]].pop()
    payload["scientific_payload_sha256"] = scientific_hash(payload)
    with pytest.raises(ValueError, match="incomplete"):
        verify_surface(payload)


def test_surface_verifier_rejects_crn_drift():
    payload = minimal_surface()
    payload["contexts"][CONTEXT_ORDER[0]][-1]["risk"] = copy.deepcopy(
        payload["contexts"][CONTEXT_ORDER[0]][-1]["risk"]
    )
    payload["contexts"][CONTEXT_ORDER[0]][-1]["risk"]["order_tape_sha256"] = "drift"
    payload["scientific_payload_sha256"] = scientific_hash(payload)
    with pytest.raises(ValueError, match="CRN"):
        verify_surface(payload)


def test_surface_verifier_rejects_changed_science():
    payload = minimal_surface()
    payload["contexts"][CONTEXT_ORDER[0]][0]["utility"] = 0.25
    with pytest.raises(ValueError, match="scientific payload"):
        verify_surface(payload)


def test_surface_verifier_checks_envelope_when_present():
    payload = minimal_surface()
    payload["self_sha256"] = "not-a-seal"
    with pytest.raises(ValueError, match="envelope"):
        verify_surface(payload)

