from __future__ import annotations

from scripts.build_transfer_confirmation_cache_v1 import (
    BASE_CONFIGS,
    EXT_CONFIGS,
    _project_base,
)


def test_confirmation_projection_is_the_frozen_288_order():
    payload = {
        "context": "R1r",
        "seed": 8_100_001,
        "horizon_hours": 52.0,
        "module_manifest": {"modules": {"physics": "frozen"}},
        "cells": [{"value": float(i), "drivers": [], "panel": {}}
                  for i in range(len(EXT_CONFIGS))],
    }
    projected = _project_base(payload, contract="contract")

    assert projected["grid_id"] == "wrap288_v1"
    assert len(projected["cells"]) == len(BASE_CONFIGS) == 288
    assert [row["value"] for row in projected["cells"]] == [float(i * 16)
                                                              for i in range(len(BASE_CONFIGS))]

