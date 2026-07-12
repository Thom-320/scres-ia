#!/usr/bin/env python3
"""Materialize the frozen Program F phase design before any screen tape."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.stats import qmc


def level(value: float, options: list):
    return options[min(int(value * len(options)), len(options) - 1)]


def main() -> int:
    sampler = qmc.LatinHypercube(d=6, seed=2026071206, optimization="random-cd")
    sample = sampler.random(24)
    efficacy = ["low", "base", "high"]
    accuracy = [0.60, 0.75, 0.90]
    dwell = [[3, 5], [4, 8], [6, 10]]
    budget = [1, 2, 3]
    amplitude = ["current_all_contexts", "dominant_increased_background_current"]
    commitment = [1, 2]
    cells = []
    for i, row in enumerate(sample, start=1):
        cells.append({
            "cell_id": f"FSC-{i:02d}",
            "efficacy_level": level(row[0], efficacy),
            "signal_accuracy": level(row[1], accuracy),
            "context_dwell_weeks": level(row[2], dwell),
            "budget_tokens": level(row[3], budget),
            "risk_amplitude": level(row[4], amplitude),
            "minimum_commitment_weeks": level(row[5], commitment),
            "screen_seed_start": 970001 + (i - 1) * 12,
            "screen_seed_end": 970001 + i * 12 - 1,
            "confirmatory_v1_admissible": level(row[3], budget) == 2,
        })
    payload = {
        "contract_id": "program_f_phase_diagram_v1", "design_seed": 2026071206,
        "n_cells": len(cells), "cells": cells,
    }
    payload["design_sha256"] = sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output = Path("results/program_f/screen")
    output.mkdir(parents=True, exist_ok=True)
    (output / "design.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "design_sha256": payload["design_sha256"],
        "budget_counts": {str(value): sum(cell["budget_tokens"] == value for cell in cells) for value in budget},
        "admissible_cells": sum(cell["confirmatory_v1_admissible"] for cell in cells),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
