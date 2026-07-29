#!/usr/bin/env python3
"""Build separated native and wartime Morris designs for Program S v1.1."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FREEZE = json.loads(
    (
        ROOT
        / "research/paper2_exhaustive_search/program_s_morris_design_freeze_v1_1.json"
    ).read_text()
)
CONTRACT = json.loads(
    (ROOT / "contracts/program_s_product_mix_risk_interaction_gsa_v1.json").read_text()
)
OUT_NATIVE = ROOT / "research/paper2_exhaustive_search/program_s_native_morris_design_v1_1.json"
OUT_WARTIME = ROOT / "research/paper2_exhaustive_search/program_s_wartime_morris_design_v1_1.json"


RANGES = {
    "uniform_frequency": (1.0, 4.0),
    "duration_impact": (1.0, 2.0),
    "r14_defect_probability": (1.0, 2.0),
    "r24_frequency": (1.0, 4.0),
    "r24_quantity": (1.0, 2.0),
}
COUPLINGS = (
    "disruption_leads_r24_72h",
    "coincident",
    "r24_leads_disruption_72h",
)


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def factors(mask: str) -> tuple[str, ...]:
    values = [
        "uniform_frequency",
        "duration_impact",
        "r24_frequency",
        "r24_quantity",
    ]
    if mask == "PRODUCTION_QUALITY_SURGE":
        values.insert(2, "r14_defect_probability")
    return tuple(values)


def candidate_trajectory(rng: np.random.Generator, names: tuple[str, ...]) -> np.ndarray:
    levels = int(FREEZE["levels"])
    delta = (levels // 2) / (levels - 1)
    grid = np.linspace(0.0, 1.0, levels)
    direction = rng.choice((-1.0, 1.0), size=len(names))
    start = np.empty(len(names), dtype=float)
    for index, sign in enumerate(direction):
        allowed = grid[grid <= 1.0 - delta + 1e-12] if sign > 0 else grid[grid >= delta - 1e-12]
        start[index] = float(rng.choice(allowed))
    order = rng.permutation(len(names))
    points = [start.copy()]
    current = start.copy()
    for index in order:
        current = current.copy()
        current[index] += direction[index] * delta
        points.append(current)
    return np.asarray(points)


def trajectory_distance(left: np.ndarray, right: np.ndarray) -> float:
    distances = np.linalg.norm(left[:, None, :] - right[None, :, :], axis=2)
    return float(np.sqrt(np.sum(distances**2)))


def optimize(candidates: list[np.ndarray], count: int) -> list[int]:
    n = len(candidates)
    matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i, j] = matrix[j, i] = trajectory_distance(candidates[i], candidates[j])
    selected = [int(np.argmax(matrix.sum(axis=1)))]
    while len(selected) < int(count):
        remaining = [index for index in range(n) if index not in selected]
        next_index = max(
            remaining,
            key=lambda index: (min(matrix[index, chosen] for chosen in selected), -index),
        )
        selected.append(int(next_index))
    return selected


def physical_value(name: str, normalized: float) -> float:
    low, high = RANGES[name]
    return float(2 ** (math.log2(low) + float(normalized) * (math.log2(high) - math.log2(low))))


def physical_point(mask: str, names: tuple[str, ...], point: np.ndarray) -> dict[str, Any]:
    values = {name: physical_value(name, point[index]) for index, name in enumerate(names)}
    values["baseline_capacity"] = 1.0
    risks = CONTRACT["physical_masks"][mask]
    phi = {
        risk: (
            values["r24_frequency"]
            if risk == "R24"
            else 1.0
            if risk == "R14"
            else values["uniform_frequency"]
        )
        for risk in risks
    }
    psi = {
        risk: (
            values["r24_quantity"]
            if risk == "R24"
            else 1.0
            if risk == "R14"
            else values["duration_impact"]
        )
        for risk in risks
    }
    # R23 remains physically live but is frozen as a non-selectable negative
    # control for the mix-only action under amendment v1.1.
    if "R23" in phi:
        phi["R23"] = 1.0
        psi["R23"] = 1.0
    return {
        "normalized": {name: float(point[index]) for index, name in enumerate(names)},
        "physical": values,
        "phi_by_risk": phi,
        "psi_by_risk": psi,
        "r14_probability_multiplier": values.get("r14_defect_probability", 1.0),
        "baseline_capacity_multiplier": 1.0,
    }


def build(stratum: str) -> dict[str, Any]:
    rng = np.random.default_rng(int(FREEZE["design_rng_seed"]))
    groups: list[dict[str, Any]] = []
    candidate_count = int(FREEZE["candidate_trajectories_per_mask_stratum"])
    selected_count = int(FREEZE["selected_optimized_trajectories_per_mask_stratum"])
    for mask in CONTRACT["physical_masks"]:
        names = factors(mask)
        for current_stratum in (stratum,):
            candidates = [candidate_trajectory(rng, names) for _ in range(candidate_count)]
            trajectories: list[dict[str, Any]] = []
            if current_stratum == "THESIS_NATIVE_INDEPENDENT":
                selected = optimize(candidates, selected_count)
                coupling_by_index = {index: "independent" for index in selected}
            else:
                selected = []
                coupling_by_index = {}
                quotas = (4, 3, 3)
                for coupling_index, (coupling, quota) in enumerate(zip(COUPLINGS, quotas)):
                    pool = [index for index in range(candidate_count) if index % 3 == coupling_index]
                    local = optimize([candidates[index] for index in pool], quota)
                    chosen = [pool[index] for index in local]
                    selected.extend(chosen)
                    coupling_by_index.update({index: coupling for index in chosen})
            for sequence, candidate_index in enumerate(selected):
                points = [
                    physical_point(mask, names, point)
                    for point in candidates[candidate_index]
                ]
                trajectories.append(
                    {
                        "trajectory_id": f"{mask}__{current_stratum}__{sequence:02d}",
                        "candidate_index": int(candidate_index),
                        "coupling": coupling_by_index[candidate_index],
                        "factor_order_inferred_from_steps": True,
                        "points": points,
                    }
                )
            groups.append(
                {
                    "mask": mask,
                    "stratum": current_stratum,
                    "factor_names": list(names),
                    "candidate_count": candidate_count,
                    "selected_count": len(trajectories),
                    "trajectories": trajectories,
                    "mandatory_capacity_1_anchors": [
                        {
                            "coupling": coupling,
                            "physical": {
                                "uniform_frequency": 1.0,
                                "duration_impact": 1.0,
                                **(
                                    {"r14_defect_probability": 1.0}
                                    if "r14_defect_probability" in names
                                    else {}
                                ),
                                "r24_frequency": 1.0,
                                "r24_quantity": 1.0,
                                "baseline_capacity": 1.0,
                            },
                            "phi_by_risk": {risk: 1.0 for risk in CONTRACT["physical_masks"][mask]},
                            "psi_by_risk": {risk: 1.0 for risk in CONTRACT["physical_masks"][mask]},
                            "r14_probability_multiplier": 1.0,
                            "baseline_capacity_multiplier": 1.0,
                        }
                        for coupling in (
                            ("independent",)
                            if current_stratum == "THESIS_NATIVE_INDEPENDENT"
                            else COUPLINGS
                        )
                    ],
                }
            )
    payload = {
        "schema_version": "program_s_morris_design_v1_1",
        "execution_family": "S_NATIVE" if stratum == "THESIS_NATIVE_INDEPENDENT" else "S_WARTIME",
        "promotion_authorized": stratum == "THESIS_NATIVE_INDEPENDENT",
        "freeze_sha256": digest(FREEZE),
        "design_rng_seed": FREEZE["design_rng_seed"],
        "groups": groups,
        "product_cells": FREEZE["product_cells"],
        "scientific_seed_block": FREEZE["native"]["seed_block"] if stratum == "THESIS_NATIVE_INDEPENDENT" else None,
        "scientific_seed_block_opened": False,
    }
    payload["design_sha256"] = digest(payload)
    return payload


def main() -> int:
    native = build("THESIS_NATIVE_INDEPENDENT")
    wartime = build("RESEARCHER_WARTIME_COUPLED")
    OUT_NATIVE.write_text(json.dumps(native, indent=2, sort_keys=True) + "\n")
    OUT_WARTIME.write_text(json.dumps(wartime, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "native_output": str(OUT_NATIVE),
        "wartime_output": str(OUT_WARTIME),
        "native_design_sha256": native["design_sha256"],
        "wartime_design_sha256": wartime["design_sha256"],
        "native_groups": len(native["groups"]),
        "native_trajectories": sum(len(group["trajectories"]) for group in native["groups"]),
        "native_points": sum(len(row["points"]) for group in native["groups"] for row in group["trajectories"]),
        "wartime_groups": len(wartime["groups"]),
        "wartime_trajectories": sum(len(group["trajectories"]) for group in wartime["groups"]),
        "wartime_points": sum(len(row["points"]) for group in wartime["groups"] for row in group["trajectories"]),
        "scientific_seed_block_opened": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
