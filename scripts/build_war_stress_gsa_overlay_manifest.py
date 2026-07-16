#!/usr/bin/env python3
"""Build the exact preregistered Morris and QMC configuration manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.stats import qmc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.paper2_exhaustive_search.war_risk_gsa_v2 import (  # noqa: E402
    MASK_FACTORS,
    FactorSpace,
    salib_morris_design,
)


COUPLINGS = (
    "independent",
    "disruption_leads_surge_72h",
    "coincident",
    "surge_leads_disruption_72h",
)
MASK_SEEDS = {
    "LOC_SURGE": 7471001,
    "THEATER_CAPACITY_SURGE": 7471002,
    "PRODUCTION_QUALITY_SURGE": 7471003,
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _point_payload(space: FactorSpace, values: np.ndarray) -> dict[str, float]:
    return {name: float(value) for name, value in zip(space.names, values, strict=True)}


def build_manifest() -> dict[str, Any]:
    morris_rows: list[dict[str, Any]] = []
    qmc_rows: list[dict[str, Any]] = []
    for mask in MASK_FACTORS:
        shared_space = FactorSpace.for_mask(mask, COUPLINGS[0])
        design = salib_morris_design(
            shared_space,
            candidate_trajectories=20,
            selected_trajectories=10,
            levels=8,
            seed=MASK_SEEDS[mask],
        )
        qmc_sampler = qmc.Sobol(
            d=len(shared_space.names),
            scramble=True,
            seed=MASK_SEEDS[mask] + 100,
        )
        qmc_unit = qmc_sampler.random_base2(m=7)
        lo = np.asarray([bounds[0] for bounds in shared_space.log2_bounds], dtype=float)
        hi = np.asarray([bounds[1] for bounds in shared_space.log2_bounds], dtype=float)
        qmc_log2 = qmc.scale(qmc_unit, lo, hi)
        qmc_physical = shared_space.physical(qmc_log2)
        for coupling in COUPLINGS:
            for index, (log2_point, physical_point) in enumerate(
                zip(design.log2_points, design.physical_points, strict=True)
            ):
                trajectory = index // (len(shared_space.names) + 1)
                step = index % (len(shared_space.names) + 1)
                morris_rows.append(
                    {
                        "config_id": f"morris::{mask}::{coupling}::{trajectory:02d}::{step:02d}",
                        "mask": mask,
                        "coupling": coupling,
                        "trajectory": trajectory,
                        "step": step,
                        "log2_factors": _point_payload(shared_space, log2_point),
                        "physical_multipliers": _point_payload(shared_space, physical_point),
                    }
                )
            for index, (log2_point, physical_point) in enumerate(
                zip(qmc_log2, qmc_physical, strict=True)
            ):
                qmc_rows.append(
                    {
                        "config_id": f"qmc::{mask}::{coupling}::{index:03d}",
                        "mask": mask,
                        "coupling": coupling,
                        "index": index,
                        "log2_factors": _point_payload(shared_space, log2_point),
                        "physical_multipliers": _point_payload(shared_space, physical_point),
                    }
                )
    cell_payload = json.dumps(
        {"morris": morris_rows, "qmc_pool": qmc_rows},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return {
        "schema_version": "war_stress_gsa_overlay_manifest_v1",
        "status": "FROZEN_BEFORE_SCIENTIFIC_SEED_ACCESS",
        "factor_parameterization": "independent uniform sampling in log2 multiplier space within each fixed mask/coupling stratum",
        "morris": {
            "candidate_trajectories": 20,
            "selected_trajectories": 10,
            "levels": 8,
            "configuration_count": len(morris_rows),
            "rows": morris_rows,
        },
        "qmc_pool": {
            "design": "scrambled Sobol space-filling pool, not Saltelli cross-sampling",
            "points_per_stratum": 128,
            "configuration_count": len(qmc_rows),
            "maximum_strata_executed": 3,
            "rows": qmc_rows,
        },
        "configuration_payload_sha256": _sha256_bytes(cell_payload),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/paper2_exhaustive_search/war_stress_gsa_overlay_manifest_20260716.json"),
    )
    args = parser.parse_args()
    payload = build_manifest()
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "morris_configurations": payload["morris"]["configuration_count"],
                "qmc_pool_configurations": payload["qmc_pool"]["configuration_count"],
                "configuration_payload_sha256": payload["configuration_payload_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
