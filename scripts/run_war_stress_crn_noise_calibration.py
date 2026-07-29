#!/usr/bin/env python3
"""Non-scientific DES fixture for the CRN two-way noise decomposition."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.paper2_exhaustive_search.restricted_timing_oracle import (  # noqa: E402
    ScheduleSpec,
    evaluate_schedule,
    posture_from_label,
)
from research.paper2_exhaustive_search.war_risk_gsa_v2 import (  # noqa: E402
    StochasticTimingResponse,
    audit_crn_noise,
)
from research.paper2_exhaustive_search.war_stress_risk_tapes import (  # noqa: E402
    MASK_RISKS,
    build_risk_tape,
)
from scripts.run_war_stress_gsa_executor import _probe_treatment_start  # noqa: E402


CALIBRATION_TAPES = (94_700_101, 94_700_102, 94_700_103, 94_700_104)
CALIBRATION_CONFIGS = (
    "morris::LOC_SURGE::independent::00::00",
    "morris::LOC_SURGE::independent::00::01",
    "morris::LOC_SURGE::coincident::00::00",
    "morris::LOC_SURGE::coincident::00::01",
)


def _configuration(manifest: dict[str, Any], config_id: str) -> dict[str, Any]:
    rows = manifest["morris"]["rows"]
    return next(dict(row) for row in rows if row["config_id"] == config_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "research/paper2_exhaustive_search/war_stress_gsa_overlay_manifest_20260716.json"
        ),
    )
    parser.add_argument("--max-daily-steps", type=int, default=28)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "research/paper2_exhaustive_search/war_stress_crn_noise_calibration_20260716.json"
        ),
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    low = posture_from_label("f0_S1")
    responses: list[StochasticTimingResponse] = []
    raw_rows: list[dict[str, Any]] = []
    for config_id in CALIBRATION_CONFIGS:
        config = _configuration(manifest, config_id)
        values: list[float] = []
        event_hashes: list[str] = []
        base_hashes: list[str] = []
        for tape_id in CALIBRATION_TAPES:
            start = _probe_treatment_start(
                seed=tape_id, max_daily_steps=int(args.max_daily_steps)
            )
            tape = build_risk_tape(
                config,
                tape_id=tape_id,
                start_hour=start,
                horizon_hours=start + int(args.max_daily_steps) * 24.0,
            )
            metrics = evaluate_schedule(
                seed=tape_id,
                risk_overrides={},
                low=low,
                high=low,
                spec=ScheduleSpec("constant", "noise-calibration-constant", False),
                max_daily_steps=int(args.max_daily_steps),
                known_risk_events=(),
                risk_event_tape=tape.events,
                enabled_risks=MASK_RISKS[tape.mask],
            )
            value = float(metrics["ret_excel"])
            values.append(value)
            event_hashes.append(tape.event_tape_sha256)
            base_hashes.append(tape.base_stream_sha256)
            raw_rows.append(
                {
                    "config_id": config_id,
                    "tape_id": tape_id,
                    "ret_excel": value,
                    "event_tape_sha256": tape.event_tape_sha256,
                    "base_stream_sha256": tape.base_stream_sha256,
                    "r3_event_count": tape.r3_event_count,
                }
            )
        vector = np.asarray(values, dtype=float)
        responses.append(
            StochasticTimingResponse(
                config_id=config_id,
                tape_ids=CALIBRATION_TAPES,
                event_tape_sha256s=tuple(event_hashes),
                exogenous_base_stream_sha256s=tuple(base_hashes),
                deltas=vector,
                selected_policy_ids=("constant::f0_S1",) * len(vector),
                mean=float(np.mean(vector)),
                standard_error=float(np.std(vector, ddof=1) / np.sqrt(len(vector))),
                favorable_tapes=0,
            )
        )
    audit = audit_crn_noise(responses)
    audit_payload = asdict(audit)
    audit_payload["residuals"] = audit.residuals.tolist()
    payload = {
        "schema_version": "war_stress_crn_noise_calibration_v1",
        "status": "NONSCIENTIFIC_DES_CRN_DECOMPOSITION_FIXTURE_PASS",
        "scientific_seeds_opened": False,
        "tape_ids": list(CALIBRATION_TAPES),
        "configuration_ids": list(CALIBRATION_CONFIGS),
        "max_daily_steps": int(args.max_daily_steps),
        "raw_rows": raw_rows,
        "crn_two_way_audit": audit_payload,
        "claim_boundary": (
            "This validates populated DES fields and the CRN decomposition on a "
            "short fixed-policy fixture. It does not calibrate an interaction or "
            "additivity threshold for H_timing_safe. Those claims remain prohibited."
        ),
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "raw_rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
