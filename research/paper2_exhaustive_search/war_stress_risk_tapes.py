"""Policy-independent risk tapes for the frozen wartime timing atlas.

The tape generator starts from fixed per-risk base uniforms.  A configuration
transforms those uniforms through its frequency/impact multipliers, so every
configuration has the same *base-stream* hash while its realized event-tape
hash may legitimately differ.  Policies within one configuration replay the
same materialized event tape exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import binom


MASK_RISKS: dict[str, tuple[str, ...]] = {
    "LOC_SURGE": ("R22", "R24"),
    "THEATER_CAPACITY_SURGE": ("R21", "R23", "R24"),
    "PRODUCTION_QUALITY_SURGE": ("R11", "R14", "R24"),
}
COUPLING_OFFSETS: dict[str, float | None] = {
    "independent": None,
    "disruption_leads_surge_72h": -72.0,
    "coincident": 0.0,
    "surge_leads_disruption_72h": 72.0,
}
UNIFORM_RISKS: dict[str, tuple[int, int, float, tuple[int, ...]]] = {
    # risk: (a, b, recovery mean, affected operations)
    "R11": (1, 168, 2.0, (5, 6)),
    "R21": (1, 16_128, 120.0, (3, 5, 6, 7, 9)),
    "R22": (1, 4_032, 24.0, (4, 8, 10, 12)),
    "R23": (1, 8_064, 120.0, (11,)),
    "R24": (1, 672, 0.0, (13,)),
}
BASE_STREAM_LENGTH = 2_048
GENERATOR_VERSION = "war_stress_policy_independent_tape_v1"


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _risk_seed(tape_id: int, risk_id: str) -> int:
    digest = hashlib.sha256(
        f"{GENERATOR_VERSION}:{int(tape_id)}:{risk_id}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _base_stream(tape_id: int, risk_id: str) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(_risk_seed(tape_id, risk_id))
    return {
        "occurrence": rng.random(BASE_STREAM_LENGTH),
        "impact": rng.random((BASE_STREAM_LENGTH, 5)),
        "target": rng.random(BASE_STREAM_LENGTH),
        "magnitude": rng.random(BASE_STREAM_LENGTH),
    }


def base_stream_sha256(tape_id: int) -> str:
    digest = hashlib.sha256()
    digest.update(GENERATOR_VERSION.encode())
    digest.update(str(int(tape_id)).encode())
    for risk_id in sorted({risk for values in MASK_RISKS.values() for risk in values}):
        digest.update(risk_id.encode())
        for name, values in sorted(_base_stream(tape_id, risk_id).items()):
            digest.update(name.encode())
            digest.update(np.asarray(values, dtype="<f8").tobytes())
    return digest.hexdigest()


def _multiplier(
    physical_multipliers: Mapping[str, float], prefix: str, risk_id: str
) -> float:
    return max(1e-9, float(physical_multipliers.get(f"{prefix}_{risk_id}", 1.0)))


def _duration(mean: float, psi: float, uniform: float) -> float:
    return max(1.0, -float(mean) * float(psi) * math.log(max(1e-15, 1.0 - uniform)))


def _event(
    risk_id: str,
    start: float,
    duration: float,
    affected_ops: Sequence[int],
    *,
    magnitude: float = 1.0,
    unit: str = "incidents",
) -> dict[str, Any]:
    return {
        "risk_id": risk_id,
        "start_time": float(start),
        "end_time": float(start + max(0.0, duration)),
        "duration": float(max(0.0, duration)),
        "affected_ops": [int(value) for value in affected_ops],
        "description": f"{GENERATOR_VERSION}:{risk_id}",
        "magnitude": float(magnitude),
        "unit": unit,
    }


def _uniform_onsets(
    risk_id: str,
    *,
    horizon_hours: float,
    start_hour: float,
    phi: float,
    stream: Mapping[str, np.ndarray],
) -> list[float]:
    a, native_b, _mean, _ops = UNIFORM_RISKS[risk_id]
    b = max(a, int(round(native_b / max(phi, 1e-9))))
    onsets: list[float] = []
    window_start = float(start_hour)
    index = 0
    while window_start < horizon_hours:
        if index >= BASE_STREAM_LENGTH:
            raise RuntimeError(f"base stream exhausted for {risk_id}")
        u = float(stream["occurrence"][index])
        delay = a + int(math.floor(u * (b - a + 1)))
        onset = window_start + min(delay, b)
        if onset < horizon_hours:
            onsets.append(float(onset))
        window_start += float(b)
        index += 1
    return onsets


def _risk_events_at_onset(
    risk_id: str,
    *,
    onset: float,
    index: int,
    physical_multipliers: Mapping[str, float],
    stream: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    if risk_id == "R24":
        psi = _multiplier(physical_multipliers, "psi", risk_id)
        lo = max(0, int(round(2_400 * psi)))
        hi = max(lo, int(round(2_600 * psi)))
        u = float(stream["magnitude"][index])
        surge = lo + int(math.floor(u * (hi - lo + 1)))
        return [_event("R24", onset, 0.0, (13,), magnitude=surge, unit="rations")]
    if risk_id == "R14":
        phi = _multiplier(physical_multipliers, "phi", risk_id)
        probability = min(0.98, 0.03 * phi)
        defects = int(binom.ppf(float(stream["magnitude"][index]), 2_564, probability))
        return (
            [_event("R14", onset, 0.0, (7,), magnitude=defects, unit="defective_products")]
            if defects > 0
            else []
        )

    _a, _b, mean, affected = UNIFORM_RISKS[risk_id]
    psi = _multiplier(physical_multipliers, "psi", risk_id)
    if risk_id == "R21":
        return [
            _event(
                risk_id,
                onset,
                _duration(mean, psi, float(stream["impact"][index, op_index])),
                (op_id,),
            )
            for op_index, op_id in enumerate(affected)
        ]
    target_index = min(
        len(affected) - 1,
        int(math.floor(float(stream["target"][index]) * len(affected))),
    )
    return [
        _event(
            risk_id,
            onset,
            _duration(mean, psi, float(stream["impact"][index, 0])),
            (affected[target_index],),
        )
    ]


@dataclass(frozen=True)
class BuiltRiskTape:
    config_id: str
    tape_id: int
    mask: str
    coupling: str
    events: tuple[dict[str, Any], ...]
    base_stream_sha256: str
    event_tape_sha256: str
    r3_event_count: int


def build_risk_tape(
    configuration: Mapping[str, Any],
    *,
    tape_id: int,
    horizon_hours: float,
    start_hour: float = 0.0,
) -> BuiltRiskTape:
    """Materialize one exact policy-independent tape from a manifest row."""
    mask = str(configuration["mask"])
    coupling = str(configuration["coupling"])
    if mask not in MASK_RISKS or coupling not in COUPLING_OFFSETS:
        raise ValueError("unknown mask/coupling")
    multipliers = {
        str(key): float(value)
        for key, value in configuration["physical_multipliers"].items()
    }
    events: list[dict[str, Any]] = []
    streams = {risk: _base_stream(tape_id, risk) for risk in MASK_RISKS[mask]}

    phi_r24 = _multiplier(multipliers, "phi", "R24")
    r24_onsets = _uniform_onsets(
        "R24",
        horizon_hours=horizon_hours,
        start_hour=start_hour,
        phi=phi_r24,
        stream=streams["R24"],
    )
    for index, onset in enumerate(r24_onsets):
        events.extend(
            _risk_events_at_onset(
                "R24",
                onset=onset,
                index=index,
                physical_multipliers=multipliers,
                stream=streams["R24"],
            )
        )

    offset = COUPLING_OFFSETS[coupling]
    for risk_id in MASK_RISKS[mask]:
        if risk_id == "R24":
            continue
        stream = streams[risk_id]
        if offset is None:
            if risk_id == "R14":
                onsets = list(np.arange(start_hour + 24.0, horizon_hours, 24.0))
            else:
                onsets = _uniform_onsets(
                    risk_id,
                    horizon_hours=horizon_hours,
                    start_hour=start_hour,
                    phi=_multiplier(multipliers, "phi", risk_id),
                    stream=stream,
                )
        else:
            onsets = [onset + offset for onset in r24_onsets]
        for index, onset in enumerate(onsets):
            if onset < 0.0 or onset >= horizon_hours:
                continue
            events.extend(
                _risk_events_at_onset(
                    risk_id,
                    onset=onset,
                    index=index,
                    physical_multipliers=multipliers,
                    stream=stream,
                )
            )

    ordered = tuple(
        sorted(events, key=lambda row: (float(row["start_time"]), str(row["risk_id"]), tuple(row["affected_ops"])))
    )
    if any(row["risk_id"] == "R3" for row in ordered):
        raise AssertionError("R3 is forbidden in the primary wartime tape")
    return BuiltRiskTape(
        config_id=str(configuration["config_id"]),
        tape_id=int(tape_id),
        mask=mask,
        coupling=coupling,
        events=ordered,
        base_stream_sha256=base_stream_sha256(tape_id),
        event_tape_sha256=_canonical_sha256(ordered),
        r3_event_count=0,
    )
