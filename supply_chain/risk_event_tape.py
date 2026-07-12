"""Risk-event tape helpers for causal Track B audits.

The normal simulator samples risks from RNG streams.  Gate-v2 prevention audits
need a stronger contract: discover a realized risk calendar once, serialize it,
and replay edited variants (event-on/off, anchor deletion, synthetic insertion)
without changing the headline Track B environment defaults.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


RISK_EVENT_TAPE_FIELDS: tuple[str, ...] = (
    "risk_id",
    "start_time",
    "end_time",
    "duration",
    "affected_ops",
    "description",
    "magnitude",
    "unit",
)


def event_to_record(event: Any) -> dict[str, Any]:
    return {
        "risk_id": str(getattr(event, "risk_id")),
        "start_time": float(getattr(event, "start_time")),
        "end_time": float(getattr(event, "end_time")),
        "duration": float(getattr(event, "duration")),
        "affected_ops": [int(op) for op in getattr(event, "affected_ops", [])],
        "description": str(getattr(event, "description", "") or ""),
        "magnitude": float(getattr(event, "magnitude", 1.0) or 1.0),
        "unit": str(getattr(event, "unit", "incidents") or "incidents"),
    }


def normalize_event_record(record: dict[str, Any]) -> dict[str, Any]:
    affected = record.get("affected_ops", [])
    if isinstance(affected, str):
        try:
            affected = json.loads(affected)
        except json.JSONDecodeError:
            affected = [x for x in affected.split(";") if x]
    return {
        "risk_id": str(record["risk_id"]),
        "start_time": float(record["start_time"]),
        "end_time": float(record.get("end_time", record["start_time"])),
        "duration": float(record.get("duration", 0.0)),
        "affected_ops": [int(op) for op in affected],
        "description": str(record.get("description", "") or ""),
        "magnitude": float(record.get("magnitude", 1.0) or 1.0),
        "unit": str(record.get("unit", "incidents") or "incidents"),
    }


def events_to_records(events: Iterable[Any]) -> list[dict[str, Any]]:
    return [event_to_record(event) for event in events]


def load_risk_event_tape(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text())
        if isinstance(payload, dict):
            payload = payload.get("events", [])
        return [normalize_event_record(dict(row)) for row in payload]
    with path.open(newline="") as fh:
        return [normalize_event_record(dict(row)) for row in csv.DictReader(fh)]


def save_risk_event_tape(events: Iterable[Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        normalize_event_record(row) if isinstance(row, dict) else event_to_record(row)
        for row in events
    ]
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps({"events": rows}, indent=2, sort_keys=True))
        return
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RISK_EVENT_TAPE_FIELDS)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["affected_ops"] = json.dumps(out["affected_ops"])
            writer.writerow(out)


def remove_event_at_index(events: list[dict[str, Any]], index: int) -> list[dict[str, Any]]:
    return [dict(event) for i, event in enumerate(events) if i != index]


def insert_event(events: list[dict[str, Any]], event: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [dict(row) for row in events]
    rows.append(normalize_event_record(event))
    return sorted(rows, key=lambda row: (float(row["start_time"]), str(row["risk_id"])))


def cluster_events(
    events: Iterable[dict[str, Any]],
    *,
    risk_id: str,
    max_gap_hours: float,
) -> list[dict[str, Any]]:
    rows = sorted(
        [normalize_event_record(row) for row in events if str(row["risk_id"]) == risk_id],
        key=lambda row: float(row["start_time"]),
    )
    clusters: list[dict[str, Any]] = []
    for row in rows:
        start = float(row["start_time"])
        end = float(row["end_time"])
        if not clusters or start - float(clusters[-1]["end_time"]) > max_gap_hours:
            clusters.append(
                {
                    "risk_id": risk_id,
                    "start_time": start,
                    "end_time": end,
                    "duration": max(0.0, end - start),
                    "affected_ops": list(row["affected_ops"]),
                    "description": "cluster",
                    "magnitude": float(row.get("magnitude", 1.0)),
                    "unit": "cluster",
                    "member_count": 1,
                }
            )
            continue
        cluster = clusters[-1]
        cluster["end_time"] = max(float(cluster["end_time"]), end)
        cluster["duration"] = max(0.0, float(cluster["end_time"]) - float(cluster["start_time"]))
        cluster["affected_ops"] = sorted(set(cluster["affected_ops"]) | set(row["affected_ops"]))
        cluster["magnitude"] = float(cluster["magnitude"]) + float(row.get("magnitude", 1.0))
        cluster["member_count"] = int(cluster.get("member_count", 1)) + 1
    return clusters

