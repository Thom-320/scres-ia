#!/usr/bin/env python3
"""Build the exact finite policy-family certificate for the wartime atlas."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator


BUFFER_FRACTIONS = (0.0, 0.125, 0.25, 0.375, 0.5, 1.0)
SHIFT_LEVELS = (1, 2, 3)
ENTRY_OFFSETS = (-168, -72, -24, 0, 24, 72)
EXIT_OFFSETS = (24, 72, 168)


def posture_label(fraction: float, shifts: int) -> str:
    return f"f{fraction:g}_S{int(shifts)}"


def posture_resource(fraction: float, shifts: int) -> float:
    return 0.5 * float(fraction) + 0.5 * ((int(shifts) - 1) / 2.0)


def postures() -> list[dict[str, Any]]:
    return [
        {
            "label": posture_label(fraction, shifts),
            "buffer_fraction": fraction,
            "shifts": shifts,
            "nominal_resource": posture_resource(fraction, shifts),
        }
        for fraction in BUFFER_FRACTIONS
        for shifts in SHIFT_LEVELS
    ]


def iter_policy_templates() -> Iterator[dict[str, Any]]:
    rows = postures()
    for posture in rows:
        yield {
            "policy_id": f"constant::{posture['label']}",
            "family": "constant",
            "low": posture["label"],
            "high": posture["label"],
            "payload": None,
            "scheduled_resource_entitlement": posture["nominal_resource"],
        }

    # Every two-posture 8-week binary calendar.  Unordered posture pairs are
    # sufficient because the 256 bit patterns include each complement.
    for low_index, low in enumerate(rows):
        for high in rows[low_index + 1 :]:
            for calendar in range(256):
                yield {
                    "policy_id": (
                        f"open_loop::{low['label']}::{high['label']}::{calendar:03d}"
                    ),
                    "family": "open_loop_8week_periodic",
                    "low": low["label"],
                    "high": high["label"],
                    "payload": calendar,
                    "scheduled_resource_entitlement": max(
                        low["nominal_resource"], high["nominal_resource"]
                    ),
                }

    for low in rows:
        for high in rows:
            if low["label"] == high["label"]:
                continue
            entitlement = max(low["nominal_resource"], high["nominal_resource"])
            for entry in ENTRY_OFFSETS:
                for exit_offset in EXIT_OFFSETS:
                    payload = {
                        "entry_offset_hours": entry,
                        "exit_offset_hours": exit_offset,
                    }
                    suffix = f"entry{entry:+g}::exit{exit_offset:+g}"
                    for family in ("restricted_privileged", "weekly_privileged"):
                        yield {
                            "policy_id": (
                                f"{family}::{low['label']}::{high['label']}::{suffix}"
                            ),
                            "family": family,
                            "low": low["label"],
                            "high": high["label"],
                            "payload": payload,
                            "scheduled_resource_entitlement": entitlement,
                        }


def build_manifest() -> dict[str, Any]:
    digest = hashlib.sha256()
    counts: dict[str, int] = {}
    total = 0
    for row in iter_policy_templates():
        encoded = json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
        digest.update(encoded)
        digest.update(b"\n")
        counts[row["family"]] = counts.get(row["family"], 0) + 1
        total += 1
    return {
        "schema_version": "war_stress_policy_manifest_v1",
        "status": "FROZEN_BEFORE_SCIENTIFIC_SEED_ACCESS",
        "postures": postures(),
        "entry_offsets_hours": list(ENTRY_OFFSETS),
        "exit_offsets_hours": list(EXIT_OFFSETS),
        "periodic_calendar_bits": 8,
        "counts_by_family": counts,
        "total_policy_templates": total,
        "ordered_template_rows_sha256": digest.hexdigest(),
        "enumeration_rule": (
            "18 constants; every unordered posture pair x all 256 binary 8-week "
            "calendars; every ordered distinct posture pair x 6 entry x 3 exit "
            "offsets for both restricted and weekly-boundary clairvoyant families"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "research/paper2_exhaustive_search/war_stress_policy_manifest_20260716.json"
        ),
    )
    args = parser.parse_args()
    payload = build_manifest()
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
