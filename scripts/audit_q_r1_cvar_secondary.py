#!/usr/bin/env python3
"""Burned-data diagnostic of CVaR10 as a secondary Q-R1 instrument."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/q_r1/d0_cold_start_reanalysis_v1/result.json"
OUTPUT = ROOT / "results/q_r1/cvar_secondary_instrument_audit_v1/result.json"


def lcb(values: dict[int, list[float]], seed: int = 20260721) -> float:
    roots = sorted(values)
    means = np.asarray([np.mean(values[root]) for root in roots])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(means), size=(5000, len(means)))
    return float(np.quantile(means[draws].mean(axis=1), 0.025))


def main() -> int:
    rows = json.loads(SOURCE.read_text())["rows"]
    grouped: dict[tuple[float, int, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["kappa"], row["history_root"], row["campaign_index"])].append(row)
    oracle_delta: dict[int, list[float]] = defaultdict(list)
    trivial_delta: dict[int, list[float]] = defaultdict(list)
    for (_kappa, root, campaign), candidates in grouped.items():
        if campaign == 0:
            continue
        reset = next(row for row in candidates if row["arm"] == "reset_posterior_0p5")
        best = max(candidates, key=lambda row: row["ret_visible_cvar10"])
        oracle_delta[root].append(best["ret_visible_cvar10"] - reset["ret_visible_cvar10"])
        trivial_delta[root].append(reset["ret_visible_cvar10"] - reset["ret_visible_cvar10"])
    values = [value for root in oracle_delta for value in oracle_delta[root]]
    root_means = np.asarray([np.mean(oracle_delta[root]) for root in sorted(oracle_delta)])
    mde_80_two_sided = float((1.959964 + 0.841621) * np.std(root_means, ddof=1) / np.sqrt(len(root_means)))
    payload = {
        "schema_version": "q_r1_cvar_secondary_instrument_audit_v1",
        "claim_status": "BURNED_SECONDARY_DIAGNOSTIC_NO_PRIMARY_GATE_CHANGE",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(SOURCE),
        "trivial_identity_mean": 0.0,
        "trivial_identity_lcb95": lcb(trivial_delta),
        "privileged_tested_arm_ceiling_mean": float(np.mean(values)),
        "privileged_tested_arm_ceiling_lcb95": lcb(oracle_delta),
        "history_clusters": int(len(root_means)),
        "approximate_two_sided_mde_80pct": mde_80_two_sided,
        "boundary": "CVaR10 remains secondary; the privileged selector is a diagnostic ceiling over tested arms, not a policy claim",
        "q_r1_delayed": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
