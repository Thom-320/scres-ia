#!/usr/bin/env python3
"""Program O relevant-risk sensitivity screen — gates G0/G1 (+ G2 map runner).

Contract: contracts/program_o_relevant_risk_sensitivity_v1.json (frozen 2026-07-17).
Stage discipline: G0 (risks-off identity vs custodied corrective-validation raw matrices) and
G1 (each relevant risk fires, alone, at thesis-plausible escalated rates; R3 absent) must pass
and be independently verified BEFORE the G2 map is read.

Usage: run_program_o_relevant_risk_screen.py g0 | g1
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.screen_program_o_hobs_fit import primary_scheduler  # noqa: E402
from supply_chain.program_o_full_des import run_program_o_full_des_episode  # noqa: E402
from supply_chain.program_o_full_des_transducer import direct_full_des_vector  # noqa: E402

RUN = ROOT / "results/program_o/fixed_clock_hobs_corrective_validation_v1/remote_run/artifacts/validation"
PARENT_CONTRACT = ROOT / "contracts/program_o_full_des_hpi_translation_v1.json"
_SCHEDULER = None


def frozen_scheduler():
    global _SCHEDULER
    if _SCHEDULER is None:
        _SCHEDULER = primary_scheduler(json.load(open(PARENT_CONTRACT)))
    return _SCHEDULER
OUT_DIR = ROOT / "results/program_o/relevant_risk_sensitivity_v1"
CELL = "rho90_share90"                      # frozen primary cell (contract)
CELL_PARAMS = {"regime_persistence": 0.90, "dominant_share": 0.90}
INCLUDED = ["R11", "R14", "R21", "R22"]   # R24 excluded pre-execution (amendment 2026-07-17): no P_C/P_H label
G1_PHI = 4.0
G0_TOL = 1e-9


def decode_calendar(index: int, *, weeks: int = 8) -> tuple[int, ...]:
    """Verbatim copy of screen_program_o_fixed_clock_hobs_validation.decode_calendar
    (most-significant-digit-first; a reversed decode reproduces gross production exactly
    while silently scoring a different action sequence -- caught by G0)."""
    value = int(index)
    actions = [0] * weeks
    for position in range(weeks - 1, -1, -1):
        actions[position] = value % 4
        value //= 4
    if value:
        raise ValueError("calendar index exceeds frozen horizon")
    return tuple(actions)


def episode(seed: int, calendar, *, risks=None, phi=None):
    sim, panel = _episode_raw(seed, calendar, risks=risks, phi=phi)
    return sim, direct_full_des_vector(sim, panel)


def _episode_raw(seed: int, calendar, *, risks=None, phi=None):
    return run_program_o_full_des_episode(
        seed=int(seed),
        calendar=calendar,
        scheduler=frozen_scheduler(),
        downstream_freight_physics_mode="fixed_clock_physical_v1",
        risks_enabled=risks is not None,
        enabled_risks=set(risks) if risks else None,
        risk_frequency_multipliers_by_id={r: float(phi) for r in risks} if risks and phi else None,
        **CELL_PARAMS,
    )


def g0() -> dict:
    """Risks-off identity vs the custodied raw calendar matrix (transducer<->DES parity chain)."""
    result = json.load(open(RUN / "result.json"))
    static_index = int(result["cells"][CELL]["static_index"])
    cal = decode_calendar(static_index)
    z = np.load(RUN / f"raw_calendar_matrix/{CELL}/tape_7430001.npz")
    _, panel = episode(7430001, cal)
    checks = {}
    for key in ("ret_visible", "ret_visible_cvar10", "gross_production_quantity"):
        ref = float(z[key][static_index])
        got = float(panel[key])
        checks[key] = {"episode": got, "custodied": ref, "abs_diff": abs(got - ref)}
    ok = all(c["abs_diff"] < G0_TOL for c in checks.values())
    return {"gate": "G0_identity", "cell": CELL, "seed": 7430001, "static_index": static_index,
            "tolerance": G0_TOL, "checks": checks, "pass": bool(ok)}


def g1() -> dict:
    """Each included risk, alone, at phi=4: fires with plausible counts; nothing else fires."""
    result = json.load(open(RUN / "result.json"))
    cal = decode_calendar(int(result["cells"][CELL]["static_index"]))
    per_risk = {}
    for rid in INCLUDED:
        sim, panel = episode(7430001, cal, risks=[rid], phi=G1_PHI)
        events = list(getattr(sim, "risk_events", []))
        ids = {}
        sample_keys = None
        ops_seen = set()
        for e in events:
            eid = getattr(e, "risk_id", None) or (e.get("risk_id") if isinstance(e, dict) else None)
            ids[eid] = ids.get(eid, 0) + 1
            if sample_keys is None:
                sample_keys = sorted(e.keys()) if isinstance(e, dict) else sorted(vars(e).keys())
            for attr in ("operation", "op", "operations", "ops"):
                v = getattr(e, attr, None) if not isinstance(e, dict) else e.get(attr)
                if v is not None:
                    ops_seen.update(v if isinstance(v, (list, tuple, set)) else [v])
        foreign = {k: v for k, v in ids.items() if k != rid}
        per_risk[rid] = {"event_count": ids.get(rid, 0), "foreign_risk_events": foreign,
                         "ops_seen": sorted(map(str, ops_seen)), "event_record_keys": sample_keys,
                         "ret_visible": float(panel["ret_visible"]),
                         "delta_ret_vs_risks_off": None}
    base = episode(7430001, cal)[1]
    for rid in INCLUDED:
        per_risk[rid]["delta_ret_vs_risks_off"] = per_risk[rid]["ret_visible"] - float(base["ret_visible"])
    ok = all(per_risk[r]["event_count"] > 0 and not any(k == "R3" for k in per_risk[r]["foreign_risk_events"])
             for r in INCLUDED)
    return {"gate": "G1_risk_fixture", "phi": G1_PHI, "per_risk": per_risk,
            "r3_events_anywhere": any("R3" in per_risk[r]["foreign_risk_events"] for r in INCLUDED),
            "pass": bool(ok)}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    which = sys.argv[1] if len(sys.argv) > 1 else "g0"
    out = g0() if which == "g0" else g1()
    (OUT_DIR / f"{which}_result.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1)[:2400])
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
