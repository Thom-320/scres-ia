#!/usr/bin/env python3
"""Learner-blind screen of the thesis-native op9 dispatch lever.

Contract: contracts/thesis_native_dispatch_lever_screen_v1.json (frozen before dev seeds).
Direct SimPy via MFSCGymEnvShifts track_a_v1 (6D); ONLY op9_q varies. Constant postures,
observable two-mode policies (switch on the LOC/op-down state, 168h hold), and clairvoyant
two-mode policies (same trigger from the recorded action-independent risk timeline, with
168h anticipation). No learner anywhere.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_garrido_risk_headroom_sensitivity import build_profiles  # noqa: E402
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.external_env_interface import make_track_b_env  # noqa: E402

NEUTRAL_Q = -1.0 / 3.0  # multiplier 1.0 under 1.25 + 0.75*a
S1_SIG = -1.0
MULTS = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
CALM_MULTS = (0.75, 1.0, 1.25)
HOLD_HOURS = 168.0
LEAD_HOURS = 168.0
WATCH_OPS = (4, 8, 9, 10, 11, 12)
CELLS = ["R1_current", "R2_current", "R2_OAT_R22_increased",
         "impact_R22_psi1.5", "impact_R22_psi2", "R2_OAT_R24_increased"]
DEV_SEEDS = list(range(7_540_001, 7_540_013))


def act(mult: float) -> np.ndarray:
    """6D track_a_v1 action with only op9_q set to `mult`; everything else thesis-neutral."""
    a9 = (float(mult) - 1.25) / 0.75
    return np.asarray([NEUTRAL_Q, a9, NEUTRAL_Q, NEUTRAL_Q, 0.0, S1_SIG], dtype=np.float32)


def make_env(profile: dict, *, seed: int, max_steps: int):
    env = make_track_b_env(
        action_contract="track_a_v1",
        action_mode="full",
        reward_mode="ReT_excel_delta",
        observation_version="v3",
        risk_level="current",
        risk_overrides=dict(profile["overrides"]),
        risk_impact_multipliers_by_id=dict(profile["impact"]),
        enabled_risks=tuple(profile["enabled"]),
        risk_rng_mode="per_risk",
        stochastic_pt=False,
        max_steps=int(max_steps),
        step_size_hours=168.0,
    )
    env.reset(seed=int(seed))
    return env


def _metrics(env) -> dict[str, float]:
    m = compute_episode_metrics(env.unwrapped.sim)
    return {k: float(m[k]) for k in (
        "ret_excel", "ration_ret_excel", "ret_excel_cvar10", "lost_orders",
        "backorder_qty_final", "backlog_age_max", "service_loss_auc_ration_hours",
    )} | {"risk_events": float(len(env.unwrapped.sim.risk_events))}


def _event_windows(sim) -> list[tuple[float, float]]:
    return sorted(
        (float(e.start_time), float(e.end_time if e.end_time is not None else e.start_time))
        for e in sim.risk_events
        if set(e.affected_ops or []) & set(WATCH_OPS)
    )


def run_policy(profile: dict, *, seed: int, max_steps: int, calm: float, active: float,
               mode: str, windows: list[tuple[float, float]] | None) -> dict:
    """mode: 'constant' (calm==active), 'obs' (switch on live op-down), 'oracle' (windows+lead)."""
    env = make_env(profile, seed=seed, max_steps=max_steps)
    sim = env.unwrapped.sim
    done = truncated = False
    switches = 0
    prev_active = False
    hold_until = -1.0
    try:
        while not (done or truncated):
            now = float(sim.env.now)
            if mode == "constant":
                on = False
            elif mode == "obs":
                down = any(sim.op_down_count.get(op, 0) > 0 for op in WATCH_OPS)
                if down:
                    hold_until = now + HOLD_HOURS
                on = down or now < hold_until
            else:  # oracle: anticipate LEAD before each recorded window, hold through +HOLD
                on = any(s - LEAD_HOURS <= now <= e + HOLD_HOURS for s, e in (windows or []))
            if on != prev_active:
                switches += 1
                prev_active = on
            action = act(active if on else calm)
            _o, _r, done, truncated, _i = env.step(action)
        row = _metrics(env)
        row.update({"calm_mult": calm, "active_mult": active, "mode": mode,
                    "switches": switches, "seed": seed, "profile": profile["id"]})
        if mode == "constant":
            # self-check: neutral posture must leave thesis dispatch bounds untouched
            if abs(calm - 1.0) < 1e-9:
                p = sim.params
                row["op9_q_bounds"] = [float(p.get("op9_q_min", -1)), float(p.get("op9_q_max", -1))]
        row["event_windows"] = _event_windows(sim)
        return row
    finally:
        env.close()


def task(spec: dict) -> dict:
    profiles = {p["id"]: p for p in build_profiles()}
    return run_policy(
        profiles[spec["profile_id"]], seed=spec["seed"], max_steps=spec["max_steps"],
        calm=spec["calm"], active=spec["active"], mode=spec["mode"],
        windows=spec.get("windows"),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--cells", nargs="+", default=CELLS)
    ap.add_argument("--seeds", nargs="+", type=int, default=DEV_SEEDS)
    ap.add_argument("--max-steps", type=int, default=520)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--hard-cap-seconds", type=float, default=21_600.0)
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")

    started = time.perf_counter()
    # Pass 1: constants (also record the action-independent risk windows per (cell, seed)).
    specs = [
        {"profile_id": c, "seed": s, "max_steps": args.max_steps,
         "calm": m, "active": m, "mode": "constant"}
        for c in args.cells for s in args.seeds for m in MULTS
    ]
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, row in enumerate(ex.map(task, specs), 1):
            rows.append(row)
            if i % 50 == 0:
                print(f"  pass1 {i}/{len(specs)} ({time.perf_counter()-started:.0f}s)", flush=True)
            if time.perf_counter() - started > args.hard_cap_seconds:
                raise TimeoutError("hard cap exceeded in pass 1")

    windows = {(r["profile"], r["seed"]): r["event_windows"]
               for r in rows if abs(r["calm_mult"] - 1.0) < 1e-9}

    # CRN self-check: identical windows across constant arms for same (cell, seed).
    for r in rows:
        key = (r["profile"], r["seed"])
        if r["event_windows"] != windows[key]:
            raise SystemExit(f"CRN violation: risk windows differ across arms for {key}")

    # Pass 2: two-mode obs + oracle.
    specs2 = []
    for c in args.cells:
        for s in args.seeds:
            w = windows[(c, s)]
            for calm in CALM_MULTS:
                for active in MULTS:
                    if abs(active - calm) < 1e-9:
                        continue
                    specs2.append({"profile_id": c, "seed": s, "max_steps": args.max_steps,
                                   "calm": calm, "active": active, "mode": "obs"})
                    specs2.append({"profile_id": c, "seed": s, "max_steps": args.max_steps,
                                   "calm": calm, "active": active, "mode": "oracle",
                                   "windows": w})
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, row in enumerate(ex.map(task, specs2), 1):
            rows.append(row)
            if i % 100 == 0:
                print(f"  pass2 {i}/{len(specs2)} ({time.perf_counter()-started:.0f}s)", flush=True)
            if time.perf_counter() - started > args.hard_cap_seconds:
                raise TimeoutError("hard cap exceeded in pass 2")

    for row in rows:
        row.pop("event_windows", None)
    out = {
        "schema_version": "thesis_native_dispatch_screen_v1",
        "claim_status": "DEVELOPMENT_SCREEN_NO_CLAIM",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract": "contracts/thesis_native_dispatch_lever_screen_v1.json",
        "cells": args.cells, "seeds": args.seeds, "max_steps": args.max_steps,
        "rows": rows, "elapsed_seconds": time.perf_counter() - started,
        "selection_performed": False, "learner_return_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"rows={len(rows)} elapsed={out['elapsed_seconds']:.0f}s -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
