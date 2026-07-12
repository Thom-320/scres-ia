#!/usr/bin/env python3
"""Static-only headroom gate for Track A's conservation-respecting contract.

**Scope: this gate's GRID varies only 3 of the contract's 5 effective
dimensions — op3_q, op9_q, shift. op3_rop and op9_rop are held fixed at
Garrido baseline (mult=1.0) for every candidate.** Do not call this a "5D
gate" or a "5D full bound" — it is a 3D screen on a contract that HAS 5
effective dims. A genuine 5D gate (varying rop too) is a separate,
not-yet-built follow-up; only run PPO confirmatory training after that one,
not after this one.

Mirrors `run_track_a_headroom_search.py`'s structure (same regimes, same
oracle-vs-best-single-constant headroom test), but swaps the action space:
instead of `per_op_buffer` (fraction targets fed into the unconditioned
`_top_up_inventory_buffer` exogenous top-up — the mechanism Garrido flagged
in the 2026-07-02 meeting as "no se puede abastecer de la nada"), this uses
the native contract on `MFSCGymEnvShifts` (`action_contract="track_a_v1"`):
(op3_q, op9_q, op3_rop, op9_rop, shift) — 5 effective dims (`op5_q`, dim 4 of
the native 6D contract, is inert per D11: it only has an effect if
`initial_buffers["op5_rm"]` is populated, which would resurrect the flawed
top-up mechanism this gate is built to avoid). Every dim this gate DOES vary
is consumed by a capacity-capped DES event (`_op3_wdc_dispatch`,
`_op9_sb_dispatch`; see docs/THESIS_INTERPRETATION_DECISIONS_2026-06-24.md
D10) — replenishment can never exceed on-hand upstream inventory.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_track_a_headroom_search import (  # noqa: E402
    FAMILY_RISKS,
    regime_name,
    summarize_gate,
)
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.external_env_interface import make_track_b_env  # noqa: E402

# Dims 0-3 (op3_q, op9_q, op3_rop, op9_rop): multiplier = 1.25 + 0.75*signal,
# so signal = (mult - 1.25) / 0.75. mult=1.0 -> signal=-1/3 (baseline).
_Q_SLOPE = 0.75
_Q_INTERCEPT = 1.25
SHIFT_SIGS = {1: -1.0, 2: 0.0, 3: 1.0}


def mult_to_signal(mult: float) -> float:
    return float(np.clip((mult - _Q_INTERCEPT) / _Q_SLOPE, -1.0, 1.0))


@dataclass(frozen=True)
class Candidate:
    label: str
    action: tuple[float, ...]  # 6D: (op3_sig, op9_sig, op3_rop_sig, op9_rop_sig, op5_sig=0, shift_sig)
    resource: float


def conservation_candidates(
    op3_mults: Iterable[float], op9_mults: Iterable[float], shifts: Iterable[int]
) -> list[Candidate]:
    baseline_rop_sig = mult_to_signal(1.0)  # rop held at Garrido baseline; see module docstring
    out: list[Candidate] = []
    for op3_mult, op9_mult, shift in itertools.product(
        sorted(set(op3_mults)), sorted(set(op9_mults)), shifts
    ):
        shift = int(shift)
        resource = 0.5 * ((op3_mult + op9_mult) / 2.0 - 0.5) / 1.5 + 0.5 * ((shift - 1) / 2.0)
        label = f"op3_{op3_mult:g}_op9_{op9_mult:g}_S{shift}"
        out.append(
            Candidate(
                label=label,
                action=(
                    mult_to_signal(op3_mult),
                    mult_to_signal(op9_mult),
                    baseline_rop_sig,
                    baseline_rop_sig,
                    0.0,  # op5_q: inert (D11), left at neutral signal
                    SHIFT_SIGS[shift],
                ),
                resource=float(np.clip(resource, 0.0, 1.0)),
            )
        )
    return out


def make_env(*, family: str, phi: float, psi: float, max_steps: int, seed: int):
    enabled_risks = FAMILY_RISKS[family]
    kwargs: dict = dict(
        action_contract="track_a_v1",
        action_mode="full",
        reward_mode="ReT_excel_delta",
        observation_version="v6",
        risk_level="current",
        risk_frequency_multiplier=float(phi),
        risk_impact_multiplier=float(psi),
        stochastic_pt=False,
        max_steps=int(max_steps),
        step_size_hours=168.0,
        ret_excel_cvar_alpha=0.0,
    )
    if enabled_risks is not None:
        kwargs["enabled_risks"] = enabled_risks
    env = make_track_b_env(**kwargs)
    env.reset(seed=int(seed))
    return env


def eval_candidate(
    *, family: str, phi: float, psi: float, candidate: Candidate, seed: int, max_steps: int
) -> dict[str, float]:
    env = make_env(family=family, phi=phi, psi=psi, max_steps=max_steps, seed=seed)
    obs, _info = env.reset(seed=int(seed))
    done = truncated = False
    resources: list[float] = []
    try:
        while not (done or truncated):
            obs, _reward, done, truncated, info = env.step(
                np.asarray(candidate.action, dtype=np.float32)
            )
            resources.append(float(info.get("resource_composite", candidate.resource)))
        metrics = compute_episode_metrics(env.unwrapped.sim)
        return {
            "excel": float(metrics.get("ret_excel", np.nan)),
            "cvar_loss": float(metrics.get("service_loss_auc_ration_hours", np.nan)),
            "flow_fill": float(metrics.get("flow_fill_rate", np.nan)),
            "lost_rate": float(metrics.get("lost_rate", np.nan)),
            "resource": float(np.nanmean(resources)) if resources else candidate.resource,
        }
    finally:
        env.close()


def parse_csv_floats(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_csv_strings(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--families", default="R13,R14,R24")
    ap.add_argument("--phis", default="1,4,8")
    ap.add_argument("--psis", default="1.5")
    ap.add_argument("--op3-mults", default="0.5,0.75,1.0,1.25,1.5,2.0")
    ap.add_argument("--op9-mults", default="0.5,0.75,1.0,1.25,1.5,2.0")
    ap.add_argument("--shifts", default="1,2,3")
    ap.add_argument("--seeds", default="7000,7001")
    ap.add_argument("--max-steps", type=int, default=52)
    ap.add_argument(
        "--output", default="outputs/experiments/track_a_v2_conservation_gate_2026-07-03"
    )
    ap.add_argument("--quick", action="store_true", help="small smoke for plumbing")
    args = ap.parse_args()

    if args.quick:
        args.families = "R24"
        args.phis = "1,4"
        args.psis = "1.5"
        args.op3_mults = "0.75,1.0,1.25"
        args.op9_mults = "0.75,1.0,1.25"
        args.shifts = "1,2"
        args.seeds = "7000"
        args.max_steps = min(args.max_steps, 12)

    families = parse_csv_strings(args.families)
    phis = parse_csv_floats(args.phis)
    psis = parse_csv_floats(args.psis)
    op3_mults = parse_csv_floats(args.op3_mults)
    op9_mults = parse_csv_floats(args.op9_mults)
    shifts = [int(x) for x in parse_csv_floats(args.shifts)]
    seeds = [int(x) for x in parse_csv_floats(args.seeds)]
    unknown = [f for f in families if f not in FAMILY_RISKS]
    if unknown:
        raise ValueError(f"unknown families: {unknown}; valid={sorted(FAMILY_RISKS)}")

    candidates = conservation_candidates(op3_mults, op9_mults, shifts)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    regimes: list[str] = []
    for family, phi, psi in itertools.product(families, phis, psis):
        reg = regime_name(family, phi, psi)
        regimes.append(reg)
        for cand in candidates:
            for seed in seeds:
                metrics = eval_candidate(
                    family=family, phi=phi, psi=psi, candidate=cand, seed=seed,
                    max_steps=args.max_steps,
                )
                row = {
                    "regime": reg, "family": family, "phi": phi, "psi": psi,
                    "mode": "conservation_v1", "candidate": cand.label,
                    "action": json.dumps(cand.action), "seed": seed, **metrics,
                }
                rows.append(row)
                print(
                    f"{reg:18} {cand.label:24} seed={seed} "
                    f"excel={metrics['excel']:.5f} res={metrics['resource']:.3f}",
                    flush=True,
                )

    fieldnames = list(rows[0].keys())
    with (out / "static_runs.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize_gate(rows, candidates, regimes, seeds)
    best_rows = []
    for reg, item in summary["best_by_regime"].items():
        family, phi_part, psi_part = reg.split("_", 2)
        best_rows.append(
            {
                "regime": reg, "family": family,
                "phi": phi_part.replace("phi", ""), "psi": psi_part.replace("psi", ""),
                "best_candidate": item["candidate"], "best_excel": item["excel"],
            }
        )
    with (out / "best_static_by_regime.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(best_rows[0].keys()))
        writer.writeheader()
        writer.writerows(best_rows)

    payload = {
        "args": vars(args),
        "contract": "track_a_v1, 5 effective dims (op3_q, op9_q, op3_rop, op9_rop, shift; "
        "op5_q inert per D11)",
        "grid_scope": "3D SCREEN, not a 5D bound: only op3_q, op9_q, shift are varied. "
        "op3_rop and op9_rop are held fixed at mult=1.0 (Garrido baseline) for every "
        "candidate. A real 5D gate (varying rop too) is a separate follow-up.",
        "n_candidates": len(candidates),
        "candidate_preview": [asdict(c) for c in candidates[:20]],
        **summary,
        "promotion_rule": [
            "oracle_minus_best_static CI95 lower bound > 0",
            "best action changes across regimes",
            "then and only then train PPO on this calibrated campaign",
        ],
    }
    (out / "gate_summary.json").write_text(json.dumps(payload, indent=2, default=float))

    print("\n=== TRACK A V2 (CONSERVATION-RESPECTING) HEADROOM SEARCH ===")
    print(f"regimes={len(regimes)} candidates={len(candidates)} seeds={len(seeds)}")
    print(f"best single: {summary['best_single_constant']}")
    print(f"oracle_excel: {summary['oracle_excel']:.6f}")
    print(
        "oracle_minus_best_static: "
        f"{summary['oracle_minus_best_static']:+.6f} "
        f"CI95={summary['oracle_minus_best_static_ci95']}"
    )
    print(f"best action changes: {summary['best_action_changes_across_regimes']}")
    print(f"opening_real: {summary['opening_real']}")
    print(f"WROTE {out}/gate_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
