#!/usr/bin/env python3
"""Static-only headroom gate — the REAL 5D version.

`run_track_a_v2_conservation_3d_gate.py` only varies op3_q, op9_q, shift and
freezes op3_rop/op9_rop at Garrido baseline. That is a cheap screen, not a
bound on the contract's actual decision space. This script varies all 5
effective dims of `track_a_v1` (op3_q, op9_q, op3_rop, op9_rop, shift;
`op5_q` stays inert per D11 — see docs/THESIS_INTERPRETATION_DECISIONS_2026-06-24.md).

Only promote to PPO confirmatory training if `opening_real` is True here.
A 3D-gate `opening_real=False` does NOT license that decision — the rop axis
might be exactly where the headroom lives (rop controls how *often* the
agent re-triggers a dispatch check, which is a real lever a static policy
can't fire dynamically in response to risk).
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
from scripts.run_track_a_v2_conservation_3d_gate import (  # noqa: E402
    eval_candidate as _eval_candidate_impl,
    make_env,
    mult_to_signal,
)
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402

SHIFT_SIGS = {1: -1.0, 2: 0.0, 3: 1.0}


@dataclass(frozen=True)
class Candidate:
    label: str
    action: tuple[float, ...]  # 6D: (op3_sig, op9_sig, op3_rop_sig, op9_rop_sig, op5_sig=0, shift_sig)
    resource: float


def conservation_candidates_5d(
    op3_mults: Iterable[float],
    op9_mults: Iterable[float],
    op3_rop_mults: Iterable[float],
    op9_rop_mults: Iterable[float],
    shifts: Iterable[int],
) -> list[Candidate]:
    out: list[Candidate] = []
    for op3_mult, op9_mult, op3_rop_mult, op9_rop_mult, shift in itertools.product(
        sorted(set(op3_mults)),
        sorted(set(op9_mults)),
        sorted(set(op3_rop_mults)),
        sorted(set(op9_rop_mults)),
        shifts,
    ):
        shift = int(shift)
        resource = 0.5 * ((op3_mult + op9_mult) / 2.0 - 0.5) / 1.5 + 0.5 * ((shift - 1) / 2.0)
        label = (
            f"op3_{op3_mult:g}_op9_{op9_mult:g}_"
            f"rop3_{op3_rop_mult:g}_rop9_{op9_rop_mult:g}_S{shift}"
        )
        out.append(
            Candidate(
                label=label,
                action=(
                    mult_to_signal(op3_mult),
                    mult_to_signal(op9_mult),
                    mult_to_signal(op3_rop_mult),
                    mult_to_signal(op9_rop_mult),
                    0.0,  # op5_q: inert (D11), left at neutral signal
                    SHIFT_SIGS[shift],
                ),
                resource=float(np.clip(resource, 0.0, 1.0)),
            )
        )
    return out


def eval_candidate(
    *, family: str, phi: float, psi: float, candidate: Candidate, seed: int, max_steps: int
) -> dict[str, float]:
    # Reuse the 3D gate's eval loop exactly (same env recipe); it is generic
    # over `candidate.action`, so a 6-tuple with rop dims populated works
    # unchanged.
    return _eval_candidate_impl(
        family=family, phi=phi, psi=psi, candidate=candidate, seed=seed, max_steps=max_steps
    )


def parse_csv_floats(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_csv_strings(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--families", default="R13,R14,R24")
    ap.add_argument("--phis", default="1,4,8")
    ap.add_argument("--psis", default="1.5")
    ap.add_argument("--op3-mults", default="0.75,1.0,1.25,1.5")
    ap.add_argument("--op9-mults", default="0.75,1.0,1.25,1.5")
    ap.add_argument("--op3-rop-mults", default="0.5,1.0")
    ap.add_argument("--op9-rop-mults", default="0.5,1.0")
    ap.add_argument("--shifts", default="1,2,3")
    ap.add_argument("--seeds", default="7100")
    ap.add_argument("--max-steps", type=int, default=52)
    ap.add_argument(
        "--output", default="outputs/experiments/track_a_v2_conservation_5d_gate_2026-07-03"
    )
    ap.add_argument("--quick", action="store_true", help="small smoke for plumbing")
    args = ap.parse_args()

    if args.quick:
        args.families = "R24"
        args.phis = "1,4"
        args.psis = "1.5"
        args.op3_mults = "0.75,1.0"
        args.op9_mults = "0.75,1.0"
        args.op3_rop_mults = "1.0"
        args.op9_rop_mults = "1.0"
        args.shifts = "1,2"
        args.seeds = "7000"
        args.max_steps = min(args.max_steps, 12)

    families = parse_csv_strings(args.families)
    phis = parse_csv_floats(args.phis)
    psis = parse_csv_floats(args.psis)
    op3_mults = parse_csv_floats(args.op3_mults)
    op9_mults = parse_csv_floats(args.op9_mults)
    op3_rop_mults = parse_csv_floats(args.op3_rop_mults)
    op9_rop_mults = parse_csv_floats(args.op9_rop_mults)
    shifts = [int(x) for x in parse_csv_floats(args.shifts)]
    seeds = [int(x) for x in parse_csv_floats(args.seeds)]
    unknown = [f for f in families if f not in FAMILY_RISKS]
    if unknown:
        raise ValueError(f"unknown families: {unknown}; valid={sorted(FAMILY_RISKS)}")

    candidates = conservation_candidates_5d(
        op3_mults, op9_mults, op3_rop_mults, op9_rop_mults, shifts
    )
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    total = len(candidates) * len(list(itertools.product(families, phis, psis))) * len(seeds)
    print(
        f"5D gate: {len(candidates)} candidates x "
        f"{len(families) * len(phis) * len(psis)} regimes x {len(seeds)} seeds "
        f"= {total} episodes",
        flush=True,
    )

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
                    "mode": "conservation_v1_5d", "candidate": cand.label,
                    "action": json.dumps(cand.action), "seed": seed, **metrics,
                }
                rows.append(row)

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
        "grid_scope": "genuine 5D grid: all 5 effective dims varied.",
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

    print("\n=== TRACK A V2 (CONSERVATION-RESPECTING, 5D) HEADROOM SEARCH ===")
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
