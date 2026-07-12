#!/usr/bin/env python3
"""Static-only Track A headroom search across risk families and stress levels.

This is the gate before any PPO training. It asks a structural question:

    Does the best static Track A action change across risk regimes enough that
    an adaptive policy could beat the best single constant policy?

If the static oracle (best action per regime) does not beat the best single
constant action with a positive bootstrap CI, there is no honest Track A
dynamic headroom to train on.
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

from supply_chain.continuous_its_env import (  # noqa: E402
    make_continuous_its_track_a_env,
    make_per_op_buffer_track_a_env,
)
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.thesis_design import R1_RISKS, R2_RISKS, R3_RISKS  # noqa: E402

SHIFT_SIGS = {1: -1.0, 2: 0.0, 3: 1.0}
FAMILY_RISKS: dict[str, tuple[str, ...] | None] = {
    "R1": tuple(R1_RISKS),
    "R1u": ("R12", "R13"),
    "R13": ("R13",),
    "R14": ("R14",),
    "R2": tuple(R2_RISKS),
    "R3": tuple(R3_RISKS),
    "R24": ("R24",),
    "mixed": None,
}


@dataclass(frozen=True)
class Candidate:
    label: str
    action: tuple[float, ...]
    resource: float


def parse_csv_floats(value: str) -> list[float]:
    vals = [float(x.strip()) for x in value.split(",") if x.strip()]
    if not vals:
        raise ValueError("expected at least one numeric value")
    return vals


def parse_csv_strings(value: str) -> list[str]:
    vals = [x.strip() for x in value.split(",") if x.strip()]
    if not vals:
        raise ValueError("expected at least one value")
    return vals


def continuous_candidates(fracs: Iterable[float], shifts: Iterable[int]) -> list[Candidate]:
    out: list[Candidate] = []
    for frac, shift in itertools.product(fracs, shifts):
        frac = float(np.clip(frac, 0.0, 1.0))
        shift = int(shift)
        resource = 0.5 * frac + 0.5 * ((shift - 1) / 2.0)
        out.append(
            Candidate(
                label=f"f{frac:g}_S{shift}",
                action=(frac, SHIFT_SIGS[shift]),
                resource=float(resource),
            )
        )
    return out


def per_op_candidates(
    fracs: Iterable[float],
    shifts: Iterable[int],
    *,
    grid: str,
) -> list[Candidate]:
    vals = sorted({float(np.clip(x, 0.0, 1.0)) for x in fracs})
    triples: set[tuple[float, float, float]] = set()
    if grid == "full":
        triples.update(itertools.product(vals, vals, vals))
    else:
        positives = [x for x in vals if x > 0]
        low = min(positives) if positives else 0.0
        for f in vals:
            triples.add((f, f, f))          # common buffer
            triples.add((0.0, 0.0, f))      # downstream-only Op9
            triples.add((0.0, f, f))        # Op5+Op9
            triples.add((low, low, f))      # light upstream + variable Op9
            triples.add((f, 0.0, 0.0))      # Op3 isolate
            triples.add((0.0, f, 0.0))      # Op5 isolate
    out: list[Candidate] = []
    for op3, op5, op9 in sorted(triples):
        for shift in shifts:
            shift = int(shift)
            buffer_resource = (op3 + op5 + op9) / 3.0
            resource = 0.5 * buffer_resource + 0.5 * ((shift - 1) / 2.0)
            label = f"op3{op3:g}_op5{op5:g}_op9{op9:g}_S{shift}"
            out.append(
                Candidate(
                    label=label,
                    action=(op3, op5, op9, SHIFT_SIGS[shift]),
                    resource=float(resource),
                )
            )
    return out


def make_env(
    *,
    mode: str,
    family: str,
    phi: float,
    psi: float,
    max_steps: int,
    seed: int,
):
    enabled_risks = FAMILY_RISKS[family]
    common = dict(
        reward_mode="ReT_excel_delta",
        observation_version="v6",
        risk_level="current",
        risk_frequency_multiplier=float(phi),
        risk_impact_multiplier=float(psi),
        stochastic_pt=False,
        max_steps=int(max_steps),
        step_size_hours=168.0,
        risk_obs=True,
        holding_cost=0.0,
        shift_cost=0.001,
    )
    if enabled_risks is not None:
        common["enabled_risks"] = enabled_risks
    if mode == "continuous":
        env = make_continuous_its_track_a_env(init_frac=0.0, **common)
    elif mode == "per_op":
        env = make_per_op_buffer_track_a_env(init_fracs=(0.0, 0.0, 0.0), **common)
    else:
        raise ValueError(f"unknown mode: {mode}")
    env.reset(seed=int(seed))
    return env


def eval_candidate(
    *,
    mode: str,
    family: str,
    phi: float,
    psi: float,
    candidate: Candidate,
    seed: int,
    max_steps: int,
) -> dict[str, float]:
    env = make_env(mode=mode, family=family, phi=phi, psi=psi, max_steps=max_steps, seed=seed)
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


def regime_name(family: str, phi: float, psi: float) -> str:
    return f"{family}_phi{phi:g}_psi{psi:g}"


def summarize_gate(rows: list[dict], candidates: list[Candidate], regimes: list[str], seeds: list[int]) -> dict:
    metric = "excel"
    by_regime_candidate_seed: dict[tuple[str, str], dict[int, float]] = {}
    for row in rows:
        key = (row["regime"], row["candidate"])
        by_regime_candidate_seed.setdefault(key, {})[int(row["seed"])] = float(row[metric])

    candidate_labels = [c.label for c in candidates]

    def mean_for(regime: str, label: str, seed_subset: Iterable[int]) -> float:
        vals = [
            by_regime_candidate_seed[(regime, label)][s]
            for s in seed_subset
            if s in by_regime_candidate_seed.get((regime, label), {})
        ]
        return float(np.mean(vals)) if vals else float("-inf")

    best_by_regime: dict[str, dict] = {}
    for reg in regimes:
        best_label = max(candidate_labels, key=lambda lab: mean_for(reg, lab, seeds))
        best_by_regime[reg] = {
            "candidate": best_label,
            "excel": mean_for(reg, best_label, seeds),
        }

    def mean_across_regimes(label: str, seed_subset: Iterable[int]) -> float:
        return float(np.mean([mean_for(reg, label, seed_subset) for reg in regimes]))

    best_single_label = max(candidate_labels, key=lambda lab: mean_across_regimes(lab, seeds))
    best_single_excel = mean_across_regimes(best_single_label, seeds)
    oracle_excel = float(np.mean([best_by_regime[reg]["excel"] for reg in regimes]))

    rng = np.random.default_rng(0)
    headrooms = []
    seed_array = np.asarray(seeds, dtype=int)
    for _ in range(2000):
        sample = list(rng.choice(seed_array, size=len(seed_array), replace=True))
        oracle = float(
            np.mean(
                [
                    max(mean_for(reg, lab, sample) for lab in candidate_labels)
                    for reg in regimes
                ]
            )
        )
        best_single = max(mean_across_regimes(lab, sample) for lab in candidate_labels)
        headrooms.append(oracle - best_single)
    h = np.asarray(headrooms, dtype=float)
    ci = [float(np.percentile(h, 2.5)), float(np.percentile(h, 97.5))]
    best_labels = {v["candidate"] for v in best_by_regime.values()}
    return {
        "primary_metric": metric,
        "best_by_regime": best_by_regime,
        "best_single_constant": {
            "candidate": best_single_label,
            "excel": best_single_excel,
        },
        "oracle_excel": oracle_excel,
        "oracle_minus_best_static": oracle_excel - best_single_excel,
        "oracle_minus_best_static_ci95": ci,
        "best_action_changes_across_regimes": len(best_labels) > 1,
        "opening_real": bool(ci[0] > 0 and len(best_labels) > 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("continuous", "per_op"), default="continuous")
    ap.add_argument("--families", default="R1,R2,R3,R24,mixed")
    ap.add_argument("--phis", default="1,2,4,6,8")
    ap.add_argument("--psis", default="1.0,1.5,2.0")
    ap.add_argument("--fracs", default="0,0.05,0.10,0.15,0.20,0.25,0.30,0.50")
    ap.add_argument("--shifts", default="1,2,3")
    ap.add_argument("--per-op-grid", choices=("targeted", "full"), default="targeted")
    ap.add_argument("--seeds", default="7000,7001")
    ap.add_argument("--max-steps", type=int, default=52)
    ap.add_argument("--output", default="outputs/experiments/track_a_headroom_search_2026-06-29")
    ap.add_argument("--quick", action="store_true", help="small R2/R24 smoke for plumbing")
    args = ap.parse_args()

    if args.quick:
        args.families = "R2,R24"
        args.phis = "1,4"
        args.psis = "1.0"
        args.fracs = "0,0.05,0.10,0.15"
        args.shifts = "1,2"
        args.seeds = "7000"
        args.max_steps = min(args.max_steps, 12)

    families = parse_csv_strings(args.families)
    phis = parse_csv_floats(args.phis)
    psis = parse_csv_floats(args.psis)
    fracs = parse_csv_floats(args.fracs)
    shifts = [int(x) for x in parse_csv_floats(args.shifts)]
    seeds = [int(x) for x in parse_csv_floats(args.seeds)]
    unknown = [f for f in families if f not in FAMILY_RISKS]
    if unknown:
        raise ValueError(f"unknown families: {unknown}; valid={sorted(FAMILY_RISKS)}")

    candidates = (
        continuous_candidates(fracs, shifts)
        if args.mode == "continuous"
        else per_op_candidates(fracs, shifts, grid=args.per_op_grid)
    )
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
                    mode=args.mode,
                    family=family,
                    phi=phi,
                    psi=psi,
                    candidate=cand,
                    seed=seed,
                    max_steps=args.max_steps,
                )
                row = {
                    "regime": reg,
                    "family": family,
                    "phi": phi,
                    "psi": psi,
                    "mode": args.mode,
                    "candidate": cand.label,
                    "action": json.dumps(cand.action),
                    "seed": seed,
                    **metrics,
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
                "regime": reg,
                "family": family,
                "phi": phi_part.replace("phi", ""),
                "psi": psi_part.replace("psi", ""),
                "best_candidate": item["candidate"],
                "best_excel": item["excel"],
            }
        )
    with (out / "best_static_by_regime.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(best_rows[0].keys()))
        writer.writeheader()
        writer.writerows(best_rows)

    payload = {
        "args": vars(args),
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

    print("\n=== TRACK A HEADROOM SEARCH ===")
    print(f"mode={args.mode} regimes={len(regimes)} candidates={len(candidates)} seeds={len(seeds)}")
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
