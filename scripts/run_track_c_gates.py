#!/usr/bin/env python3
"""Track C Gates C0-C2: oracle headroom BEFORE any training.

Frozen protocol: docs/TRACK_C_PREREGISTRATION_2026-07-10.md.
All stages are EVAL-ONLY (no learner). Stages:

  baseline  - Gate C0: thesis-anchor policies on the lambda-calibration tapes;
              freezes the J_v3 lambdas from the Cf_0 baseline statistics.
  screen    - Sobol global screen of constant 11D policies under J_v3.
  refine    - one local refinement around the screen leaders; freezes the
              best CONSTANT policy.
  pairs     - TRUE-regime switching pairs built from refinement leaders
              (plus a hand-built lean/heavy pair); freezes the best SWITCHER.
  verdict   - Gate C1: frozen switcher vs frozen constant on the C1 verdict
              battery (CRN-paired, two-way bootstrap). PROMOTE iff
              delta >= 0.05 * ReT_base AND CI95 wholly > 0.
  c2fit     - hazard-threshold detector grid (realized-event EWMA) driving
              the frozen pair; freezes the best detector.
  c2verdict - Gate C2: detector-switched pair vs frozen constant on the same
              battery. PASS iff it captures >= 50% of the C1 gap.

Tape discipline (never reuse): lambda/anchors 600001-600012; screen
600001-600008; refine 600001-600016; pair/detector fit 600017-600024;
C1/C2 verdict 600031-600054.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import qmc

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_track_b_crossed_eval import episode_metrics_row  # noqa: E402
from supply_chain.track_c_env import make_track_c_env  # noqa: E402

WEEK = 168.0
MAX_STEPS = 104

TAPES = {
    "baseline": list(range(600_001, 600_013)),
    "screen": list(range(600_001, 600_009)),
    "refine": list(range(600_001, 600_017)),
    "pairs": list(range(600_017, 600_025)),
    "verdict": list(range(600_031, 600_055)),
}

# Thesis anchors (signals). Garrido Cf_0: base multipliers (signal -1/3 on the
# 1.25+0.75x dims = 1.0x), S=1, no strategic stocks. 'garrido_I1344_S2' is his
# heaviest buffer row; 'lean'/'heavy' are the hand-built campaign pair.
ANCHORS: dict[str, list[float]] = {
    "cf0_S1_nostock": [-1 / 3, -1 / 3, -1 / 3, -1 / 3, 0.0, -1.0, -1 / 3, -1 / 3, 0.0, 0.0, 0.0],
    "garrido_I1344_S2": [-1 / 3, -1 / 3, -1 / 3, -1 / 3, 0.0, 0.0, -1 / 3, -1 / 3, 1.0, 1.0, 1.0],
    "garrido_I336_S2": [-1 / 3, -1 / 3, -1 / 3, -1 / 3, 0.0, 0.0, -1 / 3, -1 / 3, 0.25, 0.25, 0.25],
    "lean": [-1 / 3, -1 / 3, -2 / 3, -2 / 3, 0.0, -1.0, -1 / 3, -1 / 3, 0.0, 0.0, 0.0],
    "heavy": [1 / 3, 1 / 3, 0.0, 0.0, 0.0, 1.0, 1.0, 1 / 3, 0.5, 0.5, 0.5],
}


@dataclass(frozen=True)
class Policy:
    """Constant or state-switched 11D signal policy."""

    name: str
    calm: tuple[float, ...]
    campaign: tuple[float, ...] | None = None  # None => constant
    detector: tuple[float, float] | None = None  # (theta, halflife_weeks); None => true regime

    def is_switching(self) -> bool:
        return self.campaign is not None


class HazardDetector:
    """EWMA of an operator-observable hazard signal (non-privileged).

    signal_t = (# operations currently down) + (# recorded R21/R22/R23/R24
    event starts inside the last week). Recorded events appear only after
    recovery completes (sim logs at event end), which is why the live
    ops-down count is included: an operator sees outages as they happen.
    """

    RISKS = ("R21", "R22", "R23", "R24")

    def __init__(self, theta: float, halflife_weeks: float):
        self.theta = float(theta)
        self.alpha = 1.0 - 0.5 ** (1.0 / max(0.5, float(halflife_weeks)))
        self.ewma = 0.0
        self._seen = 0

    def update(self, sim: Any, now: float) -> bool:
        starts = sum(
            1
            for e in sim.risk_events[self._seen:]
            if e.risk_id in self.RISKS and e.start_time > now - WEEK
        )
        self._seen = len(sim.risk_events)
        ops_down = sum(1 for c in sim.op_down_count.values() if c > 0)
        signal = float(starts) + float(ops_down)
        self.ewma = (1.0 - self.alpha) * self.ewma + self.alpha * signal
        return self.ewma >= self.theta


def run_episode(
    policy: Policy,
    tape: int,
    campaign: dict[str, Any] | None = None,
    lead: float = 168.0,
) -> dict[str, float]:
    env = make_track_c_env(
        max_steps=MAX_STEPS,
        campaign_config=campaign,
        inventory_replenishment_lead_time=float(lead),
    )
    obs, info = env.reset(seed=tape)
    sim = env.unwrapped.sim
    detector = (
        HazardDetector(*policy.detector) if policy.detector is not None else None
    )
    calm = np.asarray(policy.calm, dtype=np.float32)
    camp = (
        np.asarray(policy.campaign, dtype=np.float32)
        if policy.campaign is not None
        else calm
    )
    in_campaign = False
    campaign_weeks_detected = 0
    terminated = truncated = False
    while not (terminated or truncated):
        if policy.is_switching():
            if detector is not None:
                in_campaign = detector.update(sim, float(sim.env.now))
            else:
                in_campaign = sim.campaign_state_at(float(sim.env.now)) == "campaign"
        action = camp if in_campaign else calm
        if in_campaign:
            campaign_weeks_detected += 1
        obs, _r, terminated, truncated, info = env.step(action)
    row = episode_metrics_row(sim)
    econ = info.get("track_c_econ", {})
    row.update({f"econ_{k}": float(v) for k, v in econ.items()})
    row["detected_campaign_weeks"] = float(campaign_weeks_detected)
    env.close()
    return row


def eval_policy(task: tuple) -> list[dict[str, Any]]:
    stage, policy, tapes, campaign, lead = task
    rows = []
    for tape in tapes:
        r = run_episode(policy, tape, campaign=campaign, lead=lead)
        rows.append({"stage": stage, "policy": policy.name, "eval_seed": tape, **r})
    return rows


def j_of(row: dict[str, Any], lam: dict[str, float]) -> float:
    return (
        float(row["ret_excel"])
        - lam["lam_h"] * float(row["econ_holding_frac_mean"])
        - lam["lam_d"] * float(row["econ_dispatch_excess_mean"])
        - lam["lam_s"] * float(row["econ_shift_excess_mean"])
    )


def two_way_bootstrap_1d(delta: np.ndarray, n_boot: int = 10_000, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(delta)
    means = np.empty(n_boot)
    for b in range(n_boot):
        means[b] = delta[rng.integers(0, n, n)].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def run_many(
    policies: list[Policy],
    tapes: list[int],
    stage: str,
    workers: int,
    campaign: dict[str, Any] | None = None,
    lead: float = 168.0,
):
    tasks = [(stage, p, tapes, campaign, lead) for p in policies]
    if workers == 1:
        blocks = [eval_policy(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            blocks = list(pool.map(eval_policy, tasks, chunksize=1))
    return [r for b in blocks for r in b]


def sobol_policies(n: int, seed: int) -> list[Policy]:
    raw = qmc.Sobol(d=11, scramble=True, seed=seed).random_base2(int(np.log2(n)))
    out = []
    for i, row in enumerate(raw):
        sig = list(2.0 * row[:8] - 1.0) + list(row[8:11])
        out.append(Policy(name=f"sobol_{i}", calm=tuple(float(v) for v in sig)))
    for name, vec in ANCHORS.items():
        out.append(Policy(name=f"anchor_{name}", calm=tuple(vec)))
    return out


def refine_policies(leaders: list[Policy], radius: float) -> list[Policy]:
    seen: dict[tuple, Policy] = {}
    for li, leader in enumerate(leaders):
        base = list(leader.calm)
        seen[tuple(np.round(base, 8))] = Policy(f"ref_{li}_base", tuple(base))
        for dim in range(11):
            lo, hi = (0.0, 1.0) if dim >= 8 else (-1.0, 1.0)
            for direction in (-1.0, 1.0):
                v = list(base)
                v[dim] = float(np.clip(v[dim] + direction * radius, lo, hi))
                key = tuple(np.round(v, 8))
                if key not in seen:
                    seen[key] = Policy(f"ref_{li}_d{dim}_{'+' if direction>0 else '-'}", tuple(v))
    return list(seen.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["baseline", "screen", "refine", "pairs", "verdict", "c2fit", "c2verdict", "all"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--sobol-n", type=int, default=96)
    parser.add_argument("--sobol-seed", type=int, default=20260711)
    parser.add_argument("--refine-top", type=int, default=6)
    parser.add_argument("--refine-radius", type=float, default=0.15)
    parser.add_argument("--pair-top", type=int, default=6)
    parser.add_argument(
        "--campaign-json",
        default=None,
        help="JSON dict of campaign_config knobs for calibration iterations "
        "(default: config.CAMPAIGN_V1_CONFIG). Logged to campaign_config.json.",
    )
    parser.add_argument("--lead", type=float, default=168.0)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    campaign = json.loads(args.campaign_json) if args.campaign_json else None
    lead = float(args.lead)
    from supply_chain.config import CAMPAIGN_V1_CONFIG as _DEFAULT_CFG

    (out / "campaign_config.json").write_text(
        json.dumps({"campaign": campaign or _DEFAULT_CFG, "lead": lead}, indent=2)
    )
    stages = ["baseline", "screen", "refine", "pairs", "verdict", "c2fit", "c2verdict"] if args.stage == "all" else [args.stage]

    def load_json(name):
        return json.loads((out / name).read_text())

    for stage in stages:
        print(f"=== stage {stage}", flush=True)
        if stage == "baseline":
            policies = [Policy(f"anchor_{n}", tuple(v)) for n, v in ANCHORS.items()]
            rows = run_many(policies, TAPES["baseline"], stage, args.workers, campaign=campaign, lead=lead)
            write_rows(out / "baseline_rows.csv", rows)
            base_rows = [r for r in rows if r["policy"] == "anchor_cf0_S1_nostock"]
            ret_base = float(np.mean([r["ret_excel"] for r in base_rows]))
            holding_base = float(np.mean([r["econ_holding_frac_mean"] for r in base_rows]))
            lam = {
                # Frozen BEFORE any optimization (pre-registration §lambdas):
                # a policy holding 2x the Cf_0 stock pays 15% of Cf_0 ReT;
                # full 2.0x/1.5x dispatch costs ~6%; each extra shift ~5%.
                "lam_h": 0.15 * ret_base / max(holding_base, 1e-9),
                "lam_d": 0.05 * ret_base,
                "lam_s": 0.05 * ret_base,
                "ret_base": ret_base,
                "holding_base": holding_base,
            }
            (out / "lambdas.json").write_text(json.dumps(lam, indent=2))
            camp_fracs = [r["econ_campaign_frac"] for r in rows]
            summary = {
                "ret_base": ret_base,
                "holding_base": holding_base,
                "lambdas": lam,
                "campaign_frac_mean": float(np.mean(camp_fracs)),
                "anchor_means_ret": {
                    n: float(np.mean([r["ret_excel"] for r in rows if r["policy"] == f"anchor_{n}"]))
                    for n in ANCHORS
                },
            }
            (out / "baseline_summary.json").write_text(json.dumps(summary, indent=2))
            print(json.dumps(summary, indent=2), flush=True)

        elif stage == "screen":
            lam = load_json("lambdas.json")
            policies = sobol_policies(args.sobol_n, args.sobol_seed)
            rows = run_many(policies, TAPES["screen"], stage, args.workers, campaign=campaign, lead=lead)
            for r in rows:
                r["J"] = j_of(r, lam)
            write_rows(out / "screen_rows.csv", rows)
            by_policy: dict[str, list[float]] = {}
            vec_by_policy: dict[str, Policy] = {p.name: p for p in policies}
            for r in rows:
                by_policy.setdefault(r["policy"], []).append(r["J"])
            ranked = sorted(by_policy.items(), key=lambda kv: -float(np.mean(kv[1])))
            leaders = [
                {"policy": name, "J_mean": float(np.mean(js)), "calm": list(vec_by_policy[name].calm)}
                for name, js in ranked[: args.refine_top]
            ]
            (out / "screen_leaders.json").write_text(json.dumps(leaders, indent=2))
            print(json.dumps(leaders[:3], indent=2), flush=True)

        elif stage == "refine":
            lam = load_json("lambdas.json")
            leaders = [Policy(l["policy"], tuple(l["calm"])) for l in load_json("screen_leaders.json")]
            policies = refine_policies(leaders, args.refine_radius)
            print(f"refine candidates: {len(policies)}", flush=True)
            rows = run_many(policies, TAPES["refine"], stage, args.workers, campaign=campaign, lead=lead)
            for r in rows:
                r["J"] = j_of(r, lam)
            write_rows(out / "refine_rows.csv", rows)
            by_policy: dict[str, list[float]] = {}
            vec_by_policy = {p.name: p for p in policies}
            for r in rows:
                by_policy.setdefault(r["policy"], []).append(r["J"])
            ranked = sorted(by_policy.items(), key=lambda kv: -float(np.mean(kv[1])))
            frozen = {
                "constant": {
                    "policy": ranked[0][0],
                    "J_mean_calibration": float(np.mean(ranked[0][1])),
                    "calm": list(vec_by_policy[ranked[0][0]].calm),
                },
                "top_for_pairs": [
                    {"policy": n, "J_mean": float(np.mean(js)), "calm": list(vec_by_policy[n].calm)}
                    for n, js in ranked[: args.pair_top]
                ],
            }
            (out / "frozen_constant.json").write_text(json.dumps(frozen, indent=2))
            print(json.dumps(frozen["constant"], indent=2), flush=True)

        elif stage == "pairs":
            lam = load_json("lambdas.json")
            frozen = load_json("frozen_constant.json")
            tops = frozen["top_for_pairs"]
            policies: list[Policy] = []
            for i, ci in enumerate(tops):
                for j, cj in enumerate(tops):
                    if i == j:
                        continue
                    policies.append(
                        Policy(f"pair_{i}_{j}", calm=tuple(ci["calm"]), campaign=tuple(cj["calm"]))
                    )
            policies.append(Policy("pair_lean_heavy", calm=tuple(ANCHORS["lean"]), campaign=tuple(ANCHORS["heavy"])))
            # The frozen constant itself, as the pair-stage reference:
            policies.append(Policy("constant_ref", calm=tuple(frozen["constant"]["calm"])))
            print(f"pair candidates: {len(policies)}", flush=True)
            rows = run_many(policies, TAPES["pairs"], stage, args.workers, campaign=campaign, lead=lead)
            for r in rows:
                r["J"] = j_of(r, lam)
            write_rows(out / "pairs_rows.csv", rows)
            by_policy: dict[str, list[float]] = {}
            spec = {p.name: p for p in policies}
            for r in rows:
                by_policy.setdefault(r["policy"], []).append(r["J"])
            switchers = {n: v for n, v in by_policy.items() if spec[n].is_switching()}
            best = max(switchers.items(), key=lambda kv: float(np.mean(kv[1])))
            frozen_pair = {
                "policy": best[0],
                "J_mean_fit": float(np.mean(best[1])),
                "calm": list(spec[best[0]].calm),
                "campaign": list(spec[best[0]].campaign),
                "constant_ref_J_mean_fit": float(np.mean(by_policy["constant_ref"])),
            }
            (out / "frozen_pair.json").write_text(json.dumps(frozen_pair, indent=2))
            print(json.dumps(frozen_pair, indent=2), flush=True)

        elif stage == "verdict":
            lam = load_json("lambdas.json")
            const = load_json("frozen_constant.json")["constant"]
            pair = load_json("frozen_pair.json")
            policies = [
                Policy("frozen_constant", calm=tuple(const["calm"])),
                Policy("frozen_switcher", calm=tuple(pair["calm"]), campaign=tuple(pair["campaign"])),
                Policy("anchor_cf0", calm=tuple(ANCHORS["cf0_S1_nostock"])),
                Policy("anchor_I1344_S2", calm=tuple(ANCHORS["garrido_I1344_S2"])),
            ]
            rows = run_many(policies, TAPES["verdict"], stage, args.workers, campaign=campaign, lead=lead)
            for r in rows:
                r["J"] = j_of(r, lam)
            write_rows(out / "verdict_rows.csv", rows)
            by: dict[tuple[str, int], float] = {(r["policy"], r["eval_seed"]): r["J"] for r in rows}
            tapes = TAPES["verdict"]
            delta = np.array([by[("frozen_switcher", t)] - by[("frozen_constant", t)] for t in tapes])
            lo, hi = two_way_bootstrap_1d(delta)
            ret_base = lam["ret_base"]
            threshold = 0.05 * ret_base
            passed = bool(lo > 0 and delta.mean() >= threshold)
            result = {
                "gate": "C1",
                "switcher_minus_constant_J": {
                    "mean": float(delta.mean()),
                    "ci95": [lo, hi],
                    "tapes_positive": int((delta > 0).sum()),
                    "n_tapes": len(tapes),
                },
                "threshold_005_ret_base": threshold,
                "means_J": {p.name: float(np.mean([by[(p.name, t)] for t in tapes])) for p in policies},
                "passed": passed,
                "verdict": "PROMOTE_TO_C2" if passed else "ITERATE_ENV_OR_STOP",
            }
            (out / "c1_verdict.json").write_text(json.dumps(result, indent=2))
            print(json.dumps(result, indent=2), flush=True)

        elif stage == "c2fit":
            lam = load_json("lambdas.json")
            pair = load_json("frozen_pair.json")
            policies = []
            for theta in (0.5, 0.75, 1.0, 1.5, 2.0):
                for hl in (2.0, 3.0, 4.0):
                    policies.append(
                        Policy(
                            f"det_t{theta}_hl{hl}",
                            calm=tuple(pair["calm"]),
                            campaign=tuple(pair["campaign"]),
                            detector=(theta, hl),
                        )
                    )
            rows = run_many(policies, TAPES["pairs"], stage, args.workers, campaign=campaign, lead=lead)
            for r in rows:
                r["J"] = j_of(r, lam)
            write_rows(out / "c2fit_rows.csv", rows)
            by_policy: dict[str, list[float]] = {}
            for r in rows:
                by_policy.setdefault(r["policy"], []).append(r["J"])
            best = max(by_policy.items(), key=lambda kv: float(np.mean(kv[1])))
            name = best[0]
            theta = float(name.split("_t")[1].split("_hl")[0])
            hl = float(name.split("_hl")[1])
            frozen_det = {"policy": name, "theta": theta, "halflife_weeks": hl,
                          "J_mean_fit": float(np.mean(best[1]))}
            (out / "frozen_detector.json").write_text(json.dumps(frozen_det, indent=2))
            print(json.dumps(frozen_det, indent=2), flush=True)

        elif stage == "c2verdict":
            lam = load_json("lambdas.json")
            const = load_json("frozen_constant.json")["constant"]
            pair = load_json("frozen_pair.json")
            det = load_json("frozen_detector.json")
            c1 = load_json("c1_verdict.json")
            policy = Policy(
                "detector_switcher",
                calm=tuple(pair["calm"]),
                campaign=tuple(pair["campaign"]),
                detector=(det["theta"], det["halflife_weeks"]),
            )
            rows = run_many([policy], TAPES["verdict"], stage, args.workers, campaign=campaign, lead=lead)
            for r in rows:
                r["J"] = j_of(r, lam)
            write_rows(out / "c2verdict_rows.csv", rows)
            # Constant J per tape from the C1 verdict ledger:
            crows = list(csv.DictReader((out / "verdict_rows.csv").open()))
            const_by = {int(r["eval_seed"]): float(r["J"]) for r in crows if r["policy"] == "frozen_constant"}
            det_by = {int(r["eval_seed"]): float(r["J"]) for r in rows}
            tapes = TAPES["verdict"]
            delta = np.array([det_by[t] - const_by[t] for t in tapes])
            lo, hi = two_way_bootstrap_1d(delta)
            c1_gap = float(c1["switcher_minus_constant_J"]["mean"])
            capture = float(delta.mean()) / c1_gap if c1_gap > 0 else float("nan")
            passed = bool(capture >= 0.5 and lo > 0)
            result = {
                "gate": "C2",
                "detector_minus_constant_J": {"mean": float(delta.mean()), "ci95": [lo, hi],
                                              "tapes_positive": int((delta > 0).sum())},
                "c1_gap": c1_gap,
                "capture_ratio": capture,
                "passed": passed,
                "verdict": "PROMOTE_TO_C3_TRAINING" if passed else "REGIME_UNDETECTABLE_ITERATE_OR_STOP",
            }
            (out / "c2_verdict.json").write_text(json.dumps(result, indent=2))
            print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
