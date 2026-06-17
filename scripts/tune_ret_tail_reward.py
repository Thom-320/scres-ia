#!/usr/bin/env python3
"""Grid-search ReT_tail_v1 reward parameters on the static policy surface.

This is deliberately a pre-training tuner.  It does not run PPO.  It asks a
smaller, safer question first: if a static policy maximizes a candidate reward,
is that same policy also good on the external tail metrics we care about?

The default grid is a bounded scout over a panel subset.  Promote the top rows
to a full-panel confirmation before using them for expensive RL training.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
AUDITOR = REPO / "scripts" / "audit_garrido_metric_saturation.py"
DEFAULT_OUTPUT_ROOT = REPO / "outputs" / "benchmarks" / "ret_tail_reward_tuning"


@dataclass(frozen=True)
class Candidate:
    w_sc: float
    w_rc: float
    w_ce: float
    cap_kappa: float
    inv_kappa: float
    boost: float

    @property
    def label(self) -> str:
        parts = [
            f"sc{self.w_sc:.2f}",
            f"rc{self.w_rc:.2f}",
            f"ce{self.w_ce:.2f}",
            f"cap{self.cap_kappa:.2f}",
            f"inv{self.inv_kappa:.2f}",
            f"boost{self.boost:.2f}",
        ]
        return "ret_tail_" + "_".join(p.replace(".", "p") for p in parts)


def parse_float_csv(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_weight_triplets(value: str) -> list[tuple[float, float, float]]:
    """Parse `w_sc:w_rc:w_ce;...`."""
    out: list[tuple[float, float, float]] = []
    for chunk in value.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        pieces = [float(part.strip()) for part in chunk.split(":")]
        if len(pieces) != 3:
            raise ValueError(f"Expected w_sc:w_rc:w_ce, got {chunk!r}")
        total = sum(pieces)
        if total <= 0.0:
            raise ValueError(f"Weight triplet must have positive sum: {chunk!r}")
        out.append(tuple(round(value / total, 6) for value in pieces))
    return out


def generate_weight_grid(
    *,
    step: float,
    w_sc_min: float,
    w_rc_min: float,
    w_ce_min: float,
) -> list[tuple[float, float, float]]:
    if not 0.0 < step <= 1.0:
        raise ValueError("--weight-step must be in (0, 1].")
    scale = int(round(1.0 / step))
    weights: list[tuple[float, float, float]] = []
    for sc_i in range(scale + 1):
        for rc_i in range(scale + 1 - sc_i):
            ce_i = scale - sc_i - rc_i
            w_sc = round(sc_i / scale, 6)
            w_rc = round(rc_i / scale, 6)
            w_ce = round(ce_i / scale, 6)
            if w_sc < w_sc_min or w_rc < w_rc_min or w_ce < w_ce_min:
                continue
            if w_rc < max(w_sc, w_ce):
                continue
            weights.append((w_sc, w_rc, w_ce))
    return weights


def build_candidates(args: argparse.Namespace) -> list[Candidate]:
    if args.weights:
        weights = parse_weight_triplets(args.weights)
    else:
        weights = generate_weight_grid(
            step=args.weight_step,
            w_sc_min=args.w_sc_min,
            w_rc_min=args.w_rc_min,
            w_ce_min=args.w_ce_min,
        )
    candidates = [
        Candidate(
            w_sc=w_sc,
            w_rc=w_rc,
            w_ce=w_ce,
            cap_kappa=cap_kappa,
            inv_kappa=inv_kappa,
            boost=boost,
        )
        for w_sc, w_rc, w_ce in weights
        for cap_kappa in parse_float_csv(args.cap_kappas)
        for inv_kappa in parse_float_csv(args.inv_kappas)
        for boost in parse_float_csv(args.boosts)
    ]
    if args.max_candidates and args.max_candidates > 0:
        candidates = candidates[: args.max_candidates]
    return candidates


def build_auditor_command(
    *, args: argparse.Namespace, candidate: Candidate, out_root: Path
) -> list[str]:
    return [
        sys.executable,
        str(AUDITOR),
        "--profiles",
        args.profiles,
        "--panel-cfis",
        args.panel_cfis,
        "--policy-set",
        args.policy_set,
        "--replications",
        str(args.replications),
        "--raw-material-flow-mode",
        "kit_equivalent_order_up_to",
        "--raw-material-order-up-to-multiplier",
        str(args.multiplier),
        "--risk-occurrence-mode",
        "thesis_periodic",
        "--reward-mode",
        "ReT_tail_v1",
        "--ret-tail-w-sc",
        str(candidate.w_sc),
        "--ret-tail-w-rc",
        str(candidate.w_rc),
        "--ret-tail-w-ce",
        str(candidate.w_ce),
        "--ret-tail-cap-kappa",
        str(candidate.cap_kappa),
        "--ret-tail-inv-kappa",
        str(candidate.inv_kappa),
        "--ret-tail-boost",
        str(candidate.boost),
        "--label",
        candidate.label,
        "--output-root",
        str(out_root),
        "--progress-every",
        str(args.progress_every),
    ]


def spearman(a: pd.Series, b: pd.Series) -> float:
    if a.nunique() < 2 or b.nunique() < 2:
        return float("nan")
    return float(a.rank().corr(b.rank()))


def score_profile(g: pd.DataFrame, *, top_k: int, p10_floor: float) -> dict[str, object]:
    best = g.loc[g["reward_total"].idxmax()]
    top_p10 = float(g["ret_p10_all"].max())
    top_flow = float(g["flow_fill_rate"].max())
    best_p10 = float(best["ret_p10_all"])
    best_flow = float(best["flow_fill_rate"])
    best_stockout = float(best["stockout_week_pct"])
    p10_rank = int((g["ret_p10_all"] > best_p10 + 1e-9).sum()) + 1
    p10_ratio = best_p10 / max(top_p10, 1e-9)
    flow_ratio = best_flow / max(top_flow, 1e-9)
    stockout_score = max(0.0, min(1.0, 1.0 - best_stockout / 100.0))
    rho_p10 = spearman(g["reward_total"], g["ret_p10_all"])
    rho_stockout = spearman(g["reward_total"], g["stockout_week_pct"])
    rho_p10_score = 0.0 if pd.isna(rho_p10) else (rho_p10 + 1.0) / 2.0
    score = (
        0.45 * p10_ratio
        + 0.25 * flow_ratio
        + 0.20 * stockout_score
        + 0.10 * rho_p10_score
    )
    return {
        "best_by_reward": best["policy"],
        "best_reward_p10": best_p10,
        "best_p10_rank": p10_rank,
        "top_p10": top_p10,
        "p10_ratio": p10_ratio,
        "best_reward_flow_fill": best_flow,
        "top_flow_fill": top_flow,
        "flow_ratio": flow_ratio,
        "best_reward_stockout_week_pct": best_stockout,
        "rho_ret_p10_all": rho_p10,
        "rho_flow_fill_rate": spearman(g["reward_total"], g["flow_fill_rate"]),
        "rho_stockout_week_pct": rho_stockout,
        "profile_score": score,
        "PASS": bool(p10_rank <= top_k and best_p10 >= p10_floor * max(top_p10, 1e-9)),
    }


def analyze_candidate(csv_path: Path, candidate: Candidate, args: argparse.Namespace) -> dict[str, object]:
    df = pd.read_csv(csv_path)
    profile_rows: list[dict[str, object]] = []
    for profile in [part.strip() for part in args.profiles.split(",") if part.strip()]:
        g = (
            df[df["profile"] == profile]
            .groupby("policy")
            .mean(numeric_only=True)
            .reset_index()
        )
        if g.empty:
            continue
        row = score_profile(g, top_k=args.top_k, p10_floor=args.p10_floor)
        row["profile"] = profile
        profile_rows.append(row)
    if not profile_rows:
        raise RuntimeError(f"No profile rows in {csv_path}")
    out: dict[str, object] = asdict(candidate)
    out["candidate"] = candidate.label
    out["profile_count"] = len(profile_rows)
    out["all_pass"] = all(bool(row["PASS"]) for row in profile_rows)
    out["mean_score"] = float(pd.Series([row["profile_score"] for row in profile_rows]).mean())
    out["mean_p10_ratio"] = float(pd.Series([row["p10_ratio"] for row in profile_rows]).mean())
    out["mean_flow_ratio"] = float(pd.Series([row["flow_ratio"] for row in profile_rows]).mean())
    out["mean_stockout_pct"] = float(
        pd.Series([row["best_reward_stockout_week_pct"] for row in profile_rows]).mean()
    )
    out["mean_best_p10_rank"] = float(
        pd.Series([row["best_p10_rank"] for row in profile_rows]).mean()
    )
    out["mean_rho_ret_p10_all"] = float(
        pd.Series([row["rho_ret_p10_all"] for row in profile_rows]).mean()
    )
    for row in profile_rows:
        profile = str(row["profile"])
        for key, value in row.items():
            if key != "profile":
                out[f"{profile}_{key}"] = value
    return out


def run_candidate(
    *, args: argparse.Namespace, candidate: Candidate, out_root: Path
) -> Path:
    csv_path = out_root / candidate.label / "episode_metric_audit.csv"
    if csv_path.exists() and csv_path.stat().st_size > 0 and not args.force:
        print(f"[skip] {candidate.label}")
        return csv_path
    cmd = build_auditor_command(args=args, candidate=candidate, out_root=out_root)
    if args.dry_run:
        print(" ".join(cmd))
        return csv_path
    print(f"[run ] {candidate.label}", flush=True)
    env_extra = {"KMP_DUPLICATE_LIB_OK": "TRUE", "OMP_NUM_THREADS": "1"}
    proc = subprocess.run(
        cmd,
        cwd=str(REPO),
        env={**os.environ, **env_extra},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or not csv_path.exists():
        print(proc.stdout[-2000:])
        raise RuntimeError(f"candidate failed: {candidate.label}")
    return csv_path


def write_report(out_root: Path, summary: pd.DataFrame, args: argparse.Namespace) -> None:
    top = summary.head(10)
    lines = [
        "# ReT_tail_v1 Reward Tuning",
        "",
        f"Created UTC: `{datetime.now(timezone.utc).isoformat()}`",
        f"Profiles: `{args.profiles}`; panel: `{args.panel_cfis}`; "
        f"policy_set: `{args.policy_set}`; multiplier: `{args.multiplier}`.",
        "",
        "Score = 0.45*p10_ratio + 0.25*flow_ratio + "
        "0.20*(1-stockout_pct/100) + 0.10*scaled_spearman_p10.",
        "The score is only a selector for reward parameters; final wins must be "
        "judged by held-out PPO/static evaluation metrics, not reward_total.",
        "",
        "## Top Candidates",
        "",
        "```text",
        top.to_string(index=False),
        "```",
        "",
    ]
    (out_root / "RET_TAIL_TUNING.md").write_text("\n".join(lines), encoding="utf-8")


def jsonable_args(args: argparse.Namespace) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--profiles", default="increased,severe")
    parser.add_argument("--panel-cfis", default="31,51,71")
    parser.add_argument("--policy-set", default="with_crossed")
    parser.add_argument("--replications", type=int, default=1)
    parser.add_argument("--multiplier", type=float, default=2.0)
    parser.add_argument("--weight-step", type=float, default=0.10)
    parser.add_argument("--w-sc-min", type=float, default=0.10)
    parser.add_argument("--w-rc-min", type=float, default=0.45)
    parser.add_argument("--w-ce-min", type=float, default=0.05)
    parser.add_argument(
        "--weights",
        default=None,
        help="Optional semicolon list of normalized or unnormalized w_sc:w_rc:w_ce.",
    )
    parser.add_argument("--cap-kappas", default="0.10,0.25,0.40")
    parser.add_argument("--inv-kappas", default="0.25,0.50,0.75")
    parser.add_argument("--boosts", default="0.0,1.0,2.0")
    parser.add_argument("--max-candidates", type=int, default=40)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--p10-floor", type=float, default=0.5)
    parser.add_argument("--progress-every", type=int, default=200)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    label = datetime.now(timezone.utc).strftime("ret_tail_tuning_%Y%m%dT%H%M%SZ")
    out_root = args.output_root / label
    out_root.mkdir(parents=True, exist_ok=True)
    candidates = build_candidates(args)
    if not candidates:
        raise ValueError("No candidates generated. Relax weight/kappa constraints.")
    print(f"Generated {len(candidates)} candidates -> {out_root}")
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        try:
            csv_path = run_candidate(args=args, candidate=candidate, out_root=out_root)
            if args.dry_run:
                continue
            rows.append(analyze_candidate(csv_path, candidate, args))
            summary = pd.DataFrame(rows).sort_values(
                ["all_pass", "mean_score", "mean_p10_ratio", "mean_best_p10_rank"],
                ascending=[False, False, False, True],
            )
            summary.to_csv(out_root / "ret_tail_tuning_summary.csv", index=False)
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {candidate.label}: {exc}")
            if args.stop_on_error:
                raise
    if args.dry_run:
        return 0
    if not rows:
        print("No successful candidates.")
        return 1
    summary = pd.DataFrame(rows).sort_values(
        ["all_pass", "mean_score", "mean_p10_ratio", "mean_best_p10_rank"],
        ascending=[False, False, False, True],
    )
    summary_path = out_root / "ret_tail_tuning_summary.csv"
    summary.to_csv(summary_path, index=False)
    (out_root / "manifest.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "args": jsonable_args(args),
                "candidate_count": len(candidates),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    write_report(out_root, summary, args)
    pd.set_option("display.width", 240)
    print(summary.head(12).to_string(index=False))
    print(f"Saved {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
