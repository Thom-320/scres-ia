#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_COMMITS: tuple[str, ...] = (
    "5f17368",
    "85dc014",
    "881c713",
    "4274857",
    "2f4e638",
    "HEAD",
)
DEFAULT_POLICIES: tuple[str, ...] = ("static_s1", "static_s2", "static_s3")
LEGACY_SUMMARY_PATH = Path(
    "outputs/benchmarks/control_reward_500k_increased_stopt/summary.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit static-policy regression across commits to identify when the "
            "benchmark backbone diverged from the legacy frozen bundle."
        )
    )
    parser.add_argument(
        "--commits",
        nargs="+",
        default=list(DEFAULT_COMMITS),
        help="Commit refs to audit in chronological order. Include HEAD if desired.",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=list(DEFAULT_POLICIES),
        choices=list(DEFAULT_POLICIES),
        help="Static policies to evaluate per commit.",
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--reward-mode", default="control_v1")
    parser.add_argument("--observation-version", default="v1")
    parser.add_argument("--frame-stack", type=int, default=1)
    parser.add_argument("--risk-level", default="increased")
    parser.add_argument("--stochastic-pt", action="store_true")
    parser.add_argument("--year-basis", default="thesis")
    parser.add_argument("--step-size-hours", type=float, default=168.0)
    parser.add_argument("--max-steps", type=int, default=260)
    parser.add_argument("--w-bo", type=float, default=4.0)
    parser.add_argument("--w-cost", type=float, default=0.02)
    parser.add_argument("--w-disr", type=float, default=0.0)
    parser.add_argument(
        "--legacy-summary",
        type=Path,
        default=LEGACY_SUMMARY_PATH,
        help="Optional frozen bundle summary used as the historical comparison point.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/audits/backbone_regression"),
    )
    return parser


def resolve_repo_root() -> Path:
    return Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )


def resolve_commit_sha(repo_root: Path, ref: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", ref],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def build_probe_code() -> str:
    return """
import json
import numpy as np
import sys
import inspect
from supply_chain.external_env_interface import make_shift_control_env
from supply_chain.external_env_interface import get_episode_terminal_metrics
from supply_chain.env_experimental_shifts import MFSCGymEnvShifts

try:
    from supply_chain.external_env_interface import run_episodes
except ImportError:  # legacy commits pre-date helper export
    run_episodes = None

FIXED_POLICY_ACTIONS = {
    "static_s1": np.array([0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32),
    "static_s2": np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    "static_s3": np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
}

def make_policy(name):
    action = FIXED_POLICY_ACTIONS[name]
    def _policy(obs, info):
        return action.copy()
    return _policy

def run_episodes_fallback(policy_fn, *, n_episodes, seed, env_kwargs, policy_name):
    del policy_name
    rows = []
    accepted = inspect.signature(MFSCGymEnvShifts.__init__).parameters
    filtered_env_kwargs = {
        key: value for key, value in env_kwargs.items() if key in accepted
    }
    env = make_shift_control_env(**filtered_env_kwargs)
    try:
        for ep in range(int(n_episodes)):
            obs, info = env.reset(seed=int(seed) + ep)
            reward_total = 0.0
            ret_corr = 0.0
            demanded_total = 0.0
            backorder_qty_total = 0.0
            terminated = False
            truncated = False
            last_info = info
            counts = {1: 0, 2: 0, 3: 0}
            steps = 0
            while not (terminated or truncated):
                action = policy_fn(obs, last_info)
                obs, reward, terminated, truncated, last_info = env.step(action)
                reward_total += float(reward)
                ret_corr += float(last_info.get("ret_thesis_corrected_step", 0.0))
                demanded_total += float(last_info.get("new_demanded", 0.0))
                backorder_qty_total += float(last_info.get("new_backorder_qty", 0.0))
                shift = int(last_info.get("shifts_active", 0))
                if shift in counts:
                    counts[shift] += 1
                steps += 1
            terminal_metrics = get_episode_terminal_metrics(env)
            denom = max(steps, 1)
            rows.append(
                {
                    "reward_total": reward_total,
                    "fill_rate": float(terminal_metrics["fill_rate_order_level"]),
                    "backorder_rate": float(
                        terminal_metrics["backorder_rate_order_level"]
                    ),
                    "ret_thesis_corrected_total": ret_corr,
                    "order_level_ret_mean": float(
                        terminal_metrics["order_level_ret_mean"]
                    ),
                    "pct_steps_S1": 100.0 * counts[1] / denom,
                    "pct_steps_S2": 100.0 * counts[2] / denom,
                    "pct_steps_S3": 100.0 * counts[3] / denom,
                }
            )
    finally:
        env.close()
    return rows

payload = json.loads(sys.argv[1])
rows = []
for policy_name in payload["policies"]:
    env_kwargs = {
        "reward_mode": payload["reward_mode"],
        "observation_version": payload["observation_version"],
        "step_size_hours": float(payload["step_size_hours"]),
        "risk_level": payload["risk_level"],
        "stochastic_pt": bool(payload["stochastic_pt"]),
        "max_steps": int(payload["max_steps"]),
        "year_basis": payload["year_basis"],
        "w_bo": float(payload["w_bo"]),
        "w_cost": float(payload["w_cost"]),
        "w_disr": float(payload["w_disr"]),
    }
    if run_episodes is not None:
        accepted = inspect.signature(make_shift_control_env).parameters
        accepts_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in accepted.values()
        )
        env_kwargs = (
            dict(env_kwargs)
            if accepts_kwargs
            else {key: value for key, value in env_kwargs.items() if key in accepted}
        )
    if run_episodes is not None:
        episode_rows = run_episodes(
            make_policy(policy_name),
            n_episodes=int(payload["episodes"]),
            seed=int(payload["seed"]),
            env_kwargs=env_kwargs,
            policy_name=policy_name,
        )
    else:
        episode_rows = run_episodes_fallback(
            make_policy(policy_name),
            n_episodes=int(payload["episodes"]),
            seed=int(payload["seed"]),
            env_kwargs=env_kwargs,
            policy_name=policy_name,
        )
    rows.append(
        {
            "policy": policy_name,
            "fill_rate_mean": float(np.mean([row["fill_rate"] for row in episode_rows])),
            "backorder_rate_mean": float(np.mean([row["backorder_rate"] for row in episode_rows])),
            "ret_thesis_corrected_total_mean": float(
                np.mean([row["ret_thesis_corrected_total"] for row in episode_rows])
            ),
            "order_level_ret_mean": float(
                np.mean(
                    [
                        row.get("order_level_ret_mean", row["fill_rate"])
                        for row in episode_rows
                    ]
                )
            ),
            "reward_total_mean": float(np.mean([row["reward_total"] for row in episode_rows])),
        }
    )
print(json.dumps(rows))
"""


def evaluate_commit(
    *,
    repo_root: Path,
    ref: str,
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    if ref == "HEAD":
        completed = subprocess.run(
            [sys.executable, "-c", build_probe_code(), json.dumps(payload)],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(completed.stdout)

    with tempfile.TemporaryDirectory(prefix=f"backbone_audit_{ref}_") as tmpdir:
        worktree_dir = Path(tmpdir) / "repo"
        subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "worktree",
                "add",
                "--detach",
                str(worktree_dir),
                ref,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        try:
            completed = subprocess.run(
                [sys.executable, "-c", build_probe_code(), json.dumps(payload)],
                cwd=worktree_dir,
                check=True,
                capture_output=True,
                text=True,
            )
            return json.loads(completed.stdout)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Failed to evaluate ref {ref} in {worktree_dir}:\n{exc.stderr}"
            ) from exc
        finally:
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo_root),
                    "worktree",
                    "remove",
                    "--force",
                    str(worktree_dir),
                ],
                check=True,
                capture_output=True,
                text=True,
            )


def load_legacy_reference(summary_path: Path) -> dict[str, dict[str, float]]:
    if not summary_path.exists():
        return {}
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = payload.get("policy_summary", [])
    legacy: dict[str, dict[str, float]] = {}
    for row in rows:
        policy = row.get("policy")
        if policy not in DEFAULT_POLICIES:
            continue
        legacy[str(policy)] = {
            "fill_rate_mean": float(row["fill_rate_mean"]),
            "backorder_rate_mean": float(row["backorder_rate_mean"]),
            "ret_thesis_corrected_total_mean": float(
                row["ret_thesis_corrected_total_mean"]
            ),
            "reward_total_mean": float(row["reward_total_mean"]),
            "order_level_ret_mean": float(
                row.get("order_level_ret_mean_mean", row["fill_rate_mean"])
            ),
        }
    return legacy


def build_markdown_report(
    *,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    legacy_reference: dict[str, dict[str, float]],
) -> str:
    lines = [
        "# Backbone Regression Audit",
        "",
        f"- Generated: {datetime.now(timezone.utc).isoformat()}",
        f"- Reward mode: `{args.reward_mode}`",
        f"- Backbone under test: `obs={args.observation_version}`, `fs={args.frame_stack}`, `year_basis={args.year_basis}`, `risk={args.risk_level}`, `stochastic_pt={args.stochastic_pt}`",
        "",
        "## Legacy reference",
        "",
    ]
    if legacy_reference:
        for policy in args.policies:
            ref = legacy_reference.get(policy)
            if not ref:
                continue
            lines.append(
                f"- `{policy}`: fill={ref['fill_rate_mean']:.3f}, backorder={ref['backorder_rate_mean']:.3f}, order-level-ReT={ref['order_level_ret_mean']:.3f}, ReTcorr={ref['ret_thesis_corrected_total_mean']:.3f}"
            )
    else:
        lines.append("- Legacy summary not found.")

    lines.extend(["", "## Commit audit", ""])
    lines.append(
        "| ref | commit | policy | fill | backorder | order-level-ReT | ReTcorr | delta_fill_vs_legacy |"
    )
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        legacy_fill = legacy_reference.get(row["policy"], {}).get("fill_rate_mean")
        delta_fill = (
            row["fill_rate_mean"] - legacy_fill
            if legacy_fill is not None
            else float("nan")
        )
        delta_fill_str = f"{delta_fill:+.3f}" if legacy_fill is not None else "n/a"
        lines.append(
            f"| {row['ref']} | `{row['commit'][:10]}` | `{row['policy']}` | "
            f"{row['fill_rate_mean']:.3f} | {row['backorder_rate_mean']:.3f} | "
            f"{row.get('order_level_ret_mean', math.nan):.3f} | "
            f"{row['ret_thesis_corrected_total_mean']:.3f} | {delta_fill_str} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    repo_root = resolve_repo_root()
    legacy_reference = load_legacy_reference(args.legacy_summary)

    payload = {
        "episodes": int(args.episodes),
        "seed": int(args.seed),
        "policies": [str(policy) for policy in args.policies],
        "reward_mode": str(args.reward_mode),
        "observation_version": str(args.observation_version),
        "frame_stack": int(args.frame_stack),
        "risk_level": str(args.risk_level),
        "stochastic_pt": bool(args.stochastic_pt),
        "year_basis": str(args.year_basis),
        "step_size_hours": float(args.step_size_hours),
        "max_steps": int(args.max_steps),
        "w_bo": float(args.w_bo),
        "w_cost": float(args.w_cost),
        "w_disr": float(args.w_disr),
    }

    rows: list[dict[str, Any]] = []
    for ref in args.commits:
        commit_sha = resolve_commit_sha(repo_root, ref)
        evaluated_rows = evaluate_commit(repo_root=repo_root, ref=ref, payload=payload)
        for row in evaluated_rows:
            rows.append(
                {
                    "ref": ref,
                    "commit": commit_sha,
                    **row,
                }
            )

    csv_path = args.output_dir / "commit_backbone_regression.csv"
    json_path = args.output_dir / "summary.json"
    md_path = args.output_dir / "audit_report.md"

    fieldnames = [
        "ref",
        "commit",
        "policy",
        "fill_rate_mean",
        "backorder_rate_mean",
        "order_level_ret_mean",
        "ret_thesis_corrected_total_mean",
        "reward_total_mean",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "legacy_summary": (
            str(args.legacy_summary.resolve()) if args.legacy_summary.exists() else None
        ),
        "legacy_reference": legacy_reference,
        "audit_backbone": {
            "reward_mode": args.reward_mode,
            "observation_version": args.observation_version,
            "frame_stack": args.frame_stack,
            "year_basis": args.year_basis,
            "risk_level": args.risk_level,
            "stochastic_pt": args.stochastic_pt,
            "step_size_hours": args.step_size_hours,
            "max_steps": args.max_steps,
            "episodes": args.episodes,
            "seed": args.seed,
        },
        "commits": list(args.commits),
        "rows": rows,
        "artifacts": {
            "csv": str(csv_path.resolve()),
            "summary_json": str(json_path.resolve()),
            "report_md": str(md_path.resolve()),
        },
    }
    json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    md_path.write_text(
        build_markdown_report(args=args, rows=rows, legacy_reference=legacy_reference),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
