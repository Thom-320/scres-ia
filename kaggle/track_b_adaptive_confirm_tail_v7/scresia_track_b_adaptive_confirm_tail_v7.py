"""Kaggle kernel: Track B adaptive ReT_tail_v2/v7 confirmatory run.

This is the clean follow-up to the 2026-07-01 adaptive sweep.  The sweep's
best raw Excel/ReT candidate was ReT_tail_v2 with v7 observations.  This kernel
runs only that candidate, writes output under one directory, and keeps the repo
clone outside /kaggle/working so Kaggle output download stays small.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from typing import Any


SCRIPT_PAYLOAD = Path(__file__).resolve().with_name("scres_ia_payload.tar.gz")
DATASET_PAYLOAD = Path("/kaggle/input/scres-ia-adaptive-sweep-payload/scres_ia_payload.tar.gz")
REPO_DIR = Path("/tmp/scres-ia")
OUTPUT_DIR = Path("/kaggle/working/track_b_adaptive_confirm_tail_v7")
LOCAL_REPO_DIR = Path("outputs/kaggle/track_b_adaptive_confirm_tail_v7/_payload_repo")
LOCAL_OUTPUT_DIR = Path("outputs/kaggle/track_b_adaptive_confirm_tail_v7")
GIT_REPO_URL = "https://github.com/Thom-320/scres-ia.git"
GIT_BRANCH = "codex/garrido-replication-experiments"


CANDIDATE: dict[str, Any] = {
    "label": "ret_tail_v2_v7",
    "reward_mode": "ReT_tail_v2",
    "observation_version": "v7",
    "ret_excel_cvar_alpha": None,
}


def is_kaggle() -> bool:
    return Path("/kaggle/working").exists()


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def find_payload() -> Path | None:
    candidates = [SCRIPT_PAYLOAD, DATASET_PAYLOAD, Path("scres_ia_payload.tar.gz")]
    if is_kaggle():
        candidates.extend(sorted(Path("/kaggle/input").glob("**/scres_ia_payload.tar.gz")))
    for candidate in candidates:
        if candidate.exists():
            print(f"[kernel] using payload {candidate}", flush=True)
            return candidate
    return None


def prepare_repo() -> Path:
    if not is_kaggle():
        cwd = Path.cwd().resolve()
        if (cwd / "scripts" / "run_track_b_smoke.py").exists() and (
            cwd / "supply_chain" / "supply_chain.py"
        ).exists():
            print(f"[kernel] local smoke using current repo {cwd}", flush=True)
            return cwd

    repo_dir = REPO_DIR if is_kaggle() else LOCAL_REPO_DIR.resolve()
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    payload = find_payload()
    if payload is None:
        print(f"[kernel] payload unavailable; cloning {GIT_BRANCH} into {repo_dir}", flush=True)
        run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                GIT_BRANCH,
                GIT_REPO_URL,
                str(repo_dir),
            ]
        )
    else:
        repo_dir.mkdir(parents=True, exist_ok=True)
        print(f"[kernel] extracting {payload} -> {repo_dir}", flush=True)
        with tarfile.open(payload, "r:gz") as archive:
            archive.extractall(repo_dir)

    required = [
        repo_dir / "requirements.txt",
        repo_dir / "scripts" / "run_track_b_smoke.py",
        repo_dir / "supply_chain" / "supply_chain.py",
        repo_dir / "supply_chain" / "external_env_interface.py",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        if payload is not None:
            print(
                f"[kernel] payload preflight failed; falling back to git clone. missing={missing}",
                flush=True,
            )
            shutil.rmtree(repo_dir, ignore_errors=True)
            run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--branch",
                    GIT_BRANCH,
                    GIT_REPO_URL,
                    str(repo_dir),
                ]
            )
            missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Repo preflight failed; missing: {missing}")
    return repo_dir


def metric(row: dict[str, Any], key: str) -> float:
    for name in (f"{key}_mean", f"{key}_mean_mean", key):
        value = row.get(name)
        if value not in (None, ""):
            return float(value)
    return 0.0


def summarize_run(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return {**CANDIDATE, "status": "missing_summary", "run_dir": str(run_dir)}

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = [row for row in summary.get("policy_summary", []) if isinstance(row, dict)]
    by_policy = {str(row.get("policy")): row for row in rows if row.get("policy")}
    ppo = by_policy.get("ppo")
    nonppo = [row for row in rows if str(row.get("policy")) != "ppo"]
    if ppo is None or not nonppo:
        return {**CANDIDATE, "status": "missing_policy_rows", "run_dir": str(run_dir)}

    best_nonppo = max(nonppo, key=lambda row: metric(row, "order_ret_excel"))
    best_static_name = summary.get("decision", {}).get("best_static_policy")
    best_static = by_policy.get(str(best_static_name)) if best_static_name else best_nonppo
    if best_static is None:
        best_static = best_nonppo

    def delta(key: str, ref: dict[str, Any] = best_static) -> float:
        return metric(ppo, key) - metric(ref, key)

    result = {
        **CANDIDATE,
        "status": "ok",
        "run_dir": str(run_dir),
        "summary_json": str(summary_path),
        "episode_metrics_csv": str(run_dir / "episode_metrics.csv"),
        "comparison_table_csv": str(run_dir / "comparison_table.csv"),
        "order_ledger_csv": str(run_dir / "order_ledger.csv"),
        "best_static_policy": best_static.get("policy"),
        "best_nonppo_policy": best_nonppo.get("policy"),
        "ppo_order_ret_excel": metric(ppo, "order_ret_excel"),
        "static_order_ret_excel": metric(best_static, "order_ret_excel"),
        "nonppo_order_ret_excel": metric(best_nonppo, "order_ret_excel"),
        "ppo_order_ret_excel_cvar05": metric(ppo, "order_ret_excel_cvar05"),
        "static_order_ret_excel_cvar05": metric(best_static, "order_ret_excel_cvar05"),
        "ppo_cost_index": metric(ppo, "assembly_cost_index"),
        "static_cost_index": metric(best_static, "assembly_cost_index"),
        "ppo_ctj_p99": metric(ppo, "order_ctj_p99"),
        "static_ctj_p99": metric(best_static, "order_ctj_p99"),
        "ppo_rpj_p99": metric(ppo, "order_rpj_p99"),
        "static_rpj_p99": metric(best_static, "order_rpj_p99"),
        "order_ret_excel_delta": delta("order_ret_excel"),
        "order_ret_excel_delta_vs_best_nonppo": delta("order_ret_excel", best_nonppo),
        "order_ret_excel_cvar05_delta": delta("order_ret_excel_cvar05"),
        "flow_fill_delta": delta("order_flow_fill_rate"),
        "cost_index_delta": delta("assembly_cost_index"),
        "ctj_p99_delta": delta("order_ctj_p99"),
        "rpj_p99_delta": delta("order_rpj_p99"),
        "service_loss_auc_per_order_delta": delta("order_service_loss_auc_per_order"),
    }
    result["raw_ret_win"] = result["order_ret_excel_delta"] > 0.0
    result["tail_win"] = result["order_ret_excel_cvar05_delta"] > 0.0
    result["cost_nonworse"] = result["cost_index_delta"] <= 0.0
    return result


def run_candidate(repo_dir: Path, out_dir: Path, profile: str) -> None:
    if profile == "smoke":
        seeds = ["1"]
        train_timesteps = "256"
        eval_episodes = "1"
        max_steps = "2"
        n_envs = "1"
    elif profile == "confirmatory":
        seeds = ["1", "2", "3", "4", "5"]
        train_timesteps = "60000"
        eval_episodes = "12"
        max_steps = "104"
        n_envs = "4"
    else:
        raise ValueError("SCRESIA_PROFILE must be smoke or confirmatory")

    cmd = [
        sys.executable,
        "scripts/run_track_b_smoke.py",
        "--output-dir",
        str(out_dir),
        "--seeds",
        *seeds,
        "--train-timesteps",
        train_timesteps,
        "--eval-episodes",
        eval_episodes,
        "--export-order-ledger",
        "--reward-mode",
        str(CANDIDATE["reward_mode"]),
        "--risk-level",
        "adaptive_benchmark_v2",
        "--observation-version",
        str(CANDIDATE["observation_version"]),
        "--max-steps",
        max_steps,
        "--n-envs",
        n_envs,
        "--n-steps",
        "256",
        "--batch-size",
        "64",
        "--learning-rate",
        "0.0001",
    ]
    run(cmd, cwd=repo_dir)


def main() -> int:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    profile = os.environ.get("SCRESIA_PROFILE", "confirmatory").strip()
    started = datetime.now(timezone.utc)
    output_dir = OUTPUT_DIR if is_kaggle() else LOCAL_OUTPUT_DIR.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"SCRESIA Track B ReT_tail_v2/v7 confirm profile={profile} started={started.isoformat()}",
        flush=True,
    )
    try:
        repo_dir = prepare_repo()
        if is_kaggle():
            run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=repo_dir)

        run_dir = output_dir / str(CANDIDATE["label"])
        run_candidate(repo_dir, run_dir, profile)
        decision = summarize_run(run_dir)
        final = {
            "profile": profile,
            "confirmatory_profile": "track_b_ret_tail_v2_v7_5seed_60k_h104",
            "started_at": started.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "primary_metric": "Garrido Excel/order ReT",
            "candidate": decision,
        }
        (output_dir / "decision.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
        print("\n=== TRACK B ReT_tail_v2/v7 DECISION ===", flush=True)
        print(json.dumps(final, indent=2), flush=True)

        if is_kaggle() and repo_dir.exists():
            shutil.rmtree(repo_dir, ignore_errors=True)
        return 0
    except Exception as exc:
        error = {
            "profile": profile,
            "confirmatory_profile": "track_b_ret_tail_v2_v7_5seed_60k_h104",
            "started_at": started.isoformat(),
            "failed_at": datetime.now(timezone.utc).isoformat(),
            "error": repr(exc),
        }
        (output_dir / "error.json").write_text(json.dumps(error, indent=2), encoding="utf-8")
        print(json.dumps(error, indent=2), flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
