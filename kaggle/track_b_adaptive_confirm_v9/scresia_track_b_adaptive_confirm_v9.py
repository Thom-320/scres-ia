"""Kaggle kernel: Track B adaptive v9 confirmatory run.

Runs the two promoted adaptive_benchmark_v2 Track B candidates from the
2026-07-01 sweep:

1. ReT_excel_plus_cvar alpha=0.05 with v9 observation.
2. control_v1 with v9 observation.

The primary reporting bar is Garrido-style Excel/order ReT.  The run also
exports a per-order ledger for CTj/RPj/DPj/APj and tail-risk audit.
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
REPO_DIR = Path("/kaggle/working/scres-ia")
OUTPUT_DIR = Path("/kaggle/working/track_b_adaptive_confirm_v9")
LOCAL_REPO_DIR = Path("outputs/kaggle/track_b_adaptive_confirm_v9/_payload_repo")
LOCAL_OUTPUT_DIR = Path("outputs/kaggle/track_b_adaptive_confirm_v9")
EMBEDDED_PAYLOAD_B64 = ""
GIT_REPO_URL = "https://github.com/Thom-320/scres-ia.git"
GIT_BRANCH = "codex/garrido-replication-experiments"


CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "label": "ret_excel_plus_cvar_a0p05_v9",
        "reward_mode": "ReT_excel_plus_cvar",
        "observation_version": "v9",
        "ret_excel_cvar_alpha": 0.05,
    },
    {
        "label": "control_v1_v9",
        "reward_mode": "control_v1",
        "observation_version": "v9",
        "ret_excel_cvar_alpha": None,
    },
)


def is_kaggle() -> bool:
    return Path("/kaggle/working").exists()


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def find_payload() -> Path:
    candidates = [SCRIPT_PAYLOAD, DATASET_PAYLOAD, Path("scres_ia_payload.tar.gz")]
    if is_kaggle():
        candidates.extend(sorted(Path("/kaggle/input").glob("**/scres_ia_payload.tar.gz")))
    for candidate in candidates:
        if candidate.exists():
            print(f"[kernel] using payload {candidate}", flush=True)
            return candidate
    if EMBEDDED_PAYLOAD_B64.strip():
        target = Path("/kaggle/working/scres_ia_payload.tar.gz") if is_kaggle() else Path("scres_ia_payload.tar.gz")
        print(f"[kernel] writing embedded payload {target}", flush=True)
        target.write_bytes(base64.b64decode(EMBEDDED_PAYLOAD_B64))
        return target
    raise FileNotFoundError(
        "Missing scres_ia_payload.tar.gz beside kernel or in Kaggle input."
    )


def extract_payload() -> Path:
    repo_dir = REPO_DIR if is_kaggle() else LOCAL_REPO_DIR.resolve()
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    repo_dir.mkdir(parents=True, exist_ok=True)
    try:
        payload = find_payload()
    except FileNotFoundError as exc:
        if not is_kaggle():
            raise
        print(f"[kernel] payload unavailable ({exc}); cloning {GIT_BRANCH}", flush=True)
        shutil.rmtree(repo_dir)
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
        raise FileNotFoundError(f"Payload preflight failed; missing: {missing}")
    return repo_dir


def metric(row: dict[str, Any], key: str) -> float:
    for name in (f"{key}_mean", f"{key}_mean_mean", key):
        value = row.get(name)
        if value not in (None, ""):
            return float(value)
    return 0.0


def summarize_run(run_dir: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return {
            **candidate,
            "status": "missing_summary",
            "run_dir": str(run_dir),
        }
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    by_policy = {
        str(row.get("policy")): row
        for row in summary.get("policy_summary", [])
        if isinstance(row, dict) and row.get("policy")
    }
    ppo = by_policy.get("ppo")
    best_static_name = summary.get("decision", {}).get("best_static_policy")
    best_static = by_policy.get(str(best_static_name)) if best_static_name else None
    if ppo is None or best_static is None:
        return {
            **candidate,
            "status": "missing_policy_rows",
            "run_dir": str(run_dir),
            "best_static_policy": best_static_name,
        }

    deltas = {
        "order_ret_excel_delta": metric(ppo, "order_ret_excel")
        - metric(best_static, "order_ret_excel"),
        "order_ret_excel_cvar05_delta": metric(ppo, "order_ret_excel_cvar05")
        - metric(best_static, "order_ret_excel_cvar05"),
        "flow_fill_delta": metric(ppo, "order_flow_fill_rate")
        - metric(best_static, "order_flow_fill_rate"),
        "cost_index_delta": metric(ppo, "assembly_cost_index")
        - metric(best_static, "assembly_cost_index"),
        "ctj_p99_delta": metric(ppo, "order_ctj_p99")
        - metric(best_static, "order_ctj_p99"),
        "rpj_p99_delta": metric(ppo, "order_rpj_p99")
        - metric(best_static, "order_rpj_p99"),
        "service_loss_auc_per_order_delta": metric(
            ppo, "order_service_loss_auc_per_order"
        )
        - metric(best_static, "order_service_loss_auc_per_order"),
    }
    return {
        **candidate,
        "status": "ok",
        "run_dir": str(run_dir),
        "summary_json": str(summary_path),
        "order_ledger_csv": str(run_dir / "order_ledger.csv"),
        "best_static_policy": best_static_name,
        "ppo_order_ret_excel": metric(ppo, "order_ret_excel"),
        "static_order_ret_excel": metric(best_static, "order_ret_excel"),
        "ppo_order_ret_excel_cvar05": metric(ppo, "order_ret_excel_cvar05"),
        "static_order_ret_excel_cvar05": metric(best_static, "order_ret_excel_cvar05"),
        "ppo_cost_index": metric(ppo, "assembly_cost_index"),
        "static_cost_index": metric(best_static, "assembly_cost_index"),
        "ppo_ctj_p99": metric(ppo, "order_ctj_p99"),
        "static_ctj_p99": metric(best_static, "order_ctj_p99"),
        "ppo_rpj_p99": metric(ppo, "order_rpj_p99"),
        "static_rpj_p99": metric(best_static, "order_rpj_p99"),
        **deltas,
        "raw_ret_win": deltas["order_ret_excel_delta"] > 0.0,
        "tail_win": deltas["order_ret_excel_cvar05_delta"] > 0.0,
        "cost_nonworse": deltas["cost_index_delta"] <= 0.0,
    }


def run_candidate(candidate: dict[str, Any], out_dir: Path, profile: str) -> None:
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
        str(candidate["reward_mode"]),
        "--risk-level",
        "adaptive_benchmark_v2",
        "--observation-version",
        str(candidate["observation_version"]),
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
    alpha = candidate.get("ret_excel_cvar_alpha")
    if alpha is not None:
        cmd.extend(["--ret-excel-cvar-alpha", str(alpha)])
        run(cmd, cwd=REPO_DIR if is_kaggle() else LOCAL_REPO_DIR.resolve())


def main() -> int:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    profile = os.environ.get("SCRESIA_PROFILE", "confirmatory").strip()
    started = datetime.now(timezone.utc)
    print(
        f"SCRESIA Track B adaptive confirm v9 profile={profile} started={started.isoformat()}",
        flush=True,
    )
    output_dir = OUTPUT_DIR if is_kaggle() else LOCAL_OUTPUT_DIR.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        repo_dir = extract_payload()
        if is_kaggle():
            run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=repo_dir)
        decisions: list[dict[str, Any]] = []
        candidates = CANDIDATES[:1] if profile == "smoke" else CANDIDATES
        for idx, candidate in enumerate(candidates, start=1):
            run_dir = output_dir / str(candidate["label"])
            print(f"[kernel] candidate {idx}/{len(candidates)} -> {run_dir}", flush=True)
            run_candidate(candidate, run_dir, profile)
            decision = summarize_run(run_dir, candidate)
            decisions.append(decision)
            (output_dir / "partial_decision.json").write_text(
                json.dumps({"candidates": decisions}, indent=2), encoding="utf-8"
            )

        final = {
            "profile": profile,
            "confirmatory_profile": "track_b_adaptive_confirm_v9_2configs_5seed_60k_h104",
            "started_at": started.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "primary_metric": "Garrido Excel/order ReT",
            "candidates": decisions,
        }
        (output_dir / "decision.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
        print("\n=== TRACK B ADAPTIVE CONFIRM V9 DECISION ===", flush=True)
        print(json.dumps(final, indent=2), flush=True)
        return 0
    except Exception as exc:
        error = {
            "profile": profile,
            "confirmatory_profile": "track_b_adaptive_confirm_v9_2configs_5seed_60k_h104",
            "started_at": started.isoformat(),
            "failed_at": datetime.now(timezone.utc).isoformat(),
            "error": repr(exc),
        }
        output_dir = OUTPUT_DIR if is_kaggle() else LOCAL_OUTPUT_DIR.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "error.json").write_text(json.dumps(error, indent=2), encoding="utf-8")
        print(json.dumps(error, indent=2), flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())


EMBEDDED_PAYLOAD_B64 = ""
