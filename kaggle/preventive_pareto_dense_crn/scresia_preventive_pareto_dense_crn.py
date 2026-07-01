"""Kaggle kernel: dense-frontier + CRN audit for the preventive Pareto lane.

Lead lane:
  action = continuous_it_s
  observation = v6 + risk_obs/hazard wrapper
  reward = ReT_excel_delta
  env = war stress phi4 / psi1.5
  horizon = 104 weekly decisions
  evaluation = resource-aware Pareto frontier on Excel ReT and CVaR
  audit = dense 21x3 static frontier + common-random-number eval seed block

This intentionally uses scripts/run_preventive_pareto.py, not the older
run_continuous_its.py free-static comparison.
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


PAYLOAD = Path("/kaggle/input/scres-ia-payload/scres_ia_payload.tar.gz")
LOCAL_PAYLOAD = Path("scres_ia_payload.tar.gz")
SCRIPT_PAYLOAD = Path(__file__).resolve().with_name("scres_ia_payload.tar.gz")
KAGGLE_REPO_DIR = Path("/kaggle/temp/scres-ia")
KAGGLE_OUTPUT_ROOT = Path("/kaggle/working/scresia_preventive_pareto_dense_crn_outputs")
LOCAL_OUTPUT_ROOT = Path("outputs/kaggle/preventive_pareto_dense_crn")
LOCAL_REPO_DIR = LOCAL_OUTPUT_ROOT / "_payload_repo"

DEBUG_PROFILE = {
    "seeds": "9901",
    "timesteps": "256",
    "n_envs": "1",
    "eval_episodes": "1",
    "max_steps": "4",
}

CONFIRMATORY_PROFILE = {
    "seeds": "1,2,3,4,5,8501,8502,8503,8504,8505",
    "timesteps": "60000",
    "n_envs": "4",
    "eval_episodes": "8",
    "max_steps": "104",
}


def is_kaggle() -> bool:
    return Path("/kaggle/working").exists()


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def ensure_dependencies(repo: Path) -> None:
    requirements = repo / "requirements.txt"
    if not is_kaggle() or not requirements.exists():
        return
    run([sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements)])


def find_payload() -> Path | None:
    candidates = [SCRIPT_PAYLOAD, PAYLOAD, LOCAL_PAYLOAD]
    if is_kaggle():
        candidates.extend(sorted(Path("/kaggle/input").glob("**/scres_ia_payload.tar.gz")))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def repo_root_from_context() -> Path:
    cwd = Path.cwd()
    if (cwd / "scripts" / "run_preventive_pareto.py").exists():
        return cwd
    parent = Path(__file__).resolve().parents[2]
    if (parent / "scripts" / "run_preventive_pareto.py").exists():
        return parent
    raise FileNotFoundError("Could not locate scres-ia repo root.")


def find_extracted_repo() -> Path | None:
    candidates: list[Path] = []
    if is_kaggle():
        candidates.extend(
            path
            for path in Path("/kaggle/input").glob("**/scres_ia_payload")
            if (path / "scripts" / "run_preventive_pareto.py").exists()
        )
        candidates.extend(
            path.parents[1]
            for path in Path("/kaggle/input").glob(
                "**/scres_ia_payload/scripts/run_preventive_pareto.py"
            )
        )
        candidates.extend(
            path.parents[1]
            for path in Path("/kaggle/input").glob("**/scripts/run_preventive_pareto.py")
            if "scres-ia-payload" in str(path)
        )
    for candidate in candidates:
        if (candidate / "supply_chain").exists():
            return candidate
    return None


def describe_kaggle_input() -> None:
    if not is_kaggle():
        return
    root = Path("/kaggle/input")
    print("[kernel] /kaggle/input listing:", flush=True)
    if not root.exists():
        print("[kernel]   MISSING /kaggle/input", flush=True)
        return
    for path in sorted(root.glob("**/*"))[:250]:
        kind = "dir " if path.is_dir() else "file"
        print(f"[kernel]   {kind} {path}", flush=True)


def extract_payload_or_use_local_repo() -> Path:
    payload = find_payload()
    if payload is not None:
        print(f"[kernel] extracting payload {payload}", flush=True)
        repo_dir = KAGGLE_REPO_DIR if is_kaggle() else LOCAL_REPO_DIR
        if repo_dir.exists():
            shutil.rmtree(repo_dir)
        repo_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(payload, "r:gz") as archive:
            archive.extractall(repo_dir)
        return repo_dir
    extracted = find_extracted_repo()
    if extracted is not None:
        print(f"[kernel] copying extracted repo {extracted}", flush=True)
        if KAGGLE_REPO_DIR.exists():
            shutil.rmtree(KAGGLE_REPO_DIR)
        shutil.copytree(extracted, KAGGLE_REPO_DIR)
        return KAGGLE_REPO_DIR
    describe_kaggle_input()
    return repo_root_from_context()


def profile() -> dict[str, str]:
    value = (os.environ.get("SCRESIA_PROFILE") or "confirmatory").strip()
    if value == "debug":
        return DEBUG_PROFILE
    if value != "confirmatory":
        raise ValueError("SCRESIA_PROFILE must be debug or confirmatory")
    return CONFIRMATORY_PROFILE


def output_root() -> Path:
    return KAGGLE_OUTPUT_ROOT if is_kaggle() else LOCAL_OUTPUT_ROOT


def run_case(repo: Path, out: Path, *, label: str, prof: dict[str, str]) -> Path:
    case_out = out / label
    cmd = [
        sys.executable,
        "scripts/run_preventive_pareto.py",
        "--regime",
        "current",
        "--phi",
        "4.0",
        "--psi",
        "1.5",
        "--reward-mode",
        "ReT_excel_delta",
        "--observation-version",
        "v6",
        "--seeds",
        prof["seeds"],
        "--n-envs",
        prof["n_envs"],
        "--timesteps",
        prof["timesteps"],
        "--eval-episodes",
        prof["eval_episodes"],
        "--max-steps",
        prof["max_steps"],
        "--n-fracs",
        "21",
        "--crn-eval",
        "--eval-seed0",
        "9000",
        "--step-size-hours",
        "168",
        "--holding-cost",
        "0.0",
        "--shift-cost",
        "0.001",
        "--output",
        str(case_out),
    ]
    run(cmd, cwd=repo)
    return case_out / "summary.json"


def decision_from_summary(path: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text())
    return {
        "summary_path": str(path),
        "args": summary["args"],
        "dynamic": summary["dynamic"],
        "excel_pareto": summary["excel_pareto"],
        "cvar_pareto": summary["cvar_pareto"],
        "win_excel_pareto": bool(summary["excel_pareto"].get("pareto_win", False)),
        "win_cvar_pareto": bool(summary["cvar_pareto"].get("pareto_win", False)),
        "n_learned_seeds": len(summary.get("learned_per_seed", [])),
        "learned_per_seed": summary.get("learned_per_seed", []),
        "action_correlations": summary.get("action_correlations", []),
    }


def main() -> int:
    # SB3 MLP PPO is CPU-oriented; force CPU because Kaggle CUDA images can mismatch torch wheels.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    repo = extract_payload_or_use_local_repo()
    ensure_dependencies(repo)
    out = output_root()
    out.mkdir(parents=True, exist_ok=True)
    prof = profile()

    started = datetime.now(timezone.utc).isoformat()
    case = "preventive_pareto_excel_delta_mixed10_60k_dense21_crn9000"
    summary_path = run_case(repo, out, label=case, prof=prof)
    decision = {
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "profile": prof,
        "primary_case": case,
        "cases": {case: decision_from_summary(summary_path)},
    }
    decision["primary_win"] = (
        decision["cases"][case]["win_excel_pareto"]
        and decision["cases"][case]["win_cvar_pareto"]
    )
    (out / "decision.json").write_text(json.dumps(decision, indent=2))
    print(json.dumps(decision, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
