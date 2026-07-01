"""Kaggle kernel: confirm continuous I_t,S + v6 + Excel reward signal.

Primary claim lane:
  action = continuous_it_s
  observation = v6 forecast visible
  reward = ReT_excel_delta
  env = war phi4 / psi1.5
  init_frac = best_excel selected from the static continuous frontier
  horizon = 104 weekly steps
  eval primary = Excel ReT
  secondary = CVaR, flow fill, lost rate

The kernel runs the free-initial-static comparison used by the local winning
gate: static policies may choose initial prepositioning and weekly fraction;
PPO receives the best-Excel initial fraction selected from that same static
frontier, then learns weekly continuous control.
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
KAGGLE_REPO_DIR = Path("/kaggle/temp/scres-ia")
KAGGLE_OUTPUT_ROOT = Path("/kaggle/working/scresia_continuous_its_confirm_outputs")
LOCAL_OUTPUT_ROOT = Path("outputs/kaggle/continuous_its_confirm")


DEBUG_PROFILE = {
    "seeds": "5101",
    "timesteps": "256",
    "n_envs": "1",
    "eval_episodes": "1",
    "max_steps": "4",
}

CONFIRMATORY_PROFILE = {
    "seeds": "8101,8102,8103,8104,8105",
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
    """Install runtime dependencies in Kaggle's clean kernel image."""
    requirements = repo / "requirements.txt"
    if not is_kaggle() or not requirements.exists():
        return
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "-r",
            str(requirements),
        ]
    )


def find_payload() -> Path | None:
    candidates = [PAYLOAD, LOCAL_PAYLOAD]
    if is_kaggle():
        candidates.extend(sorted(Path("/kaggle/input").glob("**/scres_ia_payload.tar.gz")))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def repo_root_from_context() -> Path:
    cwd = Path.cwd()
    if (cwd / "scripts" / "run_continuous_its.py").exists():
        return cwd
    parent = Path(__file__).resolve().parents[2]
    if (parent / "scripts" / "run_continuous_its.py").exists():
        return parent
    raise FileNotFoundError("Could not locate scres-ia repo root.")


def find_extracted_repo() -> Path | None:
    candidates: list[Path] = []
    if is_kaggle():
        candidates.extend(
            path
            for path in Path("/kaggle/input").glob("**/scres_ia_payload")
            if (path / "scripts" / "run_continuous_its.py").exists()
        )
        candidates.extend(
            path.parents[1]
            for path in Path("/kaggle/input").glob(
                "**/scres_ia_payload/scripts/run_continuous_its.py"
            )
        )
        candidates.extend(
            path.parents[1]
            for path in Path("/kaggle/input").glob("**/scripts/run_continuous_its.py")
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
    for path in sorted(root.glob("**/*"))[:200]:
        kind = "dir " if path.is_dir() else "file"
        print(f"[kernel]   {kind} {path}", flush=True)


def extract_payload_or_use_local_repo() -> Path:
    payload = find_payload()
    if payload is not None:
        print(f"[kernel] extracting payload {payload}", flush=True)
        if KAGGLE_REPO_DIR.exists():
            shutil.rmtree(KAGGLE_REPO_DIR)
        KAGGLE_REPO_DIR.mkdir(parents=True, exist_ok=True)
        with tarfile.open(payload, "r:gz") as archive:
            archive.extractall(KAGGLE_REPO_DIR)
        return KAGGLE_REPO_DIR
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


def run_case(
    repo: Path,
    out: Path,
    *,
    label: str,
    prof: dict[str, str],
) -> Path:
    case_out = out / label
    cmd = [
        sys.executable,
        "scripts/run_continuous_its.py",
        "--reward-mode",
        "ReT_excel_delta",
        "--observation-version",
        "v6",
        "--phi",
        "4",
        "--psi",
        "1.5",
        "--regime",
        "current",
        "--init-frac",
        "best_excel",
        "--seeds",
        prof["seeds"],
        "--timesteps",
        prof["timesteps"],
        "--n-envs",
        prof["n_envs"],
        "--eval-episodes",
        prof["eval_episodes"],
        "--max-steps",
        prof["max_steps"],
        "--static-init-fracs",
        "0,0.25,0.5,0.75,1",
        "--static-fracs",
        "0,0.1,0.2,0.25,0.3,0.4,0.5,0.6,0.75,1",
        "--output",
        str(case_out),
    ]
    run(cmd, cwd=repo)
    return case_out / "summary.json"


def decision_from_summary(path: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text())
    best_excel = summary["best_const_excel"]
    best_cvar = summary["best_const_cvar"]
    constants = summary["constants"]
    learned_excel = float(summary["learned_excel_mean"])
    learned_cvar = float(summary["learned_cvar_mean"])
    static_excel = float(constants[best_excel]["ret_excel"])
    static_cvar = float(constants[best_cvar]["cvar95"])
    learned = list(summary.get("learned", []))
    adaptive = [
        row
        for row in learned
        if float(row.get("frac_std", 0.0) or 0.0) > 0.05
    ]
    return {
        "summary_path": str(path),
        "args": summary["args"],
        "best_const_excel": best_excel,
        "best_const_excel_value": static_excel,
        "best_const_cvar": best_cvar,
        "best_const_cvar_value": static_cvar,
        "learned_excel_mean": learned_excel,
        "learned_cvar_mean": learned_cvar,
        "delta_excel": learned_excel - static_excel,
        "delta_cvar": learned_cvar - static_cvar,
        "win_excel": learned_excel > static_excel,
        "win_cvar": learned_cvar < static_cvar,
        "adaptive_seed_count": len(adaptive),
        "n_learned_seeds": len(learned),
    }


def main() -> int:
    # SB3 MLP PPO is CPU-oriented, and Kaggle's P100 image can be incompatible
    # with the latest PyTorch CUDA wheels. Force CPU for a stable confirmatory run.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    repo = extract_payload_or_use_local_repo()
    ensure_dependencies(repo)
    out = output_root()
    out.mkdir(parents=True, exist_ok=True)
    prof = profile()
    started = datetime.now(timezone.utc).isoformat()

    fixed = run_case(
        repo,
        out,
        label="h104_free_init_static",
        prof=prof,
    )

    decisions = {
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "profile": prof,
        "primary_case": "h104_free_init_static",
        "cases": {
            "h104_free_init_static": decision_from_summary(fixed),
        },
    }
    decisions["primary_win"] = bool(
        decisions["cases"]["h104_free_init_static"]["win_excel"]
        and decisions["cases"]["h104_free_init_static"]["win_cvar"]
    )
    (out / "decision.json").write_text(json.dumps(decisions, indent=2))
    print(json.dumps(decisions, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
