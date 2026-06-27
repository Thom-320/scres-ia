"""Kaggle kernel: DQN retained-transfer ladder for the frozen Env-B frontier.

This is the paper-facing Track-A memory lane, not the PPO dynamic-vs-static
screen.  It runs the cold-transfer estimand in ``scripts/retention_transfer.py``:
retained weights versus reset single-block adaptation, evaluated before each
new block's training.
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


PAYLOAD_NAME = "scres_ia_payload.tar.gz"
KAGGLE_REPO_DIR = Path("/kaggle/temp/scres-ia")
KAGGLE_OUTPUT_ROOT = Path("/kaggle/working/scresia_garrido_dqn_frontier_outputs")
LOCAL_OUTPUT_ROOT = Path("outputs/kaggle/garrido_dqn_frontier_ladder")


FULL = {
    "seeds": "8501,8502,8503,8504,8505,8506,8507,8508,8509,8510",
    "n_blocks": "24",
    "max_steps": "12",
    "train_per_block": "150",
}

DEBUG = {
    "seeds": "8501",
    "n_blocks": "4",
    "max_steps": "4",
    "train_per_block": "50",
}


def is_kaggle() -> bool:
    return Path("/kaggle/working").exists()


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def profile() -> dict[str, str]:
    value = (subprocess.os.environ.get("SCRESIA_PROFILE") or "confirmatory").strip()
    if value == "debug":
        return DEBUG
    if value != "confirmatory":
        raise ValueError("SCRESIA_PROFILE must be 'debug' or 'confirmatory'.")
    return FULL


def find_payload() -> Path | None:
    candidates = [Path(PAYLOAD_NAME)]
    if is_kaggle():
        candidates.extend(sorted(Path("/kaggle/input").glob(f"**/{PAYLOAD_NAME}")))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def find_mounted_repo() -> Path | None:
    if not is_kaggle():
        return None
    for candidate in sorted(Path("/kaggle/input").glob("**")):
        if not candidate.is_dir():
            continue
        if (candidate / "scripts" / "retention_transfer.py").exists():
            return candidate
    return None


def repo_root_from_context() -> Path:
    cwd = Path.cwd()
    if (cwd / "scripts" / "retention_transfer.py").exists():
        return cwd
    parent = Path(__file__).resolve().parents[2]
    if (parent / "scripts" / "retention_transfer.py").exists():
        return parent
    raise FileNotFoundError("Could not locate scres-ia repo root.")


def extract_payload_or_use_local_repo() -> Path:
    payload = find_payload()
    if payload is not None:
        print(f"[kernel] Extracting payload from {payload}", flush=True)
        if KAGGLE_REPO_DIR.exists():
            shutil.rmtree(KAGGLE_REPO_DIR)
        KAGGLE_REPO_DIR.mkdir(parents=True, exist_ok=True)
        with tarfile.open(payload, "r:gz") as archive:
            archive.extractall(KAGGLE_REPO_DIR)
        return KAGGLE_REPO_DIR
    mounted = find_mounted_repo()
    if mounted is not None:
        print(f"[kernel] Copying mounted repo from {mounted}", flush=True)
        if KAGGLE_REPO_DIR.exists():
            shutil.rmtree(KAGGLE_REPO_DIR)
        shutil.copytree(mounted, KAGGLE_REPO_DIR)
        return KAGGLE_REPO_DIR
    if is_kaggle():
        raise FileNotFoundError("Could not locate scres-ia payload or mounted repo.")
    return repo_root_from_context()


def output_root() -> Path:
    return KAGGLE_OUTPUT_ROOT if is_kaggle() else LOCAL_OUTPUT_ROOT


def torch_probe() -> dict[str, object]:
    try:
        import torch

        return {
            "available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
            "device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        }
    except Exception as exc:  # pragma: no cover - only diagnostic
        return {"error": repr(exc)}


def install_dependencies(repo: Path) -> None:
    """Install the payload dependencies inside Kaggle's fresh kernel image."""
    requirements = repo / "requirements.txt"
    if requirements.exists():
        run([sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements)], cwd=repo)
        return
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "stable-baselines3",
            "sb3-contrib",
            "gymnasium",
            "simpy",
            "numpy",
            "pandas",
        ],
        cwd=repo,
    )


def run_frontier_ladder(repo: Path, out: Path, prof: dict[str, str]) -> Path:
    label = "kaggle_envb_frontier_v2_control_v1_dqn"
    cmd = [
        sys.executable,
        "scripts/retention_transfer.py",
        "--label",
        label,
        "--output-root",
        str(out / "runs"),
        "--reward-mode",
        "control_v1",
        "--outcome",
        "excel_ret",
        "--risk-frequency-multiplier",
        "1.0",
        "--risk-impact-multiplier",
        "1.5",
        "--seeds",
        prof["seeds"],
        "--n-blocks",
        prof["n_blocks"],
        "--max-steps",
        prof["max_steps"],
        "--train-per-block",
        prof["train_per_block"],
        "--rho-disruption",
        "0.85",
        "--regime-seed",
        "909",
        "--mask-preset",
        "direct_disruption_blind",
        "--learning-starts",
        "50",
        "--buffer-size",
        "10000",
    ]
    run(cmd, cwd=repo)
    return out / "runs" / label / "transfer.json"


def write_report(out: Path, transfer_path: Path, prof: dict[str, str]) -> None:
    transfer = json.loads(transfer_path.read_text())
    mem = transfer["memory_retained_minus_reset"]["overall"]
    total = transfer["total_retained_minus_frozen"]["overall"]
    decision = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": "envb_frontier_v2",
        "reward_mode": transfer["reward_mode"],
        "outcome": transfer["outcome"],
        "settings": prof,
        "memory_retained_minus_reset": mem,
        "total_retained_minus_frozen": total,
        "memory_ci_strictly_positive": mem["ci95_lo"] > 0,
        "total_ci_strictly_positive": total["ci95_lo"] > 0,
        "transfer_json": str(transfer_path),
    }
    (out / "dqn_frontier_decision.json").write_text(json.dumps(decision, indent=2))
    report = [
        "# DQN Frontier Ladder",
        "",
        f"Generated: {decision['generated_at_utc']}",
        "",
        "Profile: `envb_frontier_v2` (`phi=1.0`, `psi=1.5`, `stochastic_pt=false`).",
        "Reward: `control_v1`; outcome: Garrido Excel ReT.",
        "",
        f"Memory retained-reset: {mem['mean']:+.6f} CI95 "
        f"[{mem['ci95_lo']:+.6f}, {mem['ci95_hi']:+.6f}]",
        f"Total retained-frozen: {total['mean']:+.6f} CI95 "
        f"[{total['ci95_lo']:+.6f}, {total['ci95_hi']:+.6f}]",
        "",
        f"Transfer JSON: `{transfer_path}`",
    ]
    (out / "dqn_frontier_report.md").write_text("\n".join(report) + "\n")


def main() -> None:
    prof = profile()
    repo = extract_payload_or_use_local_repo()
    out = output_root()
    out.mkdir(parents=True, exist_ok=True)
    if is_kaggle():
        # Kaggle's P100 image can expose a CUDA device that the bundled torch
        # wheel cannot execute on. The Track-A DQN job is CPU-light enough, and
        # this keeps the confirmatory run deterministic instead of failing late.
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    print("SCRESIA DQN frontier ladder", datetime.now(timezone.utc).isoformat(), flush=True)
    print("profile", prof, flush=True)
    print("repo", repo, flush=True)
    print("output", out, flush=True)
    print("torch_cuda", torch_probe(), flush=True)
    if is_kaggle():
        install_dependencies(repo)
    transfer_path = run_frontier_ladder(repo, out, prof)
    write_report(out, transfer_path, prof)
    print("outputs", out, flush=True)


if __name__ == "__main__":
    main()
