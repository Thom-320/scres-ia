"""Kaggle kernel: continuous_its retained-reset memory probe.

This tests the strongest remaining upside claim after the confirmed preventive
Pareto win:

    Delta memory = ExcelReT(retained, cold) - ExcelReT(reset, cold)

on the winning lane:
    continuous_its + v6 + risk_obs/hazard + ReT_excel_delta + phi4/psi1.5.

The default profile is a directional probe, not the final confirmatory. Use
SCRESIA_PROFILE=confirmatory only after the probe has a positive signal.
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
KAGGLE_OUTPUT_ROOT = Path("/kaggle/working/scresia_continuous_dmemory_outputs")
LOCAL_OUTPUT_ROOT = Path("outputs/kaggle/continuous_dmemory_probe")

PROFILES: dict[str, dict[str, str]] = {
    "debug": {
        "seeds": "9901",
        "n_blocks": "1",
        "max_steps": "2",
        "train_per_block": "8",
        "n_steps": "8",
        "n_epochs": "1",
    },
    "probe": {
        "seeds": "8201,8202",
        "n_blocks": "4",
        "max_steps": "12",
        "train_per_block": "1000",
        "n_steps": "128",
        "n_epochs": "6",
    },
    "confirmatory": {
        "seeds": "8201,8202,8203,8204,8205,8501,8502,8503,8504,8505",
        "n_blocks": "12",
        "max_steps": "12",
        "train_per_block": "2000",
        "n_steps": "128",
        "n_epochs": "6",
    },
}


def is_kaggle() -> bool:
    return Path("/kaggle/working").exists()


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def find_payload() -> Path | None:
    candidates = [PAYLOAD, LOCAL_PAYLOAD, SCRIPT_PAYLOAD]
    if is_kaggle():
        candidates.extend(sorted(Path("/kaggle/input").glob("**/scres_ia_payload.tar.gz")))
    for candidate in candidates:
        if candidate.exists():
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


def repo_root_from_context() -> Path:
    cwd = Path.cwd()
    if (cwd / "scripts" / "retention_transfer.py").exists():
        return cwd
    parent = Path(__file__).resolve().parents[2]
    if (parent / "scripts" / "retention_transfer.py").exists():
        return parent
    raise FileNotFoundError("Could not locate scres-ia repo root.")


def find_extracted_repo() -> Path | None:
    candidates: list[Path] = []
    if is_kaggle():
        candidates.extend(
            path
            for path in Path("/kaggle/input").glob("**/scres_ia_payload")
            if (path / "scripts" / "retention_transfer.py").exists()
        )
        candidates.extend(
            path.parents[1]
            for path in Path("/kaggle/input").glob(
                "**/scres_ia_payload/scripts/retention_transfer.py"
            )
        )
        candidates.extend(
            path.parents[1]
            for path in Path("/kaggle/input").glob("**/scripts/retention_transfer.py")
            if "scres-ia-payload" in str(path)
        )
    for candidate in candidates:
        if (candidate / "supply_chain").exists():
            return candidate
    return None


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


def ensure_dependencies(repo: Path) -> None:
    requirements = repo / "requirements.txt"
    if is_kaggle() and requirements.exists():
        run([sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements)])


def profile() -> tuple[str, dict[str, str]]:
    name = (os.environ.get("SCRESIA_PROFILE") or "probe").strip()
    if name not in PROFILES:
        raise ValueError(f"SCRESIA_PROFILE must be one of {sorted(PROFILES)}")
    return name, PROFILES[name]


def output_root() -> Path:
    return KAGGLE_OUTPUT_ROOT if is_kaggle() else LOCAL_OUTPUT_ROOT


def run_memory_probe(repo: Path, out: Path, prof: dict[str, str]) -> Path:
    cmd = [
        sys.executable,
        "scripts/retention_transfer.py",
        "--label",
        "continuous_dmemory",
        "--track",
        "continuous",
        "--algo",
        "ppo",
        "--observation-version",
        "v6",
        "--reward-mode",
        "ReT_excel_delta",
        "--risk-obs",
        "--risk-frequency-multiplier",
        "4.0",
        "--risk-impact-multiplier",
        "1.5",
        "--holding-cost",
        "0.0",
        "--shift-cost",
        "0.001",
        "--init-frac",
        "1.0",
        "--outcome",
        "excel_ret",
        "--rho-disruption",
        "0.85",
        "--regime-seed",
        "909",
        "--learning-rate",
        "3e-4",
        "--output-root",
        str(out),
        "--seeds",
        prof["seeds"],
        "--n-blocks",
        prof["n_blocks"],
        "--max-steps",
        prof["max_steps"],
        "--train-per-block",
        prof["train_per_block"],
        "--n-steps",
        prof["n_steps"],
        "--n-epochs",
        prof["n_epochs"],
    ]
    run(cmd, cwd=repo)
    return out / "continuous_dmemory" / "transfer.json"


def main() -> int:
    repo = extract_payload_or_use_local_repo()
    ensure_dependencies(repo)
    out = output_root()
    out.mkdir(parents=True, exist_ok=True)
    profile_name, prof = profile()

    started = datetime.now(timezone.utc).isoformat()
    transfer_path = run_memory_probe(repo, out, prof)
    transfer = json.loads(transfer_path.read_text())
    memory = transfer.get("memory_retained_minus_reset", {}).get("overall", {})
    total = transfer.get("total_retained_minus_frozen", {}).get("overall", {})
    decision: dict[str, Any] = {
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "profile": profile_name,
        "profile_args": prof,
        "transfer_path": str(transfer_path),
        "memory_retained_minus_reset": memory,
        "total_retained_minus_frozen": total,
        "positive_memory_signal": float(memory.get("mean", 0.0)) > 0.0,
        "raw_transfer": transfer,
    }
    (out / "decision.json").write_text(json.dumps(decision, indent=2, default=float))
    print(json.dumps(decision, indent=2, default=float), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
