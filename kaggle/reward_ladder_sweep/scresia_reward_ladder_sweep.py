from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

REPO_URL = "https://github.com/Thom-320/scres-ia.git"
TARGET_COMMIT = "2541a43f3cf1016b32a671c731bd7bb87414000a"
REPO_DIR = Path("/kaggle/working/scres-ia")
OUTPUT_ROOT = Path("/kaggle/working/scresia_reward_ladder_sweep_outputs")


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def main() -> None:
    print("SCRESIA reward ladder sweep", flush=True)
    print("started_at", datetime.now(timezone.utc).isoformat(), flush=True)
    print("target_commit", TARGET_COMMIT, flush=True)

    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)
    run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)])
    run(["git", "fetch", "--depth", "1", "origin", TARGET_COMMIT], cwd=REPO_DIR)
    run(["git", "checkout", TARGET_COMMIT], cwd=REPO_DIR)

    run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
        cwd=REPO_DIR,
    )

    output_root = OUTPUT_ROOT
    output_root.mkdir(parents=True, exist_ok=True)
    profile = os.environ.get("SCRESIA_REWARD_PROFILE", "serious_common")
    if profile == "debug":
        replications = "1"
        max_steps = "12"
        extra_args: list[str] = []
    elif profile == "capacity_panel":
        replications = "1"
        max_steps = "260"
        extra_args = ["--garrido-cfis", "61-90"]
    else:
        replications = "3"
        max_steps = "260"
        extra_args = []

    label = f"kaggle_{profile}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    cmd = [
        sys.executable,
        "scripts/audit_thesis_reward_surface.py",
        "--label",
        label,
        "--output-root",
        str(output_root),
        "--reward-modes",
        "ReT_cd_v1",
        "ReT_seq_v1",
        "ReT_ladder_v1",
        "--replications",
        replications,
        "--risk-level",
        "increased",
        "--observation-version",
        "v5",
        "--observation-mode",
        "env_sdm_history_reward",
        "--step-size-hours",
        "168",
        "--max-steps",
        max_steps,
        "--stochastic-pt",
        "--progress-every",
        "25",
        *extra_args,
    ]
    run(cmd, cwd=REPO_DIR)

    run_dir = output_root / label
    manifest = {
        "profile": profile,
        "target_commit": TARGET_COMMIT,
        "run_dir": str(run_dir),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    report_path = run_dir / "REWARD_SURFACE_AUDIT.md"
    if report_path.exists():
        print(report_path.read_text(encoding="utf-8"), flush=True)
    print("outputs:", output_root, flush=True)


if __name__ == "__main__":
    main()
