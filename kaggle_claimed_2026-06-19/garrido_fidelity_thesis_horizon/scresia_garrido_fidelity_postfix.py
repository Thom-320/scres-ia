from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import sys

REPO_URL = "https://github.com/Thom-320/scres-ia.git"
TARGET_REF = os.environ.get("SCRESIA_TARGET_REF", "codex/garrido-postfix-reruns")
REPO_DIR = Path("/kaggle/working/scres-ia")
OUTPUT_ROOT = Path("/kaggle/working/scresia_garrido_fidelity_postfix_outputs")
EXPORT_ROOT = Path("/kaggle/working/kaggle_outputs")


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def profile_args(profile: str) -> dict[str, str]:
    if profile == "debug":
        return {
            "panel_cfis": "31,61",
            "profiles": "current,severe",
            "replications": "1",
            "horizon_mode": "fixed",
            "max_steps": "16",
            "policy_set": "minimal",
        }
    if profile == "full_260w":
        return {
            "panel_cfis": "31-90",
            "profiles": "thesis_pattern,current,increased,severe,severe_extended",
            "replications": "3",
            "horizon_mode": "fixed",
            "max_steps": "260",
            "policy_set": "minimal",
        }
    if profile == "thesis_minimal":
        return {
            "panel_cfis": "31-90",
            "profiles": "thesis_pattern,current,increased,severe,severe_extended",
            "replications": "3",
            "horizon_mode": "thesis",
            "max_steps": "260",
            "policy_set": "minimal",
        }
    raise ValueError(f"Unknown SCRESIA_FIDELITY_PROFILE={profile!r}")


def main() -> None:
    started_at = datetime.now(timezone.utc)
    profile = os.environ.get("SCRESIA_FIDELITY_PROFILE", "thesis_minimal")
    cfg = profile_args(profile)

    print("SCRESIA Garrido fidelity post-fix", flush=True)
    print("started_at", started_at.isoformat(), flush=True)
    print("target_ref", TARGET_REF, flush=True)
    print("profile", profile, flush=True)

    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)
    run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            TARGET_REF,
            REPO_URL,
            str(REPO_DIR),
        ]
    )
    run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
        cwd=REPO_DIR,
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    label = f"kaggle_{profile}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    cmd = [
        sys.executable,
        "scripts/run_garrido_static_fidelity_stress.py",
        "--label",
        label,
        "--output-root",
        str(OUTPUT_ROOT),
        "--panel-cfis",
        cfg["panel_cfis"],
        "--profiles",
        cfg["profiles"],
        "--policy-set",
        cfg["policy_set"],
        "--replications",
        cfg["replications"],
        "--horizon-mode",
        cfg["horizon_mode"],
        "--max-steps",
        cfg["max_steps"],
        "--reward-mode",
        "ReT_thesis",
        "--raw-material-flow-mode",
        "bom_total_units_order_up_to",
        "--raw-material-order-up-to-multiplier",
        "2.0",
        "--risk-occurrence-mode",
        "thesis_periodic",
        "--progress-every",
        "250",
    ]
    run(cmd, cwd=REPO_DIR)

    run_dir = OUTPUT_ROOT / label
    manifest = {
        "profile": profile,
        "profile_args": cfg,
        "target_ref": TARGET_REF,
        "run_dir": str(run_dir),
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    (OUTPUT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    report_path = run_dir / "GARRIDO_STATIC_FIDELITY_STRESS.md"
    if report_path.exists():
        print(report_path.read_text(encoding="utf-8"), flush=True)

    export_dir = EXPORT_ROOT / label
    if export_dir.exists():
        shutil.rmtree(export_dir)
    shutil.copytree(run_dir, export_dir)
    shutil.copy2(OUTPUT_ROOT / "manifest.json", export_dir / "kernel_manifest.json")
    print("exported:", export_dir, flush=True)
    print("outputs:", OUTPUT_ROOT, flush=True)
    shutil.rmtree(REPO_DIR, ignore_errors=True)
    shutil.rmtree(OUTPUT_ROOT, ignore_errors=True)


if __name__ == "__main__":
    main()
