from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import sys

REPO_URL = "https://github.com/Thom-320/scres-ia.git"
TARGET_COMMIT = "2ad6a62da11007fb0d7bf5dc91fbe14ba53627cc"
REPO_DIR = Path("/kaggle/working/scres-ia")
OUTPUT_ROOT = Path("/kaggle/working/scresia_garrido_fidelity_h2_thesis_outputs")
EXPORT_ROOT = Path("/kaggle/working/kaggle_outputs")


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def main() -> None:
    started_at = datetime.now(timezone.utc)
    print("SCRESIA Garrido H2/H3 thesis-horizon post-fix gate", flush=True)
    print("started_at", started_at.isoformat(), flush=True)
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

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    label = f"kaggle_h2_thesis_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    cmd = [
        sys.executable,
        "scripts/run_garrido_static_fidelity_stress.py",
        "--label",
        label,
        "--output-root",
        str(OUTPUT_ROOT),
        "--panel-cfis",
        "31-90",
        "--profiles",
        "thesis_pattern",
        "--policy-set",
        "minimal",
        "--replications",
        "3",
        "--horizon-mode",
        "thesis",
        "--reward-mode",
        "ReT_thesis",
        "--raw-material-flow-mode",
        "kit_equivalent_order_up_to",
        "--raw-material-order-up-to-multiplier",
        "2.0",
        "--progress-every",
        "100",
    ]
    run(cmd, cwd=REPO_DIR)

    run_dir = OUTPUT_ROOT / label
    manifest = {
        "target_commit": TARGET_COMMIT,
        "label": label,
        "run_dir": str(run_dir),
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "H2/H3 thesis-horizon post-fix gate, Cf31-90, thesis_pattern, minimal policies, 3 reps",
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
