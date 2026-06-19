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
OUTPUT_ROOT = Path("/kaggle/working/scresia_confirmatory_ppo_postfix_outputs")
EXPORT_ROOT = Path("/kaggle/working/kaggle_outputs")
PPO_ROOT_CANDIDATES = [
    Path("/kaggle/input/scresia-ppo-bestshot-artifacts"),
    Path("/kaggle/input/scresia-ppo-bestshot"),
    Path("/kaggle/input/scresia-ppo-models"),
]


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def find_ppo_root() -> Path:
    for candidate in PPO_ROOT_CANDIDATES:
        if candidate.exists():
            return candidate
    input_root = Path("/kaggle/input")
    if input_root.exists():
        if list(input_root.glob("**/kaggle_ppo_bestshot_seed*/summary.json")):
            return input_root
        if list(input_root.glob("**/ppo_adaptive/*/summary.json")):
            return input_root
        visible = sorted(str(path) for path in input_root.glob("*"))
    else:
        visible = []
    raise FileNotFoundError(
        "No PPO artifact dataset found. Checked: "
        + ", ".join(str(path) for path in PPO_ROOT_CANDIDATES)
        + f". Visible /kaggle/input entries: {visible}"
    )


def main() -> None:
    started_at = datetime.now(timezone.utc)
    ppo_root = find_ppo_root()
    print("SCRESIA confirmatory PPO post-fix", flush=True)
    print("started_at", started_at.isoformat(), flush=True)
    print("target_ref", TARGET_REF, flush=True)
    print("ppo_root", ppo_root, flush=True)

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
    label = f"kaggle_confirmatory_ppo_postfix_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    cmd = [
        sys.executable,
        "scripts/run_confirmatory_ppo_ladder.py",
        "--label",
        label,
        "--output-root",
        str(OUTPUT_ROOT),
        "--ppo-root",
        str(ppo_root),
        "--panel-cfis",
        "31-90",
        "--replications",
        "10",
        "--max-steps",
        "260",
        "--reward-mode",
        "ReT_thesis",
        "--raw-material-flow-mode",
        "kit_equivalent_order_up_to",
        "--raw-material-order-up-to-multiplier",
        "2.0",
        "--risk-occurrence-mode",
        "thesis_periodic",
        "--bootstrap-draws",
        "5000",
    ]
    run(cmd, cwd=REPO_DIR)

    run_dir = OUTPUT_ROOT / label
    manifest = {
        "target_ref": TARGET_REF,
        "label": label,
        "run_dir": str(run_dir),
        "ppo_root": str(ppo_root),
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Confirmatory PPO evaluation-only rerun under kit_equivalent_order_up_to",
    }
    (OUTPUT_ROOT / "kernel_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    export_dir = EXPORT_ROOT / label
    if export_dir.exists():
        shutil.rmtree(export_dir)
    shutil.copytree(run_dir, export_dir)
    shutil.copy2(
        OUTPUT_ROOT / "kernel_manifest.json", export_dir / "kernel_manifest.json"
    )

    report_path = export_dir / "CONFIRMATORY_PPO_LADDER.md"
    if report_path.exists():
        print(report_path.read_text(encoding="utf-8"), flush=True)
    print("exported:", export_dir, flush=True)
    shutil.rmtree(REPO_DIR, ignore_errors=True)
    shutil.rmtree(OUTPUT_ROOT, ignore_errors=True)


if __name__ == "__main__":
    main()
