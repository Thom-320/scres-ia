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
PROFILE = os.environ.get("SCRESIA_HISTORY_PROFILE", "screen").strip().lower()
REPO_DIR = Path("/kaggle/working/scres-ia")
OUTPUT_ROOT = Path("/kaggle/working/scresia_track_a_history_outputs")
EXPORT_ROOT = Path("/kaggle/working/kaggle_outputs")


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def profile_args() -> dict[str, str]:
    if PROFILE == "debug":
        return {
            "train_timesteps": "1024",
            "panel_cfis": "31,41",
            "eval_episodes": "1",
            "max_steps": "8",
            "n_steps": "64",
            "batch_size": "32",
            "n_epochs": "1",
        }
    if PROFILE != "screen":
        raise ValueError(
            "SCRESIA_HISTORY_PROFILE must be 'screen' or 'debug', " f"got {PROFILE!r}."
        )
    return {
        "train_timesteps": os.environ.get("SCRESIA_TRAIN_TIMESTEPS", "100000"),
        "panel_cfis": os.environ.get("SCRESIA_PANEL_CFIS", "31-90"),
        "eval_episodes": os.environ.get("SCRESIA_EVAL_EPISODES", "5"),
        "max_steps": os.environ.get("SCRESIA_MAX_STEPS", "260"),
        "n_steps": os.environ.get("SCRESIA_N_STEPS", "1024"),
        "batch_size": os.environ.get("SCRESIA_BATCH_SIZE", "64"),
        "n_epochs": os.environ.get("SCRESIA_N_EPOCHS", "10"),
    }


def main() -> None:
    started_at = datetime.now(timezone.utc)
    label = (
        f"kaggle_track_a_history_{PROFILE}_" f"{started_at.strftime('%Y%m%dT%H%M%SZ')}"
    )
    cfg = profile_args()

    print("SCRESIA Track A history screen", flush=True)
    print("started_at", started_at.isoformat(), flush=True)
    print("target_ref", TARGET_REF, flush=True)
    print("profile", PROFILE, flush=True)

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

    run(
        [
            sys.executable,
            "-c",
            (
                "import torch; "
                "print('torch_cuda_available', torch.cuda.is_available()); "
                "print('torch_device', torch.cuda.get_device_name(0) "
                "if torch.cuda.is_available() else 'cpu')"
            ),
        ],
        cwd=REPO_DIR,
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "scripts/run_track_a_exhaustion_sweep.py",
        "--label-prefix",
        label,
        "--output-root",
        str(OUTPUT_ROOT),
        "--algos",
        "recurrent_ppo",
        "dmlpa_ppo",
        "--action-space-modes",
        "thesis_factorized",
        "continuous_it_s",
        "--reward-profiles",
        "ret_ladder_steep",
        "--risk-levels",
        "severe_training",
        "war_stress_v1",
        "--pt-profiles",
        "stoch_pt_hist",
        "--use-cf-risk-profile",
        "--panel-cfis",
        cfg["panel_cfis"],
        "--train-timesteps",
        cfg["train_timesteps"],
        "--eval-episodes",
        cfg["eval_episodes"],
        "--max-steps",
        cfg["max_steps"],
        "--n-steps",
        cfg["n_steps"],
        "--batch-size",
        cfg["batch_size"],
        "--n-epochs",
        cfg["n_epochs"],
        "--history-window",
        os.environ.get("SCRESIA_HISTORY_WINDOW", "30"),
        "--device",
        "auto",
        "--stop-on-error",
    ]
    run(cmd, cwd=REPO_DIR)

    sweep_dirs = sorted(OUTPUT_ROOT.glob(f"{label}_*"))
    if not sweep_dirs:
        raise FileNotFoundError(f"No sweep output found for prefix {label}")
    run_dir = sweep_dirs[-1]
    manifest = {
        "target_ref": TARGET_REF,
        "profile": PROFILE,
        "label": label,
        "run_dir": str(run_dir),
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Track A history screening: RecurrentPPO vs DMLPA under "
            "severe_training and war_stress_v1."
        ),
        "environment_contract": {
            "risk_occurrence_mode": "thesis_periodic",
            "raw_material_flow_mode": "kit_equivalent_order_up_to",
            "raw_material_order_up_to_multiplier": 2.0,
            "stochastic_pt": True,
            "stochastic_pt_spread": 1.0,
            "reward_profile": "ret_ladder_steep",
            "observation_version": "v5",
            "observation_mode": "env_sdm_history_reward",
        },
    }
    (OUTPUT_ROOT / "kernel_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    export_dir = EXPORT_ROOT / run_dir.name
    if export_dir.exists():
        shutil.rmtree(export_dir)
    shutil.copytree(run_dir, export_dir)
    shutil.copy2(
        OUTPUT_ROOT / "kernel_manifest.json", export_dir / "kernel_manifest.json"
    )

    summary_path = export_dir / "sweep_summary.csv"
    if summary_path.exists():
        print(summary_path.read_text(encoding="utf-8"), flush=True)
    print("exported:", export_dir, flush=True)
    shutil.rmtree(REPO_DIR, ignore_errors=True)
    shutil.rmtree(OUTPUT_ROOT, ignore_errors=True)


if __name__ == "__main__":
    main()
