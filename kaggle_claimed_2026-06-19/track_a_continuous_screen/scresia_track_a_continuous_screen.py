from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import sys

# Continuous_it_s sibling of the Track A tail screen. Reuses the SAME validated
# harness (preflight + run_track_a_exhaustion_sweep + smoke + paired eval + stop
# rule) but de-discretizes Garrido's inventory decision: the static baseline is
# now a FINE continuous buffer-fraction grid (smoke --include-static-grid emits a
# continuous grid for continuous_it_s), so "best static" is the best CONTINUOUS
# static, evaluated with equal reps. Reward is the same audited ReT_tail_v1.

REPO_URL = "https://github.com/Thom-320/scres-ia.git"
TARGET_REF = os.environ.get("SCRESIA_TARGET_REF", "codex/garrido-postfix-reruns")
PROFILE = os.environ.get("SCRESIA_TAIL_PROFILE", "screen").strip().lower()
REPO_DIR = Path("/kaggle/working/scres-ia")
OUTPUT_ROOT = Path("/kaggle/working/scresia_track_a_continuous_outputs")
EXPORT_ROOT = Path("/kaggle/working/kaggle_outputs")


def env_list(name: str, default: str) -> list[str]:
    return [part.strip() for part in os.environ.get(name, default).split(",") if part.strip()]


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
            "n_envs": "2",
            "n_steps": "64",
            "batch_size": "32",
            "n_epochs": "1",
            "algos": "ppo_mlp",
            "risk_levels": "increased",
        }
    if PROFILE != "screen":
        raise ValueError(
            "SCRESIA_TAIL_PROFILE must be 'screen' or 'debug', "
            f"got {PROFILE!r}."
        )
    return {
        "train_timesteps": os.environ.get("SCRESIA_TRAIN_TIMESTEPS", "100000"),
        "panel_cfis": os.environ.get("SCRESIA_PANEL_CFIS", "31-90"),
        "eval_episodes": os.environ.get("SCRESIA_EVAL_EPISODES", "20"),
        "max_steps": os.environ.get("SCRESIA_MAX_STEPS", "260"),
        "n_envs": os.environ.get("SCRESIA_N_ENVS", "8"),
        "n_steps": os.environ.get("SCRESIA_N_STEPS", "1024"),
        "batch_size": os.environ.get("SCRESIA_BATCH_SIZE", "64"),
        "n_epochs": os.environ.get("SCRESIA_N_EPOCHS", "10"),
        "algos": os.environ.get("SCRESIA_ALGOS", "ppo_mlp,recurrent_ppo,dmlpa_ppo"),
        "risk_levels": os.environ.get("SCRESIA_RISK_LEVELS", "increased,severe"),
    }


def resolve_torch_device_arg() -> str:
    import torch

    cuda_available = torch.cuda.is_available()
    print("torch_cuda_available", cuda_available, flush=True)
    if not cuda_available:
        raise RuntimeError(
            "Kaggle GPU is not available. Enable GPU acceleration before running "
            "the Track A continuous screen."
        )
    capability = torch.cuda.get_device_capability(0)
    device_name = torch.cuda.get_device_name(0)
    print("torch_device", device_name, flush=True)
    print("torch_cuda_capability", capability, flush=True)
    if capability[0] < 7:
        raise RuntimeError(
            f"Kaggle assigned {device_name} with compute capability {capability}. "
            "Use machine_shape=NvidiaTeslaT4 or select a T4/A100 GPU in the Kaggle UI."
        )
    return "cuda"


def main() -> None:
    started_at = datetime.now(timezone.utc)
    label = f"kaggle_track_a_continuous_{PROFILE}_{started_at.strftime('%Y%m%dT%H%M%SZ')}"
    cfg = profile_args()
    algos = env_list("SCRESIA_ALGOS", cfg["algos"])
    risk_levels = env_list("SCRESIA_RISK_LEVELS", cfg["risk_levels"])
    action_space_modes = env_list("SCRESIA_ACTION_SPACE_MODES", "continuous_it_s")
    device_arg = resolve_torch_device_arg()

    print("SCRESIA Track A continuous_it_s tail/recovery screen", flush=True)
    print("started_at", started_at.isoformat(), flush=True)
    print("target_ref", TARGET_REF, flush=True)
    print("profile", PROFILE, flush=True)
    print("algos", algos, flush=True)
    print("risk_levels", risk_levels, flush=True)
    print("action_space_modes", action_space_modes, flush=True)

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
    preflight_cmd = [
        sys.executable,
        "scripts/track_a_preflight_check.py",
        "--output-root",
        str(OUTPUT_ROOT / "preflight"),
        "--algo",
        algos[0],
        "--risk-level",
        risk_levels[0],
        "--panel-cfis",
        cfg["panel_cfis"],
        "--train-timesteps",
        cfg["train_timesteps"],
        "--eval-episodes",
        cfg["eval_episodes"],
        "--max-steps",
        cfg["max_steps"],
        "--n-envs",
        cfg["n_envs"],
        "--n-steps",
        cfg["n_steps"],
        "--batch-size",
        cfg["batch_size"],
        "--n-epochs",
        cfg["n_epochs"],
        "--device",
        device_arg,
        "--eval-seed-base",
        os.environ.get("SCRESIA_EVAL_SEED_BASE", "900000"),
        "--profile-eval-common-seed",
        "--ret-tail-transform",
        os.environ.get("SCRESIA_RET_TAIL_TRANSFORM", "power"),
        "--ret-tail-gamma",
        os.environ.get("SCRESIA_RET_TAIL_GAMMA", "1.25"),
        "--ret-tail-beta",
        os.environ.get("SCRESIA_RET_TAIL_BETA", "2.0"),
    ]
    run(preflight_cmd, cwd=REPO_DIR)

    sweep_cmd = [
        sys.executable,
        "scripts/run_track_a_exhaustion_sweep.py",
        "--label-prefix",
        label,
        "--output-root",
        str(OUTPUT_ROOT),
        "--algos",
        *algos,
        "--action-space-modes",
        *action_space_modes,
        "--reward-profiles",
        "ret_tail",
        "--risk-levels",
        *risk_levels,
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
        "--n-envs",
        cfg["n_envs"],
        "--n-steps",
        cfg["n_steps"],
        "--batch-size",
        cfg["batch_size"],
        "--n-epochs",
        cfg["n_epochs"],
        "--history-window",
        os.environ.get("SCRESIA_HISTORY_WINDOW", "30"),
        "--device",
        device_arg,
        "--norm-reward",
        "--eval-seed-base",
        os.environ.get("SCRESIA_EVAL_SEED_BASE", "900000"),
        "--profile-eval-common-seed",
        "--ret-tail-transform",
        os.environ.get("SCRESIA_RET_TAIL_TRANSFORM", "power"),
        "--ret-tail-gamma",
        os.environ.get("SCRESIA_RET_TAIL_GAMMA", "1.25"),
        "--ret-tail-beta",
        os.environ.get("SCRESIA_RET_TAIL_BETA", "2.0"),
    ]
    run(sweep_cmd, cwd=REPO_DIR)

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
            "Track A continuous_it_s tail/recovery screen. De-discretizes "
            "Garrido's inventory level (continuous buffer fraction) while keeping "
            "the same audited ReT_tail_v1 reward; static baseline is the best "
            "CONTINUOUS static grid with equal eval reps."
        ),
        "environment_contract": {
            "action_space_mode": action_space_modes[0],
            "risk_occurrence_mode": "thesis_periodic",
            "raw_material_flow_mode": "kit_equivalent_order_up_to",
            "raw_material_order_up_to_multiplier": 2.0,
            "inventory_period_mode": "thesis_strict",
            "stochastic_pt": True,
            "stochastic_pt_spread": 1.0,
            "reward_profile": "ret_tail",
            "reward_mode": "ReT_tail_v1",
            "ret_tail_transform": os.environ.get("SCRESIA_RET_TAIL_TRANSFORM", "power"),
            "ret_tail_gamma": float(os.environ.get("SCRESIA_RET_TAIL_GAMMA", "1.25")),
            "vec_normalize": True,
            "norm_reward": True,
            "observation_version": "v5",
            "observation_mode": "env_sdm_history_reward",
            "device": device_arg,
            "panel_cfis": cfg["panel_cfis"],
            "eval_seed_base": int(os.environ.get("SCRESIA_EVAL_SEED_BASE", "900000")),
            "profile_eval_common_seed": True,
        },
        "matrix": {
            "algos": algos,
            "risk_levels": risk_levels,
            "action_space_modes": action_space_modes,
            "pt_profiles": ["stoch_pt_hist"],
        },
        "stop_rule": {
            "promote_if": (
                "any config improves best CONTINUOUS static by >=0.02 in all-order "
                "ReT or flow_fill, or improves ret_p10_all with lower stockout and "
                "no material fill loss"
            ),
            "otherwise": "continuous adds no value over best continuous static; "
            "move to Track B or report Track A closed",
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
    preflight_report = OUTPUT_ROOT / "preflight" / "track_a_preflight_latest.json"
    if preflight_report.exists():
        shutil.copy2(preflight_report, export_dir / "track_a_preflight_latest.json")

    summary_path = export_dir / "sweep_summary.csv"
    if summary_path.exists():
        print(summary_path.read_text(encoding="utf-8"), flush=True)
    print("exported:", export_dir, flush=True)
    shutil.rmtree(REPO_DIR, ignore_errors=True)
    shutil.rmtree(OUTPUT_ROOT, ignore_errors=True)


if __name__ == "__main__":
    main()
