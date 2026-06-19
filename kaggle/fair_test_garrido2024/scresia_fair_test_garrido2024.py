from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import sys

# FAIR test: train RL directly on the resilience index R (Garrido's intent:
# reward = metric), judged on R (variance_log, de-saturated). Same-action-space
# comparison only: RL-discrete vs Garrido-discrete + best discrete static;
# RL-continuous vs best continuous static. Three architectures (mlp/dmlpa/
# recurrent); recurrent gets more steps (memory needs long training). Difficulty
# = increased (where the local scout showed RL competitive). Standard resilience
# metrics (ret_p10, flow, stockout) are kept in the output as thesis-audit signals.

REPO_URL = "https://github.com/Thom-320/scres-ia.git"
TARGET_REF = os.environ.get("SCRESIA_TARGET_REF", "codex/garrido-postfix-reruns")
PROFILE = os.environ.get("SCRESIA_PROFILE", "screen").strip().lower()
REPO_DIR = Path("/kaggle/working/scres-ia")
OUTPUT_ROOT = Path("/kaggle/working/scresia_fair_outputs")
EXPORT_ROOT = Path("/kaggle/working/kaggle_outputs")

# (algo, train_timesteps). Recurrent/DMLPA get more steps than MLP.
ALGOS = [
    ("ppo_mlp", int(os.environ.get("SCRESIA_MLP_STEPS", "150000"))),
    ("dmlpa_ppo", int(os.environ.get("SCRESIA_DMLPA_STEPS", "250000"))),
    ("recurrent_ppo", int(os.environ.get("SCRESIA_RECURRENT_STEPS", "350000"))),
]
ACTION_SPACES = os.environ.get(
    "SCRESIA_ACTION_SPACES", "thesis_factorized,continuous_it_s"
).split(",")
RISK_LEVELS = os.environ.get("SCRESIA_RISK_LEVELS", "increased").split(",")
SEEDS = [int(s) for s in os.environ.get("SCRESIA_SEEDS", "42").split(",")]
EVAL_EPISODES = os.environ.get("SCRESIA_EVAL_EPISODES", "20")


def run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> int:
    print("$ " + " ".join(map(str, cmd)), flush=True)
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if check and proc.returncode != 0:
        raise RuntimeError(f"cmd failed rc={proc.returncode}")
    return proc.returncode


def resolve_device() -> str:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle GPU unavailable; enable the T4 accelerator.")
    cap = torch.cuda.get_device_capability(0)
    print("torch_device", torch.cuda.get_device_name(0), "cap", cap, flush=True)
    if cap[0] < 7:
        raise RuntimeError(f"GPU compute capability {cap} < 7; use a T4/A100.")
    return "cuda"


def smoke_cmd(*, label, algo, action_space, risk, seed, steps, device) -> list[str]:
    cmd = [
        sys.executable, "scripts/run_thesis_decision_ppo_smoke.py",
        "--label", label, "--output-root", str(OUTPUT_ROOT),
        "--algo", algo, "--action-space-mode", action_space,
        "--reward-mode", "ReT_garrido2024_raw",  # R as the training objective
        "--risk-level", risk, "--risk-occurrence-mode", "thesis_periodic",
        "--raw-material-flow-mode", "kit_equivalent_order_up_to",
        "--raw-material-order-up-to-multiplier", "2.0", "--stochastic-pt",
        "--train-timesteps", str(steps), "--eval-episodes", EVAL_EPISODES,
        "--seed", str(seed), "--eval-seed-base", "900000",
        "--garrido-cfis", "31", "--eval-ai-on-garrido-cfis",
        "--include-static-grid", "--n-envs", "8", "--norm-reward",
        "--device", device,
    ]
    if action_space == "continuous_it_s":
        cmd += ["--continuous-buffer-steps", "11"]
    if algo == "dmlpa_ppo":
        cmd += ["--history-window", "30"]
    return cmd


def profile_steps(steps: int) -> int:
    return 1024 if PROFILE == "debug" else steps


def main() -> None:
    started = datetime.now(timezone.utc)
    label_root = f"fair_g24_{PROFILE}_{started.strftime('%Y%m%dT%H%M%SZ')}"
    device = resolve_device()
    print("SCRESIA fair Garrido2024 R-as-reward test", flush=True)
    print("target_ref", TARGET_REF, "| algos", ALGOS, "| spaces", ACTION_SPACES,
          "| risks", RISK_LEVELS, "| seeds", SEEDS, flush=True)

    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)
    run(["git", "clone", "--depth", "1", "--branch", TARGET_REF, REPO_URL, str(REPO_DIR)])
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=REPO_DIR)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    cells = []
    for action_space in ACTION_SPACES:
        for algo, steps in ALGOS:
            for risk in RISK_LEVELS:
                for seed in SEEDS:
                    label = f"{label_root}_{algo}_{action_space}_{risk}_s{seed}"
                    cells.append(label)
                    print("\n" + "=" * 80 + f"\nCELL {label}\n" + "=" * 80, flush=True)
                    try:
                        run(
                            smoke_cmd(
                                label=label, algo=algo, action_space=action_space,
                                risk=risk, seed=seed, steps=profile_steps(steps),
                                device=device,
                            ),
                            cwd=REPO_DIR,
                        )
                    except Exception as exc:  # noqa: BLE001
                        print(f"CELL {label} FAILED: {exc}", flush=True)

    manifest = {
        "target_ref": TARGET_REF,
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Fair same-action-space test: train RL on R (ReT_garrido2024_raw, "
            "variance_log de-saturated index), judge on R. RL-discrete vs "
            "Garrido-discrete + best discrete static; RL-continuous vs best "
            "continuous static. Judge with scripts/analyze_track_a_cost_aware.py."
        ),
        "contract": {
            "reward_mode": "ReT_garrido2024_raw",
            "eval_index": "ReT_garrido2024 (variance_log, balance_c=0.15)",
            "risk_occurrence_mode": "thesis_periodic",
            "raw_material_flow_mode": "kit_equivalent_order_up_to",
            "raw_material_order_up_to_multiplier": 2.0,
            "stochastic_pt": True,
            "n_envs": 8,
            "norm_reward": True,
            "eval_seed_base": 900000,
            "algos": ALGOS,
            "action_spaces": ACTION_SPACES,
            "risk_levels": RISK_LEVELS,
            "seeds": SEEDS,
        },
        "fairness_rule": "compare RL only against the SAME action space (no continuous-vs-discrete)",
    }
    (OUTPUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
    export_dir = EXPORT_ROOT / label_root
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True)
    for cell in cells:
        src = OUTPUT_ROOT / cell / "policy_summary.csv"
        if src.exists():
            (export_dir / cell).mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, export_dir / cell / "policy_summary.csv")
    shutil.copy2(OUTPUT_ROOT / "manifest.json", export_dir / "manifest.json")
    print("exported:", export_dir, flush=True)
    shutil.rmtree(REPO_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
