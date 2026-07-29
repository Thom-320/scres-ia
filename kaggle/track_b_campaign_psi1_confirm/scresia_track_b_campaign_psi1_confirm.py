"""Kaggle kernel: Track B R2/R24 optimized campaign confirmation.

Runs the optimized Track B campaign runner with Excel ReT + CVaR reward
(`alpha=0.1`), BC warm-start, checkpoint selection, dense static frontier,
and CRN evaluation.  The payload tarball is shipped with the kernel folder so
the run does not depend only on the shared Kaggle dataset.
"""

from __future__ import annotations

from datetime import datetime, timezone
import base64
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile


PAYLOAD = Path("/kaggle/input/scres-ia-payload/scres_ia_payload.tar.gz")
SCRIPT_PAYLOAD = Path(__file__).resolve().with_name("scres_ia_payload.tar.gz")
REPO_DIR = Path("/kaggle/working/scres-ia")
OUTPUT_DIR = Path("/kaggle/working/track_b_campaign_optimized")
LOCAL_REPO_DIR = Path("outputs/kaggle/track_b_campaign_optimized/_payload_repo")
LOCAL_OUTPUT_DIR = Path("outputs/kaggle/track_b_campaign_optimized")
GATE_DIR = "outputs/experiments/track_b_headroom_matrix_risklevel_2026-06-30"


def is_kaggle() -> bool:
    return Path("/kaggle/working").exists()


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def find_payload() -> Path:
    candidates = [SCRIPT_PAYLOAD, PAYLOAD, Path("scres_ia_payload.tar.gz")]
    if is_kaggle():
        candidates.extend(sorted(Path("/kaggle/input").glob("**/scres_ia_payload.tar.gz")))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if EMBEDDED_PAYLOAD_B64.strip():
        target = Path("/kaggle/working/scres_ia_payload.tar.gz") if is_kaggle() else Path("scres_ia_payload.tar.gz")
        print(f"[kernel] writing embedded payload {target}", flush=True)
        target.write_bytes(base64.b64decode(EMBEDDED_PAYLOAD_B64))
        return target
    raise FileNotFoundError(
        "Missing scres_ia_payload.tar.gz. Expected Kaggle dataset thomaschisica/scres-ia-payload to be mounted."
    )


def extract_payload() -> Path:
    payload = find_payload()
    repo_dir = REPO_DIR if is_kaggle() else LOCAL_REPO_DIR
    print(f"[kernel] extracting payload {payload} -> {repo_dir}", flush=True)
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    repo_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(payload, "r:gz") as archive:
        archive.extractall(repo_dir)
    return repo_dir


def output_dir() -> Path:
    return OUTPUT_DIR if is_kaggle() else LOCAL_OUTPUT_DIR.resolve()


def profile_args(profile: str, out_dir: Path) -> list[str]:
    common = [
        "scripts/run_track_b_kaggle.py",
        "--gate-dir",
        GATE_DIR,
        "--output-dir",
        str(out_dir),
        "--reward-mode",
        "ReT_excel_plus_cvar",
        "--ret-excel-cvar-alpha",
        "0.1",
        "--families",
        "R2,R24",
        "--risk-levels",
        "current,increased,severe",
        "--cells-per-group",
        "1",
        "--max-cells",
        "6",
        "--target-psi",
        "1.0",
        "--observation-version",
        "v7",
        "--shifts",
        "1,2,3",
        "--op9-mults",
        "1.0",
        "--op10-mults",
        "0.5,1.0,1.5,2.0",
        "--op12-mults",
        "0.5,1.0,1.5,2.0",
    ]
    if profile == "smoke":
        return common + [
            "--seeds",
            "1",
            "--timesteps",
            "128",
            "--checkpoint-every",
            "64",
            "--bc-epochs",
            "1",
            "--max-steps",
            "4",
            "--n-envs",
            "1",
            "--n-steps",
            "16",
            "--batch-size",
            "16",
            "--n-epochs",
            "1",
            "--shifts",
            "1",
            "--op10-mults",
            "1.0",
            "--op12-mults",
            "1.0",
            "--families",
            "R2",
            "--risk-levels",
            "current",
            "--max-cells",
            "1",
        ]
    if profile != "confirmatory":
        raise ValueError("SCRESIA_PROFILE must be smoke or confirmatory")
    return common + [
        "--seeds",
        "1,2,3,4,5",
        "--timesteps",
        "60000",
        "--checkpoint-every",
        "5000",
        "--bc-epochs",
        "200",
        "--max-steps",
        "104",
        "--n-envs",
        "4",
        "--n-steps",
        "256",
        "--batch-size",
        "64",
        "--n-epochs",
        "10",
        "--learning-rate",
        "1e-4",
    ]


def summarize(out_dir: Path) -> dict:
    summary_path = out_dir / "summary.json"
    if not summary_path.exists():
        return {"summary_path": str(summary_path), "summary_exists": False}
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    ppo = payload.get("ppo", {})
    best = payload.get("best_static", {})
    verdict = payload.get("verdict", {})
    return {
        "summary_path": str(summary_path),
        "summary_exists": True,
        "ppo_ret_excel": ppo.get("ret_excel_mean"),
        "best_static_ret_excel": best.get("ret_excel_mean"),
        "ret_delta": payload.get("deltas", {}).get("ret_excel"),
        "raw_ret_win": verdict.get("raw_ret_win"),
        "raw_ret_ci_win": verdict.get("raw_ret_ci_win"),
        "ret_delta_paired_ci95_low": verdict.get("ret_delta_paired_ci95_low"),
        "ret_delta_paired_ci95_high": verdict.get("ret_delta_paired_ci95_high"),
        "same_reward_win": verdict.get("same_reward_win"),
        "pareto_ret_cost": verdict.get("pareto_ret_cost"),
        "resource_efficient_win": verdict.get("resource_efficient_win"),
        "cohens_d": verdict.get("cohens_d"),
        "h3_static_cv": verdict.get("h3_static_cv"),
        "h3_ppo_cv": verdict.get("h3_ppo_cv"),
    }


def main() -> int:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    profile = (os.environ.get("SCRESIA_PROFILE") or "confirmatory").strip()
    started = datetime.now(timezone.utc)
    print("SCRESIA Track B optimized campaign", started.isoformat(), flush=True)
    print(f"[kernel] profile={profile}", flush=True)

    repo = extract_payload()
    if is_kaggle():
        run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=repo)
    out_dir = output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    run([sys.executable, *profile_args(profile, out_dir)], cwd=repo)

    decision = summarize(out_dir)
    decision.update(
        {
            "started_at": started.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "profile": profile,
        }
    )
    (out_dir / "decision.json").write_text(json.dumps(decision, indent=2, default=float))
    print("\n=== TRACK B OPTIMIZED CAMPAIGN DECISION ===", flush=True)
    print(json.dumps(decision, indent=2, default=float), flush=True)
    return 0


EMBEDDED_PAYLOAD_B64 = """
"""


if __name__ == "__main__":
    raise SystemExit(main())
