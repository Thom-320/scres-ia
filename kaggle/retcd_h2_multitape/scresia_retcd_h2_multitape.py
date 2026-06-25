"""Kaggle kernel: powered multi-tape H2 dose-response for retained-vs-reset.

Training-tape only (anti-fishing). Estimates `retained - reset` per rho over
several INDEPENDENT tapes so the dose-response is not driven by a single tape's
phase composition, and reports a seed-CLUSTERED CI (tape seed = inferential unit).
CPU-bound (SimPy + block env); GPU is disabled on purpose.

Honest budget (~93 online ts/s locally; Kaggle CPU may be slower):
  3 rho x 5 tapes x 12 cycles x online 1200 + pretrain 1500
  ~= 454k online timesteps ~= 80-150 min online + eval/overhead.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile


PAYLOAD = Path("/kaggle/input/scres-ia-payload/scres_ia_payload.tar.gz")
LOCAL_PAYLOAD = Path("scres_ia_payload.tar.gz")
REPO_DIR = Path("/kaggle/working/scres-ia")
OUTPUT_ROOT = Path("/kaggle/working/scresia_retcd_h2_multitape_outputs")


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def extract_payload() -> None:
    payload = PAYLOAD if PAYLOAD.exists() else LOCAL_PAYLOAD
    if not payload.exists():
        raise FileNotFoundError(
            "Missing scres_ia_payload.tar.gz. Rebuild it locally before pushing."
        )
    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)
    REPO_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(payload, "r:gz") as archive:
        archive.extractall(REPO_DIR)


def main() -> None:
    started = datetime.now(timezone.utc)
    print("SCRESIA ReT_cd H2 multi-tape dose-response", flush=True)
    print("started_at", started.isoformat(), flush=True)

    extract_payload()
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=REPO_DIR)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    label = f"kaggle_h2_multitape_{started.strftime('%Y%m%dT%H%M%SZ')}"
    cmd = [
        sys.executable,
        "scripts/pilot_learning_regime.py",
        "--label", label,
        "--output-root", str(OUTPUT_ROOT),
        "--reward-mode", "ReT_cd",
        "--rhos", "0.3334,0.6,0.9",
        "--regime-seeds", "5000,5001,5002,5003,5004",
        "--cycles", "12",
        "--pretrain-timesteps", "1500",
        "--online-timesteps-per-cycle", "1200",
        "--learning-starts", "100",
        "--buffer-size", "5000",
        "--max-steps", "8",
    ]
    run(cmd, cwd=REPO_DIR)

    run_dir = OUTPUT_ROOT / label
    pilot = run_dir / "pilot.json"
    if pilot.exists():
        data = json.loads(pilot.read_text(encoding="utf-8"))
        print(f"\n=== {label} (clustered) ===", flush=True)
        for row in data["results"]:
            rr = row["retained_minus_reset_clustered"]
            print(
                f"rho={row['rho_disruption']:.4f} "
                f"ret-reset={rr['mean']:+.5f} +/-{rr['sem']:.5f} "
                f"(n_tapes={rr['n_tapes']}, ci95=[{rr['ci95_lo']:+.4f},{rr['ci95_hi']:+.4f}])",
                flush=True,
            )
    (OUTPUT_ROOT / "manifest.json").write_text(
        json.dumps(
            {
                "purpose": "Powered multi-tape H2 dose-response, training tapes only",
                "started_at": started.isoformat(),
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "run_dir": str(run_dir),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print("outputs:", OUTPUT_ROOT, flush=True)


if __name__ == "__main__":
    main()
