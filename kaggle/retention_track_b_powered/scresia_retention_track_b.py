"""Kaggle kernel: POWERED Track B retained-vs-reset with observability ablation.

Rigorous backstop before declaring the retention null: gives Track B retention a
fair, well-powered test (10 seeds, real PPO budget) with seed-clustered CIs, under
both regime-observable and regime-masked (v7 idx 30-36) conditions.

CPU-bound PPO (~450 ts/s locally). Budget ~5.4M PPO timesteps ~= 3-4h.
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
OUTPUT_ROOT = Path("/kaggle/working/scresia_retention_track_b_outputs")


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def extract_payload() -> None:
    payload = PAYLOAD if PAYLOAD.exists() else LOCAL_PAYLOAD
    if not payload.exists():
        raise FileNotFoundError("Missing scres_ia_payload.tar.gz. Rebuild before pushing.")
    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)
    REPO_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(payload, "r:gz") as archive:
        archive.extractall(REPO_DIR)


def main() -> None:
    started = datetime.now(timezone.utc)
    print("SCRESIA Track B retention (powered)", started.isoformat(), flush=True)
    extract_payload()
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=REPO_DIR)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    label = f"kaggle_track_b_retention_{started.strftime('%Y%m%dT%H%M%SZ')}"
    run(
        [
            sys.executable,
            "scripts/retention_track_b.py",
            "--label", label,
            "--output-root", str(OUTPUT_ROOT),
            "--reward-mode", "ReT_seq_v1",
            "--seeds", "8101,8102,8103,8104,8105,8106,8107,8108,8109,8110",
            "--cycles", "8",
            "--max-steps", "10",
            "--pretrain-timesteps", "80000",
            "--online-timesteps-per-cycle", "12000",
            "--n-steps", "256",
            "--batch-size", "128",
            "--n-epochs", "10",
        ],
        cwd=REPO_DIR,
    )

    out = OUTPUT_ROOT / label / "retention_track_b.json"
    if out.exists():
        data = json.loads(out.read_text(encoding="utf-8"))
        print("\n=== TRACK B RETENTION (powered) ===", flush=True)
        for cond in ("obs_full", "obs_hidden"):
            rr = data["results"][cond]["retained_minus_reset"]
            rf = data["results"][cond]["retained_minus_frozen"]
            print(
                f"{cond:10} ret-reset={rr['mean']:+.4f} +/-{rr['sem']:.4f} "
                f"ci95=[{rr['ci95_lo']:+.4f},{rr['ci95_hi']:+.4f}] (n={rr['n']})  "
                f"ret-frozen={rf['mean']:+.4f}",
                flush=True,
            )
    print("finished", datetime.now(timezone.utc).isoformat(), flush=True)


if __name__ == "__main__":
    main()
