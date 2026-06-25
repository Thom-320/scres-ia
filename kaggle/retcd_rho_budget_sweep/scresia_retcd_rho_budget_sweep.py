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
OUTPUT_ROOT = Path("/kaggle/working/scresia_retcd_rho_budget_sweep_outputs")


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def extract_payload() -> None:
    payload = PAYLOAD if PAYLOAD.exists() else LOCAL_PAYLOAD
    if not payload.exists():
        raise FileNotFoundError(
            "Missing scres_ia_payload.tar.gz. Build it locally before pushing."
        )
    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)
    REPO_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(payload, "r:gz") as archive:
        archive.extractall(REPO_DIR)


def run_pilot(label_suffix: str, *, cycles: int, pretrain: int, online: int) -> Path:
    label = (
        f"kaggle_retcd_{label_suffix}_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    cmd = [
        sys.executable,
        "scripts/pilot_learning_regime.py",
        "--label",
        label,
        "--output-root",
        str(OUTPUT_ROOT),
        "--reward-mode",
        "ReT_cd",
        "--rhos",
        "0.3334,0.6,0.9",
        "--cycles",
        str(cycles),
        "--pretrain-timesteps",
        str(pretrain),
        "--online-timesteps-per-cycle",
        str(online),
        "--learning-starts",
        "100",
        "--buffer-size",
        "5000",
        "--max-steps",
        "8",
    ]
    run(cmd, cwd=REPO_DIR)
    return OUTPUT_ROOT / label


def main() -> None:
    started = datetime.now(timezone.utc)
    print("SCRESIA ReT_cd rho-budget sweep", flush=True)
    print("started_at", started.isoformat(), flush=True)

    extract_payload()
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=REPO_DIR)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Cheap diagnostic ladder: test whether H2 appears as online budget grows.
    run_dirs = [
        run_pilot("budget100", cycles=4, pretrain=300, online=100),
        run_pilot("budget300", cycles=6, pretrain=600, online=300),
        run_pilot("budget600", cycles=8, pretrain=1000, online=600),
    ]

    manifest = {
        "purpose": "Training-tape ReT_cd rho/budget diagnostic before powered retained-vs-reset run",
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "run_dirs": [str(path) for path in run_dirs],
        "payload": str(PAYLOAD if PAYLOAD.exists() else LOCAL_PAYLOAD),
    }
    (OUTPUT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    for run_dir in run_dirs:
        pilot = run_dir / "pilot.json"
        if pilot.exists():
            data = json.loads(pilot.read_text(encoding="utf-8"))
            print(f"\n=== {run_dir.name} ===", flush=True)
            for row in data["results"]:
                rr = row["retained_minus_reset_ret"]["mean_delta"]
                rf = row["retained_minus_frozen_ret"]["mean_delta"]
                slope = row["retained_adaptation_slope"]
                print(
                    f"rho={row['rho_disruption']:.4f} "
                    f"ret-reset={rr:+.5f} "
                    f"ret-frozen={rf:+.5f} "
                    f"adapt_slope={slope:+.5f}",
                    flush=True,
                )
    print("outputs:", OUTPUT_ROOT, flush=True)


if __name__ == "__main__":
    main()
