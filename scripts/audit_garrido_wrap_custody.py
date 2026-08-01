#!/usr/bin/env python3
"""Build the custody manifest for the active Garrido--WRAP paper lane."""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUTPUT = ROOT / "results/garrido_wrap_custody_manifest_v1.json"


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return sha256(path.read_bytes()).hexdigest()


def repository_head() -> str:
    """Return the commit being audited, rather than a stale hand-copied hash."""
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def record(
    *,
    artifact: str,
    runner: str | None,
    contract: str,
    status: str,
    allowed_use: str,
) -> dict[str, object]:
    artifact_path = ROOT / artifact
    runner_path = ROOT / runner if runner else None
    contract_path = ROOT / contract
    return {
        "artifact": artifact,
        "artifact_exists": artifact_path.exists(),
        "artifact_sha256": file_sha256(artifact_path),
        "runner": runner,
        "runner_exists": bool(runner_path and runner_path.exists()),
        "runner_sha256": file_sha256(runner_path) if runner_path else None,
        "contract": contract,
        "contract_exists": contract_path.exists(),
        "contract_sha256": file_sha256(contract_path),
        "status": status,
        "allowed_use": allowed_use,
    }


def build_manifest() -> dict[str, object]:
    contract = "docs/GARRIDO_WRAP_SCRES_AI_CONTRACT_V1.md"
    return {
        "schema_version": "garrido_wrap_custody_manifest_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "paper_lane": "garrido_wrap_v0_cie",
        "repository_head": repository_head(),
        "records": [
            record(
                artifact="results/garrido_wrap_source_audit/result.json",
                runner="scripts/audit_garrido_wrap_sources.py",
                contract=contract,
                status="DEVELOPMENT_SOURCE_AUDIT",
                allowed_use="source and Cf coverage only",
            ),
            record(
                artifact="results/garrido_wrap_q1/result.json",
                runner="scripts/build_garrido_fig5_surrogate.py",
                contract=contract,
                status="NO_GO_NEURAL_PREMIUM_IN_WRAP_PANEL",
                allowed_use="development Q1 result; no neural promotion",
            ),
            record(
                artifact="results/garrido_wrap_q2_smoke_2016h/result.json",
                runner="scripts/run_garrido_wrap_closed_loop.py",
                contract=contract,
                status="DEVELOPMENT_Q2_SMOKE_ONLY",
                allowed_use="interface smoke; not H4 confirmation",
            ),
            record(
                artifact="results/garrido_meta_learner_thesis90_v2/result.json",
                runner="scripts/run_meta_learner_thesis90_v1.py",
                contract="docs/PREREGISTRO_META_APRENDIZ_V2_2026-08-01.md",
                status="SURFACE_REPLAY_ONLY",
                allowed_use="algorithmic replay and leakage audit only",
            ),
            record(
                artifact="results/garrido_cssu_liveness_gate_v1/result.json",
                runner="scripts/run_cssu_liveness_gate.py",
                contract="docs/PREREGISTRO_CSSU_LIVENESS_2026-08-01.md",
                status="GATE_A_PASS_GATE_B_HOLD",
                allowed_use="split CSSU action liveness only",
            ),
            record(
                artifact="results/garrido_neural_headroom_gate_v1/result.json",
                runner="scripts/adjudicate_neural_headroom_gate_v1.py",
                contract="docs/PREREGISTRO_NEURAL_HEADROOM_ENV_V1_2026-08-01.md",
                status="HOLD_E1_PLACEBO_NOT_OPENED",
                allowed_use="E1 incomplete boundary screen; no MLP/PPO training authorization",
            ),
            record(
                artifact="results/garrido_q2_des288_v1/result.json",
                runner="scripts/run_garrido_q2_des288_v1.py",
                contract="docs/PREREGISTRO_GARRIDO_Q2_DES288_V1_2026-08-01.md",
                status="READY_NOT_STARTED_DES288",
                allowed_use="none until full DES-288 seal and falsifier merge",
            ),
            record(
                artifact="results/garrido_meta_learner_h3power_merge_v1/result.json",
                runner="scripts/merge_garrido_h3_power_v1.py",
                contract="docs/PREREGISTRO_H3_POTENCIA_2026-08-01.md",
                status="PENDING_H3_INPUTS",
                allowed_use="none until local/VPS seals, source hash and merge falsifiers pass",
            ),
            record(
                artifact="results/garrido_meta_learner_h3power_local/result.json",
                runner="scripts/run_meta_learner_over_configs_v1.py",
                contract="docs/PREREGISTRO_H3_POTENCIA_2026-08-01.md",
                status="PENDING_ACTIVE_LOCAL_RUN",
                allowed_use="none until merge and falsifiers",
            ),
            record(
                artifact="results/garrido_meta_learner_h3power_vps/result.json",
                runner="scripts/run_meta_learner_over_configs_v1.py",
                contract="docs/PREREGISTRO_H3_POTENCIA_2026-08-01.md",
                status="PENDING_ACTIVE_VPS_RUN",
                allowed_use="none until merge and falsifiers",
            ),
        ],
        "retired_claims": {
            "old_meta_learner_contrasts": "RETIRED_DRIVER_LEAK",
            "old_h2_curve": "RETIRED_DRIVER_LEAK",
            "ret_excel_family_orientation": "RETRACTED_SEED_BLOCK_ARTIFACT",
        },
    }


def main() -> int:
    payload = build_manifest()
    body = json.dumps(payload, indent=2, sort_keys=True)
    payload["self_sha256"] = sha256(body.encode()).hexdigest()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Saved: {OUTPUT}")
    print(f"self_sha256: {payload['self_sha256']}")
    for item in payload["records"]:
        print(f"{item['status']}: {item['artifact']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
