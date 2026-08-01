#!/usr/bin/env python3
"""Adjudicate the already-sealed E1 screen before authorising neural training."""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "results/metric_audit/contention_service_first_v2/result.json"
CONTRACT = ROOT / "docs/PREREGISTRO_NEURAL_HEADROOM_ENV_V1_2026-08-01.md"
OUTPUT = ROOT / "results/garrido_neural_headroom_gate_v1/result.json"


def main() -> int:
    source = json.loads(SOURCE.read_text())
    h = source["H_regime_leading_component"]
    falsifiers = source.get("falsifiers", {})
    source_passed = bool(falsifiers.get("all_passed", False))
    headroom_passed = float(h["H_regime"]) >= 0.01 and float(h["lcb95"]) > 0.0
    placebo_status = "NOT_OPENED_BY_E1_CONTRACT"
    status = (
        "NEURAL_SEARCH_AUTHORIZED"
        if source_passed and headroom_passed
        else "NO_GO_NEURAL_PREMIUM_E1_HEADROOM_CLOSED"
    )
    payload = {
        "schema_version": "garrido_neural_headroom_gate_v1",
        "claim_status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_artifact": str(SOURCE),
        "source_sha256": sha256(SOURCE.read_bytes()).hexdigest(),
        "contract": str(CONTRACT),
        "contract_sha256": sha256(CONTRACT.read_bytes()).hexdigest(),
        "environment": "E1_CSSU_SPLIT_NONFUNGIBLE",
        "headroom": h,
        "falsifiers_all_passed": source_passed,
        "placebo_status": placebo_status,
        "training_authorized": bool(status == "NEURAL_SEARCH_AUTHORIZED"),
        "next_environment": (
            "E2_CSSU_PARTIALLY_OBSERVABLE_ONLY_IF_NEW_PREREGISTERED"
            if status == "NEURAL_SEARCH_AUTHORIZED"
            else "NONE_UNDER_CURRENT_GATE"
        ),
    }
    body = json.dumps(payload, indent=2, sort_keys=True)
    payload["self_sha256"] = sha256(body.encode()).hexdigest()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Saved: {OUTPUT}")
    print(f"claim_status: {status}")
    print(f"training_authorized: {payload['training_authorized']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
