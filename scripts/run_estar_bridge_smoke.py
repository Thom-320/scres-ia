#!/usr/bin/env python3
"""Burned-only E* DES bridge smoke and conservation receipt."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from supply_chain.arm_runner import canonical_payload_sha256  # noqa: E402
from supply_chain.estar_bridge import (  # noqa: E402
    check_expanded_bridge_smoke,
    load_tape,
)
from supply_chain.seed_custody import module_manifest  # noqa: E402


RUN_ROLE = "BURNED_BRIDGE_SMOKE"
DEFAULT_CONTRACT = ROOT / "contracts/garrido_expanded_des_e_star_v2_hcompute.json"
DEFAULT_TAPE = ROOT / (
    "results/expanded_contract_comparators_v2_preflight_1dc40c1/preflight/"
    "R1r_actual_tapes.json"
)
DEFAULT_OUTPUT = ROOT / "results/estar_expanded_bridge_smoke_v1/result.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--run-role", choices=(RUN_ROLE,), required=True)
    parser.add_argument("--replay-of", required=True)
    parser.add_argument("--tape-file", type=Path, default=DEFAULT_TAPE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    tape = load_tape(args.tape_file)
    expected_m000 = contract["flags_off_bridge"]["golden_payload_sha256"]
    result = check_expanded_bridge_smoke(
        tape,
        contract.get("expanded_des_bridge", {}).get("smoke_payload_sha256"),
        m000_expected_digest=expected_m000,
    )
    payload: dict[str, Any] = {
        "schema_version": "estar_expanded_bridge_smoke_v1",
        "claim_status": (
            "BRIDGE_SMOKE_PASS"
            if result["passed"]
            else "STOP_ESTAR_DES_BRIDGE_NOT_READY"
        ),
        "run_role": args.run_role,
        "replay_of": args.replay_of,
        "engineering_only": True,
        "scientific_execution_authorized": False,
        "fresh_seeds_opened": False,
        "learner_trained": False,
        "contract_path": str(args.contract),
        "contract_sha256": hashlib.sha256(args.contract.read_bytes()).hexdigest(),
        "fixture": {
            "path": str(args.tape_file),
            "seed": int(tape["seed"]),
            "family": str(tape["family"]),
            "horizon": float(tape["horizon"]),
        },
        "command_argv": [str(value) for value in sys.argv],
        "hardware_and_protocol": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "protocol": "one burned fixture, all eight masks, no fresh roots",
        },
        "bridge_check": result,
        "module_manifest": module_manifest(
            modules=(
                "supply_chain/estar_des_adapter.py",
                "supply_chain/estar_bridge.py",
                "supply_chain/estar_kernel.py",
                "supply_chain/supply_chain.py",
                "supply_chain/seed_custody.py",
                "supply_chain/arm_runner.py",
            ),
            script=Path(__file__),
        ),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    payload["canonical_payload_sha256"] = canonical_payload_sha256(payload)
    payload["self_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "claim_status": payload["claim_status"],
                "output": str(args.output),
                "self_sha256": payload["self_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
