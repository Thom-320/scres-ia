#!/usr/bin/env python3
"""Gate-0 preflight for Program L(e-1)."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys
from typing import Any

import gymnasium
import numpy
import simpy
import stable_baselines3
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.l_program_env import (  # noqa: E402
    CampaignTape,
    GarridoLearningEnv,
    materialize_campaign_tape,
)


def rollout(actions: list[int]) -> list[dict[str, Any]]:
    tape = materialize_campaign_tape(
        CampaignTape(
            campaign_id="gate0-identity",
            family="mixed",
            risk_level="current",
            base_seed=750001,
            horizon_weeks=len(actions),
            split="calibration",
        )
    )
    env = GarridoLearningEnv(max_steps=len(actions), buffer_level=504)
    states = []
    try:
        env.reset(
            seed=tape.base_seed,
            options={
                "campaign_tape": tape,
                "buffer_level": 504,
                "initial_state_seed": tape.base_seed,
                "initial_shift": 1,
            },
        )
        states.append(env.audit_state())
        for action in actions:
            _obs, _reward, term, trunc, _info = env.step(action)
            states.append(env.audit_state())
            if term or trunc:
                break
        return states
    finally:
        env.close()


def stable_payload(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/preflight/l_program/gate0.json"),
    )
    args = parser.parse_args()
    proxy = Path("supply_chain/data/garrido_proxy_v1_freeze_2026-07-10.json")
    proxy_payload = json.loads(proxy.read_text(encoding="utf-8"))
    expected_proxy_sha = (
        "3d7aaa14263596cd68dce8a79f1f04bcb8fca9639c10cd9ac8160586fda4ed96"
    )
    actual_proxy_sha = sha256(proxy.read_bytes()).hexdigest()
    requirements = Path("requirements-pinned.txt")
    requirements_sha = sha256(requirements.read_bytes()).hexdigest()

    actions = [0, 2, 1, 0, 2, 2, 1, 0]
    first = rollout(actions)
    second = rollout(actions)
    identity = stable_payload(first) == stable_payload(second)
    flow_ok = all(
        max(
            abs(float(value))
            for key, value in row["flow_ledger"].items()
            if key.endswith("_residual")
        )
        <= 1e-6
        for row in first
    )
    buffer_ok = len(
        {stable_payload(row["buffer_targets"]) for row in first}
    ) == 1
    delay_ok = [row["effective_shift"] for row in first[1:4]] == [1, 1, 3]

    checks = {
        "proxy_contract_id": proxy_payload.get("contract_id") == "garrido_proxy_v1",
        "proxy_training_authorized": bool(proxy_payload.get("rl_training_allowed")),
        "proxy_sha256": actual_proxy_sha == expected_proxy_sha,
        "trajectory_identity": identity,
        "mass_conservation": flow_ok,
        "buffer_invariance": buffer_ok,
        "one_week_shift_delay": delay_ok,
    }
    passed = all(checks.values())
    payload = {
        "kind": "l_program_gate0_preflight",
        "contract_id": "garrido_learning_v1",
        "status": "READY_FOR_GATE1" if passed else "BLOCKED_GATE0",
        "passed": passed,
        "checks": checks,
        "proxy_sha256": actual_proxy_sha,
        "requirements_pinned_sha256": requirements_sha,
        "trajectory_sha256": sha256(stable_payload(first)).hexdigest(),
        "runtime": {
            "python": platform.python_version(),
            "gymnasium": gymnasium.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "numpy": numpy.__version__,
            "simpy": simpy.__version__,
            "torch": torch.__version__,
        },
        "ppo_trained": False,
        "virgin_tapes_opened": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
