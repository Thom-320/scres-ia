#!/usr/bin/env python3
"""Merge Program B learner/classical validation artifacts without promotion."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"missing required evaluation: {path}")
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppo", type=Path, required=True)
    parser.add_argument("--recurrent", type=Path, required=True)
    parser.add_argument("--classical", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    ppo = read(args.ppo)
    recurrent = read(args.recurrent)
    classical = read(args.classical)
    learner_results = {"PPO_MLP": ppo, "RecurrentPPO_MLP": recurrent}

    adjudication = {
        architecture: result.get("adjudication", {})
        for architecture, result in learner_results.items()
    }
    learner_primary = {
        architecture: {
            model_name: payload.get("primary")
            for model_name, payload in result.get("contrasts", {}).items()
        }
        for architecture, result in learner_results.items()
    }
    classical_primary = {
        policy: payload.get("primary")
        for policy, payload in classical.get("contrasts", {}).items()
    }

    result = {
        "schema_version": "program_b_policy_comparison_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_artifacts": {
            "ppo_mlp": str(args.ppo),
            "recurrent_ppo_mlp": str(args.recurrent),
            "classical": str(args.classical),
        },
        "source_sha256": {
            "ppo_mlp": sha256(args.ppo),
            "recurrent_ppo_mlp": sha256(args.recurrent),
            "classical": sha256(args.classical),
        },
        "validation_tape_claim": "same burned validation block; exploratory only",
        "learner_adjudication": adjudication,
        "learner_primary_contrasts": learner_primary,
        "classical_primary_contrasts": classical_primary,
        "comparative_reading": {
            "confirmatory_promotion": False,
            "neural_superiority_claim": False,
            "rule": "Report paired service-safe endpoint, guardrails, secondary metrics, and classical-policy contrasts; do not promote from this post-hoc development/validation screen.",
        },
        "raw_evaluations": {
            "PPO_MLP": ppo,
            "RecurrentPPO_MLP": recurrent,
            "classical": classical,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "confirmatory_promotion": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
