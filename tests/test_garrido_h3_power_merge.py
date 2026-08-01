from __future__ import annotations

from pathlib import Path

from scripts.merge_garrido_h3_power_v1 import (
    EXPECTED_CONTEXTS,
    EXPECTED_LOCAL_SEEDS,
    EXPECTED_VPS_SEEDS,
    RUNNER,
    checks,
    file_sha256,
    recompute_seal,
    variance_trace,
)


def _payload(seeds):
    strategies = {}
    for strategy in ("neuron_memory", "neuron_reset", "ofat", "random"):
        strategies[strategy] = [
            {
                context: {"runs_to_within_1pct": float(i + 1)}
                for i, context in enumerate(EXPECTED_CONTEXTS)
            }
            for _ in seeds
        ]
    payload = {
        "contexts": list(EXPECTED_CONTEXTS),
        "seeds": list(seeds),
        "budget": 24,
        "n_configurations": 288,
        "per_context": strategies,
        "falsifiers": {"all_passed": True},
    }
    payload["self_sha256"] = recompute_seal(payload)
    return payload


def test_variance_trace_uses_six_contexts_and_sample_variance():
    payload = _payload([EXPECTED_LOCAL_SEEDS[0]])
    assert variance_trace(payload, "ofat").tolist() == [3.5]


def test_merge_checks_accept_disjoint_matching_slices():
    local = _payload(EXPECTED_LOCAL_SEEDS)
    vps = _payload(EXPECTED_VPS_SEEDS)
    checks_out = checks(local, vps, remote_runner_sha256=file_sha256(RUNNER))
    assert all(item["passed"] for item in checks_out.values())


def test_merge_checks_reject_seed_overlap():
    local = _payload(EXPECTED_LOCAL_SEEDS)
    vps = _payload(EXPECTED_LOCAL_SEEDS)
    checks_out = checks(local, vps, remote_runner_sha256=file_sha256(RUNNER))
    assert checks_out["f_merge_seeds_are_disjoint"]["passed"] is False
