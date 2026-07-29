#!/usr/bin/env python3
"""Custody-aware DES executor for the frozen wartime GSA overlay.

Only ``--mode benchmark`` is authorized by the current freeze.  Scientific mode
fails closed until a separately committed authorization names this exact source,
contract, configuration manifest, policy manifest and command manifest.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from types import SimpleNamespace
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research.paper2_exhaustive_search.restricted_timing_oracle import (  # noqa: E402
    ScheduleSpec,
    evaluate_schedule,
    posture_from_label,
)
from research.paper2_exhaustive_search.war_risk_gsa_v2 import (  # noqa: E402
    PolicyMetrics,
    TimingTapeEvaluation,
    compute_h_timing_safe,
)
from research.paper2_exhaustive_search.war_stress_risk_tapes import (  # noqa: E402
    MASK_RISKS,
    build_risk_tape,
)
from supply_chain.event_triggered_env import make_event_triggered_track_a_env  # noqa: E402
from scripts.build_war_stress_policy_manifest import (  # noqa: E402
    build_manifest as build_policy_manifest,
)


SCIENTIFIC_SEED_MIN = 7_470_001
SCIENTIFIC_SEED_MAX = 7_470_012
BENCHMARK_SEED = 94_700_001
MAX_AUTHORIZABLE_EPISODES = 2_000_000
MAX_AUTHORIZABLE_PROJECTED_WALL_DAYS = 7.0


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _load_configuration(manifest: Mapping[str, Any], config_id: str) -> dict[str, Any]:
    rows = [*manifest["morris"]["rows"], *manifest["qmc_pool"]["rows"]]
    matches = [row for row in rows if row["config_id"] == config_id]
    if len(matches) != 1:
        raise ValueError(f"configuration {config_id!r} occurs {len(matches)} times")
    return dict(matches[0])


def _event_objects(events: tuple[dict[str, Any], ...]) -> list[SimpleNamespace]:
    return [SimpleNamespace(**row) for row in events]


def _policy_spec(family: str, payload: Any) -> ScheduleSpec:
    if family == "constant":
        return ScheduleSpec("constant", "benchmark_constant", False)
    return ScheduleSpec(family, f"benchmark_{family}", payload)


def _evaluate(
    *,
    tape: Any,
    low_label: str,
    high_label: str,
    family: str,
    payload: Any,
    max_daily_steps: int,
) -> dict[str, Any]:
    return evaluate_schedule(
        seed=int(tape.tape_id),
        risk_overrides={},
        low=posture_from_label(low_label),
        high=posture_from_label(high_label),
        spec=_policy_spec(family, payload),
        max_daily_steps=int(max_daily_steps),
        known_risk_events=_event_objects(tape.events),
        risk_event_tape=tape.events,
        enabled_risks=MASK_RISKS[tape.mask],
    )


def _probe_treatment_start(*, seed: int, max_daily_steps: int) -> float:
    env = make_event_triggered_track_a_env(
        init_frac=0.0,
        init_shifts=1,
        max_steps=int(max_daily_steps),
        enabled_risks=(),
        stochastic_pt=False,
        priming_enabled=False,
    )
    try:
        env.reset(seed=int(seed))
        return float(env.unwrapped.sim.env.now)
    finally:
        env.close()


def run_benchmark(
    *,
    configuration: Mapping[str, Any],
    max_daily_steps: int,
    projected_workers: int,
) -> dict[str, Any]:
    treatment_start = _probe_treatment_start(
        seed=BENCHMARK_SEED, max_daily_steps=max_daily_steps
    )
    horizon_hours = treatment_start + float(max_daily_steps) * 24.0
    tape = build_risk_tape(
        configuration,
        tape_id=BENCHMARK_SEED,
        horizon_hours=horizon_hours,
        start_hour=treatment_start,
    )
    policies = (
        ("constant::f0_S1", "f0_S1", "f0_S1", "constant", None),
        (
            "restricted::f0_S1::f0.125_S2::entry0::exit72",
            "f0_S1",
            "f0.125_S2",
            "restricted_privileged",
            {"entry_offset_hours": 0, "exit_offset_hours": 72},
        ),
    )
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for policy_id, low, high, family, payload in policies:
        row_started = time.perf_counter()
        metrics = _evaluate(
            tape=tape,
            low_label=low,
            high_label=high,
            family=family,
            payload=payload,
            max_daily_steps=max_daily_steps,
        )
        rows.append(
            {
                "policy_id": policy_id,
                "family": family,
                "wall_seconds": time.perf_counter() - row_started,
                "metrics": metrics,
            }
        )
    elapsed = time.perf_counter() - started
    comparator = PolicyMetrics(
        policy_id=rows[0]["policy_id"],
        metrics=rows[0]["metrics"],
        event_tape_sha256=tape.event_tape_sha256,
        exogenous_base_stream_sha256=tape.base_stream_sha256,
    )
    candidate = PolicyMetrics(
        policy_id=rows[1]["policy_id"],
        metrics=rows[1]["metrics"],
        event_tape_sha256=tape.event_tape_sha256,
        exogenous_base_stream_sha256=tape.base_stream_sha256,
    )
    response = compute_h_timing_safe(
        str(configuration["config_id"]),
        [TimingTapeEvaluation(BENCHMARK_SEED, comparator, (candidate,))],
    )
    policy_manifest = build_policy_manifest()
    morris_configurations = 570
    scientific_tapes = 3
    projected_episodes = (
        morris_configurations
        * scientific_tapes
        * int(policy_manifest["total_policy_templates"])
    )
    seconds_per_episode = elapsed / len(rows)
    horizon_scale = (104 * 7) / max_daily_steps
    projected_seconds = (
        projected_episodes
        * seconds_per_episode
        * horizon_scale
        / max(1, int(projected_workers))
    )
    projected_wall_days = projected_seconds / 86_400.0
    compute_gate_pass = bool(
        projected_episodes <= MAX_AUTHORIZABLE_EPISODES
        and projected_wall_days <= MAX_AUTHORIZABLE_PROJECTED_WALL_DAYS
    )
    return {
        "schema_version": "war_stress_gsa_executor_benchmark_v1",
        "status": (
            "PASS_NONSCIENTIFIC_PIPELINE_BUT_STOP_COMPUTE_INFEASIBLE"
            if not compute_gate_pass
            else "PASS_NONSCIENTIFIC_PIPELINE_AND_COMPUTE_GATE"
        ),
        "scientific_seeds_opened": False,
        "benchmark_seed": BENCHMARK_SEED,
        "configuration_id": configuration["config_id"],
        "configuration": dict(configuration),
        "horizon_days": int(max_daily_steps),
        "treatment_start_hour": treatment_start,
        "risk_tape": {
            "event_count": len(tape.events),
            "r3_event_count": tape.r3_event_count,
            "base_stream_sha256": tape.base_stream_sha256,
            "event_tape_sha256": tape.event_tape_sha256,
        },
        "policy_rows": rows,
        "h_timing_safe_fixture": {
            **asdict(response),
            "deltas": response.deltas.tolist(),
        },
        "runtime": {
            "wall_seconds": elapsed,
            "seconds_per_short_episode": seconds_per_episode,
            "projected_scientific_episodes": projected_episodes,
            "projected_workers": int(projected_workers),
            "linear_horizon_scale": horizon_scale,
            "projected_wall_days": projected_wall_days,
            "maximum_authorizable_episodes": MAX_AUTHORIZABLE_EPISODES,
            "maximum_authorizable_projected_wall_days": (
                MAX_AUTHORIZABLE_PROJECTED_WALL_DAYS
            ),
            "compute_gate_pass": compute_gate_pass,
            "projection_is_diagnostic_only": True,
        },
        "claim_boundary": {
            "non_scientific_fixture_only": True,
            "morris_authorized": False,
            "sobol_authorized": False,
            "prim_authorized": False,
            "learner_authorized": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("benchmark", "scientific"), required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "research/paper2_exhaustive_search/war_stress_gsa_overlay_manifest_20260716.json"
        ),
    )
    parser.add_argument(
        "--config-id",
        default="morris::LOC_SURGE::coincident::00::00",
    )
    parser.add_argument("--max-daily-steps", type=int, default=14)
    parser.add_argument("--projected-workers", type=int, default=32)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--scientific-authorization", type=Path)
    args = parser.parse_args()

    if args.mode == "scientific":
        # The current repository deliberately has no valid authorization file.
        # Presence alone is insufficient: a future implementation must add a
        # content-addressed schema check before this branch can execute.
        raise RuntimeError(
            "scientific GSA execution is not authorized; benchmark and freeze only"
        )
    if args.scientific_authorization is not None:
        raise RuntimeError("benchmark must not receive scientific authorization")
    if SCIENTIFIC_SEED_MIN <= BENCHMARK_SEED <= SCIENTIFIC_SEED_MAX:
        raise AssertionError("benchmark seed overlaps scientific custody")

    manifest = json.loads(args.manifest.read_text())
    configuration = _load_configuration(manifest, args.config_id)
    run_dir = args.run_dir.resolve()
    if run_dir.exists():
        existing = [path for path in run_dir.rglob("*") if path.is_file()]
        allowed = {
            run_dir / "custody" / "watcher_ready.json",
            run_dir / "custody" / "heartbeat.json",
        }
        if any(path not in allowed for path in existing):
            raise FileExistsError("run identity already contains producer artifacts")
    else:
        run_dir.mkdir(parents=True)
    custody = run_dir / "custody"
    custody.mkdir(exist_ok=True)
    _write_json_atomic(
        custody / "producer_control.json",
        {
            "schema_version": "war_stress_gsa_producer_control_v1",
            "created_at_utc": _now(),
            "mode": args.mode,
            "pid": os.getpid(),
            "pgid": os.getpgid(0),
            "sid": os.getsid(0),
            "cwd": str(Path.cwd()),
            "argv": sys.argv,
            "manifest_sha256": _sha256(args.manifest),
        },
    )
    _write_json_atomic(
        custody / "environment_manifest.json",
        {
            "captured_at_utc": _now(),
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        },
    )
    _write_json_atomic(
        custody / "resume_cursor.json",
        {"status": "RUNNING_BENCHMARK", "completed_policies": 0, "scientific": False},
    )
    try:
        result = run_benchmark(
            configuration=configuration,
            max_daily_steps=int(args.max_daily_steps),
            projected_workers=int(args.projected_workers),
        )
        _write_json_atomic(run_dir / "result.json", result)
        _write_json_atomic(
            custody / "resume_cursor.json",
            {
                "status": "COMPLETE_BENCHMARK",
                "completed_policies": len(result["policy_rows"]),
                "scientific": False,
            },
        )
        _write_json_atomic(
            custody / "producer_exit.json",
            {"finished_at_utc": _now(), "returncode": 0, "result_exists": True},
        )
        return 0
    except Exception as exc:
        _write_json_atomic(
            custody / "producer_exit.json",
            {
                "finished_at_utc": _now(),
                "returncode": 1,
                "result_exists": False,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
