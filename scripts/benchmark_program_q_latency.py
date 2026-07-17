#!/usr/bin/env python3
"""Hardware-matched online decision-latency benchmark for Program Q policies."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import platform
import statistics
import time
import tracemalloc
from typing import Callable, Iterable

import numpy as np
import torch


def benchmark_callable(
    policy: Callable[[np.ndarray], object],
    observations: Iterable[np.ndarray],
    *,
    warmup: int = 100,
    repeats: int = 1_000,
) -> dict:
    rows = [np.asarray(row, dtype=np.float32) for row in observations]
    if not rows:
        raise ValueError("observations cannot be empty")
    for index in range(warmup):
        policy(rows[index % len(rows)])
    latencies = []
    failures = 0
    tracemalloc.start()
    for index in range(repeats):
        started = time.perf_counter_ns()
        try:
            policy(rows[index % len(rows)])
        except Exception:
            failures += 1
        latencies.append((time.perf_counter_ns() - started) / 1_000_000.0)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "warmup": warmup,
        "repeats": repeats,
        "batch_size": 1,
        "median_ms": float(statistics.median(latencies)),
        "p95_ms": float(np.quantile(latencies, 0.95)),
        "mean_ms": float(statistics.mean(latencies)),
        "failures": failures,
        "tracemalloc_peak_bytes": int(peak),
    }


def load_factory(specification: str):
    module_name, function_name = specification.split(":", 1)
    return getattr(importlib.import_module(module_name), function_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factory", required=True, help="module:function -> (policy, observations)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=1_000)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    torch.set_num_threads(args.torch_threads)
    setup_started = time.perf_counter()
    policy, observations = load_factory(args.factory)()
    setup_seconds = time.perf_counter() - setup_started
    payload = benchmark_callable(
        policy, observations, warmup=args.warmup, repeats=args.repeats
    )
    payload.update(
        {
            "schema_version": "program_q_latency_benchmark_v1",
            "factory": args.factory,
            "setup_seconds_excluded_from_online_latency": setup_seconds,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
