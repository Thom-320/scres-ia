#!/usr/bin/env python3
"""Quick analysis of the crosseval check run to verify PPO appears in cross-eval."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


def main() -> None:
    run_dir = Path("outputs/benchmarks/control_reward_crosseval_check")
    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}")
        return

    # Load data
    episode_path = run_dir / "episode_metrics.csv"
    if not episode_path.exists():
        print("episode_metrics.csv not found — run may still be in progress.")
        return

    with episode_path.open() as f:
        rows = list(csv.DictReader(f))
    print(f"Total episode rows: {len(rows)}")

    # Phase breakdown
    phase_counts = Counter(r["phase"] for r in rows)
    print("\nRows by phase:")
    for phase, count in sorted(phase_counts.items()):
        print(f"  {phase}: {count}")

    # Check PPO in cross-eval (the key verification)
    for phase in ["cross_eval_current", "cross_eval_severe"]:
        ppo_rows = [r for r in rows if r["phase"] == phase and r["policy"] == "ppo"]
        if ppo_rows:
            rewards = [float(r["reward_total"]) for r in ppo_rows]
            print(f"\n  PPO in {phase}: {len(ppo_rows)} rows, mean={np.mean(rewards):.3f}")
        else:
            print(f"\n  PPO in {phase}: MISSING (bug not fixed!)")

    # Summary table: all policies × all phases
    print("\n=== Policy × Phase reward means ===")
    phases = sorted(phase_counts.keys())
    policies = sorted(set(r["policy"] for r in rows))

    header = f"{'Policy':<25}" + "".join(f"{p:<25}" for p in phases)
    print(header)
    print("-" * len(header))

    for pol in policies:
        line = f"{pol:<25}"
        for phase in phases:
            phase_rows = [r for r in rows if r["phase"] == phase and r["policy"] == pol]
            if phase_rows:
                mean_r = np.mean([float(r["reward_total"]) for r in phase_rows])
                line += f"{mean_r:<25.3f}"
            else:
                line += f"{'—':<25}"
        print(line)

    # Comparison table
    comp_path = run_dir / "comparison_table.csv"
    if comp_path.exists():
        with comp_path.open() as f:
            comp_rows = list(csv.DictReader(f))
        print(f"\nComparison rows: {len(comp_rows)}")
        for r in comp_rows:
            print(f"  PPO mean={r.get('ppo_reward_mean', '?')}")
            print(f"  Best static: {r.get('best_static_policy', '?')} = {r.get('best_static_reward_mean', '?')}")
            print(f"  Beats best static: {r.get('learned_beats_best_static', '?')}")
            if "best_heuristic_policy" in r:
                print(f"  Best heuristic: {r.get('best_heuristic_policy', '?')} = {r.get('best_heuristic_reward_mean', '?')}")
                print(f"  Beats best heuristic: {r.get('learned_beats_best_heuristic', '?')}")

    # Summary JSON config
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        s = json.loads(summary_path.read_text())
        c = s.get("config", {})
        print(f"\nConfig: algo={c.get('algo')}, seeds={c.get('seeds')}, ts={c.get('train_timesteps')}")
        print(f"  eval_risk_levels={c.get('eval_risk_levels')}")


if __name__ == "__main__":
    main()
