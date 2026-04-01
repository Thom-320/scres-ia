#!/usr/bin/env python3
"""
Compare ReT_cd_v1, ReT_cd_sigmoid, and ReT_thesis across static policies.

Runs 3 static policies (S1, S2, S3) × 20 episodes × 3 reward modes in the
increased + stochastic_pt environment.

Reports:
  - Mean/std of each reward signal per policy
  - Correlation matrix between reward signals
  - Sigmoid bias demonstration (raw C-D vs sigmoid)
  - Comparison of reward distributions

Saves:
  outputs/ret_cd_comparison/comparison_results.json
  outputs/ret_cd_comparison/comparison_report.txt
  ~/Downloads/RET_CD_COMPARISON.json
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import numpy as np

# --- path setup ---
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from supply_chain.env_experimental_shifts import MFSCGymEnvShifts  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "outputs" / "ret_cd_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DOWNLOADS = Path.home() / "Downloads"

# --- Configuration ---
RISK_LEVEL = "increased"
STOCHASTIC_PT = True
N_EPISODES = 20
N_STEPS = 500
BASE_SEED = 777
STEP_SIZE_HOURS = 168.0

REWARD_MODES = ["ReT_thesis", "ReT_cd_v1", "ReT_cd_sigmoid"]

STATIC_POLICIES = {
    "S1": [-1.0, -1.0, -1.0, -1.0, -1.0],  # shift_signal < -0.33 → S=1
    "S2": [0.0, 0.0, 0.0, 0.0, 0.0],        # shift_signal in [-0.33,0.33) → S=2
    "S3": [1.0, 1.0, 1.0, 1.0, 1.0],        # shift_signal ≥ 0.33 → S=3
}


def run_episodes(
    policy_action: list[float],
    reward_mode: str,
    n_episodes: int,
    base_seed: int,
    max_steps: int,
) -> dict:
    """Run episodes with a fixed static policy and collect per-step rewards."""
    action = np.array(policy_action, dtype=np.float32)
    all_episode_rewards = []
    all_fill_rates = []
    all_step_rewards = []

    for ep in range(n_episodes):
        seed = base_seed + ep
        env = MFSCGymEnvShifts(
            risk_level=RISK_LEVEL,
            stochastic_pt=STOCHASTIC_PT,
            reward_mode=reward_mode,
            step_size_hours=STEP_SIZE_HOURS,
            max_steps=max_steps,
        )
        obs, info = env.reset(seed=seed)
        ep_reward = 0.0
        step_rewards = []
        steps = 0
        terminated = truncated = False

        while not (terminated or truncated):
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            step_rewards.append(reward)
            steps += 1

        # Terminal fill rate
        sim = getattr(getattr(env, "unwrapped", env), "sim", None)
        fill_rate = float(sim._fill_rate()) if sim and hasattr(sim, "_fill_rate") else float("nan")

        all_episode_rewards.append(ep_reward)
        all_fill_rates.append(fill_rate)
        all_step_rewards.extend(step_rewards)
        env.close()

    return {
        "episode_rewards": all_episode_rewards,
        "step_rewards": all_step_rewards,
        "fill_rates": all_fill_rates,
        "mean_episode_reward": statistics.mean(all_episode_rewards),
        "std_episode_reward": statistics.stdev(all_episode_rewards) if len(all_episode_rewards) > 1 else 0.0,
        "mean_step_reward": statistics.mean(all_step_rewards),
        "std_step_reward": statistics.stdev(all_step_rewards) if len(all_step_rewards) > 1 else 0.0,
        "mean_fill_rate": statistics.mean(all_fill_rates),
        "p10_step_reward": float(np.percentile(all_step_rewards, 10)),
        "p50_step_reward": float(np.percentile(all_step_rewards, 50)),
        "p90_step_reward": float(np.percentile(all_step_rewards, 90)),
    }


def compute_correlation(a: list[float], b: list[float]) -> float:
    """Pearson correlation between two lists."""
    if len(a) < 2:
        return float("nan")
    arr_a = np.array(a)
    arr_b = np.array(b)
    if np.std(arr_a) == 0 or np.std(arr_b) == 0:
        return float("nan")
    return float(np.corrcoef(arr_a, arr_b)[0, 1])


def main() -> None:
    print("=" * 70)
    print("ReT_cd_v1 vs ReT_thesis vs ReT_cd_sigmoid Comparison")
    print(f"Config: risk={RISK_LEVEL}, stochastic_pt={STOCHASTIC_PT}")
    print(f"       {N_EPISODES} episodes × {N_STEPS} steps per policy/mode")
    print("=" * 70)

    results: dict = {}

    for policy_name, policy_action in STATIC_POLICIES.items():
        print(f"\n--- Policy {policy_name} ---")
        results[policy_name] = {}
        for mode in REWARD_MODES:
            print(f"  Running {mode}...", end=" ", flush=True)
            r = run_episodes(
                policy_action,
                mode,
                n_episodes=N_EPISODES,
                base_seed=BASE_SEED,
                max_steps=N_STEPS,
            )
            results[policy_name][mode] = r
            print(
                f"mean_step={r['mean_step_reward']:.4f} "
                f"[p10={r['p10_step_reward']:.3f} p50={r['p50_step_reward']:.3f} "
                f"p90={r['p90_step_reward']:.3f}] "
                f"fill={r['mean_fill_rate']:.3f}"
            )

    # Compute cross-mode correlations (step-level)
    print("\n--- Step-Level Reward Correlations ---")
    corr_results: dict = {}
    for policy_name in STATIC_POLICIES:
        corr_results[policy_name] = {}
        step_rewards = {
            mode: results[policy_name][mode]["step_rewards"]
            for mode in REWARD_MODES
        }
        for i, m1 in enumerate(REWARD_MODES):
            for m2 in REWARD_MODES[i+1:]:
                key = f"{m1}_vs_{m2}"
                corr = compute_correlation(step_rewards[m1], step_rewards[m2])
                corr_results[policy_name][key] = corr
                print(f"  {policy_name}: {key} r={corr:.4f}")

    # Sigmoid bias analysis
    print("\n--- Sigmoid Bias Analysis ---")
    print("  When FR=1, AT=1: raw_CD = exp(0) = 1.0, sigmoid = σ(0) = 0.500")
    print("  Maximum sigmoid reward is ALWAYS ≤ 0.5 (because log-inputs ≤ 0)")
    print("  This represents a ~50% compression of the effective reward range.")

    sigmoid_bias: dict = {}
    for policy_name in STATIC_POLICIES:
        raw_cd_steps = results[policy_name]["ReT_cd_v1"]["step_rewards"]
        sig_steps = results[policy_name]["ReT_cd_sigmoid"]["step_rewards"]
        ratio = (
            statistics.mean(sig_steps) / statistics.mean(raw_cd_steps)
            if statistics.mean(raw_cd_steps) > 0 else float("nan")
        )
        sigmoid_bias[policy_name] = {
            "mean_raw_cd": statistics.mean(raw_cd_steps),
            "mean_sigmoid": statistics.mean(sig_steps),
            "ratio_sigmoid_over_raw": ratio,
        }
        print(f"  {policy_name}: raw_CD={statistics.mean(raw_cd_steps):.4f} "
              f"sigmoid={statistics.mean(sig_steps):.4f} "
              f"ratio={ratio:.3f}")

    # Full results object
    full_results = {
        "config": {
            "risk_level": RISK_LEVEL,
            "stochastic_pt": STOCHASTIC_PT,
            "n_episodes": N_EPISODES,
            "n_steps": N_STEPS,
            "base_seed": BASE_SEED,
            "reward_modes": REWARD_MODES,
            "policies": {k: v for k, v in STATIC_POLICIES.items()},
        },
        "results": {
            policy: {
                mode: {
                    k: v for k, v in data.items()
                    if k not in ("episode_rewards", "step_rewards", "fill_rates")
                }
                for mode, data in modes.items()
            }
            for policy, modes in results.items()
        },
        "correlations": corr_results,
        "sigmoid_bias": sigmoid_bias,
        "conclusion": {
            "recommended_mode": "ReT_cd_v1",
            "reason": (
                "Raw Cobb-Douglas uses inputs in (0,1] → output in (0,1]. "
                "Sigmoid wraps log-linear sum that is always ≤ 0 when inputs ∈ (0,1], "
                "producing maximum reward of σ(0)=0.5 — a 50% compression. "
                "ReT_cd_v1 is the correct continuous bridge for ReT_thesis piecewise."
            ),
        },
    }

    # Save outputs
    json_path = OUTPUT_DIR / "comparison_results.json"
    with open(json_path, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"\n✓ Saved: {json_path}")

    downloads_path = DOWNLOADS / "RET_CD_COMPARISON.json"
    with open(downloads_path, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"✓ Saved: {downloads_path}")

    # Text report
    lines = [
        "ReT_cd_v1 vs ReT_thesis vs ReT_cd_sigmoid Comparison Report",
        "=" * 70,
        "",
        f"Config: risk={RISK_LEVEL}, stochastic_pt={STOCHASTIC_PT}",
        f"       {N_EPISODES} episodes × {N_STEPS} steps",
        "",
        "MEAN STEP-LEVEL REWARDS",
        "-" * 50,
    ]
    header = f"{'Policy':<6} {'ReT_thesis':<14} {'ReT_cd_v1':<14} {'ReT_cd_sigmoid':<15}"
    lines.append(header)
    lines.append("-" * len(header))
    for policy_name in STATIC_POLICIES:
        row = f"{policy_name:<6} "
        for mode in REWARD_MODES:
            r = results[policy_name][mode]
            row += f"{r['mean_step_reward']:<14.4f} "
        lines.append(row)

    lines += [
        "",
        "FILL RATES",
        "-" * 50,
        f"{'Policy':<6} {'ReT_thesis':<14} {'ReT_cd_v1':<14} {'ReT_cd_sigmoid':<15}",
    ]
    for policy_name in STATIC_POLICIES:
        row = f"{policy_name:<6} "
        for mode in REWARD_MODES:
            r = results[policy_name][mode]
            row += f"{r['mean_fill_rate']:<14.3f} "
        lines.append(row)

    lines += [
        "",
        "SIGMOID BIAS (ratio sigmoid/raw_CD — should be ~0.50 at max)",
        "-" * 50,
    ]
    for policy_name, bias in sigmoid_bias.items():
        lines.append(
            f"  {policy_name}: raw_CD={bias['mean_raw_cd']:.4f}  "
            f"sigmoid={bias['mean_sigmoid']:.4f}  "
            f"ratio={bias['ratio_sigmoid_over_raw']:.3f}"
        )

    lines += [
        "",
        "CORRELATIONS (step-level)",
        "-" * 50,
    ]
    for policy_name, corrs in corr_results.items():
        for key, val in corrs.items():
            lines.append(f"  {policy_name} {key}: r={val:.4f}")

    lines += [
        "",
        "CONCLUSION",
        "-" * 50,
        "Recommended: ReT_cd_v1 (raw Cobb-Douglas, output ∈ (0,1])",
        "NOT recommended: ReT_cd_sigmoid (max reward capped at 0.5)",
        "NOT recommended for training: ReT_thesis (piecewise discontinuous)",
    ]

    txt_path = OUTPUT_DIR / "comparison_report.txt"
    txt_path.write_text("\n".join(lines))
    print(f"✓ Saved: {txt_path}")

    print("\n✓ Comparison complete.")


if __name__ == "__main__":
    main()
