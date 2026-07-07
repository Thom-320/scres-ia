#!/usr/bin/env python3
"""Interpretability audit for the trained Real-KAN (pykan) Track B policy.

Answers "is KAN actually learning something, or just saturating?" by loading
a frozen, already-trained checkpoint (no retraining), re-enabling pykan's
``save_act``/``symbolic_enabled`` bookkeeping (disabled during training only
for speed, per ``scripts/real_kan_extractor.py``'s docstring), running a
forward pass over real observations collected from an evaluation rollout, and
then using pykan's own introspection API:

- ``model.attribute()``: per-input-dimension attribution score -- which
  observation fields the network actually depends on.
- ``model.plot()``: renders the learned univariate edge (spline) functions.
- ``model.auto_symbolic()`` (best-effort): tries to fit a closed-form
  symbolic expression to each edge.

A degenerate/non-learning network would show near-uniform attribution across
all inputs (no discrimination) and near-flat/constant spline shapes. A real
learned representation shows concentrated attribution on a subset of inputs
and non-trivial (non-flat, non-linear) spline shapes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.audit_track_b_prevention_mechanism import (  # noqa: E402
    env_kwargs,
    load_runtime,
    predict_action,
)
from supply_chain.external_env_interface import (  # noqa: E402
    get_observation_fields,
    make_track_b_env,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experiments/real_kan_interpretability_2026-07-04"),
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n-episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=104)
    parser.add_argument("--reward-mode", default="control_v1")
    parser.add_argument("--risk-level", default="adaptive_benchmark_v2")
    parser.add_argument("--observation-version", default="v7")
    parser.add_argument("--step-size-hours", type=float, default=168.0)
    parser.add_argument(
        "--real-kan-bundles",
        nargs="+",
        type=Path,
        default=[
            Path("outputs/experiments/track_b_real_kan_fixed_rng_confirm_5seed_60k_2026-07-03"),
        ],
    )
    parser.add_argument(
        "--n-forward-samples",
        type=int,
        default=256,
        help="Max number of collected observations fed through the KAN for attribution/plot.",
    )
    return parser.parse_args()


def collect_observations(args: argparse.Namespace, runtime) -> np.ndarray:
    obs_list: list[np.ndarray] = []
    for episode in range(args.n_episodes):
        env = make_track_b_env(**env_kwargs(args))
        obs, _info = env.reset(seed=1000 + episode)
        terminated = truncated = False
        while not (terminated or truncated):
            obs_before = np.asarray(obs, dtype=np.float32).copy()
            obs_list.append(obs_before)
            action = predict_action(runtime, obs_before)
            obs, _reward, terminated, truncated, _info = env.step(action)
        env.close()
    return np.stack(obs_list, axis=0)


def main() -> None:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    figures_dir = out / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    runtime = load_runtime("real_kan", args.seed, args)
    extractor = runtime.model.policy.features_extractor
    kan = extractor.kan
    print(f"Loaded Real-KAN extractor: width={kan.width}, grid={kan.grid}, k={kan.k}")

    obs_raw = collect_observations(args, runtime)
    print(f"Collected {obs_raw.shape[0]} real observations from {args.n_episodes} episodes.")
    obs_norm = runtime.vec_norm.normalize_obs(obs_raw)

    n = min(int(args.n_forward_samples), obs_norm.shape[0])
    rng = np.random.default_rng(0)
    idx = rng.choice(obs_norm.shape[0], size=n, replace=False)
    sample = obs_norm[idx]
    x = torch.clamp(
        torch.as_tensor(sample, dtype=torch.float32), -extractor.clamp_input, extractor.clamp_input
    )

    # Re-enable interpretability bookkeeping on the frozen, already-trained
    # model -- this does not change any learned parameter.
    kan.save_act = True
    kan.symbolic_enabled = True
    for layer in kan.act_fun:
        layer.save_act = True

    with torch.no_grad():
        _ = kan(x)

    fields = list(get_observation_fields(args.observation_version))
    n_input = len(fields)

    kan.attribute(plot=False)
    input_score = kan.node_scores[0].detach().cpu().numpy().reshape(-1)
    if input_score.shape[0] != n_input:
        print(
            f"WARNING: attribute() input score length {input_score.shape[0]} != "
            f"n observation fields {n_input}; saving raw scores without field names."
        )
        field_scores = [
            {"field_index": i, "field_name": "", "attribution_score": float(s)}
            for i, s in enumerate(input_score)
        ]
    else:
        order = np.argsort(-input_score)
        field_scores = [
            {
                "field_index": int(i),
                "field_name": fields[i],
                "attribution_score": float(input_score[i]),
            }
            for i in order
        ]

    import csv

    with open(out / "input_attribution.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["field_index", "field_name", "attribution_score"])
        writer.writeheader()
        writer.writerows(field_scores)

    top10 = field_scores[:10]
    print("Top-10 attributed input fields:")
    for row in top10:
        print(f"  {row['field_name'] or row['field_index']}: {row['attribution_score']:.4f}")

    total = sum(r["attribution_score"] for r in field_scores) or 1.0
    top10_share = sum(r["attribution_score"] for r in top10) / total
    n_fields = len(field_scores)
    uniform_share = 10.0 / n_fields if n_fields else 0.0
    print(
        f"Top-10 share of total attribution: {top10_share:.3f} "
        f"(uniform/no-learning baseline would be ~{uniform_share:.3f})"
    )

    try:
        import matplotlib

        matplotlib.use("Agg")
        kan.plot(folder=str(figures_dir), beta=3, scale=0.6)
        import matplotlib.pyplot as plt

        plt.savefig(out / "kan_splines.png", dpi=150, bbox_inches="tight")
        plt.close("all")
        print(f"Saved spline plot to {out / 'kan_splines.png'}")
    except Exception as exc:  # pragma: no cover - best-effort visualization
        print(f"plot() failed (non-fatal): {exc}")

    summary = {
        "n_input_fields": n_input,
        "n_forward_samples": n,
        "top10_attribution_share": float(top10_share),
        "uniform_baseline_share": float(uniform_share),
        "learning_signal": bool(top10_share > 2.0 * uniform_share),
    }
    import json

    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
