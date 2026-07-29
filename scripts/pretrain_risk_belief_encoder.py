#!/usr/bin/env python3
"""Pretrain a belief encoder trunk (Ruta A) with a supervised BCE loss only.

No PPO loss, no RL. Loads (full v10 observation, future-R24 label) pairs from
``build_track_b_v10_belief_pretrain_dataset.py``'s output, trains the encoder
trunk (``MLPBeliefExtractor`` or ``RealKANBeliefExtractor``) plus a temporary
linear head with ``BCEWithLogitsLoss``, then saves ONLY the trunk's
``state_dict()`` -- PPO's own actor/critic heads are separate and untouched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym  # noqa: E402

from scripts.belief_extractor import MLPBeliefExtractor, RealKANBeliefExtractor  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("outputs/experiments/track_b_v10_belief_pretrain_dataset_2026-07-04"),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/experiments/track_b_v10_belief_pretrain_dataset_2026-07-04/mlp_belief_trunk.pt"),
    )
    parser.add_argument("--architecture", choices=("ppo_mlp", "real_kan"), default="ppo_mlp")
    parser.add_argument("--targets", nargs="+", default=["y_R24_1w", "y_R24_2w"])
    parser.add_argument("--features-dim", type=int, default=64)
    parser.add_argument("--hidden-width", type=int, default=64)
    parser.add_argument("--kan-features-dim", type=int, default=32)
    parser.add_argument("--kan-hidden-width", type=int, default=32)
    parser.add_argument("--kan-grid", type=int, default=3)
    parser.add_argument("--kan-k", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    x = np.load(args.dataset_dir / "x_v10.npy")
    import csv

    with open(args.dataset_dir / "labels.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    y = np.asarray(
        [[int(row[target]) for target in args.targets] for row in rows],
        dtype=np.float32,
    )

    groups = np.asarray([int(row["seed"]) for row in rows])
    unique_groups = np.unique(groups)
    train_groups, test_groups = train_test_split(unique_groups, test_size=0.2, random_state=args.seed)
    train_mask = np.isin(groups, train_groups)
    test_mask = np.isin(groups, test_groups)

    x_train, x_test = x[train_mask], x[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(x.shape[1],), dtype=np.float32)
    if args.architecture == "ppo_mlp":
        trunk = MLPBeliefExtractor(obs_space, features_dim=args.features_dim, hidden_width=args.hidden_width)
        features_dim = args.features_dim
    else:
        trunk = RealKANBeliefExtractor(
            obs_space,
            features_dim=args.kan_features_dim,
            hidden_width=args.kan_hidden_width,
            grid=args.kan_grid,
            k=args.kan_k,
            seed=args.seed,
        )
        features_dim = args.kan_features_dim

    head = torch.nn.Linear(features_dim, len(args.targets))
    params = list(trunk.parameters()) + list(head.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    x_train_t = torch.as_tensor(x_train, dtype=torch.float32)
    y_train_t = torch.as_tensor(y_train, dtype=torch.float32)
    n = x_train_t.shape[0]

    trunk.train()
    head.train()
    for epoch in range(int(args.epochs)):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for start in range(0, n, int(args.batch_size)):
            idx = perm[start : start + int(args.batch_size)]
            xb, yb = x_train_t[idx], y_train_t[idx]
            optimizer.zero_grad()
            features = trunk(xb)
            logits = head(features)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * len(idx)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"epoch {epoch + 1}/{args.epochs} train_bce={epoch_loss / n:.4f}")

    trunk.eval()
    head.eval()
    with torch.no_grad():
        test_logits = head(trunk(torch.as_tensor(x_test, dtype=torch.float32)))
        test_probs = torch.sigmoid(test_logits).numpy()
    aucs = {}
    for i, target in enumerate(args.targets):
        if len(np.unique(y_test[:, i])) < 2:
            aucs[target] = None
        else:
            aucs[target] = float(roc_auc_score(y_test[:, i], test_probs[:, i]))
    print("Held-out AUC (grouped by seed):", aucs)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trunk.state_dict(), args.output_path)
    head_path = args.output_path.with_name(args.output_path.stem + "_head.pt")
    torch.save(head.state_dict(), head_path)
    meta = {
        "architecture": args.architecture,
        "targets": args.targets,
        "features_dim": features_dim,
        "n_train": int(n),
        "n_test": int(x_test.shape[0]),
        "held_out_auc": aucs,
        "dataset_dir": str(args.dataset_dir),
        "head_path": str(head_path),
    }
    meta_path = args.output_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved trunk state_dict to {args.output_path}")
    print(f"Saved head state_dict to {head_path}")
    print(f"Saved meta to {meta_path}")


if __name__ == "__main__":
    main()
