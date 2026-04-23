#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.dkana import DKANAPolicy


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the DKANA starter policy by behavior cloning from a "
            "DKANA-ready trajectory dataset."
        )
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    row_matrices = np.load(args.dataset_dir / "dkana_row_matrices.npy").astype(
        np.float32
    )
    config_context = np.load(args.dataset_dir / "dkana_config_context.npy").astype(
        np.float32
    )
    action_targets = np.load(args.dataset_dir / "dkana_action_targets.npy").astype(
        np.float32
    )
    time_mask = np.load(args.dataset_dir / "dkana_time_mask.npy").astype(bool)
    metadata = load_json(args.dataset_dir / "metadata.json")

    dataset = TensorDataset(
        torch.from_numpy(row_matrices),
        torch.from_numpy(config_context),
        torch.from_numpy(time_mask),
        torch.from_numpy(action_targets),
    )
    val_size = int(round(len(dataset) * args.validation_fraction))
    val_size = (
        min(max(val_size, 1), max(1, len(dataset) - 1)) if len(dataset) > 1 else 0
    )
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(args.seed)
    if val_size > 0:
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=generator,
        )
    else:
        train_dataset = dataset
        val_dataset = dataset

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    model = DKANAPolicy(
        config_dim=config_context.shape[-1],
        action_dim=action_targets.shape[-1],
        latent_dim=args.latent_dim,
        num_heads=args.num_heads,
        max_rows=row_matrices.shape[2] + 1,
        max_sequence_length=row_matrices.shape[1],
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    def run_epoch(loader: DataLoader, *, train: bool) -> float:
        model.train(train)
        losses: list[float] = []
        for rows, config, mask, target in loader:
            distribution = model(rows, config, mask)
            nll = -distribution.log_prob(target).mean()
            mse = torch.nn.functional.mse_loss(distribution.mean, target)
            loss = nll + mse
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            losses.append(float(loss.detach().cpu()))
        return float(np.mean(losses)) if losses else float("nan")

    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(train_loader, train=True)
        with torch.no_grad():
            val_loss = run_epoch(val_loader, train=False)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )
        print(f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "config_dim": int(config_context.shape[-1]),
            "action_dim": int(action_targets.shape[-1]),
            "latent_dim": int(args.latent_dim),
            "num_heads": int(args.num_heads),
            "max_rows": int(row_matrices.shape[2] + 1),
            "max_sequence_length": int(row_matrices.shape[1]),
        },
        "dataset_metadata": metadata,
        "training_history": history,
    }
    torch.save(checkpoint, args.output_dir / "dkana_policy.pt")
    with (args.output_dir / "training_metrics.json").open(
        "w", encoding="utf-8"
    ) as file_obj:
        json.dump(
            {"history": history, "checkpoint": "dkana_policy.pt"}, file_obj, indent=2
        )
    print(f"Saved DKANA checkpoint to {args.output_dir / 'dkana_policy.pt'}")


if __name__ == "__main__":
    main()
