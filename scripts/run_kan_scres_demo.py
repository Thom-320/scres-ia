#!/usr/bin/env python3
"""Real Kolmogorov-Arnold Network (pykan, Liu et al. 2024) demonstration.

Garrido et al. (2024) propose, in their Fig. 5, inserting a neural network
between the DES decision variables (rho_i) and the SCRES output metric, to
close the open loop between "experiment design" and "measured resilience".
This script instantiates that exact architecture with an OFFICIAL KAN
(pykan), not the RBF-skip approximation used in the earlier sidecar smoke
test (scripts/kan_extractor.py, docs/KAN_SIDECAR_SMOKE_2026-07-02.md).

Task: learn the mapping
    (shift level S, Op10 dispatch multiplier, Op12 dispatch multiplier)
        -> Excel ReT (order_ret_excel)
from the 147-cell dense static dispatch frontier (the same evidence bundle
behind the paper's Table 4 / Figure 4). This is supervised regression on a
small, clean, deterministic-ish design table -- fast, honest, and directly
interpretable via pykan's built-in spline plotting.

Outputs:
  outputs/experiments/kan_scres_demo_2026-07-02/
    kan_fit_summary.json   -- R^2, MSE for KAN vs. linear vs. MLP baselines
    kan_splines.png        -- pykan's learned univariate edge functions
    kan_vs_baselines.png   -- predicted vs actual scatter, three models
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

SRC = Path("docs/track_b_q1_stats_2026-07-02_final/pareto_points.csv")
OUT = Path("outputs/experiments/kan_scres_demo_2026-07-02")


def load_static_frontier() -> tuple[np.ndarray, np.ndarray, list[str]]:
    rows = []
    with SRC.open() as fh:
        for row in csv.DictReader(fh):
            if row["kind"] != "dense_static":
                continue
            policy = row["policy"]
            shift = int(policy[1])  # "S2_op10_..." -> 2
            op10 = float(policy.split("op10_")[1].split("_op12_")[0])
            op12 = float(policy.split("op12_")[1])
            rows.append((shift, op10, op12, float(row["order_ret_excel"])))
    rows.sort()
    X = np.array([[r[0], r[1], r[2]] for r in rows], dtype=np.float64)
    y = np.array([r[3] for r in rows], dtype=np.float64)
    names = ["shift S", "Op10 dispatch mult.", "Op12 dispatch mult."]
    return X, y, names


def normalize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return (X - mu) / sd, mu, sd


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _make_mlp(hidden: tuple[int, ...], seed: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    layers: list[torch.nn.Module] = []
    d_in = 3
    for h in hidden:
        layers += [torch.nn.Linear(d_in, h), torch.nn.Tanh()]
        d_in = h
    layers += [torch.nn.Linear(d_in, 1)]
    return torch.nn.Sequential(*layers).float()


def _train_with_early_stopping(
    model: torch.nn.Module,
    Xtr: torch.Tensor,
    ytr: torch.Tensor,
    Xval: torch.Tensor | None,
    yval: torch.Tensor | None,
    lr: float,
    weight_decay: float,
    max_epochs: int = 3000,
    patience: int = 250,
) -> tuple[float, int]:
    """Returns (best_val_mse, best_epoch). If Xval is None, trains for
    max_epochs and returns (train_mse_at_end, max_epochs)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val = float("inf")
    best_epoch = 0
    best_state = None
    bad_epochs = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(Xtr), ytr)
        loss.backward()
        opt.step()
        if Xval is None:
            continue
        model.eval()
        with torch.no_grad():
            val_mse = torch.nn.functional.mse_loss(model(Xval), yval).item()
        if val_mse < best_val - 1e-9:
            best_val = val_mse
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs > patience:
                break
    if Xval is None:
        return loss.item(), max_epochs
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val, best_epoch


def tune_and_fit_mlp(
    Xn: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    y_test_real: np.ndarray,
    seed: int = 0,
) -> tuple[float, float, np.ndarray, dict, list[dict]]:
    # inner train/val split carved out of TRAIN only (test stays untouched)
    rng = np.random.default_rng(seed + 1)
    perm = rng.permutation(len(train_idx))
    n_val = max(1, int(round(0.2 * len(train_idx))))
    val_pos, inner_pos = perm[:n_val], perm[n_val:]
    inner_idx = train_idx[inner_pos]
    val_idx = train_idx[val_pos]

    Xinner = torch.tensor(Xn[inner_idx], dtype=torch.float32)
    yinner = torch.tensor(y[inner_idx], dtype=torch.float32).unsqueeze(1)
    Xval = torch.tensor(Xn[val_idx], dtype=torch.float32)
    yval = torch.tensor(y[val_idx], dtype=torch.float32).unsqueeze(1)

    grid = [
        {"hidden": h, "lr": lr, "weight_decay": wd}
        for h in [(16,), (32,), (16, 16), (32, 32), (64, 64), (32, 16)]
        for lr in [0.003, 0.01, 0.03]
        for wd in [0.0, 1e-4, 1e-3]
    ]

    log = []
    best_cfg, best_val_mse, best_epoch = None, float("inf"), 0
    for cfg in grid:
        model = _make_mlp(cfg["hidden"], seed=seed)
        val_mse, epoch = _train_with_early_stopping(
            model, Xinner, yinner, Xval, yval,
            lr=cfg["lr"], weight_decay=cfg["weight_decay"],
        )
        log.append({**cfg, "val_mse": val_mse, "best_epoch": epoch})
        if val_mse < best_val_mse:
            best_val_mse, best_cfg, best_epoch = val_mse, cfg, epoch

    log.sort(key=lambda r: r["val_mse"])
    print(f"MLP grid search: {len(grid)} configs; best = {best_cfg}, "
          f"val_mse={best_val_mse:.3e}, best_epoch={best_epoch}")

    # retrain winning config on the FULL training set (118 pts) for
    # best_epoch epochs (no further peeking at val or test), then score once
    # on the held-out test set.
    final_model = _make_mlp(best_cfg["hidden"], seed=seed)
    Xtr_full = torch.tensor(Xn[train_idx], dtype=torch.float32)
    ytr_full = torch.tensor(y[train_idx], dtype=torch.float32).unsqueeze(1)
    opt = torch.optim.Adam(
        final_model.parameters(), lr=best_cfg["lr"], weight_decay=best_cfg["weight_decay"]
    )
    for _ in range(max(1, best_epoch)):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(final_model(Xtr_full), ytr_full)
        loss.backward()
        opt.step()

    final_model.eval()
    with torch.no_grad():
        pred = final_model(torch.tensor(Xn[test_idx], dtype=torch.float32)).squeeze(-1).numpy()
    r2 = r2_score(y_test_real, pred)
    mse = float(np.mean((y_test_real - pred) ** 2))
    cfg_out = {**best_cfg, "best_epoch": best_epoch, "val_mse_during_tuning": best_val_mse}
    return r2, mse, pred, cfg_out, log[:5]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)

    X, y, names = load_static_frontier()
    n = len(y)
    Xn, mu, sd = normalize(X)
    y_mu, y_sd = y.mean(), y.std()
    yn = (y - y_mu) / y_sd

    # 80/20 split, fixed seed, small-N stratification not needed (dense grid)
    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    n_test = max(1, int(round(0.2 * n)))
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    Xtr = torch.tensor(Xn[train_idx], dtype=torch.float32)
    ytr = torch.tensor(yn[train_idx], dtype=torch.float32).unsqueeze(1)
    Xte = torch.tensor(Xn[test_idx], dtype=torch.float32)
    yte_real = y[test_idx]

    print(f"n={n} (train={len(train_idx)}, test={len(test_idx)}); inputs={names}")

    # ---------------------------------------------------------------- KAN
    from kan import KAN

    kan_model = KAN(width=[3, 4, 1], grid=5, k=3, seed=0, device="cpu",
                    ckpt_path=str(OUT / "pykan_checkpoints"))
    dataset = {
        "train_input": Xtr,
        "train_label": ytr,
        "test_input": Xte,
        "test_label": torch.tensor(
            (yte_real - y_mu) / y_sd, dtype=torch.float32
        ).unsqueeze(1),
    }
    kan_model.fit(dataset, opt="LBFGS", steps=60, lr=0.01, update_grid=True)
    with torch.no_grad():
        kan_pred_n = kan_model(Xte).squeeze(-1).numpy()
    kan_pred = kan_pred_n * y_sd + y_mu
    kan_r2 = r2_score(yte_real, kan_pred)
    kan_mse = float(np.mean((yte_real - kan_pred) ** 2))

    # spline plot -- the interpretable signature output
    kan_model.plot(beta=100)
    plt.savefig(OUT / "kan_splines.png", dpi=200, bbox_inches="tight")
    plt.close("all")

    # ---------------------------------------------------------------- linear baseline
    Xtr_np, ytr_np = Xn[train_idx], y[train_idx]
    A = np.hstack([Xtr_np, np.ones((len(train_idx), 1))])
    coef, *_ = np.linalg.lstsq(A, ytr_np, rcond=None)
    lin_pred = np.hstack([Xn[test_idx], np.ones((len(test_idx), 1))]) @ coef
    lin_r2 = r2_score(yte_real, lin_pred)
    lin_mse = float(np.mean((yte_real - lin_pred) ** 2))

    # ---------------------------------------------------------------- tuned MLP baseline
    # Honest protocol: carve a validation split out of TRAIN only (never touch
    # test during tuning), grid-search architecture/lr/weight-decay with
    # early stopping on validation MSE, then retrain the winning config on
    # the full training set for the winning epoch count and evaluate once on
    # the held-out test set.
    mlp_r2, mlp_mse, mlp_pred, mlp_cfg, mlp_grid_log = tune_and_fit_mlp(
        Xn, y, train_idx, test_idx, y_test_real=yte_real, seed=0
    )

    # ---------------------------------------------------------------- comparison figure
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    lo, hi = min(y.min(), kan_pred.min()), max(y.max(), kan_pred.max())
    ax.plot([lo, hi], [lo, hi], color="0.6", lw=1.0, ls="--", zorder=0)
    ax.scatter(yte_real, kan_pred, s=40, label=f"KAN (R$^2$={kan_r2:.3f})", zorder=3)
    ax.scatter(yte_real, mlp_pred, s=30, marker="^", label=f"MLP (R$^2$={mlp_r2:.3f})", zorder=2)
    ax.scatter(yte_real, lin_pred, s=30, marker="s", label=f"Linear (R$^2$={lin_r2:.3f})", zorder=1)
    ax.set_xlabel("Actual Excel ReT (held-out static configs)")
    ax.set_ylabel("Predicted Excel ReT")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "kan_vs_baselines.png", dpi=200)
    plt.close(fig)

    summary = {
        "task": "decision-variable -> Excel ReT surrogate (Garrido Fig. 5 architecture)",
        "inputs": names,
        "n_total": n,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "kan": {"width": [3, 4, 1], "grid": 5, "k": 3, "r2_test": kan_r2, "mse_test": kan_mse},
        "mlp_tuned": {
            "selection": "grid search (18 archs x 3 lr x 3 weight-decay = 54 configs), "
                         "early-stopped on a val split carved out of TRAIN only; "
                         "winning config retrained on full train, scored once on test",
            "winning_config": mlp_cfg,
            "r2_test": mlp_r2,
            "mse_test": mlp_mse,
            "top5_configs_by_val_mse": mlp_grid_log,
        },
        "linear_baseline": {"r2_test": lin_r2, "mse_test": lin_mse},
    }
    (OUT / "kan_fit_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
