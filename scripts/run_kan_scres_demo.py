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

    # ---------------------------------------------------------------- small MLP baseline
    mlp = torch.nn.Sequential(
        torch.nn.Linear(3, 16), torch.nn.Tanh(),
        torch.nn.Linear(16, 16), torch.nn.Tanh(),
        torch.nn.Linear(16, 1),
    ).float()
    opt = torch.optim.Adam(mlp.parameters(), lr=0.02)
    Xtr_t = torch.tensor(Xn[train_idx], dtype=torch.float32)
    ytr_t = torch.tensor(y[train_idx], dtype=torch.float32).unsqueeze(1)
    for _ in range(400):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(mlp(Xtr_t), ytr_t)
        loss.backward()
        opt.step()
    with torch.no_grad():
        mlp_pred = mlp(torch.tensor(Xn[test_idx], dtype=torch.float32)).squeeze(-1).numpy()
    mlp_r2 = r2_score(yte_real, mlp_pred)
    mlp_mse = float(np.mean((yte_real - mlp_pred) ** 2))

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
        "mlp_baseline": {"hidden": [16, 16], "r2_test": mlp_r2, "mse_test": mlp_mse},
        "linear_baseline": {"r2_test": lin_r2, "mse_test": lin_mse},
    }
    (OUT / "kan_fit_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
