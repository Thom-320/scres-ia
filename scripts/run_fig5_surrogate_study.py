#!/usr/bin/env python3
"""Fig-5 surrogate study — frozen endpoints (Track B, prereg 2026-07-23).

Implements docs/FIG5_SURROGATE_PREREGISTRATION_2026-07-23.md on the exact
frontier dataset produced by scripts/dump_fig5_dataset.py (48 burned
campaigns x 65,536 weekly-count calendars; label early_ret_complete_cohort).

Families:
  * MLP  — [64, 64] ReLU, Adam lr 1e-3, MSE, early stop on a val split
           carved from TRAIN (scaffold: scripts/run_bc_de_sweetspot.py:89-120)
  * KAN  — pykan, width [d_in, 10, 8, 1], grid 5, k 3, LBFGS 60 steps
           (scaled from scripts/run_kan_scres_demo.py); train sets larger
           than KAN_MAX_TRAIN=2048 rows are subsampled (documented; the
           mission allows <=8192 — 2048 keeps ~270 KAN fits inside the wall
           clock cap)
  * Ridge — linear baseline, closed form, alpha=1.0 on standardized features
  * Sim-opt — random search / CMA-ES-style (mu/mu_w, lambda) gaussian ES on
           the continuous relaxation [0,3]^8 rounded to ints / GP-BO
           (sklearn GaussianProcessRegressor + expected improvement), each
           limited to eval budgets {64, 256, 1024, 4096} true-simulator
           calls per campaign; a "call" = one lookup into the stored exact
           frontier.

Feature representation (documented choice): 8 weekly counts as scalars
normalized by /3 (NOT one-hot) + kappa + retained_prior +
initial_regime_is_PC (0/1) -> 11 dims.  Within-campaign models see the 3
context features as constants; the transfer model (endpoint 4) uses them.

All sampling/splits derive from the fixed RNG seed 20260723 via
np.random.default_rng([SEED, *stream tags]).

Endpoints (frozen; no additions):
  1. argmax regret per (family, n), mean/median/q95 across campaigns
  2. top-k recall (k=10,100) at n=4096
  3. sample efficiency: attained regret at budget for surrogate-guided
     (MLP-guided, ridge-guided; KAN-guided omitted for wall-clock cost —
     documented deviation) vs random/CMA/BO, plus evals-needed summaries
     for regret <= {0.005, 0.01}
  4. transfer: one global MLP + one global KAN trained on 36 campaigns,
     argmax regret on 12 held-out campaigns (last 6 sorted roots x both
     kappa strata)
  5. R^2 MLP vs KAN vs linear on identical 80/20 splits at n=4096
  D2. dispatch-screen replication (3,096 rows) — 80/20 R^2 + per-profile
     43-arm argmax regret with models trained at n=30

Output: results/fig5_surrogate_v1/study_results.json
claim_status: BURNED_DEVELOPMENT_NO_CLAIM_METHODOLOGICAL

Usage:
    OMP_NUM_THREADS=1 .venv/bin/python scripts/run_fig5_surrogate_study.py
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

FRONTIER_DIR = _ROOT / "results/fig5_surrogate_v1/frontiers"
OUT_PATH = _ROOT / "results/fig5_surrogate_v1/study_results.json"
SEED = 20260723
BUDGETS = (64, 256, 1024, 4096)
N_CAL = 65536
KAN_MAX_TRAIN = 2048
KAN_STEPS = 60
EPS_THRESHOLDS = (0.005, 0.01)
TOPK = (10, 100)
CLAIM_STATUS = "BURNED_DEVELOPMENT_NO_CLAIM_METHODOLOGICAL"

import torch  # noqa: E402

torch.set_num_threads(1)


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------

def load_campaigns() -> list[dict]:
    manifest = json.loads((FRONTIER_DIR / "manifest.json").read_text())
    calendars = np.load(FRONTIER_DIR / "calendars.npz")["calendars"]
    campaigns = []
    for entry in manifest["files"]:
        meta = json.loads((FRONTIER_DIR / entry["meta"]).read_text())
        labels = np.load(FRONTIER_DIR / entry["npz"])["labels_f64"]
        campaigns.append(
            {
                "tag": entry["npz"].replace("campaign_", "").replace(".npz", ""),
                "root": meta["history_root"],
                "index": meta["campaign_index"],
                "kappa": meta["kappa"],
                "prior": meta["retained_prior"],
                "regime_pc": 1.0 if meta["initial_regime"] == "P_C" else 0.0,
                "labels": labels,
                "true_max": float(labels.max()),
            }
        )
    campaigns.sort(key=lambda c: (c["root"], c["kappa"]))
    return campaigns, calendars


def features_for(camp: dict, calendars: np.ndarray) -> np.ndarray:
    n = calendars.shape[0]
    ctx = np.array([camp["kappa"], camp["prior"], camp["regime_pc"]])
    return np.hstack(
        [calendars.astype(np.float64) / 3.0, np.tile(ctx, (n, 1))]
    )


def rng_for(*tags) -> np.random.Generator:
    return np.random.default_rng([SEED, *[int(t) for t in tags]])


# --------------------------------------------------------------------------
# model families
# --------------------------------------------------------------------------

class _Standardizer:
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.mu = X.mean(axis=0)
        self.sd = np.where(X.std(axis=0) < 1e-9, 1.0, X.std(axis=0))
        self.ymu = float(y.mean())
        self.ysd = float(y.std()) or 1.0

    def x(self, X):
        return (X - self.mu) / self.sd

    def y(self, y):
        return (y - self.ymu) / self.ysd

    def y_inv(self, yn):
        return yn * self.ysd + self.ymu


def fit_predict_ridge(
    Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, alpha: float = 1.0
) -> np.ndarray:
    st = _Standardizer(Xtr, ytr)
    A = np.hstack([st.x(Xtr), np.ones((len(Xtr), 1))])
    reg = alpha * np.eye(A.shape[1])
    reg[-1, -1] = 0.0
    coef = np.linalg.solve(A.T @ A + reg, A.T @ st.y(ytr))
    B = np.hstack([st.x(Xte), np.ones((len(Xte), 1))])
    return st.y_inv(B @ coef)


def fit_mlp(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    torch_seed: int,
    max_epochs: int = 600,
    patience: int = 60,
):
    """[64,64] ReLU + Adam(1e-3) + MSE + early stop on a 20% val split of
    TRAIN (scaffold: scripts/run_bc_de_sweetspot.py:89-120; linear head, no
    Tanh, because the target is a regression score, not a bounded action).
    Returns (model, standardizer)."""
    import torch.nn as nn

    st = _Standardizer(Xtr, ytr)
    rng = np.random.default_rng([SEED, torch_seed & 0x7FFFFFFF, 77])
    n = len(Xtr)
    idx = rng.permutation(n)
    n_val = max(4, int(round(0.2 * n)))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    Xt = torch.tensor(st.x(Xtr[tr_idx]), dtype=torch.float32)
    yt = torch.tensor(st.y(ytr[tr_idx]), dtype=torch.float32).unsqueeze(1)
    Xv = torch.tensor(st.x(Xtr[val_idx]), dtype=torch.float32)
    yv = torch.tensor(st.y(ytr[val_idx]), dtype=torch.float32).unsqueeze(1)

    torch.manual_seed(torch_seed & 0x7FFFFFFF)
    model = nn.Sequential(
        nn.Linear(Xtr.shape[1], 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, 1),
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    best = float("inf")
    best_state = None
    bad = 0
    for _epoch in range(max_epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(Xt), yt)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(Xv), yv).item()
        if vl < best - 1e-9:
            best = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad > patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, st


def predict_mlp(model, st: _Standardizer, Xte: np.ndarray) -> np.ndarray:
    preds = []
    with torch.no_grad():
        for start in range(0, len(Xte), 16384):
            chunk = torch.tensor(
                st.x(Xte[start:start + 16384]), dtype=torch.float32
            )
            preds.append(model(chunk).squeeze(-1).numpy())
    return st.y_inv(np.concatenate(preds).astype(np.float64))


def fit_predict_mlp(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    torch_seed: int,
    max_epochs: int = 600,
    patience: int = 60,
) -> np.ndarray:
    model, st = fit_mlp(Xtr, ytr, torch_seed, max_epochs, patience)
    return predict_mlp(model, st, Xte)


_KAN_CKPT = Path(
    "/private/tmp/claude-501/-Users-thom-Projects-research-scres-ia/"
    "79a20ac6-ef07-49db-ab45-48bab5e28d63/scratchpad/kan_ckpt_study"
)


def fit_predict_kan(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    torch_seed: int,
    steps: int = KAN_STEPS,
) -> tuple[np.ndarray, int]:
    """pykan KAN width [d_in,10,8,1], LBFGS `steps`, lr 0.01 (scaled from
    scripts/run_kan_scres_demo.py).  Returns (pred, n_train_used).  Train
    sets larger than KAN_MAX_TRAIN are subsampled deterministically."""
    from kan import KAN

    n_used = len(Xtr)
    if len(Xtr) > KAN_MAX_TRAIN:
        sub = np.random.default_rng(
            [SEED, torch_seed & 0x7FFFFFFF, 88]
        ).choice(len(Xtr), KAN_MAX_TRAIN, replace=False)
        Xtr, ytr = Xtr[sub], ytr[sub]
        n_used = KAN_MAX_TRAIN
    st = _Standardizer(Xtr, ytr)
    Xt = torch.tensor(st.x(Xtr), dtype=torch.float32)
    yt = torch.tensor(st.y(ytr), dtype=torch.float32).unsqueeze(1)
    d_in = Xtr.shape[1]
    torch.manual_seed(torch_seed & 0x7FFFFFFF)
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        model = KAN(
            width=[d_in, 10, 8, 1], grid=5, k=3,
            seed=torch_seed & 0x7FFFFFFF, device="cpu",
            ckpt_path=str(_KAN_CKPT), auto_save=False,
        )
        dataset = {
            "train_input": Xt,
            "train_label": yt,
            "test_input": Xt[: min(64, len(Xt))],
            "test_label": yt[: min(64, len(yt))],
        }
        model.fit(dataset, opt="LBFGS", steps=steps, lr=0.01, update_grid=True)
    preds = []
    with torch.no_grad():
        for start in range(0, len(Xte), 16384):
            chunk = torch.tensor(
                st.x(Xte[start:start + 16384]), dtype=torch.float32
            )
            preds.append(model(chunk).squeeze(-1).numpy())
    return st.y_inv(np.concatenate(preds).astype(np.float64)), n_used


# --------------------------------------------------------------------------
# sim-opt baselines (oracle = stored frontier lookup; a call = one lookup)
# --------------------------------------------------------------------------

def _cal_to_index(cals: np.ndarray) -> np.ndarray:
    idx = np.zeros(len(cals), dtype=np.int64)
    for w in range(8):
        idx = idx * 4 + cals[:, w].astype(np.int64)
    return idx


def random_search(labels: np.ndarray, budget: int, rng) -> float:
    idx = rng.choice(N_CAL, size=budget, replace=False)
    return float(labels[idx].max())


def cma_es_search(labels: np.ndarray, budget: int, rng) -> float:
    """Minimal (mu/mu_w, lambda) gaussian ES on the continuous relaxation
    [0,3]^8, rounded to ints; every rounded-candidate evaluation counts as
    one oracle call (duplicates included, budget-honest)."""
    lam, mu = 16, 8
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= weights.sum()
    mean = rng.uniform(0.0, 3.0, size=8)
    sigma = 0.9
    best = -np.inf
    used = 0
    while used < budget:
        k = min(lam, budget - used)
        pop = mean[None, :] + sigma * rng.standard_normal((k, 8))
        pop = np.clip(pop, 0.0, 3.0)
        cand = np.rint(pop).astype(np.uint8)
        vals = labels[_cal_to_index(cand)]
        used += k
        best = max(best, float(vals.max()))
        order = np.argsort(-vals)[: min(mu, k)]
        w = weights[: len(order)] / weights[: len(order)].sum()
        mean = (w[:, None] * pop[order]).sum(axis=0)
        sigma = max(sigma * 0.93, 0.12)
    return best


def gp_bo_search(labels: np.ndarray, calendars: np.ndarray, budget: int,
                 rng) -> float:
    """GP-BO: sklearn GaussianProcessRegressor (Matern 2.5 + white noise),
    expected improvement over 2048 random unevaluated candidates per batch;
    batch size q = max(1, budget//32); GP fit data subsampled to the 512
    best-plus-recent points when larger (wall-clock necessity, documented)."""
    import warnings

    from scipy.stats import norm
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

    warnings.simplefilter("ignore", ConvergenceWarning)

    X8 = calendars.astype(np.float64) / 3.0
    n_init = min(32, budget)
    evaluated = list(rng.choice(N_CAL, size=n_init, replace=False))
    seen = set(evaluated)
    used = n_init
    q = max(1, budget // 32)
    while used < budget:
        idx_arr = np.asarray(evaluated, dtype=np.int64)
        yv = labels[idx_arr]
        if len(idx_arr) > 512:
            best_part = np.argsort(-yv)[:256]
            recent = np.arange(len(idx_arr))[-256:]
            keep = np.unique(np.concatenate([best_part, recent]))
            fit_idx, fit_y = idx_arr[keep], yv[keep]
        else:
            fit_idx, fit_y = idx_arr, yv
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(length_scale=np.ones(8), nu=2.5,
                     length_scale_bounds=(1e-2, 1e2))
            + WhiteKernel(1e-6, (1e-9, 1e-1))
        )
        gp = GaussianProcessRegressor(
            kernel=kernel, normalize_y=True, n_restarts_optimizer=0,
            random_state=int(rng.integers(2**31)),
        )
        gp.fit(X8[fit_idx], fit_y)
        cand = rng.choice(N_CAL, size=2048, replace=False)
        cand = np.asarray([c for c in cand if c not in seen], dtype=np.int64)
        mu_c, sd_c = gp.predict(X8[cand], return_std=True)
        sd_c = np.maximum(sd_c, 1e-9)
        y_best = float(yv.max())
        z = (mu_c - y_best) / sd_c
        ei = (mu_c - y_best) * norm.cdf(z) + sd_c * norm.pdf(z)
        k = min(q, budget - used)
        pick = cand[np.argsort(-ei)[:k]]
        evaluated.extend(int(p) for p in pick)
        seen.update(int(p) for p in pick)
        used += k
    return float(labels[np.asarray(evaluated, dtype=np.int64)].max())


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def summarize(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "q95": float(np.quantile(arr, 0.95)),
        "n": int(arr.size),
    }


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def save_partial(results: dict) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2, sort_keys=True))


# --------------------------------------------------------------------------
# endpoints
# --------------------------------------------------------------------------

def run_endpoints_1_2_5(campaigns, calendars, results, timings):
    """Endpoint 1 (argmax regret), 2 (top-k recall at n=4096), 5 (R^2 on
    identical 80/20 splits at n=4096) in one pass over campaigns."""
    regret = {f: {n: [] for n in BUDGETS} for f in ("ridge", "mlp", "kan")}
    recall = {f: {k: [] for k in TOPK} for f in ("ridge", "mlp", "kan")}
    r2 = {f: [] for f in ("ridge", "mlp", "kan")}
    kan_train_sizes = set()
    t0 = time.time()
    for ci, camp in enumerate(campaigns):
        X_all = features_for(camp, calendars)
        labels = camp["labels"]
        true_max = camp["true_max"]
        true_order = np.argsort(-labels)
        for n in BUDGETS:
            rng = rng_for(camp["root"], camp["index"], int(camp["kappa"] * 100), n, 1)
            tr = rng.choice(N_CAL, size=n, replace=False)
            Xtr, ytr = X_all[tr], labels[tr]
            tseed = int(rng.integers(2**31))
            preds = {
                "ridge": fit_predict_ridge(Xtr, ytr, X_all),
                "mlp": fit_predict_mlp(Xtr, ytr, X_all, tseed),
            }
            kan_pred, kan_n = fit_predict_kan(Xtr, ytr, X_all, tseed)
            preds["kan"] = kan_pred
            kan_train_sizes.add((n, kan_n))
            for fam, p in preds.items():
                regret[fam][n].append(true_max - float(labels[int(p.argmax())]))
                if n == 4096:
                    pred_order = np.argsort(-p)
                    for k in TOPK:
                        inter = len(
                            set(true_order[:k].tolist())
                            & set(pred_order[:k].tolist())
                        )
                        recall[fam][k].append(inter / k)
            # endpoint 5 shares the n=4096 sample: 80/20 split, refit, R^2
            if n == 4096:
                rng5 = rng_for(camp["root"], camp["index"],
                               int(camp["kappa"] * 100), 5)
                perm = rng5.permutation(n)
                n_te = int(round(0.2 * n))
                te_i, tr_i = perm[:n_te], perm[n_te:]
                Xa, ya = Xtr[tr_i], ytr[tr_i]
                Xb, yb = Xtr[te_i], ytr[te_i]
                t5 = int(rng5.integers(2**31))
                r2["ridge"].append(
                    r2_score(yb, fit_predict_ridge(Xa, ya, Xb)))
                r2["mlp"].append(
                    r2_score(yb, fit_predict_mlp(Xa, ya, Xb, t5)))
                kp, _ = fit_predict_kan(Xa, ya, Xb, t5)
                r2["kan"].append(r2_score(yb, kp))
        print(f"[e125] campaign {ci + 1}/48 done ({time.time() - t0:.0f}s)",
              flush=True)

    results["endpoint_1_argmax_regret"] = {
        fam: {str(n): summarize(v) for n, v in per_n.items()}
        for fam, per_n in regret.items()
    }
    results["endpoint_1_raw"] = {
        fam: {str(n): v for n, v in per_n.items()}
        for fam, per_n in regret.items()
    }
    results["endpoint_2_topk_recall_n4096"] = {
        fam: {
            str(k): {
                "mean": float(np.mean(v)), "median": float(np.median(v)),
            }
            for k, v in per_k.items()
        }
        for fam, per_k in recall.items()
    }
    results["endpoint_5_r2_n4096_80_20"] = {
        fam: {"mean": float(np.mean(v)), "median": float(np.median(v)),
              "min": float(np.min(v))}
        for fam, v in r2.items()
    }
    results["endpoint_5_raw"] = r2
    results["kan_train_sizes_used"] = sorted(
        [list(t) for t in kan_train_sizes])
    timings["endpoints_1_2_5_seconds"] = round(time.time() - t0, 1)


def guided_search(labels, X_all, budget, rng, family: str) -> float:
    """Surrogate-guided iterative: fit on budget/2 random, propose top
    budget/4 unevaluated by prediction, verify (oracle calls), refit ONCE on
    the union, verify the top remaining budget/4.  Total calls = budget."""
    n1 = budget // 2
    n2 = budget // 4
    n3 = budget - n1 - n2
    evaluated = list(rng.choice(N_CAL, size=n1, replace=False))
    seen = set(evaluated)

    def fit_pred(idx_list):
        idx_arr = np.asarray(idx_list, dtype=np.int64)
        Xtr, ytr = X_all[idx_arr], labels[idx_arr]
        if family == "mlp":
            return fit_predict_mlp(Xtr, ytr, X_all, int(rng.integers(2**31)))
        return fit_predict_ridge(Xtr, ytr, X_all)

    # proposal 1: fit on the n1 random points, verify top n2 predictions
    pred = fit_pred(evaluated)
    picks = [int(i) for i in np.argsort(-pred) if int(i) not in seen][:n2]
    evaluated.extend(picks)
    seen.update(picks)
    # refit ONCE on the union, verify top n3 predictions
    pred = fit_pred(evaluated)
    picks = [int(i) for i in np.argsort(-pred) if int(i) not in seen][:n3]
    evaluated.extend(picks)
    seen.update(picks)
    assert len(evaluated) == budget
    return float(labels[np.asarray(evaluated, dtype=np.int64)].max())


def run_endpoint_3(campaigns, calendars, results, timings):
    methods = ("mlp_guided", "ridge_guided", "random", "cma_es", "gp_bo")
    attained = {m: {n: [] for n in BUDGETS} for m in methods}
    t0 = time.time()
    for ci, camp in enumerate(campaigns):
        X_all = features_for(camp, calendars)
        labels = camp["labels"]
        true_max = camp["true_max"]
        for n in BUDGETS:
            base = [camp["root"], camp["index"], int(camp["kappa"] * 100), n]
            attained["mlp_guided"][n].append(
                true_max - guided_search(labels, X_all, n,
                                         rng_for(*base, 31), "mlp"))
            attained["ridge_guided"][n].append(
                true_max - guided_search(labels, X_all, n,
                                         rng_for(*base, 32), "ridge"))
            attained["random"][n].append(
                true_max - random_search(labels, n, rng_for(*base, 33)))
            attained["cma_es"][n].append(
                true_max - cma_es_search(labels, n, rng_for(*base, 34)))
            attained["gp_bo"][n].append(
                true_max - gp_bo_search(labels, calendars, n,
                                        rng_for(*base, 35)))
        print(f"[e3] campaign {ci + 1}/48 done ({time.time() - t0:.0f}s)",
              flush=True)

    ep3 = {"attained_regret_at_budget": {}, "evals_needed": {}}
    for m in methods:
        ep3["attained_regret_at_budget"][m] = {
            str(n): summarize(v) for n, v in attained[m].items()
        }
        per_eps = {}
        for eps in EPS_THRESHOLDS:
            frac = {
                str(n): float(np.mean(np.asarray(attained[m][n]) <= eps))
                for n in BUDGETS
            }
            min_budgets = []
            for i in range(len(campaigns)):
                hit = [n for n in BUDGETS if attained[m][n][i] <= eps]
                min_budgets.append(min(hit) if hit else None)
            reached = [b for b in min_budgets if b is not None]
            per_eps[str(eps)] = {
                "fraction_of_campaigns_at_budget": frac,
                "median_min_budget_when_reached": (
                    float(np.median(reached)) if reached else None
                ),
                "n_campaigns_reached_at_any_budget": len(reached),
            }
        ep3["evals_needed"][m] = per_eps
    ep3["raw_attained_regret"] = {
        m: {str(n): v for n, v in attained[m].items()} for m in methods
    }
    results["endpoint_3_sample_efficiency"] = ep3
    timings["endpoint_3_seconds"] = round(time.time() - t0, 1)


def run_endpoint_4(campaigns, calendars, results, timings):
    t0 = time.time()
    roots = sorted({c["root"] for c in campaigns})
    held_roots = set(roots[-6:])
    train_c = [c for c in campaigns if c["root"] not in held_roots]
    test_c = [c for c in campaigns if c["root"] in held_roots]
    assert len(train_c) == 36 and len(test_c) == 12

    Xtr_parts, ytr_parts = [], []
    for camp in train_c:
        rng = rng_for(camp["root"], camp["index"],
                      int(camp["kappa"] * 100), 4096, 1)
        tr = rng.choice(N_CAL, size=4096, replace=False)
        Xtr_parts.append(features_for(camp, calendars)[tr])
        ytr_parts.append(camp["labels"][tr])
    Xtr = np.vstack(Xtr_parts)
    ytr = np.concatenate(ytr_parts)

    trng = rng_for(4)
    tseed = int(trng.integers(2**31))
    ep4 = {"held_out_roots": sorted(held_roots),
           "n_train_rows": int(len(Xtr))}
    for fam in ("mlp", "kan"):
        regrets = {}
        per_camp = []
        # fit once, predict per campaign
        if fam == "mlp":
            model, st = fit_mlp(Xtr, ytr, tseed, max_epochs=800, patience=80)
            for camp in test_c:
                pred = predict_mlp(model, st, features_for(camp, calendars))
                r = camp["true_max"] - float(camp["labels"][int(pred.argmax())])
                regrets[camp["tag"]] = r
                per_camp.append(r)
        else:
            # one global KAN, train subsampled to 8192 rows (mission cap;
            # documented)
            sub = trng.choice(len(Xtr), 8192, replace=False)
            Xk, yk = Xtr[sub], ytr[sub]
            st = _Standardizer(Xk, yk)
            from kan import KAN
            torch.manual_seed(tseed & 0x7FFFFFFF)
            sink = io.StringIO()
            with contextlib.redirect_stdout(sink), \
                    contextlib.redirect_stderr(sink):
                model = KAN(width=[Xk.shape[1], 10, 8, 1], grid=5, k=3,
                            seed=tseed & 0x7FFFFFFF, device="cpu",
                            ckpt_path=str(_KAN_CKPT), auto_save=False)
                dataset = {
                    "train_input": torch.tensor(st.x(Xk), dtype=torch.float32),
                    "train_label": torch.tensor(
                        st.y(yk), dtype=torch.float32).unsqueeze(1),
                    "test_input": torch.tensor(
                        st.x(Xk[:64]), dtype=torch.float32),
                    "test_label": torch.tensor(
                        st.y(yk[:64]), dtype=torch.float32).unsqueeze(1),
                }
                model.fit(dataset, opt="LBFGS", steps=KAN_STEPS, lr=0.01,
                          update_grid=True)
            for camp in test_c:
                Xte = features_for(camp, calendars)
                preds = []
                with torch.no_grad():
                    for s in range(0, len(Xte), 16384):
                        chunk = torch.tensor(
                            st.x(Xte[s:s + 16384]), dtype=torch.float32)
                        preds.append(model(chunk).squeeze(-1).numpy())
                pred = st.y_inv(np.concatenate(preds).astype(np.float64))
                r = camp["true_max"] - float(camp["labels"][int(pred.argmax())])
                regrets[camp["tag"]] = r
                per_camp.append(r)
        ep4[fam] = {"summary": summarize(per_camp), "per_campaign": regrets}
        if fam == "kan":
            ep4[fam]["train_rows_used"] = 8192
    results["endpoint_4_transfer"] = ep4
    timings["endpoint_4_seconds"] = round(time.time() - t0, 1)


# --------------------------------------------------------------------------
# D2 dispatch-screen replication
# --------------------------------------------------------------------------

def run_d2(results, timings):
    t0 = time.time()
    data = json.loads(
        (_ROOT / "results/thesis_native_dispatch_screen_v1/result.json")
        .read_text()
    )
    rows = data["rows"]
    modes = sorted({r["mode"] for r in rows})
    profiles = sorted({r["profile"] for r in rows})

    def feats(row, with_profile: bool) -> list[float]:
        f = [float(row["calm_mult"]), float(row["active_mult"])]
        f += [1.0 if row["mode"] == m else 0.0 for m in modes]
        if with_profile:
            f += [1.0 if row["profile"] == p else 0.0 for p in profiles]
        return f

    X_all = np.asarray([feats(r, True) for r in rows])
    y_all = np.asarray([float(r["ret_excel"]) for r in rows])

    # global 80/20 R^2
    rng = rng_for(2001)
    perm = rng.permutation(len(rows))
    n_te = int(round(0.2 * len(rows)))
    te, tr = perm[:n_te], perm[n_te:]
    tseed = int(rng.integers(2**31))
    d2 = {"n_rows": len(rows), "label": "ret_excel",
          "features_global": "calm_mult, active_mult, mode one-hot (3), "
                             "profile one-hot (6); seed excluded",
          "global_r2_80_20": {}}
    d2["global_r2_80_20"]["ridge"] = r2_score(
        y_all[te], fit_predict_ridge(X_all[tr], y_all[tr], X_all[te]))
    d2["global_r2_80_20"]["mlp"] = r2_score(
        y_all[te], fit_predict_mlp(X_all[tr], y_all[tr], X_all[te], tseed))
    kp, kn = fit_predict_kan(X_all[tr], y_all[tr], X_all[te], tseed)
    d2["global_r2_80_20"]["kan"] = r2_score(y_all[te], kp)
    d2["global_r2_80_20"]["kan_train_rows_used"] = kn

    # per-profile 43-arm argmax regret, models trained at n=30 (5 replicates)
    n_train = 30
    replicates = 5
    arm_regret = {f: [] for f in ("ridge", "mlp", "kan")}
    for pi, prof in enumerate(profiles):
        prows = [r for r in rows if r["profile"] == prof]
        arms = sorted({(r["calm_mult"], r["active_mult"], r["mode"])
                       for r in prows})
        assert len(arms) == 43
        arm_mean = {}
        for a in arms:
            vals = [float(r["ret_excel"]) for r in prows
                    if (r["calm_mult"], r["active_mult"], r["mode"]) == a]
            arm_mean[a] = float(np.mean(vals))
        best_arm_val = max(arm_mean.values())
        Xp = np.asarray([feats(r, False) for r in prows])
        yp = np.asarray([float(r["ret_excel"]) for r in prows])
        X_arms = np.asarray([
            [a[0], a[1]] + [1.0 if a[2] == m else 0.0 for m in modes]
            for a in arms
        ])
        for rep in range(replicates):
            rrng = rng_for(2002, pi, rep)
            tr_i = rrng.choice(len(prows), size=n_train, replace=False)
            ts = int(rrng.integers(2**31))
            preds = {
                "ridge": fit_predict_ridge(Xp[tr_i], yp[tr_i], X_arms),
                "mlp": fit_predict_mlp(Xp[tr_i], yp[tr_i], X_arms, ts),
                "kan": fit_predict_kan(Xp[tr_i], yp[tr_i], X_arms, ts)[0],
            }
            for fam, p in preds.items():
                chosen = arms[int(np.argmax(p))]
                arm_regret[fam].append(best_arm_val - arm_mean[chosen])
    d2["per_profile_argmax_regret_n30"] = {
        fam: summarize(v) for fam, v in arm_regret.items()
    }
    d2["per_profile_argmax_regret_raw"] = arm_regret
    d2["argmax_protocol"] = (
        f"per profile: {replicates} replicates x n={n_train} sampled rows "
        "(seeded), predict all 43 arms, regret vs the 12-seed arm-mean of "
        "the best arm"
    )
    results["d2_dispatch_replication"] = d2
    timings["d2_seconds"] = round(time.time() - t0, 1)


# --------------------------------------------------------------------------

def main() -> None:
    t_start = time.time()
    campaigns, calendars = load_campaigns()
    assert len(campaigns) == 48

    timings: dict = {}
    results: dict = {
        "claim_status": CLAIM_STATUS,
        "preregistration": "docs/FIG5_SURROGATE_PREREGISTRATION_2026-07-23.md",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "seed": SEED,
        "budgets": list(BUDGETS),
        "config": {
            "label": "early_ret_complete_cohort",
            "features": "8 weekly counts (scalar, /3) + kappa + "
                        "retained_prior + initial_regime_is_PC -> 11 dims "
                        "(scalar counts chosen over one-hot; documented)",
            "mlp": "[64,64] ReLU, Adam lr 1e-3, MSE, full batch, early stop "
                   "patience 60 on 20% val split of train, max 600 epochs",
            "kan": f"pykan width [d_in,10,8,1], grid 5, k 3, LBFGS "
                   f"{KAN_STEPS} steps, lr 0.01, train subsampled to "
                   f"<= {KAN_MAX_TRAIN} rows (documented deviation for wall "
                   f"clock; mission allows <= 8192)",
            "ridge": "alpha=1.0 on standardized features, unpenalized "
                     "intercept",
            "random": "n distinct uniform lookups",
            "cma_es": "(mu/mu_w, lambda)-ES, lambda=16, mu=8, log-rank "
                      "weights, sigma 0.9 * 0.93^gen floored at 0.12, "
                      "continuous [0,3]^8 rounded; duplicates count "
                      "against budget",
            "gp_bo": "sklearn GP (Matern 2.5 + white), EI over 2048 random "
                     "unevaluated candidates, init 32, batch q=budget//32, "
                     "GP fit data capped at 512 points (256 best + 256 "
                     "recent)",
            "guided": "fit on budget/2 random, verify top budget/4 "
                      "predicted, refit once, verify top budget/4 again; "
                      "KAN-guided omitted (wall-clock; documented deviation)",
        },
        "deviations_from_preregistration": [
            f"KAN train sets subsampled to <= {KAN_MAX_TRAIN} rows (mission "
            "allowed <= 8192; chosen smaller to fit the 2-3h wall-clock "
            "cap across ~270 KAN fits); global transfer KAN uses 8192.",
            "Endpoint 3 surrogate-guided uses MLP-guided and ridge-guided "
            "only; KAN-guided omitted for wall-clock cost.",
            "GP-BO fits on at most 512 of the evaluated points and "
            "proposes in batches (q=budget//32) — exact GP at n=4096 is "
            "computationally infeasible; budgets remain matched (every "
            "lookup counted).",
        ],
    }

    # Descriptive landscape diagnostics (NOT a graded endpoint): how much
    # probability mass sits near the optimum — needed to interpret zero
    # regrets (a fat optimal plateau makes every method look perfect).
    diag = {}
    for camp in campaigns:
        lab = camp["labels"]
        diag[camp["tag"]] = {
            "n_within_1e-9_of_max": int((lab >= camp["true_max"] - 1e-9).sum()),
            "frac_within_0.005": float((lab >= camp["true_max"] - 0.005).mean()),
            "frac_within_0.01": float((lab >= camp["true_max"] - 0.01).mean()),
            "label_std": float(lab.std()),
        }
    results["landscape_diagnostics_not_an_endpoint"] = diag

    run_endpoints_1_2_5(campaigns, calendars, results, timings)
    results["timings"] = timings
    save_partial(results)

    run_endpoint_3(campaigns, calendars, results, timings)
    results["timings"] = timings
    save_partial(results)

    run_endpoint_4(campaigns, calendars, results, timings)
    results["timings"] = timings
    save_partial(results)

    run_d2(results, timings)
    timings["total_seconds"] = round(time.time() - t_start, 1)
    results["timings"] = timings
    save_partial(results)
    print(f"[study] complete in {timings['total_seconds']}s -> {OUT_PATH}",
          flush=True)


if __name__ == "__main__":
    main()
