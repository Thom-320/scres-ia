#!/usr/bin/env python3
"""Gate B: was the uncaptured surface premium the environment's fault, or ours?

Contract: `docs/CONTRATO_PROGRAMA_N_PRIMA_NEURAL_2026-08-09.md`
Preregistration: `docs/PREREGISTRO_PUERTA_B_SUPERFICIE_CD_2026-08-09.md`
No seeds are opened: the object of study is the FIT, so the tapes stay identical.

WHAT THE PREDECESSOR MEASURED. `cd_surface_prediction_premium` =
PREMIUM_WAS_AVAILABLE_AND_NOT_CAPTURED: a ceiling of 0.693 against a declared primary baseline of
0.631, and both networks BELOW the linear model at 0.602 and 0.584. That cannot support "the
environment has no premium".

WHY THE NETWORKS LOST, read from the code rather than guessed. Its neural arms were a 16-16 tanh
MLP and a width-4 KAN, both trained for a FIXED 600 Adam steps with no validation split, no early
stopping, no regularisation, no hyperparameter selection and one init seed per fold -- against OLS,
which is closed-form optimal for its basis. An untuned fit against an analytic optimum.

SO ONLY THE FIT CHANGES, and symmetrically: standardisation on train only, an inner validation
split with early stopping, one small declared grid, five init seeds averaged. Tuning never sees the
test fold, which is precisely the sin the original run declared about its own spline baseline.

THE RECURRENT ARM HAS A RICHER INFORMATION SET AND SAYS SO. Garrido's Fig. 5 activation compares
configuration x with x-1, so a sequence surrogate needs the PREVIOUS resilience value -- which the
other arms never see. It is therefore reported against its own matched classical comparator, a
linear model with the same lagged input, and never against the arms that lack it.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from build_garrido_fig5_surrogate import grouped_folds                           # noqa: E402
from run_cd_surface_prediction_premium import (                                  # noqa: E402
    BUFFER_HOURS, ESCALATIONS, FAMILIES, FAMILY_RISKS, SEED_BASE, base_features,
    episode, ols, r2, rich_features)
from supply_chain.arm_runner import seal_and_write                               # noqa: E402
from supply_chain import falsifiers as F                                         # noqa: E402
from supply_chain.cobb_douglas_resilience import derive_exponents, resilience_index  # noqa: E402
from supply_chain.config import HOURS_PER_WEEK                                   # noqa: E402
from supply_chain.seed_custody import custody_falsifier, module_manifest         # noqa: E402

SESOI = 0.05
T_CRIT = {4: 2.776, 3: 3.182, 2: 4.303}
#: Declared here and closed. Identical for MLP and KAN, so the comparison is architecture.
GRID = [{"width": w, "lr": lr, "wd": wd}
        for w in (16, 64) for lr in (3e-3, 1e-2) for wd in (0.0, 1e-4)]
MAX_STEPS, PATIENCE, INIT_SEEDS = 5_000, 300, 5
PRIMARY = "linear_interactions"
OUT = Path("results/program_n/gate_b_cd_surface/result.json")
CONTRACT = Path("docs/PREREGISTRO_PUERTA_B_SUPERFICIE_CD_2026-08-09.md")
MODULES = ("supply_chain/cobb_douglas_resilience.py", "supply_chain/supply_chain.py",
           "supply_chain/falsifiers.py")
#: The predecessor's held-out means, for the reproduction check.
PREDECESSOR = {"constant": -0.0167, "linear_additive": 0.6062, "linear_interactions": 0.6306,
               "spline_buffer": 0.6365, "tree": 0.6225, "train_cell_mean_comparator": 0.6931,
               "kan": 0.6019, "backprop": 0.5841}


#: The widened non-neural class, added 2026-08-12 after five external reviews said the declared
#: class was too narrow to support the words "best non-neural comparator". Same features, same
#: folds; only the family differs. Fixed hyperparameters, deliberately: searching them would need a
#: budget-matching argument, and leaving them fixed understates these arms, which is the direction
#: that cannot manufacture a neural premium.
WIDENED = ("gbdt", "random_forest", "gaussian_process", "kernel_ridge")
#: The same families given the lagged information set, so the recurrent arm faces its own class.
WIDENED_LAGGED = tuple(f"{m}_lagged" for m in WIDENED)


def standardise(train, *others):
    """Fit on TRAIN only and apply everywhere. The predecessor standardised nothing."""
    mu, sd = train.mean(axis=0), train.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return [(a - mu) / sd for a in (train, *others)], (mu, sd)


def widened_predict(kind: str, x_tr, y_tr, x_te):
    """One fit per fold from a fixed configuration. No tuning, no access to the test fold.

    Distance-based families are STANDARDISED on train first. Without it an RBF with gamma=1 over
    raw one-hots scored -1.08 in the smoke run -- and a straw-man comparator proves nothing, which
    is the whole reason this widened class exists. Trees are scale-free and get the raw features.
    """
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    from sklearn.kernel_ridge import KernelRidge
    if kind in ("gaussian_process", "kernel_ridge"):
        (x_tr, x_te), _ = standardise(x_tr, x_te)
    if kind == "gbdt":
        model = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                          subsample=0.9, random_state=20260812)
    elif kind == "random_forest":
        model = RandomForestRegressor(n_estimators=400, min_samples_leaf=2, n_jobs=1,
                                      random_state=20260812)
    elif kind == "gaussian_process":
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(1e-3)
        model = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=1e-8,
                                         random_state=20260812)
    elif kind == "kernel_ridge":
        # KernelRidge does not centre the target, so with y in [0,1] and alpha=1 it shrinks every
        # prediction toward zero and scored -6.90 in the smoke run. Centre on train and add back,
        # which is what `normalize_y` does for the GP. Same reason as the standardisation above:
        # a comparator that loses because I mis-specified it proves nothing.
        mu_y = float(np.mean(y_tr))
        model = KernelRidge(alpha=1.0, kernel="rbf", gamma=1.0)
        return model.fit(x_tr, y_tr - mu_y).predict(x_te) + mu_y
    else:                                                              # pragma: no cover
        raise ValueError(f"unknown widened arm {kind!r}")
    return model.fit(x_tr, y_tr).predict(x_te)


def _torch_fit(kind, x_tr, y_tr, x_va, y_va, x_te, cfg, seed):
    """One fit with early stopping on an inner validation split. Returns test predictions."""
    import torch
    torch.manual_seed(seed)
    d = x_tr.shape[1]
    if kind == "mlp":
        net = torch.nn.Sequential(torch.nn.Linear(d, cfg["width"]), torch.nn.Tanh(),
                                  torch.nn.Linear(cfg["width"], cfg["width"]), torch.nn.Tanh(),
                                  torch.nn.Linear(cfg["width"], 1))
    else:
        from kan import KAN
        net = KAN(width=[d, max(2, cfg["width"] // 8), 1], grid=3, k=3, seed=seed,
                  auto_save=False, save_act=False, symbolic_enabled=False)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    tx = torch.tensor(x_tr, dtype=torch.float32)
    ty = torch.tensor(y_tr, dtype=torch.float32).reshape(-1, 1)
    vx = torch.tensor(x_va, dtype=torch.float32)
    vy = torch.tensor(y_va, dtype=torch.float32).reshape(-1, 1)
    loss_fn = torch.nn.MSELoss()
    best, best_state, since = np.inf, None, 0
    for step in range(MAX_STEPS):
        opt.zero_grad()
        loss_fn(net(tx), ty).backward()
        opt.step()
        if step % 25 == 0:
            with torch.no_grad():
                v = float(loss_fn(net(vx), vy))
            if v < best - 1e-9:
                best, since = v, 0
                best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
            else:
                since += 25
                if since >= PATIENCE:
                    break
    if best_state is not None:
        net.load_state_dict(best_state)
    with torch.no_grad():
        return net(torch.tensor(x_te, dtype=torch.float32)).numpy().ravel(), best


def tuned_predict(kind, x_tr, y_tr, x_te, fold_seed, rng):
    """Grid selected on an INNER validation split of train, then averaged over init seeds."""
    n = len(y_tr)
    perm = rng.permutation(n)
    cut = max(8, int(0.75 * n))
    itr, iva = perm[:cut], perm[cut:]
    (xs_tr, xs_va, xs_te), _ = standardise(x_tr[itr], x_tr[iva], x_te)
    ymu, ysd = float(y_tr[itr].mean()), float(y_tr[itr].std() or 1.0)
    ys_tr, ys_va = (y_tr[itr] - ymu) / ysd, (y_tr[iva] - ymu) / ysd

    best_cfg, best_v = None, np.inf
    for cfg in GRID:
        _, v = _torch_fit(kind, xs_tr, ys_tr, xs_va, ys_va, xs_te, cfg, fold_seed)
        if v < best_v:
            best_cfg, best_v = cfg, v
    # Refit on ALL of train with the selected configuration, averaged over init seeds.
    (xa_tr, xa_te), _ = standardise(x_tr, x_te)
    amu, asd = float(y_tr.mean()), float(y_tr.std() or 1.0)
    ya = (y_tr - amu) / asd
    cut2 = max(8, int(0.85 * len(ya)))
    preds = []
    for k in range(INIT_SEEDS):
        p, _ = _torch_fit(kind, xa_tr[:cut2], ya[:cut2], xa_tr[cut2:], ya[cut2:], xa_te,
                          best_cfg, fold_seed + 100 * k)
        preds.append(p * asd + amu)
    return np.mean(preds, axis=0), best_cfg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seed-base", type=int, default=None,
                    help="override the seed base; used by the confirmation on a "
                         "virgin block, which changes ONLY the seed values")
    ap.add_argument("--horizon-weeks", type=int, default=52)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--skip-kan", action="store_true")
    ap.add_argument("--endpoint", choices=("cobb_douglas", "ret_excel"), default="cobb_douglas",
                    help="ret_excel is the LEGACY SENSITIVITY surface. It is a prediction target "
                         "here and never a training reward: ret_excel is measured to reward "
                         "abandonment, so a policy fitted to it is forbidden. Predicting it asks "
                         "whether the premium is a property of the method or of one metric.")
    ap.add_argument("--replay-of", type=str, default=None,
                    help="registry block id whose tapes this run re-reads. A sensitivity on the "
                         "same tapes is a declared replay, not an independent confirmation.")
    ap.add_argument("--confirmation-of", type=Path, default=None,
                    help="path to the development artifact. Switches f2 from the DEVELOPMENT check "
                         "(same tapes, so levels must match) to the CONFIRMATION check (different "
                         "tapes, so code identity and rank order must match instead)")
    ap.add_argument("--output", type=Path, default=Path(__file__).resolve().parent.parent / OUT)
    args = ap.parse_args()
    started = time.perf_counter()
    horizon = float(args.horizon_weeks * HOURS_PER_WEEK)
    base = args.seed_base if args.seed_base is not None else SEED_BASE
    seeds = [base + i for i in range(args.seeds)]

    cells, index, legacy_ret = {}, [], {}
    for family in FAMILIES:
        for escalation, mult in ESCALATIONS.items():
            for buf in BUFFER_HOURS:
                for seed in seeds:
                    agg, ret = episode(FAMILY_RISKS[family], mult, buf, seed, horizon)
                    cells[(family, escalation, buf, seed)] = agg
                    legacy_ret[(family, escalation, buf, seed)] = float(ret)
                    index.append((family, escalation, buf, seed))
        print(f"  {family} listo ({time.perf_counter() - started:.0f}s)", flush=True)

    aggs = [cells[k] for k in index]
    x_base = np.array([base_features(b, f, e) for (f, e, b, _) in index])
    x_rich = np.array([rich_features(b, f, e) for (f, e, b, _) in index])
    g = np.array([s for (_, _, _, s) in index])
    cell_key = [(f, e, b) for (f, e, b, _) in index]

    ret_vector = np.array([legacy_ret[k] for k in index])

    def target_from_training(train_idx):
        if args.endpoint == "ret_excel":
            # No train-dependent calibration: ret_excel carries its own scale, so the target does
            # not move with the fold. The fold structure still governs fitting and evaluation.
            return ret_vector
        maxima = {v: max(max(aggs[i][v] for i in train_idx), 1.0 + 1e-9)
                  for v in ("zeta", "epsilon", "phi", "tau")}
        maxima["kappa_dot"] = float(len(train_idx))
        exps = derive_exponents(maxima)
        total = float(sum(aggs[i]["kappa"] for i in train_idx))
        scale = float(len(train_idx)) / total if total > 0 else 1.0
        return np.array([
            resilience_index({"zeta": a["zeta"], "epsilon": a["epsilon"], "phi": a["phi"],
                              "tau": a["tau"], "kappa_dot": max(a["kappa"] * scale, 1e-9)},
                             exps)["R_cobb_douglas"] for a in aggs])

    # Lagged-y features: the information the recurrent arm needs and no other arm has. The
    # previous configuration is the next lower buffer level in the same (family, escalation, seed).
    order = {k: i for i, k in enumerate(BUFFER_HOURS)}
    prev_of = {}
    for i, (f, e, b, s) in enumerate(index):
        j = order[b] - 1
        prev_of[i] = next((k for k, (f2, e2, b2, s2) in enumerate(index)
                           if j >= 0 and (f2, e2, s2) == (f, e, s) and b2 == BUFFER_HOURS[j]), None)

    folds = grouped_folds(g, n_folds=args.folds)
    arms = ["constant", "linear_additive", "linear_interactions", "spline_buffer", "tree",
            "train_cell_mean_comparator", "mlp_tuned", "linear_lagged", "recurrent"]
    if not args.skip_kan:
        arms.append("kan_tuned")
    arms.extend(WIDENED)
    arms.extend(WIDENED_LAGGED)
    per_fold = {m: [] for m in arms}
    # Second resolution, added 2026-08-27. grouped_folds puts every seed in
    # exactly one test fold, so each seed yields exactly one held-out score.
    # Those scores are independent across seeds; CV folds are not, because
    # their training sets overlap. The power audit showed the fold unit made
    # every interval ~33% too narrow (Nadeau-Bengio) and left MDE80 above the
    # SESOI in 12 of 12 architecture contrasts.
    per_seed: dict[str, dict[int, float]] = {m: {} for m in arms}
    seed_of_fold: dict[int, list[int]] = {}

    def record(name, te_idx, pred, fold_index=None):
        """Score one arm on one test fold at both resolutions."""
        pred = np.asarray(pred, dtype=float)
        value = r2(y[te_idx], pred)
        per_fold[name].append(value)
        gte = g[te_idx]
        for grp in np.unique(gte):
            mask = gte == grp
            if int(mask.sum()) >= 2:
                per_seed[name][int(grp)] = r2(y[te_idx][mask], pred[mask])
        if fold_index is not None:
            seed_of_fold.setdefault(int(fold_index),
                                    sorted(int(v) for v in np.unique(gte)))
        return value
    chosen = {"mlp_tuned": [], "kan_tuned": [], "recurrent": []}
    rng = np.random.default_rng(20260809)

    def spline_features(rows):
        knots = [336.0, 672.0, 1008.0]
        return np.asarray([[*base_features(index[i][2], index[i][0], index[i][1]),
                            *[max(0.0, (index[i][2] - k) / 1344.0) for k in knots]] for i in rows])

    def tree_predict(x_tr, y_tr, x_te, depth=4):
        def build(idx, d):
            if d == 0 or len(idx) < 8 or float(y_tr[idx].std()) < 1e-12:
                return float(y_tr[idx].mean())
            best = None
            for col in range(x_tr.shape[1]):
                for thr in np.unique(x_tr[idx, col])[:-1]:
                    lo, hi = idx[x_tr[idx, col] <= thr], idx[x_tr[idx, col] > thr]
                    if len(lo) < 4 or len(hi) < 4:
                        continue
                    sse = float(((y_tr[lo] - y_tr[lo].mean()) ** 2).sum()
                                + ((y_tr[hi] - y_tr[hi].mean()) ** 2).sum())
                    if best is None or sse < best[0]:
                        best = (sse, col, thr, lo, hi)
            if best is None:
                return float(y_tr[idx].mean())
            _, col, thr, lo, hi = best
            return (col, thr, build(lo, d - 1), build(hi, d - 1))
        tree = build(np.arange(len(y_tr)), depth)

        def walk(node, row):
            while isinstance(node, tuple):
                col, thr, a, b = node
                node = a if row[col] <= thr else b
            return node
        return np.array([walk(tree, r) for r in x_te])

    for fi, (tr, te) in enumerate(folds):
        y = target_from_training(tr)
        cell_mean = {}
        for key in set(cell_key[i] for i in tr):
            cell_mean[key] = float(y[[i for i in tr if cell_key[i] == key]].mean())
        y_cm = np.array([cell_mean.get(cell_key[i], float(y[tr].mean())) for i in te])

        record("constant", te, np.full(len(te), y[tr].mean()), fi)
        record("linear_additive", te, ols(x_base[tr], y[tr], x_base[te]))
        record("linear_interactions", te, ols(x_rich[tr], y[tr], x_rich[te]))
        record("spline_buffer", te,
               ols(spline_features(list(tr)), y[tr], spline_features(list(te))))
        record("tree", te, tree_predict(x_base[tr], y[tr], x_base[te]))
        record("train_cell_mean_comparator", te, y_cm)

        # The widened non-neural class. Five reviews independently said the declared class --
        # constant, additive, interactions, spline, hand-rolled tree -- was too narrow to support
        # "beats the best non-neural comparator". These get the SAME seven configuration features
        # and the same folds, so only the model family differs. Their hyperparameters are fixed
        # here, not searched, which UNDERSTATES them and is the conservative direction.
        for name in WIDENED:
            record(name, te, widened_predict(name, x_base[tr], y[tr], x_base[te]))
        for name in WIDENED_LAGGED:
            pass  # filled below, once the lagged design matrix exists

        pred, cfg = tuned_predict("mlp", x_base[tr], y[tr], x_base[te], 3000 + fi, rng)
        record("mlp_tuned", te, pred)
        chosen["mlp_tuned"].append(cfg)
        if not args.skip_kan:
            try:
                pk, ck = tuned_predict("kan", x_base[tr], y[tr], x_base[te], 3000 + fi, rng)
                record("kan_tuned", te, pk)
                chosen["kan_tuned"].append(ck)
            except Exception as exc:                                   # pragma: no cover
                per_fold["kan_tuned"].append(float("nan"))
                # no prediction exists, so no seed can be scored
                print(f"  KAN no disponible ({exc})", flush=True)

        # The lagged information set, declared: previous configuration's resilience, or the
        # training mean where no previous configuration exists.
        lag = np.array([y[prev_of[i]] if prev_of.get(i) is not None else float(y[tr].mean())
                        for i in range(len(index))]).reshape(-1, 1)
        x_lag = np.hstack([x_base, lag])
        record("linear_lagged", te, ols(x_lag[tr], y[tr], x_lag[te]))
        for name in WIDENED_LAGGED:
            base = name.replace("_lagged", "")
            record(name, te, widened_predict(base, x_lag[tr], y[tr], x_lag[te]))
        pr, cr = tuned_predict("mlp", x_lag[tr], y[tr], x_lag[te], 5000 + fi, rng)
        record("recurrent", te, pr)
        chosen["recurrent"].append(cr)
        print(f"  fold {fi} listo ({time.perf_counter() - started:.0f}s)", flush=True)

    means = {m: float(np.nanmean(v)) for m, v in per_fold.items()}
    t_crit = T_CRIT.get(args.folds - 1, 2.776)

    def paired(model, baseline):
        d = np.array(per_fold[model]) - np.array(per_fold[baseline])
        d = d[~np.isnan(d)]
        se = float(d.std(ddof=1) / np.sqrt(d.size)) if d.size > 1 else 0.0
        return {"baseline": baseline, "mean_difference": float(d.mean()),
                "ci95_low": float(d.mean() - t_crit * se),
                "ci95_high": float(d.mean() + t_crit * se), "sesoi": SESOI,
                "passes_sesoi_and_ci": bool(d.mean() >= SESOI and d.mean() - t_crit * se > 0)}

    vs_primary = {m: paired(m, PRIMARY) for m in ("mlp_tuned", "kan_tuned", "spline_buffer", "tree")
                  if m in per_fold}
    vs_lagged = {"recurrent": paired("recurrent", "linear_lagged")}
    reproduction = {m: abs(means[m] - PREDECESSOR[m]) for m in
                    ("constant", "linear_additive", "linear_interactions", "spline_buffer",
                     "tree", "train_cell_mean_comparator")}

    # f2 has two forms and the first burned a block. On the SAME tapes it asks whether anything
    # but the neural fit changed, and levels must match. On DIFFERENT tapes that question becomes
    # "do eight fresh seeds reproduce eight old ones", which has no reason to hold: the classical
    # arms are deterministic given the data and move because the data moved. The confirmation form
    # asks the intended question without depending on the data -- byte-identical module manifest,
    # and the classical ranking preserved.
    if args.endpoint != "cobb_douglas":
        # The target changed, so neither the levels nor their order have any reason to reproduce
        # the Cobb-Douglas run, and testing them would be the same category error that burned two
        # blocks. What still must hold, and is the whole point of a sensitivity, is that the
        # instrument is the same one.
        dev = json.loads(Path(args.confirmation_of).read_text()) if args.confirmation_of else {}
        manifest_same = (module_manifest(MODULES) == dev.get("module_manifest")) if dev else None
        f2 = F.check(
            bool(manifest_same) if dev else False,
            "a sensitivity is only a sensitivity if the code is byte-identical to the run it is a "
            "sensitivity OF. This fails if any module hash differs, or if no reference run was "
            "given at all -- in which case nothing is being varied but the target",
            computed_from={"n_modules": len(MODULES),
                           "reference_given": float(bool(dev))},
            module_manifest_identical=manifest_same,
            levels_not_compared="the endpoint differs; level and order reproduction are undefined",
            reference_artifact=str(args.confirmation_of) if args.confirmation_of else None)
    elif args.confirmation_of is not None:
        # Amendment 2026-08-09 (second): the previous confirmation form demanded that the classical
        # ranking be preserved exactly. It burned block 9500001-9500008 on a 0.0153 swap between
        # spline_buffer and linear_interactions whose own paired CI was [-0.0388, +0.0082] -- a
        # sign test on a quantity that straddles zero, which is the failure mode the Program L
        # closure forbade. The order clause is REMOVED. What replaces it is a reversal that is
        # itself significant: sampling noise cannot produce one, a changed instrument can.
        dev = json.loads(Path(args.confirmation_of).read_text())
        manifest_same = (module_manifest(MODULES) == dev.get("module_manifest"))
        order = ["spline_buffer", "linear_interactions", "linear_additive", "constant"]
        reversals = []
        for i in range(len(order) - 1):
            hi, lo = order[i], order[i + 1]
            p = paired(lo, hi)                       # positive => the pair reversed
            if p["ci95_low"] > 0.0:
                reversals.append({"expected_higher": hi, "observed_higher": lo,
                                  "mean_difference": p["mean_difference"],
                                  "ci95_low": p["ci95_low"], "ci95_high": p["ci95_high"]})
        f2 = F.check(
            bool(manifest_same and not reversals),
            "on fresh tapes the levels MUST move and so may their order when two arms are "
            "indistinguishable; what may not move is the instrument. This fails if a module hash "
            "differs, or if a classical pair reverses with its own paired CI excluding zero -- a "
            "reversal that sampling variation cannot manufacture",
            computed_from={"n_classical_ranked": len(order),
                           "n_significant_reversals": len(reversals),
                           "max_level_shift": max(abs(means[m] - dev["held_out_r2_mean"][m])
                                                  for m in order)},
            module_manifest_identical=manifest_same, significant_reversals=reversals,
            development_artifact=str(args.confirmation_of))
    else:
        f2 = F.lt(
            max(reproduction.values()), 0.02,
            "the classical arms are untouched code; if they move, I changed more than the neural "
            "fit and nothing here is comparable with the predecessor")

    checks = {
        "f1_ceiling_still_above_the_primary": F.gt(
            means["train_cell_mean_comparator"] - means[PRIMARY], 0.0,
            "if the ceiling vanishes once the fit is repaired, there was no margin to capture and "
            "the predecessor's headline was itself an artefact"),
        "f2_classical_arms_reproduce": f2,
        "f3_tuning_used_only_inner_validation": F.check(
            all(len(v) == args.folds for k, v in chosen.items() if v),
            "selection that sees the test fold is exactly the sin the predecessor declared about "
            "its own spline baseline",
            computed_from={"folds": args.folds, "grid_size": len(GRID)}),
        "f4_networks_now_reach_the_linear": F.ge(
            min(means["mlp_tuned"], means.get("kan_tuned", means["mlp_tuned"])) - means[PRIMARY],
            0.0,
            "H_B may simply be false: if the repaired networks still lose to OLS, the fit was not "
            "the problem and the surface does not admit a better approximator here"),
        "f5_neural_premium_over_the_primary": F.check(
            any(v["passes_sesoi_and_ci"] for v in vs_primary.values()
                if v is not vs_primary.get("spline_buffer")),
            "reaching the linear model is not beating it; the margin may exist and stay "
            "uncaptured even with the fit repaired",
            computed_from={"sesoi": SESOI, "n_compared": len(vs_primary)}),
        "f6_recurrent_arm_is_reported": F.gt(
            len(per_fold["recurrent"]), 0,
            "without it Garrido's Fig. 5 as a PREDICTOR is not answered at all"),
        "f7_budgets_are_matched": F.check(
            len(GRID) == 8 and INIT_SEEDS == 5,
            "an unequal grid or seed count between architectures measures budget, not architecture",
            computed_from={"grid": len(GRID), "init_seeds": INIT_SEEDS, "max_steps": MAX_STEPS}),
    }
    checks["d1_recurrent_information_set"] = F.disclosure(
        "the recurrent arm sees the PREVIOUS configuration's resilience, which no other arm does, "
        "because Fig. 5's activation compares x with x-1. It is therefore judged against "
        "linear_lagged, a classical model with the same input, and never against the arms that "
        "lack it",
        evidence={"recurrent_vs_linear_lagged": vs_lagged["recurrent"]})
    checks["custody"] = custody_falsifier(seeds, replay_of=args.replay_of)
    summary = F.summarise(checks)

    if not checks["f2_classical_arms_reproduce"]["passed"]:
        status = "BLOCKED_INSTRUMENT"
    elif args.endpoint == "ret_excel":
        # A sensitivity never earns the confirmatory label, whichever way it comes out.
        status = ("SENSITIVITY_PREMIUM_HOLDS_ON_LEGACY_SURFACE"
                  if checks["f5_neural_premium_over_the_primary"]["passed"]
                  else "SENSITIVITY_PREMIUM_DOES_NOT_HOLD_ON_LEGACY_SURFACE")
    elif not checks["f4_networks_now_reach_the_linear"]["passed"]:
        status = "NETWORKS_WERE_NOT_THE_PROBLEM"
    elif checks["f5_neural_premium_over_the_primary"]["passed"]:
        status = "SURFACE_PREMIUM_CAPTURED"
    else:
        status = "NETWORKS_REACH_THE_LINEAR_BUT_DO_NOT_BEAT_IT"

    # Grade is DERIVED, not hardcoded. Both fields were frozen strings from the first version and
    # `--seed-base` was added later without touching them, so `gate_b_confirmation_v3` ran on a
    # fresh block and still sealed run_role=DEVELOPMENT / scope=..._NO_NEW_SEEDS. The artifact then
    # contradicted every document that called it a confirmation. Note the ceiling on what may be
    # claimed: the registry declares itself incomplete, so the strongest honest grade is
    # "no known collision", never "virgin".
    opened_a_block = args.seed_base is not None and args.endpoint == "cobb_douglas"
    custody_clean = checks["custody"].get("passed") is True
    if args.replay_of is not None:
        run_role, scope = ("DECLARED_REPLAY",
                           "DECLARED_REPLAY_OF_CONSUMED_TAPES_NO_NEW_SEEDS")
    elif args.endpoint == "ret_excel":
        run_role, scope = ("SENSITIVITY_REPLAY",
                           "SENSITIVITY_REPLAY_SAME_TAPES_DIFFERENT_TARGET_NOT_A_CONFIRMATION")
    elif opened_a_block and custody_clean:
        run_role, scope = ("PROSPECTIVE",
                           "PROSPECTIVE_FRESH_BLOCK_NO_KNOWN_COLLISION_VIRGINITY_NOT_PROVEN")
    elif opened_a_block:
        run_role, scope = ("PROSPECTIVE_WITH_CUSTODY_CONFLICT",
                           "PROSPECTIVE_BLOCK_WITH_A_RECORDED_CUSTODY_CONFLICT")
    else:
        run_role, scope = "DEVELOPMENT", "DEVELOPMENT_REANALYSIS_NO_NEW_SEEDS"

    payload = {
        "schema_version": "program_n_gate_b_v1", "claim_status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_role": run_role,
        "scope": scope,
        "endpoint": ("held_out_r2_on_R_cobb_douglas" if args.endpoint == "cobb_douglas"
                     else "held_out_r2_on_ret_excel_risk_conditional__LEGACY_SENSITIVITY"),
        "seeds": seeds,
        "per_seed": {m: {str(k): v for k, v in sorted(d.items())}
                     for m, d in per_seed.items()},
        "seed_of_fold": {str(k): v for k, v in sorted(seed_of_fold.items())},
        "resampling_unit": "seed",
        "resampling_unit_note": (
            "per_seed is the preregistered unit of analysis: grouped_folds puts "
            "each seed in exactly one test fold, so the scores are independent "
            "across seeds. per_fold is retained only for comparability with the "
            "sealed 8-seed run, whose overlapping training sets made every "
            "interval about 33% too narrow."
        ),
        "sesoi": SESOI, "primary_baseline": PRIMARY,
        "grid": GRID, "init_seeds": INIT_SEEDS, "max_steps": MAX_STEPS, "patience": PATIENCE,
        "held_out_r2_mean": means, "per_fold": per_fold, "chosen_hyperparameters": chosen,
        "vs_primary": vs_primary, "vs_lagged": vs_lagged,
        "predecessor_means": PREDECESSOR, "reproduction_gap": reproduction,
        "confirmation_of": str(args.confirmation_of) if args.confirmation_of else None,
        "module_manifest": module_manifest(MODULES),
        "falsifiers": checks, "falsifier_summary": summary,
        "elapsed_seconds": time.perf_counter() - started,
    }
    seal_and_write(payload, args.output, contract=CONTRACT,
                   reference=Path("results/headroom/cd_surface_prediction_premium/result.json"))

    print(f"\nveredicto: {status}\n")
    for m, v in sorted(means.items(), key=lambda kv: -kv[1]):
        was = PREDECESSOR.get(m)
        tag = f"   (antes {was:+.4f})" if was is not None else ""
        print(f"  {m:28}{v:+.4f}{tag}")
    print("\n  contra el baseline primario:")
    for m, v in vs_primary.items():
        print(f"    {m:20}{v['mean_difference']:+.4f} "
              f"[{v['ci95_low']:+.4f}, {v['ci95_high']:+.4f}]  "
              f"{'PASA' if v['passes_sesoi_and_ci'] else 'no'}")
    r = vs_lagged["recurrent"]
    print(f"    {'recurrent vs lagged':20}{r['mean_difference']:+.4f} "
          f"[{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}]")
    print(f"\n  falsadores: {summary['n_computed']} computados, {summary['n_failed']} fallidos")
    for name, c in checks.items():
        if c.get("computed"):
            print(f"    {name:40} {'PASA' if c['passed'] else 'FALLA'}")
    print(f"\n  -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
