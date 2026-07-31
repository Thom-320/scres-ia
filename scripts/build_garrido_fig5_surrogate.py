#!/usr/bin/env python3
"""Build Figure 5 of Garrido, Pongutá & Adarme (2024): backpropagation against KAN.

His Figure 5 is a neuron. The SCRES **drivers** `d_i` are the dendrites, the simulation
**decision variables** `ρ` are the weights, and the activation asks *"is the SCRES measure at
configuration `x` higher than at configuration `x−1`?"*. His conclusions name backpropagation
and Kolmogorov-Arnold networks as two of the three candidates. This runs both, on his own
90-configuration design.

**Task A is the figure as drawn, and it is an identity.** With the four drivers as inputs and
ReT as the output there is nothing to learn: `results/garrido_drivers_per_configuration/`
established that ReT is *exactly* the sum of the four driver contributions, so any model that
learns "add the inputs" scores a perfect fit. `f1` proves it -- an ordinary least squares fit
returns R² = 1 with all four coefficients at 1. **That is a finding about the figure, not a
result about learning**, and reporting a perfect R² from it without saying so would be the
single easiest way to mislead with this data.

**Task B is the learnable version**: predict SCRES from `ρ` and the risk design ALONE, which is
the question a closed loop would actually have to answer -- how the decision variables shape
resilience. Two heads:

  B1  regression:     (rho, family, risk pattern) -> ReT
  B2  classification: his activation question, over consecutive configurations of a family

Both are evaluated by **grouped** cross-validation on `seed`: his design reuses one seed across
each (Cf_b, Cf_b+30, Cf_b+60) triple, so a random split would put the same trajectory on both
sides and inflate everything. `f2` checks no group leaks.

And both are reported **against baselines** -- a constant predictor and a linear/logistic model.
With 90 rows and 9 features, "the network fits" is not evidence; only "the network beats the
linear rule" is. The baselines are the result, not decoration.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

DRIVERS = Path("results/garrido_drivers_per_configuration/result.json")
FAMILIES = ("R1r", "R2r", "R3")
DRIVER_NAMES = ("Re_APj", "Re_RPj", "Re_DPj_RPj", "Re_FRt")


def load_rows() -> list[dict]:
    blob = json.loads(DRIVERS.read_text())
    if blob["claim_status"] != "DEVELOPMENT_DRIVER_TABLE":
        raise SystemExit(f"driver table is {blob['claim_status']}; refusing to fit on it")
    return blob["rows"], blob["self_sha256"]


def design_features(rows: list[dict]) -> np.ndarray:
    """`rho` plus the risk design -- deliberately WITHOUT the drivers.

    Including a driver would leak the answer: the four contributions sum to ReT exactly.
    """
    out = []
    for r in rows:
        pattern = r["pattern"].ljust(4, "0")
        out.append([
            r["rho"]["buffer_hours"] / 1344.0,          # scaled to [0, 1]
            (r["rho"]["shifts"] - 1) / 2.0,             # 1,2,3 -> 0,0.5,1
            *[1.0 if r["family"] == f else 0.0 for f in FAMILIES],
            *[1.0 if ch == "+" else 0.0 for ch in pattern],
        ])
    return np.asarray(out, dtype=np.float64)


def fit_mlp(x_tr, y_tr, x_te, *, seed: int, classify: bool):
    import torch

    torch.manual_seed(seed)
    net = torch.nn.Sequential(
        torch.nn.Linear(x_tr.shape[1], 16), torch.nn.Tanh(),
        torch.nn.Linear(16, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1))
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    xt = torch.tensor(x_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32).reshape(-1, 1)
    loss_fn = torch.nn.BCEWithLogitsLoss() if classify else torch.nn.MSELoss()
    first = last = None
    for step in range(600):
        opt.zero_grad()
        loss = loss_fn(net(xt), yt)
        loss.backward()
        opt.step()
        if step == 0:
            first = float(loss)
        last = float(loss)
    with torch.no_grad():
        pred = net(torch.tensor(x_te, dtype=torch.float32)).numpy().ravel()
    if classify:
        pred = 1.0 / (1.0 + np.exp(-pred))
    return pred, {"loss_first": first, "loss_last": last}


def fit_kan(x_tr, y_tr, x_te, *, seed: int, classify: bool):
    import torch
    from kan import KAN

    torch.manual_seed(seed)
    model = KAN(width=[x_tr.shape[1], 4, 1], grid=3, k=3, seed=seed,
                auto_save=False, save_act=False, symbolic_enabled=False)
    xt = torch.tensor(x_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32).reshape(-1, 1)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCEWithLogitsLoss() if classify else torch.nn.MSELoss()
    first = last = None
    for step in range(600):
        opt.zero_grad()
        loss = loss_fn(model(xt), yt)
        loss.backward()
        opt.step()
        if step == 0:
            first = float(loss)
        last = float(loss)
    with torch.no_grad():
        pred = model(torch.tensor(x_te, dtype=torch.float32)).numpy().ravel()
    if classify:
        pred = 1.0 / (1.0 + np.exp(-pred))
    return pred, {"loss_first": first, "loss_last": last}


def grouped_folds(groups: np.ndarray, n_folds: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    unique = np.array(sorted(set(groups.tolist())))
    folds = []
    for k in range(n_folds):
        held = set(unique[k::n_folds].tolist())
        test = np.array([i for i, g in enumerate(groups) if g in held])
        train = np.array([i for i, g in enumerate(groups) if g not in held])
        folds.append((train, test))
    return folds


def evaluate(x, y, groups, *, classify: bool, seed: int) -> dict:
    """Grouped CV for every model and every baseline, on identical folds."""
    from sklearn.linear_model import LinearRegression, LogisticRegression

    scores: dict[str, list[float]] = {m: [] for m in
                                      ("constant", "linear", "backprop", "kan")}
    training: dict[str, list[dict]] = {"backprop": [], "kan": []}
    for train, test in grouped_folds(groups):
        x_tr, y_tr, x_te, y_te = x[train], y[train], x[test], y[test]
        if classify:
            major = 1.0 if y_tr.mean() >= 0.5 else 0.0
            scores["constant"].append(float((y_te == major).mean()))
            lin = LogisticRegression(max_iter=2000).fit(x_tr, y_tr)
            scores["linear"].append(float((lin.predict(x_te) == y_te).mean()))
            for name, fit in (("backprop", fit_mlp), ("kan", fit_kan)):
                pred, info = fit(x_tr, y_tr, x_te, seed=seed, classify=True)
                scores[name].append(float(((pred >= 0.5).astype(float) == y_te).mean()))
                training[name].append(info)
        else:
            ss_tot = float(((y_te - y_tr.mean()) ** 2).sum())
            r2 = lambda p: 1.0 - float(((y_te - p) ** 2).sum()) / ss_tot  # noqa: E731
            scores["constant"].append(r2(np.full_like(y_te, y_tr.mean())))
            lin = LinearRegression().fit(x_tr, y_tr)
            scores["linear"].append(r2(lin.predict(x_te)))
            for name, fit in (("backprop", fit_mlp), ("kan", fit_kan)):
                pred, info = fit(x_tr, y_tr, x_te, seed=seed, classify=False)
                scores[name].append(r2(pred))
                training[name].append(info)
    return {
        "per_fold": scores,
        "mean": {k: float(np.mean(v)) for k, v in scores.items()},
        "sd": {k: float(np.std(v, ddof=1)) for k, v in scores.items()},
        "training": training,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=20260731)
    ap.add_argument("--output", type=Path,
                    default=Path("results/garrido_fig5_surrogate/result.json"))
    args = ap.parse_args()
    started = time.perf_counter()

    rows, drivers_seal = load_rows()
    y_ret = np.array([r["ret_excel"] for r in rows], dtype=np.float64)
    groups = np.array([r["seed"] for r in rows])
    x_design = design_features(rows)

    # --- Task A: the figure as drawn ------------------------------------------------------
    contributions = np.array([[r[d]["contribution"] for d in DRIVER_NAMES] for r in rows]
                             + [], dtype=np.float64)
    fifth = np.array([r["not_in_his_ReT_unfulfilled"]["contribution"] for r in rows])
    from sklearn.linear_model import LinearRegression
    ols = LinearRegression(fit_intercept=False).fit(
        np.column_stack([contributions, fifth]), y_ret)
    design_a = np.column_stack([contributions, fifth])
    names_a = list(DRIVER_NAMES) + ["not_in_his_ReT"]
    # A column that is identically zero carries no information about its own coefficient, so
    # sklearn returns 0 for it. Three of the five ARE identically zero here -- Re(APj) is
    # structurally unreachable, Re(DPj) is zero by his Eq. 5.3, and our fifth term never fires.
    # The first version of `f1` demanded every coefficient equal 1 and failed on exactly those
    # three; the identity was fine, the check was not.
    live_a = [i for i in range(design_a.shape[1]) if design_a[:, i].std() > 0.0]
    task_a = {
        "status": "IDENTITY_NOT_A_LEARNING_TASK",
        "r2": float(ols.score(design_a, y_ret)),
        "max_abs_identity_error": float(np.max(np.abs(design_a.sum(axis=1) - y_ret))),
        "coefficients": {n: float(c) for n, c in zip(names_a, ols.coef_)},
        "identified_coefficients": [names_a[i] for i in live_a],
        "degenerate_all_zero_columns": [names_a[i] for i in range(len(names_a))
                                        if i not in live_a],
        "note": ("ReT is exactly the sum of the driver contributions, so the neuron as drawn "
                 "has nothing to learn. Reported so that a perfect fit here is never mistaken "
                 "for evidence"),
    }

    # --- Task B1: regression from rho + risk design ---------------------------------------
    b1 = evaluate(x_design, y_ret, groups, classify=False, seed=args.seed)

    # --- Task B2: his activation question -------------------------------------------------
    # "is SCRES at configuration x higher than at x-1?" -- consecutive cells WITHIN a family,
    # which is the only ordering his design defines.
    pair_x, pair_y, pair_g = [], [], []
    for family in FAMILIES:
        idx = [i for i, r in enumerate(rows) if r["family"] == family]
        idx.sort(key=lambda i: rows[i]["cf"])
        for a, b in zip(idx, idx[1:]):
            pair_x.append(np.concatenate([x_design[b], x_design[a]]))
            pair_y.append(1.0 if y_ret[b] > y_ret[a] else 0.0)
            # group by the PAIR's seeds so neither trajectory can leak across the split
            pair_g.append(f"{groups[a]}|{groups[b]}")
    pair_x = np.asarray(pair_x)
    pair_y = np.asarray(pair_y)
    pair_g = np.asarray(pair_g)
    b2 = evaluate(pair_x, pair_y, pair_g, classify=True, seed=args.seed)

    # --- determinism ----------------------------------------------------------------------
    repeat = evaluate(x_design, y_ret, groups, classify=False, seed=args.seed)

    falsifiers = {
        "f1_task_A_is_an_identity": {
            "passed": (task_a["max_abs_identity_error"] < 1e-12
                       and abs(task_a["r2"] - 1.0) < 1e-9
                       and all(abs(task_a["coefficients"][n] - 1.0) < 1e-6
                               for n in task_a["identified_coefficients"])),
            "evidence": {
                "why_it_can_fail": ("if ReT were NOT the exact sum of the drivers, the neuron "
                                    "as drawn would be a real learning problem and this whole "
                                    "framing would be wrong. Checked on the identity itself, "
                                    "and on the coefficients only where they are IDENTIFIED -- "
                                    "an all-zero column has no coefficient to check, which is "
                                    "what the first version of this falsifier got wrong"),
                **task_a},
        },
        "f2_no_group_leaks_across_a_fold": {
            "passed": all(not (set(groups[tr].tolist()) & set(groups[te].tolist()))
                          for tr, te in grouped_folds(groups)),
            "evidence": {
                "why_it_can_fail": ("his design reuses one seed across each Cf_b/Cf_b+30/"
                                    "Cf_b+60 triple; a random split would put the same "
                                    "trajectory on both sides and inflate every score"),
                "n_groups": int(len(set(groups.tolist()))), "n_rows": int(len(rows))},
        },
        "f3_baselines_are_non_degenerate": {
            "passed": (b2["mean"]["constant"] < 0.95 and float(np.std(y_ret)) > 0.0),
            "evidence": {
                "why_it_can_fail": ("a constant target, or a class balance so skewed that the "
                                    "majority baseline is already ~1.0, would make every "
                                    "comparison below meaningless"),
                "majority_baseline_accuracy": b2["mean"]["constant"],
                "positive_class_share": float(pair_y.mean()),
                "ret_sd": float(np.std(y_ret))},
        },
        "f4_both_networks_actually_train": {
            "passed": all(info["loss_last"] < info["loss_first"]
                          for task in (b1, b2) for name in ("backprop", "kan")
                          for info in task["training"][name]),
            "evidence": {
                "why_it_can_fail": "a dead net would score like the constant baseline in silence",
                "loss_ratio_median": {
                    name: float(np.median([i["loss_last"] / max(i["loss_first"], 1e-12)
                                           for task in (b1, b2)
                                           for i in task["training"][name]]))
                    for name in ("backprop", "kan")}},
        },
        "f5_deterministic_under_a_fixed_seed": {
            "passed": all(abs(b1["mean"][k] - repeat["mean"][k]) < 1e-9 for k in b1["mean"]),
            "evidence": {
                "why_it_can_fail": ("with 90 rows a run-to-run swing could be mistaken for a "
                                    "model difference"),
                "delta": {k: abs(b1["mean"][k] - repeat["mean"][k]) for k in b1["mean"]}},
        },
    }
    falsifiers["all_passed"] = all(v["passed"] for k, v in falsifiers.items()
                                   if k != "all_passed")

    def beats(task: dict, model: str) -> bool:
        """A model counts only if it beats the LINEAR rule by more than a fold SD."""
        return (task["mean"][model] - task["mean"]["linear"]) > task["sd"]["linear"]

    verdict = {
        "B1_regression": {m: beats(b1, m) for m in ("backprop", "kan")},
        "B2_activation": {m: beats(b2, m) for m in ("backprop", "kan")},
        "rule": ("a network counts as learning something only if it beats the linear/logistic "
                 "baseline by more than one between-fold SD of that baseline"),
    }

    print("  === Task A: la figura tal cual está dibujada ===")
    print(f"    R2 = {task_a['r2']:.12f}   coeficientes {task_a['coefficients']}")
    print(f"    -> {task_a['status']}\n")
    for label, task, metric in (("B1 regresion (rho -> ReT)", b1, "R2"),
                                ("B2 activacion (ReT(x) > ReT(x-1)?)", b2, "acc")):
        print(f"  === {label} — {metric}, CV agrupada por semilla ===")
        for model in ("constant", "linear", "backprop", "kan"):
            print(f"    {model:<10} {task['mean'][model]:>8.4f} ± {task['sd'][model]:.4f}")
        print()
    print("  falsadores:")
    for name, check in falsifiers.items():
        if name != "all_passed":
            print(f"    {name:<40} {'PASA' if check['passed'] else 'FALLA'}")
    print(f"\n  ¿alguna red supera al lineal? {verdict['B1_regression']} / "
          f"{verdict['B2_activation']}")

    payload = {
        "schema_version": "garrido_fig5_surrogate_v1",
        "claim_status": ("DEVELOPMENT_FIG5_SURROGATE" if falsifiers["all_passed"]
                         else "HALTED_FALSIFIER_FAILED"),
        "question": ("Figure 5 of Garrido, Ponguta & Adarme (2024): can a neural network map "
                     "the SCRES drivers, weighted by the decision variables, onto SCRES -- and "
                     "does backpropagation or KAN do it better than a linear rule?"),
        "driver_table": str(DRIVERS), "driver_table_sha256": drivers_seal,
        "task_A_figure_as_drawn": task_a,
        "task_B1_regression": b1,
        "task_B2_activation_question": b2,
        "verdict": verdict,
        "n_configurations": len(rows), "n_pairs": int(len(pair_y)),
        "features": ("buffer_hours, shifts, family one-hot, risk pattern flags -- "
                     "the drivers are deliberately EXCLUDED, they sum to the target"),
        "falsifiers": falsifiers,
        "seed": args.seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    body = json.dumps(payload, indent=1, sort_keys=True, default=str)
    payload["self_sha256"] = sha256(body.encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str) + "\n")
    print(f"\n  -> {args.output} (sello {payload['self_sha256'][:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
