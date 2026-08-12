#!/usr/bin/env python3
"""Phase 3: does a decision-focused surrogate CHOOSE a better configuration?

Preregistration: `docs/PREREGISTRO_FASE_3_SURROGATE_ORIENTADO_A_DECISION_2026-08-12.md`

Fitting better is not choosing better, and this project has already measured the first without ever
measuring the second. Gate B's design is nine context cells and seventeen buffer levels, so the
decision -- which buffer to deploy given the context -- already exists and does not have to be
invented.

THE OPTIMIZER IS FROZEN. Every arm calls the SAME exhaustive argmax over the seventeen levels, so
no arm can win by searching better; only the model that scores the levels differs. That is the
ablation that makes a positive result attributable to the surrogate rather than to the pipeline.

TWO NEURAL ARMS, on purpose. `mlp_mse` is Gate B's architecture on the usual squared loss;
`mlp_decision` is the same architecture on expected decision regret under a soft policy. The gap
between them isolates the LOSS from the ARCHITECTURE, which no run in this tree has done.

No seed is opened: declared replay of the tapes gate_b_confirmation_v3 consumed.
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
    BUFFER_HOURS, ESCALATIONS, FAMILIES, FAMILY_RISKS, base_features, episode, ols, rich_features)
from run_program_n_gate_b_v1 import standardise, widened_predict                 # noqa: E402
from supply_chain.arm_runner import seal_and_write                               # noqa: E402
from supply_chain import falsifiers as F                                         # noqa: E402
from supply_chain.cobb_douglas_resilience import derive_exponents, resilience_index  # noqa: E402
from supply_chain.config import HOURS_PER_WEEK                                   # noqa: E402
from supply_chain.seed_custody import custody_falsifier                          # noqa: E402

CONTRACT = Path("docs/PREREGISTRO_FASE_3_SURROGATE_ORIENTADO_A_DECISION_2026-08-12.md")
OUT = Path("results/program_n/phase3_decision_surrogate/result.json")
SEED_BASE, N_SEEDS, N_FOLDS = 9600001, 8, 5
REPLAY_OF = "program_n_gate_b_confirmation_v3"
TAU = 0.02                       # declared in the preregistration, never tuned here
MAX_STEPS, PATIENCE, INIT_SEEDS = 5_000, 300, 5
WIDTH, LR, WD = 64, 3e-3, 0.0    # the Gate B grid's midpoint, frozen so the loss is the variable
CLASSICAL = ("linear_interactions", "spline_buffer", "gbdt", "random_forest",
             "gaussian_process", "kernel_ridge")
NEURAL = ("mlp_mse", "mlp_decision")
FLOOR = "random_surrogate"

#: Counts every call to the shared argmax, so k2 can prove no arm searched more than another.
ARGMAX_CALLS: dict[str, int] = {}


def choose(arm: str, scores_by_cell: np.ndarray) -> np.ndarray:
    """THE frozen optimizer. `scores_by_cell` is (cells, actions); returns the index per cell."""
    ARGMAX_CALLS[arm] = ARGMAX_CALLS.get(arm, 0) + 1
    return np.asarray(scores_by_cell).argmax(axis=1)


def spline_features(rows_buf, rows_fam, rows_esc):
    knots = [336.0, 672.0, 1008.0]
    return np.asarray([[*base_features(b, f, e), *[max(0.0, (b - k) / 1344.0) for k in knots]]
                       for b, f, e in zip(rows_buf, rows_fam, rows_esc)])


def _fit_net(x_tr, y_tr, cells_tr, x_all, seed, decision_focused):
    """One MLP. With `decision_focused`, the loss is expected regret under a soft policy."""
    import torch
    torch.manual_seed(seed)
    (xs_tr, xs_all), _ = standardise(x_tr, x_all)
    mu, sd = float(y_tr.mean()), float(y_tr.std() or 1.0)
    ys = (y_tr - mu) / sd
    d = xs_tr.shape[1]
    net = torch.nn.Sequential(torch.nn.Linear(d, WIDTH), torch.nn.Tanh(),
                              torch.nn.Linear(WIDTH, WIDTH), torch.nn.Tanh(),
                              torch.nn.Linear(WIDTH, 1))
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WD)
    tx = torch.tensor(xs_tr, dtype=torch.float32)
    ty = torch.tensor(ys, dtype=torch.float32).reshape(-1, 1)

    groups = []
    if decision_focused:
        # Rows of the TRAINING set grouped by decision cell, with the true standardised value of
        # every action in that cell. The regret weights come from training data only.
        for cell in sorted(set(cells_tr)):
            idx = np.where(np.asarray(cells_tr) == cell)[0]
            groups.append((torch.tensor(idx, dtype=torch.long),
                           torch.tensor(ys[idx], dtype=torch.float32)))

    best, best_state, since = np.inf, None, 0
    for step in range(MAX_STEPS):
        opt.zero_grad()
        pred = net(tx)
        if decision_focused:
            loss = 0.0
            for idx, yv in groups:
                p = torch.softmax(pred[idx].squeeze(-1) / TAU, dim=0)
                loss = loss + (p * (yv.max() - yv)).sum()
            loss = loss / max(len(groups), 1)
        else:
            loss = torch.nn.functional.mse_loss(pred, ty)
        loss.backward()
        opt.step()
        v = float(loss)
        if v < best - 1e-9:
            best, best_state, since = v, {k: t.detach().clone()
                                          for k, t in net.state_dict().items()}, 0
        else:
            since += 1
            if since >= PATIENCE:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    with torch.no_grad():
        return net(torch.tensor(xs_all, dtype=torch.float32)).numpy().ravel() * sd + mu


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    ap.add_argument("--horizon-weeks", type=int, default=52)
    ap.add_argument("--output", type=Path,
                    default=Path(__file__).resolve().parent.parent / OUT)
    args = ap.parse_args()
    started = time.perf_counter()
    seeds = [SEED_BASE + i for i in range(args.seeds)]
    horizon = float(args.horizon_weeks * HOURS_PER_WEEK)

    cells, index = {}, []
    for family in FAMILIES:
        for escalation, mult in ESCALATIONS.items():
            for buf in BUFFER_HOURS:
                for seed in seeds:
                    agg, _ = episode(FAMILY_RISKS[family], mult, buf, seed, horizon)
                    cells[(family, escalation, buf, seed)] = agg
                    index.append((family, escalation, buf, seed))
        print(f"  {family} listo ({time.perf_counter() - started:.0f}s)", flush=True)

    aggs = [cells[k] for k in index]
    x_base = np.array([base_features(b, f, e) for (f, e, b, _) in index])
    x_rich = np.array([rich_features(b, f, e) for (f, e, b, _) in index])
    x_spl = spline_features([b for (_, _, b, _) in index], [f for (f, _, _, _) in index],
                            [e for (_, e, _, _) in index])
    g = np.array([s for (_, _, _, s) in index])
    cell_of = np.array([f"{f}|{e}" for (f, e, _, _) in index])
    buf_of = np.array([b for (_, _, b, _) in index])
    cell_names = sorted(set(cell_of))
    buf_levels = list(BUFFER_HOURS)

    def target_from_training(train_idx):
        maxima = {v: max(max(aggs[i][v] for i in train_idx), 1.0 + 1e-9)
                  for v in ("zeta", "epsilon", "phi", "tau")}
        maxima["kappa_dot"] = float(len(train_idx))
        exps = derive_exponents(maxima)
        total = float(sum(aggs[i]["kappa"] for i in train_idx))
        scale = float(len(train_idx)) / total if total > 0 else 1.0
        return np.array([resilience_index(
            {"zeta": a["zeta"], "epsilon": a["epsilon"], "phi": a["phi"], "tau": a["tau"],
             "kappa_dot": max(a["kappa"] * scale, 1e-9)}, exps)["R_cobb_douglas"] for a in aggs])

    arms = (*CLASSICAL, *NEURAL, FLOOR)
    regret_per_fold = {a: [] for a in arms}
    chosen_per_fold = {a: [] for a in arms}
    optimum_moves = []
    rng = np.random.default_rng(20260812)

    for fi, (tr, te) in enumerate(grouped_folds(g, n_folds=args.folds)):
        y = target_from_training(tr)
        tr_set, te_set = set(tr), set(te)

        # The TEST truth: mean resilience of every (cell, buffer) on held-out seeds.
        truth = np.full((len(cell_names), len(buf_levels)), np.nan)
        for ci, cname in enumerate(cell_names):
            for bi, b in enumerate(buf_levels):
                rows = [i for i in te_set if cell_of[i] == cname and buf_of[i] == b]
                if rows:
                    truth[ci, bi] = float(y[rows].mean())
        best_value = np.nanmax(truth, axis=1)
        optimum_moves.append(int(len(set(np.nanargmax(truth, axis=1).tolist()))))

        # One prediction per (cell, buffer) from each surrogate, trained on TRAIN rows only.
        grid_rows = [[next(i for i in range(len(index))
                           if cell_of[i] == c and buf_of[i] == b) for b in buf_levels]
                     for c in cell_names]

        def to_grid(pred_all):
            return np.array([[pred_all[r] for r in row] for row in grid_rows])

        preds = {}
        preds["linear_interactions"] = ols(x_rich[tr], y[tr], x_rich)
        preds["spline_buffer"] = ols(x_spl[tr], y[tr], x_spl)
        for name in ("gbdt", "random_forest", "gaussian_process", "kernel_ridge"):
            preds[name] = widened_predict(name, x_base[tr], y[tr], x_base)
        cells_tr = cell_of[tr]
        preds["mlp_mse"] = np.mean([_fit_net(x_base[tr], y[tr], cells_tr, x_base, 7000 + fi + 97 * k,
                                             False) for k in range(INIT_SEEDS)], axis=0)
        preds["mlp_decision"] = np.mean([_fit_net(x_base[tr], y[tr], cells_tr, x_base,
                                                  7000 + fi + 97 * k, True)
                                         for k in range(INIT_SEEDS)], axis=0)
        preds[FLOOR] = rng.normal(size=len(index))

        for arm in arms:
            picks = choose(arm, to_grid(preds[arm]))
            regret = float(np.nanmean([best_value[ci] - truth[ci, picks[ci]]
                                       for ci in range(len(cell_names))]))
            regret_per_fold[arm].append(regret)
            chosen_per_fold[arm].append([buf_levels[p] for p in picks])
        print(f"  fold {fi} listo ({time.perf_counter() - started:.0f}s)", flush=True)

    mean_regret = {a: float(np.mean(v)) for a, v in regret_per_fold.items()}
    t_crit = {4: 2.776, 3: 3.182, 2: 4.303}.get(args.folds - 1, 2.776)

    def paired(a, b):
        """Positive means `a` has LOWER regret than `b`, i.e. `a` chooses better."""
        d = np.array(regret_per_fold[b]) - np.array(regret_per_fold[a])
        se = float(d.std(ddof=1) / np.sqrt(d.size)) if d.size > 1 else 0.0
        return {"mean": float(d.mean()), "ci95_low": float(d.mean() - t_crit * se),
                "ci95_high": float(d.mean() + t_crit * se), "n_folds": int(d.size),
                "lcb_positive": bool(d.mean() - t_crit * se > 0.0)}

    best_classical = min(CLASSICAL, key=lambda m: mean_regret[m])
    best_neural = min(NEURAL, key=lambda m: mean_regret[m])
    vs = {"neural_vs_best_classical": paired(best_neural, best_classical),
          "decision_loss_vs_mse": paired("mlp_decision", "mlp_mse"),
          "best_neural_vs_floor": paired(best_neural, FLOOR),
          "best_classical_vs_floor": paired(best_classical, FLOOR)}

    checks = {
        "k1_the_decision_is_live": F.check(
            max(optimum_moves) > 1,
            "if one buffer level is optimal in all nine cells the argmax is trivial, every arm "
            "picks the same and there is nothing a surrogate could do better -- the degeneracy "
            "this project has already met as 'the optimum does not move'",
            computed_from={"max_distinct_optima_in_a_fold": max(optimum_moves),
                           "n_cells": len(cell_names)},
            distinct_optima_by_fold=optimum_moves),
        "k2_the_optimizer_is_identical": F.check(
            len(set(ARGMAX_CALLS.values())) == 1,
            "an arm that called the shared argmax more often than another would be searching more, "
            "not scoring better, and the whole ablation would be void",
            computed_from={"n_arms": len(ARGMAX_CALLS),
                           "distinct_call_counts": len(set(ARGMAX_CALLS.values()))},
            calls=ARGMAX_CALLS),
        "k3_the_decision_loss_helps_over_mse": F.check(
            vs["decision_loss_vs_mse"]["lcb_positive"],
            "same architecture, same budget, same features: if the decision-focused loss adds "
            "nothing over squared error, the idea is wrong and not merely insufficient",
            computed_from={"mean": vs["decision_loss_vs_mse"]["mean"],
                           "ci95_low": vs["decision_loss_vs_mse"]["ci95_low"]}),
        "k4_the_network_beats_the_best_classical": F.check(
            vs["neural_vs_best_classical"]["lcb_positive"],
            "the headline, and it is expected to fail: gbdt_lagged already beats the recurrent arm "
            "at prediction on this same surface",
            computed_from={"mean": vs["neural_vs_best_classical"]["mean"],
                           "ci95_low": vs["neural_vs_best_classical"]["ci95_low"]},
            best_classical=best_classical, best_neural=best_neural),
        "k5_a_control_must_be_worse": F.check(
            vs["best_neural_vs_floor"]["lcb_positive"]
            or vs["best_classical_vs_floor"]["lcb_positive"],
            "if an unfitted random surrogate matches the fitted ones, the decision problem has no "
            "resolution and no comparison here means anything",
            computed_from={"neural_lcb": vs["best_neural_vs_floor"]["ci95_low"],
                           "classical_lcb": vs["best_classical_vs_floor"]["ci95_low"]}),
    }
    checks["custody"] = custody_falsifier(seeds, replay_of=REPLAY_OF)
    summary = F.summarise(checks)

    if not checks["k1_the_decision_is_live"]["passed"]:
        status = "DECISION_PROBLEM_IS_DEGENERATE_NOTHING_TO_MEASURE"
    elif not checks["k5_a_control_must_be_worse"]["passed"]:
        status = "BLOCKED_NO_RESOLUTION"
    elif checks["k4_the_network_beats_the_best_classical"]["passed"]:
        status = "DECISION_PREMIUM_FOR_THE_NEURAL_SURROGATE"
    elif checks["k3_the_decision_loss_helps_over_mse"]["passed"]:
        status = "DECISION_LOSS_HELPS_BUT_NOT_ENOUGH_TO_BEAT_THE_CLASSICAL"
    else:
        status = "NO_DECISION_PREMIUM"

    payload = {
        "schema_version": "program_n_phase3_decision_surrogate_v1", "claim_status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_role": "DECLARED_REPLAY",
        "scope": "DECLARED_REPLAY_OF_CONSUMED_TAPES_NO_NEW_SEEDS_OUTER_LOOP_DECISION_ONLY",
        "endpoint": "mean_decision_regret_on_R_cobb_douglas__lower_is_better",
        "seeds": seeds, "tau": TAU, "n_cells": len(cell_names), "n_actions": len(buf_levels),
        "mean_regret_by_arm": dict(sorted(mean_regret.items(), key=lambda kv: kv[1])),
        "regret_per_fold": regret_per_fold, "chosen_buffers_per_fold": chosen_per_fold,
        "best_classical": best_classical, "best_neural": best_neural, "comparisons": vs,
        "distinct_optima_by_fold": optimum_moves, "argmax_calls": ARGMAX_CALLS,
        "falsifiers": checks, "falsifier_summary": summary,
        "elapsed_seconds": time.perf_counter() - started, "contract_path": str(CONTRACT),
    }
    seal_and_write(payload, args.output, contract=CONTRACT,
                   reference=Path("results/program_n/gate_b_confirmation_v3/result.json"))

    print(f"\nveredicto: {status}\n")
    print("  regret medio de decision (menor es mejor):")
    for a, v in sorted(mean_regret.items(), key=lambda kv: kv[1]):
        tag = "  <- red" if a in NEURAL else ("  <- suelo" if a == FLOOR else "")
        print(f"    {a:22}{v:.6f}{tag}")
    print(f"\n  {best_neural} vs {best_classical}: {vs['neural_vs_best_classical']['mean']:+.6f} "
          f"[{vs['neural_vs_best_classical']['ci95_low']:+.6f}, "
          f"{vs['neural_vs_best_classical']['ci95_high']:+.6f}]")
    print(f"  decision vs mse:          {vs['decision_loss_vs_mse']['mean']:+.6f} "
          f"[{vs['decision_loss_vs_mse']['ci95_low']:+.6f}, "
          f"{vs['decision_loss_vs_mse']['ci95_high']:+.6f}]")
    print("\n  falsadores:")
    for k, v in checks.items():
        mark = "N/A " if v.get("not_applicable") else ("PASA" if v["passed"] else "FALLA")
        print(f"    {k:44} {mark}")
    print(f"\n  -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
