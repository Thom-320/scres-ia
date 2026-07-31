#!/usr/bin/env python3
"""Executes `docs/PREREGISTRO_ESPACIO_CONTINUO_2026-07-31.md`.

Is the 0.970 linear fit a property of Garrido's METRIC or of his GRID? On his 6x3 design a
linear model explains 0.970 of his own ReT and neither network beats it on his activation
question. He asked us on 2026-07-28 to add decision variables and to prefer CONTINUOUS ones, so
the test is to decouple what his design ties together and re-run the identical comparison.

The space stays inside his Table 6.16 ranges: the replenishment period becomes continuous in
[0, 1344] h and the three stock quantities vary INDEPENDENTLY instead of being pinned to one
index. Nothing else changes -- same models, same optimiser, same grouped CV, same acceptance
bar. If the surface turns non-linear, the networks win and the earlier conclusion was about his
grid. If it does not, the earlier conclusion is stronger, because it can no longer be blamed on
the design.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from build_garrido_fig5_surrogate import evaluate, grouped_folds  # noqa: E402
from supply_chain.config import HOURS_PER_WEEK  # noqa: E402
from supply_chain.config import THESIS_FAITHFUL_PROTOCOL as P  # noqa: E402
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402

FAMILIES = {"R1r": ("R11", "R12", "R13", "R14"), "R2r": ("R21", "R22", "R23", "R24"),
            "R3": ("R3",)}
# Table 6.16's maxima -- the ceiling of HIS design, never exceeded.
BOUNDS = {"period_hours": (0.0, 1344.0), "op3_rm": (0.0, 122_880.0),
          "op5_rm": (0.0, 122_880.0), "op9_rations": (0.0, 126_000.0)}
N_CONFIG = 384
ROOT_BASE = 4_100_001


def sample_design(n: int, seed: int) -> list[dict]:
    from scipy.stats import qmc

    engine = qmc.Sobol(d=4, scramble=True, seed=seed)
    unit = engine.random(n)
    rng = np.random.default_rng(seed)
    names = list(BOUNDS)
    rows = []
    for i in range(n):
        rho = {name: float(BOUNDS[name][0]
                           + unit[i, k] * (BOUNDS[name][1] - BOUNDS[name][0]))
               for k, name in enumerate(names)}
        family = FAMILIES and list(FAMILIES)[int(rng.integers(0, 3))]
        risks = FAMILIES[family]
        pattern = [bool(rng.integers(0, 2)) for _ in risks]
        rows.append({"cf": i + 1, "family": family, "shifts": int(rng.integers(1, 4)),
                     "pattern": "".join("+" if p else "-" for p in pattern),
                     "increased": [r for r, p in zip(risks, pattern) if p],
                     "rho": rho, "seed": ROOT_BASE + i})
    return rows


def run_configuration(row: dict, horizon: float) -> float:
    rho = row["rho"]
    period = float(rho["period_hours"])
    buffers = {"op3_rm": float(rho["op3_rm"]), "op5_rm": float(rho["op5_rm"]),
               "op9_rations": float(rho["op9_rations"])}
    sim = MFSCSimulation(
        shifts=row["shifts"], initial_buffers=buffers,
        inventory_replenishment_period=period, seed=row["seed"], horizon=horizon,
        risks_enabled=True, risk_level="current",
        enabled_risks=set(FAMILIES[row["family"]]),
        risk_overrides={r: "increased" for r in row["increased"]},
        strict_exogenous_crn=True, year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"], r14_defect_mode=P["r14_defect_mode"])
    sim.step(action=None, step_hours=horizon)
    return float(compute_episode_metrics(sim)["ret_excel"])


def features(rows: list[dict]) -> np.ndarray:
    out = []
    for r in rows:
        rho = r["rho"]
        pattern = r["pattern"].ljust(4, "0")
        out.append([
            rho["period_hours"] / BOUNDS["period_hours"][1],
            rho["op3_rm"] / BOUNDS["op3_rm"][1],
            rho["op5_rm"] / BOUNDS["op5_rm"][1],
            rho["op9_rations"] / BOUNDS["op9_rations"][1],
            (r["shifts"] - 1) / 2.0,
            *[1.0 if r["family"] == f else 0.0 for f in FAMILIES],
            *[1.0 if ch == "+" else 0.0 for ch in pattern],
        ])
    return np.asarray(out, dtype=np.float64)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=N_CONFIG)
    ap.add_argument("--seed", type=int, default=20260731)
    ap.add_argument("--horizon-weeks", type=int, default=52)
    ap.add_argument("--output", type=Path,
                    default=Path("results/garrido_continuous_space/result.json"))
    args = ap.parse_args()

    horizon = float(args.horizon_weeks * HOURS_PER_WEEK)
    started = time.perf_counter()
    rows = sample_design(args.n, args.seed)
    for i, row in enumerate(rows):
        row["ret_excel"] = run_configuration(row, horizon)
        if (i + 1) % 64 == 0:
            print(f"  {i + 1}/{len(rows)} ({time.perf_counter() - started:.0f}s)", flush=True)

    x = features(rows)
    y = np.array([r["ret_excel"] for r in rows])
    groups = np.array([r["seed"] for r in rows])
    b1 = evaluate(x, y, groups, classify=False, seed=args.seed)

    # B2: his activation question over consecutive configurations of the same family.
    pair_x, pair_y, pair_g = [], [], []
    for family in FAMILIES:
        idx = [i for i, r in enumerate(rows) if r["family"] == family]
        for a, b in zip(idx, idx[1:]):
            pair_x.append(np.concatenate([x[b], x[a]]))
            pair_y.append(1.0 if y[b] > y[a] else 0.0)
            pair_g.append(f"{groups[a]}|{groups[b]}")
    pair_x, pair_y = np.asarray(pair_x), np.asarray(pair_y)
    b2 = evaluate(pair_x, pair_y, np.asarray(pair_g), classify=True, seed=args.seed)

    distinct = {name: len({round(r["rho"][name], 6) for r in rows}) for name in BOUNDS}
    over = [r["cf"] for r in rows
            for name in BOUNDS
            if not (BOUNDS[name][0] - 1e-9 <= r["rho"][name] <= BOUNDS[name][1] + 1e-9)]

    falsifiers = {
        "f1_the_sample_covers_the_continuous_space": {
            "passed": all(v >= 100 for v in distinct.values()),
            "evidence": {"why_it_can_fail": ("a broken mapping would collapse the design back "
                                             "to his grid and the test would be vacuous"),
                         "distinct_values": distinct}},
        "f2_no_configuration_leaves_his_table_6_16_ranges": {
            "passed": not over,
            "evidence": {"why_it_can_fail": ("inventing physics outside his bounds would make "
                                             "the comparison illegitimate"),
                         "bounds": BOUNDS, "violations": over[:10]}},
        "f3_no_group_leaks": {
            "passed": all(not (set(groups[tr].tolist()) & set(groups[te].tolist()))
                          for tr, te in grouped_folds(groups)),
            "evidence": {"why_it_can_fail": "leakage inflates every score",
                         "n_groups": int(len(set(groups.tolist())))}},
        "f4_both_networks_train": {
            "passed": all(i["loss_last"] < i["loss_first"] for task in (b1, b2)
                          for name in ("backprop", "kan") for i in task["training"][name]),
            "evidence": {"why_it_can_fail": "a dead net scores like the baseline in silence"}},
        "f5_baselines_are_non_degenerate": {
            "passed": float(np.std(y)) > 0.0 and b2["mean"]["constant"] < 0.95,
            "evidence": {"why_it_can_fail": "a constant target makes every comparison empty",
                         "ret_sd": float(np.std(y)),
                         "majority_baseline": b2["mean"]["constant"]}},
    }
    falsifiers["all_passed"] = all(v["passed"] for k, v in falsifiers.items()
                                   if k != "all_passed")

    def beats(task, model):
        return (task["mean"][model] - task["mean"]["linear"]) > task["sd"]["linear"]

    verdict = {"B1_regression": {m: beats(b1, m) for m in ("backprop", "kan")},
               "B2_activation": {m: beats(b2, m) for m in ("backprop", "kan")},
               "rule": "beat the linear baseline by more than one of its between-fold SDs"}

    for label, task in (("B1 regresion R2", b1), ("B2 activacion acc", b2)):
        print(f"\n  === {label} ===")
        for model in ("constant", "linear", "backprop", "kan"):
            print(f"    {model:<10} {task['mean'][model]:>8.4f} ± {task['sd'][model]:.4f}")
    print("\n  falsadores:")
    for name, check in falsifiers.items():
        if name != "all_passed":
            print(f"    {name:<48} {'PASA' if check['passed'] else 'FALLA'}")
    print(f"\n  ¿gana alguna red? B1 {verdict['B1_regression']}  B2 {verdict['B2_activation']}")

    payload = {
        "schema_version": "garrido_continuous_space_v1",
        "claim_status": ("DEVELOPMENT_CONTINUOUS_SPACE" if falsifiers["all_passed"]
                         else "HALTED_FALSIFIER_FAILED"),
        "contract": "docs/PREREGISTRO_ESPACIO_CONTINUO_2026-07-31.md",
        "question": ("is the 0.970 linear fit a property of his metric or of his grid?"),
        "space": BOUNDS, "n_configurations": len(rows),
        "horizon_hours": horizon,
        "task_B1_regression": b1, "task_B2_activation": b2, "verdict": verdict,
        "grid_comparison": {
            "his_grid_B1_linear": 0.9697, "his_grid_B1_kan": 0.9913,
            "his_grid_B2_logistic": 0.7111, "his_grid_B2_kan": 0.7711,
            "source": "results/garrido_fig5_surrogate/result.json"},
        "rows": rows, "falsifiers": falsifiers, "seed": args.seed,
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
