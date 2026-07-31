#!/usr/bin/env python3
"""Which resilience metric shows the most headroom -- measured, on one set of runs.

`H_regime` came out at 1.8e-4 on `ret_excel`. That could be the system, or it could be the
METRIC: `ret_excel` is heavy-tailed (its Sobol indices on the raw scale came back outside
[0,1]), it collapses to `0.5/RPj` whenever a risk touches an order, and under R1r every single
order lands in that branch. So the same runs are re-scored under every resilience metric the
panel offers, plus the Cobb-Douglas index of his IJPR 2024 paper, plus the CVaR tails.

**The comparison must be normalised or it is meaningless.** A metric with a wider range shows a
larger raw headroom for free. Every `H_regime` is therefore also reported as `H / SD(metric)` --
headroom in units of the metric's own dispersion -- and that is the number the ranking uses.

**A cost correction.** The previous contract deferred Cobb-Douglas on an estimated ~300x cost.
Measured, the recorder sampled weekly costs **1.0x**: 0.054 s against 0.053 s for a single-step
episode. The deferral was based on a bad estimate, and this runner includes it.

**A cadence caveat, stated up front.** `ret_excel` is step-cadence dependent -- identical
trajectories score differently at different `step_hours`. Every cell here uses the SAME weekly
cadence, so the comparison is internally valid; the levels are not comparable with runs made at
another cadence, and `f1` measures that difference instead of hiding it.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.cobb_douglas_resilience import (  # noqa: E402
    CobbDouglasRecorder, score_comparison_set,
)
from supply_chain.config import HOURS_PER_WEEK  # noqa: E402
from supply_chain.config import THESIS_FAITHFUL_PROTOCOL as P  # noqa: E402
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402

R1R = ("R11", "R12", "R13", "R14")
R2R = ("R21", "R22", "R23", "R24")
R3 = ("R3",)
REGIMES = {"R1r": R1R, "R2r": R2R, "R3": R3, "R1r+R2r": R1R + R2R,
           "R1r+R3": R1R + R3, "R2r+R3": R2R + R3, "R1r+R2r+R3": R1R + R2R + R3}
PURE = ("R1r", "R2r", "R3")
LEVERS = {"op9_rop": (12.0, 48.0), "op10_q_max": (1_200.0, 5_200.0),
          "op12_q_max": (1_200.0, 5_200.0)}
SEED_BASE = 4_400_001
PERIOD_HOURS = float(HOURS_PER_WEEK)
# Every resilience-shaped column the panel exposes, plus the tails the PI asked for.
PANEL_METRICS = (
    "ret_excel", "ret_excel_full_ledger", "ret_excel_visible_clipped_0_1",
    "ret_thesis", "ret_continuous", "ration_ret_excel",
    "ret_excel_cvar05", "ret_excel_cvar10", "ret_excel_p05", "ret_excel_p10",
    "ret_excel_p50", "ret_excel_risk_conditional",
    "ret_excel_rolling_4w_min", "ret_excel_rolling_4w_mean",
    "flow_fill_rate",
)
COBB = "cobb_douglas_index"
CALIBRATION = Path("results/cobb_douglas/score_v1.json")


def grid(levels: int) -> list[dict[str, float]]:
    axes = {n: np.linspace(lo, hi, levels) for n, (lo, hi) in LEVERS.items()}
    return [dict(zip(axes, v)) for v in itertools.product(*axes.values())]


def action_for(setting: dict[str, float]) -> dict[str, float]:
    return {"op9_rop": setting["op9_rop"], "op9_q_min": 2_400.0, "op9_q_max": 2_600.0,
            "op10_rop": 24.0, "op10_q_min": setting["op10_q_max"] * 0.92,
            "op10_q_max": setting["op10_q_max"], "op12_rop": 24.0,
            "op12_q_min": setting["op12_q_max"] * 0.92,
            "op12_q_max": setting["op12_q_max"]}


def run(setting: dict[str, float], risks: tuple[str, ...], seed: int, horizon: float,
        *, weekly: bool = True) -> tuple[dict[str, float], dict[str, float] | None]:
    sim = MFSCSimulation(
        shifts=1, initial_buffers={n: 0.0 for n in ("op3_rm", "op5_rm", "op9_rations")},
        inventory_replenishment_period=0.0, seed=seed, horizon=horizon,
        risks_enabled=True, risk_level="current", enabled_risks=set(risks),
        strict_exogenous_crn=True, year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"], r14_defect_mode=P["r14_defect_mode"])
    if not weekly:
        sim.step(action=action_for(setting), step_hours=horizon)
        return compute_episode_metrics(sim), None
    recorder = CobbDouglasRecorder(period_hours=PERIOD_HOURS)
    steps = int(round(horizon / PERIOD_HOURS))
    for index in range(steps):
        sim.step(action=action_for(setting) if index == 0 else None,
                 step_hours=PERIOD_HOURS)
        recorder.sample(sim)
    return compute_episode_metrics(sim), recorder.aggregate()


def headroom(block: np.ndarray, *, n_boot: int = 2000, seed: int = 0) -> dict:
    """`H_regime` on one metric: knowing the regime, minus one setting for all of them."""
    def point(sample: np.ndarray) -> float:
        per_regime = sample.mean(axis=2)
        return float(per_regime.max(axis=0).mean() - per_regime.mean(axis=1).max())

    rng = np.random.default_rng(seed)
    n_seeds = block.shape[2]
    boot = [point(block[:, :, rng.integers(0, n_seeds, n_seeds)]) for _ in range(n_boot)]
    return {"H_regime": point(block), "lcb95": float(np.percentile(boot, 5)),
            "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--levels", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--horizon-weeks", type=int, default=52)
    ap.add_argument("--output", type=Path,
                    default=Path("results/sensitivity/multi_metric_headroom_v1/result.json"))
    args = ap.parse_args()
    horizon = float(args.horizon_weeks * HOURS_PER_WEEK)
    settings = grid(args.levels)
    names = list(REGIMES)
    started = time.perf_counter()

    panels: dict[str, np.ndarray] = {m: np.zeros((len(settings), len(names), args.seeds))
                                     for m in PANEL_METRICS}
    cd_aggregates: dict[tuple[int, int, int], dict] = {}
    for s_i, setting in enumerate(settings):
        for r_i, label in enumerate(names):
            for k in range(args.seeds):
                metrics, agg = run(setting, REGIMES[label], SEED_BASE + k, horizon)
                for m in PANEL_METRICS:
                    panels[m][s_i, r_i, k] = float(metrics[m])
                cd_aggregates[(s_i, r_i, k)] = agg
        if (s_i + 1) % 25 == 0:
            print(f"  {s_i + 1}/{len(settings)} ({time.perf_counter() - started:.0f}s)",
                  flush=True)

    # --- Cobb-Douglas. The exponents come from the FROZEN calibration, not from this sweep.
    # Two reasons, and the first is decisive: `derive_exponents` refuses a maximum at or below
    # 1, and `tau` maxes at 0.28 here, so a sweep-local derivation is not even defined. The
    # second is that re-deriving exponents per experiment would make the index incomparable
    # between experiments, which defeats the point of having a second metric.
    exponents = dict(json.loads(CALIBRATION.read_text())["exponents"])
    cd = np.zeros((len(settings), len(names), args.seeds))
    for r_i in range(len(names)):
        for k in range(args.seeds):
            comparison = {str(s_i): cd_aggregates[(s_i, r_i, k)] for s_i in range(len(settings))}
            scored = score_comparison_set(comparison, exponents)
            for s_i in range(len(settings)):
                cd[s_i, r_i, k] = float(scored[str(s_i)]["R_cobb_douglas"])
    panels[COBB] = cd

    pure_idx = [names.index(r) for r in PURE]
    mixed_idx = [i for i, r in enumerate(names) if r not in PURE]
    table: dict[str, dict] = {}
    for metric, block in panels.items():
        sd = float(np.std(block.mean(axis=2)))
        rows = {"sd": sd}
        for label, idx in (("pure", pure_idx), ("mixed", mixed_idx),
                           ("all_seven", list(range(len(names))))):
            h = headroom(block[:, idx, :], seed=hash(metric + label) % 2**31)
            rows[label] = {**h,
                           "H_over_sd": (h["H_regime"] / sd) if sd > 0 else float("nan"),
                           "lcb95_over_sd": (h["lcb95"] / sd) if sd > 0 else float("nan")}
        table[metric] = rows

    ranked = sorted(table, key=lambda m: -table[m]["all_seven"]["H_over_sd"])

    # --- falsifiers ---
    single, _ = run(settings[0], REGIMES["R1r"], SEED_BASE, horizon, weekly=False)
    cadence_gap = abs(float(single["ret_excel"]) - float(panels["ret_excel"][0, 0, 0]))
    repeat, _ = run(settings[0], REGIMES["R1r"], SEED_BASE, horizon)
    falsifiers = {
        "f1_cadence_effect_is_measured_not_hidden": {
            "passed": True,
            "evidence": {"why_this_is_recorded_not_gated": (
                "ret_excel is step-cadence dependent by a known defect; every cell here uses "
                "the same weekly cadence so the comparison is internally valid, and the size "
                "of the cadence difference is reported rather than assumed away"),
                "weekly": float(panels["ret_excel"][0, 0, 0]),
                "single_step": float(single["ret_excel"]), "absolute_gap": cadence_gap}},
        "f2_cobb_douglas_is_well_defined": {
            "passed": all(np.isfinite(v) for v in exponents.values()) and bool(np.isfinite(cd).all()),
            "evidence": {"why_it_can_fail": ("the exponent rule 0.20/ln(x_max) is undefined for "
                                             "a maximum at or below 1"),
                         "exponents": exponents,
                         "source": str(CALIBRATION)}},
        "f3_headroom_is_non_negative_for_every_metric": {
            "passed": all(table[m][s]["H_regime"] >= -1e-12
                          for m in table for s in ("pure", "mixed", "all_seven")),
            "evidence": {"why_it_can_fail": "a negative value would betray a bad aggregation"}},
        "f4_crn_is_real": {
            "passed": abs(float(repeat["ret_excel"]) - float(panels["ret_excel"][0, 0, 0]))
            < 1e-12,
            "evidence": {"why_it_can_fail": "without pairing the headroom is seed noise"}},
        "f5_normaliser_is_non_degenerate": {
            "passed": all(table[m]["sd"] > 0 for m in table),
            "evidence": {"why_it_can_fail": ("a metric constant across the grid would divide "
                                             "by zero and top the ranking for free"),
                         "sd_by_metric": {m: table[m]["sd"] for m in table}}},
    }
    falsifiers["all_passed"] = all(v["passed"] for k, v in falsifiers.items()
                                   if k != "all_passed")

    print(f"\n  === headroom por métrica (H / SD propia, los 7 regímenes) ===")
    print(f"  {'métrica':<34}{'H':>12}{'LCB95':>12}{'H/SD':>9}{'SD':>12}")
    for m in ranked:
        row = table[m]["all_seven"]
        print(f"  {m:<34}{row['H_regime']:>12.6f}{row['lcb95']:>12.6f}"
              f"{row['H_over_sd']:>9.3f}{table[m]['sd']:>12.6f}")
    print("\n  falsadores:")
    for name, check in falsifiers.items():
        if name != "all_passed":
            print(f"    {name:<46} {'PASA' if check['passed'] else 'FALLA'}")

    payload = {
        "schema_version": "multi_metric_headroom_v1",
        "claim_status": ("DEVELOPMENT_MULTI_METRIC_HEADROOM" if falsifiers["all_passed"]
                         else "HALTED_FALSIFIER_FAILED"),
        "question": ("is the 1.8e-4 headroom the system or the metric? Same runs, every "
                     "resilience metric the panel offers plus Cobb-Douglas and the CVaR tails"),
        "normalisation": ("H / SD(metric) -- a metric with a wider range would otherwise show a "
                          "larger raw headroom for free"),
        "regimes": {k: list(v) for k, v in REGIMES.items()},
        "levers": {k: list(v) for k, v in LEVERS.items()},
        "cadence_hours": PERIOD_HOURS, "levels": args.levels, "seeds": args.seeds,
        "n_runs": int(len(settings) * len(names) * args.seeds),
        "cobb_douglas_exponents": exponents,
        "headroom_by_metric": table, "ranking_by_H_over_sd": ranked,
        "falsifiers": falsifiers,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(
        payload, args.output,
        contract=Path("docs/PREREGISTRO_MEZCLA_RIESGOS_2026-07-31.md"),
        reference=Path("results/sensitivity/mixed_risk_downstream_v1/result.json"))
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
