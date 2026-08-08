#!/usr/bin/env python3
"""Garrido's point 7: switch each risk on and off, escalate it, and plot what it does over time.

Contract: docs/PREREGISTRO_GARRIDO_R2_RANDOMIZED_BENCHMARK_V1.md section 7, as amended by
docs/ENMIENDA_ALCANCE_R1_R3_BENCHMARK_2026-08-08.md. Custody: declared replay, no fresh seeds.

WHAT HE ASKED FOR, literally: "activar/desactivar riesgos individualmente y graficar su
comportamiento temporal", with marginal contribution reported per risk.

THE LADDER IS UNIFORM AND FROZEN: off, source, x4, x16 frequency, applied to ONE risk at a time
with every other risk left at source. Uniform because calibrating a different multiplier per risk
so each fires a comparable number of times would be tuning the environment, and the exposure
disparity is itself the finding. R1 and R3 keep their distribution families and move only in
parameters, which is what the PI's clarification permits.

WHY x16 IS IN THE LADDER AT ALL. R21 fires ZERO times at source on seed 8600001 and five times at
x16: its source window reaches 16,128 h against a 26-week episode. Every earlier conclusion that
named R21 was measured where R21 does not occur, and a ladder that stopped at x4 would have
repeated that.

MARGINAL CONTRIBUTION is leave-one-out from a common baseline -- L*(all at source) minus L*(this
risk off, rest at source) -- so it is the risk's own contribution and not a comparison between two
differently-configured worlds.

The policy is held fixed at the train-selected calendar, so the sensitivity is about risks and not
about a policy reacting to them. Trajectories are recorded weekly and serialised in full.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.continuous_its_env import make_continuous_its_track_a_env  # noqa: E402
from supply_chain.seed_custody import custody_falsifier, module_manifest  # noqa: E402

R1 = ("R11", "R12", "R13", "R14")
R2 = ("R21", "R22", "R23", "R24")
ALL_RISKS = R1 + R2
MAX_STEPS, STEP_HOURS = 26, 168.0
PREFIX_WEEKS, K_BUFFER = 4, 11
CHOICE_WEEKS = list(range(PREFIX_WEEKS, MAX_STEPS))
POLICY_WEEKS = [CHOICE_WEEKS[i] for i in range(K_BUFFER)]     # the train-selected calendar
LEVELS = {"off": None, "source": 1.0, "x4": 4.0, "x16": 16.0}
#: The ladder the sensitivity CLAIM covers. x16 is run and reported, but it is a stress probe:
#: measured, it collapses R13 to a one-step episode at L* = 1.0 and drives R12's recorded events to
#: ZERO -- at p = 0.98 the binomial saturates, each event lasts 12 x 168 h, and none completes
#: inside a 26-week horizon, while events are recorded on completion. Escalating a BINOMIAL risk by
#: a frequency multiplier is not the same operation as escalating a uniform one, and claiming
#: across both would be comparing two different things.
CLAIMED_LEVELS = ("off", "source", "x4")
SEED_BLOCK = tuple(range(8600001, 8600013))
MODULES = ("supply_chain/supply_chain.py", "supply_chain/continuous_its_env.py",
           "supply_chain/arm_runner.py", "supply_chain/seed_custody.py")


def exposure(sim) -> float:
    horizon, start = float(sim.env.now), float(sim.warmup_time)
    num = den = 0.0
    for o in sim.orders:
        if bool(getattr(o, "metrics_excluded", False)):
            continue
        opt = float(getattr(o, "OPTj", 0.0) or 0.0)
        if opt < start:
            continue
        q = float(o.quantity or 0.0)
        due = opt + float(o.LTj or 0.0)
        end = float(o.OATj) if getattr(o, "OATj", None) is not None else horizon
        num += q * max(0.0, end - due)
        den += q * max(0.0, horizon - due)
    return num / den if den > 0 else 0.0


def on_hand(sim) -> float:
    total = 0.0
    for name in ("rations_al", "rations_sb", "rations_sb_dispatch",
                 "rations_cssu", "rations_theatre"):
        node = getattr(sim, name, None)
        if node is not None:
            total += float(node.level)
    return total


def play(target: str | None, level: str, seed: int) -> dict:
    """One episode with `target` set to `level` and every other risk at source."""
    enabled = ALL_RISKS
    freq = {}
    if target is not None:
        if level == "off":
            enabled = tuple(r for r in ALL_RISKS if r != target)
        elif LEVELS[level] != 1.0:
            freq = {target: LEVELS[level]}
    kw = dict(init_frac=0.0, reward_mode="ReT_excel_delta", observation_version="v6",
              risk_level="current", enabled_risks=enabled, risk_rng_mode="per_risk",
              stochastic_pt=False, max_steps=MAX_STEPS, step_size_hours=STEP_HOURS,
              risk_obs=True, holding_cost=0.0, shift_cost=0.0)
    if freq:
        kw["risk_frequency_multipliers_by_id"] = freq
    env = make_continuous_its_track_a_env(**kw)
    env.reset(seed=int(seed))
    sim = env.unwrapped.sim
    weeks = set(POLICY_WEEKS)
    done = truncated = False
    step, seen = 0, 0
    traj = {"backlog": [], "on_hand": [], "events": [], "action": []}
    try:
        while not (done or truncated):
            on = step in weeks
            traj["action"].append(int(on))
            _o, _r, done, truncated, _i = env.step(
                np.array([1.0 if on else 0.0, -1.0], dtype=np.float32))
            now = len(getattr(sim, "risk_events", []) or [])
            traj["events"].append(now - seen)
            seen = now
            traj["backlog"].append(float(getattr(sim, "pending_backorder_qty", 0.0) or 0.0))
            traj["on_hand"].append(on_hand(sim))
            step += 1
        by_id: dict[str, int] = {}
        for e in getattr(sim, "risk_events", []) or []:
            rid = str(e.get("risk_id") if isinstance(e, dict) else getattr(e, "risk_id", "?"))
            by_id[rid] = by_id.get(rid, 0) + 1
        return {"L": exposure(sim), "events_by_id": by_id, "trajectory": traj,
                "n_steps": step, "completed_horizon": step >= MAX_STEPS}
    finally:
        env.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--replay-of", required=True)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--figure", type=Path,
                    default=Path("results/per_risk_sensitivity/per_risk_sensitivity.png"))
    ap.add_argument("--output", type=Path,
                    default=Path("results/per_risk_sensitivity/result.json"))
    args = ap.parse_args()
    started = time.perf_counter()
    seeds = list(SEED_BLOCK[:args.seeds])
    n_cfg = len(ALL_RISKS) * len(LEVELS) + 1
    print(f"  {n_cfg} configuraciones x {len(seeds)} semillas = {n_cfg * len(seeds)} episodios")

    base = [play(None, "source", s) for s in seeds]
    L_base = np.array([r["L"] for r in base])
    print(f"    baseline (todos en fuente) L* {L_base.mean():.6f}")

    cells = {}
    for risk in ALL_RISKS:
        for level in LEVELS:
            runs = [play(risk, level, s) for s in seeds]
            L = np.array([r["L"] for r in runs])
            own = float(np.mean([r["events_by_id"].get(risk, 0) for r in runs]))
            cells[f"{risk}|{level}"] = {
                "risk": risk, "level": level,
                "n_steps_mean": float(np.mean([r["n_steps"] for r in runs])),
                "completed_horizon": bool(all(r["completed_horizon"] for r in runs)),
                "L_mean": float(L.mean()), "L_by_seed": [float(x) for x in L],
                "own_events_mean": own,
                "delta_vs_baseline": float(L.mean() - L_base.mean()),
                "trajectory_seed0": runs[0]["trajectory"],
                "events_by_id_mean": {k: float(np.mean([r["events_by_id"].get(k, 0) for r in runs]))
                                      for k in ALL_RISKS},
            }
        m = cells[f"{risk}|off"]
        print(f"    {risk}: fuente {cells[f'{risk}|source']['own_events_mean']:6.2f} ev  "
              f"x4 {cells[f'{risk}|x4']['own_events_mean']:6.2f}  "
              f"x16 {cells[f'{risk}|x16']['own_events_mean']:6.2f}  "
              f"marginal {L_base.mean() - m['L_mean']:+.6f}")

    # MARGINAL = leave-one-out from the SAME baseline, so it is this risk's own contribution.
    marginal = {r: {"mean": float(L_base.mean() - cells[f"{r}|off"]["L_mean"]),
                    "escalation_x4": cells[f"{r}|x4"]["delta_vs_baseline"],
                    "escalation_x16": cells[f"{r}|x16"]["delta_vs_baseline"],
                    "exposure_source": cells[f"{r}|source"]["own_events_mean"],
                    "exposure_x16": cells[f"{r}|x16"]["own_events_mean"]}
                for r in ALL_RISKS}
    ranked = sorted(ALL_RISKS, key=lambda r: -marginal[r]["mean"])

    under_exposed = [r for r in ALL_RISKS if marginal[r]["exposure_source"] < 1.0]
    inert_even_at_x16 = [r for r in ALL_RISKS if marginal[r]["exposure_x16"] < 1.0]

    falsifiers = {
        "f1_off_really_disables_the_risk": {
            "passed": all(cells[f"{r}|off"]["own_events_mean"] == 0.0 for r in ALL_RISKS),
            "evidence": {"why_it_can_fail": "if a risk still fires when disabled, every marginal "
                                            "here is a difference between two worlds that both "
                                            "contain it",
                         "own_events_when_off": {r: cells[f"{r}|off"]["own_events_mean"]
                                                 for r in ALL_RISKS}}},
        "f2_escalation_raises_exposure_on_the_claimed_ladder": {
            "passed": all(cells[f"{r}|x4"]["own_events_mean"]
                          >= cells[f"{r}|source"]["own_events_mean"] for r in ALL_RISKS),
            "evidence": {"why_it_can_fail": "a frequency multiplier that does not raise realised "
                                            "events is inert, and the escalation column would be "
                                            "measuring nothing",
                         "claimed_ladder": list(CLAIMED_LEVELS),
                         "x16_is_a_stress_probe_not_a_claim": {
                             "R13": "collapses to a one-step episode at L* = 1.0",
                             "R12": ("recorded events fall to ZERO: at p = 0.98 the binomial "
                                     "saturates, each event lasts 12 x 168 h, and none completes "
                                     "inside 26 weeks -- events are recorded on completion")},
                         "exposure": {r: {lv: cells[f"{r}|{lv}"]["own_events_mean"]
                                          for lv in LEVELS} for r in ALL_RISKS}}},
        "f3_marginal_is_leave_one_out_from_one_baseline": {
            "passed": True, "not_applicable": False,
            "evidence": {"why_it_can_fail": "disclosure carried as a falsifier: every marginal is "
                                            "L*(all at source) minus L*(this risk off, rest at "
                                            "source), so it is the risk's own contribution rather "
                                            "than a contrast between two differently-configured "
                                            "worlds",
                         "baseline_L": float(L_base.mean()), "n_seeds": len(seeds)}},
        "f4_exposure_is_disclosed_per_risk": {
            "passed": True, "not_applicable": False,
            "evidence": {"why_it_can_fail": "disclosure. A marginal measured where the risk barely "
                                            "occurs says nothing about the risk, and every earlier "
                                            "conclusion in this repository that named R21 was "
                                            "measured exactly there",
                         "under_exposed_at_source": under_exposed,
                         "still_inert_at_x16": inert_even_at_x16,
                         "exposure_source": {r: marginal[r]["exposure_source"] for r in ALL_RISKS},
                         "exposure_x16": {r: marginal[r]["exposure_x16"] for r in ALL_RISKS}}},
        "f5_policy_is_held_fixed": {
            "passed": all(cells[f"{r}|{lv}"]["trajectory_seed0"]["action"]
                          == cells[f"{ALL_RISKS[0]}|source"]["trajectory_seed0"]["action"]
                          for r in ALL_RISKS for lv in CLAIMED_LEVELS),
            "evidence": {"why_it_can_fail": "if the action trajectory moved between cells, the "
                                            "sensitivity would mix the risk's effect with a policy "
                                            "responding to it",
                         "policy_weeks": POLICY_WEEKS,
                         "scoped_to": list(CLAIMED_LEVELS),
                         "collapsed_cells_excluded": [k for k, v in cells.items()
                                                      if not v["completed_horizon"]]}},
        "f6_trajectories_are_recorded": {
            "passed": all(len(cells[f"{r}|{lv}"]["trajectory_seed0"]["backlog"]) == MAX_STEPS
                          for r in ALL_RISKS for lv in CLAIMED_LEVELS),
            "evidence": {"why_it_can_fail": "Garrido asked for temporal behaviour, not a table; a "
                                            "missing series makes point 7 unanswered",
                         "series": ["backlog", "on_hand", "events", "action"],
                         "weeks": MAX_STEPS, "scoped_to": list(CLAIMED_LEVELS),
                         "episode_length_by_cell": {k: v["n_steps_mean"]
                                                    for k, v in cells.items()}}},
        "f7_no_fresh_seeds": custody_falsifier(seeds, replay_of=args.replay_of,
                                               exclude=args.output),
    }
    all_ok = all(v["passed"] for k, v in falsifiers.items()
                 if isinstance(v, dict) and not v.get("not_applicable"))
    falsifiers["all_passed"] = all_ok
    verdict = ("BLOCKED_INSTRUMENT" if not all_ok
               else "PER_RISK_SENSITIVITY_MEASURED_WITH_EXPOSURE_DISCLOSED")

    # ---- the figure Garrido asked for ------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
        weeks = np.arange(MAX_STEPS)
        ax = axes[0, 0]
        for r in ALL_RISKS:
            ax.plot(weeks, np.cumsum(cells[f"{r}|x4"]["trajectory_seed0"]["events"]),
                    label=r, lw=1.6)
        ax.set_title("Eventos acumulados por riesgo (x4, semilla 8600001)")
        ax.set_xlabel("semana"); ax.set_ylabel("eventos"); ax.legend(fontsize=7, ncol=2)
        ax = axes[0, 1]
        for r in ALL_RISKS:
            ax.plot(weeks, cells[f"{r}|x4"]["trajectory_seed0"]["backlog"], lw=1.4)
        ax.plot(weeks, cells[f"{ALL_RISKS[0]}|source"]["trajectory_seed0"]["backlog"],
                "k--", lw=1.8, label="fuente")
        ax.set_title("Backlog pendiente"); ax.set_xlabel("semana"); ax.legend(fontsize=7)
        ax = axes[1, 0]
        for r in ALL_RISKS:
            ax.plot(weeks, cells[f"{r}|x4"]["trajectory_seed0"]["on_hand"], lw=1.4)
        ax.set_title("Inventario disponible"); ax.set_xlabel("semana")
        ax = axes[1, 1]
        vals = [marginal[r]["mean"] for r in ranked]
        ax.barh(range(len(ranked)), vals)
        ax.set_yticks(range(len(ranked))); ax.set_yticklabels(ranked, fontsize=8)
        ax.invert_yaxis()
        ax.set_title("Contribución marginal a L* (leave-one-out)")
        ax.set_xlabel("L*(base) - L*(riesgo apagado)")
        fig.suptitle("Sensibilidad riesgo por riesgo — petición 7 de Garrido", fontsize=12)
        fig.tight_layout()
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.figure, dpi=150, bbox_inches="tight")
        fig.savefig(args.figure.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        figure_written = str(args.figure)
    except Exception as exc:                                    # noqa: BLE001
        figure_written = f"NOT_WRITTEN: {type(exc).__name__}: {exc}"

    print(f"\n  ranking por contribución marginal: {ranked}")
    print(f"  infra-expuestos en fuente (<1 evento): {under_exposed}")
    print(f"  inertes incluso a x16: {inert_even_at_x16 or 'ninguno'}")
    print(f"  figura: {figure_written}")
    print(f"\n  veredicto: {verdict}\n")
    for name, f in falsifiers.items():
        if name == "all_passed" or not isinstance(f, dict):
            continue
        label = ("NO APLICA" if f.get("not_applicable")
                 else "PASA" if f["passed"] else "FALLA")
        print(f"    {name:<48} {label}")

    payload = {
        "schema_version": "per_risk_sensitivity_v1",
        "claim_status": verdict,
        "scope": "DEVELOPMENT_ON_DECLARED_REPLAY_FIXED_POLICY",
        "run_role": "PER_RISK_SENSITIVITY", "replay_of": args.replay_of,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "module_manifest": module_manifest(MODULES, script=__file__),
        "answers": "Garrido 2026-08-07 request point 7: toggle each risk, escalate it, plot it",
        "ladder": {"levels": list(LEVELS), "uniform_across_risks": True,
                   "why_uniform": ("calibrating a different multiplier per risk so each fires a "
                                   "comparable number of times would be tuning the environment; "
                                   "the exposure disparity is itself the finding"),
                   "why_x16": ("R21 fires zero times at source on seed 8600001 and five at x16, "
                               "because its source window reaches 16,128 h against 26 weeks")},
        "policy": {"weeks_buffer_on": POLICY_WEEKS, "held_fixed": True,
                   "why": "so the sensitivity is about risks, not about a policy reacting to them"},
        "baseline_L": float(L_base.mean()), "baseline_L_by_seed": [float(x) for x in L_base],
        "cells": cells, "marginal": marginal, "ranked_by_marginal": ranked,
        "under_exposed_at_source": under_exposed, "inert_even_at_x16": inert_even_at_x16,
        "figure": figure_written, "seeds": seeds,
        "falsifiers": falsifiers, "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(payload, args.output, contract=args.contract,
                            reference=Path("results/kan_mlp_r2_benchmark_v2/result.json"))
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
