#!/usr/bin/env python3
"""H1 and H3 as the v.0 draft actually words them, third attempt, on the n=120 block.

WHAT THE FIRST TWO ATTEMPTS ESTABLISHED, AND WHY THIS IS NOT A RE-RUN OF EITHER.

v1 (2026-08-01) halted: the arms deployed the SAME configuration, so both hypotheses were empty
by construction, and `system_ttr` was right-censored at 1.000 in every arm, so H1 had no
estimand at all. v2 answered two DECLARED reformulations -- H1' on cumulative lost service and
H3' on the variance of search cost -- and both are now supported and are not reopened here.

Two things changed since, and neither is a loosened criterion:

  * v1 collapsed each arm to its MODAL configuration over 12 replicates. Over the 120-replicate
    power block the arms deploy 21 / 43 / 33 / 87 distinct configurations, so the estimand exists
    per cell. This runner never takes a mode.
  * `restricted_ttr = min(TTR, tau)` with a paired placebo was written on 2026-08-06 in
    supply_chain/garrido_v0_recovery.py, for the v0 lane and before this preregistration. An
    absorbed shock scores 0 and an unrestored one scores tau, so censoring can no longer
    manufacture a fast arm -- which is exactly how `system_ttr` produced 1.000 everywhere.

It is still a redefinition of the endpoint and the manuscript has to say so.

WHY H1 IS MEASURED UNDER ISOLATED SHOCKS. Under the recurrent R11-R24 regime at 52 weeks the
events merge into a single cluster that never ends, so there is no return-to-normal to time. That
is a property of Garrido's own risk regime, not a defect of the instrument, and it is reported
whichever way the contrast falls.

Preregistration: docs/PREREGISTRO_H1_H3_ORIGINALES_V3_2026-08-07.md
Development. Seeds 6_000_001-120 are an already-open block; no virgin seed is consumed.
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

from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.config import HOURS_PER_WEEK  # noqa: E402
from supply_chain.config import THESIS_FAITHFUL_PROTOCOL as P  # noqa: E402
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.garrido_v0_recovery import (  # noqa: E402
    CONTEXT_ORDER, EVENT_ONSET_HOURS, RECOVERY_CONSECUTIVE_DAYS, RECOVERY_FRACTION,
    RECOVERY_WINDOW_HOURS, placebo_event_rows, restricted_recovery_summary, risk_event_rows,
)
from supply_chain.resilience_temporal import compute_temporal_resilience_panel  # noqa: E402
from supply_chain.seed_custody import module_manifest  # noqa: E402
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402

# Imported, not re-implemented: f4 compares against values this exact function produced.
from run_meta_learner_over_configs_v1 import evaluate as meta_evaluate  # noqa: E402

SLICES = (
    Path("results/garrido_meta_learner_h3power_h3_contract_local_v2/result.json"),
    Path("results/garrido_meta_learner_h3power_h3_contract_vps_v2/result.json"),
)
ARMS = {"hybrid": "neuron_memory", "static": "ofat", "reset": "neuron_reset"}
DESCRIPTIVE_ARM = ("random", "random")
# Frozen in v1 and validated there by its own f2; not re-chosen after seeing anything.
INTENSITIES = (1.0, 2.0, 3.0, 4.0)
BASE_CONTEXTS = ("R1r", "R2r", "R1r+R2r")
R1R = ("R11", "R12", "R13", "R14")
R2R = ("R21", "R22", "R23", "R24")
CONTEXT_RISKS = {"R1r": R1R, "R2r": R2R, "R1r+R2r": R1R + R2R}
# Fixed in the preregistration: R11-R14 use the configuration deployed under R1r, R21-R24 under
# R2r. No other pairing is tried.
SHOCK_TO_META = {r: ("R1r" if r in R1R else "R2r") for r in R1R + R2R}
H1_HORIZON = 36.0 * HOURS_PER_WEEK          # the v0 grid's frozen horizon
H3_HORIZON = 52.0 * HOURS_PER_WEEK          # the meta-learner's own horizon
N_BOOT = 5_000
MODULES = ("supply_chain/arm_runner.py", "supply_chain/garrido_v0_recovery.py",
           "supply_chain/resilience_temporal.py", "supply_chain/episode_metrics.py",
           "supply_chain/supply_chain.py")


def cfg_key(config: dict) -> tuple:
    return tuple(sorted((k, float(v)) for k, v in config.items()))


def build_sim(config: dict, seed: int, horizon: float, **kw) -> MFSCSimulation:
    """One simulator under a meta-learner configuration. The decision variables enter exactly as
    they do in run_meta_learner_over_configs_v1.evaluate; only the risk source differs."""
    sim = MFSCSimulation(
        shifts=int(config["shifts"]),
        initial_buffers={"op3_rm": 0.0, "op5_rm": 0.0,
                         "op9_rations": float(config["buffer_hours"]) * 2_500.0 / 24.0},
        inventory_replenishment_period=0.0, seed=int(seed), horizon=float(horizon),
        strict_exogenous_crn=True, year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"], r14_defect_mode=P["r14_defect_mode"], **kw)
    sim.params["op9_rop"] = float(config["op9_rop"])
    sim.params["op12_rop"] = float(config["op12_rop"])
    return sim


def isolated_episode(config: dict, seed: int, events: list) -> dict:
    """One isolated-shock (or placebo) episode with the temporal recovery panel."""
    sim = build_sim(config, seed, H1_HORIZON, risks_enabled=True, enabled_risks=set(),
                    risk_event_tape=events)
    sim.step(action=None, step_hours=H1_HORIZON)
    temporal = compute_temporal_resilience_panel(
        sim, cluster_window_hours=RECOVERY_WINDOW_HOURS, recovery_fraction=RECOVERY_FRACTION,
        recovery_consecutive_days=RECOVERY_CONSECUTIVE_DAYS)
    episode = compute_episode_metrics(sim)
    return {"temporal": temporal,
            "physical_risk_events": sum(1 for e in sim.risk_events
                                        if str(getattr(e, "risk_id", None)
                                               or (e.get("risk_id") if isinstance(e, dict)
                                                   else "")) != "PLACEBO"),
            "demanded_rations": float(episode["demanded_rations"]),
            "flow_fill_rate": float(episode["flow_fill_rate"])}


def ladder_episode(config: dict, context: str, intensity: float, seed: int) -> dict:
    """One rung of the intensity ladder, under the recurrent regime the arms were chosen in."""
    risks = CONTEXT_RISKS[context]
    sim = build_sim(config, seed, H3_HORIZON, risks_enabled=True, risk_level="current",
                    enabled_risks=set(risks),
                    risk_frequency_multipliers_by_id={r: float(intensity) for r in risks})
    sim.run()
    panel = compute_episode_metrics(sim)
    return {"service_loss_auc": float(panel["service_loss_auc_ration_hours"]),
            "ret_excel": float(panel["ret_excel_risk_conditional"]),
            "n_risk_events": float(len(sim.risk_events))}


def boot_paired(diff: np.ndarray, rng: np.random.Generator) -> dict:
    """Bootstrap over CELLS. Resampling observations was the 1 August audit's fourth defect."""
    draws = diff[rng.integers(0, diff.size, size=(N_BOOT, diff.size))].mean(axis=1)
    return {"mean": float(diff.mean()), "lcb95": float(np.percentile(draws, 2.5)),
            "ucb95": float(np.percentile(draws, 97.5)), "n_cells": int(diff.size),
            "p_one_sided": float(np.mean(draws <= 0.0))}


def holm(pvals: dict[str, float]) -> dict[str, dict]:
    order = sorted(pvals, key=lambda k: pvals[k])
    k, out, running = len(order), {}, 0.0
    for i, name in enumerate(order):
        adj = min(1.0, max(running, pvals[name] * (k - i)))
        running = adj
        out[name] = {"p_raw": pvals[name], "p_holm": adj, "rejected_at_05": adj < 0.05}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed-limit", type=int, default=0,
                    help="use only the first N seeds (smoke runs); 0 = the full 120")
    ap.add_argument("--f4-sample", type=int, default=24,
                    help="sealed cells re-evaluated against their stored value")
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--output", type=Path,
                    default=Path("results/manuscript/h1_h3_originales_v3/result.json"))
    args = ap.parse_args()
    started = time.perf_counter()
    rng = np.random.default_rng(20260807)

    # ---- the arms: per-cell deployed configurations, never a mode ------------------------------
    seeds, deployed, sealed_meta = [], {}, []
    for path in SLICES:
        d = json.loads(path.read_text())
        sealed_meta.append({"path": str(path), "self_sha256": d.get("self_sha256"),
                            "contract_sha256": d.get("contract_sha256"),
                            "repeats": d.get("repeats")})
        for i, seed in enumerate(d["seeds"]):
            seeds.append(int(seed))
            for arm, strategy in list(ARMS.items()) + [DESCRIPTIVE_ARM]:
                run = d["per_context"][strategy][i]
                for ctx in d["contexts"]:
                    deployed[(arm, int(seed), ctx)] = (dict(run[ctx]["chosen_config"]),
                                                       float(run[ctx]["chosen_value"]))
    if len(set(seeds)) != len(seeds):
        raise SystemExit("the two slices share a seed; they are not independent replicates")
    seeds = sorted(seeds)[: args.seed_limit or None]
    print(f"  {len(seeds)} semillas · {len(ARMS)} brazos + descriptivo · "
          f"{len(CONTEXT_ORDER)} choques aislados · escalera {INTENSITIES}", flush=True)

    # ---- f4 FIRST: if the physics drifted, nothing below means anything -----------------------
    cells = [(a, s, c) for (a, s, c) in deployed if s in set(seeds) and a in ARMS]
    sample = [cells[i] for i in rng.choice(len(cells), size=min(args.f4_sample, len(cells)),
                                           replace=False)]
    f4_rows, f4_ok = [], True
    for arm, seed, ctx in sample:
        config, stored = deployed[(arm, seed, ctx)]
        got, _ = meta_evaluate(config, ctx, seed, H3_HORIZON)
        match = abs(got - stored) <= 1e-12
        f4_ok &= match
        f4_rows.append({"arm": arm, "seed": seed, "context": ctx, "stored": stored,
                        "reproduced": got, "abs_delta": abs(got - stored), "match": match})
    print(f"  f4 {'PASA' if f4_ok else 'FALLA'}: {sum(r['match'] for r in f4_rows)}/"
          f"{len(f4_rows)} celdas selladas reproducidas ({time.perf_counter()-started:.0f}s)",
          flush=True)
    if not f4_ok:
        worst = max(f4_rows, key=lambda r: r["abs_delta"])
        payload = {
            "schema_version": "h1_h3_originales_v3",
            "claim_status": "HALTED_SEALED_SURFACE_NO_LONGER_REPRODUCES",
            "scope": "DEVELOPMENT_HALTED_BEFORE_ANY_ENDPOINT_WAS_COMPUTED",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "module_manifest": module_manifest(MODULES, script=__file__),
            "preregistration": str(args.contract),
            "why": ("The n=120 block was sealed under a different physics than the one installed "
                    "today, so no endpoint computed from its deployed configurations would be "
                    "comparable to it. This is gap A2 of docs/REGISTRO_DE_HUECOS_2026-08-07.md "
                    "reaching a runner."),
            "source_slices": sealed_meta, "f4_rows": f4_rows, "worst_cell": worst,
            "elapsed_seconds": time.perf_counter() - started,
        }
        seal_and_write(payload, args.output, contract=args.contract, reference=SLICES[0])
        print(f"  -> DETENIDO. peor celda delta={worst['abs_delta']:.3e}")
        return 1

    # ---- H1: restricted TTR of what each arm deploys, under isolated shocks --------------------
    placebo_cache: dict[tuple, dict] = {}
    placebo_events = placebo_event_rows(onset_hours=EVENT_ONSET_HOURS)
    h1_cells: dict[str, dict[tuple, float]] = {a: {} for a in ARMS}
    h1_flags = {a: {"absorbed": 0, "censored": 0, "recovered": 0, "n": 0} for a in ARMS}
    h1_config = {}
    for si, seed in enumerate(seeds):
        for shock in CONTEXT_ORDER:
            meta_ctx = SHOCK_TO_META[shock]
            events = risk_event_rows(shock, onset_hours=EVENT_ONSET_HOURS)
            for arm in ARMS:
                config, _ = deployed[(arm, seed, meta_ctx)]
                key = (cfg_key(config), seed)
                if key not in placebo_cache:
                    placebo_cache[key] = isolated_episode(config, seed, placebo_events)
                placebo = placebo_cache[key]
                risk = isolated_episode(config, seed, events)
                rec = restricted_recovery_summary(risk["temporal"], placebo["temporal"],
                                                  tau_hours=RECOVERY_WINDOW_HOURS)
                h1_cells[arm][(seed, shock)] = float(rec["restricted_ttr_hours"])
                h1_flags[arm]["absorbed"] += int(bool(rec["absorbed"]))
                h1_flags[arm]["censored"] += int(bool(rec["right_censored_at_tau"]))
                h1_flags[arm]["recovered"] += int(bool(rec["recovered_within_tau"]))
                h1_flags[arm]["n"] += 1
                h1_config[(arm, seed, shock)] = cfg_key(config)
                if risk["physical_risk_events"] == 0:
                    raise SystemExit(f"shock {shock} produced no physical event")
                if placebo["physical_risk_events"] != 0:
                    raise SystemExit("placebo carried a physical risk event")
        if (si + 1) % 10 == 0:
            print(f"  H1 {si+1}/{len(seeds)} semillas ({time.perf_counter()-started:.0f}s)",
                  flush=True)

    # ---- H3: variance ACROSS the intensity ladder, per (seed, base context) --------------------
    h3_cells = {a: {} for a in ARMS}
    h3_ret = {a: {} for a in ARMS}
    ladder_events = {i: [] for i in INTENSITIES}
    within_rung: list[float] = []
    for si, seed in enumerate(seeds):
        for ctx in BASE_CONTEXTS:
            for arm in ARMS:
                config, _ = deployed[(arm, seed, ctx)]
                rows = [ladder_episode(config, ctx, i, seed) for i in INTENSITIES]
                for i, row in zip(INTENSITIES, rows):
                    ladder_events[i].append(row["n_risk_events"])
                h3_cells[arm][(seed, ctx)] = float(np.var([r["service_loss_auc"] for r in rows],
                                                          ddof=1))
                h3_ret[arm][(seed, ctx)] = float(np.var([r["ret_excel"] for r in rows], ddof=1))
        if (si + 1) % 10 == 0:
            print(f"  H3 {si+1}/{len(seeds)} semillas ({time.perf_counter()-started:.0f}s)",
                  flush=True)
    # f6's control: the same quantity taken along the WRONG axis, so it can be shown to differ.
    for arm in ARMS:
        for ctx in BASE_CONTEXTS:
            within_rung.append(float(np.var([h3_cells[arm][(s, ctx)] for s in seeds], ddof=1)))

    # ---- contrasts, declared family, Holm ------------------------------------------------------
    h1_keys = sorted(h1_cells["hybrid"])
    h3_keys = sorted(h3_cells["hybrid"])

    def arr(store, arm, keys):
        return np.array([store[arm][k] for k in keys], dtype=float)

    # Positive = the hybrid is better (recovers sooner / varies less).
    contrasts = {
        "H1_hybrid_vs_static": boot_paired(arr(h1_cells, "static", h1_keys)
                                           - arr(h1_cells, "hybrid", h1_keys), rng),
        "H1_hybrid_vs_reset": boot_paired(arr(h1_cells, "reset", h1_keys)
                                          - arr(h1_cells, "hybrid", h1_keys), rng),
        "H3_hybrid_vs_static": boot_paired(arr(h3_cells, "static", h3_keys)
                                           - arr(h3_cells, "hybrid", h3_keys), rng),
        "H3_hybrid_vs_reset": boot_paired(arr(h3_cells, "reset", h3_keys)
                                          - arr(h3_cells, "hybrid", h3_keys), rng),
    }
    holm_table = holm({k: v["p_one_sided"] for k, v in contrasts.items()})

    differing = [k for k in h1_keys
                 if h1_config[("hybrid", *k)] != h1_config[("static", *k)]]
    differing_share = len(differing) / len(h1_keys)
    secondary = {
        "H1_hybrid_vs_static_on_differing_cells": boot_paired(
            np.array([h1_cells["static"][k] - h1_cells["hybrid"][k] for k in differing]), rng)
        if differing else None,
        "H3_ret_excel_not_primary_rewards_abandonment": {
            arm: float(np.mean([h3_ret[arm][k] for k in h3_keys])) for arm in ARMS},
    }
    levels = {
        "H1_restricted_ttr_hours": {a: float(np.mean(list(h1_cells[a].values()))) for a in ARMS},
        "H3_variance_service_loss_auc": {a: float(np.mean(list(h3_cells[a].values())))
                                         for a in ARMS},
    }

    h1_sup = (contrasts["H1_hybrid_vs_static"]["lcb95"] > 0
              and holm_table["H1_hybrid_vs_static"]["rejected_at_05"])
    h3_sup = (contrasts["H3_hybrid_vs_static"]["lcb95"] > 0
              and holm_table["H3_hybrid_vs_static"]["rejected_at_05"])

    absorbed_frac = sum(f["absorbed"] for f in h1_flags.values()) / sum(f["n"] for f
                                                                       in h1_flags.values())
    censored_frac = sum(f["censored"] for f in h1_flags.values()) / sum(f["n"] for f
                                                                       in h1_flags.values())
    mean_events = {str(i): float(np.mean(ladder_events[i])) for i in INTENSITIES}

    falsifiers = {
        "f1_the_arms_deploy_different_configurations": {
            "passed": bool(differing_share >= 0.30),
            "evidence": {"why_it_can_fail": "if hybrid and static deploy the same configuration "
                                            "the hypotheses are empty by construction, which is "
                                            "exactly how the v1 attempt halted",
                         "differing_share": differing_share,
                         "n_differing_cells": len(differing), "n_cells": len(h1_keys),
                         "threshold": 0.30}},
        "f2_the_recovery_endpoint_has_range": {
            "passed": bool(absorbed_frac <= 0.999 and censored_frac <= 0.999),
            "evidence": {"why_it_can_fail": "an endpoint that is absorbed everywhere, or censored "
                                            "everywhere, is identical across arms precisely "
                                            "because it measures nothing -- this is the criterion "
                                            "that killed system_ttr at 1.000",
                         "absorbed_fraction": absorbed_frac,
                         "censored_at_tau_fraction": censored_frac,
                         "per_arm": h1_flags}},
        "f3_the_placebo_is_really_shock_free": {
            "passed": True,
            "evidence": {"why_it_can_fail": "a placebo carrying a real risk event, or an impact "
                                            "decided without the incremental comparison, would "
                                            "let routine backlog masquerade as shock impact",
                         "checked_every_episode": True,
                         "n_placebo_episodes": len(placebo_cache),
                         "impact_rule": "excess AUC or excess max drop over the paired placebo"}},
        "f4_the_sealed_surface_still_reproduces": {
            "passed": bool(f4_ok),
            "evidence": {"why_it_can_fail": "if supply_chain.py drifted since the block was "
                                            "sealed, the deployed configurations are not the "
                                            "ones this artifact names",
                         "n_sampled": len(f4_rows), "tolerance": 1e-12,
                         "max_abs_delta": max(r["abs_delta"] for r in f4_rows)}},
        "f5_the_intensity_ladder_escalates": {
            "passed": bool(mean_events[str(max(INTENSITIES))]
                           > mean_events[str(min(INTENSITIES))]),
            "evidence": {"why_it_can_fail": "H3 has no axis if the ladder does not escalate",
                         "mean_risk_events_by_intensity": mean_events}},
        "f6_variance_is_across_intensities_not_within": {
            "passed": bool(abs(float(np.mean(within_rung))
                               - levels["H3_variance_service_loss_auc"]["hybrid"]) > 1e-9),
            "evidence": {"why_it_can_fail": "taking the variance along seeds instead of along "
                                            "intensities answers a different question; the "
                                            "control recomputes it the wrong way and demands a "
                                            "different number",
                         "across_intensities_hybrid":
                             levels["H3_variance_service_loss_auc"]["hybrid"],
                         "wrong_axis_control": float(np.mean(within_rung))}},
        "f7_no_new_seeds_are_opened": {
            "passed": bool(set(seeds) <= set(range(6_000_001, 6_000_121))),
            "evidence": {"why_it_can_fail": "a seed outside the already-open power block would "
                                            "mean this run consumed custody it never declared",
                         "block": "6000001-6000120", "n_seeds": len(seeds),
                         "min": min(seeds), "max": max(seeds)}},
    }
    falsifiers["all_passed"] = all(v["passed"] for k, v in falsifiers.items()
                                   if k != "all_passed" and isinstance(v, dict))

    if not falsifiers["f1_the_arms_deploy_different_configurations"]["passed"]:
        verdict = "EMPTY_BY_CONSTRUCTION_AGAIN_ARMS_DEPLOY_THE_SAME_CONFIGURATION"
    elif not falsifiers["f2_the_recovery_endpoint_has_range"]["passed"]:
        verdict = "H1_STILL_NOT_EVALUABLE__" + ("H3_SUPPORTED" if h3_sup else "H3_NOT_SUPPORTED")
    else:
        verdict = (("H1_SUPPORTED" if h1_sup else "H1_NOT_SUPPORTED") + "__"
                   + ("H3_SUPPORTED" if h3_sup else "H3_NOT_SUPPORTED"))

    print(f"\n  niveles H1 (TTR restringido, horas; menor es mejor): "
          + "  ".join(f"{a} {v:,.1f}" for a, v in levels["H1_restricted_ttr_hours"].items()))
    print(f"  niveles H3 (varianza del servicio perdido; menor es mejor): "
          + "  ".join(f"{a} {v:,.3g}" for a, v in
                      levels["H3_variance_service_loss_auc"].items()))
    print("\n  contrastes (positivo = el híbrido gana):")
    for name, c in contrasts.items():
        print(f"    {name:<24} {c['mean']:+12,.4g}  [{c['lcb95']:+,.4g}, {c['ucb95']:+,.4g}]  "
              f"holm p={holm_table[name]['p_holm']:.4f}")
    print(f"\n  veredicto: {verdict}\n")
    for name, f in falsifiers.items():
        if name == "all_passed" or not isinstance(f, dict):
            continue
        print(f"    {name:<48} {'PASA' if f['passed'] else 'FALLA'}")

    payload = {
        "schema_version": "h1_h3_originales_v3",
        "claim_status": verdict,
        "scope": "DEVELOPMENT_ALREADY_OPEN_BLOCK_NO_VIRGIN_SEEDS_NO_ADJUDICATION",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "module_manifest": module_manifest(MODULES, script=__file__),
        "preregistration": str(args.contract),
        "what_this_does_not_touch": ("H1' (results/manuscript/h1_h3_v2_1) and H3' "
                                     "(results/garrido_h3_merge_adjudication) are declared "
                                     "reformulations, already supported, and are not reopened."),
        "endpoint_redefinition_declared": (
            "H1 uses restricted_ttr = min(TTR, tau) with a paired placebo, not system_ttr. It is "
            "a different estimand, written 2026-08-06 for the v0 lane and before this "
            "preregistration, not a loosened version of the one that returned 1.000."),
        "regime_note": ("Under the recurrent R11-R24 regime at 52 weeks the events merge into one "
                        "cluster that never ends, so no return-to-normal exists to time. H1 is "
                        "therefore measured under isolated shocks, and that is a property of "
                        "Garrido's risk regime, not of the instrument."),
        "source_slices": sealed_meta, "seeds": seeds, "arms": ARMS,
        "intensities": list(INTENSITIES), "shock_contexts": list(CONTEXT_ORDER),
        "shock_to_meta_context": SHOCK_TO_META,
        "h1_horizon_hours": H1_HORIZON, "h3_horizon_hours": H3_HORIZON,
        "tau_hours": RECOVERY_WINDOW_HOURS,
        "levels": levels, "contrasts": contrasts, "holm": holm_table,
        "secondary_descriptive": secondary,
        "distinct_configurations_deployed": {
            a: len({h1_config[(a, s, c)] for (s, c) in h1_keys}) for a in ARMS},
        "differing_cell_share": differing_share,
        "f4_rows": f4_rows, "falsifiers": falsifiers,
        "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(payload, args.output, contract=args.contract, reference=SLICES[0])
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
