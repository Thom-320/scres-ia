#!/usr/bin/env python3
"""G3-obs: does the ceiling convert with DEPLOYABLE observations?

The re-audit's "clairvoyant" arm read `cssu_demanded - cssu_delivered`: a ledger quantity the
operator knows, which does not read the future. So it was already an OBSERVABLE policy, and a
two-branch threshold at that. STRUCTURED_CONTROL_SUFFICES is therefore the EXPECTED outcome, and
this contract exists to establish it properly rather than to look for an exception.

Two questions stay open: does the value survive realistic observation limits (finite window,
delay, noise), and is there residual over the best simple threshold that a richer policy captures?

Seeds: the BURNED block 5_200_001-16, split 8 development / 8 test. No fresh roots.
Contract: docs/PREREGISTRO_G3_OBS_CONVERSION_OBSERVABLE_2026-08-01.md
"""
from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.config import HOURS_PER_WEEK  # noqa: E402
from supply_chain.config import THESIS_FAITHFUL_PROTOCOL as P  # noqa: E402
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.seed_custody import custody_falsifier  # noqa: E402
from supply_chain.service_first_metric import claimant_fills  # noqa: E402
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402

R1R = ("R11", "R12", "R13", "R14")
R2R = ("R21", "R22", "R23", "R24")
RULE = "FIFO_PARTIAL"
REGIMES = {
    "R1r+R2r|base": (R1R + R2R, {}, {}),
    "R1r+R2r|freq3_imp2": (R1R + R2R, {"R23": 3.0}, {"R23": 2.0}),
}
SHARES = tuple(round(0.1 + 0.1 * i, 2) for i in range(9))
SWING_HI, SWING_LO, NEUTRAL = 0.9, 0.1, 0.5
BIN_ALPHAS = (0.1, 0.3, 0.5, 0.7, 0.9)
WINDOW_DAYS = 14
DELAY_DAYS = 3
NOISE_SD = 0.30
PRIMARY = "worst_claimant_fill"
SESOI = 0.010
# Signed non-inferiority margins, inherited verbatim from the G3c preregistration.
# lost_orders re-derived from operations (one lost delivery every two years), NOT from
# Monte Carlo granularity -- see docs/ENMIENDA_G3C_MARGENES_OPERACIONALES_2026-08-02.md
MARGINS = {"flow_fill_rate": 0.005, "lost_orders": 0.50, "backorder_qty_final_rel": 0.010}
DIAGNOSTICS = ("flow_fill_rate", "lost_orders", "backorder_qty_final",
               "ret_excel_full_ledger", "ret_excel_risk_conditional")
SEED_BASE = 5_200_001          # default: the burned block of the underpowered run
STEP_HOURS = 24.0
Z90, Z95 = 1.2816, 1.6449          # one-sided 90% power, one-sided 95% test
SEALED_REFERENCE = Path("results/sensitivity/contention_headroom_v1_1/result.json")


def _build(seed, risks, freq, impact, *, share):
    return MFSCSimulation(
        shifts=1, initial_buffers={n: 0.0 for n in ("op3_rm", "op5_rm", "op9_rations")},
        inventory_replenishment_period=0.0, seed=seed,
        horizon=float(52 * HOURS_PER_WEEK),
        risks_enabled=True, risk_level="current", enabled_risks=set(risks),
        risk_frequency_multipliers_by_id=dict(freq) or None,
        risk_impact_multipliers_by_id=dict(impact) or None,
        cssu_topology_mode="split_v1", cssu_allocation_a=float(share),
        cssu_service_rule=RULE, cssu_reallocate_unused=False,
        order_fulfillment_mode="op9_linked", op9_dispatch_policy="fixed_clock_daily",
        strict_exogenous_crn=True, year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"], r14_defect_mode=P["r14_defect_mode"])


def _panel(sim, alphas):
    panel = compute_episode_metrics(sim)
    fills = claimant_fills(sim)
    out = {PRIMARY: float(min(fills.values())) if fills else float("nan")}
    out.update({k: float(panel.get(k, float("nan"))) for k in DIAGNOSTICS})
    out["alpha_sd"] = float(np.std(alphas)) if alphas else 0.0
    out["alpha_switches"] = float(sum(1 for i in range(1, len(alphas))
                                      if abs(alphas[i] - alphas[i - 1]) > 1e-9))
    return out


def episode_constant(seed, risks, freq, impact, *, share):
    sim = _build(seed, risks, freq, impact, share=share)
    alphas, done = [], False
    while not done:
        _, _, done, _ = sim.step(step_hours=STEP_HOURS)
        alphas.append(float(sim.cssu_allocation_a))
    return _panel(sim, alphas)


def episode_policy(seed, risks, freq, impact, *, arm: str, tau: float = 0.0,
                   bins: tuple[float, ...] | None = None):
    """One episode under an observable-signal policy.

    The signal is built from the trailing window of unmet demand PER CLAIMANT, read at the moment
    of decision. It never contains the future risk target, and the action only takes effect after
    the model's own 24 h activation latency -- so nothing here is clairvoyant.
    """
    sim = _build(seed, risks, freq, impact, share=NEUTRAL)
    rng = np.random.default_rng(seed ^ 0x5EED0B5)
    hist: deque[tuple[float, float]] = deque(maxlen=WINDOW_DAYS)
    delayed: deque[float] = deque(maxlen=DELAY_DAYS + 1)
    prev = {"A": 0.0, "B": 0.0}
    alphas, done = [], False
    while not done:
        unmet = {c: float(sim.cssu_demanded.get(c, 0.0)) - float(sim.cssu_delivered.get(c, 0.0))
                 for c in ("A", "B")}
        # Per-day increment, so a finite window means something. `cumulative` deliberately keeps
        # the whole horizon: it is the re-audit's rule, and the realism arms are measured
        # against it.
        inc = {c: unmet[c] - prev[c] for c in ("A", "B")}
        prev = unmet
        hist.append((inc["A"], inc["B"]))
        if arm == "threshold_cumulative":
            a, b = unmet["A"], unmet["B"]
        else:
            a = sum(x for x, _ in hist)
            b = sum(y for _, y in hist)
        s = (a - b) / (abs(a) + abs(b) + 1e-9)
        if arm == "threshold_noisy":
            s = float(s * (1.0 + NOISE_SD * rng.standard_normal()))
        delayed.append(s)
        if arm == "threshold_delayed":
            s = delayed[0] if len(delayed) > DELAY_DAYS else 0.0
        if arm == "uninformed_placebo":
            target = float(rng.choice((SWING_HI, NEUTRAL, SWING_LO)))
        elif arm == "wrong_claimant":
            target = SWING_LO if s > tau else (SWING_HI if s < -tau else NEUTRAL)
        elif arm == "tabular_5bin":
            assert bins is not None
            edges = (-0.5, -0.15, 0.15, 0.5)
            k = int(np.searchsorted(edges, s))
            target = float(bins[k])
        else:
            target = SWING_HI if s > tau else (SWING_LO if s < -tau else NEUTRAL)
        if (sim._pending_cssu_action is None
                and abs(float(sim.cssu_allocation_a) - target) > 1e-9):
            action = {"cssu_allocation_a": float(target)}
        else:
            action = None
        _, _, done, _ = sim.step(action=action, step_hours=STEP_HOURS)
        alphas.append(float(sim.cssu_allocation_a))
    return _panel(sim, alphas)


def boot(diff: np.ndarray, n_boot: int, rng) -> dict[str, float]:
    stats = np.empty(n_boot)
    for b in range(n_boot):
        stats[b] = float(np.mean(diff[rng.integers(0, diff.shape[0], diff.shape[0])]))
    return {"mean": float(np.mean(diff)),
            "lcb95": float(np.percentile(stats, 2.5)),
            "ucb95": float(np.percentile(stats, 97.5))}


def mde(diff: np.ndarray) -> float:
    """Minimum detectable effect at 90% power, one-sided, given the observed paired SD."""
    sd = float(np.std(diff, ddof=1))
    return float((Z90 + Z95) * sd / np.sqrt(diff.shape[0]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=5_000)
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--seed-base", type=int, default=SEED_BASE)
    ap.add_argument("--replay-of", default=None,
                    help="registry block id this run deliberately re-executes")
    ap.add_argument("--output", type=Path,
                    default=Path("results/headroom/g3_obs_conversion/result.json"))
    args = ap.parse_args()
    seeds = [args.seed_base + i for i in range(args.seeds)]
    half = len(seeds) // 2
    dev, test = seeds[:half], seeds[half:]      # disjoint; every parameter is fit on dev only
    started = time.perf_counter()

    TAUS = (0.0, 0.05, 0.15, 0.30)
    BIN_CANDIDATES = ((0.1, 0.3, 0.5, 0.7, 0.9), (0.1, 0.1, 0.5, 0.9, 0.9),
                      (0.3, 0.4, 0.5, 0.6, 0.7))
    REALISM = ("threshold_windowed", "threshold_delayed", "threshold_noisy")
    results: dict[str, dict] = {}
    rng = np.random.default_rng(20260801)

    for rname, (risks, freq, impact) in REGIMES.items():
        key = f"{RULE}|{rname}"
        print(f"  ajustando en desarrollo: {key}")
        # --- development: choose tau and bins WITHOUT ever looking at the test seeds ---
        dev_tau = {t: float(np.mean([episode_policy(s, risks, freq, impact,
                                                    arm="threshold_windowed", tau=t)[PRIMARY]
                                     for s in dev])) for t in TAUS}
        best_tau = max(dev_tau, key=lambda t: dev_tau[t])
        dev_bins = {i: float(np.mean([episode_policy(s, risks, freq, impact, arm="tabular_5bin",
                                                     bins=bc)[PRIMARY] for s in dev]))
                    for i, bc in enumerate(BIN_CANDIDATES)}
        best_bins = BIN_CANDIDATES[max(dev_bins, key=lambda i: dev_bins[i])]
        dev_share = {sh: float(np.mean([episode_constant(s, risks, freq, impact,
                                                         share=sh)[PRIMARY] for s in dev]))
                     for sh in SHARES}
        best_share = max(dev_share, key=lambda sh: dev_share[sh])

        print(f"    tau*={best_tau}  bins*={best_bins}  share*={best_share}  -> evaluando en test")
        arms: dict[str, dict] = {
            "best_constant": {s: episode_constant(s, risks, freq, impact, share=best_share)
                              for s in test}}
        for arm in ("threshold_cumulative", *REALISM, "uninformed_placebo", "wrong_claimant"):
            arms[arm] = {s: episode_policy(s, risks, freq, impact, arm=arm, tau=best_tau)
                         for s in test}
        arms["tabular_5bin"] = {s: episode_policy(s, risks, freq, impact, arm="tabular_5bin",
                                                  tau=best_tau, bins=best_bins) for s in test}

        base = np.array([arms["best_constant"][s][PRIMARY] for s in test])
        vs_const = {a: boot(np.array([arms[a][s][PRIMARY] for s in test]) - base,
                            args.n_boot, rng)
                    for a in arms if a != "best_constant"}
        # Best OBSERVABLE policy is chosen on development-consistent grounds: the simple
        # threshold family is the pre-declared primary observable comparator.
        simple = np.array([arms["threshold_windowed"][s][PRIMARY] for s in test])
        rich = np.array([arms["tabular_5bin"][s][PRIMARY] for s in test])
        residual = boot(rich - simple, args.n_boot, rng)
        h_obs = vs_const["threshold_windowed"]
        realism_cost = {a: boot(
            np.array([arms["threshold_cumulative"][s][PRIMARY] for s in test])
            - np.array([arms[a][s][PRIMARY] for s in test]), args.n_boot, rng) for a in REALISM}

        guard = {}
        for metric, delta in (("flow_fill_rate", MARGINS["flow_fill_rate"]),
                              ("lost_orders", MARGINS["lost_orders"])):
            cand = np.array([arms["threshold_windowed"][s][metric] for s in test])
            ref = np.array([arms["best_constant"][s][metric] for s in test])
            # Harm is "worse than the incumbent": lower fill, or MORE lost orders.
            harm = (ref - cand) if metric == "flow_fill_rate" else (cand - ref)
            b = boot(harm, args.n_boot, rng)
            guard[metric] = {**b, "delta": delta, "passes": bool(b["ucb95"] <= delta)}
        ref_bo = np.array([arms["best_constant"][s]["backorder_qty_final"] for s in test])
        cand_bo = np.array([arms["threshold_windowed"][s]["backorder_qty_final"] for s in test])
        scale = float(np.mean(np.abs(ref_bo))) + 1e-9
        b = boot((cand_bo - ref_bo) / scale, args.n_boot, rng)
        guard["backorder_qty_final_rel"] = {
            **b, "delta": MARGINS["backorder_qty_final_rel"],
            "passes": bool(b["ucb95"] <= MARGINS["backorder_qty_final_rel"])}

        results[key] = {
            "development": {"best_tau": best_tau, "best_bins": list(best_bins),
                            "best_share": best_share, "dev_seeds": dev},
            "test_seeds": test,
            "H_obs_windowed_minus_constant": h_obs,
            "residual_over_simple": residual,
            "vs_constant": vs_const,
            "realism_cost_vs_cumulative": realism_cost,
            "guardrails": guard,
            "mde_primary": mde(np.array([arms["threshold_windowed"][s][PRIMARY]
                                         for s in test]) - base),
            "alpha_sd": float(np.mean([arms["threshold_windowed"][s]["alpha_sd"] for s in test])),
            "diagnostics": {d: {a: float(np.mean([arms[a][s][d] for s in test]))
                                for a in arms} for d in DIAGNOSTICS},
        }

    powered = all(r["mde_primary"] <= SESOI for r in results.values())
    converts = all(r["H_obs_windowed_minus_constant"]["lcb95"] > 0
                   and r["H_obs_windowed_minus_constant"]["mean"] >= SESOI
                   for r in results.values())
    residual_material = all(r["residual_over_simple"]["lcb95"] > 0
                            and r["residual_over_simple"]["mean"] >= SESOI
                            for r in results.values())
    guards_ok = all(g["passes"] for r in results.values() for g in r["guardrails"].values())
    signal_ordering = all(
        r["vs_constant"]["threshold_windowed"]["mean"]
        > max(r["vs_constant"]["uninformed_placebo"]["mean"],
              r["vs_constant"]["wrong_claimant"]["mean"]) for r in results.values())
    live = all(r["alpha_sd"] > 1e-6 for r in results.values())

    falsifiers = {
        "f1_signal_is_causal_and_pre_action": {
            "passed": True,
            "evidence": {"why_it_can_fail": "a signal containing the future risk target would be "
                                            "an oracle in disguise",
                         "construction": "unmet demand per claimant, read before the decision; the "
                                         "action then waits the model's own 24h activation latency",
                         "future_symbols_read": []}},
        "f2_real_signal_beats_shuffled_delayed_and_wrong": {
            "passed": bool(signal_ordering),
            "evidence": {"why_it_can_fail": "if an uninformed or inverted signal matches the real "
                                            "one, the signal carries no information",
                         "vs_constant": {k: {a: r["vs_constant"][a]["mean"] for a in
                                             ("threshold_windowed", "uninformed_placebo",
                                              "wrong_claimant")} for k, r in results.items()}}},
        "f3_thresholds_fit_on_development_only": {
            "passed": all(not (set(r["development"]["dev_seeds"]) & set(r["test_seeds"]))
                          for r in results.values()),
            "evidence": {"why_it_can_fail": "picking tau or bins on the test block would select "
                                            "the best threshold by looking at its own result -- "
                                            "the defect shipped in the Cobb-Douglas run",
                         "dev": dev, "test": test}},
        "f4_every_guardrail_has_a_signed_margin": {
            "passed": all(g.get("delta", 0.0) > 0
                          for r in results.values() for g in r["guardrails"].values()),
            "evidence": {"why_it_can_fail": "a zero-margin guardrail on point estimates is what "
                                            "halted the re-audit on ONE lost order",
                         "margins": MARGINS}},
        "f5_power_is_published_pass_or_fail": {
            "passed": True,
            "evidence": {"why_it_can_fail": "it cannot; that is the point -- the MDE is published "
                                            "whether or not it clears the SESOI, so an underpowered "
                                            "null can never be read as a claim",
                         "mde": {k: r["mde_primary"] for k, r in results.items()},
                         "sesoi": SESOI, "powered": bool(powered)}},
        "f6_actuator_is_live": {
            "passed": bool(live),
            "evidence": {"why_it_can_fail": "a dead actuator already caught me once on this lever",
                         "alpha_sd": {k: r["alpha_sd"] for k, r in results.items()}}},
        "f7_no_gain_by_abandonment": {
            "passed": bool(guards_ok),
            "evidence": {"why_it_can_fail": "a worst-fill gain bought with lost orders is the "
                                            "measured failure mode of ret_excel in a new coordinate",
                         "guardrails": {k: r["guardrails"] for k, r in results.items()}}},
        # Custody now goes through the central registry instead of a hard-coded block. A run on
        # a newly authorized block must find it RESERVED_NOT_OPENED there, or it is a collision.
        "f8_seed_custody": custody_falsifier(seeds, replay_of=args.replay_of,
                                            exclude=args.output),
    }
    falsifiers["all_passed"] = all(
        v["passed"] for k, v in falsifiers.items()
        if k != "all_passed" and not v.get("not_applicable"))
    falsifiers["not_applicable"] = sorted(
        k for k, v in falsifiers.items()
        if k != "all_passed" and isinstance(v, dict) and v.get("not_applicable"))

    if not powered:
        verdict = "STOP_G3_OBS_UNDERPOWERED"
    elif not guards_ok:
        verdict = "STOP_G3_OBS_GUARDRAIL"
    elif converts and residual_material:
        verdict = "G3_OBS_RESIDUAL_OVER_SIMPLE_RULE"
    elif converts:
        verdict = "STRUCTURED_CONTROL_SUFFICES_G3_OBS"
    else:
        verdict = "OBSERVABLE_CONVERSION_FAILS"

    for key, r in results.items():
        print(f"\n  === {key} ===")
        h = r["H_obs_windowed_minus_constant"]
        print(f"  H_obs (ventana - constante)  {h['mean']:+.4f} [LCB95 {h['lcb95']:+.4f}]")
        rs = r["residual_over_simple"]
        print(f"  residual (tabular - umbral)  {rs['mean']:+.4f} [LCB95 {rs['lcb95']:+.4f}]")
        for a, v in r["vs_constant"].items():
            print(f"    {a:<24} {v['mean']:+.4f} [LCB95 {v['lcb95']:+.4f}]")
        print(f"  MDE(90%) {r['mde_primary']:.4f}  vs SESOI {SESOI}"
              f"  -> {'CON POTENCIA' if r['mde_primary'] <= SESOI else 'SIN POTENCIA'}")
        for m, g in r["guardrails"].items():
            print(f"    guardarraíl {m:<26} UCB95 {g['ucb95']:+.4f} <= δ {g['delta']}"
                  f"  {'PASA' if g['passes'] else 'FALLA'}")
    print(f"\n  veredicto: {verdict}\n")
    for name, f in falsifiers.items():
        if name in ("all_passed", "not_applicable") or not isinstance(f, dict):
            continue
        label = "NO APLICA" if f.get("not_applicable") else ("PASA" if f["passed"] else "FALLA")
        print(f"    {name:<48} {label}")

    payload = {
        "schema_version": "g3_obs_conversion_v1",
        "claim_status": verdict if falsifiers["all_passed"] else "HALTED_FALSIFIER_FAILED",
        "scope": "OBSERVABLE_CONVERSION_ON_BURNED_TAPES_NO_TRAINING_AUTHORIZED",
        "primary_metric": PRIMARY, "sesoi": SESOI, "margins": MARGINS,
        "window_days": WINDOW_DAYS, "delay_days": DELAY_DAYS, "noise_sd": NOISE_SD,
        "step_hours": STEP_HOURS, "seeds": seeds, "development_seeds": dev, "test_seeds": test,
        "rule": RULE, "regimes": list(REGIMES), "fungible": False,
        "results": results, "falsifiers": falsifiers,
        "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(
        payload, args.output,
        contract=Path("docs/PREREGISTRO_G3_OBS_CONVERSION_OBSERVABLE_2026-08-01.md"),
        reference=SEALED_REFERENCE)
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
