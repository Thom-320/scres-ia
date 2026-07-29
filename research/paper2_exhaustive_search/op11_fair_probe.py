"""op11 fair-allocation conversion probe — gates G0–G4 (contract op11_fair_allocation_conversion_probe_v1).

Implements the frozen fair-policy family and runs the development gates on BURNED blocks only.
The environment (supply_chain/headroom_sensitivity.py, supply_chain/program_g.py) is NOT modified.
G5 (virgin block 4800001-4800200) is NOT runnable from this module's main(): it requires the
independent verifier's sign-off first (contract implementation_split.verifier_gate).

Run: .venv/bin/python -m research.paper2_exhaustive_search.op11_fair_probe
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from supply_chain.headroom_sensitivity import (
    WEEKS, ARM, theta_to_cell, materialize_tape_theta, _belief_policy,
)
from supply_chain.program_g import (
    ACTIONS, CONVOY_LOAD, CSSU_CAP, CYCLES_PER_WEEK, DEMAND_DAYS, MULT, S1_DAILY, SB_INITIAL,
    metrics_all,
)
from supply_chain.program_h_belief import CSSUFilter

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "contracts/op11_fair_allocation_conversion_probe_v1.json"
OUT = ROOT / "research/paper2_exhaustive_search/op11_fair_probe_g0_g4_20260716.json"

# Frozen verbatim from the contract (source: results/headroom_gsa/oos_guardrail_check.json).
THETA_STAR = {"signal_q": 0.532, "lead": 2.0, "surge_mult": 1.946,
              "persistence": 0.0, "commonality": 0.887, "r22_prob": 0.107}
ROBUSTNESS = [{"surge_mult": 1.751}, {"surge_mult": 2.141}, {"commonality": 0.798}]
DEV_BLOCKS = {"3000001": 3_000_001, "4200001": 4_200_001, "4500001": 4_500_001}
N_TAPES = 200
GUARDRAIL_FLOOR = -0.02
BOOT_N, BOOT_SEED = 10_000, 20_260_716


# ----------------------------------------------------------------------------- shared bookkeeping
def _step_with_served(inv, sb, a, demand_w, r22_w):
    """program_g._week_step dynamics + per-CSSU served accounting (equivalence pinned by test)."""
    inv = inv.copy(); sb = sb + S1_DAILY * DEMAND_DAYS
    served = np.zeros(2)
    if a in ("A", "B"):
        i = 0 if a == "A" else 1
        cycles = CYCLES_PER_WEEK - (1 if int(r22_w[i]) else 0)
        deliver = min(cycles * CONVOY_LOAD, sb, max(0.0, CSSU_CAP - inv[i]))
        inv[i] += deliver; sb -= deliver
    unmet = 0.0
    for j in range(2):
        s = min(inv[j], demand_w[j]); inv[j] -= s; served[j] += s
        unmet += demand_w[j] - s
    return inv, sb, unmet, served


class _BeliefState:
    """The Program I belief machinery, exposed step-by-step (identical information set)."""

    def __init__(self, cell):
        self.fA = CSSUFilter(cell["persistence"], cell["surge_mult"], cell["signal_q"], cell["lead_weeks"])
        self.fB = CSSUFilter(cell["persistence"], cell["surge_mult"], cell["signal_q"], cell["lead_weeks"])
        self.means = np.array([2500 / 2 * MULT["low"] * DEMAND_DAYS, 2500 / 2 * DEMAND_DAYS,
                               2500 / 2 * cell["surge_mult"] * DEMAND_DAYS])

    def observe(self, t, w):
        if w > 0:
            self.fA.update(demand=t.demand[w - 1, 0]); self.fB.update(demand=t.demand[w - 1, 1])
            self.fA.predict(); self.fB.predict()
        self.fA.update(sig=int(t.signal[w, 0])); self.fB.update(sig=int(t.signal[w, 1]))

    def expected(self):
        return (float(self.fA.tempo_marginal() @ self.means),
                float(self.fB.tempo_marginal() @ self.means))


def _rollout(t, choose):
    """Non-anticipative rollout: `choose(w, inv, belief, fills)` -> 'A'|'B'. Never HOLD."""
    cell = t.cell
    bs = _BeliefState(cell)
    inv = np.zeros(2); sb = float(SB_INITIAL)
    cum_served = np.zeros(2); cum_demand = np.zeros(2)
    acts = []
    for w in range(t.weeks):
        bs.observe(t, w)
        fills = cum_served / np.maximum(cum_demand, 1.0)  # realized fill BEFORE week w
        a = choose(w, inv, bs, fills, acts)
        assert a in ("A", "B")
        acts.append(a)
        inv, sb, _, served = _step_with_served(inv, sb, a, t.demand[w], t.r22[w])
        cum_served += served; cum_demand += t.demand[w]
    return tuple(acts)


def _ratio_choice(inv, bs):
    eA, eB = bs.expected()
    return "A" if inv[0] / max(eA, 1) <= inv[1] / max(eB, 1) else "B"


# ----------------------------------------------------------------------------- frozen candidates
def policy_maxmin_realized_fill(t):
    def choose(w, inv, bs, fills, acts):
        if w == 0 or abs(fills[0] - fills[1]) < 1e-12:
            return _ratio_choice(inv, bs)
        return "A" if fills[0] < fills[1] else "B"
    return _rollout(t, choose)


def policy_fair_gate_ratio(t, delta):
    def choose(w, inv, bs, fills, acts):
        a = _ratio_choice(inv, bs)
        other = "B" if a == "A" else "A"
        ia, io = (0, 1) if a == "A" else (1, 0)
        if w > 0 and fills[io] < fills[ia] - delta:
            return other
        return a
    return _rollout(t, choose)


def policy_min_service_alternation(t):
    def choose(w, inv, bs, fills, acts):
        a = _ratio_choice(inv, bs)
        if w > 0:
            unserved = "B" if acts[w - 1] == "A" else "A"
            if a != unserved:
                return unserved  # never leave a CSSU unserved 2 consecutive weeks
        return a
    return _rollout(t, choose)


CANDIDATES = {
    "maxmin_realized_fill": policy_maxmin_realized_fill,
    "fair_gate_ratio_d0.02": lambda t: policy_fair_gate_ratio(t, 0.02),
    "fair_gate_ratio_d0.05": lambda t: policy_fair_gate_ratio(t, 0.05),
    "fair_gate_ratio_d0.10": lambda t: policy_fair_gate_ratio(t, 0.10),
    "min_service_alternation": policy_min_service_alternation,
}


# ----------------------------------------------------------------------------- evaluation
@dataclass
class SeqEval:
    ret_order: float
    ret_quantity: float
    attended_frac: float
    worst_cssu_fill: float


def eval_seq(t, seq) -> SeqEval:
    # single daily-order-adapter rollout; worst_cssu_fill is the CONTRACT-frozen lens
    # (program_g.py:420 in metrics_all), NOT simulate()'s weekly-fill lens. The first G0 run
    # used the weekly lens and failed the -0.10 check at -0.096 while reproducing H/CI/rq of
    # the custodied OOS artifact exactly -- a lens mismatch in this module, corrected here.
    m = metrics_all(t, seq, ARM)
    return SeqEval(ret_order=m["ret_order"], ret_quantity=m["ret_quantity"],
                   attended_frac=m["attended_orders"] / max(len(m["orders"]), 1),
                   worst_cssu_fill=m["worst_cssu_fill"])


ALL_SEQS = list(itertools.product(ACTIONS, repeat=WEEKS))  # 81; superset of periodic_calendars(4)


def block_tapes(seed0, theta, n=N_TAPES):
    cell = theta_to_cell(theta)
    return [materialize_tape_theta(seed0 + i, cell) for i in range(n)]


def eval_block(tapes, policies: dict):
    """Evaluate all 81 open-loop sequences (-> in-sample static comparator + clairvoyant oracle)
    and every policy in `policies` on each tape. Returns per-tape panels."""
    seq_ret = np.array([[eval_seq(t, s).ret_order for s in ALL_SEQS] for t in tapes])  # (n, 81)
    static_idx = int(seq_ret.mean(axis=0).argmax())          # strongest open-loop ON MEAN, in-sample
    static_seq = ALL_SEQS[static_idx]
    static = [eval_seq(t, static_seq) for t in tapes]
    oracle = seq_ret.max(axis=1)                              # per-tape clairvoyant (report-only)
    out = {"static_seq": list(static_seq),
           "static": static,
           "oracle_ret": oracle,
           "policies": {}}
    for name, fn in policies.items():
        out["policies"][name] = [eval_seq(t, fn(t)) for t in tapes]
    return out


def deltas(policy_evals, static_evals):
    return {
        "H": np.array([p.ret_order - s.ret_order for p, s in zip(policy_evals, static_evals)]),
        "wcf": np.array([p.worst_cssu_fill - s.worst_cssu_fill for p, s in zip(policy_evals, static_evals)]),
        "att": np.array([p.attended_frac - s.attended_frac for p, s in zip(policy_evals, static_evals)]),
        "rq": np.array([p.ret_quantity - s.ret_quantity for p, s in zip(policy_evals, static_evals)]),
    }


def boot_ci(x, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(BOOT_N, len(x)), replace=True).mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)),
            float(np.percentile(means, 5.0))]  # [lo95, hi95, one-sided LCB95]


def summarize(d):
    return {"H_mean": float(d["H"].mean()), "H_ci": boot_ci(d["H"]),
            "worst_cssu_fill_delta": float(d["wcf"].mean()),
            "attended_delta": float(d["att"].mean()),
            "ret_quantity_delta": float(d["rq"].mean()),
            "favorable_tapes": int((d["H"] > 0).sum()), "n": int(len(d["H"]))}


def eligible(summary):
    return (summary["worst_cssu_fill_delta"] >= GUARDRAIL_FLOOR
            and summary["attended_delta"] >= GUARDRAIL_FLOOR
            and summary["ret_quantity_delta"] >= GUARDRAIL_FLOOR)


# ----------------------------------------------------------------------------- gates
def main() -> int:
    contract_sha = __import__("hashlib").sha256(CONTRACT.read_bytes()).hexdigest()
    out = {"schema_version": "op11_fair_probe_g0_g4_v1", "date": "2026-07-16",
           "contract": str(CONTRACT.relative_to(ROOT)), "contract_sha256": contract_sha,
           "virgin_block_opened": False,
           "delta_definitions": {
               "H": "policy.ret_order - static.ret_order (per tape, paired)",
               "worst_cssu_fill_delta": "raw difference of program_g metrics_all() worst_cssu_fill (daily order adapter, program_g.py:420 -- the contract-frozen lens)",
               "attended_delta": "difference of attended-order FRACTION (attended/total orders)",
               "ret_quantity_delta": "raw difference of quantity-weighted ret",
               "static": "argmax-mean open-loop sequence over all 81, in-sample per block (superset of periodic_calendars(4))"}}

    # --- G0 + G4 development panels (theta*, three burned blocks)
    anchor = {"anchor_ratio_belief": _belief_policy}
    dev = {}
    for label, seed0 in DEV_BLOCKS.items():
        tapes = block_tapes(seed0, THETA_STAR)
        panel = eval_block(tapes, {**anchor, **CANDIDATES})
        dev[label] = panel
    g0 = summarize(deltas(dev["3000001"]["policies"]["anchor_ratio_belief"], dev["3000001"]["static"]))
    g0_pass = bool(g0["H_ci"][0] > 0 and g0["worst_cssu_fill_delta"] <= -0.10)
    out["G0_reproduction"] = {**g0, "criteria": "CI95 low > 0 AND worst_cssu_fill_delta <= -0.10",
                              "pass": g0_pass}
    if not g0_pass:
        out["status"] = "STOP_ENV_MISMATCH"
        OUT.write_text(json.dumps(out, indent=1)); print(json.dumps(out["G0_reproduction"], indent=1))
        return 1

    # --- G1 anchor null (commonality=0)
    from supply_chain.headroom_sensitivity import headroom_at
    th0 = dict(THETA_STAR); th0["commonality"] = 0.0
    h = headroom_at(th0, n_tapes=N_TAPES, seed0=DEV_BLOCKS["3000001"])
    g1_pass = bool(abs(h.H_PI - 0.015) <= 0.01 and abs(h.H_obs - (-0.02)) <= 0.01)
    out["G1_anchor_null"] = {"H_PI": h.H_PI, "H_obs": h.H_obs, "eta": h.eta,
                             "criteria": "|H_PI-0.015|<=0.01 AND |H_obs-(-0.02)|<=0.01", "pass": g1_pass}

    # --- G4 selection (eligibility on EVERY dev block; argmax pooled mean H)
    g4 = {"per_candidate": {}}
    pooled_H = {}
    for name in CANDIDATES:
        per_block = {lbl: summarize(deltas(dev[lbl]["policies"][name], dev[lbl]["static"])) for lbl in dev}
        elig = all(eligible(s) for s in per_block.values())
        pooled = float(np.mean([s["H_mean"] for s in per_block.values()]))
        g4["per_candidate"][name] = {"blocks": per_block, "eligible_all_blocks": elig,
                                     "pooled_H_mean": pooled}
        if elig:
            pooled_H[name] = pooled
    if not pooled_H:
        out["G4_development_selection"] = {**g4, "selected": None}
        out["status"] = "DEVELOPMENT_NO_ELIGIBLE_CANDIDATE__DOOR_CLOSES_WITHOUT_OPENING_VIRGIN_SEEDS"
        OUT.write_text(json.dumps(out, indent=1)); print(out["status"])
        return 0
    selected = max(pooled_H, key=pooled_H.get)
    g4["selected"] = selected
    out["G4_development_selection"] = g4

    # eta_fair (report-only) on pooled dev blocks
    ora = np.concatenate([dev[l]["oracle_ret"] for l in dev])
    sta = np.concatenate([[s.ret_order for s in dev[l]["static"]] for l in dev])
    H_PI_pooled = float((ora - sta).mean())
    out["eta_fair_report_only"] = {"H_PI_pooled": H_PI_pooled,
                                   "eta_fair": float(pooled_H[selected] / H_PI_pooled) if abs(H_PI_pooled) > 1e-9 else None}

    # --- G2 information null (signal_q = 0.5) for the selected candidate
    thq = dict(THETA_STAR); thq["signal_q"] = 0.5
    tapes_q = block_tapes(DEV_BLOCKS["3000001"], thq)
    panel_q = eval_block(tapes_q, {selected: CANDIDATES[selected]})
    g2 = summarize(deltas(panel_q["policies"][selected], panel_q["static"]))
    g2_pass = not (g2["H_ci"][0] > 0)
    out["G2_information_null"] = {**g2, "criteria": "CI95 must NOT be entirely above 0", "pass": g2_pass}

    # --- G3 permuted-signal placebo (shift-by-1 signal swap across tapes)
    tapes_p = block_tapes(DEV_BLOCKS["3000001"], THETA_STAR)
    sigs = [t.signal.copy() for t in tapes_p]
    for i, t in enumerate(tapes_p):
        t.signal = sigs[(i + 1) % len(tapes_p)]
    panel_p = eval_block(tapes_p, {selected: CANDIDATES[selected]})
    g3 = summarize(deltas(panel_p["policies"][selected], panel_p["static"]))
    g3_pass = not (g3["H_ci"][0] > 0)
    out["G3_signal_placebo"] = {**g3, "criteria": "CI95 must NOT be entirely above 0", "pass": g3_pass}

    # --- robustness thetas (report-only)
    rob = {}
    for pert in ROBUSTNESS:
        th = dict(THETA_STAR); th.update(pert)
        tapes_r = block_tapes(DEV_BLOCKS["3000001"], th)
        panel_r = eval_block(tapes_r, {selected: CANDIDATES[selected]})
        rob[json.dumps(pert)] = summarize(deltas(panel_r["policies"][selected], panel_r["static"]))
    out["robustness_report_only"] = rob

    all_pass = g0_pass and g1_pass and g2_pass and g3_pass
    out["execution_freeze"] = {
        "selected_candidate": selected,
        "frozen_for_G5": bool(all_pass),
        "G5_block": "4800001-4800200",
        "G5_requires": "independent verifier sign-off on this artifact BEFORE opening (contract implementation_split.verifier_gate)"}
    out["status"] = ("G0_G4_COMPLETE_AWAITING_INDEPENDENT_VERIFICATION_BEFORE_G5" if all_pass
                     else "GATE_FAILURE_SEE_FIELDS")
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("G0_reproduction", "G1_anchor_null", "G2_information_null",
                                          "G3_signal_placebo", "status")}, indent=1))
    print("selected:", selected, "| pooled H:", pooled_H)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
