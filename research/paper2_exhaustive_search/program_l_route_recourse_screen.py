"""Program L — pre-RL headroom screen for EVENT-DRIVEN ALTERNATE-ROUTE RECOURSE
with a FINITE single vehicle and partially-observed, route-specific persistent R22.

This is the family my 12-region adversarial screen classified R03_BLOCKED_PENDING_PI.
The PI (user) reports Garrido will authorize the physics that unlocks Paper 2, so the
permission barrier is removed and the family becomes RUNNABLE. This module does NOT
train any learner: it runs the mandatory pre-learner gate (H_PI clairvoyant ceiling,
H_obs deployable observable policy, null cell, placebo, resource/guardrail checks) to
answer the ONLY question Garrido's authorization cannot: does convertible observable
headroom exist under the canonical ReT endpoint?

Design (single destination -> avoids the A/B equity confound that contaminated Program G):
  - one finite vehicle; dispatch decision at each daily opportunity: ROUTE_1 / ROUTE_2 / HOLD.
  - committing route r locks the vehicle for a MANDATORY round trip 2*tau_r(Z) (finite-fleet
    intertemporal commitment); tau inflates by PEN when that route is degraded (route-specific R22).
  - route condition Z_r,t in {normal, degraded} is semi-Markov (persistent, Garrido-validated knob).
  - pre-departure signal sigma_r observes Z_r with balanced accuracy q (imperfect, no future info).
  - orders are the project's OrderRecord ledger; scored by the CANONICAL ret_excel formula.

CRN: per-tape RNG draws Z and demand independently of actions; action changes only route
choice, never the exogenous Z/demand tape. Same snapshot + same tape + different action =>
identical exogenous events.  No inventory is created; the vehicle is the binding resource.
"""
from __future__ import annotations
import itertools, json, math
from dataclasses import dataclass
import numpy as np

from supply_chain.supply_chain import OrderRecord
from supply_chain.ret_thesis import compute_order_level_ret_excel_formula
from supply_chain.config import LEAD_TIME_PROMISE, HOURS_PER_WEEK

DEMAND_DAYS = 6
DAY = 24.0
S_BATTALION_AMPLE = 1e12   # supply battalion NOT binding: isolate the finite-FLEET routing channel
CONVOY_LOAD = 5000.0
ACTIONS = ("R1", "R2", "HOLD")


@dataclass
class Cell:
    tau_base: float        # normal one-way transit (h); thesis nominal leg = 24
    pen: float             # degraded-route transit penalty (h); route-specific R22 rehab-scale
    q: float               # signal balanced accuracy (sens=spec); 0.5 = uninformative
    persistence: float     # P(stay in current route condition) per day; higher = autocorrelated
    p_degraded: float      # stationary P(route degraded)
    demand_day: float      # daily destination demand (rations); thesis 2400-2600
    cssu_cap: float        # finite CSSU buffer (rations); large ~ thesis unlimited storage
    common_mode: float     # P(route 2 shares route 1's condition) = joint degradation coupling
    weeks: int = 8
    init_inv_frac: float = 1.0   # warm-start buffer fraction (1.0=full steady-state; 0=empty startup)


@dataclass
class Tape:
    Z: np.ndarray          # (days, 2) route condition 0/1
    sigma: np.ndarray      # (days, 2) noisy pre-departure signal 0/1
    demand: np.ndarray     # (days,)
    cell: Cell


def materialize(seed: int, cell: Cell) -> Tape:
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x2222]))
    days = cell.weeks * DEMAND_DAYS
    Z = np.zeros((days, 2), dtype=int)
    # route 1 semi-Markov
    for r in range(2):
        z = 1 if rng.random() < cell.p_degraded else 0
        for d in range(days):
            if r == 1 and rng.random() < cell.common_mode:
                Z[d, 1] = Z[d, 0]                     # common-mode: route 2 tracks route 1 today
                z = Z[d, 1]; continue
            # semi-Markov persistence toward stationary p_degraded
            if rng.random() < cell.persistence:
                pass                                   # stay
            else:
                z = 1 if rng.random() < cell.p_degraded else 0
            Z[d, r] = z
    sigma = np.zeros((days, 2), dtype=int)
    for d in range(days):
        for r in range(2):
            correct = rng.random() < cell.q
            sigma[d, r] = Z[d, r] if correct else (1 - Z[d, r])
    demand = rng.integers(int(cell.demand_day * 0.96), int(cell.demand_day * 1.04) + 1, size=days).astype(float)
    return Tape(Z=Z, sigma=sigma, demand=demand, cell=cell)


def tau(cell: Cell, z: int) -> float:
    return cell.tau_base + (cell.pen if z == 1 else 0.0)


def simulate_orders(tape: Tape, policy) -> list:
    """policy(day, obs) -> action index. obs exposes ONLY deployable info (signals, inv, backlog,
    vehicle availability, past trip outcomes). Emits canonical OrderRecord ledger."""
    cell = tape.cell
    days = cell.weeks * DEMAND_DAYS
    inv = float(cell.init_inv_frac) * cell.cssu_cap    # warm-start buffer (steady-state)
    veh_free_at = 0.0                       # hour the single vehicle is next available
    arrivals: list[tuple[float, float]] = []  # (arrival_hour, qty) in transit
    queue: list = []; orders = []; j = 0
    last_trip_outcome = 0                   # 1 if last trip used a route that turned out degraded
    n_trips = 0; veh_busy_h = 0.0           # resource accounting
    for d in range(days):
        h0 = d * DAY
        # land any arrivals due by now
        arrivals.sort()
        keep = []
        for (ah, qty) in arrivals:
            if ah <= h0 + 1e-9:
                inv = min(cell.cssu_cap, inv + qty)
            else:
                keep.append((ah, qty))
        arrivals = keep
        # place the day's order
        j += 1
        o = OrderRecord(j=j, OPTj=h0, quantity=float(tape.demand[d]),
                        LTj=float(LEAD_TIME_PROMISE), remaining_qty=float(tape.demand[d]))
        queue.append(o); orders.append(o)
        # dispatch decision if the vehicle is available today
        if veh_free_at <= h0 + 1e-9:
            obs = {
                "sigma": tape.sigma[d], "inv": inv,
                "backlog": sum(x.remaining_qty for x in queue),
                "veh_free": True, "last_trip_outcome": last_trip_outcome, "day": d,
            }
            a = policy(d, obs, tape)
            if a in (0, 1):                 # R1 or R2
                z = int(tape.Z[d, a])
                t = tau(cell, z)
                load = min(CONVOY_LOAD, S_BATTALION_AMPLE, cell.cssu_cap - inv)
                arrivals.append((h0 + t, load))
                veh_free_at = h0 + 2.0 * t   # mandatory return
                last_trip_outcome = z
                n_trips += 1; veh_busy_h += 2.0 * t
            # HOLD: vehicle stays, no dispatch today
        # serve demand FIFO from inventory
        for o2 in queue:
            if o2.remaining_qty <= 1e-9 or inv <= 0:
                continue
            take = min(inv, o2.remaining_qty)
            inv -= take; o2.remaining_qty -= take
            if o2.remaining_qty <= 1e-9 and o2.OATj is None:
                o2.OATj = h0; o2.CTj = h0 - o2.OPTj
                o2.backorder = bool(o2.CTj > o2.LTj)
        queue = [o2 for o2 in queue if o2.remaining_qty > 1e-9]
    horizon = days * DAY
    n_lost = 0
    for o in orders:
        if o.OATj is None:
            o.lost = True; o.lost_time = float(horizon); n_lost += 1
    info = {"trips": n_trips, "veh_busy_h": veh_busy_h, "lost": n_lost}
    return orders, info


def ret(orders) -> float:
    return float(compute_order_level_ret_excel_formula(orders, j_source="row_index")["mean_ret_excel"])


# ---- policies ----
def const_route(r):
    return lambda d, obs, tape: r

def alternate():
    return lambda d, obs, tape: d % 2

def hold_never():  # always dispatch fastest-a-priori (route 0)
    return lambda d, obs, tape: 0

def signal_threshold():
    """Deployable observable policy: pick the route whose pre-departure signal says 'normal';
    if both say normal or both degraded, keep route 0 (a-priori primary). HOLD if both degraded."""
    def pol(d, obs, tape):
        s = obs["sigma"]
        n0, n1 = (s[0] == 0), (s[1] == 0)
        if n0 and not n1: return 0
        if n1 and not n0: return 1
        if not n0 and not n1: return 0     # both look degraded -> still must move (finite vehicle); primary
        return 0
    return pol

def belief_policy():
    """Bayesian nowcast per route from the signal (accuracy q) toward stationary p_degraded;
    choose the route with lower expected transit."""
    def pol(d, obs, tape):
        cell = tape.cell; s = obs["sigma"]; q = cell.q; pd = cell.p_degraded
        best, ba = 0, 1e18
        for r in range(2):
            # P(degraded | signal) via Bayes
            if s[r] == 1:
                num = q * pd; den = num + (1 - q) * (1 - pd)
            else:
                num = (1 - q) * pd; den = num + q * (1 - pd)
            p_deg = num / max(den, 1e-12)
            exp_tau = cell.tau_base + p_deg * cell.pen
            if exp_tau < ba: ba, best = exp_tau, r
        return best
    return pol

def clairvoyant():
    """Perfect-information routing: pick the truly-faster (normal) route each dispatch."""
    def pol(d, obs, tape):
        z = tape.Z[d]
        if z[0] == z[1]: return 0
        return int(np.argmin(z))            # choose the normal (z=0) route
    return pol

def placebo_threshold(seed):
    """Wrong-route/shuffled placebo: same structure as signal_threshold but on a shuffled signal
    independent of Z -> must not beat the real signal."""
    rng = np.random.default_rng(seed)
    def pol(d, obs, tape):
        fake = rng.integers(0, 2, size=2)
        n0, n1 = (fake[0] == 0), (fake[1] == 0)
        if n0 and not n1: return 0
        if n1 and not n0: return 1
        return 0
    return pol


def eval_policy(tapes, policy):
    rets = np.empty(len(tapes)); trips = np.empty(len(tapes)); lost = np.empty(len(tapes)); busy = np.empty(len(tapes))
    for i, t in enumerate(tapes):
        orders, info = simulate_orders(t, policy)
        rets[i] = ret(orders); trips[i] = info["trips"]; lost[i] = info["lost"]; busy[i] = info["veh_busy_h"]
    return rets, trips, lost, busy


def screen_cell(cell: Cell, n_tapes: int = 80, seed0: int = 9_100_001) -> dict:
    tapes = [materialize(seed0 + i, cell) for i in range(n_tapes)]
    statics = {
        "const_R1": const_route(0), "const_R2": const_route(1), "alternate": alternate(),
    }
    static_eval = {k: eval_policy(tapes, p) for k, p in statics.items()}
    static_vals = {k: v[0] for k, v in static_eval.items()}
    best_static_key = max(static_vals, key=lambda k: static_vals[k].mean())
    static, static_trips, static_lost, static_busy = static_eval[best_static_key]
    oracle, _, _, _ = eval_policy(tapes, clairvoyant())
    obs_thr, thr_trips, thr_lost, thr_busy = eval_policy(tapes, signal_threshold())
    obs_bel, bel_trips, bel_lost, bel_busy = eval_policy(tapes, belief_policy())
    plac, _, _, _ = eval_policy(tapes, placebo_threshold(seed0))
    obs_best_key = "belief" if obs_bel.mean() >= obs_thr.mean() else "threshold"
    obs = obs_bel if obs_best_key == "belief" else obs_thr
    obs_trips = bel_trips if obs_best_key == "belief" else thr_trips
    obs_lost = bel_lost if obs_best_key == "belief" else thr_lost
    obs_busy = bel_busy if obs_best_key == "belief" else thr_busy

    def boot_ci(delta, nb=3000):
        rng = np.random.default_rng(1234)
        idx = rng.integers(0, len(delta), size=(nb, len(delta)))
        bm = delta[idx].mean(axis=1)
        return [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))]

    H_PI = float((oracle - static).mean())
    H_obs = float((obs - static).mean())
    return {
        "cell": cell.__dict__,
        "best_static": best_static_key,
        "static_mean": float(static.mean()),
        "oracle_mean": float(oracle.mean()),
        "H_PI": H_PI, "H_PI_ci95": boot_ci(oracle - static),
        "obs_policy": obs_best_key,
        "H_obs": H_obs, "H_obs_ci95": boot_ci(obs - static),
        "H_obs_threshold": float((obs_thr - static).mean()),
        "H_obs_belief": float((obs_bel - static).mean()),
        "H_obs_placebo": float((plac - static).mean()),
        "eta": float(H_obs / H_PI) if abs(H_PI) > 1e-9 else 0.0,
        "real_beats_placebo": bool(H_obs > (plac - static).mean()),
        # resource + guardrail (adaptive must NOT buy performance with more trips or more lost orders)
        "trips_obs_minus_static": float(obs_trips.mean() - static_trips.mean()),
        "trips_static": float(static_trips.mean()), "trips_obs": float(obs_trips.mean()),
        "lost_obs_minus_static": float(obs_lost.mean() - static_lost.mean()),
        "resource_ok": bool(obs_trips.mean() <= static_trips.mean() + 1e-6),
        "veh_busy_obs_minus_static": float(obs_busy.mean() - static_busy.mean()),
        "veh_busy_static": float(static_busy.mean()), "veh_busy_obs": float(obs_busy.mean()),
        "lost_guardrail_ok": bool(obs_lost.mean() <= static_lost.mean() + 1e-6),
        "n": n_tapes,
    }
