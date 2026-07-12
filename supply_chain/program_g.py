"""Program G V1.2 — two-CSSU shared-transport spatial-commitment study.

Stylized weekly-resolution extension of the thesis MFSC (disclosed, not a reproduction).
One aggregate two-day convoy-equivalent (2500 rations/day = thesis flow) is the scarce
downstream transport shared between CSSU-A and CSSU-B. The weekly action is a dispatch
PRIORITY in {A, B, HOLD} applied to every convoy cycle that week; the convoy never
auto-reorients. Tempo is a semi-Markov latent regime per CSSU; an imperfect balanced-
accuracy signal announces next-week local surge. Physics is exact and CRN-clean so the
open-loop oracle is enumerable (3^H). See contracts/program_g_domain_envelope_v1_2.json.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import itertools
import json
from typing import Any, Callable, Sequence

import numpy as np

from .config import HOURS_PER_WEEK

# --- frozen constants (from config anchors / V1.2 envelope) ---
S1_DAILY = 2564          # RATIONS_PER_SHIFT (config.py:70)
DEMAND_DAYS = 6          # thesis demand days/week
CONVOY_LOAD = 5000       # RATIONS_PER_BATCH (config.py:44)
CYCLES_PER_WEEK = 3      # 168h / 48h round trip -> 3 full cycles
CSSU_CAP = 10000
SB_INITIAL = 10000
TEMPO = ("low", "routine", "surge")
STATIONARY = {"routine": 0.50, "low": 0.25, "surge": 0.25}
MULT = {"low": 0.75, "routine": 1.00}   # surge multiplier is a cell parameter
ACTIONS = ("A", "B", "HOLD")


def digest(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def central_cell() -> dict[str, Any]:
    """The G1 central cell: high signal, lead 1, surge 1.5, short persistence."""
    return {"cell_id": "G1-central", "signal_q": 0.75, "lead_weeks": 1,
            "surge_mult": 1.50, "persistence": "short", "r22_weekly_prob": 0.05}


@dataclass
class Tape:
    seed: int
    weeks: int
    cell: dict[str, Any]
    z: np.ndarray          # (weeks, 2) latent tempo index per CSSU
    demand: np.ndarray     # (weeks, 2) realized weekly demand per CSSU
    signal: np.ndarray     # (weeks, 2) binary surge-alert per CSSU (lead-shifted)
    r22: np.ndarray        # (weeks, 2) route-down flag per CSSU
    tape_id: str = ""
    threat_sha256: str = ""


def _semi_markov(rng: np.random.Generator, weeks: int, persistence: str) -> np.ndarray:
    dwell_set = {"short": (4, 5, 6), "long": (6, 7, 8)}[persistence]
    out = []
    # initial state from stationary distribution
    state = rng.choice(TEMPO, p=[STATIONARY[s] for s in TEMPO])
    while len(out) < weeks:
        dwell = int(rng.choice(dwell_set))
        out.extend([state] * dwell)
        if state == "routine":
            state = rng.choice(("low", "surge"))          # 0.5/0.5
        else:
            state = "routine"
    return np.array([TEMPO.index(s) for s in out[:weeks]], dtype=int)


def _iid_tempo(rng: np.random.Generator, weeks: int) -> np.ndarray:
    return np.array([TEMPO.index(rng.choice(TEMPO, p=[STATIONARY[s] for s in TEMPO]))
                     for _ in range(weeks)], dtype=int)


def materialize_tape(seed: int, cell: dict[str, Any], weeks: int, *, persistent: bool = True) -> Tape:
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x60260712]))
    if persistent:
        z = np.stack([_semi_markov(rng, weeks, cell["persistence"]) for _ in range(2)], axis=1)
    else:
        z = np.stack([_iid_tempo(rng, weeks) for _ in range(2)], axis=1)
    surge_mult = float(cell["surge_mult"])
    demand = np.zeros((weeks, 2), dtype=float)
    for w in range(weeks):
        base = rng.integers(2400, 2601, size=DEMAND_DAYS)   # daily base, exogenous
        for i in range(2):
            m = MULT.get(TEMPO[z[w, i]], surge_mult)         # surge -> cell multiplier
            demand[w, i] = float(np.round(base / 2.0 * m).sum())
    # signal: balanced-accuracy alert that CSSU i is in surge at week w+lead
    q = float(cell["signal_q"]); lead = int(cell["lead_weeks"])
    signal = np.zeros((weeks, 2), dtype=int)
    for w in range(weeks):
        for i in range(2):
            tgt = w + lead
            future_surge = (tgt < weeks) and (TEMPO[z[tgt, i]] == "surge")
            # sens = spec = q
            if future_surge:
                signal[w, i] = 1 if rng.random() < q else 0
            else:
                signal[w, i] = 1 if rng.random() > q else 0
    r22 = (rng.random((weeks, 2)) < float(cell["r22_weekly_prob"])).astype(int)
    tape = Tape(seed=seed, weeks=weeks, cell=cell, z=z, demand=demand, signal=signal, r22=r22)
    tape.tape_id = f"program-g-{cell['cell_id']}-{seed}-{weeks}w"
    tape.threat_sha256 = digest({"z": z.tolist(), "demand": demand.tolist(),
                                 "signal": signal.tolist(), "r22": r22.tolist()})
    return tape


@dataclass
class Result:
    service_loss: float            # total unmet demand (rations) -- primary, lower better
    fill_rate: float               # served / demand
    ret_proxy: float               # 1 - service_loss/demand (disclosed proxy, not ret_excel)
    convoy_missions: int
    worst_cssu_fill: float
    per_week_actions: list[str] = field(default_factory=list)


def simulate(tape: Tape, actions: Sequence[str], *, arm: str = "TRS") -> Result:
    """Weekly-resolution exact dynamics. `arm` toggles the mechanism ladder."""
    inv = np.zeros(2)                                  # CSSU inventories, start empty
    sb = float(SB_INITIAL)
    served = np.zeros(2); demand_tot = np.zeros(2); unmet = np.zeros(2)
    missions = 0
    persistent_convoy = arm in ("TR", "TRS", "TRSC")   # convoy physical memory
    for w in range(tape.weeks):
        sb += S1_DAILY * DEMAND_DAYS                   # production feeds SB
        a = actions[w] if w < len(actions) else "HOLD"
        # convoy cycles available this week (R22 pauses a route's cycles)
        if a in ("A", "B"):
            i = 0 if a == "A" else 1
            down = int(tape.r22[w, i])
            cycles = CYCLES_PER_WEEK - (1 if down else 0)
            if not persistent_convoy:
                cycles = CYCLES_PER_WEEK               # no carry-over penalty, still scarce
            free = CSSU_CAP - inv[i]
            deliver = min(cycles * CONVOY_LOAD, sb, max(0.0, free))
            inv[i] += deliver; sb -= deliver
            missions += max(0, cycles) if deliver > 0 else 0
        # demand realization (both CSSU), served from inventory
        for i in range(2):
            d = tape.demand[w, i]; demand_tot[i] += d
            s = min(inv[i], d); inv[i] -= s; served[i] += s
            unmet[i] += (d - s)
    service_loss = float(unmet.sum())
    dtot = float(demand_tot.sum())
    fills = served / np.maximum(demand_tot, 1.0)
    return Result(service_loss=service_loss, fill_rate=float(served.sum() / max(dtot, 1.0)),
                  ret_proxy=1.0 - service_loss / max(dtot, 1.0), convoy_missions=missions,
                  worst_cssu_fill=float(fills.min()), per_week_actions=list(actions))


def enumerate_oracle(tape: Tape, arm: str = "TRS") -> tuple[float, tuple[str, ...]]:
    """Exact clairvoyant open-loop oracle over 3^weeks sequences (min service-loss)."""
    best = np.inf; best_seq = None
    for seq in itertools.product(ACTIONS, repeat=tape.weeks):
        r = simulate(tape, seq, arm=arm)
        if r.service_loss < best:
            best, best_seq = r.service_loss, seq
    return best, best_seq


def periodic_calendars(weeks: int) -> list[tuple[str, ...]]:
    """All period-1..4 calendars, tiled to `weeks`, deduped."""
    seen = set(); cals = []
    for period in range(1, 5):
        for base in itertools.product(ACTIONS, repeat=period):
            cal = tuple((base * (weeks // period + 1))[:weeks])
            if cal not in seen:
                seen.add(cal); cals.append(cal)
    return cals


def _week_step(inv, sb, a, demand_w, r22_w, persistent):
    inv = inv.copy(); sb = sb + S1_DAILY * DEMAND_DAYS
    if a in ("A", "B"):
        i = 0 if a == "A" else 1
        cycles = CYCLES_PER_WEEK - (1 if (persistent and int(r22_w[i])) else 0)
        deliver = min(cycles * CONVOY_LOAD, sb, max(0.0, CSSU_CAP - inv[i]))
        inv[i] += deliver; sb -= deliver
    unmet = 0.0
    for j in range(2):
        s = min(inv[j], demand_w[j]); inv[j] -= s; unmet += demand_w[j] - s
    return inv, sb, unmet


def cover_signal_policy(tape: Tape, arm: str = "TRS", *, use_signal: bool = True) -> tuple[str, ...]:
    """Observable: dispatch to the CSSU with the lowest projected days-of-cover.
    With use_signal, projected demand uses the surge alert; without, it is cover-blind."""
    acts = []; inv = np.zeros(2); sb = float(SB_INITIAL)
    persistent = arm in ("TR", "TRS", "TRSC")
    surge_mult = float(tape.cell["surge_mult"])
    for w in range(tape.weeks):
        proj = np.array([
            (2600.0 / 2 * DEMAND_DAYS) * (surge_mult if (use_signal and int(tape.signal[w, i])) else 1.0)
            for i in range(2)
        ])
        cover = inv / np.maximum(proj, 1.0)
        a = "A" if cover[0] <= cover[1] else "B"
        acts.append(a)
        inv, sb, _ = _week_step(inv, sb, a, tape.demand[w], tape.r22[w], persistent)
    return tuple(acts)


def mpc_policy(tape: Tape, arm: str = "TRS", horizon: int = 2) -> tuple[str, ...]:
    """Observable receding-horizon: pick the action minimizing expected `horizon`-week
    service loss, using the signal as a surge forecast and a cover follow-on heuristic."""
    acts = []; inv = np.zeros(2); sb = float(SB_INITIAL)
    persistent = arm in ("TR", "TRS", "TRSC")
    surge_mult = float(tape.cell["surge_mult"])

    def forecast(w, i):
        return (2600.0 / 2 * DEMAND_DAYS) * (surge_mult if (w < tape.weeks and int(tape.signal[w, i])) else 1.0)

    for w in range(tape.weeks):
        best_a, best_cost = "HOLD", np.inf
        for a0 in ACTIONS:
            inv_s, sb_s = inv.copy(), sb; cost = 0.0
            for h in range(horizon):
                ww = w + h
                if ww >= tape.weeks:
                    break
                a = a0 if h == 0 else ("A" if (inv_s[0] / max(forecast(ww, 0), 1) <=
                                               inv_s[1] / max(forecast(ww, 1), 1)) else "B")
                dem = np.array([forecast(ww, 0), forecast(ww, 1)])
                r22w = tape.r22[ww] if ww < tape.weeks else np.zeros(2)
                inv_s, sb_s, unmet = _week_step(inv_s, sb_s, a, dem, r22w, persistent)
                cost += unmet
            if cost < best_cost:
                best_cost, best_a = cost, a0
        acts.append(best_a)
        inv, sb, _ = _week_step(inv, sb, best_a, tape.demand[w], tape.r22[w], persistent)
    return tuple(acts)


OBS_KEYS = ("signal_A", "signal_B", "inv_A_frac", "inv_B_frac", "cover_A", "cover_B", "week_phase")


def observe(inv, sb, tape: Tape, w: int) -> np.ndarray:
    """Deployable observation at the START of week w (no latent tempo, no future)."""
    routine_wk = 2600.0 / 2 * DEMAND_DAYS
    return np.array([
        float(tape.signal[w, 0]), float(tape.signal[w, 1]),
        inv[0] / CSSU_CAP, inv[1] / CSSU_CAP,
        inv[0] / routine_wk, inv[1] / routine_wk,
        w / max(tape.weeks - 1, 1),
    ], dtype=np.float32)


def rollout_policy(tape: Tape, policy_fn: Callable[[np.ndarray], int], arm: str = "TRS"):
    """Closed-loop rollout: policy_fn(obs)->action index. Returns (service_loss, obs_rows, acts)."""
    inv = np.zeros(2); sb = float(SB_INITIAL); persistent = arm in ("TR", "TRS", "TRSC")
    unmet = 0.0; rows = []; acts = []
    for w in range(tape.weeks):
        o = observe(inv, sb, tape, w); rows.append(o)
        a_idx = int(policy_fn(o)); acts.append(ACTIONS[a_idx])
        inv, sb, u = _week_step(inv, sb, ACTIONS[a_idx], tape.demand[w], tape.r22[w], persistent)
        unmet += u
    return unmet, rows, acts


def oracle_action_dataset(tape: Tape, arm: str = "TRS"):
    """(observable state at week w, clairvoyant open-loop best action) along the oracle path."""
    _, best_seq = enumerate_oracle(tape, arm=arm)
    inv = np.zeros(2); sb = float(SB_INITIAL); persistent = arm in ("TR", "TRS", "TRSC")
    X, y = [], []
    for w in range(tape.weeks):
        X.append(observe(inv, sb, tape, w)); y.append(ACTIONS.index(best_seq[w]))
        inv, sb, _ = _week_step(inv, sb, best_seq[w], tape.demand[w], tape.r22[w], persistent)
    return X, y


def simulate_orders(tape: Tape, actions: Sequence[str], arm: str = "TRS") -> list:
    """Daily order-level rollout emitting real OrderRecord objects, so the project's
    ret_excel visible-ledger machinery can score Program G under its PRIMARY metric.
    R22 is off in the primary screen -> orders carry no ReT risk indicator -> the
    workbook no-risk fill-rate branch (1 - (Bt+Ut)/j) applies."""
    from .supply_chain import OrderRecord
    from .config import LEAD_TIME_PROMISE
    inv = [0.0, 0.0]; sb = float(SB_INITIAL)
    queues = [[], []]; orders = []; j = 0
    for w in range(tape.weeks):
        a = actions[w] if w < len(actions) else "HOLD"
        pri = 0 if a == "A" else 1 if a == "B" else None
        daily = [tape.demand[w, 0] / DEMAND_DAYS, tape.demand[w, 1] / DEMAND_DAYS]
        for dow in range(DEMAND_DAYS):
            hour = w * HOURS_PER_WEEK + dow * 24.0
            sb += S1_DAILY
            if pri is not None and dow in (1, 3, 5):        # 3 convoy cycles/week
                deliver = min(CONVOY_LOAD, sb, CSSU_CAP - inv[pri])
                inv[pri] += deliver; sb -= deliver
            for i in range(2):
                j += 1
                o = OrderRecord(j=j, OPTj=hour, quantity=daily[i],
                                LTj=float(LEAD_TIME_PROMISE), remaining_qty=daily[i])
                queues[i].append(o); orders.append(o)
            for i in range(2):
                for o in queues[i]:
                    if o.remaining_qty <= 1e-9 or inv[i] <= 0:
                        continue
                    take = min(inv[i], o.remaining_qty)
                    inv[i] -= take; o.remaining_qty -= take
                    if o.remaining_qty <= 1e-9 and o.OATj is None:
                        o.OATj = hour; o.CTj = hour - o.OPTj
                        o.backorder = bool(o.CTj > o.LTj)
                queues[i] = [o for o in queues[i] if o.remaining_qty > 1e-9]
    horizon = tape.weeks * HOURS_PER_WEEK
    for o in orders:                       # unattended at horizon -> lost (increments Ut)
        if o.OATj is None:
            o.lost = True; o.lost_time = float(horizon)
    return orders


def ret_order_metrics(orders) -> dict:
    """Canonical order-level ret_excel (no-risk fill-rate branch) via the REPO aggregator
    (correct cumulative Bt/Ut ordering), plus quantity-weighted ReT_quantity — the clean
    probe of the order-vs-mass reversal. Unattended orders are marked lost -> score 0 -> Ut."""
    from .ret_thesis import (
        compute_order_level_ret_excel_formula, compute_ret_per_order_excel_formula,
        order_counts_as_backorder_for_fill_rate,
    )
    canonical = compute_order_level_ret_excel_formula(orders, j_source="row_index")
    # replicate the canonical loop EXACTLY (increment Bt/Ut BEFORE scoring) to expose per-order
    order_sorted = sorted(orders, key=lambda o: (int(getattr(o, "j", 0) or 0),
                                                 float(getattr(o, "OPTj", 0.0) or 0.0)))
    cb = cu = 0; rv = []; q = []
    for idx, o in enumerate(order_sorted, start=1):
        if bool(getattr(o, "lost", False)):
            cu += 1
        elif order_counts_as_backorder_for_fill_rate(o, current_time=None):
            cb += 1
        ret, _ = compute_ret_per_order_excel_formula(o, j=idx, cumulative_backorders=cb,
                                                     cumulative_unattended=cu)
        rv.append(ret); q.append(float(getattr(o, "quantity", 0.0) or 0.0))
    rv = np.array(rv); q = np.array(q)
    assert abs(float(rv.mean()) - float(canonical["mean_ret_excel"])) < 1e-9, "canonical mismatch"
    return {
        "ret_order": float(canonical["mean_ret_excel"]),         # canonical, each order equal
        "ret_quantity": float((rv * q).sum() / max(q.sum(), 1e-9)),  # each ration weighted equally
        "attended": int(sum(getattr(o, "OATj", None) is not None for o in orders)),
        "lost": int(sum(bool(getattr(o, "lost", False)) for o in orders)),
    }


# Cobb-Douglas-INSPIRED index (repo ReT_garrido2024 exponents; NOT paper-calibrated for MFSC).
# The published G24 coefficients were fit for FACTORY resilience (APP), not this military SC; used
# here only as a fixed preference profile for a construct-sensitivity probe, NOT "Garrido 2024 exact".
G24 = {"a_zeta": 0.0240, "b_epsilon": 0.0260, "c_phi": 0.0400, "d_tau": 0.0600, "n_kappa": 0.1771}
_ROUTINE_WK = 2600.0 / 2 * DEMAND_DAYS


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def metrics_all(tape: Tape, actions: Sequence[str], arm: str = "TRS") -> dict:
    """One trajectory, many lenses. Runs the daily order adapter once and returns
    service-loss, ret_excel inputs, and the repo's Cobb-Douglas index (frozen G24
    exponents) plus a spatial geometric-mean variant. phi (spare capacity) and
    kappa (cost) are held CONSTANT in v1.2 (S1 fixed, resources matched) -- disclosed;
    so the index is driven by zeta (inventory), epsilon (backorders), tau (fulfil time)."""
    from .supply_chain import OrderRecord
    from .config import LEAD_TIME_PROMISE
    inv = [0.0, 0.0]; sb = float(SB_INITIAL); queues = [[], []]; orders = []; j = 0
    inv_time = 0.0; bo_units_time = [0.0, 0.0]; days = 0
    for w in range(tape.weeks):
        a = actions[w] if w < len(actions) else "HOLD"
        pri = 0 if a == "A" else 1 if a == "B" else None
        daily = [tape.demand[w, 0] / DEMAND_DAYS, tape.demand[w, 1] / DEMAND_DAYS]
        for dow in range(DEMAND_DAYS):
            hour = w * HOURS_PER_WEEK + dow * 24.0; sb += S1_DAILY
            if pri is not None and dow in (1, 3, 5):
                d = min(CONVOY_LOAD, sb, CSSU_CAP - inv[pri]); inv[pri] += d; sb -= d
            for i in range(2):
                j += 1; o = OrderRecord(j=j, OPTj=hour, quantity=daily[i],
                    LTj=float(LEAD_TIME_PROMISE), remaining_qty=daily[i])
                queues[i].append(o); orders.append(o)
            for i in range(2):
                for o in queues[i]:
                    if o.remaining_qty <= 1e-9 or inv[i] <= 0:
                        continue
                    take = min(inv[i], o.remaining_qty); inv[i] -= take; o.remaining_qty -= take
                    if o.remaining_qty <= 1e-9 and o.OATj is None:
                        o.OATj = hour; o.CTj = hour - o.OPTj; o.backorder = bool(o.CTj > o.LTj)
                bo_units_time[i] += sum(o.remaining_qty for o in queues[i])
                queues[i] = [o for o in queues[i] if o.remaining_qty > 1e-9]
            inv_time += inv[0] + inv[1] + sb; days += 1
    horizon = tape.weeks * HOURS_PER_WEEK
    for o in orders:
        if o.OATj is None:
            o.lost = True; o.lost_time = float(horizon)
    # per-CSSU aggregates
    attended = [[o for o in orders if o.OATj is not None and (o.j % 2 == (1 - i))] for i in range(2)]
    ct = [np.mean([o.CTj for o in attended[i]]) if attended[i] else float(LEAD_TIME_PROMISE) for i in range(2)]
    unmet = [sum(o.remaining_qty for o in orders if o.OATj is None and (o.j % 2 == (1 - i))) for i in range(2)]
    demand_i = [tape.demand[:, i].sum() for i in range(2)]
    zeta = inv_time / max(days, 1) / _ROUTINE_WK
    eps = [bo_units_time[i] / max(days, 1) / _ROUTINE_WK + 1e-3 for i in range(2)]
    tau = [max(ct[i], 1.0) / float(LEAD_TIME_PROMISE) for i in range(2)]
    phi, kappa = 1.0, 1.0                                  # held constant (disclosed)
    # paper-faithful aggregate CD
    eps_agg = float(np.mean(eps)); tau_agg = float(np.mean(tau))
    Z = (G24["a_zeta"] * np.log(zeta + 1e-3) - G24["b_epsilon"] * np.log(eps_agg)
         + G24["c_phi"] * np.log(phi) - G24["d_tau"] * np.log(tau_agg) - G24["n_kappa"] * np.log(kappa))
    # spatial geometric mean (penalizes abandoning one CSSU)
    zeta_i = [zeta, zeta]  # inventory pooled at SB; use aggregate for both (disclosed)
    R_i = [(zeta_i[i] + 1e-3) ** G24["a_zeta"] * eps[i] ** (-G24["b_epsilon"]) * tau[i] ** (-G24["d_tau"]) for i in range(2)]
    R_spatial = float(np.sqrt(R_i[0] * R_i[1]) * phi ** G24["c_phi"] * kappa ** (-G24["n_kappa"]))
    ret = ret_order_metrics(orders)
    return {
        "unfulfilled_rations_at_horizon": float(sum(unmet)),
        # Compatibility alias only; this endpoint is not a time-integrated service-loss AUC.
        "service_loss": float(sum(unmet)),
        "orders": orders,
        "ret_order": ret["ret_order"], "ret_quantity": ret["ret_quantity"],
        "attended_orders": ret["attended"], "lost_orders": ret["lost"],
        "worst_cssu_fill": float(min(1.0 - unmet[i] / max(demand_i[i], 1.0) for i in range(2))),
        "cd_raw_Z": float(Z), "cd_sigmoid": float(_sigmoid(Z)), "cd_spatial": R_spatial,
        "zeta": float(zeta), "epsilon": eps_agg, "tau": tau_agg,
    }


def signal_hysteresis_policy(tape: Tape) -> tuple[str, ...]:
    """Observable: send convoy to the CSSU whose signal fires; tie/none -> lower inventory."""
    acts = []
    inv = np.zeros(2); sb = float(SB_INITIAL)
    for w in range(tape.weeks):
        sb += S1_DAILY * DEMAND_DAYS
        sa, sb_ = int(tape.signal[w, 0]), int(tape.signal[w, 1])
        if sa and not sb_:
            a = "A"
        elif sb_ and not sa:
            a = "B"
        else:
            a = "A" if inv[0] <= inv[1] else "B"
        acts.append(a)
        # shadow inventory update to inform tie-breaks
        i = 0 if a == "A" else 1
        deliver = min(CYCLES_PER_WEEK * CONVOY_LOAD, sb, CSSU_CAP - inv[i])
        inv[i] += deliver; sb -= deliver
        for j in range(2):
            inv[j] = max(0.0, inv[j] - tape.demand[w, j])
    return tuple(acts)
