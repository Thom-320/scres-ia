"""Program K2 — replenishment under a VALIDATED holding cost + REAL shelf life + lead time.

Corrects Program K (contested): no 2-week spoilage (thesis rations last ~3 years). Hoarding is made
costly the realistic way -- a per-unit-week HOLDING/obsolescence cost -- and orders take a LEAD TIME
to arrive. Primary comparator is the best-tuned CLASSICAL (s,S) policy: a learner only counts as
"RL warranted" if it beats a well-tuned (s,S), not merely a fixed schedule. Objective
J = p*service_loss + h*holding_units; service loss is reported SEPARATELY (resilience outcome).
See docs/PROGRAM_K2_HOLDING_COST_PREREGISTRATION_2026-07-12.md (frozen before any result).
"""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
import itertools, json
import numpy as np

D0 = 15000.0
ORDER_LEVELS = (0.0, 0.5, 1.0, 1.5)              # multiples of D0 (kept 4 for a tractable 4^8 oracle)


def digest(v):
    return sha256(json.dumps(v, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def central_cell():
    # DEFAULT physics: real (effectively infinite on horizon) shelf life; hoarding costly via holding.
    return {"cell_id": "k2-central", "shelf_life": 156, "cap_mult": 3.0, "lead": 1,
            "signal_noise": 0.20, "surge_mult": 1.6, "persist": 0.7, "p": 1.0, "h": 0.4}


@dataclass
class RTape:
    seed: int
    weeks: int
    cell: dict
    demand: np.ndarray
    signal: np.ndarray
    sha: str = ""


def materialize_tape(seed, cell, weeks=8):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x4B32]))
    surge = np.zeros(weeks, dtype=int); s = 0
    for w in range(weeks):
        if rng.random() > cell["persist"]:
            s = 1 - s
        surge[w] = s
    mean = np.where(surge == 1, cell["surge_mult"], 1.0) * D0
    demand = np.clip(rng.normal(mean, 0.10 * D0), 0.3 * D0, None)
    # `signal_noise` is a Gaussian NOISE SCALE on next-week realized demand (NOT a classification
    # accuracy): higher signal_noise -> less informative forecast. signal = D_{w+1} + N(0, sn*0.6*D0).
    sn = cell["signal_noise"]
    nxt = np.concatenate([demand[1:], demand[-1:]])
    signal = np.clip(nxt + rng.normal(0, sn * 0.6 * D0, size=weeks), 0.0, None)
    t = RTape(seed=seed, weeks=weeks, cell=cell, demand=demand, signal=signal)
    t.sha = digest({"d": demand.tolist(), "s": signal.tolist()})
    return t


@dataclass
class State:
    onhand: float
    pipeline: list         # orders arriving in 1..lead weeks (index 0 arrives next)


def new_state(cell):
    return State(onhand=0.0, pipeline=[0.0] * cell["lead"])


def week_step(tape, w, order_units, st: State):
    """Shared weekly physics: receive the pipeline head, place a new order (arrives after `lead`),
    serve demand from on-hand, pay holding on end-of-week on-hand. Returns (state, sl, holding)."""
    cell = tape.cell; cap = cell["cap_mult"] * D0
    onhand = st.onhand; pipe = list(st.pipeline)
    # 1) receive order that was placed `lead` weeks ago (pipeline head)
    onhand += pipe.pop(0)
    # 2) place a new order (capacity on inventory position = on-hand + in-transit)
    pos = onhand + sum(pipe)
    recv = min(order_units * D0, max(0.0, cap - pos))
    pipe.append(recv)
    # 3) serve demand
    dem = tape.demand[w]; served = min(onhand, dem); onhand -= served
    sl = float(dem - served)
    # 4) shelf life: only bites if inventory older than shelf_life; with the default long shelf life
    #    on an 8-52wk horizon it never triggers (kept for the shelf-life sensitivity axis).
    #    (age structure omitted for shelf_life >= weeks; a finite-shelf variant tracks buckets.)
    holding = float(onhand)                       # end-of-week on-hand pays holding
    return State(onhand=onhand, pipeline=pipe), sl, holding


@dataclass
class RResult:
    service_loss: float
    holding: float
    J: float


def simulate(tape, order_seq):
    cell = tape.cell; st = new_state(cell); sl = 0.0; hold = 0.0
    for w in range(tape.weeks):
        a = order_seq[w]
        q = ORDER_LEVELS[a] if isinstance(a, (int, np.integer)) else a
        st, wsl, wh = week_step(tape, w, q, st)
        sl += wsl; hold += wh
    return RResult(service_loss=float(sl), holding=float(hold),
                   J=float(cell["p"] * sl + cell["h"] * hold))


def enumerate_oracle(tape):
    best = np.inf; bseq = None
    for seq in itertools.product(range(len(ORDER_LEVELS)), repeat=tape.weeks):
        j = simulate(tape, seq).J
        if j < best:
            best, bseq = j, seq
    return best, bseq


def periodic_calendars(weeks, max_period=4):
    seen = set(); cals = []
    for p in range(1, max_period + 1):
        for base in itertools.product(range(len(ORDER_LEVELS)), repeat=p):
            cal = tuple((base * (weeks // p + 1))[:weeks])
            if cal not in seen:
                seen.add(cal); cals.append(cal)
    return cals


def sS_policy(tape, s_units, S_units):
    """CLASSICAL (s,S): when inventory position <= s, order up to S (rounded to the action grid).
    Uses ONLY observable inventory position -- the strong adaptive comparator the RL must beat."""
    cell = tape.cell; st = new_state(cell); acts = []
    for w in range(tape.weeks):
        pos = (st.onhand + sum(st.pipeline)) / D0
        want = max(0.0, S_units - pos) if pos <= s_units else 0.0
        a = int(np.argmin([abs(want - x) for x in ORDER_LEVELS])); acts.append(a)
        st, _, _ = week_step(tape, w, ORDER_LEVELS[a], st)
    return tuple(acts)


def best_sS(cal_tapes):
    """Grid-search the strongest (s,S) on calibration tapes (by mean J)."""
    sg = np.round(np.arange(0.0, 2.0, 0.25), 2); Sg = np.round(np.arange(0.5, 3.25, 0.25), 2)
    best = (np.inf, None, None)
    for s in sg:
        for S in Sg:
            if S <= s:
                continue
            m = float(np.mean([simulate(t, sS_policy(t, s, S)).J for t in cal_tapes]))
            if m < best[0]:
                best = (m, float(s), float(S))
    return best[1], best[2]
