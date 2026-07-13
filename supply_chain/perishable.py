"""Paper 2b / Program K — perishable replenishment: the legitimate test of WHEN RL IS warranted.

Every closed lane failed for one recorded reason: "unlimited storage + no holding cost + weekly-mean
ReT make always-max feasible" -> no observable state can matter. This lane REMOVES that unrealistic
assumption (it is a military FOOD supply chain: rations perish). With finite storage + perishability,
hoarding is no longer free (perished stock is scrapped and blocks capacity), so a state-feedback
policy that replenishes in response to an OBSERVABLE leading demand signal -- within the shelf-life
window -- genuinely beats the best fixed schedule. This is the classic (s,S) result and the honest
flip side of the paper: it delineates the RL-warranted boundary from BOTH sides.

Objective J = service_loss + lambda * waste (a REAL perishability cost; lambda must be Garrido-real,
frozen before any RL). CRN: demand regime + noise + signal are fixed per tape; only realized
inventory depends on the ordering action. No metric is chosen to make the learner win; the win (if
any) is conditional on the waste cost being a genuine operational fact.
"""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
import itertools, json
import numpy as np

D0 = 15000.0                        # ~ weekly S1 demand (2564/day * 6), the demand scale
ORDER_LEVELS = (0.0, 0.5, 1.0, 1.5)  # order quantity as a multiple of D0 (4 discrete actions)
ACTIONS = tuple(f"Q{int(x*100)}" for x in ORDER_LEVELS)


def digest(v):
    return sha256(json.dumps(v, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def central_cell():
    return {"cell_id": "k-central", "shelf_life": 2, "cap_mult": 2.0, "signal_q": 0.80,
            "surge_mult": 1.6, "lam": 0.5, "persist": 0.7}


@dataclass
class PTape:
    seed: int
    weeks: int
    cell: dict
    demand: np.ndarray      # (weeks,) realized demand
    signal: np.ndarray      # (weeks,) noisy 1-wk-ahead demand signal (observable)
    sha: str = ""


def materialize_tape(seed, cell, weeks=8):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x9E12]))
    # persistent calm/surge regime (semi-Markov via a sticky Bernoulli)
    surge = np.zeros(weeks, dtype=int); s = 0
    for w in range(weeks):
        if rng.random() > cell["persist"]:
            s = 1 - s
        surge[w] = s
    mean = np.where(surge == 1, cell["surge_mult"], 1.0) * D0
    demand = np.clip(rng.normal(mean, 0.10 * D0), 0.3 * D0, None)
    # observable 1-week-ahead signal: noisy forecast of NEXT week's demand (balanced quality signal_q)
    q = cell["signal_q"]
    nxt = np.concatenate([demand[1:], demand[-1:]])
    noise = rng.normal(0, (1 - q) * 0.6 * D0, size=weeks)
    signal = np.clip(nxt + noise, 0.0, None)
    t = PTape(seed=seed, weeks=weeks, cell=cell, demand=demand, signal=signal)
    t.sha = digest({"d": demand.tolist(), "s": signal.tolist()})
    return t


def week_step(tape, w, order_units, inv):
    """Shared weekly physics. `order_units` in multiples of D0. `inv` = age-bucketed array
    (index 0 = freshest ... index shelf_life-1 = oldest). Returns (inv_new, weekly_sl, weekly_waste).
    Order placed this week arrives THIS week (lead folded in), capacity-capped, perishable, FIFO issue.
    """
    L = tape.cell["shelf_life"]; cap = tape.cell["cap_mult"] * D0
    inv = np.asarray(inv, float).copy()
    # 1) receive order into freshest bucket, capped by remaining storage capacity
    room = max(0.0, cap - inv.sum())
    recv = min(order_units * D0, room)
    inv[0] += recv
    # 2) serve demand FIFO from OLDEST first (minimize waste)
    dem = tape.demand[w]; need = dem
    for a in range(L - 1, -1, -1):
        take = min(inv[a], need); inv[a] -= take; need -= take
    weekly_sl = float(need)                       # unmet demand
    # 3) age: oldest bucket perishes (scrapped -> waste), shift the rest up one age
    weekly_waste = float(inv[L - 1])
    inv[1:] = inv[:-1]; inv[0] = 0.0
    return inv, weekly_sl, weekly_waste


@dataclass
class PResult:
    service_loss: float
    waste: float
    J: float


def simulate(tape, order_seq):
    L = tape.cell["shelf_life"]; lam = tape.cell["lam"]
    inv = np.zeros(L); sl = 0.0; wst = 0.0
    for w in range(tape.weeks):
        q = ORDER_LEVELS[order_seq[w]] if isinstance(order_seq[w], (int, np.integer)) else \
            ORDER_LEVELS[ACTIONS.index(order_seq[w])]
        inv, wsl, ww = week_step(tape, w, q, inv)
        sl += wsl; wst += ww
    return PResult(service_loss=float(sl), waste=float(wst), J=float(sl + lam * wst))


def enumerate_oracle(tape):
    """Exact clairvoyant open-loop oracle: min J over 4^weeks order sequences."""
    best = np.inf; bseq = None
    for seq in itertools.product(range(len(ORDER_LEVELS)), repeat=tape.weeks):
        j = simulate(tape, seq).J
        if j < best:
            best, bseq = j, seq
    return best, bseq


def constant_and_periodic(weeks, max_period=4):
    seen = set(); cals = []
    for p in range(1, max_period + 1):
        for base in itertools.product(range(len(ORDER_LEVELS)), repeat=p):
            cal = tuple((base * (weeks // p + 1))[:weeks])
            if cal not in seen:
                seen.add(cal); cals.append(cal)
    return cals


def basestock_policy(tape, S_units):
    """Observable state-feedback: order up to a target level S (in D0 units), adjusted by the
    observable leading demand signal. Uses ONLY observable inventory + signal (no future access)."""
    L = tape.cell["shelf_life"]; inv = np.zeros(L); acts = []
    for w in range(tape.weeks):
        on_hand = inv.sum() / D0
        target = S_units + (tape.signal[w] / D0 - 1.0)      # raise target when signal predicts surge
        q = np.clip(target - on_hand, 0.0, ORDER_LEVELS[-1])
        a = int(np.argmin([abs(q - x) for x in ORDER_LEVELS]))
        acts.append(a)
        inv, _, _ = week_step(tape, w, ORDER_LEVELS[a], inv)
    return tuple(acts)
