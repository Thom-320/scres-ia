"""Brown-Smith-Sun (2010) penalized information-relaxation bound for Program Q (burned calibration tapes only).

Components
- exogenous model of a Program O-R tape (verified elsewhere): only `order_products` (2-state Markov
  regime, persistence rho, dominant share s; supply_chain.program_o_full_des.product_demand_tape) and
  `order_quantities` (iid U{2400..2600} from the dedicated demand stream) vary across seeds.
- exact Bayes filter of the latent regime given the labels revealed before decision t (orders of weeks < t);
  the conditional sampler draws the future (weeks >= t) from the true cell law given that history.
- vectorized classical controller (max_pressure__P, min_cost_flow__K) replaying
  supply_chain.program_o_state_rich.state_rich_calendar exactly, with the first L actions overridden.
"""
from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np

# Default: the repo this artifact lives in (results/program_q/<artifact>/penalized/scripts/).
REPO = Path(os.environ.get("SCRES_REPO", Path(__file__).resolve().parents[5]))
PRODUCTS = ("P_C", "P_H")
QMIN, QMAX = 2400, 2600
WEEKS, PER_WEEK = 8, 6


def setup(repo: Path = REPO):
    import sys
    sys.path.insert(0, str(repo))


def scheduler(repo: Path = REPO):
    p = json.loads((repo / "contracts/program_o_full_des_hpi_translation_v1.json").read_text())
    return p["action"]["within_week_schedulers"][p["action"]["primary_scheduler"]]


def enc(cal: Sequence[int]) -> int:
    i = 0
    for a in cal:
        i = i * 4 + int(a)
    return i


def all_prefixes(L: int) -> np.ndarray:
    """(4**L, L) uint8 in lexicographic order == base-4 index order."""
    if L == 0:
        return np.zeros((1, 0), dtype=np.uint8)
    idx = np.arange(4 ** L)
    return np.stack([(idx >> (2 * (L - 1 - k))) & 3 for k in range(L)], 1).astype(np.uint8)


# ----------------------------------------------------------------------------- exogenous model
def labels_to_bits(products: Sequence[str]) -> np.ndarray:
    return np.asarray([1 if p == "P_C" else 0 for p in products], dtype=np.int8)  # 1 = P_C


def regime_posterior_c(label_bits: np.ndarray, t: int, rho: float, share: float) -> float:
    """P(regime of week t == P_C | labels of weeks 0..t-1), exact forward filter, true cell law."""
    b = 0.5
    for k in range(t):
        w = label_bits[k * PER_WEEK:(k + 1) * PER_WEEK]
        nc = int(w.sum()); nh = PER_WEEK - nc
        lc = share ** nc * (1 - share) ** nh
        lh = share ** nh * (1 - share) ** nc
        b = b * lc / (b * lc + (1 - b) * lh)
        b = rho * b + (1 - rho) * (1 - b)
    return b


def sample_future(rng: np.random.Generator, label_bits: np.ndarray, quantities: np.ndarray, t: int,
                  rho: float, share: float, m: int) -> tuple[np.ndarray, np.ndarray]:
    """m draws of (label_bits, quantities) equal to the realized history for weeks < t and drawn from
    the conditional law for weeks >= t."""
    bc = regime_posterior_c(label_bits, t, rho, share)
    L = np.repeat(label_bits[None, :], m, 0).copy()
    Q = np.repeat(np.asarray(quantities, dtype=float)[None, :], m, 0).copy()
    if t >= WEEKS:
        return L, Q
    reg = (rng.random(m) < bc).astype(np.int8)  # 1 = P_C
    for k in range(t, WEEKS):
        dom = rng.random((m, PER_WEEK)) < share
        L[:, k * PER_WEEK:(k + 1) * PER_WEEK] = np.where(dom, reg[:, None], 1 - reg[:, None])
        switch = rng.random(m) > rho
        reg = np.where(switch, 1 - reg, reg).astype(np.int8)
    Q[:, t * PER_WEEK:] = rng.integers(QMIN, QMAX + 1, size=(m, (WEEKS - t) * PER_WEEK)).astype(float)
    return L, Q


def make_skeleton(base, label_bits: np.ndarray, quantities: np.ndarray):
    return replace(base, order_products=tuple("P_C" if b else "P_H" for b in label_bits),
                   order_quantities=tuple(float(q) for q in quantities))


# ----------------------------------------------------------------------------- classical controller
def _policy_rows(sk, cfg_id: str):
    from supply_chain.program_o_hobs import observable_calendar
    from supply_chain.program_o_state_rich import _forecast_shares
    _u, belief_rows = observable_calendar(
        request_times=np.asarray(sk.order_times, float), request_products=tuple(sk.order_products),
        decision_start=float(sk.decision_start), decision_weeks=int(sk.decision_weeks),
        policy_id="belief_extreme_v1", initial_action=1, regime_persistence=0.75, dominant_share=0.90)
    pol, par = cfg_id.split("__")
    par = int(par)
    rows = []
    for w in range(int(sk.decision_weeks)):
        br = belief_rows[w]
        extra = None
        if pol == "min_cost_flow":
            shares = _forecast_shares(br.belief_c, regime_persistence=0.75, dominant_share=0.90, horizon=par)
            extra = 15000.0 * sum(shares)
        rows.append((float(br.belief_c), float(br.predicted_share_c), extra))
    return pol, par, rows


def classical_continuations(sk, prefixes: np.ndarray, cfg_id: str, sched) -> np.ndarray:
    """Vectorized state_rich_calendar with actions[:L] = prefixes (n, L); returns (n, 8) uint8."""
    n, L = prefixes.shape
    weeks = int(sk.decision_weeks)
    if L >= weeks:
        return prefixes[:, :weeks].astype(np.uint8)
    pol, par, prow = _policy_rows(sk, cfg_id)
    counts_c = np.asarray([sum(x == "P_C" for x in sched[str(a)]) for a in range(4)], dtype=float)
    lab_c = np.asarray([[1 if x == "P_C" else 0 for x in sched[str(a)]] for a in range(4)], dtype=np.int64)
    start = float(sk.decision_start)
    order_times = np.asarray(sk.order_times, float)
    q = np.asarray(sk.order_quantities, float)
    prod = np.asarray([0 if p == "P_C" else 1 for p in sk.order_products], dtype=np.int64)
    no = len(order_times)
    inv = np.repeat(np.asarray(sk.opening_inventory, float)[None, :], n, 0)
    pending = np.zeros((n, no), bool)
    created = np.zeros(no, bool)
    actions = np.zeros((n, weeks), dtype=np.int64)
    release_slots = tuple(map(float, sk.release_slots))
    rel_avail = tuple(map(bool, sk.release_available or [True] * len(release_slots)))
    batch = [(float(a), int(w), int(p)) for a, w, p in sk.batch_arrivals]
    events = [(start + 168.0 * w, -1, "decision", w) for w in range(weeks)]
    events += [(float(t), 0, "release", i) for i, t in enumerate(release_slots)]
    events += [(t, 1, "batch", (w, p)) for t, w, p in batch]
    events += [(float(t), 2, "demand", i) for i, t in enumerate(order_times)]
    contingent = np.asarray(sk.order_contingent or [False] * no, dtype=bool)
    prio = np.lexsort((np.arange(no), order_times, q, 1 - contingent.astype(np.uint8)))
    q_p, prod_p = q[prio], prod[prio]
    rows_idx = np.arange(n)
    onehot = np.stack([prod == 0, prod == 1], 1).astype(float)  # (no, 2)
    for now, _pr, kind, payload in sorted(events):
        if kind == "decision":
            w = int(payload)
            if w < L:
                actions[:, w] = prefixes[:, w]
                continue
            locked = np.zeros((n, 2))
            for at, tw, pos in batch:
                if tw >= w or at < now - 1e-12:
                    continue
                isc = lab_c[actions[:, tw], pos]  # 1 -> P_C
                locked[:, 0] += 5000.0 * isc
                locked[:, 1] += 5000.0 * (1 - isc)
            backlog = (pending * q[None, :]) @ onehot
            position = inv + locked - backlog
            belief_c, ps, extra = prow[w]
            prev = actions[:, w - 1] if w > 0 else None
            if pol == "max_pressure":
                d0, d1 = 15000.0 * ps, 15000.0 * (1.0 - ps)
                p0, p1 = d0 - position[:, 0], d1 - position[:, 1]
                diff = p0 - p1
                pos0, pos1 = np.maximum(p0, 0.0), np.maximum(p1, 0.0)
                tot = pos0 + pos1
                share_c = np.where(tot > 1e-12, pos0 / np.where(tot > 1e-12, tot, 1.0), 0.5)
                obj = np.abs(counts_c[None, :] / 3.0 - share_c[:, None])  # (n,4)
                best = obj.min(1, keepdims=True)
                act = np.argmax(obj == best, 1)
                if prev is not None:
                    keep = np.abs(diff) <= float(par)
                    act = np.where(keep, prev, act)
            elif pol == "min_cost_flow":
                tc = extra
                target = np.asarray([tc, 15000.0 * par - tc])
                post = position[:, None, :] + 5000.0 * np.stack([counts_c, 3.0 - counts_c], 1)[None, :, :]
                short = np.maximum(0.0, target[None, None, :] - post)
                exc = np.maximum(0.0, post - target[None, None, :])
                pa = prev if prev is not None else np.ones(n, dtype=np.int64)
                keys = [short[:, :, 0] + short[:, :, 1], short.max(2), exc[:, :, 0] + exc[:, :, 1],
                        np.abs(np.arange(4)[None, :] - pa[:, None]).astype(float)]
                cand = np.ones((n, 4), bool)
                for k in keys:
                    kk = np.where(cand, k, np.inf)
                    cand &= kk == kk.min(1, keepdims=True)
                act = np.argmax(cand, 1)
            else:
                raise ValueError(cfg_id)
            actions[:, w] = act
            continue
        if kind == "batch":
            w, pos = payload
            isc = lab_c[actions[:, w], pos]
            inv[:, 0] += 5000.0 * isc
            inv[:, 1] += 5000.0 * (1 - isc)
            continue
        if kind == "demand":
            created[payload] = True
            pending[:, payload] = True
            continue
        if not rel_avail[payload]:
            continue
        pend_p = pending[:, prio] & created[prio][None, :]
        feas = pend_p & (inv[:, prod_p] + 1e-9 >= q_p[None, :])
        has = feas.any(1)
        if not has.any():
            continue
        first = np.argmax(feas, 1)
        r = rows_idx[has]; o = prio[first[has]]
        inv[r, prod[o]] -= q[o]
        pending[r, o] = False
    return actions.astype(np.uint8)


def reference_continuation(sk, prefix: Sequence[int], cfg_id: str, sched) -> tuple[int, ...]:
    """Repo state_rich_calendar with the first len(prefix) decisions overridden (monkeypatch)."""
    import supply_chain.program_o_state_rich as sr
    pol, par = cfg_id.split("__")
    cfg = sr.StateRichConfiguration(pol, int(par))
    orig = sr.choose_state_rich_action
    pre = tuple(int(a) for a in prefix)

    def patched(observation, config, **kw):
        if observation.week < len(pre):
            return pre[observation.week], (), (pre[observation.week],)
        return orig(observation, config, **kw)
    sr.choose_state_rich_action = patched
    try:
        cal, _ = sr.state_rich_calendar(skeleton=sk.as_dict(), scheduler=sched, config=cfg,
                                        regime_persistence=0.75, dominant_share=0.90)
    finally:
        sr.choose_state_rich_action = orig
    return tuple(cal)
