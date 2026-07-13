"""Paper 2 — adaptive maintenance on Op5-Op7 (condition-based, single shared crew).

Stylized serial-WIP extension (disclosed, thesis-anchored: Op5-Op7 serial, R11 on Op5/6,
R14 on Op7, one 24h weekly maintenance block). The decisive NEW structure vs the closed lanes:
a PERSISTENT, OBSERVABLE, controllable degradation state per station + a SINGLE shared crew ->
real intertemporal opportunity cost, and R11 becomes ENDOGENOUS (degradation decides whether the
exogenous threat becomes a failure). Weekly action = give the 24h crew to PM5/PM6/PM7. Corrective
repair preempts PM (single crew). CRN: exogenous wear-noise, threat and demand are fixed per tape;
only realized damage depends on the action. Metric = service fill (ret proxy; ret_excel port later).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import itertools
import json
from typing import Any

import numpy as np

STATIONS = 3                    # Op5, Op6, Op7
ACTIONS = ("PM5", "PM6", "PM7")
CAP = 2564.0                    # station weekly capacity (RATIONS_PER_SHIFT-ish), S1
DEMAND_WK = 2500.0 * 6 / 3      # downstream weekly draw (~ per the 3-echelon split, stylized)
WIP_UNIT = 2564.0              # 1 "day of S1 production" WIP unit
PM_DOWN = 24.0 / 168.0         # fraction of the week a maintained/failed station is down


def digest(v):
    return sha256(json.dumps(v, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def central_cell():
    return {"cell_id": "m-central", "sensor_q": 0.75, "pm_efficacy": 0.50, "wip_days": 2,
            "wear_hetero": "high", "r11_level": "current"}


@dataclass
class MTape:
    seed: int
    weeks: int
    cell: dict
    wear: np.ndarray       # (weeks, 3) exogenous wear increments
    threat: np.ndarray     # (weeks, 3) R11 exogenous threat fires
    demand: np.ndarray     # (weeks,)
    tape_id: str = ""
    sha: str = ""


def materialize_tape(seed, cell, weeks=12):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x5A17CE]))
    base = {"low": 0.05, "high": 0.05}[cell["wear_hetero"]]
    # heterogeneous mean wear per station
    hetero = {"low": np.array([1.0, 1.0, 1.0]), "high": np.array([1.6, 1.0, 0.6])}[cell["wear_hetero"]]
    wear = np.clip(rng.normal(base * hetero, 0.02, size=(weeks, 3)), 0.0, None)
    r11_b = {"current": 168.0, "increased": 42.0}[cell["r11_level"]]   # thesis window (hours)
    p_threat = min(1.0, 168.0 / r11_b * 0.30)                          # weekly threat prob per station
    threat = (rng.random((weeks, 3)) < p_threat).astype(int)
    demand = np.round(rng.uniform(2400, 2600, size=weeks) * 6 / 3)
    t = MTape(seed=seed, weeks=weeks, cell=cell, wear=wear, threat=threat, demand=demand)
    t.tape_id = f"maint-{cell['cell_id']}-{seed}"
    t.sha = digest({"wear": wear.tolist(), "threat": threat.tolist(), "demand": demand.tolist()})
    return t


def _hazard(d):
    return float(np.clip((d - 0.4) / 0.6, 0.0, 1.0))    # degradation>0.4 starts converting threat->failure


@dataclass
class MResult:
    service_loss: float
    fill_rate: float
    worst_station_downtime: float
    crew_hours: float
    per_week_actions: list = field(default_factory=list)


def week_step(tape: MTape, w, action, d, wip, *, degrade=True):
    """Single shared weekly physics step (env and oracle both call this -> identical dynamics).

    Mutates copies of (d, wip); returns (d_new, wip_new, weekly_service_loss, down, crew_hours).
    `action` is an int in {0,1,2}. CRN: tape.threat/wear/demand fixed; only realized damage
    depends on the action (which station the single crew serves).
    """
    cell = tape.cell; eff = cell["pm_efficacy"]; Wcap = cell["wip_days"] * WIP_UNIT
    d = d.copy(); wip = wip.copy(); a = int(action); down = np.zeros(3); crew = 0.0
    # 1) corrective repairs (R11 endogenous): exogenous threat becomes a failure iff degraded > TAU.
    #    Corrective repair preempts the single crew, so PM is skipped that week (opportunity cost).
    repaired = False
    for i in range(3):
        if degrade and tape.threat[w, i] and d[i] > TAU:
            rep = min(1.0, 0.5 + d[i])            # full-ish week down, scales with degradation
            down[i] = max(down[i], rep); d[i] = max(0.0, d[i] - eff * d[i]); repaired = True; crew += 24.0
    # 2) preventive maintenance only if the crew was not preempted by a corrective repair
    if not repaired:
        down[a] = max(down[a], PM_DOWN); d[a] = max(0.0, d[a] - eff * d[a]); crew += 24.0
    # 3) wear accrues (exogenous + endogenous utilization)
    if degrade:
        d = np.clip(d + tape.wear[w] + 0.02 * (1 - down), 0.0, 1.0)
    # 4) production through serial WIP with block/starvation
    prod0 = CAP * (1 - down[0])
    take0 = min(prod0, Wcap - wip[0]); wip[0] += take0
    prod1 = min(CAP * (1 - down[1]), wip[0]); wip[0] -= prod1
    take1 = min(prod1, Wcap - wip[1]); wip[1] += take1
    prod2 = min(CAP * (1 - down[2]), wip[1]); wip[1] -= prod2
    throughput = prod2
    # 5) demand served
    dem = tape.demand[w]; s = min(throughput, dem)
    weekly_sl = float(max(0.0, dem - s))
    return d, wip, weekly_sl, down, crew


def simulate(tape: MTape, actions, *, degrade=True):
    d = np.zeros(3); wip = np.zeros(2)
    served_loss = 0.0; downtime = np.zeros(3); crew = 0.0; demand_tot = 0.0
    for w in range(tape.weeks):
        a = ACTIONS.index(actions[w]) if w < len(actions) else 0
        d, wip, wsl, down, cw = week_step(tape, w, a, d, wip, degrade=degrade)
        served_loss += wsl; downtime += down; crew += cw; demand_tot += tape.demand[w]
    return MResult(service_loss=float(served_loss),
                   fill_rate=float(1 - served_loss / max(demand_tot, 1)),
                   worst_station_downtime=float(downtime.max()), crew_hours=crew,
                   per_week_actions=list(actions))


def enumerate_oracle(tape: MTape):
    """Exact clairvoyant open-loop oracle over 3^weeks PM sequences (min service-loss)."""
    best = np.inf; bseq = None
    for seq in itertools.product(ACTIONS, repeat=tape.weeks):
        r = simulate(tape, seq)
        if r.service_loss < best:
            best, bseq = r.service_loss, seq
    return best, bseq


def periodic_calendars(weeks, max_period=6):
    seen = set(); cals = []
    for p in range(1, max_period + 1):
        for base in itertools.product(ACTIONS, repeat=p):
            cal = tuple((base * (weeks // p + 1))[:weeks])
            if cal not in seen:
                seen.add(cal); cals.append(cal)
    return cals


def condition_index(tape, w, d_true):
    """Noisy observable condition sensor: observes TRUE degradation d_true with sensor noise
    (balanced accuracy sensor_q). This is a legitimate CBM sensor -- real state, noisily seen."""
    q = tape.cell["sensor_q"]
    rng = np.random.default_rng(np.random.SeedSequence([tape.seed, w, 0xC04D]))
    noise = rng.normal(0, (1 - q) * 0.3, size=3)
    return np.clip(np.asarray(d_true, float) + noise, 0.0, 1.0)


def worst_condition_policy(tape: MTape):
    """Observable adaptive, closed-loop over TRUE physics: each week observe the noisy true
    condition and maintain the worst station."""
    d = np.zeros(3); wip = np.zeros(2); acts = []
    for w in range(tape.weeks):
        ci = condition_index(tape, w, d)               # noisy TRUE degradation
        a = int(ci.argmax()); acts.append(ACTIONS[a])
        d, wip, _, _, _ = week_step(tape, w, a, d, wip)
    return tuple(acts)


def _threat_forecast(tape, w):
    """Noisy 1-week-ahead threat forecast (balanced accuracy sensor_q, fail-closed).

    A deployed maintenance planner legitimately has threat/weather forecasts; this is a
    DISCLOSED observable, not privileged state. Fail-closed: on a miss it predicts no threat.
    """
    q = tape.cell["sensor_q"]
    if w + 1 >= tape.weeks:
        return np.zeros(3, dtype=int)
    truth = tape.threat[w + 1]
    rng = np.random.default_rng(np.random.SeedSequence([tape.seed, w, 0xF0EC]))
    flip = rng.random(3) > q                       # with prob (1-q) the forecast is wrong
    fc = np.where(flip, 1 - truth, truth)
    return fc.astype(int)


TAU = 0.4   # degradation above which an exogenous threat converts to a failure (matches simulate)


def forecast_policy(tape: MTape):
    """Observable adaptive WITH predictive signal, closed-loop over TRUE physics: protect the
    station whose *imminent* (forecast next-week) threat, given its observed true condition near
    the failure threshold, poses the largest expected loss. This is the project's identified
    missing ingredient: a signal that predicts the future BEFORE acting.
    """
    d = np.zeros(3); wip = np.zeros(2); acts = []
    for w in range(tape.weeks):
        ci = condition_index(tape, w, d)               # noisy TRUE degradation obs
        fc = _threat_forecast(tape, w)                 # noisy next-week threat obs
        # a threat only converts to failure if d>TAU; protect the station with imminent threat
        # AND condition at/over the threshold (else the crew is wasted).
        risk = fc * np.clip((ci - (TAU - 0.15)) / 0.3, 0.0, 1.0)
        a = int(ci.argmax()) if risk.max() <= 0 else int(risk.argmax())
        acts.append(ACTIONS[a])
        d, wip, _, _, _ = week_step(tape, w, a, d, wip)
    return tuple(acts)
