"""Program L — FULL-DES event-driven alternate-route recourse over the canonical
finite-convoy MFSC.  This extends the frozen DRA-2/Program-E Op8 finite-convoy physics
(``ProgramEConvoyEnv``) with a single DISCLOSED, Garrido-pending alternate route:

  action 0 = HOLD
  action 1 = ROUTE_1  -> the thesis Op8 leg; real R22 stall (fast when up, stalls when down)
  action 2 = ROUTE_2  -> a disclosed alternate that BYPASSES Op8 R22 at a committed transit
                         (slower base; inflated when the alternate is itself degraded)

All order / mass / ReT / warm-up accounting is the canonical DES (``make_sim`` +
``compute_episode_metrics``).  The ONLY researcher-introduced physics is the alternate route
(thesis Section 6.5 takes route planning as given) and its pre-departure signal — both are
flagged and require Garrido face validation before any paper claim.

Faithfulness contract:
  * route mode OFF (never issuing action 2) => bit-identical to ProgramEConvoyEnv (tested).
  * R22 is CRN-fixed on the tape => same tape + different action => identical Op8 outages.
  * ROUTE_1 signal is a NOISY nowcast of the KNOWN Op8 R22 schedule (honest; no future leak
    beyond a declared lead, accuracy q); ROUTE_2 condition Z2 + signal are a per-tape CRN draw.

NO learner is trained here. This module exists to run the pre-learner headroom gate on the
full DES.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import gymnasium as gym
import numpy as np

from .dra2_experiment import advance_including, make_sim
from .dra2_policy_env import OBSERVATION_KEYS as BASE_OBS_KEYS
from .episode_metrics import compute_episode_metrics
from .config import HOURS_PER_WEEK

ROUTE_OBS_KEYS = ("sig_route1_down", "sig_route2_degraded", "route2_last_degraded")
OBSERVATION_KEYS = tuple(BASE_OBS_KEYS) + ROUTE_OBS_KEYS
ACTIONS = ("HOLD", "ROUTE_1", "ROUTE_2")


@dataclass(frozen=True)
class RouteContract:
    route2_base_outbound_h: float = 36.0    # alternate is slower than the 24h primary
    route2_base_return_h: float = 36.0
    route2_degraded_penalty_h: float = 24.0  # extra transit when the alternate is itself degraded
    route2_persistence: float = 0.85         # semi-Markov daily stay prob for Z2
    route2_p_degraded: float = 0.25
    signal_accuracy: float = 0.85            # balanced accuracy for BOTH route signals
    signal_lead_hours: float = 24.0          # nowcast window for the Op8-down signal


def _op8_down_stepfun(tape: dict[str, Any], start: float, horizon_h: float) -> np.ndarray:
    """Hourly 0/1 Op8-down indicator from the CRN R22 tape (absolute time, warmup-shifted)."""
    n = int(horizon_h) + 2
    down = np.zeros(n, dtype=np.int8)
    for ev in tape.get("risk_events", []):
        if str(ev.get("risk_id")) != "R22":
            continue
        a = int(max(0.0, float(ev["start_time"])))
        b = int(min(float(horizon_h) + 1.0, float(ev["end_time"])))
        if b > a:
            down[a:b] = 1
    return down


class ProgramLRouteRecourseEnv(gym.Env):
    """Discrete(3) HOLD/ROUTE_1/ROUTE_2 over the canonical finite-convoy DES."""
    metadata = {"render_modes": []}

    def __init__(self, tapes: Sequence[dict[str, Any]], normalizers: Mapping[str, Any], *,
                 contract: RouteContract | None = None, episode_days: int = 56,
                 random_tapes: bool = True, route_recourse_enabled: bool = True) -> None:
        super().__init__()
        if not tapes:
            raise ValueError("Program L requires at least one tape")
        self.tapes = list(tapes)
        self.normalizers = dict(normalizers)
        self.contract = contract or RouteContract()
        self.episode_days = int(episode_days)
        self.random_tapes = bool(random_tapes)
        self.route_recourse_enabled = bool(route_recourse_enabled)
        self.action_space = gym.spaces.Discrete(3 if route_recourse_enabled else 2)
        self.observation_space = gym.spaces.Box(
            low=-10.0, high=10.0, shape=(len(OBSERVATION_KEYS),), dtype=np.float32)
        self._tape_cursor = 0
        self.sim = None
        self.start = 0.0
        self.end = 0.0
        self.steps = 0
        self._resource_start: dict[str, float] = {}

    # --- tape / route state ---
    def _select_tape(self) -> dict[str, Any]:
        if self.random_tapes:
            idx = int(self.np_random.integers(0, len(self.tapes)))
        else:
            idx = self._tape_cursor % len(self.tapes)
            self._tape_cursor += 1
        return self.tapes[idx]

    def _build_route2_tape(self, seed: int):
        c = self.contract
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x1052]))
        days = self.episode_days + 2
        Z2 = np.zeros(days, dtype=np.int8)
        z = 1 if rng.random() < c.route2_p_degraded else 0
        for d in range(days):
            if rng.random() >= c.route2_persistence:
                z = 1 if rng.random() < c.route2_p_degraded else 0
            Z2[d] = z
        # per-day balanced-accuracy signals (drawn once, CRN; action-independent)
        sig2 = np.array([Z2[d] if rng.random() < c.signal_accuracy else 1 - Z2[d] for d in range(days)], dtype=np.int8)
        sig1_noise = rng.random(days)   # applied to the true Op8-down nowcast at obs time
        return Z2, sig2, sig1_noise

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        tape = (options or {}).get("tape") or self._select_tape()
        self.sim, self.start = make_sim(tape)
        self.end = self.start + self.episode_days * 24.0
        self.steps = 0
        self._resource_start = dict(self.sim.op8_convoy_metrics())
        self._op8_down = _op8_down_stepfun(tape, self.start, self.end - self.start + self.contract.signal_lead_hours + 2)
        self._Z2, self._sig2, self._sig1_noise = self._build_route2_tape(int(tape["seed"]))
        self._route2_last_degraded = 0.0
        self._tape_id = tape.get("tape_id", "?")
        return self._normalized_observation(), {"tape_id": self._tape_id, "family": tape.get("family")}

    # --- observation ---
    def _cur_day(self) -> int:
        return int(max(0, (float(self.sim.env.now) - self.start) // 24.0))

    def _route_signals(self) -> dict[str, float]:
        d = min(self._cur_day(), len(self._Z2) - 1)
        c = self.contract
        # ROUTE_1 signal: noisy nowcast of whether Op8 is down anywhere in [now, now+lead]
        now_rel = int(float(self.sim.env.now) - self.start)
        lead = int(c.signal_lead_hours)
        lo = max(0, now_rel)
        hi = min(len(self._op8_down), lo + max(1, lead))
        true_down = 1 if (hi > lo and self._op8_down[lo:hi].any()) else 0
        sig1 = true_down if self._sig1_noise[d] < c.signal_accuracy else 1 - true_down
        return {"sig_route1_down": float(sig1),
                "sig_route2_degraded": float(self._sig2[d]),
                "route2_last_degraded": float(self._route2_last_degraded)}

    def raw_observation(self) -> dict[str, float]:
        obs = dict(self.sim.get_op8_convoy_observation())
        obs["departures_to_date"] = float(self.sim.op8_convoy_departures
                                          - self._resource_start.get("op8_convoy_departures", 0.0))
        obs["unavailable_hours_to_date"] = float(self.sim.op8_convoy_vehicle_hours
                                          - self._resource_start.get("op8_convoy_unavailable_hours", 0.0))
        obs.update(self._route_signals())
        return {k: float(obs[k]) for k in OBSERVATION_KEYS}

    def _normalized_observation(self) -> np.ndarray:
        raw = self.raw_observation()
        scales = self.normalizers["observation_scales"]
        return np.asarray([np.clip(raw[k] / max(float(scales.get(k, 1.0)), 1e-9), -10.0, 10.0)
                           for k in OBSERVATION_KEYS], dtype=np.float32)

    def action_masks(self) -> np.ndarray:
        avail = bool(self.sim.op8_convoy_available) and float(self.sim.rations_al.level) > 1e-9
        route1_ok = avail and not self.sim._is_down(8)
        route2_ok = avail  # alternate bypasses Op8 R22
        if self.route_recourse_enabled:
            return np.asarray([True, bool(route1_ok), bool(route2_ok)])
        return np.asarray([True, bool(route1_ok)])

    def _daily_loss_terms(self) -> tuple[float, float]:
        now = float(self.sim.env.now)
        service = float(self.sim.pending_backorder_qty) * 24.0
        backlog_age = sum(float(o.remaining_qty) * max(0.0, now - float(o.OPTj)) * 24.0
                          for o in self.sim.pending_backorders)
        return service, backlog_age

    def _dispatch(self, action: int) -> dict[str, Any]:
        c = self.contract
        if action == 1:   # ROUTE_1 = thesis Op8 leg (defaults => real R22)
            return self.sim.apply_op8_convoy_action("DISPATCH_NOW", source="program_l", route_id="R1")
        if action == 2:   # ROUTE_2 = disclosed alternate, committed transit, bypasses Op8
            d = min(self._cur_day(), len(self._Z2) - 1)
            deg = int(self._Z2[d])
            self._route2_last_degraded = float(deg)
            out = c.route2_base_outbound_h + (c.route2_degraded_penalty_h if deg else 0.0)
            ret = c.route2_base_return_h + (c.route2_degraded_penalty_h if deg else 0.0)
            return self.sim.apply_op8_convoy_action(
                "DISPATCH_NOW", source="program_l", route_id="R2",
                route_outbound_hours=out, route_return_hours=ret,
                route_exposed_op=None, route_bypasses_op8=True)
        return self.sim.apply_op8_convoy_action("HOLD", source="program_l")

    def step(self, action: int):
        requested = int(action)
        mask = self.action_masks()
        effective = requested if (requested < len(mask) and mask[requested]) else 0
        event = self._dispatch(effective)
        advance_including(self.sim, min(self.end, float(self.sim.env.now) + 24.0))
        self.steps += 1
        service, backlog_age = self._daily_loss_terms()
        rs = self.normalizers["reward_scales"]
        reward = -(service / max(float(rs["daily_service_loss_p95"]), 1.0)
                   + 0.1 * backlog_age / max(float(rs["daily_backlog_age_p95"]), 1.0))
        terminated = float(self.sim.env.now) >= self.end - 1e-9
        info = {"requested_action": requested, "effective_action": effective,
                "masked": requested != effective, "departed": bool(event["departed"]),
                "route_id": event.get("route_id")}
        if terminated:
            info.update(compute_episode_metrics(self.sim, treatment_start=self.start))
            res = self.sim.op8_convoy_metrics()
            info["episode_departures"] = float(res["op8_convoy_departures"]
                                        - self._resource_start.get("op8_convoy_departures", 0.0))
            info["episode_unavailable_hours"] = float(res["op8_convoy_unavailable_hours"]
                                        - self._resource_start.get("op8_convoy_unavailable_hours", 0.0))
        return self._normalized_observation(), float(reward), terminated, False, info


def make_identity_normalizers() -> dict[str, Any]:
    return {"observation_scales": {k: 1.0 for k in OBSERVATION_KEYS},
            "reward_scales": {"daily_service_loss_p95": 1.0, "daily_backlog_age_p95": 1.0}}
