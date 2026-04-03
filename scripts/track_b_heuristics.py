"""Track B heuristic baselines (7D action space).

Each heuristic outputs a 7-dim action array in [-1, 1]:
  [0-3]: inventory multiplier signals (op3_q, op9_q, op3_rop, op9_rop)
  [4]:   shift selector signal (-1→S1, 0→S2, +1→S3)
  [5]:   op10 downstream multiplier signal
  [6]:   op12 downstream multiplier signal

Multiplier mapping: signal ∈ [-1, 1] → multiplier = 1.25 + 0.75 * signal ∈ [0.5, 2.0]

Observation v7 indices used:
  6: fill_rate, 7: backorder_rate, 8: assembly_line_down, 9: any_location_down,
  40: op10_down, 41: op12_down, 42: op10_queue_pressure_norm,
  43: op12_queue_pressure_norm, 44: rolling_fill_rate_4w, 45: rolling_backorder_rate_4w
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

SHIFT_SIGNAL = {1: -1.0, 2: 0.0, 3: 1.0}


class TrackBHeuristicPolicy(Protocol):
    def reset(self) -> None: ...
    def __call__(self, obs: np.ndarray, info: dict[str, Any]) -> np.ndarray: ...


def _latest_frame(obs: np.ndarray) -> np.ndarray:
    return obs[-1] if obs.ndim == 2 else obs


class TrackBHysteresis:
    """Shift hysteresis on backorder_rate. Downstream stays neutral."""

    def __init__(self, tau_high: float = 0.15, tau_low: float = 0.05) -> None:
        self.tau_high = tau_high
        self.tau_low = tau_low
        self._current_shift = 2

    def reset(self) -> None:
        self._current_shift = 2

    def __call__(self, obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        frame = _latest_frame(obs)
        backorder_rate = float(frame[7])
        if backorder_rate > self.tau_high:
            self._current_shift = 3
        elif backorder_rate < self.tau_low:
            self._current_shift = 1
        return np.array(
            [0.0, 0.0, 0.0, 0.0, SHIFT_SIGNAL[self._current_shift], 0.0, 0.0],
            dtype=np.float32,
        )


class TrackBDisruptionAware:
    """Disruption-reactive shift + inventory + downstream control.

    When disruption active → S3 + max inventory + max downstream.
    When fill_rate low → S2 + moderate boost + moderate downstream.
    Otherwise → S1 + neutral.
    """

    def __init__(
        self,
        fill_rate_caution: float = 0.90,
        inventory_boost: float = 0.5,
        inventory_large_boost: float = 1.0,
    ) -> None:
        self.fill_rate_caution = fill_rate_caution
        self.inventory_boost = inventory_boost
        self.inventory_large_boost = inventory_large_boost

    def reset(self) -> None:
        pass

    def __call__(self, obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        frame = _latest_frame(obs)
        assembly_down = float(frame[8]) > 0.5
        any_loc_down = float(frame[9]) > 0.5
        fill_rate = float(frame[6])
        if assembly_down or any_loc_down:
            shift_signal = 1.0
            inv_signal = self.inventory_large_boost
            ds_signal = 1.0
        elif fill_rate < self.fill_rate_caution:
            shift_signal = 0.0
            inv_signal = self.inventory_boost
            ds_signal = 0.5
        else:
            shift_signal = -1.0
            inv_signal = 0.0
            ds_signal = 0.0
        return np.array(
            [inv_signal, inv_signal, inv_signal, inv_signal,
             shift_signal, ds_signal, ds_signal],
            dtype=np.float32,
        )


class TrackBTuned:
    """Dual-criteria hysteresis with downstream following upstream boost."""

    def __init__(
        self,
        tau_up: float = 0.18,
        tau_down: float = 0.08,
        fr_low: float = 0.80,
        fr_high: float = 0.90,
    ) -> None:
        self.tau_up = tau_up
        self.tau_down = tau_down
        self.fr_low = fr_low
        self.fr_high = fr_high
        self._current_shift = 2

    def reset(self) -> None:
        self._current_shift = 2

    def __call__(self, obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        frame = _latest_frame(obs)
        fill_rate = float(frame[6])
        backorder_rate = float(frame[7])
        assembly_down = float(frame[8]) > 0.5
        any_loc_down = float(frame[9]) > 0.5

        if assembly_down or any_loc_down:
            self._current_shift = min(3, self._current_shift + 1)
        elif backorder_rate >= self.tau_up or fill_rate <= self.fr_low:
            self._current_shift = min(3, self._current_shift + 1)
        elif backorder_rate <= self.tau_down and fill_rate >= self.fr_high:
            self._current_shift = max(1, self._current_shift - 1)

        ds_signal = 0.5 if self._current_shift >= 2 else 0.0
        return np.array(
            [0.0, 0.0, 0.0, 0.0, SHIFT_SIGNAL[self._current_shift],
             ds_signal, ds_signal],
            dtype=np.float32,
        )


class TrackBDownstreamReactive:
    """S1 fixed + downstream reactive to queue pressure and disruption flags.

    Uses v7 obs: op10_queue_pressure_norm (42), op12_queue_pressure_norm (43),
    op10_down (40), op12_down (41).
    """

    def __init__(
        self,
        pressure_threshold: float = 0.3,
        boost_signal: float = 0.8,
    ) -> None:
        self.pressure_threshold = pressure_threshold
        self.boost_signal = boost_signal

    def reset(self) -> None:
        pass

    def __call__(self, obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        frame = _latest_frame(obs)
        op10_down = float(frame[40]) > 0.5 if len(frame) > 40 else False
        op12_down = float(frame[41]) > 0.5 if len(frame) > 41 else False
        op10_pressure = float(frame[42]) if len(frame) > 42 else 0.0
        op12_pressure = float(frame[43]) if len(frame) > 43 else 0.0

        op10_signal = 0.0
        if op10_down or op10_pressure > self.pressure_threshold:
            op10_signal = self.boost_signal

        op12_signal = 0.0
        if op12_down or op12_pressure > self.pressure_threshold:
            op12_signal = self.boost_signal

        return np.array(
            [0.0, 0.0, 0.0, 0.0, -1.0, op10_signal, op12_signal],
            dtype=np.float32,
        )


class TrackBMaxDownstream:
    """S1 fixed + downstream always at max (signal=+1.0 → multiplier=2.0).

    This is the critical strawman baseline: it mimics PPO's discovered strategy
    (mostly S1 + aggressive downstream) as a fixed heuristic. If PPO cannot beat
    this, the reactive component adds no value.
    """

    def reset(self) -> None:
        pass

    def __call__(self, obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        return np.array(
            [0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 1.0],
            dtype=np.float32,
        )


HEURISTIC_SPECS: dict[str, TrackBHeuristicPolicy] = {}


def make_heuristic_defaults() -> dict[str, TrackBHeuristicPolicy]:
    return {
        "heur_hysteresis": TrackBHysteresis(),
        "heur_disruption_aware": TrackBDisruptionAware(),
        "heur_tuned": TrackBTuned(),
        "heur_downstream_reactive": TrackBDownstreamReactive(),
        "heur_s1_max_downstream": TrackBMaxDownstream(),
    }


HEURISTIC_POLICY_NAMES = tuple(make_heuristic_defaults().keys())
