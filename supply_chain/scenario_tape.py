"""Exogenous scenario tapes for the frozen learning regime (`learning_extension_v1`).

See `docs/PAPER_CONTRACT_2026-06-24.md`. The Track A learning experiment requires
recurring, *persistent* disruption exposure so that retained policy state ``L_{k-1}``
can be valuable (H1) and so that the retained-minus-reset advantage can grow with
persistence ``rho`` (H2). Per the user decision (2026-06-24), persistence is modelled
as **two independent processes**:

1. a campaign-phase **disruption-intensity** chain with persistence ``rho_disruption``
   that selects a per-block thesis risk-severity level;
2. an independent **operational-tempo demand** chain with persistence ``rho_demand``
   that selects a per-block demand-mean multiplier.

Keeping them independent lets the analysis ablate *which* source of persistence the
retained learner exploits.

These tapes are generated once and replayed identically across every policy condition
(common random numbers), and the tape RNG is kept separate from any endogenous decision
RNG. A tape is pure data: deterministic given its seed, never tuned against held-out
results.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Default phase grids. These are operational-realism choices frozen before the
# held-out contrast, NOT thesis values; they are recorded in the experiment manifest.
DEFAULT_DISRUPTION_LEVELS: tuple[str, ...] = ("current", "increased", "severe")
DEFAULT_DEMAND_MULTIPLIERS: tuple[float, ...] = (0.85, 1.0, 1.20)


@dataclass(frozen=True)
class RegimePhase:
    """One disruption block's exogenous regime, drawn from the two phase chains."""

    disruption_phase: int
    demand_phase: int
    disruption_level: str
    demand_multiplier: float


@dataclass(frozen=True)
class ScenarioTape:
    """A replayable sequence of per-block regimes plus the parameters that made it."""

    blocks: tuple[RegimePhase, ...]
    rho_disruption: float
    rho_demand: float
    seed: int
    disruption_levels: tuple[str, ...]
    demand_multipliers: tuple[float, ...]

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, index: int) -> RegimePhase:
        return self.blocks[index]


def _validate_rho(rho: float, n_states: int, name: str) -> float:
    rho = float(rho)
    lo = 1.0 / n_states
    if not (lo - 1e-9 <= rho <= 1.0 + 1e-9):
        raise ValueError(
            f"{name}={rho} out of range; symmetric persistence must lie in "
            f"[1/n_states, 1] = [{lo:.4f}, 1.0] for {n_states} states. "
            f"{lo:.4f} is memoryless (uniform); 1.0 is absorbing."
        )
    return min(max(rho, lo), 1.0)


def _markov_phase_sequence(
    rng: np.random.Generator, n_blocks: int, n_states: int, rho: float
) -> np.ndarray:
    """Symmetric-persistence Markov chain over ``n_states``.

    Stay-probability is ``rho``; the remaining mass is split evenly across the other
    states, so the stationary distribution is uniform and ``rho`` is the single
    persistence knob. ``rho = 1/n_states`` is memoryless; ``rho -> 1`` is sticky.
    """
    if n_blocks <= 0:
        return np.empty(0, dtype=int)
    off = (1.0 - rho) / (n_states - 1) if n_states > 1 else 0.0
    seq = np.empty(n_blocks, dtype=int)
    # Start from the (uniform) stationary distribution.
    seq[0] = int(rng.integers(n_states))
    for k in range(1, n_blocks):
        prev = seq[k - 1]
        probs = np.full(n_states, off)
        probs[prev] = rho
        seq[k] = int(rng.choice(n_states, p=probs))
    return seq


def generate_scenario_tape(
    n_blocks: int,
    *,
    rho_disruption: float,
    rho_demand: float,
    seed: int,
    disruption_levels: tuple[str, ...] = DEFAULT_DISRUPTION_LEVELS,
    demand_multipliers: tuple[float, ...] = DEFAULT_DEMAND_MULTIPLIERS,
) -> ScenarioTape:
    """Generate a deterministic, replayable scenario tape of ``n_blocks`` regimes.

    The disruption and demand phase chains are driven by independent RNG streams spawned
    from ``seed``, so their persistence parameters move independently and the two
    processes are statistically uncorrelated.
    """
    if n_blocks < 0:
        raise ValueError(f"n_blocks must be >= 0, got {n_blocks}")
    n_disr = len(disruption_levels)
    n_dem = len(demand_multipliers)
    if n_disr == 0 or n_dem == 0:
        raise ValueError("disruption_levels and demand_multipliers must be non-empty")
    rho_disruption = _validate_rho(rho_disruption, n_disr, "rho_disruption")
    rho_demand = _validate_rho(rho_demand, n_dem, "rho_demand")

    # Independent streams: spawning from one SeedSequence guarantees the two chains are
    # uncorrelated while the whole tape stays reproducible from a single integer seed.
    disr_ss, dem_ss = np.random.SeedSequence(seed).spawn(2)
    disr_seq = _markov_phase_sequence(
        np.random.default_rng(disr_ss), n_blocks, n_disr, rho_disruption
    )
    dem_seq = _markov_phase_sequence(
        np.random.default_rng(dem_ss), n_blocks, n_dem, rho_demand
    )

    blocks = tuple(
        RegimePhase(
            disruption_phase=int(d),
            demand_phase=int(m),
            disruption_level=disruption_levels[int(d)],
            demand_multiplier=float(demand_multipliers[int(m)]),
        )
        for d, m in zip(disr_seq, dem_seq)
    )
    return ScenarioTape(
        blocks=blocks,
        rho_disruption=rho_disruption,
        rho_demand=rho_demand,
        seed=int(seed),
        disruption_levels=tuple(disruption_levels),
        demand_multipliers=tuple(demand_multipliers),
    )
