"""Exact reduced POMDP benchmark for prospective Program T development.

This module is deliberately independent of the full DES and opens no scientific
seed.  It provides a finite, exactly solvable product-mix problem used to audit
claims about truncated MPC, open-loop schedules, and learned terminal values.

The hidden weekly regime is either C-dominant or H-dominant.  The controller
observes only the realised product count after committing three equal-capacity
slots.  The belief is therefore a sufficient statistic.  Total supply and
demand are both three units per week, so the scalar net C position also fixes
the H position and no clipping or mass creation is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
import math
from typing import Callable, Sequence


ActionRule = Callable[[int, float, int], int]


@dataclass(frozen=True)
class ExactProductMixPOMDP:
    """Small belief-state product-mix POMDP with an exact dynamic program."""

    horizon: int = 6
    regime_persistence: float = 0.85
    dominant_share: float = 0.85
    slots_per_week: int = 3
    backlog_weight: float = 1.0
    worst_product_weight: float = 0.25
    holding_weight: float = 0.05

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError("horizon must be positive")
        if self.slots_per_week < 1:
            raise ValueError("slots_per_week must be positive")
        if not 0.5 <= self.regime_persistence <= 1.0:
            raise ValueError("regime_persistence must be in [0.5, 1]")
        if not 0.5 <= self.dominant_share < 1.0:
            raise ValueError("dominant_share must be in [0.5, 1)")
        if min(
            self.backlog_weight,
            self.worst_product_weight,
            self.holding_weight,
        ) < 0.0:
            raise ValueError("cost weights must be non-negative")

    @property
    def actions(self) -> tuple[int, ...]:
        """Number of C slots; remaining slots are assigned to H."""
        return tuple(range(self.slots_per_week + 1))

    def count_probability(self, count_c: int, belief_c: float) -> float:
        """Predict the next observed C count under the current regime belief."""
        n = self.slots_per_week
        if count_c not in range(n + 1):
            return 0.0
        p = self.dominant_share
        likelihood_c = math.comb(n, count_c) * p**count_c * (1.0 - p) ** (n - count_c)
        likelihood_h = math.comb(n, count_c) * (1.0 - p) ** count_c * p ** (n - count_c)
        return float(belief_c) * likelihood_c + (1.0 - float(belief_c)) * likelihood_h

    def next_belief(self, count_c: int, belief_c: float) -> float:
        """Bayes-update the current regime and then apply its Markov transition."""
        p = self.dominant_share
        n = self.slots_per_week
        likelihood_c = math.comb(n, count_c) * p**count_c * (1.0 - p) ** (n - count_c)
        likelihood_h = math.comb(n, count_c) * (1.0 - p) ** count_c * p ** (n - count_c)
        denominator = float(belief_c) * likelihood_c + (1.0 - float(belief_c)) * likelihood_h
        posterior = 0.5 if denominator <= 0.0 else float(belief_c) * likelihood_c / denominator
        rho = self.regime_persistence
        transitioned = rho * posterior + (1.0 - rho) * (1.0 - posterior)
        return round(float(transitioned), 12)

    def step_cost(self, net_c: int) -> float:
        """Service-oriented cost after one week's supply and demand."""
        net_h = -int(net_c)
        backlog_c = max(0, -int(net_c))
        backlog_h = max(0, -net_h)
        holding = max(0, int(net_c)) + max(0, net_h)
        return (
            self.backlog_weight * (backlog_c + backlog_h)
            + self.worst_product_weight * max(backlog_c, backlog_h)
            + self.holding_weight * holding
        )

    def transition_net_c(self, net_c: int, action: int, count_c: int) -> int:
        if action not in self.actions:
            raise ValueError("invalid action")
        return int(net_c) + int(action) - int(count_c)

    def _action_cost(
        self,
        net_c: int,
        belief_c: float,
        remaining: int,
        action: int,
        continuation: Callable[[int, float, int], float],
    ) -> float:
        expected = 0.0
        for count_c in range(self.slots_per_week + 1):
            probability = self.count_probability(count_c, belief_c)
            next_net = self.transition_net_c(net_c, action, count_c)
            next_belief = self.next_belief(count_c, belief_c)
            expected += probability * (
                self.step_cost(next_net)
                + continuation(next_net, next_belief, remaining - 1)
            )
        return float(expected)

    @lru_cache(maxsize=None)
    def exact_cost(self, net_c: int, belief_c: float, remaining: int) -> float:
        """Globally optimal expected cost for the finite benchmark."""
        if remaining <= 0:
            return 0.0
        belief = round(float(belief_c), 12)
        return min(
            self._action_cost(
                int(net_c), belief, int(remaining), action, self.exact_cost
            )
            for action in self.actions
        )

    def exact_action(self, net_c: int, belief_c: float, remaining: int) -> int:
        if remaining <= 0:
            raise ValueError("remaining must be positive")
        rows = [
            (
                self._action_cost(
                    int(net_c),
                    round(float(belief_c), 12),
                    int(remaining),
                    action,
                    self.exact_cost,
                ),
                action,
            )
            for action in self.actions
        ]
        return min(rows)[1]

    def truncated_action(
        self, net_c: int, belief_c: float, remaining: int, lookahead: int
    ) -> int:
        """Receding-horizon belief MPC with zero terminal value."""
        if lookahead < 1 or remaining < 1:
            raise ValueError("lookahead and remaining must be positive")
        depth = min(int(lookahead), int(remaining))

        @lru_cache(maxsize=None)
        def truncated_cost(position: int, belief: float, steps: int) -> float:
            if steps <= 0:
                return 0.0
            return min(
                self._action_cost(position, belief, steps, action, truncated_cost)
                for action in self.actions
            )

        rows = [
            (
                self._action_cost(
                    int(net_c),
                    round(float(belief_c), 12),
                    depth,
                    action,
                    truncated_cost,
                ),
                action,
            )
            for action in self.actions
        ]
        return min(rows)[1]

    def policy_cost(
        self,
        rule: ActionRule,
        *,
        net_c: int = 0,
        belief_c: float = 0.5,
        remaining: int | None = None,
    ) -> float:
        """Evaluate a non-anticipative action rule exactly, without Monte Carlo."""
        steps = self.horizon if remaining is None else int(remaining)

        @lru_cache(maxsize=None)
        def value(position: int, belief: float, depth: int) -> float:
            if depth <= 0:
                return 0.0
            action = int(rule(position, belief, depth))
            return self._action_cost(position, belief, depth, action, value)

        return value(int(net_c), round(float(belief_c), 12), steps)

    def truncated_mpc_cost(self, lookahead: int) -> float:
        return self.policy_cost(
            lambda net, belief, remaining: self.truncated_action(
                net, belief, remaining, lookahead
            )
        )

    def open_loop_cost(self, sequence: Sequence[int]) -> float:
        actions = tuple(int(action) for action in sequence)
        if len(actions) != self.horizon or any(action not in self.actions for action in actions):
            raise ValueError("sequence must contain one valid action per horizon step")
        return self.policy_cost(
            lambda _net, _belief, remaining: actions[self.horizon - remaining]
        )

    def best_open_loop(self) -> tuple[tuple[int, ...], float]:
        rows = [
            (self.open_loop_cost(sequence), tuple(sequence))
            for sequence in product(self.actions, repeat=self.horizon)
        ]
        cost, sequence = min(rows)
        return sequence, float(cost)

    def diagnostic(self) -> dict[str, object]:
        """Return the exact ladder used by the Program T theory gate."""
        sequence, open_loop = self.best_open_loop()
        exact = self.exact_cost(0, 0.5, self.horizon)
        truncated = {
            str(depth): self.truncated_mpc_cost(depth)
            for depth in sorted({1, min(3, self.horizon), self.horizon})
        }
        return {
            "schema_version": "program_t_exact_pomdp_diagnostic_v1",
            "horizon": self.horizon,
            "best_open_loop_sequence": list(sequence),
            "best_open_loop_cost": open_loop,
            "exact_belief_dp_cost": exact,
            "truncated_mpc_cost": truncated,
            "exact_adaptive_gain_over_open_loop": open_loop - exact,
            "residual_cost_gap_by_lookahead": {
                key: value - exact for key, value in truncated.items()
            },
        }
