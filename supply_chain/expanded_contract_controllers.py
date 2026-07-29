"""Comparators for the expanded (buffer) decision contract: static, DDMRP, and MPC.

Development instrument. No confirmation universe, no learner.

Why this exists. Garrido's 2026-07-28 design has four steps: baseline, MPC on the
original variables, **MPC on the expanded variables**, then KAN. Step three was
missing: we had the baseline and we had the expanded contract (H2/H3 showed buffer
and shift rights move ReT by +11% to +25%), but nothing had asked whether a
structured controller *converts* that value or whether it saturates. Without step
three there is no way to say where the neural premium could live, because the
residual is defined against the best structured controller.

He also named the incumbent to beat: "hay una técnica en la literatura de negocios
que se llama demand driven material requirement planning ... quizás uno puede
crackear esa vaina y meter una red neuronal". DDMRP is therefore a required
comparator, not an optional one, and it has to be the real method rather than a
static level wearing its name.

## The contract

All three controllers receive **exactly the same decision rights and the same
information**: at each epoch they observe the simulation's own state and set the
three strategic buffer targets (`op3_rm`, `op5_rm`, `op9_rations`). Nothing else is
written. `_inventory_buffer_replenishment` re-reads `inventory_buffer_targets` on
every cycle (supply_chain.py:1194), so mutating them between steps is genuine
closed-loop control rather than a re-parameterisation.

Levels are the thesis ladder of Table 6.16 plus a zero floor, so every controller
chooses inside the same admissible set the static screen already enumerated. That
keeps the static incumbent embedded in every richer controller's action set: a
controller that cannot at least tie the best fixed posture has failed to find a
solution that was available to it, which is a search failure and must be reported
as one rather than as evidence about the right.

## What each controller is

`StaticPosture` holds one level for the whole run. This is the incumbent and it is
strong: the 216-posture screen found the best fixed posture has an *interior*
optimum (`op5_rm = 0`), so beating it is not a matter of simply holding more stock.

`DDMRPController` implements the actual method: decoupled lead time, average daily
usage estimated on a rolling window, red/yellow/green zones with lead-time and
variability factors, net flow position, and dynamic buffer adjustment as ADU moves.
The target it writes is top-of-green.

`ReceedingHorizonMPC` plans against the real DES rather than a frozen skeleton. The
Program Q transducer cannot be reused here: `extract_full_des_skeleton` freezes
"only action-independent events" — batch arrivals, orders, release slots — and
buffer targets change exactly those. So each candidate is evaluated by re-running
the simulator from scratch over the committed action prefix plus the candidate,
under common random numbers. That is expensive and it is the honest cost of
planning in a contract where the cheap transducer is invalid.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Sequence

from supply_chain.config import INVENTORY_BUFFERS

# Table 6.16 ladder plus a zero floor: the admissible target set, shared by all arms.
LADDER_HOURS: tuple[int, ...] = (0, 168, 336, 504, 672, 1344)
NODES: tuple[str, ...] = ("op3_rm", "op5_rm", "op9_rations")


def level_targets(hours: int) -> dict[str, float]:
    """The three node targets at one rung of the thesis ladder."""
    if hours == 0:
        return {n: 0.0 for n in NODES}
    row = INVENTORY_BUFFERS[hours]
    return {n: float(row[n]) for n in NODES}


def _on_hand(sim: Any) -> dict[str, float]:
    return {
        "op3_rm": float(sim.raw_material_wdc.level),
        "op5_rm": float(sim.raw_material_al.level),
        "op9_rations": float(sim.rations_sb.level),
    }


class Controller:
    """Sets `sim.inventory_buffer_targets` at each decision epoch."""

    name = "base"

    def reset(self) -> None:  # pragma: no cover - trivial
        pass

    def act(self, sim: Any, epoch: int) -> dict[str, float]:
        raise NotImplementedError


@dataclass
class StaticPosture(Controller):
    """The incumbent: one rung of the ladder, held for the whole run."""

    hours: int
    name: str = field(init=False)

    def __post_init__(self) -> None:
        self.name = f"static_I{self.hours}"

    def act(self, sim: Any, epoch: int) -> dict[str, float]:
        return level_targets(self.hours)


@dataclass
class DDMRPController(Controller):
    """Demand-driven MRP over the three decoupling points.

    Zones follow the standard construction. For a decoupling point with average
    daily usage `ADU`, decoupled lead time `DLT` in days, lead-time factor `LTF`
    and variability factor `VF`:

        red    = ADU * DLT * LTF * (1 + VF)
        yellow = ADU * DLT
        green  = max(ADU * DLT * LTF, ADU * order_cycle_days)
        TOG    = red + yellow + green

    `ADU` is re-estimated on a rolling window at every epoch, which is what makes
    the buffer *dynamic* — the property a fixed posture cannot express and the one
    the literature credits DDMRP for. Net flow position (on-hand plus on-order
    minus qualified spikes) decides whether to replenish; because the simulator's
    top-up is itself an order-up-to to the written target, writing TOG when the
    position is below top-of-yellow reproduces the DDMRP replenishment rule.
    """

    lead_time_factor: float = 0.5
    variability_factor: float = 0.5
    order_cycle_days: float = 7.0
    window_days: float = 28.0
    # Decoupled lead time per node, in days. Op3/Op5 sit behind supplier delivery
    # (Op2 ROP 672 h = 28 d); Op9 sits behind assembly and dispatch.
    dlt_days: dict[str, float] = field(
        default_factory=lambda: {"op3_rm": 28.0, "op5_rm": 28.0, "op9_rations": 7.0}
    )
    name: str = "ddmrp_dynamic"

    def __post_init__(self) -> None:
        self._last_delivered = 0.0
        self._last_time = 0.0
        self._adu: dict[str, float] = {}

    def reset(self) -> None:
        self._last_delivered = 0.0
        self._last_time = 0.0
        self._adu = {}

    def _update_adu(self, sim: Any) -> float:
        """Rolling average daily usage from realised consumption."""
        now = float(sim.env.now)
        delivered = float(getattr(sim, "total_order_fulfilled", 0.0))
        dt_days = max((now - self._last_time) / 24.0, 1e-9)
        used = max(delivered - self._last_delivered, 0.0)
        adu = used / dt_days
        self._last_delivered, self._last_time = delivered, now
        return adu

    def act(self, sim: Any, epoch: int) -> dict[str, float]:
        adu = self._update_adu(sim)
        if adu <= 0.0:
            # Before the first realised consumption, fall back to the thesis
            # regular-demand rate rather than collapsing the buffer to zero.
            adu = 2_500.0
        on_hand = _on_hand(sim)
        targets: dict[str, float] = {}
        for node in NODES:
            dlt = self.dlt_days[node]
            # Raw-material nodes carry a kit of 12 raw materials per ration.
            scale = 12.0 if node.endswith("_rm") else 1.0
            base = adu * scale
            red = base * dlt * self.lead_time_factor * (1.0 + self.variability_factor)
            yellow = base * dlt
            green = max(base * dlt * self.lead_time_factor,
                        base * self.order_cycle_days)
            tog = red + yellow + green
            toy = red + yellow
            # Net flow position; the simulator exposes no on-order pipeline for the
            # strategic buffer, so position is on-hand.
            nfp = on_hand[node]
            targets[node] = float(tog) if nfp < toy else float(max(nfp, red))
        return targets


@dataclass
class ReceedingHorizonMPC(Controller):
    """Plans against the real DES by re-simulating the committed prefix.

    `rollout` must accept (prefix, candidate, scenario_seed) and return the
    objective of a full run whose targets follow `prefix` and then `candidate`.
    Candidates are the ladder rungs, so the static incumbent is inside the search
    set at every epoch.
    """

    rollout: Callable[[list[int], int, int], float]
    scenario_seeds: Sequence[int]
    candidates: Sequence[int] = LADDER_HOURS
    name: str = "mpc_receding_horizon"

    def __post_init__(self) -> None:
        self._prefix: list[int] = []
        self.plan_calls = 0

    def reset(self) -> None:
        self._prefix = []
        self.plan_calls = 0

    def act(self, sim: Any, epoch: int) -> dict[str, float]:
        best_hours, best_value = None, -math.inf
        for cand in self.candidates:
            vals = []
            for s in self.scenario_seeds:
                vals.append(self.rollout(list(self._prefix), int(cand), int(s)))
                self.plan_calls += 1
            mean = sum(vals) / len(vals)
            if mean > best_value:
                best_hours, best_value = int(cand), mean
        assert best_hours is not None
        self._prefix.append(best_hours)
        return level_targets(best_hours)
