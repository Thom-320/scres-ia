"""Program K3: canonical ReT under equal-budget replenishment timing."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Sequence

import numpy as np

from .program_g import ret_order_metrics
from .replenish import D0, RTape
from .supply_chain import OrderRecord

LEVELS = tuple(float(x) for x in np.arange(0.0, 1.5001, 0.25))
WEEKS = 8
BUDGET_D0 = 10.0
WEEKLY_CAP_D0 = 1.5
LEAD_WEEKS = 1


@dataclass
class RetResult:
    ret_order: float
    ret_quantity: float
    attended: int
    lost: int
    remaining_qty: float
    ordered_D0: float
    holding_ration_weeks: float
    actions: tuple[float, ...]


def periodic_calendars(max_period: int = 4) -> tuple[tuple[float, ...], ...]:
    rows: dict[tuple[float, ...], None] = {}
    for period in range(1, max_period + 1):
        for base in product(LEVELS, repeat=period):
            sequence = tuple((base * (WEEKS // period + 1))[:WEEKS])
            if sum(sequence) <= BUDGET_D0 + 1e-9:
                rows[sequence] = None
    return tuple(rows)


def _complete_pending(pending: list[OrderRecord], inventory: float, hour: float):
    for order in pending:
        take = min(inventory, float(order.remaining_qty))
        inventory -= take
        order.remaining_qty -= take
        if order.remaining_qty <= 1e-9 and order.OATj is None:
            order.OATj = float(hour)
            order.CTj = float(hour - order.OPTj)
            order.backorder = bool(order.CTj > order.LTj)
    return [order for order in pending if order.remaining_qty > 1e-9], inventory


def rollout_actions(tape: RTape, actions: Sequence[float]) -> RetResult:
    inventory = D0
    pipeline = [0.0] * LEAD_WEEKS
    pending: list[OrderRecord] = []
    orders: list[OrderRecord] = []
    spent = 0.0
    holding = 0.0
    used: list[float] = []
    for week in range(WEEKS):
        hour = float(week * 168)
        inventory += pipeline.pop(0)
        requested = float(actions[week])
        quantity = min(WEEKLY_CAP_D0, max(0.0, requested), BUDGET_D0 - spent)
        quantity = round(quantity / 0.25) * 0.25
        spent += quantity
        used.append(quantity)
        pipeline.append(quantity * D0)
        demand = float(tape.demand[week])
        order = OrderRecord(
            j=week + 1, OPTj=hour, quantity=demand, remaining_qty=demand,
            LTj=168.0,
        )
        pending.append(order)
        orders.append(order)
        pending, inventory = _complete_pending(pending, inventory, hour)
        holding += inventory
    horizon = float(WEEKS * 168)
    for order in pending:
        order.lost = True
        order.lost_time = horizon
    metrics = ret_order_metrics(orders)
    return RetResult(
        ret_order=float(metrics["ret_order"]),
        ret_quantity=float(metrics["ret_quantity"]),
        attended=int(metrics["attended"]), lost=int(metrics["lost"]),
        remaining_qty=float(sum(order.remaining_qty for order in pending)),
        ordered_D0=float(spent), holding_ration_weeks=float(holding),
        actions=tuple(used),
    )


def rollout_policy(
    tape: RTape,
    policy: Callable[[dict[str, float]], float],
) -> RetResult:
    inventory = D0
    pipeline = [0.0] * LEAD_WEEKS
    pending: list[OrderRecord] = []
    orders: list[OrderRecord] = []
    spent = 0.0
    holding = 0.0
    actions: list[float] = []
    for week in range(WEEKS):
        hour = float(week * 168)
        inventory += pipeline.pop(0)
        backlog = float(sum(order.remaining_qty for order in pending))
        obs = {
            "on_hand_D0": inventory / D0,
            "pipeline_D0": sum(pipeline) / D0,
            "backlog_D0": backlog / D0,
            "remaining_budget_D0": BUDGET_D0 - spent,
            "weeks_remaining": float(WEEKS - week),
            "forecast_D0": float(tape.signal[week] / D0),
        }
        requested = float(policy(obs))
        quantity = min(WEEKLY_CAP_D0, max(0.0, requested), BUDGET_D0 - spent)
        quantity = round(quantity / 0.25) * 0.25
        spent += quantity
        actions.append(quantity)
        pipeline.append(quantity * D0)
        demand = float(tape.demand[week])
        order = OrderRecord(j=week + 1, OPTj=hour, quantity=demand, remaining_qty=demand, LTj=168.0)
        pending.append(order)
        orders.append(order)
        pending, inventory = _complete_pending(pending, inventory, hour)
        holding += inventory
    horizon = float(WEEKS * 168)
    for order in pending:
        order.lost = True
        order.lost_time = horizon
    metrics = ret_order_metrics(orders)
    return RetResult(
        ret_order=float(metrics["ret_order"]), ret_quantity=float(metrics["ret_quantity"]),
        attended=int(metrics["attended"]), lost=int(metrics["lost"]),
        remaining_qty=float(sum(order.remaining_qty for order in pending)),
        ordered_D0=float(spent), holding_ration_weeks=float(holding), actions=tuple(actions),
    )


def paced_policy(alpha: float, beta: float):
    """Budget pace plus forecast and inventory feedback; no latent state."""
    def policy(obs: dict[str, float]) -> float:
        pace = obs["remaining_budget_D0"] / obs["weeks_remaining"]
        return (
            pace
            + float(alpha) * (obs["forecast_D0"] - 1.0)
            + float(beta) * (1.0 - obs["on_hand_D0"])
        )
    return policy


def sS_policy(s: float, S: float):
    def policy(obs: dict[str, float]) -> float:
        position = obs["on_hand_D0"] + obs["pipeline_D0"]
        return max(0.0, S - position) if position <= s else 0.0
    return policy
