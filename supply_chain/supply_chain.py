"""
supply_chain.py — DES of the 13-operation Military Food Supply Chain.

Phase 1: Deterministic baseline (Cf0).
Phase 2: Stochastic risks R11-R14, R21-R24, R3.
Phase 3: RL-ready architecture (mutable params, hourly granularity, step API).

Key changes from v1:
  - Assembly line runs at HOURLY granularity (not daily) to correctly
    capture sub-day risk events like R11 (avg 2.2h repair).
  - All decision parameters stored in self.params (mutable at runtime).
  - step() method for Gymnasium integration.
  - get_observation() returns normalized state vector.
"""

import simpy
import numpy as np
from dataclasses import dataclass
from typing import Any, Optional
from collections import Counter

from supply_chain.config import (
    OPERATIONS,
    DEMAND,
    ASSEMBLY_RATE,
    BACKORDER_QUEUE_CAP,
    CAPACITY_BY_SHIFTS,
    HOURS_PER_SHIFT,
    HOURS_PER_DAY,
    SIMULATION_HORIZON,
    NUM_RAW_MATERIALS,
    RISKS_CURRENT,
    RISKS_INCREASED,
    RISKS_SEVERE,
    RISKS_SEVERE_EXTENDED,
    DEFAULT_YEAR_BASIS,
    HOURS_PER_YEAR_GREGORIAN,
    HOURS_PER_YEAR_THESIS,
    YEAR_BASIS_OPTIONS,
)

RATIONS_PER_HOUR = ASSEMBLY_RATE  # 320.5 rations/hr


def resolve_hours_per_year(year_basis: str) -> int:
    """Resolve annualization basis to hours/year."""
    if year_basis == "thesis":
        return HOURS_PER_YEAR_THESIS
    if year_basis == "gregorian":
        return HOURS_PER_YEAR_GREGORIAN
    valid = ", ".join(YEAR_BASIS_OPTIONS)
    raise ValueError(f"Invalid year_basis={year_basis!r}. Expected one of: {valid}.")


@dataclass
class OrderRecord:
    j: int
    OPTj: float
    quantity: float = 0.0
    OATj: Optional[float] = None
    CTj: Optional[float] = None
    backorder: bool = False
    remaining_qty: float = 0.0
    contingent: bool = False
    lost: bool = False


@dataclass
class RiskEvent:
    risk_id: str
    start_time: float
    end_time: float
    duration: float
    affected_ops: list
    description: str = ""


class MFSCSimulation:
    """
    SimPy DES of the 13-operation MFSC.
    RL-ready: mutable params, hourly assembly, step() API.
    """

    def __init__(
        self,
        shifts: int = 1,
        initial_buffers: Optional[dict[str, float]] = None,
        seed: Optional[int] = 42,
        horizon: float = SIMULATION_HORIZON,
        risks_enabled: bool = False,
        risk_level: str = "current",
        year_basis: str = DEFAULT_YEAR_BASIS,
        stochastic_pt: bool = False,
        deterministic_baseline: bool = False,
    ) -> None:
        self.env = simpy.Environment()
        self.shifts = shifts
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.horizon = horizon
        self.risks_enabled = risks_enabled
        self.risk_level = risk_level
        self.year_basis = year_basis
        self.stochastic_pt = stochastic_pt
        self.deterministic_baseline = deterministic_baseline
        self.hours_per_year = resolve_hours_per_year(year_basis)

        # =================================================================
        # MUTABLE PARAMETERS — RL agent can modify these at runtime
        # =================================================================
        self.params = {
            # Procurement
            "op1_rop": OPERATIONS[1]["rop"],  # 4,032 hrs (biannual)
            "op1_pt": OPERATIONS[1]["pt"],  # 672 hrs
            "op2_rop": OPERATIONS[2]["rop"],  # 672 hrs (monthly)
            "op2_pt": OPERATIONS[2]["pt"],  # 24 hrs
            "op2_q": OPERATIONS[2]["q"],  # 190,000 per RM
            "op3_rop": OPERATIONS[3]["rop"],  # 168 hrs (weekly)
            "op3_pt": OPERATIONS[3]["pt"],  # 24 hrs
            "op3_q": OPERATIONS[3]["q"],  # 15,500 per RM
            "op4_pt": OPERATIONS[4]["pt"],  # 24 hrs (transport WDC → AL)
            # Assembly
            "assembly_shifts": shifts,
            "batch_size": OPERATIONS[7]["q"],  # 5,000
            # Distribution
            "op8_pt": OPERATIONS[8]["pt"],  # 24 hrs
            "op9_rop": OPERATIONS[9]["rop"],  # 24 hrs
            "op9_pt": OPERATIONS[9]["pt"],  # 24 hrs
            "op9_q_min": OPERATIONS[9]["q"][0],  # 2,400
            "op9_q_max": OPERATIONS[9]["q"][1],  # 2,600
            "op10_rop": OPERATIONS[10]["rop"],  # 24 hrs
            "op10_pt": OPERATIONS[10]["pt"],  # 24 hrs
            "op12_rop": OPERATIONS[12]["rop"],  # 24 hrs
            "op12_pt": OPERATIONS[12]["pt"],  # 24 hrs
        }

        # =================================================================
        # MATERIAL BUFFERS
        # =================================================================
        INF = 10_000_000
        self.raw_material_wdc = simpy.Container(self.env, capacity=INF, init=0)
        self.raw_material_al = simpy.Container(self.env, capacity=INF, init=0)
        self.rations_al = simpy.Container(self.env, capacity=INF, init=0)
        self.rations_sb = simpy.Container(self.env, capacity=INF, init=0)
        self.rations_sb_dispatch = simpy.Container(self.env, capacity=INF, init=0)
        self.rations_cssu = simpy.Container(self.env, capacity=INF, init=0)
        self.rations_theatre = simpy.Container(self.env, capacity=INF, init=0)

        if initial_buffers:
            self.raw_material_wdc.put(initial_buffers.get("op3_rm", 0))
            self.raw_material_al.put(initial_buffers.get("op5_rm", 0))
            self.rations_sb.put(initial_buffers.get("op9_rations", 0))

        # =================================================================
        # METRICS
        # =================================================================
        self.orders = []
        self.total_produced = 0
        self.total_delivered = 0
        self.total_demanded = 0
        self.total_backorders = 0
        self.cumulative_backorder_qty = 0
        self.pending_backorders: list[OrderRecord] = []
        self.pending_backorder_qty = 0.0
        self.total_unattended_orders = 0
        self.warmup_complete = False
        self.warmup_time = 0.0

        # Time-series (sampled daily)
        self.daily_production = []
        self.daily_demand = []
        self.delivery_events = []
        self.daily_inventory_sb = []
        self.daily_inventory_theatre = []

        # Hourly assembly tracking
        self._hour_in_week = 0
        self._today_produced = 0
        self._pending_batch = 0
        self._in_transit = 0  # Material between containers (Op8/Op10 transport)

        # =================================================================
        # RISK STATE
        # =================================================================
        self.op_down_count = {i: 0 for i in range(1, 14)}
        self._op_down_since = {i: None for i in range(1, 14)}  # Time when op went down
        self.risk_events = []
        self._contingent_demand_pending = 0
        self._cumulative_down_hours = 0.0  # Accumulated op-hours of disruption

        # =================================================================
        # STEP API STATE (Phase 3)
        # =================================================================
        self._processes_started = False
        self._step_size = HOURS_PER_DAY  # Default: 24h per step

    # =====================================================================
    # RUN MODES
    # =====================================================================

    def _start_processes(self):
        """Launch all SimPy processes (called once)."""
        if self._processes_started:
            return
        self.env.process(self._op1_contracting())
        self.env.process(self._op2_supplier_delivery())
        self.env.process(self._op3_wdc_dispatch())
        self.env.process(self._assembly_hourly())  # HOURLY granularity
        self.env.process(self._op8_transport_to_sb())
        self.env.process(self._op9_sb_dispatch())
        self.env.process(self._op10_transport_to_cssu())
        self.env.process(self._op12_transport_to_theatre())
        self.env.process(self._op13_demand())
        self.env.process(self._daily_tracker())

        if self.risks_enabled:
            self.env.process(self._risk_R11())
            self.env.process(self._risk_R12())
            self.env.process(self._risk_R13())
            self.env.process(self._risk_R14())
            self.env.process(self._risk_R21())
            self.env.process(self._risk_R22())
            self.env.process(self._risk_R23())
            self.env.process(self._risk_R24())
            self.env.process(self._risk_R3())

        self._processes_started = True

    def run(self) -> "MFSCSimulation":
        """Full run (for validation and batch experiments)."""
        self._start_processes()
        self.env.run(until=self.horizon)
        return self

    def step(
        self,
        action: Optional[dict[str, float]] = None,
        step_hours: Optional[float] = None,
    ) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """
        RL step: advance simulation by step_hours, return (obs, reward, done, info).

        Parameters
        ----------
        action : dict or None
            Parameter overrides, e.g. {'op9_rop': 12, 'op3_q': 20000}
        step_hours : float
            Hours to advance. Default: self._step_size (24h).
        """
        self._start_processes()

        # Apply action (modify mutable params)
        if action:
            for k, v in action.items():
                if k in self.params:
                    self.params[k] = v

            # Auto-couple shift-dependent batch size per Table 6.20:
            # S=1/2 → 5,000 rations/batch, S=3 → 7,000 rations/batch.
            new_shifts = int(self.params["assembly_shifts"])
            if new_shifts in CAPACITY_BY_SHIFTS:
                cap = CAPACITY_BY_SHIFTS[new_shifts]
                self.params["batch_size"] = cap["op7_q"]

        # Advance simulation
        dt = step_hours or self._step_size
        target = min(self.env.now + dt, self.horizon)

        prev_backorders = self.total_backorders
        prev_delivered = self.total_delivered
        prev_demanded = self.total_demanded
        prev_backorder_qty = self.cumulative_backorder_qty
        prev_unattended_orders = self.total_unattended_orders
        prev_pending_backorders = len(self.pending_backorders)
        prev_pending_backorder_qty = self.pending_backorder_qty
        prev_down_hours = self._cumulative_down_hours
        # Flush ongoing disruptions at step start
        for op_id in range(1, 14):
            if self.op_down_count[op_id] > 0 and self._op_down_since[op_id] is not None:
                self._cumulative_down_hours += self.env.now - self._op_down_since[op_id]
                self._op_down_since[op_id] = self.env.now
        prev_down_hours = self._cumulative_down_hours

        self.env.run(until=target)

        # Flush ongoing disruptions at step end
        for op_id in range(1, 14):
            if self.op_down_count[op_id] > 0 and self._op_down_since[op_id] is not None:
                self._cumulative_down_hours += self.env.now - self._op_down_since[op_id]
                self._op_down_since[op_id] = self.env.now

        # Step deltas
        new_backorders = self.total_backorders - prev_backorders  # order count
        new_delivered = self.total_delivered - prev_delivered  # rations
        new_demanded = self.total_demanded - prev_demanded  # rations
        new_backorder_qty = (
            self.cumulative_backorder_qty - prev_backorder_qty
        )  # rations
        new_unattended_orders = self.total_unattended_orders - prev_unattended_orders
        step_disruption_hours = self._cumulative_down_hours - prev_down_hours

        # Proxy reward (default — env.py may override)
        reward = new_delivered - 10 * new_backorders

        inventory_detail = self._inventory_detail()
        total_inventory = sum(inventory_detail.values())

        done = self.env.now >= self.horizon
        obs = self.get_observation()
        info = {
            "time": self.env.now,
            "new_delivered": new_delivered,
            "new_backorders": new_backorders,  # order count
            "new_demanded": new_demanded,  # rations
            "new_backorder_qty": new_backorder_qty,  # rations short
            "new_unattended_orders": new_unattended_orders,  # order count
            "step_disruption_hours": step_disruption_hours,  # op-hours down
            "total_inventory": total_inventory,
            "inventory_detail": inventory_detail,
            "pending_backorders": len(self.pending_backorders),
            "pending_backorder_delta": len(self.pending_backorders)
            - prev_pending_backorders,
            "pending_backorder_qty": self.pending_backorder_qty,
            "pending_backorder_qty_delta": self.pending_backorder_qty
            - prev_pending_backorder_qty,
            "unattended_orders_total": self.total_unattended_orders,
        }
        return obs, reward, done, info

    def _inventory_detail(self) -> dict[str, float]:
        """Return a named snapshot of all tracked material buffers."""
        return {
            "raw_material_wdc": float(self.raw_material_wdc.level),
            "raw_material_al": float(self.raw_material_al.level),
            "rations_al": float(self.rations_al.level),
            "rations_sb": float(self.rations_sb.level),
            "rations_sb_dispatch": float(self.rations_sb_dispatch.level),
            "rations_cssu": float(self.rations_cssu.level),
            "rations_theatre": float(self.rations_theatre.level),
        }

    def _backorder_priority_key(
        self, order: OrderRecord
    ) -> tuple[int, float, float, int]:
        """
        Return the Garrido backlog priority key.

        Contingent demand takes precedence over regular demand. Within each
        priority class, delayed orders are sorted in increasing order of size
        as a proxy for the SPT scheduling rule described in the thesis.
        """
        return (
            0 if order.contingent else 1,
            float(order.remaining_qty),
            float(order.OPTj),
            int(order.j),
        )

    def _refresh_pending_backorder_qty(self) -> None:
        """Recompute the outstanding delayed-demand quantity."""
        self.pending_backorder_qty = float(
            sum(order.remaining_qty for order in self.pending_backorders)
        )

    def _enqueue_backorder(self, order: OrderRecord) -> None:
        """Insert a delayed order into the capped Garrido-style backlog queue."""
        self.pending_backorders.append(order)
        self.pending_backorders.sort(key=self._backorder_priority_key)
        while len(self.pending_backorders) > BACKORDER_QUEUE_CAP:
            dropped = self.pending_backorders.pop()
            dropped.lost = True
            self.total_unattended_orders += 1
        self._refresh_pending_backorder_qty()

    def _serve_pending_backorders(self):
        """
        Serve delayed orders according to the queue head.

        The queue is blocking: if the highest-priority delayed order cannot be
        fully served from on-hand theatre inventory, lower-priority orders wait
        behind it.
        """
        while self.pending_backorders:
            next_order = self.pending_backorders[0]
            if self.rations_theatre.level + 1e-9 < next_order.remaining_qty:
                break
            yield self.rations_theatre.get(next_order.remaining_qty)
            next_order.OATj = self.env.now
            next_order.CTj = self.env.now - next_order.OPTj
            next_order.backorder = False
            next_order.remaining_qty = 0.0
            self.pending_backorders.pop(0)
            self._refresh_pending_backorder_qty()

    def _backorder_rate(self) -> float:
        """Current delayed/lost-order fraction per Garrido's Bt + Ut logic."""
        if not self.orders:
            return 0.0
        delayed_orders = len(self.pending_backorders) + self.total_unattended_orders
        return min(1.0, delayed_orders / len(self.orders))

    def get_observation(self) -> np.ndarray:
        """
        Return normalized state vector for RL agent.

        Vector (15 dims):
          [0]  raw_material_wdc / 1e6
          [1]  raw_material_al / 1e6
          [2]  rations_al / 1e5
          [3]  rations_sb / 1e5
          [4]  rations_cssu / 1e5
          [5]  rations_theatre / 1e5
          [6]  fill_rate (served-or-recoverable orders / total_orders)
          [7]  backorder_rate ((pending + unattended) / total_orders)
          [8]  assembly_line_down (0 or 1)
          [9]  any_loc_down (0 or 1)
          [10] op9_down (0 or 1)
          [11] op11_down (0 or 1)
          [12] time_fraction (env.now / horizon)
          [13] pending_batch / batch_size
          [14] contingent_demand_pending / 2600
        """
        return np.array(
            [
                self.raw_material_wdc.level / 1e6,
                self.raw_material_al.level / 1e6,
                self.rations_al.level / 1e5,
                self.rations_sb.level / 1e5,
                self.rations_cssu.level / 1e5,
                self.rations_theatre.level / 1e5,
                self._fill_rate(),
                self._backorder_rate(),
                float(self._is_down(5) or self._is_down(6) or self._is_down(7)),
                float(
                    self._is_down(4)
                    or self._is_down(8)
                    or self._is_down(10)
                    or self._is_down(12)
                ),
                float(self._is_down(9)),
                float(self._is_down(11)),
                self.env.now / self.horizon,
                self._pending_batch / max(1, self.params["batch_size"]),
                self._contingent_demand_pending / 2600.0,
            ],
            dtype=np.float32,
        )

    # =====================================================================
    # HELPERS
    # =====================================================================

    def _is_down(self, op_id: int) -> bool:
        return self.op_down_count[op_id] > 0

    def _take_down(self, op_id: int) -> None:
        if self.op_down_count[op_id] == 0:
            self._op_down_since[op_id] = self.env.now
        self.op_down_count[op_id] += 1

    def _bring_up(self, op_id: int) -> None:
        self.op_down_count[op_id] = max(0, self.op_down_count[op_id] - 1)
        if self.op_down_count[op_id] == 0 and self._op_down_since[op_id] is not None:
            self._cumulative_down_hours += self.env.now - self._op_down_since[op_id]
            self._op_down_since[op_id] = None

    def _delayed_bring_up(self, op_id: int, delay: float):
        yield self.env.timeout(delay)
        self._bring_up(op_id)

    def _is_workday(self, hour_of_week: float) -> bool:
        """Mon-Sat = workday (days 0-5), Sun = off (day 6)."""
        return (hour_of_week // HOURS_PER_DAY) < 6

    def _is_work_hour(self, hour_of_day: float) -> bool:
        """Work hours: 0-7 (8h shift for S=1), 0-15 (S=2), 0-23 (S=3)."""
        return hour_of_day < (HOURS_PER_SHIFT * self.params["assembly_shifts"])

    # =====================================================================
    # STOCHASTIC PROCESSING TIMES
    # =====================================================================

    def _pt(self, param_key: str) -> float:
        """
        Return processing time for the given param, optionally with noise.

        When stochastic_pt is enabled, applies a right-skewed triangular
        distribution: Tri(0.75×base, base, 1.5×base). This models
        realistic variability where delays are more likely than speed-ups.

        Returns the deterministic base value when stochastic_pt is False.
        """
        base = self.params[param_key]
        if not self.stochastic_pt or self.deterministic_baseline or base <= 0:
            return base
        return float(self.rng.triangular(0.75 * base, base, 1.5 * base))

    def _select_uniform_discrete(self, lower: int, upper: int) -> int:
        """Return the midpoint for Cf0 validation or sample otherwise."""
        if self.deterministic_baseline:
            return int(round((lower + upper) / 2))
        return int(self.rng.integers(lower, upper + 1))

    # =====================================================================
    # UPSTREAM: Procurement (Op1-Op4)
    # =====================================================================

    def _op1_contracting(self):
        while True:
            yield self.env.timeout(self.params["op1_rop"])
            while self._is_down(1):
                yield self.env.timeout(1)
            yield self.env.timeout(self._pt("op1_pt"))

    def _op2_supplier_delivery(self):
        while True:
            yield self.env.timeout(self.params["op2_rop"])
            while self._is_down(2):
                yield self.env.timeout(1)
            yield self.env.timeout(self._pt("op2_pt"))
            total_delivery = self.params["op2_q"] * NUM_RAW_MATERIALS
            yield self.raw_material_wdc.put(total_delivery)

    def _op3_wdc_dispatch(self):
        while True:
            yield self.env.timeout(self.params["op3_rop"])
            while self._is_down(3):
                yield self.env.timeout(1)
            total_dispatch = self.params["op3_q"] * NUM_RAW_MATERIALS
            available = self.raw_material_wdc.level
            dispatch = min(total_dispatch, available)
            if dispatch > 0:
                yield self.raw_material_wdc.get(dispatch)
                yield self.env.timeout(self._pt("op3_pt"))
                # Op4: transport WDC → AL (separate operation per thesis)
                while self._is_down(4):
                    yield self.env.timeout(1)
                yield self.env.timeout(self._pt("op4_pt"))
                yield self.raw_material_al.put(dispatch)

    # =====================================================================
    # ASSEMBLY LINE — HOURLY GRANULARITY (fixes daily-check bias)
    # =====================================================================

    def _assembly_hourly(self):
        """
        Op5-7: Assembly line at HOURLY resolution.

        Each hour during work shifts on workdays (Mon-Sat):
        - Check if assembly line is down (Op5/6/7)
        - If up and raw materials available: produce RATIONS_PER_HOUR
        - Accumulate into batches (size read from params each tick)

        This correctly captures sub-day risks (R11 avg 2.2h).
        At S=1: 8 work hours/day × 320.5 = 2,564 rations/day (matches thesis).
        """
        week_hours = 7 * HOURS_PER_DAY  # 168

        while True:
            yield self.env.timeout(1)  # Hourly tick

            self._hour_in_week = (self._hour_in_week + 1) % week_hours
            day_of_week = self._hour_in_week // HOURS_PER_DAY
            hour_of_day = self._hour_in_week % HOURS_PER_DAY

            # Skip: Sunday or outside shift hours
            if day_of_week >= 6:
                continue
            shifts = self.params["assembly_shifts"]
            if hour_of_day >= (HOURS_PER_SHIFT * shifts):
                continue

            # Skip: assembly line down
            if self._is_down(5) or self._is_down(6) or self._is_down(7):
                continue

            # Produce
            rm_available = self.raw_material_al.level
            can_produce = min(RATIONS_PER_HOUR, rm_available)

            if can_produce > 0:
                yield self.raw_material_al.get(can_produce)
                self._pending_batch += can_produce
                self._today_produced += can_produce
                self.total_produced += can_produce

                # Ship complete batches (read batch_size live for shift changes)
                batch_size = self.params["batch_size"]
                while self._pending_batch >= batch_size:
                    self._pending_batch -= batch_size
                    yield self.rations_al.put(batch_size)

                if not self.warmup_complete and self.total_produced >= batch_size:
                    self.warmup_complete = True
                    self.warmup_time = self.env.now

    # =====================================================================
    # DOWNSTREAM: Distribution (Op8-Op12)
    # =====================================================================

    def _op8_transport_to_sb(self):
        while True:
            batch_size = self.params["batch_size"]
            yield self.rations_al.get(batch_size)
            self._in_transit += batch_size
            while self._is_down(8):
                yield self.env.timeout(1)
            yield self.env.timeout(self._pt("op8_pt"))
            self._in_transit -= batch_size
            yield self.rations_sb.put(batch_size)

    def _op9_sb_dispatch(self):
        """Op9: Supply Battalion — dispatch U(q_min, q_max), async PT=24h."""
        while True:
            yield self.env.timeout(self.params["op9_rop"])
            if self._is_down(9):
                continue
            q_min = self.params["op9_q_min"]
            q_max = self.params["op9_q_max"]
            available = self.rations_sb.level
            if available > 0:
                target = self._select_uniform_discrete(q_min, q_max)
                dispatch_qty = min(target, available)
                yield self.rations_sb.get(dispatch_qty)
                self._in_transit += dispatch_qty
                self.env.process(self._op9_deliver(dispatch_qty))

    def _op9_deliver(self, qty: float):
        """Async delivery for Op9 (PT=24h concurrent with next cycle)."""
        yield self.env.timeout(self._pt("op9_pt"))
        self._in_transit -= qty
        yield self.rations_sb_dispatch.put(qty)

    def _op10_transport_to_cssu(self):
        """Op10: LOC SB→CSSUs — dispatch U(q_min, q_max), async PT=24h."""
        while True:
            yield self.env.timeout(self.params["op10_rop"])
            if self._is_down(10):
                continue
            q_min = OPERATIONS[10]["q"][0]
            q_max = OPERATIONS[10]["q"][1]
            available = self.rations_sb_dispatch.level
            if available > 0:
                target = self._select_uniform_discrete(q_min, q_max)
                dispatch_qty = min(target, available)
                yield self.rations_sb_dispatch.get(dispatch_qty)
                self._in_transit += dispatch_qty
                self.env.process(self._op10_deliver(dispatch_qty))

    def _op10_deliver(self, qty: float):
        """Async delivery for Op10 (PT=24h transit)."""
        yield self.env.timeout(self._pt("op10_pt"))
        self._in_transit -= qty
        yield self.rations_cssu.put(qty)

    def _op12_transport_to_theatre(self):
        """Op12: LOC CSSUs→Theatre — dispatch U(q_min, q_max), async PT=24h."""
        while True:
            yield self.env.timeout(self.params["op12_rop"])
            if self._is_down(12) or self._is_down(11):
                continue
            q_min = OPERATIONS[12]["q"][0]
            q_max = OPERATIONS[12]["q"][1]
            available = self.rations_cssu.level
            if available > 0:
                target = self._select_uniform_discrete(q_min, q_max)
                dispatch_qty = min(target, available)
                yield self.rations_cssu.get(dispatch_qty)
                self._in_transit += dispatch_qty
                self.env.process(self._op12_deliver(dispatch_qty))

    def _op12_deliver(self, qty: float):
        """Async delivery for Op12 (PT=24h transit)."""
        yield self.env.timeout(self._pt("op12_pt"))
        self._in_transit -= qty
        yield self.rations_theatre.put(qty)
        self.total_delivered += qty
        self.delivery_events.append((self.env.now, qty))
        yield from self._serve_pending_backorders()

    # =====================================================================
    # DEMAND SINK: Op13
    # =====================================================================

    def _op13_demand(self):
        order_num = 0
        hour_of_week = 0
        while True:
            yield self.env.timeout(DEMAND["frequency_hrs"])
            hour_of_week = (hour_of_week + HOURS_PER_DAY) % (7 * HOURS_PER_DAY)
            day_of_week = hour_of_week // HOURS_PER_DAY
            if day_of_week >= 6:
                continue

            demand_qty = float(self._select_uniform_discrete(DEMAND["a"], DEMAND["b"]))
            contingent_qty = float(self._contingent_demand_pending)
            if contingent_qty > 0:
                demand_qty += contingent_qty
                self._contingent_demand_pending = 0

            self.total_demanded += demand_qty
            order_num += 1
            order = OrderRecord(
                j=order_num,
                OPTj=self.env.now,
                quantity=demand_qty,
                remaining_qty=demand_qty,
                contingent=contingent_qty > 0,
            )

            available = self.rations_theatre.level
            if not self.pending_backorders and available >= demand_qty:
                yield self.rations_theatre.get(demand_qty)
                order.OATj = self.env.now
                order.CTj = 0.0
                order.remaining_qty = 0.0
            else:
                order.backorder = True
                self.total_backorders += 1
                self.cumulative_backorder_qty += demand_qty
                self._enqueue_backorder(order)
                yield from self._serve_pending_backorders()

            self.orders.append(order)
            self.daily_demand.append((self.env.now, demand_qty))

    # =====================================================================
    # RISK PROCESSES
    # =====================================================================

    _RISK_TABLES = {
        "increased": RISKS_INCREASED,
        "severe": RISKS_SEVERE,
        "severe_extended": RISKS_SEVERE_EXTENDED,
    }

    def _get_risk_b(self, risk_id: str) -> float:
        table = self._RISK_TABLES.get(self.risk_level)
        if table and risk_id in table:
            return table[risk_id].get("b", RISKS_CURRENT[risk_id]["occurrence"]["b"])
        return RISKS_CURRENT[risk_id]["occurrence"]["b"]

    def _get_risk_p(self, risk_id: str) -> float:
        table = self._RISK_TABLES.get(self.risk_level)
        if table and risk_id in table:
            return table[risk_id].get("p", RISKS_CURRENT[risk_id]["occurrence"]["p"])
        return RISKS_CURRENT[risk_id]["occurrence"]["p"]

    def _get_risk_recovery_mean(self, risk_id: str) -> float:
        table = self._RISK_TABLES.get(self.risk_level)
        if table and risk_id in table:
            return table[risk_id].get(
                "recovery_mean", RISKS_CURRENT[risk_id]["recovery"]["mean"]
            )
        return RISKS_CURRENT[risk_id]["recovery"]["mean"]

    def _get_risk_surge(self) -> tuple[int, int]:
        table = self._RISK_TABLES.get(self.risk_level)
        base_lo = RISKS_CURRENT["R24"]["surge"]["lo"]
        base_hi = RISKS_CURRENT["R24"]["surge"]["hi"]
        if table and "R24" in table:
            return table["R24"].get("surge_lo", base_lo), table["R24"].get(
                "surge_hi", base_hi
            )
        return base_lo, base_hi

    def _risk_R11(self):
        a = RISKS_CURRENT["R11"]["occurrence"]["a"]
        b_val = self._get_risk_b("R11")
        beta = self._get_risk_recovery_mean("R11")
        while True:
            yield self.env.timeout(self.rng.integers(a, b_val + 1))
            start = self.env.now
            self._take_down(5)
            self._take_down(6)
            repair = max(1, self.rng.exponential(beta))
            yield self.env.timeout(repair)
            self._bring_up(5)
            self._bring_up(6)
            self.risk_events.append(
                RiskEvent("R11", start, self.env.now, self.env.now - start, [5, 6])
            )

    def _risk_R12(self):
        n = RISKS_CURRENT["R12"]["occurrence"]["n"]
        p = self._get_risk_p("R12")
        while True:
            yield self.env.timeout(self.params["op1_rop"])
            delayed = self.rng.binomial(n, p)
            if delayed > 0:
                delay = delayed * 168
                start = self.env.now
                self._take_down(1)
                yield self.env.timeout(delay)
                self._bring_up(1)
                self.risk_events.append(
                    RiskEvent("R12", start, self.env.now, delay, [1])
                )

    def _risk_R13(self):
        n = RISKS_CURRENT["R13"]["occurrence"]["n"]
        p = self._get_risk_p("R13")
        while True:
            yield self.env.timeout(self.params["op2_rop"])
            delayed = self.rng.binomial(n, p)
            if delayed > 0:
                delay = delayed * 24
                start = self.env.now
                self._take_down(2)
                yield self.env.timeout(delay)
                self._bring_up(2)
                self.risk_events.append(
                    RiskEvent("R13", start, self.env.now, delay, [2])
                )

    def _risk_R14(self):
        p = self._get_risk_p("R14")
        while True:
            yield self.env.timeout(HOURS_PER_DAY)
            produced = self._today_produced
            self._today_produced = 0  # Reset daily counter
            if produced > 0:
                defects = self.rng.binomial(produced, p)
                if defects > 0:
                    # Cap at available pending to maintain mass balance
                    defects = min(defects, int(self._pending_batch))
                    if defects > 0:
                        self._pending_batch -= defects
                        self.total_produced -= defects
                        self.risk_events.append(
                            RiskEvent(
                                "R14",
                                self.env.now,
                                self.env.now,
                                0,
                                [7],
                                f"{defects} defective",
                            )
                        )

    def _risk_R21(self):
        a = RISKS_CURRENT["R21"]["occurrence"]["a"]
        b_val = self._get_risk_b("R21")
        beta = RISKS_CURRENT["R21"]["recovery"]["mean"]
        affected = RISKS_CURRENT["R21"]["affected_ops"]
        while True:
            yield self.env.timeout(self.rng.integers(a, b_val + 1))
            start = self.env.now
            for op_id in affected:
                self._take_down(op_id)
            recovery_times = {}
            for op_id in affected:
                rt = max(1, self.rng.exponential(beta))
                recovery_times[op_id] = rt
                self.env.process(self._delayed_bring_up(op_id, rt))
            max_rt = max(recovery_times.values())
            yield self.env.timeout(max_rt)
            self.risk_events.append(
                RiskEvent("R21", start, self.env.now, max_rt, list(affected))
            )

    def _risk_R22(self):
        a = RISKS_CURRENT["R22"]["occurrence"]["a"]
        b_val = self._get_risk_b("R22")
        beta = RISKS_CURRENT["R22"]["recovery"]["mean"]
        loc_ops = RISKS_CURRENT["R22"]["affected_ops"]
        while True:
            yield self.env.timeout(self.rng.integers(a, b_val + 1))
            target = int(self.rng.choice(loc_ops))
            start = self.env.now
            self._take_down(target)
            recovery = max(1, self.rng.exponential(beta))
            yield self.env.timeout(recovery)
            self._bring_up(target)
            self.risk_events.append(
                RiskEvent("R22", start, self.env.now, recovery, [target])
            )

    def _risk_R23(self):
        a = RISKS_CURRENT["R23"]["occurrence"]["a"]
        b_val = self._get_risk_b("R23")
        beta = RISKS_CURRENT["R23"]["recovery"]["mean"]
        while True:
            yield self.env.timeout(self.rng.integers(a, b_val + 1))
            start = self.env.now
            self._take_down(11)
            recovery = max(1, self.rng.exponential(beta))
            yield self.env.timeout(recovery)
            self._bring_up(11)
            self.risk_events.append(
                RiskEvent("R23", start, self.env.now, recovery, [11])
            )

    def _risk_R24(self):
        a = RISKS_CURRENT["R24"]["occurrence"]["a"]
        b_val = self._get_risk_b("R24")
        while True:
            yield self.env.timeout(self.rng.integers(a, b_val + 1))
            surge_lo, surge_hi = self._get_risk_surge()
            surge = self.rng.integers(surge_lo, surge_hi + 1)
            self._contingent_demand_pending += surge
            self.risk_events.append(
                RiskEvent("R24", self.env.now, self.env.now, 0, [13], f"+{surge}")
            )

    def _risk_R3(self):
        a = RISKS_CURRENT["R3"]["occurrence"]["a"]
        b_val = self._get_risk_b("R3")
        duration = RISKS_CURRENT["R3"]["recovery"]["duration"]
        affected = RISKS_CURRENT["R3"]["affected_ops"]
        while True:
            yield self.env.timeout(self.rng.integers(a, b_val + 1))
            start = self.env.now
            for op_id in affected:
                self._take_down(op_id)
            yield self.env.timeout(duration)
            for op_id in affected:
                self._bring_up(op_id)
            self.risk_events.append(
                RiskEvent("R3", start, self.env.now, duration, list(affected))
            )

    # =====================================================================
    # REPORTING
    # =====================================================================

    def _daily_tracker(self):
        prev_total = 0
        while True:
            yield self.env.timeout(HOURS_PER_DAY)
            self.daily_inventory_sb.append((self.env.now, self.rations_sb.level))
            self.daily_inventory_theatre.append(
                (self.env.now, self.rations_theatre.level)
            )
            # Use cumulative delta — avoids race with R14 and works in det mode
            today_net = self.total_produced - prev_total
            prev_total = self.total_produced
            self.daily_production.append((self.env.now, today_net))

    def get_annual_throughput(
        self,
        *,
        start_time: float = 0.0,
        num_years: Optional[int] = None,
    ) -> dict[str, Any]:
        hours_per_year = self.hours_per_year
        end_time = (
            start_time + num_years * hours_per_year
            if num_years is not None
            else self.horizon
        )
        yearly_produced = {}
        for t, qty in self.daily_production:
            if not (start_time <= t < end_time):
                continue
            year = int((t - start_time) // hours_per_year) + 1
            yearly_produced.setdefault(year, 0)
            yearly_produced[year] += qty
        produced_total = sum(
            qty for t, qty in self.daily_production if start_time <= t < end_time
        )
        delivered_total = sum(
            qty for t, qty in self.delivery_events if start_time <= t < end_time
        )
        years = (
            float(num_years)
            if num_years is not None
            else (end_time - start_time) / hours_per_year
        )
        return {
            "produced_by_year": yearly_produced,
            "avg_annual_delivery": delivered_total / years,
            "avg_annual_production": produced_total / years,
            "hours_per_year": hours_per_year,
            "year_basis": self.year_basis,
            "start_time": start_time,
            "num_years": num_years,
        }

    def risk_summary(self):
        if not self.risk_events:
            print("  No risk events.")
            return
        counts = Counter(e.risk_id for e in self.risk_events)
        total_dt = {}
        for e in self.risk_events:
            total_dt.setdefault(e.risk_id, 0)
            total_dt[e.risk_id] += e.duration
        print(f"\n{'='*60}")
        print("  Risk Event Summary")
        print(f"{'='*60}")
        print(f"  {'Risk':<8} {'Count':>8} {'Total Down':>12} {'Avg':>10}")
        print(f"  {'-'*40}")
        for rid in sorted(counts.keys()):
            c = counts[rid]
            dt = total_dt[rid]
            print(f"  {rid:<8} {c:>8} {dt:>11,.0f}h {dt/c:>9,.1f}h")
        print(f"\n  Total: {len(self.risk_events)} events")
        print(f"{'='*60}\n")

    def summary(self) -> None:
        years = self.horizon / self.hours_per_year
        mode = f"ENABLED ({self.risk_level})" if self.risks_enabled else "DISABLED"
        print(f"\n{'='*60}")
        print("  MFSC Simulation Summary")
        print(f"  Horizon: {self.horizon:,} hrs ({years:.1f} years)")
        print(f"  Shifts: S={self.shifts}  |  Risks: {mode}")
        print(
            f"  Granularity: HOURLY (assembly)  |  Seed: {self.seed}  |"
            f"  Year basis: {self.year_basis} ({self.hours_per_year:,}h)"
        )
        print(f"{'='*60}")
        print(f"  Warmup:         {self.warmup_time:,.0f} hrs")
        print(f"  Produced:       {self.total_produced:,}")
        print(f"  Delivered:      {self.total_delivered:,}")
        print(f"  Demanded:       {self.total_demanded:,}")
        print(f"  Orders:         {len(self.orders):,}")
        print(f"  Backorders:     {self.total_backorders:,}")
        print(f"  Pending queue:  {len(self.pending_backorders):,}")
        print(f"  Unattended Ut:  {self.total_unattended_orders:,}")
        print(f"  Fill rate:      {self._fill_rate():.1%}")
        print(f"  Avg ann. prod:  {self.total_produced / years:,.0f}")
        print(f"  Avg ann. del:   {self.total_delivered / years:,.0f}")
        print(f"{'='*60}")
        if self.risks_enabled:
            self.risk_summary()

    def _fill_rate(self):
        """Current fill rate per Garrido's order-based Bt + Ut formulation."""
        return max(0.0, 1.0 - self._backorder_rate())
