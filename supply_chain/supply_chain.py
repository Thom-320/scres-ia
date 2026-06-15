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

from .config import (
    ADAPTIVE_BENCHMARK_INITIAL_REGIME,
    ADAPTIVE_BENCHMARK_MAINTENANCE,
    ADAPTIVE_BENCHMARK_REGIME_PARAMS,
    ADAPTIVE_BENCHMARK_REGIMES,
    ADAPTIVE_BENCHMARK_REVIEW_HOURS,
    ADAPTIVE_BENCHMARK_TRANSITIONS,
    ADAPTIVE_BENCHMARK_V2_RECOVERY_MULTIPLIERS,
    ADAPTIVE_BENCHMARK_V2_RISK_MULTIPLIERS,
    ADAPTIVE_BENCHMARK_V2_SURGE_SCALE_MULTIPLIER,
    OPERATIONS,
    DEMAND,
    ASSEMBLY_RATE,
    BACKORDER_QUEUE_CAP,
    CAPACITY_BY_SHIFTS,
    HOURS_PER_SHIFT,
    HOURS_PER_DAY,
    HOURS_PER_WEEK,
    SIMULATION_HORIZON,
    NUM_RAW_MATERIALS,
    RISKS_CURRENT,
    RISKS_INCREASED,
    RISKS_SEVERE,
    RISKS_SEVERE_EXTENDED,
    RISKS_SEVERE_TRAINING,
    TRACK_B_QUEUE_PRESSURE_LOOKAHEAD_CYCLES,
    TRACK_B_ROLLING_WINDOW_HOURS,
    DEFAULT_YEAR_BASIS,
    HOURS_PER_YEAR_GREGORIAN,
    HOURS_PER_YEAR_THESIS,
    R14_DEFECT_MODE_OPTIONS,
    RAW_MATERIAL_FLOW_MODE_OPTIONS,
    canonical_raw_material_flow_mode,
    YEAR_BASIS_OPTIONS,
    LEAD_TIME_PROMISE,
    THESIS_FAITHFUL_PROTOCOL,
    THESIS_DOWNSTREAM_Q_RANGES,
    WARMUP_TRIGGER_OPTIONS,
)
from .ret_thesis import (
    compute_order_level_ret as compute_thesis_order_level_ret,
    compute_ret_per_order,
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
    # LTj: fixed lead time promise per Garrido-Rios (2017) Section 6.8.2
    # An order is a backorder if CTj > LTj (i.e., not delivered within 48h).
    LTj: float = LEAD_TIME_PROMISE  # 48 hours — thesis-defined fixed lead time
    backorder: bool = False
    remaining_qty: float = 0.0
    contingent: bool = False
    lost: bool = False
    # Thesis ReT sub-indicators (Garrido-Rios 2017, Eq. 5.1-5.5):
    APj: float = 0.0  # Autotomy period (hours): CTj=LTj and risks impact in [OPTj,OATj]
    RPj: float = 0.0  # Recovery period (hours): OATj - first R0cr detection
    DPj: float = 0.0  # Disruption period (hours): CTj when CTj > LTj


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
        warmup_trigger: str = "production",
        downstream_q_source: str = "figure_6_2",
        r14_defect_mode: str = "reprocess",
        enabled_risks: Optional[set[str]] = None,
        risk_overrides: Optional[dict[str, str]] = None,
        inventory_replenishment_period: Optional[float] = None,
        raw_material_flow_mode: str = "legacy_validated",
        raw_material_order_up_to_multiplier: float = 2.0,
    ) -> None:
        if warmup_trigger not in WARMUP_TRIGGER_OPTIONS:
            valid = ", ".join(WARMUP_TRIGGER_OPTIONS)
            raise ValueError(
                f"Invalid warmup_trigger={warmup_trigger!r}. Expected one of: {valid}."
            )
        if downstream_q_source not in THESIS_DOWNSTREAM_Q_RANGES:
            valid = ", ".join(sorted(THESIS_DOWNSTREAM_Q_RANGES))
            raise ValueError(
                f"Invalid downstream_q_source={downstream_q_source!r}. Expected one of: {valid}."
            )
        if r14_defect_mode not in R14_DEFECT_MODE_OPTIONS:
            valid = ", ".join(R14_DEFECT_MODE_OPTIONS)
            raise ValueError(
                f"Invalid r14_defect_mode={r14_defect_mode!r}. Expected one of: {valid}."
            )
        if raw_material_flow_mode not in RAW_MATERIAL_FLOW_MODE_OPTIONS:
            valid = ", ".join(RAW_MATERIAL_FLOW_MODE_OPTIONS)
            raise ValueError(
                "Invalid raw_material_flow_mode="
                f"{raw_material_flow_mode!r}. Expected one of: {valid}."
            )
        raw_material_flow_mode = canonical_raw_material_flow_mode(raw_material_flow_mode)
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
        self.warmup_trigger = warmup_trigger
        self.downstream_q_source = downstream_q_source
        self.r14_defect_mode = r14_defect_mode
        self.raw_material_flow_mode = raw_material_flow_mode
        self.raw_material_order_up_to_multiplier = float(
            raw_material_order_up_to_multiplier
        )
        if self.raw_material_order_up_to_multiplier <= 0.0:
            raise ValueError("raw_material_order_up_to_multiplier must be > 0.")
        self._raw_units_per_ration = (
            float(NUM_RAW_MATERIALS)
            if raw_material_flow_mode.startswith("bom_total_units")
            else 1.0
        )
        self.enabled_risks = set(enabled_risks) if enabled_risks is not None else None
        self.risk_overrides = dict(risk_overrides or {})
        self.inventory_replenishment_period = inventory_replenishment_period
        self.hours_per_year = resolve_hours_per_year(year_basis)
        downstream_ranges = THESIS_DOWNSTREAM_Q_RANGES[downstream_q_source]
        op9_q = downstream_ranges["op9"]
        op10_q = downstream_ranges["op10"]
        op12_q = downstream_ranges["op12"]

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
            "op9_q_min": op9_q[0],
            "op9_q_max": op9_q[1],
            "op10_rop": OPERATIONS[10]["rop"],  # 24 hrs
            "op10_pt": OPERATIONS[10]["pt"],  # 24 hrs
            "op10_q_min": op10_q[0],
            "op10_q_max": op10_q[1],
            "op12_rop": OPERATIONS[12]["rop"],  # 24 hrs
            "op12_pt": OPERATIONS[12]["pt"],  # 24 hrs
            "op12_q_min": op12_q[0],
            "op12_q_max": op12_q[1],
        }
        if shifts in CAPACITY_BY_SHIFTS:
            capacity = CAPACITY_BY_SHIFTS[shifts]
            self.params["op3_q"] = capacity["op3_q"]
            self.params["batch_size"] = capacity["op7_q"]

        # =================================================================
        # MATERIAL BUFFERS
        # =================================================================
        INF = 10_000_000
        self.raw_material_wdc = simpy.Container(self.env, capacity=INF, init=0)
        self.raw_material_al = simpy.Container(self.env, capacity=INF, init=0)
        self.rework_op6 = simpy.Container(self.env, capacity=INF, init=0)
        self.rations_al = simpy.Container(self.env, capacity=INF, init=0)
        self.rations_sb = simpy.Container(self.env, capacity=INF, init=0)
        self.rations_sb_dispatch = simpy.Container(self.env, capacity=INF, init=0)
        self.rations_cssu = simpy.Container(self.env, capacity=INF, init=0)
        self.rations_theatre = simpy.Container(self.env, capacity=INF, init=0)

        self.inventory_buffer_targets = self._normalize_inventory_buffer_targets(
            initial_buffers or {}
        )
        if self.inventory_buffer_targets:
            op3_rm = float(self.inventory_buffer_targets.get("op3_rm", 0))
            op5_rm = float(self.inventory_buffer_targets.get("op5_rm", 0))
            op9_rations = float(self.inventory_buffer_targets.get("op9_rations", 0))
            if op3_rm > 0:
                self.raw_material_wdc.put(op3_rm)
            if op5_rm > 0:
                self.raw_material_al.put(op5_rm)
            if op9_rations > 0:
                self.rations_sb.put(op9_rations)
        # Cache the original Op5 buffer target as a separate attribute (do not pollute
        # inventory_buffer_targets, which other tests compare as a strict dict).
        # See Table 6.16 (Op5,j) — the agent's a5 multiplier scales this baseline.
        self._op5_rm_base: Optional[float] = None
        if "op5_rm" in self.inventory_buffer_targets:
            self._op5_rm_base = float(self.inventory_buffer_targets["op5_rm"])
        # Thesis-aligned Op5 multiplier rule: a5 in [-1, 1] maps linearly to
        # m = 1 + 0.5 * a5 in [0.5, 1.5] (centre at 1.0x, neutral when a5 = 0).
        # This is the one place the repo diverges from the generic 1.25 + 0.75
        # rule used for op3/op9/op10/op12: op5 is anchored around 1.0x baseline
        # so the agent can leave the thesis buffer unchanged by emitting a5 = 0.
        self._op5_multiplier_rule: str = "thesis_anchored_centered"

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
        self._cumulative_available_assembly_hours = 0.0

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
        self.adaptive_benchmark_enabled = self.risk_level in (
            "adaptive_benchmark_v1",
            "adaptive_benchmark_v2",
        )
        self.adaptive_benchmark_v2_enabled = self.risk_level == "adaptive_benchmark_v2"
        self.adaptive_regime = ADAPTIVE_BENCHMARK_INITIAL_REGIME
        self.adaptive_risk_forecast_48h = 0.0
        self.adaptive_risk_forecast_168h = 0.0
        self.maintenance_debt = 0.0

        # =================================================================
        # STEP API STATE (Phase 3)
        # =================================================================
        self._processes_started = False
        self._step_size = HOURS_PER_DAY  # Default: 24h per step

    def _mark_warmup_complete(self) -> None:
        """Mark the first thesis warm-up trigger time once."""
        if not self.warmup_complete:
            self.warmup_complete = True
            self.warmup_time = self.env.now

    def _normalize_inventory_buffer_targets(
        self, targets: dict[str, float]
    ) -> dict[str, float]:
        """Convert external thesis buffer targets to internal container units."""
        normalized = {key: float(value) for key, value in targets.items()}
        if self.raw_material_flow_mode.startswith("bom_total_units"):
            for key in ("op3_rm", "op5_rm"):
                if key in normalized:
                    normalized[key] *= float(NUM_RAW_MATERIALS)
        return normalized

    def _top_up_inventory_buffer(self, key: str, target: float) -> Any:
        if key == "op3_rm":
            container = self.raw_material_wdc
        elif key == "op5_rm":
            container = self.raw_material_al
        elif key == "op9_rations":
            container = self.rations_sb
        else:
            return None
        shortfall = max(0.0, float(target) - float(container.level))
        if shortfall > 0.0:
            return container.put(shortfall)
        return None

    def _target_for_raw_node(self, key: str, fallback: float) -> float:
        """Return an order-up-to target for raw-material operating stock."""
        operating_target = float(fallback) * self.raw_material_order_up_to_multiplier
        return max(operating_target, float(self.inventory_buffer_targets.get(key, 0.0)))

    def _inventory_buffer_replenishment(self):
        """Top up thesis strategic buffers to their target level every t hours."""
        while True:
            period = float(self.inventory_replenishment_period or 0.0)
            if period <= 0.0:
                return
            yield self.env.timeout(period)
            for key, target in self.inventory_buffer_targets.items():
                event = self._top_up_inventory_buffer(key, float(target))
                if event is not None:
                    yield event

    def get_sdm_history_context(
        self, window_hours: float = HOURS_PER_WEEK
    ) -> dict[str, float]:
        """Return compact Table 6.25-style order/risk history for recent operations."""
        current_time = float(self.env.now)
        window_start = max(0.0, current_time - float(window_hours))
        recent_orders = [
            order
            for order in self.orders
            if float(order.OPTj) >= window_start
            or (order.OATj is not None and float(order.OATj) >= window_start)
        ]
        completed_orders = [order for order in recent_orders if order.OATj is not None]
        backorder_orders = [order for order in recent_orders if order.backorder]
        lost_orders = [order for order in recent_orders if order.lost]
        cycle_times = [
            float(order.CTj) for order in completed_orders if order.CTj is not None
        ]
        ret_case_counts: Counter[str] = Counter()
        for order in completed_orders:
            _, case = compute_ret_per_order(order, fill_rate=self._fill_rate())
            ret_case_counts[str(case)] += 1

        recent_risks = [
            event
            for event in self.risk_events
            if float(event.end_time) >= window_start
            and float(event.start_time) <= current_time
        ]
        risk_category_counts = Counter(
            "R3" if event.risk_id == "R3" else event.risk_id[:2]
            for event in recent_risks
        )
        risk_category_duration = Counter()
        op_risk_counts = Counter()
        for event in recent_risks:
            overlap_start = max(float(event.start_time), window_start)
            overlap_end = min(float(event.end_time), current_time)
            overlap_duration = max(0.0, overlap_end - overlap_start)
            category = "R3" if event.risk_id == "R3" else event.risk_id[:2]
            risk_category_duration[category] += overlap_duration
            for op_id in event.affected_ops:
                op_risk_counts[f"op{int(op_id)}"] += 1

        recent_demand_qty = sum(float(order.quantity) for order in recent_orders)
        recent_remaining_qty = sum(
            float(order.remaining_qty) for order in recent_orders
        )
        return {
            "sdm_recent_order_count": float(len(recent_orders)),
            "sdm_completed_order_count": float(len(completed_orders)),
            "sdm_backorder_order_count": float(len(backorder_orders)),
            "sdm_lost_order_count": float(len(lost_orders)),
            "sdm_recent_demand_qty": float(recent_demand_qty),
            "sdm_recent_remaining_qty": float(recent_remaining_qty),
            "sdm_mean_ct_hours": float(np.mean(cycle_times)) if cycle_times else 0.0,
            "sdm_max_ct_hours": float(np.max(cycle_times)) if cycle_times else 0.0,
            "sdm_sum_ap_hours": float(sum(order.APj for order in recent_orders)),
            "sdm_sum_rp_hours": float(sum(order.RPj for order in recent_orders)),
            "sdm_sum_dp_hours": float(sum(order.DPj for order in recent_orders)),
            "sdm_ret_case_fill_rate_count": float(ret_case_counts["fill_rate"]),
            "sdm_ret_case_autotomy_count": float(ret_case_counts["autotomy"]),
            "sdm_ret_case_recovery_count": float(ret_case_counts["recovery"]),
            "sdm_ret_case_non_recovery_count": float(ret_case_counts["non_recovery"]),
            "sdm_r1_event_count": float(risk_category_counts["R1"]),
            "sdm_r2_event_count": float(risk_category_counts["R2"]),
            "sdm_r3_event_count": float(risk_category_counts["R3"]),
            "sdm_r1_duration_hours": float(risk_category_duration["R1"]),
            "sdm_r2_duration_hours": float(risk_category_duration["R2"]),
            "sdm_r3_duration_hours": float(risk_category_duration["R3"]),
            "sdm_risk_affected_ops_count": float(sum(op_risk_counts.values())),
        }

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
        if (
            self.inventory_buffer_targets
            and self.inventory_replenishment_period is not None
        ):
            self.env.process(self._inventory_buffer_replenishment())

        if self.risks_enabled:
            risk_processes = {
                "R11": self._risk_R11,
                "R12": self._risk_R12,
                "R13": self._risk_R13,
                "R14": self._risk_R14,
                "R21": self._risk_R21,
                "R22": self._risk_R22,
                "R23": self._risk_R23,
                "R24": self._risk_R24,
                "R3": self._risk_R3,
            }
            enabled = self.enabled_risks or set(risk_processes)
            for risk_id, process_factory in risk_processes.items():
                if risk_id in enabled:
                    self.env.process(process_factory())
        if self.adaptive_benchmark_enabled:
            self.env.process(self._adaptive_regime_controller())

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
                # Thesis-aligned Op5 buffer target (Table 6.16 I_{t,S} on Op5,j).
                # The thesis replenishes raw-material buffer at Op5 every
                # 168/336/504/672/1344 h to a target level. Track A exposes
                # this as a continuous multiplier on the Op5 target.
                elif k == "op5_q" and "op5_rm" in self.inventory_buffer_targets:
                    if self._op5_rm_base is not None and self._op5_rm_base > 0.0:
                        new_target = self._op5_rm_base * float(v)
                        self.inventory_buffer_targets["op5_rm"] = new_target

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
        prev_produced = self.total_produced
        prev_backorder_qty = self.cumulative_backorder_qty
        prev_unattended_orders = self.total_unattended_orders
        prev_pending_backorders = len(self.pending_backorders)
        prev_pending_backorder_qty = self.pending_backorder_qty
        prev_available_assembly_hours = self._cumulative_available_assembly_hours
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
        new_produced = self.total_produced - prev_produced  # rations
        new_backorder_qty = (
            self.cumulative_backorder_qty - prev_backorder_qty
        )  # rations
        new_unattended_orders = self.total_unattended_orders - prev_unattended_orders
        new_available_assembly_hours = (
            self._cumulative_available_assembly_hours - prev_available_assembly_hours
        )
        new_available_assembly_capacity = (
            new_available_assembly_hours * RATIONS_PER_HOUR
        )
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
            "new_produced": new_produced,  # rations produced at Op5-Op7
            "new_backorder_qty": new_backorder_qty,  # rations short
            "new_unattended_orders": new_unattended_orders,  # order count
            "new_available_assembly_hours": new_available_assembly_hours,
            "new_available_assembly_capacity": new_available_assembly_capacity,
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
            "rework_op6": float(self.rework_op6.level),
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
            sum(
                max(0.0, float(order.remaining_qty))
                for order in self.pending_backorders
            )
        )

    def _remove_pending_backorder(self, order: OrderRecord) -> None:
        """Remove a queued delayed order if it is still present.

        The Track B transport lanes can trigger overlapping theatre-delivery
        callbacks. Those callbacks may attempt to finalize the same queued order
        in adjacent simulation events. Removal therefore needs to be idempotent
        rather than assuming the current order is still the queue head.
        """
        if order in self.pending_backorders:
            self.pending_backorders.remove(order)
        self._refresh_pending_backorder_qty()

    def _finalize_pending_backorder(self, order: OrderRecord) -> None:
        """Mark a delayed order as fulfilled at the current simulation time."""
        order.OATj = self.env.now
        order.CTj = self.env.now - order.OPTj
        order.backorder = False
        order.remaining_qty = 0.0
        self._set_order_ret_indicators(order)

    def _enqueue_backorder(self, order: OrderRecord) -> None:
        """Insert a delayed order into the capped Garrido-style backlog queue."""
        self.pending_backorders.append(order)
        self.pending_backorders.sort(key=self._backorder_priority_key)
        while len(self.pending_backorders) > BACKORDER_QUEUE_CAP:
            dropped = self.pending_backorders.pop()
            dropped.lost = True
            self.total_unattended_orders += 1
        self._refresh_pending_backorder_qty()

    def _delayed_backorder_check(self, order: OrderRecord):
        """Wait LTj hours, then classify as backorder if still unfulfilled.

        Per thesis Sec. 6.8.2: backorder = order not delivered within the
        pre-set lead time of 48 hours. Orders fulfilled within LTj are
        on-time (not backorders).
        """
        yield self.env.timeout(order.LTj)
        if order.remaining_qty > 0:
            order.backorder = True
            self.total_backorders += 1
            self.cumulative_backorder_qty += order.quantity

    def _serve_pending_backorders(self):
        """
        Serve delayed orders according to the queue head.

        The queue is blocking: if the highest-priority delayed order cannot be
        fully served from on-hand theatre inventory, lower-priority orders wait
        behind it.
        """
        while self.pending_backorders:
            next_order = self.pending_backorders[0]
            requested_qty = float(next_order.remaining_qty)
            if requested_qty <= 1e-9:
                # Order already fully served or numerically empty.
                self._finalize_pending_backorder(next_order)
                self._remove_pending_backorder(next_order)
                continue
            if self.rations_theatre.level + 1e-9 < requested_qty:
                break
            yield self.rations_theatre.get(requested_qty)
            self._finalize_pending_backorder(next_order)
            self._remove_pending_backorder(next_order)

    def _backorder_rate(self) -> float:
        """Current delayed/lost-order fraction per Garrido's Bt + Ut logic.

        Per thesis Sec. 6.8.2, only orders pending beyond LTj=48h count
        as backorders (Bt). Orders still within their lead-time window
        are not yet classified as backorders.
        """
        if not self.orders:
            return 0.0
        now = self.env.now
        delayed_orders = (
            sum(1 for o in self.pending_backorders if (now - o.OPTj) > o.LTj)
            + self.total_unattended_orders
        )
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

    def get_observation_v4_extra(self) -> np.ndarray:
        """
        Return 4 additional state features for obs v4.

        [0]  rations_sb_dispatch / 1e5  (intermediate buffer between Op9 and Op10)
        [1]  assembly_shifts_active / 3  (current shifts normalized to [0,1])
        [2]  op1_down (0 or 1)  (Military Logistics Agency, affected by R12)
        [3]  op2_down (0 or 1)  (Suppliers, affected by R13)
        """
        return np.array(
            [
                self.rations_sb_dispatch.level / 1e5,
                self.params["assembly_shifts"] / 3.0,
                float(self._is_down(1)),
                float(self._is_down(2)),
            ],
            dtype=np.float32,
        )

    def get_observation_v5_extra(self) -> np.ndarray:
        """
        Return 6 research-only cycle features for obs v5.

        [0]  op1_cycle_phase_norm   = (env.now mod op1_rop) / op1_rop
        [1]  op2_cycle_phase_norm   = (env.now mod op2_rop) / op2_rop
        [2]  workweek_phase_sin_norm = (sin(2π·week_phase) + 1) / 2
        [3]  workweek_phase_cos_norm = (cos(2π·week_phase) + 1) / 2
        [4]  workday_phase_sin_norm  = (sin(2π·day_phase) + 1) / 2
        [5]  workday_phase_cos_norm  = (cos(2π·day_phase) + 1) / 2

        These features do not change the DES dynamics. They expose the
        thesis-faithful operational calendar and reorder-cycle phases so
        adaptive policies can exploit anticipation without modifying the
        underlying stochastic processes.
        """
        op1_rop = max(float(self.params["op1_rop"]), 1.0)
        op2_rop = max(float(self.params["op2_rop"]), 1.0)
        now = float(self.env.now)
        op1_phase = (now % op1_rop) / op1_rop
        op2_phase = (now % op2_rop) / op2_rop
        week_phase = (now % HOURS_PER_WEEK) / HOURS_PER_WEEK
        day_phase = (now % HOURS_PER_DAY) / HOURS_PER_DAY
        return np.array(
            [
                op1_phase,
                op2_phase,
                0.5 * (1.0 + float(np.sin(2.0 * np.pi * week_phase))),
                0.5 * (1.0 + float(np.cos(2.0 * np.pi * week_phase))),
                0.5 * (1.0 + float(np.sin(2.0 * np.pi * day_phase))),
                0.5 * (1.0 + float(np.cos(2.0 * np.pi * day_phase))),
            ],
            dtype=np.float32,
        )

    def get_observation_v6_extra(self) -> np.ndarray:
        """
        Return 10 Track-B adaptive-control features for obs v6.

        [0:5]  operating regime one-hot: nominal, strained, pre_disruption,
               disrupted, recovery
        [5]    disruption forecast over next 48h (normalized)
        [6]    disruption forecast over next 168h (normalized)
        [7]    maintenance debt carried from sustained S3 usage
        [8]    average pending-backorder age normalized by config horizon
        [9]    theatre cover days normalized by config horizon
        """
        regime_one_hot = np.zeros(len(ADAPTIVE_BENCHMARK_REGIMES), dtype=np.float32)
        regime_index = ADAPTIVE_BENCHMARK_REGIMES.index(self.adaptive_regime)
        regime_one_hot[regime_index] = 1.0
        return np.concatenate(
            [
                regime_one_hot,
                np.array(
                    [
                        float(np.clip(self.adaptive_risk_forecast_48h, 0.0, 1.0)),
                        float(np.clip(self.adaptive_risk_forecast_168h, 0.0, 1.0)),
                        float(np.clip(self.maintenance_debt, 0.0, 1.0)),
                        float(np.clip(self._pending_backorder_age_norm(), 0.0, 1.0)),
                        float(np.clip(self._theatre_cover_days_norm(), 0.0, 1.0)),
                    ],
                    dtype=np.float32,
                ),
            ]
        )

    def _rolling_service_metrics(self) -> tuple[float, float]:
        """Return 4-week rolling fill and backorder rates for Track B."""
        if not self.orders:
            return float(self._fill_rate()), float(self._backorder_rate())

        window_start = max(
            0.0, float(self.env.now) - float(TRACK_B_ROLLING_WINDOW_HOURS)
        )
        recent_orders = [
            order for order in self.orders if float(order.OPTj) >= window_start
        ]
        if not recent_orders:
            return float(self._fill_rate()), float(self._backorder_rate())

        demanded_qty = float(sum(order.quantity for order in recent_orders))
        filled_qty = float(
            sum(
                max(0.0, float(order.quantity) - float(order.remaining_qty))
                for order in recent_orders
            )
        )
        fill_rate = min(1.0, filled_qty / max(demanded_qty, 1.0))
        delayed_or_lost = sum(
            1
            for order in recent_orders
            if order.lost
            or order.backorder
            or (
                order.remaining_qty > 0.0
                and (float(self.env.now) - float(order.OPTj)) > float(order.LTj)
            )
        )
        backorder_rate = min(1.0, delayed_or_lost / max(len(recent_orders), 1))
        return fill_rate, backorder_rate

    @staticmethod
    def _queue_pressure_norm(buffer_level: float, dispatch_q_max: float) -> float:
        """Normalize downstream queue pressure against several dispatch cycles."""
        norm_capacity = max(
            1.0,
            float(dispatch_q_max) * float(TRACK_B_QUEUE_PRESSURE_LOOKAHEAD_CYCLES),
        )
        return min(1.0, max(0.0, float(buffer_level)) / norm_capacity)

    def get_observation_v7_extra(self) -> np.ndarray:
        """
        Return 6 Track-B bottleneck features for obs v7.

        [0]    op10_down
        [1]    op12_down
        [2]    op10_queue_pressure_norm
        [3]    op12_queue_pressure_norm
        [4]    rolling_fill_rate_4w
        [5]    rolling_backorder_rate_4w
        """
        rolling_fill_rate, rolling_backorder_rate = self._rolling_service_metrics()
        return np.array(
            [
                float(self._is_down(10)),
                float(self._is_down(12)),
                self._queue_pressure_norm(
                    self.rations_sb_dispatch.level,
                    float(self.params["op10_q_max"]),
                ),
                self._queue_pressure_norm(
                    self.rations_cssu.level,
                    float(self.params["op12_q_max"]),
                ),
                float(np.clip(rolling_fill_rate, 0.0, 1.0)),
                float(np.clip(rolling_backorder_rate, 0.0, 1.0)),
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

    def _adaptive_regime_params(self) -> dict[str, float]:
        return ADAPTIVE_BENCHMARK_REGIME_PARAMS[self.adaptive_regime]

    def _adaptive_expected_intensity(self, regime: str) -> float:
        return float(
            ADAPTIVE_BENCHMARK_REGIME_PARAMS[regime]["risk_intensity"]
        ) / float(ADAPTIVE_BENCHMARK_REGIME_PARAMS["disrupted"]["risk_intensity"])

    def _adaptive_risk_intensity_for(self, risk_id: str) -> float:
        """Return adaptive risk intensity, with Track B v2 downstream uplift."""
        intensity = float(self._adaptive_regime_params()["risk_intensity"])
        if self.adaptive_benchmark_v2_enabled:
            intensity *= float(ADAPTIVE_BENCHMARK_V2_RISK_MULTIPLIERS.get(risk_id, 1.0))
        return intensity

    def _adaptive_recovery_scale_for(self, risk_id: str) -> float:
        """Return adaptive recovery scaling, with Track B v2 downstream uplift."""
        recovery_scale = float(self._adaptive_regime_params()["recovery_scale"])
        if self.adaptive_benchmark_v2_enabled:
            recovery_scale *= float(
                ADAPTIVE_BENCHMARK_V2_RECOVERY_MULTIPLIERS.get(risk_id, 1.0)
            )
        return recovery_scale

    def _adaptive_surge_scale(self) -> float:
        """Return adaptive surge scaling, with Track B v2 demand uplift."""
        surge_scale = float(self._adaptive_regime_params()["surge_scale"])
        if self.adaptive_benchmark_v2_enabled:
            surge_scale *= float(ADAPTIVE_BENCHMARK_V2_SURGE_SCALE_MULTIPLIER)
        return surge_scale

    def _update_adaptive_forecasts(self) -> None:
        """Refresh noisy but informative disruption forecasts for v6."""
        current_params = self._adaptive_regime_params()
        transitions = ADAPTIVE_BENCHMARK_TRANSITIONS[self.adaptive_regime]
        next_intensity = 0.0
        for next_regime, prob in transitions.items():
            next_intensity += prob * self._adaptive_expected_intensity(next_regime)
        current_intensity = self._adaptive_expected_intensity(self.adaptive_regime)
        noise_std = float(ADAPTIVE_BENCHMARK_MAINTENANCE["forecast_noise_std"])
        forecast_48h = next_intensity + float(self.rng.normal(0.0, noise_std))
        forecast_168h = (
            0.35 * current_intensity
            + 0.65 * next_intensity
            + 0.15 * float(current_params["forecast_base"])
            + float(self.rng.normal(0.0, noise_std))
        )
        self.adaptive_risk_forecast_48h = float(np.clip(forecast_48h, 0.0, 1.0))
        self.adaptive_risk_forecast_168h = float(np.clip(forecast_168h, 0.0, 1.0))

    def _adaptive_regime_controller(self):
        """Persistent regime process for the adaptive benchmark lane."""
        self._update_adaptive_forecasts()
        review_hours = float(ADAPTIVE_BENCHMARK_REVIEW_HOURS)
        while True:
            yield self.env.timeout(review_hours)
            transitions = ADAPTIVE_BENCHMARK_TRANSITIONS[self.adaptive_regime]
            next_regimes = tuple(transitions.keys())
            probs = np.array(tuple(transitions.values()), dtype=float)
            probs = probs / probs.sum()
            self.adaptive_regime = str(self.rng.choice(next_regimes, p=probs))
            self._update_adaptive_forecasts()

    def _pending_backorder_age_norm(self) -> float:
        if not self.pending_backorders:
            return 0.0
        avg_age = float(
            np.mean(
                [
                    max(0.0, self.env.now - order.OPTj)
                    for order in self.pending_backorders
                ]
            )
        )
        norm_hours = float(ADAPTIVE_BENCHMARK_MAINTENANCE["backlog_age_norm_hours"])
        return min(1.0, avg_age / max(norm_hours, 1.0))

    def _theatre_cover_days_norm(self) -> float:
        mean_daily_demand = 0.5 * float(DEMAND["a"] + DEMAND["b"])
        current_scale = (
            float(self._adaptive_regime_params()["demand_scale"])
            if self.adaptive_benchmark_enabled
            else 1.0
        )
        effective_daily_demand = max(1.0, mean_daily_demand * current_scale)
        cover_days = float(self.rations_theatre.level) / effective_daily_demand
        norm_days = float(ADAPTIVE_BENCHMARK_MAINTENANCE["theatre_cover_norm_days"])
        return min(1.0, cover_days / max(norm_days, 1.0))

    def _apply_maintenance_debt(self, shifts: int) -> None:
        if shifts >= 3:
            self.maintenance_debt = min(
                1.0,
                self.maintenance_debt
                + float(ADAPTIVE_BENCHMARK_MAINTENANCE["s3_debt_gain_per_hour"]),
            )
        elif shifts == 2:
            self.maintenance_debt = max(
                0.0,
                self.maintenance_debt
                - float(ADAPTIVE_BENCHMARK_MAINTENANCE["s2_debt_decay_per_hour"]),
            )
        else:
            self.maintenance_debt = max(
                0.0,
                self.maintenance_debt
                - float(ADAPTIVE_BENCHMARK_MAINTENANCE["s1_debt_decay_per_hour"]),
            )

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
            pt_remaining = self._pt("op1_pt")
            while pt_remaining > 0:
                while self._is_down(1):
                    yield self.env.timeout(1)
                yield self.env.timeout(1)
                pt_remaining -= 1

    def _op2_supplier_delivery(self):
        while True:
            yield self.env.timeout(self.params["op2_rop"])
            while self._is_down(2):
                yield self.env.timeout(1)
            pt_remaining = self._pt("op2_pt")
            while pt_remaining > 0:
                while self._is_down(2):
                    yield self.env.timeout(1)
                yield self.env.timeout(1)
                pt_remaining -= 1
            total_delivery = self.params["op2_q"] * NUM_RAW_MATERIALS
            if self.raw_material_flow_mode == "bom_total_units_order_up_to":
                target = self._target_for_raw_node("op3_rm", total_delivery)
                total_delivery = max(0.0, target - float(self.raw_material_wdc.level))
                if total_delivery <= 0.0:
                    continue
            yield self.raw_material_wdc.put(total_delivery)

    def _op3_wdc_dispatch(self):
        while True:
            yield self.env.timeout(self.params["op3_rop"])
            while self._is_down(3):
                yield self.env.timeout(1)
            total_dispatch = self.params["op3_q"] * NUM_RAW_MATERIALS
            available = self.raw_material_wdc.level
            if self.raw_material_flow_mode == "bom_total_units_order_up_to":
                target = self._target_for_raw_node("op5_rm", total_dispatch)
                total_dispatch = max(0.0, target - float(self.raw_material_al.level))
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

            self._cumulative_available_assembly_hours += 1.0
            if self.adaptive_benchmark_enabled:
                self._apply_maintenance_debt(int(shifts))

            # Produce. Thesis-strict R14 rework returns defects to Op6, so
            # rework consumes assembly capacity before new raw material.
            rework_available = self.rework_op6.level
            rm_available = self.raw_material_al.level
            effective_rate = RATIONS_PER_HOUR
            if self.adaptive_benchmark_enabled:
                penalty = float(
                    ADAPTIVE_BENCHMARK_MAINTENANCE["throughput_penalty_max"]
                )
                effective_rate *= max(0.0, 1.0 - penalty * self.maintenance_debt)
            rework_qty = min(effective_rate, rework_available)
            raw_capacity = max(0.0, effective_rate - rework_qty)
            raw_ration_capacity = rm_available / max(self._raw_units_per_ration, 1.0)
            raw_produced_qty = min(raw_capacity, raw_ration_capacity)
            raw_units_qty = raw_produced_qty * self._raw_units_per_ration
            can_produce = rework_qty + raw_produced_qty

            if can_produce > 0:
                if rework_qty > 0:
                    yield self.rework_op6.get(rework_qty)
                if raw_units_qty > 0:
                    yield self.raw_material_al.get(raw_units_qty)
                self._pending_batch += can_produce
                self._today_produced += can_produce
                self.total_produced += can_produce

                # Ship complete batches (read batch_size live for shift changes)
                batch_size = self.params["batch_size"]
                while self._pending_batch >= batch_size:
                    self._pending_batch -= batch_size
                    yield self.rations_al.put(batch_size)

                if (
                    self.warmup_trigger == "production"
                    and self.total_produced >= batch_size
                ):
                    self._mark_warmup_complete()

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
            if self.warmup_trigger == "op9_arrival":
                self._mark_warmup_complete()

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
            q_min = int(round(float(self.params["op10_q_min"])))
            q_max = int(round(float(self.params["op10_q_max"])))
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
            q_min = int(round(float(self.params["op12_q_min"])))
            q_max = int(round(float(self.params["op12_q_max"])))
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
            if self.adaptive_benchmark_enabled:
                demand_scale = float(self._adaptive_regime_params()["demand_scale"])
                demand_qty *= demand_scale
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
                self._set_order_ret_indicators(order)
            else:
                # Enqueue for future delivery. Per thesis Sec. 6.8.2,
                # backorder classification deferred until LTj=48h elapses.
                self._enqueue_backorder(order)
                yield from self._serve_pending_backorders()
                # Start 48h timer — only count as backorder if still pending
                self.env.process(self._delayed_backorder_check(order))

            self.orders.append(order)
            self.daily_demand.append((self.env.now, demand_qty))

    # =====================================================================
    # RISK PROCESSES
    # =====================================================================

    _RISK_TABLES = {
        "increased": RISKS_INCREASED,
        "severe": RISKS_SEVERE,
        "severe_extended": RISKS_SEVERE_EXTENDED,
        "severe_training": RISKS_SEVERE_TRAINING,
    }

    def _risk_table_for(self, risk_id: str) -> dict[str, Any] | None:
        level = self.risk_overrides.get(risk_id, self.risk_level)
        return self._RISK_TABLES.get(level)

    def _get_risk_b(self, risk_id: str) -> float:
        table = self._risk_table_for(risk_id)
        base_a = RISKS_CURRENT[risk_id]["occurrence"]["a"]
        if table and risk_id in table:
            base_b = table[risk_id].get("b", RISKS_CURRENT[risk_id]["occurrence"]["b"])
        else:
            base_b = RISKS_CURRENT[risk_id]["occurrence"]["b"]
        if self.adaptive_benchmark_enabled:
            intensity = self._adaptive_risk_intensity_for(risk_id)
            return max(float(base_a), round(float(base_b) / max(intensity, 1e-6)))
        return float(base_b)

    def _get_risk_p(self, risk_id: str) -> float:
        table = self._risk_table_for(risk_id)
        if table and risk_id in table:
            base_p = table[risk_id].get("p", RISKS_CURRENT[risk_id]["occurrence"]["p"])
        else:
            base_p = RISKS_CURRENT[risk_id]["occurrence"]["p"]
        if self.adaptive_benchmark_enabled:
            intensity = self._adaptive_risk_intensity_for(risk_id)
            return min(0.98, float(base_p) * intensity)
        return float(base_p)

    def _get_risk_recovery_mean(self, risk_id: str) -> float:
        table = self._risk_table_for(risk_id)
        if table and risk_id in table:
            base_mean = table[risk_id].get(
                "recovery_mean", RISKS_CURRENT[risk_id]["recovery"]["mean"]
            )
        else:
            base_mean = RISKS_CURRENT[risk_id]["recovery"]["mean"]
        if self.adaptive_benchmark_enabled:
            recovery_scale = self._adaptive_recovery_scale_for(risk_id)
            return float(base_mean) * recovery_scale
        return float(base_mean)

    def _get_risk_surge(self) -> tuple[int, int]:
        table = self._risk_table_for("R24")
        base_lo = RISKS_CURRENT["R24"]["surge"]["lo"]
        base_hi = RISKS_CURRENT["R24"]["surge"]["hi"]
        if table and "R24" in table:
            base_lo = table["R24"].get("surge_lo", base_lo)
            base_hi = table["R24"].get("surge_hi", base_hi)
        if self.adaptive_benchmark_enabled:
            surge_scale = self._adaptive_surge_scale()
            return (
                int(round(base_lo * surge_scale)),
                int(round(base_hi * surge_scale)),
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
        """R14 defective products — Binomial(n, p) per day (Table 6.6b).

        n = actual daily production (shift-adjusted): S=1→2564, S=2→5128,
        S=3→7692. Config n=2564 is the S=1 reference value from the thesis.
        """
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
                        if self.r14_defect_mode == "thesis_strict_op6":
                            yield self.rework_op6.put(defects)
                            description = f"{defects} defective (returned to Op6)"
                        elif self.r14_defect_mode == "reprocess":
                            # Thesis Table 6.6b: defects returned to Op6 for
                            # re-processing. Model by feeding back to raw material
                            # so they re-enter the assembly pipeline later.
                            yield self.raw_material_al.put(defects)
                            description = (
                                f"{defects} defective (returned to raw_material_al)"
                            )
                        else:
                            description = f"{defects} defective (discarded)"
                        self.risk_events.append(
                            RiskEvent(
                                "R14",
                                self.env.now,
                                self.env.now,
                                0,
                                [7],
                                description,
                            )
                        )

    def _risk_R21(self):
        """R21 natural disaster generator — non-blocking mode.

        Each event takes down all affected operations simultaneously; each
        operation recovers independently with Exp(beta) hours. The generator
        spawns a new process for each event so they can theoretically overlap.
        """
        a = RISKS_CURRENT["R21"]["occurrence"]["a"]
        b_val = self._get_risk_b("R21")
        beta = self._get_risk_recovery_mean("R21")
        affected = RISKS_CURRENT["R21"]["affected_ops"]
        while True:
            yield self.env.timeout(self.rng.integers(a, b_val + 1))
            self.env.process(self._r21_event(affected, beta))

    def _r21_event(self, affected: list[int], beta: float):
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
        beta = self._get_risk_recovery_mean("R22")
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
        beta = self._get_risk_recovery_mean("R23")
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
            # Cap accumulated contingent demand to prevent unbounded obs[14]
            # spikes when multiple R24 events fire before demand is consumed.
            # 5×2600 = 13000 ≈ 5 regular demand cycles, well above any
            # realistic surge accumulation.
            max_contingent = 5 * 2600
            self._contingent_demand_pending = min(
                self._contingent_demand_pending, max_contingent
            )
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
        print(f"\n{'=' * 60}")
        print("  Risk Event Summary")
        print(f"{'=' * 60}")
        print(f"  {'Risk':<8} {'Count':>8} {'Total Down':>12} {'Avg':>10}")
        print(f"  {'-' * 40}")
        for rid in sorted(counts.keys()):
            c = counts[rid]
            dt = total_dt[rid]
            print(f"  {rid:<8} {c:>8} {dt:>11,.0f}h {dt / c:>9,.1f}h")
        print(f"\n  Total: {len(self.risk_events)} events")
        print(f"{'=' * 60}\n")

    def summary(self) -> None:
        years = self.horizon / self.hours_per_year
        mode = f"ENABLED ({self.risk_level})" if self.risks_enabled else "DISABLED"
        print(f"\n{'=' * 60}")
        print("  MFSC Simulation Summary")
        print(f"  Horizon: {self.horizon:,} hrs ({years:.1f} years)")
        print(f"  Shifts: S={self.shifts}  |  Risks: {mode}")
        print(
            f"  Granularity: HOURLY (assembly)  |  Seed: {self.seed}  |"
            f"  Year basis: {self.year_basis} ({self.hours_per_year:,}h)"
        )
        print(f"{'=' * 60}")
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
        print(f"{'=' * 60}")
        if self.risks_enabled:
            self.risk_summary()

    def _fill_rate(self):
        """Current fill rate per Garrido's order-based Bt + Ut formulation."""
        return max(0.0, 1.0 - self._backorder_rate())

    def _order_level_fill_rate(self) -> float:
        """
        Thesis-exact fill rate (Garrido-Rios 2017, Equation 5.4):

            Re(FRt) = 1 - (Bt + Ut) / Dt

        where Bt = cumulative backorder *order count*, Ut = unattended order count,
        and Dt = total orders demanded.

        This uses order counts (not ration quantities), matching the thesis
        formulation exactly. See also _fill_rate() which is equivalent for
        the current RL reward computation.
        """
        total_orders = len(self.orders)
        if total_orders == 0:
            return 1.0
        bt_count = len(self.pending_backorders)
        ut_count = self.total_unattended_orders
        return max(0.0, 1.0 - (bt_count + ut_count) / total_orders)

    def _set_order_ret_indicators(self, order: OrderRecord) -> None:
        """
        Populate ReT sub-indicators APj, RPj, DPj for a completed order.

        Per Garrido-Rios (2017) Eq. 5.1-5.5:
        - Autotomy (CTj <= LTj, disruptions present): APj = disruption overlap hours
        - Recovery (CTj > LTj, disruptions present): RPj = OATj - earliest_risk_start
        - Non-recovery (CTj > LTj, no recovery): DPj set, RPj=0
        - Fill rate (no disruptions): APj=RPj=DPj=0
        """
        if order.OATj is None or order.CTj is None:
            return

        # Compute disruption overlap during [OPTj, OATj]
        total_disruption_hours = 0.0
        earliest_risk_start = float("inf")

        for event in self.risk_events:
            overlap_start = max(event.start_time, order.OPTj)
            overlap_end = min(event.end_time, order.OATj)
            if overlap_start < overlap_end:
                total_disruption_hours += overlap_end - overlap_start
                earliest_risk_start = min(earliest_risk_start, event.start_time)

        # Include ongoing disruptions at fulfillment time
        for op_id in range(1, 14):
            down_since = self._op_down_since.get(op_id)
            if self.op_down_count[op_id] > 0 and down_since is not None:
                overlap_start = max(down_since, order.OPTj)
                overlap_end = order.OATj
                if overlap_start < overlap_end:
                    total_disruption_hours += overlap_end - overlap_start
                    earliest_risk_start = min(earliest_risk_start, down_since)

        if total_disruption_hours <= 0:
            return  # No disruption: fill_rate case

        if order.CTj <= order.LTj:
            # Autotomy: SC absorbed disruption, order still on time
            order.APj = min(total_disruption_hours, order.LTj)
        else:
            # Recovery / Non-recovery: order delayed beyond lead time
            order.DPj = order.CTj
            eff_risk_start = max(earliest_risk_start, order.OPTj)
            order.RPj = max(0.0, order.OATj - eff_risk_start)

    def _order_ret_value(self, order: OrderRecord) -> tuple[float, str]:
        """
        Compute per-order ReT value per Garrido Eq. 5.1-5.5.

        Returns (ret_value, case_label).
        Uses thesis constants: Re^max=1.0, Re=0.5 (Figure 5.6), Re^min=0.0.
        """
        return compute_ret_per_order(
            order,
            fill_rate=self._order_level_fill_rate(),
            ret_weights=THESIS_FAITHFUL_PROTOCOL["ret_weights"],
        )

    def compute_order_level_ret(self) -> dict[str, Any]:
        """
        Compute thesis-exact order-level ReT per Garrido Eq. 5.1-5.5.

        Returns dict with aggregate ReT metrics:
        - mean_ret: average ReT across all completed orders
        - fill_rate_order_level: Re(FRt) = 1 - (Bt+Ut)/Dt (order counts, Eq. 5.4)
        - case_counts: orders per case (autotomy, recovery, non_recovery, fill_rate)
        - n_orders, n_completed: total and completed order counts
        """
        return compute_thesis_order_level_ret(
            self.orders,
            fill_rate=self._order_level_fill_rate(),
            ret_weights=THESIS_FAITHFUL_PROTOCOL["ret_weights"],
        )
