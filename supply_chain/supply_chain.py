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
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional
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
    DEMAND_SOURCE_OPTIONS,
    ASSEMBLY_RATE,
    BACKORDER_QUEUE_CAP,
    BACKORDER_OVERFLOW_MODE,
    BACKORDER_OVERFLOW_MODE_OPTIONS,
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
    GARRIDO_FULFILLMENT_DELAY_HOURS,
    GARRIDO_R14_RET_PERIOD_HOURS,
    HOURS_PER_YEAR_GREGORIAN,
    HOURS_PER_YEAR_THESIS,
    R14_DEFECT_MODE_OPTIONS,
    RATIONS_PER_SHIFT,
    RAW_MATERIAL_FLOW_MODE_OPTIONS,
    RET_RECOVERY_PERIOD_MODE,
    RET_RECOVERY_PERIOD_MODE_OPTIONS,
    RISK_ATTRIBUTION_SOURCE_OPTIONS,
    RISK_OCCURRENCE_MODE_OPTIONS,
    SEED_STREAM_MODE_OPTIONS,
    canonical_raw_material_flow_mode,
    YEAR_BASIS_OPTIONS,
    LEAD_TIME_PROMISE,
    THESIS_FAITHFUL_PROTOCOL,
    THESIS_DOWNSTREAM_Q_RANGES,
    THESIS_REPLICATION_DOWNSTREAM_Q_SOURCE,
    WARMUP_TRIGGER_OPTIONS,
)
from .ret_thesis import (
    compute_fill_rate_from_orders,
    compute_order_level_ret_excel_formula,
    compute_order_level_ret as compute_thesis_order_level_ret,
    compute_ret_per_order,
)

RATIONS_PER_HOUR = ASSEMBLY_RATE  # 320.5 rations/hr
REALIZED_RISK_OBSERVATION_IDS = (
    "R11",
    "R12",
    "R13",
    "R14",
    "R21",
    "R22",
    "R23",
    "R24",
    "R3",
)


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
    metrics_excluded: bool = False
    # Thesis ReT sub-indicators (Garrido-Rios 2017, Eq. 5.1-5.5):
    APj: float = 0.0  # Autotomy period (hours): CTj=LTj and risks impact in [OPTj,OATj]
    RPj: float = 0.0  # Recovery period (hours): OATj - first R0cr detection
    DPj: float = 0.0  # Disruption period (hours): CTj when CTj > LTj
    ret_risk_indicators: dict[str, float] = field(default_factory=dict)
    ret_risk_event_refs: list[dict[str, Any]] = field(default_factory=list)
    ret_attribution_override: Optional[dict[str, Any]] = None


@dataclass
class RiskEvent:
    risk_id: str
    start_time: float
    end_time: float
    duration: float
    affected_ops: list
    description: str = ""
    magnitude: float = 1.0
    unit: str = "incidents"


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
        warmup_trigger: str = "op9_arrival",
        downstream_q_source: str = THESIS_REPLICATION_DOWNSTREAM_Q_SOURCE,
        r14_defect_mode: str = "thesis_strict_op6",
        enabled_risks: Optional[set[str]] = None,
        risk_overrides: Optional[dict[str, str]] = None,
        risk_occurrence_mode: str = "thesis_window",
        risk_attribution_source: str = "des_events",
        inventory_replenishment_period: Optional[float] = None,
        inventory_replenishment_lead_time: float = 0.0,
        raw_material_flow_mode: str = "kit_equivalent_order_up_to",
        raw_material_order_up_to_multiplier: float = 2.0,
        demand_mean_multiplier: float = 1.0,
        demand_source: str = "thesis_calendar",
        excel_order_tape: Optional[list[dict[str, Any]]] = None,
        demand_on_hand_fulfillment_delay: float = GARRIDO_FULFILLMENT_DELAY_HOURS,
        seed_stream_mode: Optional[str] = None,
        ret_recovery_period_mode: str = RET_RECOVERY_PERIOD_MODE,
        backorder_overflow_mode: str = BACKORDER_OVERFLOW_MODE,
        risk_frequency_multiplier: float = 1.0,
        risk_impact_multiplier: float = 1.0,
        risk_frequency_multipliers_by_id: Optional[dict[str, float]] = None,
        risk_impact_multipliers_by_id: Optional[dict[str, float]] = None,
        risk_event_tape: Optional[Iterable[dict[str, Any]]] = None,
        strict_exogenous_crn: bool = False,
        risk_recovery_window_hours: float = 0.0,
        risk_recovery_release_rations: float = 0.0,
        risk_recovery_boost_downstream: bool = True,
        risk_recovery_enabled_risks: tuple[str, ...] = (
            "R21",
            "R22",
            "R23",
            "R24",
        ),
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
        if risk_occurrence_mode not in RISK_OCCURRENCE_MODE_OPTIONS:
            valid = ", ".join(RISK_OCCURRENCE_MODE_OPTIONS)
            raise ValueError(
                "Invalid risk_occurrence_mode="
                f"{risk_occurrence_mode!r}. Expected one of: {valid}."
            )
        if risk_attribution_source not in RISK_ATTRIBUTION_SOURCE_OPTIONS:
            valid = ", ".join(RISK_ATTRIBUTION_SOURCE_OPTIONS)
            raise ValueError(
                "Invalid risk_attribution_source="
                f"{risk_attribution_source!r}. Expected one of: {valid}."
            )
        if demand_source not in DEMAND_SOURCE_OPTIONS:
            valid = ", ".join(DEMAND_SOURCE_OPTIONS)
            raise ValueError(
                f"Invalid demand_source={demand_source!r}. Expected one of: {valid}."
            )
        if seed_stream_mode is None:
            seed_stream_mode = "split" if strict_exogenous_crn else "single"
        elif strict_exogenous_crn and seed_stream_mode != "split":
            raise ValueError(
                "strict_exogenous_crn=True is equivalent to seed_stream_mode='split'."
            )
        if seed_stream_mode not in SEED_STREAM_MODE_OPTIONS:
            valid = ", ".join(SEED_STREAM_MODE_OPTIONS)
            raise ValueError(
                "Invalid seed_stream_mode="
                f"{seed_stream_mode!r}. Expected one of: {valid}."
            )
        if ret_recovery_period_mode not in RET_RECOVERY_PERIOD_MODE_OPTIONS:
            valid = ", ".join(RET_RECOVERY_PERIOD_MODE_OPTIONS)
            raise ValueError(
                "Invalid ret_recovery_period_mode="
                f"{ret_recovery_period_mode!r}. Expected one of: {valid}."
            )
        if backorder_overflow_mode not in BACKORDER_OVERFLOW_MODE_OPTIONS:
            valid = ", ".join(BACKORDER_OVERFLOW_MODE_OPTIONS)
            raise ValueError(
                "Invalid backorder_overflow_mode="
                f"{backorder_overflow_mode!r}. Expected one of: {valid}."
            )
        raw_material_flow_mode = canonical_raw_material_flow_mode(raw_material_flow_mode)
        self.env = simpy.Environment()
        self.shifts = shifts
        self.seed = seed
        self.seed_stream_mode = seed_stream_mode
        self.strict_exogenous_crn = seed_stream_mode == "split"
        if self.strict_exogenous_crn:
            general_ss, demand_ss, risk_ss, regime_ss = np.random.SeedSequence(seed).spawn(4)
            self.rng = np.random.default_rng(general_ss)
            self.demand_rng = np.random.default_rng(demand_ss)
            self.risk_rng = np.random.default_rng(risk_ss)
            self.regime_rng = np.random.default_rng(regime_ss)
        else:
            self.rng = np.random.default_rng(seed)
            self.demand_rng = self.rng
            self.risk_rng = self.rng
            self.regime_rng = self.rng
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
        self.demand_source = demand_source
        self.excel_order_tape = self._normalize_excel_order_tape(excel_order_tape)
        if self.demand_source == "excel_order_tape" and not self.excel_order_tape:
            raise ValueError("demand_source='excel_order_tape' requires excel_order_tape.")
        self.demand_on_hand_fulfillment_delay = max(
            0.0, float(demand_on_hand_fulfillment_delay)
        )
        if self.raw_material_order_up_to_multiplier <= 0.0:
            raise ValueError("raw_material_order_up_to_multiplier must be > 0.")
        # Operational-tempo demand lever for `learning_extension_v1` (see
        # docs/PAPER_CONTRACT_2026-06-24.md). 1.0 = thesis baseline; >1 raises the
        # regular daily-demand mean. Independent of the adaptive_benchmark machinery.
        self.demand_mean_multiplier = float(demand_mean_multiplier)
        # Garrido-authorized risk modulation (fine-tuning, frozen per calibrated regime):
        # phi scales FREQUENCY (smaller window b, larger binomial p); psi scales IMPACT
        # (longer recovery, bigger demand surge). 1.0 = thesis baseline.
        self.risk_frequency_multiplier = max(1e-6, float(risk_frequency_multiplier))
        self.risk_impact_multiplier = max(1e-6, float(risk_impact_multiplier))
        self.risk_frequency_multipliers_by_id = {
            str(risk_id): max(1e-6, float(value))
            for risk_id, value in (risk_frequency_multipliers_by_id or {}).items()
        }
        self.risk_impact_multipliers_by_id = {
            str(risk_id): max(1e-6, float(value))
            for risk_id, value in (risk_impact_multipliers_by_id or {}).items()
        }
        self.risk_event_tape = self._normalize_risk_event_tape(risk_event_tape)
        if self.demand_mean_multiplier <= 0.0:
            raise ValueError("demand_mean_multiplier must be > 0.")
        self._raw_units_per_ration = (
            float(NUM_RAW_MATERIALS)
            if raw_material_flow_mode.startswith("bom_total_units")
            else 1.0
        )
        self.enabled_risks = set(enabled_risks) if enabled_risks is not None else None
        self.risk_overrides = dict(risk_overrides or {})
        self.risk_occurrence_mode = risk_occurrence_mode
        self.risk_attribution_source = risk_attribution_source
        self.ret_recovery_period_mode = ret_recovery_period_mode
        self.backorder_overflow_mode = backorder_overflow_mode
        if self.risk_attribution_source == "excel_risk_tape":
            if self.demand_source != "excel_order_tape":
                raise ValueError(
                    "risk_attribution_source='excel_risk_tape' requires "
                    "demand_source='excel_order_tape'."
                )
            missing = [
                int(row["j"])
                for row in self.excel_order_tape
                if row.get("ret_attribution") is None
            ]
            if missing:
                raise ValueError(
                    "risk_attribution_source='excel_risk_tape' requires "
                    f"ret_attribution on every tape row; missing j={missing[:5]}."
                )
        self.inventory_replenishment_period = inventory_replenishment_period
        # Lead time to rebuild the strategic buffer (Ed.2 for inventory): the refill
        # arrives this many hours after it is triggered, so a sustained disruption can
        # drain the buffer faster than it refills -> anticipating the regime pays.
        self.inventory_replenishment_lead_time = max(0.0, float(inventory_replenishment_lead_time))
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
        # v9 observation: backorder health, EWMA trends, and step throughput.
        self._prev_step_produced: float = 0.0
        self._prev_step_delivered: float = 0.0
        self._prev_step_available_assembly_hours: float = 0.0
        self._prev_step_fill_rate: float = 1.0
        self._ewma_fill_rate: float = 1.0
        self._ewma_backlog_growth: float = 0.0
        self._delta_fill_rate: float = 0.0
        self._delta_backlog_momentum: float = 0.0
        self._prev_pending_backorder_qty: float = 0.0

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
        self._ret_quantity_risk_units = {"R14": 0.0, "R24": 0.0}
        self._ret_quantity_risk_refs: dict[str, list[dict[str, Any]]] = {
            "R14": [],
            "R24": [],
        }
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

        # =================================================================
        # R2 RECOVERY WINDOW (opt-in sensitivity, not freeze-default)
        # =================================================================
        # Bounded stock-release after R2 events. When `risk_recovery_window_hours`
        # and `risk_recovery_release_rations` are both > 0, a simpy controller
        # fires for each R2 event in `risk_recovery_enabled_risks`: at the end
        # of the event + window, it injects `release_rations` to theatre and
        # drains the pending backorder queue while the window is open.
        # This is the diagnostic mechanism identified by
        # `docs/R2_AUDIT_DECOMPOSITION_2026-06-29.md` §5. It is NOT a freeze
        # default; the thesis-faithful lane keeps these at 0.
        self.risk_recovery_window_hours = max(0.0, float(risk_recovery_window_hours))
        self.risk_recovery_release_rations = max(
            0.0, float(risk_recovery_release_rations)
        )
        self.risk_recovery_boost_downstream = bool(risk_recovery_boost_downstream)
        self.risk_recovery_enabled_risks = tuple(
            str(r) for r in risk_recovery_enabled_risks
        )
        self._risk_recovery_seen: set[tuple[str, float, float]] = set()
        self._risk_recovery_window_until: float = 0.0
        self._risk_recovery_release_emitted: float = 0.0
        self._risk_recovery_base_params: dict[str, Any] = {}
        self._risk_recovery_boosted: bool = False
        if (
            self.risk_recovery_window_hours > 0.0
            and self.risk_recovery_release_rations > 0.0
            and self.risk_recovery_boost_downstream
        ):
            for key in (
                "op9_rop",
                "op10_rop",
                "op12_rop",
                "op9_q_min",
                "op9_q_max",
                "op10_q_min",
                "op10_q_max",
                "op12_q_min",
                "op12_q_max",
            ):
                self._risk_recovery_base_params[key] = self.params[key]

    def _normalize_excel_order_tape(
        self, tape: Optional[list[dict[str, Any]]]
    ) -> list[dict[str, Any]]:
        if not tape:
            return []
        normalized: list[dict[str, Any]] = []
        for index, row in enumerate(tape, start=1):
            if "OPTj" in row:
                optj = row["OPTj"]
            else:
                optj = row.get("optj")
            if "Q" in row:
                quantity = row["Q"]
            else:
                quantity = row.get("q", row.get("quantity"))
            if optj is None or quantity is None:
                raise ValueError("Excel order tape rows require OPTj and Q.")
            j_value = row.get("j", index)
            quantity_float = float(quantity)
            normalized_row = {
                "j": int(j_value),
                "OPTj": float(optj),
                "Q": quantity_float,
                "contingent": bool(
                    row.get("contingent", quantity_float > float(DEMAND["b"]))
                ),
            }
            attribution = row.get("ret_attribution")
            if attribution is None and any(
                key in row for key in ("APj", "apj", "RPj", "rpj", "DPj", "dpj", "risk_values")
            ):
                attribution = {
                    "APj": row.get("APj", row.get("apj", 0.0)),
                    "RPj": row.get("RPj", row.get("rpj", 0.0)),
                    "DPj": row.get("DPj", row.get("dpj", 0.0)),
                    "LTj": row.get("LTj", row.get("LT", row.get("ltj", LEAD_TIME_PROMISE))),
                    "risk_values": row.get("risk_values", {}),
                }
            if attribution is not None:
                risk_values = dict(attribution.get("risk_values", {}) or {})
                normalized_row["ret_attribution"] = {
                    "APj": float(attribution.get("APj", attribution.get("apj", 0.0)) or 0.0),
                    "RPj": float(attribution.get("RPj", attribution.get("rpj", 0.0)) or 0.0),
                    "DPj": float(attribution.get("DPj", attribution.get("dpj", 0.0)) or 0.0),
                    "LTj": float(
                        attribution.get("LTj", attribution.get("LT", LEAD_TIME_PROMISE))
                        or LEAD_TIME_PROMISE
                    ),
                    "risk_values": {
                        str(key): float(value or 0.0)
                        for key, value in risk_values.items()
                    },
                }
            normalized.append(normalized_row)
        return sorted(normalized, key=lambda item: (float(item["OPTj"]), int(item["j"])))

    def _add_ret_quantity_risk(self, event: RiskEvent) -> None:
        """Track quantity-style risks for order-level Excel-ReT attribution.

        R14 and R24 are not workstation downtime intervals.  The raw Garrido
        workbooks nevertheless expose them as per-order risk indicators, often
        across orders placed after the point event.  Keep a separate attribution
        queue so affected quantities can be consumed by subsequent orders
        without changing the physical DES flow.
        """
        risk_id = str(event.risk_id)
        if risk_id not in self._ret_quantity_risk_units:
            return
        magnitude = max(0.0, float(event.magnitude or 0.0))
        if magnitude <= 0.0:
            return
        # The workbook risk columns are used as order-level gates/counts, not
        # ration-quantity terms in the ReT formula.  One R14 defect event can
        # therefore mark a later order even if it contains many defective units.
        # R24 retains approximate order-scale magnitude because its source is a
        # surge of rations rather than a count of defective pieces.
        if risk_id == "R14":
            indicator_units = 1.0
        else:
            indicator_units = max(1.0, magnitude / max(float(DEMAND["b"]), 1.0))
        cap = 10000.0
        self._ret_quantity_risk_units[risk_id] = min(
            cap, self._ret_quantity_risk_units[risk_id] + indicator_units
        )
        self._ret_quantity_risk_refs[risk_id].append(
            {
                "risk_id": risk_id,
                "start_time": float(event.start_time),
                "end_time": float(event.end_time),
                "duration": float(event.duration),
                "affected_ops": list(event.affected_ops),
                "magnitude": indicator_units,
                "unit": str(event.unit),
                "raw_magnitude": magnitude,
            }
        )
        # Avoid a stale reference list when a long run has many daily R14 events.
        self._ret_quantity_risk_refs[risk_id] = self._ret_quantity_risk_refs[risk_id][-25:]

    def _consume_ret_quantity_risk_for_order(
        self, order: OrderRecord
    ) -> tuple[float, float | None]:
        """Apply queued quantity-risk attribution to an order.

        Returns ``(period_contribution, earliest_start)`` for the AP/RP/DP
        period logic.  The contribution is deliberately a small indicator-like
        value because the Excel columns behave as risk gates, not as ration
        quantities in the ReT formula.
        """
        contribution = 0.0
        earliest: float | None = None
        for risk_id in ("R14", "R24"):
            available = float(self._ret_quantity_risk_units.get(risk_id, 0.0))
            if available <= 0.0:
                continue
            consumed = min(available, 1.0)
            if risk_id == "R14":
                # R14 behaves like a persistent quality-risk gate in the raw
                # Excel sheets: once defective-product risk appears, later
                # orders continue to carry the R14 indicator rather than
                # exhausting a ration-sized quantity bucket.
                self._ret_quantity_risk_units[risk_id] = available
            else:
                self._ret_quantity_risk_units[risk_id] = max(0.0, available - consumed)
            refs = self._ret_quantity_risk_refs.get(risk_id, [])
            ref_start = (
                min(float(ref["start_time"]) for ref in refs)
                if refs
                else float(order.OPTj)
            )
            earliest = ref_start if earliest is None else min(earliest, ref_start)
            order.ret_risk_indicators[risk_id] = max(
                1.0, order.ret_risk_indicators.get(risk_id, 0.0)
            )
            order.ret_risk_event_refs.append(
                {
                    "risk_id": risk_id,
                    "start_time": ref_start,
                    "end_time": float(order.OATj if order.OATj is not None else order.OPTj),
                    "duration": (
                        float(GARRIDO_R14_RET_PERIOD_HOURS)
                        if risk_id == "R14"
                        else 0.0
                    ),
                    "affected_ops": [7] if risk_id == "R14" else [13],
                    "magnitude": consumed,
                    "unit": "quantity_risk_attribution",
                }
            )
            contribution += (
                float(GARRIDO_R14_RET_PERIOD_HOURS) if risk_id == "R14" else 1.0
            )
        return contribution, earliest

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
            lead = float(self.inventory_replenishment_lead_time)
            if lead > 0.0:
                # Refill arrives `lead` hours later, without shifting the period clock.
                self.env.process(self._delayed_buffer_top_up(lead))
            else:
                for key, target in self.inventory_buffer_targets.items():
                    event = self._top_up_inventory_buffer(key, float(target))
                    if event is not None:
                        yield event

    def _delayed_buffer_top_up(self, lead: float):
        """Strategic-buffer refill that arrives after a rebuild lead time (Ed.2)."""
        yield self.env.timeout(lead)
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

        if self.risks_enabled and self.risk_event_tape is not None:
            self.env.process(self._risk_event_tape_replay())
        elif self.risks_enabled:
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
        if self.adaptive_benchmark_enabled and self.risk_event_tape is None:
            self.env.process(self._adaptive_regime_controller())

        self._processes_started = True

        if (
            self.risk_recovery_window_hours > 0.0
            and self.risk_recovery_release_rations > 0.0
        ):
            self.env.process(self._r2_recovery_window_controller())

    def _normalize_risk_event_tape(
        self, risk_event_tape: Optional[Iterable[dict[str, Any]]]
    ) -> list[RiskEvent] | None:
        if risk_event_tape is None:
            return None
        normalized: list[RiskEvent] = []
        for row in risk_event_tape:
            if isinstance(row, RiskEvent):
                event = row
            else:
                affected = row.get("affected_ops", [])
                event = RiskEvent(
                    risk_id=str(row["risk_id"]),
                    start_time=float(row["start_time"]),
                    end_time=float(row.get("end_time", row["start_time"])),
                    duration=float(row.get("duration", 0.0)),
                    affected_ops=[int(op) for op in affected],
                    description=str(row.get("description", "") or ""),
                    magnitude=float(row.get("magnitude", 1.0) or 1.0),
                    unit=str(row.get("unit", "incidents") or "incidents"),
                )
            normalized.append(event)
        return sorted(normalized, key=lambda ev: (float(ev.start_time), str(ev.risk_id)))

    def _risk_event_tape_replay(self):
        """Replay a pre-serialized risk calendar for counterfactual audits.

        This is intentionally opt-in.  It does not alter the default thesis or
        Track B risk generators.  The replay covers the physical semantics used
        by the current prevention-gate work: R22/R23 duration outages and R24
        point demand surges.  Other event IDs are replayed conservatively as
        recorded down/up intervals where affected operations are present.
        """
        for event in self.risk_event_tape or []:
            start_time = max(0.0, float(event.start_time))
            if start_time > self.env.now:
                yield self.env.timeout(start_time - self.env.now)
            self.env.process(self._risk_event_tape_event(event))

    def _risk_event_tape_event(self, event: RiskEvent):
        risk_id = str(event.risk_id)
        start = float(self.env.now)
        duration = max(0.0, float(event.duration))
        affected_ops = [int(op) for op in event.affected_ops]

        if risk_id == "R24":
            surge = float(event.magnitude)
            self._contingent_demand_pending += surge
            self._contingent_demand_pending = min(
                self._contingent_demand_pending, 5 * 2600
            )
            replayed = RiskEvent(
                risk_id,
                start,
                start,
                0.0,
                affected_ops or [13],
                event.description,
                magnitude=surge,
                unit=event.unit or "rations",
            )
            self.risk_events.append(replayed)
            self._add_ret_quantity_risk(replayed)
            return

        if duration > 0.0 and affected_ops:
            for op_id in affected_ops:
                self._take_down(op_id)
            yield self.env.timeout(duration)
            for op_id in affected_ops:
                self._bring_up(op_id)
            end = float(self.env.now)
            self.risk_events.append(
                RiskEvent(
                    risk_id,
                    start,
                    end,
                    end - start,
                    affected_ops,
                    event.description,
                    magnitude=float(event.magnitude),
                    unit=event.unit,
                )
            )
            return

        replayed = RiskEvent(
            risk_id,
            start,
            start,
            0.0,
            affected_ops,
            event.description,
            magnitude=float(event.magnitude),
            unit=event.unit,
        )
        self.risk_events.append(replayed)
        if risk_id in {"R14", "R24"}:
            self._add_ret_quantity_risk(replayed)

    def _r2_recovery_window_controller(self):
        """
        Bounded stock-release after R2 events (opt-in sensitivity).

        Mirrors the diagnostic `_r2_recovery_window_controller` in
        `scripts/audit_garrido_r2_recovery_transient.py`, but reads its
        configuration from the sim attributes set in `__init__`. The freeze
        keeps `risk_recovery_window_hours=0` and
        `risk_recovery_release_rations=0` so this controller never runs in the
        default thesis-faithful lane.

        Mechanism
        ---------
        For each R2 event whose end time has not yet been processed, when
        `env.now >= event.end_time`, the controller:
          1. injects `risk_recovery_release_rations` to `rations_theatre` (NOT
             stock-conserving: the audit's "release" path was injection-based;
             the "move" path was stock-conserving and underperformed, see
             `R2_AUDIT_DECOMPOSITION_2026-06-29.md` §5),
          2. extends the active window to `event.end_time + window_hours`,
          3. if `risk_recovery_boost_downstream` is True, temporarily boosts
             op9/op10/op12 ROP to 12 h and Q to 2x while the window is open
             (reverts when it closes), matching the audit's window+release
             combination that achieved CT p99 ratio 1.15x Excel at release=2,500.
          4. yields to let `_serve_pending_backorders` consume the injection.
        """
        window_hours = self.risk_recovery_window_hours
        release_qty = self.risk_recovery_release_rations
        boost_downstream = self.risk_recovery_boost_downstream
        enabled_risks = set(self.risk_recovery_enabled_risks)
        if boost_downstream and not self._risk_recovery_base_params:
            for key in (
                "op9_rop",
                "op10_rop",
                "op12_rop",
                "op9_q_min",
                "op9_q_max",
                "op10_q_min",
                "op10_q_max",
                "op12_q_min",
                "op12_q_max",
            ):
                self._risk_recovery_base_params[key] = self.params[key]
        while self.env.now < self.horizon:
            activated = False
            for event in list(self.risk_events):
                risk_id = str(event.risk_id)
                if risk_id not in enabled_risks:
                    continue
                key = (risk_id, float(event.start_time), float(event.end_time))
                if key in self._risk_recovery_seen:
                    continue
                self._risk_recovery_seen.add(key)
                yield self.rations_theatre.put(release_qty)
                self._risk_recovery_window_until = max(
                    self._risk_recovery_window_until,
                    float(event.end_time) + window_hours,
                )
                self._risk_recovery_release_emitted += release_qty
                activated = True
            should_boost = (
                boost_downstream
                and self.env.now < self._risk_recovery_window_until
            )
            if should_boost and not self._risk_recovery_boosted:
                for key in ("op9_rop", "op10_rop", "op12_rop"):
                    self.params[key] = 12
                for prefix in ("op9", "op10", "op12"):
                    base_min = float(self._risk_recovery_base_params[f"{prefix}_q_min"])
                    base_max = float(self._risk_recovery_base_params[f"{prefix}_q_max"])
                    self.params[f"{prefix}_q_min"] = int(round(base_min * 2.0))
                    self.params[f"{prefix}_q_max"] = int(round(base_max * 2.0))
                self._risk_recovery_boosted = True
            elif self._risk_recovery_boosted and not should_boost:
                for key, value in self._risk_recovery_base_params.items():
                    self.params[key] = value
                self._risk_recovery_boosted = False
            if (
                (activated or should_boost)
                and self.pending_backorders
                and self.rations_theatre.level > 0
            ):
                yield self.env.process(self._serve_pending_backorders())
            yield self.env.timeout(1.0)

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
        if target <= self.env.now:
            inventory_detail = self._inventory_detail()
            total_inventory = sum(inventory_detail.values())
            return (
                self.get_observation(),
                0.0,
                True,
                {
                    "time": self.env.now,
                    "new_delivered": 0.0,
                    "new_backorders": 0.0,
                    "new_demanded": 0.0,
                    "new_produced": 0.0,
                    "new_backorder_qty": 0.0,
                    "new_unattended_orders": 0.0,
                    "new_available_assembly_hours": 0.0,
                    "new_available_assembly_capacity": 0.0,
                    "step_disruption_hours": 0.0,
                    "total_inventory": total_inventory,
                    "inventory_detail": inventory_detail,
                    "pending_backorders": len(self.pending_backorders),
                    "pending_backorder_delta": 0.0,
                    "pending_backorder_qty": self.pending_backorder_qty,
                    "pending_backorder_qty_delta": 0.0,
                    "unattended_orders_total": self.total_unattended_orders,
                },
            )

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

        self._update_observation_v9_step_features(
            new_delivered=float(new_delivered),
            new_demanded=float(new_demanded),
            new_produced=float(new_produced),
            new_backorder_qty=float(new_backorder_qty),
            new_available_assembly_hours=float(new_available_assembly_hours),
        )

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

    def _finalize_reserved_on_hand_order_after_delay(
        self, order: OrderRecord, delay: float
    ):
        """Finalize a demand order reserved from on-hand stock after lead time."""
        yield self.env.timeout(float(delay))
        order.OATj = self.env.now
        order.CTj = self.env.now - order.OPTj
        order.backorder = False
        order.remaining_qty = 0.0
        self._set_order_ret_indicators(order)

    def _finalize_order_after_fulfillment_delay(self, order: OrderRecord) -> None:
        """Finalize a reserved order after the configured minimum fulfilment delay.

        Garrido's order ledger records delivery after a downstream fulfilment
        cycle, even when theatre stock is available.  The delay is a minimum
        elapsed time from OPTj to OATj; orders that have already waited longer
        than the delay are finalized immediately.
        """
        remaining_delay = max(
            0.0,
            float(self.demand_on_hand_fulfillment_delay)
            - max(0.0, float(self.env.now) - float(order.OPTj)),
        )
        order.remaining_qty = 0.0
        if remaining_delay > 0.0:
            self.env.process(
                self._finalize_reserved_on_hand_order_after_delay(
                    order, remaining_delay
                )
            )
        else:
            self._finalize_pending_backorder(order)

    def _enqueue_backorder(self, order: OrderRecord) -> None:
        """Insert a delayed order into the capped Garrido-style backlog queue.

        Serving priority is always SPT (contingent first, then increasing size).
        Overflow handling follows ``backorder_overflow_mode``: "largest" drops
        the SPT-tail, matching the thesis "last order in the list" rule once
        the list is SPT-sorted. "oldest" drops the earliest OPTj as an
        age-based sensitivity.
        """
        self.pending_backorders.append(order)
        self.pending_backorders.sort(key=self._backorder_priority_key)
        while len(self.pending_backorders) > BACKORDER_QUEUE_CAP:
            if self.backorder_overflow_mode == "oldest":
                # Evict the order that has waited longest (earliest OPTj).
                drop_idx = max(
                    range(len(self.pending_backorders)),
                    key=lambda i: -float(self.pending_backorders[i].OPTj),
                )
                dropped = self.pending_backorders.pop(drop_idx)
            else:
                dropped = self.pending_backorders.pop()
            dropped.lost = True
            # Lost orders were demanded during R14-ubiquitous production, so they
            # carry the R14 risk gate like every Garrido order (AVERAGE(R..)>0).
            # Without it they fall into the no-risk fill-rate branch and score ~1.0,
            # inflating mean ReT ~30x; with the gate and no recovery they score ~0,
            # matching Garrido's lost-order ReT (~0.002).
            dropped.ret_risk_indicators["R14"] = max(
                1.0, dropped.ret_risk_indicators.get("R14", 0.0)
            )
            if not dropped.metrics_excluded:
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
            if not order.metrics_excluded:
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
            self._finalize_order_after_fulfillment_delay(next_order)
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
        Return 12 Track-B adaptive-control + production-aware features for obs v6.

        [0:5]  operating regime one-hot: nominal, strained, pre_disruption,
               disrupted, recovery
        [5]    disruption forecast over next 48h (normalized)
        [6]    disruption forecast over next 168h (normalized)
        [7]    maintenance debt carried from sustained S3 usage
        [8]    average pending-backorder age normalized by config horizon
        [9]    theatre cover days normalized by config horizon
        [10]   daily production rate (normalized by S3 max capacity)
        [11]   R14 defect probability (regime-dependent Binomial p)
        """
        regime_one_hot = np.zeros(len(ADAPTIVE_BENCHMARK_REGIMES), dtype=np.float32)
        regime_index = ADAPTIVE_BENCHMARK_REGIMES.index(self.adaptive_regime)
        regime_one_hot[regime_index] = 1.0
        max_daily_capacity = 3.0 * RATIONS_PER_SHIFT  # S3 = 7,692 rations/day
        production_rate = float(
            np.clip(self._today_produced / max(1.0, max_daily_capacity), 0.0, 1.0)
        )
        r14_p = self._get_risk_p("R14") if self.risks_enabled else 0.0
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
                        float(production_rate),
                        float(np.clip(r14_p, 0.0, 1.0)),
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
        Return 10 Track-B bottleneck + hazard features for obs v7.

        [0]    op10_down
        [1]    op12_down
        [2]    op10_queue_pressure_norm
        [3]    op12_queue_pressure_norm
        [4]    rolling_fill_rate_4w
        [5]    rolling_backorder_rate_4w
        [6]    weeks_since_last_R22 (downstream LOC attack memory)
        [7]    weeks_since_last_R23 (forward unit destruction memory)
        [8]    weeks_since_last_R24 (demand surge memory)
        [9]    ewma_downstream_risk_rate (EWMA of active downstream disruptions)
        """
        rolling_fill_rate, rolling_backorder_rate = self._rolling_service_metrics()
        now = float(self.env.now)
        # Compute weeks-since-last for downstream risks
        def weeks_since(risk_id, window_hours=672.0):
            last_end = 0.0
            for ev in self.risk_events:
                if str(ev.risk_id) == risk_id:
                    last_end = max(last_end, float(ev.end_time))
            return min(1.0, (now - last_end) / window_hours) if last_end > 0 else 1.0
        
        wsl_r22 = weeks_since("R22", 672.0)
        wsl_r23 = weeks_since("R23", 672.0)
        wsl_r24 = weeks_since("R24", 336.0)
        # EWMA of active downstream risks
        active_count = sum(1 for ev in self.risk_events 
                          if str(ev.risk_id) in ("R22","R23","R24") 
                          and float(ev.start_time) <= now <= float(ev.end_time))
        # Also count op10/op12 down status
        active_count += (1 if self._is_down(10) else 0) + (1 if self._is_down(12) else 0)
        ewma_down = min(1.0, active_count / 4.0)  # normalize by max possible
        
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
                float(np.clip(wsl_r22, 0.0, 1.0)),
                float(np.clip(wsl_r23, 0.0, 1.0)),
                float(np.clip(wsl_r24, 0.0, 1.0)),
                float(np.clip(ewma_down, 0.0, 1.0)),
            ],
            dtype=np.float32,
        )

    def get_observation_v8_extra(self, window_hours: float = HOURS_PER_WEEK) -> np.ndarray:
        """
        Return realized risk-ID observability features for obs v8.

        This is not a future oracle. It exposes what has actually occurred:

        - active_<risk>: an event of this risk is active at ``env.now``.
        - recent_<risk>: an event of this risk overlapped the recent window.
        - recent_<risk>_duration_norm: overlapped duration / window_hours.

        Zero-duration quantity risks such as R14 are represented in the recent
        flag when their timestamp falls in the window.
        """
        now = float(self.env.now)
        window = max(1.0, float(window_hours))
        start = max(0.0, now - window)
        active = {risk_id: 0.0 for risk_id in REALIZED_RISK_OBSERVATION_IDS}
        recent = {risk_id: 0.0 for risk_id in REALIZED_RISK_OBSERVATION_IDS}
        duration = {risk_id: 0.0 for risk_id in REALIZED_RISK_OBSERVATION_IDS}
        for event in self.risk_events:
            risk_id = str(event.risk_id)
            if risk_id not in active:
                continue
            event_start = float(event.start_time)
            event_end = float(event.end_time)
            event_duration = max(0.0, float(event.duration))
            if event_duration <= 1e-9:
                if start <= event_start <= now:
                    recent[risk_id] = 1.0
                continue
            if event_start <= now < event_end:
                active[risk_id] = 1.0
            overlap = max(0.0, min(now, event_end) - max(start, event_start))
            if overlap > 0.0:
                recent[risk_id] = 1.0
                duration[risk_id] += overlap
        return np.array(
            [active[risk_id] for risk_id in REALIZED_RISK_OBSERVATION_IDS]
            + [recent[risk_id] for risk_id in REALIZED_RISK_OBSERVATION_IDS]
            + [
                float(np.clip(duration[risk_id] / window, 0.0, 1.0))
                for risk_id in REALIZED_RISK_OBSERVATION_IDS
            ],
            dtype=np.float32,
        )

    def _expected_step_demand(self) -> float:
        mean_daily_demand = 0.5 * float(DEMAND["a"] + DEMAND["b"])
        return max(1.0, mean_daily_demand * max(1.0, float(self._step_size)) / HOURS_PER_DAY)

    def _update_observation_v9_step_features(
        self,
        *,
        new_delivered: float,
        new_demanded: float,
        new_produced: float,
        new_backorder_qty: float,
        new_available_assembly_hours: float,
    ) -> None:
        """Update v9 trend features once per simulation step."""
        demand_scale = max(1.0, float(new_demanded), self._expected_step_demand())
        step_fill = (
            float(new_delivered) / max(1.0, float(new_demanded))
            if float(new_demanded) > 0.0
            else self._fill_rate()
        )
        step_backlog_growth = max(0.0, float(new_backorder_qty)) / demand_scale
        pending_delta = float(self.pending_backorder_qty) - float(
            self._prev_pending_backorder_qty
        )

        self._prev_step_produced = max(0.0, float(new_produced))
        self._prev_step_delivered = max(0.0, float(new_delivered))
        self._prev_step_available_assembly_hours = max(
            0.0, float(new_available_assembly_hours)
        )
        self._ewma_fill_rate = 0.9 * float(self._ewma_fill_rate) + 0.1 * float(
            step_fill
        )
        self._ewma_backlog_growth = 0.8 * float(self._ewma_backlog_growth) + 0.2 * float(
            step_backlog_growth
        )
        self._delta_fill_rate = float(
            np.clip(float(step_fill) - float(self._prev_step_fill_rate), -1.0, 1.0)
        )
        self._delta_backlog_momentum = float(
            np.clip(pending_delta / demand_scale, -1.0, 1.0)
        )
        self._prev_step_fill_rate = float(step_fill)
        self._prev_pending_backorder_qty = float(self.pending_backorder_qty)

    def get_observation_v9_extra(self) -> np.ndarray:
        """
        Return 10 backorder-health, trend, and throughput features for obs v9.

        [0]  backorder_queue_count_norm
        [1]  unattended_total_norm
        [2]  oldest_backorder_age_norm
        [3]  ewma_fill_rate
        [4]  ewma_backlog_growth
        [5]  delta_fill_rate
        [6]  delta_backlog_momentum
        [7]  prev_step_produced_norm
        [8]  prev_step_delivered_norm
        [9]  prev_step_available_assembly_hours_norm
        """
        now = float(self.env.now)
        oldest_age = 0.0
        if self.pending_backorders:
            oldest_opt = min(float(order.OPTj) for order in self.pending_backorders)
            oldest_age = max(0.0, now - oldest_opt) / float(LEAD_TIME_PROMISE)

        step_demand_scale = self._expected_step_demand()
        max_step_production = (
            3.0
            * RATIONS_PER_HOUR
            * max(1.0, float(self._step_size))
            * (6.0 / 7.0)
        )
        max_assembly_hours = 3.0 * max(1.0, float(self._step_size)) * (6.0 / 7.0)

        return np.array(
            [
                float(np.clip(len(self.pending_backorders) / BACKORDER_QUEUE_CAP, 0.0, 1.0)),
                float(np.clip(self.total_unattended_orders / BACKORDER_QUEUE_CAP, 0.0, 1.0)),
                float(np.clip(oldest_age, 0.0, 1.0)),
                float(np.clip(self._ewma_fill_rate, 0.0, 1.0)),
                float(np.clip(self._ewma_backlog_growth, 0.0, 1.0)),
                float(np.clip(self._delta_fill_rate, -1.0, 1.0)),
                float(np.clip(self._delta_backlog_momentum, -1.0, 1.0)),
                float(np.clip(self._prev_step_produced / max(1.0, max_step_production), 0.0, 1.0)),
                float(np.clip(self._prev_step_delivered / step_demand_scale, 0.0, 1.0)),
                float(
                    np.clip(
                        self._prev_step_available_assembly_hours
                        / max(1.0, max_assembly_hours),
                        0.0,
                        1.0,
                    )
                ),
            ],
            dtype=np.float32,
        )

    def get_observation_v10_extra(self) -> np.ndarray:
        """
        Return 12 observed risk-memory features for obs v10.

        For R11/R13/R24:

        [0] weeks_since_last_Ri normalized by the episode horizon
        [1] count_Ri_8w normalized by 8 events
        [2] count_Ri_26w normalized by 26 events
        [3] EWMA-like decayed event count with an 8-week time constant

        These are historical features only. They do not inspect future events or
        the adaptive regime transition matrix.
        """
        now_step = float(self.env.now) / float(HOURS_PER_WEEK)
        features: list[float] = []
        for risk_id in ("R11", "R13", "R24"):
            starts = [
                float(ev.start_time) / float(HOURS_PER_WEEK)
                for ev in self.risk_events
                if str(ev.risk_id) == risk_id
                and float(ev.start_time) < float(self.env.now)
            ]
            if starts:
                weeks_since = max(0.0, now_step - max(starts))
            else:
                weeks_since = 104.0
            count_8w = sum(1 for start in starts if now_step - start <= 8.0)
            count_26w = sum(1 for start in starts if now_step - start <= 26.0)
            ewma = sum(float(np.exp(-(now_step - start) / 8.0)) for start in starts)
            features.extend(
                [
                    float(np.clip(weeks_since / 104.0, 0.0, 1.0)),
                    float(np.clip(count_8w / 8.0, 0.0, 1.0)),
                    float(np.clip(count_26w / 26.0, 0.0, 1.0)),
                    float(np.clip(ewma / 8.0, 0.0, 1.0)),
                ]
            )
        return np.asarray(features, dtype=np.float32)

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
        forecast_48h = next_intensity + float(self.regime_rng.normal(0.0, noise_std))
        forecast_168h = (
            0.35 * current_intensity
            + 0.65 * next_intensity
            + 0.15 * float(current_params["forecast_base"])
            + float(self.regime_rng.normal(0.0, noise_std))
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
            self.adaptive_regime = str(self.regime_rng.choice(next_regimes, p=probs))
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

    def _place_demand_order(self, order: OrderRecord):
        available = self.rations_theatre.level
        if not self.pending_backorders and available >= order.quantity:
            yield self.rations_theatre.get(order.quantity)
            self._finalize_order_after_fulfillment_delay(order)
        else:
            # Enqueue for future delivery. Per thesis Sec. 6.8.2,
            # backorder classification deferred until LTj=48h elapses.
            self._enqueue_backorder(order)
            yield from self._serve_pending_backorders()
            # Start 48h timer — only count as backorder if still pending
            self.env.process(self._delayed_backorder_check(order))
            if self.risk_attribution_source == "excel_risk_tape":
                self._set_order_ret_indicators_from_excel_tape(order)

        self.orders.append(order)
        self.daily_demand.append((self.env.now, order.quantity))

    def _op13_demand_from_excel_order_tape(self):
        for row in self.excel_order_tape:
            optj = float(row["OPTj"])
            if optj > self.env.now:
                yield self.env.timeout(optj - self.env.now)

            demand_qty = float(row["Q"])
            # In tape mode Q already includes any contingent surge visible in
            # the workbook, so R24 must not add demand a second time.
            self._contingent_demand_pending = 0
            self.total_demanded += demand_qty
            order = OrderRecord(
                j=int(row["j"]),
                OPTj=float(self.env.now),
                quantity=demand_qty,
                remaining_qty=demand_qty,
                contingent=bool(row.get("contingent", False)),
                ret_attribution_override=row.get("ret_attribution"),
            )
            yield from self._place_demand_order(order)

    def _sample_calendar_demand_quantity(self) -> tuple[float, bool]:
        demand_qty = float(
            self.demand_rng.integers(int(DEMAND["a"]), int(DEMAND["b"]) + 1)
        )
        demand_qty *= self.demand_mean_multiplier
        if self.adaptive_benchmark_enabled:
            demand_scale = float(self._adaptive_regime_params()["demand_scale"])
            demand_qty *= demand_scale
        contingent_qty = float(self._contingent_demand_pending)
        if contingent_qty > 0:
            demand_qty += contingent_qty
            self._contingent_demand_pending = 0
        return demand_qty, contingent_qty > 0

    def _exclude_current_order_ledger_from_metrics(self) -> None:
        for order in self.orders:
            order.metrics_excluded = True
        for order in self.pending_backorders:
            order.metrics_excluded = True
        self.orders = []
        self.daily_demand = []
        self.total_demanded = 0
        self.total_backorders = 0
        self.total_unattended_orders = 0
        self.cumulative_backorder_qty = 0.0

    def _op13_demand_from_excel_order_tape_after_calendar_warmup(self):
        first_optj = float(self.excel_order_tape[0]["OPTj"])
        order_num = 0
        hour_of_week = 0

        while self.env.now + float(DEMAND["frequency_hrs"]) < first_optj:
            yield self.env.timeout(DEMAND["frequency_hrs"])
            hour_of_week = (hour_of_week + HOURS_PER_DAY) % HOURS_PER_WEEK
            day_of_week = hour_of_week // HOURS_PER_DAY
            if day_of_week >= 6:
                continue

            demand_qty, is_contingent = self._sample_calendar_demand_quantity()
            self.total_demanded += demand_qty
            order_num += 1
            order = OrderRecord(
                j=order_num,
                OPTj=self.env.now,
                quantity=demand_qty,
                remaining_qty=demand_qty,
                contingent=is_contingent,
                metrics_excluded=True,
            )
            yield from self._place_demand_order(order)

        if first_optj > self.env.now:
            yield self.env.timeout(first_optj - self.env.now)

        self._exclude_current_order_ledger_from_metrics()
        yield from self._op13_demand_from_excel_order_tape()

    def _op13_demand(self):
        if self.demand_source == "excel_order_tape":
            yield from self._op13_demand_from_excel_order_tape()
            return
        if self.demand_source == "excel_order_tape_after_calendar_warmup":
            yield from self._op13_demand_from_excel_order_tape_after_calendar_warmup()
            return

        order_num = 0
        hour_of_week = 0
        while True:
            yield self.env.timeout(DEMAND["frequency_hrs"])
            hour_of_week = (hour_of_week + HOURS_PER_DAY) % (7 * HOURS_PER_DAY)
            day_of_week = hour_of_week // HOURS_PER_DAY
            if day_of_week >= 6:
                continue

            demand_qty, is_contingent = self._sample_calendar_demand_quantity()

            self.total_demanded += demand_qty
            order_num += 1
            order = OrderRecord(
                j=order_num,
                OPTj=self.env.now,
                quantity=demand_qty,
                remaining_qty=demand_qty,
                contingent=is_contingent,
            )
            yield from self._place_demand_order(order)

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
        if risk_id == "R3":
            return max(float(base_a), float(base_b))
        return max(float(base_a), float(base_b) / self._risk_frequency_multiplier_for(risk_id))

    def _get_risk_p(self, risk_id: str) -> float:
        table = self._risk_table_for(risk_id)
        if table and risk_id in table:
            base_p = table[risk_id].get("p", RISKS_CURRENT[risk_id]["occurrence"]["p"])
        else:
            base_p = RISKS_CURRENT[risk_id]["occurrence"]["p"]
        if self.adaptive_benchmark_enabled:
            intensity = self._adaptive_risk_intensity_for(risk_id)
            return min(0.98, float(base_p) * intensity)
        return min(0.98, float(base_p) * self._risk_frequency_multiplier_for(risk_id))

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
        return float(base_mean) * self._risk_impact_multiplier_for(risk_id)

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
        psi = self._risk_impact_multiplier_for("R24")
        return int(round(base_lo * psi)), int(round(base_hi * psi))

    def _risk_frequency_multiplier_for(self, risk_id: str) -> float:
        return float(
            self.risk_frequency_multipliers_by_id.get(
                str(risk_id), self.risk_frequency_multiplier
            )
        )

    def _risk_impact_multiplier_for(self, risk_id: str) -> float:
        return float(
            self.risk_impact_multipliers_by_id.get(
                str(risk_id), self.risk_impact_multiplier
            )
        )

    def _sample_uniform_risk_window(self, risk_id: str) -> tuple[float, float]:
        """Sample the event offset for a thesis uniform-occurrence window."""
        a = int(RISKS_CURRENT[risk_id]["occurrence"]["a"])
        b_val = max(a, int(round(self._get_risk_b(risk_id))))
        delay = float(self.risk_rng.integers(a, b_val + 1))
        return delay, float(b_val)

    def _tail_after_uniform_occurrence(self, delay: float, window: float) -> float:
        if self.risk_occurrence_mode == "thesis_window":
            return max(0.0, float(window) - float(delay))
        return 0.0

    def _risk_R11(self):
        beta = self._get_risk_recovery_mean("R11")
        while True:
            delay, window = self._sample_uniform_risk_window("R11")
            yield self.env.timeout(delay)
            if self.risk_occurrence_mode == "thesis_window":
                self.env.process(self._risk_R11_event(beta))
                yield self.env.timeout(self._tail_after_uniform_occurrence(delay, window))
            else:
                yield from self._risk_R11_event(beta)

    def _risk_R11_event(self, beta: float):
        target = int(self.risk_rng.choice(RISKS_CURRENT["R11"]["affected_ops"]))
        start = self.env.now
        self._take_down(target)
        repair = max(1, self.risk_rng.exponential(beta))
        yield self.env.timeout(repair)
        self._bring_up(target)
        self.risk_events.append(
            RiskEvent("R11", start, self.env.now, self.env.now - start, [target])
        )

    def _risk_R12(self):
        n = RISKS_CURRENT["R12"]["occurrence"]["n"]
        p = self._get_risk_p("R12")
        while True:
            yield self.env.timeout(self.params["op1_rop"])
            delayed = self.risk_rng.binomial(n, p)
            if delayed > 0:
                delay = delayed * 168
                if self.risk_occurrence_mode == "thesis_window":
                    self.env.process(self._risk_R12_event(delay, delayed))
                else:
                    yield from self._risk_R12_event(delay, delayed)

    def _risk_R12_event(self, delay: float, delayed: float):
        start = self.env.now
        self._take_down(1)
        yield self.env.timeout(delay)
        self._bring_up(1)
        self.risk_events.append(
            RiskEvent(
                "R12",
                start,
                self.env.now,
                delay,
                [1],
                magnitude=float(delayed),
                unit="delayed_contracts",
            )
        )

    def _risk_R13(self):
        n = RISKS_CURRENT["R13"]["occurrence"]["n"]
        p = self._get_risk_p("R13")
        interval = (
            HOURS_PER_WEEK
            if self.risk_occurrence_mode == "thesis_window"
            else self.params["op2_rop"]
        )
        while True:
            yield self.env.timeout(interval)
            delayed = self.risk_rng.binomial(n, p)
            if delayed > 0:
                delay = delayed * 24
                if self.risk_occurrence_mode == "thesis_window":
                    self.env.process(self._risk_R13_event(delay, delayed))
                else:
                    yield from self._risk_R13_event(delay, delayed)

    def _risk_R13_event(self, delay: float, delayed: float):
        start = self.env.now
        self._take_down(2)
        yield self.env.timeout(delay)
        self._bring_up(2)
        self.risk_events.append(
            RiskEvent(
                "R13",
                start,
                self.env.now,
                delay,
                [2],
                magnitude=float(delayed),
                unit="delayed_deliveries",
            )
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
                        event = RiskEvent(
                            "R14",
                            self.env.now,
                            self.env.now,
                            0,
                            [7],
                            description,
                            magnitude=float(defects),
                            unit="defective_products",
                        )
                        self.risk_events.append(event)
                        self._add_ret_quantity_risk(event)

    def _risk_R21(self):
        """R21 natural disaster generator — non-blocking mode.

        Each event takes down all affected operations simultaneously; each
        operation recovers independently with Exp(beta) hours. The generator
        spawns a new process for each event so they can theoretically overlap.
        """
        beta = self._get_risk_recovery_mean("R21")
        affected = RISKS_CURRENT["R21"]["affected_ops"]
        while True:
            delay, window = self._sample_uniform_risk_window("R21")
            yield self.env.timeout(delay)
            self.env.process(self._r21_event(affected, beta))
            tail = self._tail_after_uniform_occurrence(delay, window)
            if tail > 0:
                yield self.env.timeout(tail)

    def _r21_event(self, affected: list[int], beta: float):
        start = self.env.now
        for op_id in affected:
            self._take_down(op_id)
        recovery_times = {}
        for op_id in affected:
            rt = max(1, self.risk_rng.exponential(beta))
            recovery_times[op_id] = rt
            self.env.process(self._delayed_bring_up(op_id, rt))
        max_rt = max(recovery_times.values())
        yield self.env.timeout(max_rt)
        self.risk_events.append(
            RiskEvent("R21", start, self.env.now, max_rt, list(affected))
        )

    def _risk_R22(self):
        beta = self._get_risk_recovery_mean("R22")
        loc_ops = RISKS_CURRENT["R22"]["affected_ops"]
        while True:
            delay, window = self._sample_uniform_risk_window("R22")
            yield self.env.timeout(delay)
            if self.risk_occurrence_mode == "thesis_window":
                self.env.process(self._risk_R22_event(beta, loc_ops))
                yield self.env.timeout(self._tail_after_uniform_occurrence(delay, window))
            else:
                yield from self._risk_R22_event(beta, loc_ops)

    def _risk_R22_event(self, beta: float, loc_ops: list[int]):
        target = int(self.risk_rng.choice(loc_ops))
        start = self.env.now
        self._take_down(target)
        recovery = max(1, self.risk_rng.exponential(beta))
        yield self.env.timeout(recovery)
        self._bring_up(target)
        self.risk_events.append(
            RiskEvent("R22", start, self.env.now, recovery, [target])
        )

    def _risk_R23(self):
        beta = self._get_risk_recovery_mean("R23")
        while True:
            delay, window = self._sample_uniform_risk_window("R23")
            yield self.env.timeout(delay)
            if self.risk_occurrence_mode == "thesis_window":
                self.env.process(self._risk_R23_event(beta))
                yield self.env.timeout(self._tail_after_uniform_occurrence(delay, window))
            else:
                yield from self._risk_R23_event(beta)

    def _risk_R23_event(self, beta: float):
        start = self.env.now
        self._take_down(11)
        recovery = max(1, self.risk_rng.exponential(beta))
        yield self.env.timeout(recovery)
        self._bring_up(11)
        self.risk_events.append(
            RiskEvent("R23", start, self.env.now, recovery, [11])
        )

    def _risk_R24(self):
        while True:
            delay, window = self._sample_uniform_risk_window("R24")
            yield self.env.timeout(delay)
            self._apply_risk_R24_event()
            tail = self._tail_after_uniform_occurrence(delay, window)
            if tail > 0:
                yield self.env.timeout(tail)

    def _apply_risk_R24_event(self) -> None:
        surge_lo, surge_hi = self._get_risk_surge()
        surge = self.risk_rng.integers(surge_lo, surge_hi + 1)
        self._contingent_demand_pending += surge
        # Cap accumulated contingent demand to prevent unbounded obs[14]
        # spikes when multiple R24 events fire before demand is consumed.
        # 5×2600 = 13000 ≈ 5 regular demand cycles, well above any
        # realistic surge accumulation.
        max_contingent = 5 * 2600
        self._contingent_demand_pending = min(
            self._contingent_demand_pending, max_contingent
        )
        event = RiskEvent(
            "R24",
            self.env.now,
            self.env.now,
            0,
            [13],
            f"+{surge}",
            magnitude=float(surge),
            unit="rations",
        )
        self.risk_events.append(event)
        self._add_ret_quantity_risk(event)

    def _risk_R3(self):
        duration = RISKS_CURRENT["R3"]["recovery"]["duration"]
        affected = RISKS_CURRENT["R3"]["affected_ops"]
        while True:
            delay, window = self._sample_uniform_risk_window("R3")
            yield self.env.timeout(delay)
            if self.risk_occurrence_mode == "thesis_window":
                self.env.process(self._risk_R3_event(duration, affected))
                yield self.env.timeout(self._tail_after_uniform_occurrence(delay, window))
            else:
                yield from self._risk_R3_event(duration, affected)

    def _risk_R3_event(self, duration: float, affected: list[int]):
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
        return compute_fill_rate_from_orders(self.orders, current_time=self.env.now)

    def _set_order_ret_indicators(self, order: OrderRecord) -> None:
        """
        Populate ReT sub-indicators APj, RPj, DPj for a completed order.

        The Garrido raw workbooks gate ReT with visible risk columns
        (``AVERAGE(R..)>0``).  Duration events contribute by time overlap and
        point events such as R14/R24 contribute when their timestamp falls
        inside the order window.  Earlier code missed point events because it
        required positive duration overlap, which suppressed the Excel risk
        branch for defective-product rows.
        """
        if order.OATj is None or order.CTj is None:
            return

        order.ret_risk_indicators.clear()
        order.ret_risk_event_refs.clear()
        if self.risk_attribution_source == "excel_risk_tape":
            self._set_order_ret_indicators_from_excel_tape(order)
            return

        total_disruption_hours = 0.0
        earliest_risk_start = float("inf")

        def mark_event(event: RiskEvent, contribution: float) -> None:
            nonlocal earliest_risk_start
            key = str(event.risk_id)
            order.ret_risk_indicators[key] = (
                order.ret_risk_indicators.get(key, 0.0) + float(contribution)
            )
            order.ret_risk_event_refs.append(
                {
                    "risk_id": key,
                    "start_time": float(event.start_time),
                    "end_time": float(event.end_time),
                    "duration": float(event.duration),
                    "affected_ops": list(event.affected_ops),
                    "magnitude": float(event.magnitude),
                    "unit": str(event.unit),
                }
            )
            earliest_risk_start = min(earliest_risk_start, float(event.start_time))

        for event in self.risk_events:
            if float(event.duration) <= 0.0:
                if float(order.OPTj) <= float(event.start_time) <= float(order.OATj):
                    mark_event(event, float(event.magnitude))
                continue

            overlap_start = max(float(event.start_time), float(order.OPTj))
            overlap_end = min(float(event.end_time), float(order.OATj))
            if overlap_start < overlap_end:
                overlap = overlap_end - overlap_start
                total_disruption_hours += overlap
                mark_event(event, overlap)

        # Include ongoing disruptions at fulfillment time
        for op_id in range(1, 14):
            down_since = self._op_down_since.get(op_id)
            if self.op_down_count[op_id] > 0 and down_since is not None:
                overlap_start = max(down_since, order.OPTj)
                overlap_end = order.OATj
                if overlap_start < overlap_end:
                    total_disruption_hours += overlap_end - overlap_start
                    key = f"ongoing_op{op_id}"
                    order.ret_risk_indicators[key] = (
                        order.ret_risk_indicators.get(key, 0.0)
                        + overlap_end
                        - overlap_start
                    )
                    earliest_risk_start = min(earliest_risk_start, down_since)

        quantity_risk_hours, quantity_risk_start = self._consume_ret_quantity_risk_for_order(
            order
        )
        if quantity_risk_hours > 0.0:
            total_disruption_hours += quantity_risk_hours
            earliest_risk_start = min(
                earliest_risk_start,
                float(quantity_risk_start)
                if quantity_risk_start is not None
                else float(order.OPTj),
            )

        # R24 is a point event that materializes as a contingent-demand order.
        # If the event happened before OPTj, pure time-overlap attribution misses
        # the order even though Garrido's spreadsheet marks the order-level risk
        # column. Treat the contingent order itself as the R24 impact window.
        if bool(getattr(order, "contingent", False)) and "R24" not in order.ret_risk_indicators:
            contribution = max(1.0, min(float(order.CTj), float(order.LTj)))
            total_disruption_hours += contribution
            order.ret_risk_indicators["R24"] = contribution
            order.ret_risk_event_refs.append(
                {
                    "risk_id": "R24",
                    "start_time": float(order.OPTj),
                    "end_time": float(order.OATj),
                    "duration": 0.0,
                    "affected_ops": [13],
                    "magnitude": float(order.quantity),
                    "unit": "contingent_order",
                }
            )
            earliest_risk_start = min(earliest_risk_start, float(order.OPTj))

        if not order.ret_risk_indicators:
            return  # No disruption: fill_rate case

        if order.CTj <= order.LTj:
            # Autotomy: SC absorbed disruption, order still on time
            order.APj = min(total_disruption_hours, order.LTj)
        else:
            # Recovery / Non-recovery: order delayed beyond lead time
            order.DPj = order.CTj
            if self.ret_recovery_period_mode == "disruption":
                # Garrido raw-workbook semantics (thesis Eq. 5.3): RPj is the
                # recovery/disruption duration of the risk(s) affecting the
                # order, NOT the elapsed wall-clock since the first risk onset.
                # The elapsed mode lets plain queue wait inflate RPj up to CTj,
                # which diverges from the bounded workbook RPj distribution.
                order.RPj = max(0.0, total_disruption_hours)
            else:
                eff_risk_start = max(earliest_risk_start, order.OPTj)
                order.RPj = max(0.0, order.OATj - eff_risk_start)

    def _set_order_ret_indicators_from_excel_tape(self, order: OrderRecord) -> None:
        """
        Replay Garrido workbook-visible risk attribution for replication mode.

        This does not copy OATj, CTj, or ReT. It only imports the columns that
        the raw workbook itself uses as the operational risk gate and periods:
        APj, RPj, DPj, LTj, and the visible R... indicators for the same order.
        """
        attribution = order.ret_attribution_override
        if attribution is None:
            raise ValueError(
                "risk_attribution_source='excel_risk_tape' requires an order "
                "ret_attribution override."
            )

        order.ret_risk_indicators.clear()
        order.ret_risk_event_refs.clear()
        order.APj = float(attribution.get("APj", 0.0) or 0.0)
        order.RPj = float(attribution.get("RPj", 0.0) or 0.0)
        order.DPj = float(attribution.get("DPj", 0.0) or 0.0)
        order.LTj = float(attribution.get("LTj", order.LTj) or order.LTj)
        event_end = (
            float(order.OATj)
            if order.OATj is not None
            else float(order.OPTj) + max(order.DPj, order.RPj, order.LTj)
        )
        event_duration = (
            float(order.CTj)
            if order.CTj is not None
            else max(order.DPj, order.RPj, order.LTj)
        )
        for risk_id, value in (attribution.get("risk_values", {}) or {}).items():
            risk_value = float(value or 0.0)
            if risk_value <= 0.0:
                continue
            key = str(risk_id)
            order.ret_risk_indicators[key] = risk_value
            order.ret_risk_event_refs.append(
                {
                    "risk_id": key,
                    "start_time": float(order.OPTj),
                    "end_time": event_end,
                    "duration": event_duration,
                    "affected_ops": [],
                    "magnitude": risk_value,
                    "unit": "excel_visible_risk_indicator",
                    "source": "excel_risk_tape",
                }
            )

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

    def compute_order_level_ret(
        self,
        *,
        orders: Optional[Iterable[OrderRecord]] = None,
        j_source: str = "row_index",
    ) -> dict[str, Any]:
        """
        Compute order-level ReT with Garrido's raw Excel formula as primary.

        Returns dict with aggregate ReT metrics:
        - mean_ret: Excel-faithful raw workbook ReT
        - mean_ret_text_formula: older textual Eq. 5.1-5.5 interpretation
        - fill_rate_order_level: Re(FRt) = 1 - (Bt+Ut)/Dt (order counts, Eq. 5.4)
        - n_orders, n_completed: total and completed order counts

        ``orders`` and ``j_source`` are optional forensic controls for matching
        Garrido's workbook-visible ledgers.  The default remains the full DES
        order ledger with consecutive visible rows.
        """
        order_list = list(self.orders if orders is None else orders)
        fill_rate = compute_fill_rate_from_orders(
            order_list,
            current_time=self.env.now,
        )
        text_summary = compute_thesis_order_level_ret(
            order_list,
            fill_rate=fill_rate,
            ret_weights=THESIS_FAITHFUL_PROTOCOL["ret_weights"],
        )
        excel_summary = compute_order_level_ret_excel_formula(
            order_list,
            current_time=float(self.env.now),
            j_source=j_source,
        )
        return {
            "mean_ret": excel_summary["mean_ret_excel"],
            "mean_ret_excel_formula": excel_summary["mean_ret_excel"],
            "mean_ret_text_formula": text_summary["mean_ret"],
            "fill_rate_order_level": text_summary["fill_rate_order_level"],
            "case_counts": excel_summary["case_counts"],
            "case_counts_excel_formula": excel_summary["case_counts"],
            "case_counts_text_formula": text_summary["case_counts"],
            "n_orders": excel_summary["n_orders"],
            "n_completed": excel_summary["n_completed"],
            "j_source": j_source,
        }
