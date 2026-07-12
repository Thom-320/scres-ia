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

import hashlib
import math

import simpy
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional
from collections import Counter

from .cssu_allocation import (
    ALLOCATION_LEVELS,
    SERVICE_RULES,
    allocate_shared_capacity,
    stable_cssu_destination,
)

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
    BACKORDER_PRIORITY_RULE_OPTIONS,
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
    PROCUREMENT_CONTRACT_MODE_OPTIONS,
    ORDER_FULFILLMENT_MODE_OPTIONS,
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
    in_flight_qty: float = 0.0
    contingent: bool = False
    # Opt-in Program D topology. ``None`` preserves the aggregate Garrido lane.
    cssu_destination: Optional[str] = None
    lost: bool = False
    lost_time: Optional[float] = None
    metrics_excluded: bool = False
    # Thesis ReT sub-indicators (Garrido-Rios 2017, Eq. 5.1-5.5):
    APj: float = 0.0  # Autotomy period (hours): CTj=LTj and risks impact in [OPTj,OATj]
    RPj: float = 0.0  # Recovery period (hours): OATj - first R0cr detection
    DPj: float = 0.0  # Disruption period (hours): CTj when CTj > LTj
    ret_risk_indicators: dict[str, float] = field(default_factory=dict)
    ret_risk_event_refs: list[dict[str, Any]] = field(default_factory=list)
    ret_attribution_override: Optional[dict[str, Any]] = None
    # Diagnostic-only causal timing ledger. These fields do not enter ReT.
    # They let the fidelity audit distinguish release/stock waiting, convoy
    # resource queues, and time physically blocked by an affected operation.
    op9_release_time: Optional[float] = None
    causal_wait_hours: dict[str, float] = field(default_factory=dict)
    causal_block_intervals: list[dict[str, Any]] = field(default_factory=list)
    causal_r24_event_ids: set[str] = field(default_factory=set)
    # Opt-in material genealogy. ``consumed_material_lineage`` is descriptive:
    # it proves which affected lots supplied the order. ``lineage_shortage_refs``
    # is the stricter candidate: it contains only open upstream debts observed
    # while this order was physically blocked for stock. A counterfactual
    # delayed-quantity model is still required before calling it causal.
    consumed_material_lineage: list[dict[str, Any]] = field(default_factory=list)
    lineage_shortage_refs: list[dict[str, Any]] = field(default_factory=list)


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
    affected_cssu: Optional[str] = None


@dataclass
class MaterialLineageSlice:
    """FIFO quantity slice carried alongside a physical SimPy container."""

    quantity: float
    lot_id: str
    risk_event_refs: tuple[str, ...] = ()
    source_stage: str = "external"
    created_at: float = 0.0


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
        backorder_priority_rule: str = "spt_contingent",
        backorder_age_threshold_hours: float = 336.0,
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
        campaign_config: Optional[dict[str, Any]] = None,
        replenishment_route_aware: bool = False,
        procurement_contract_mode: str = "legacy_independent",
        order_fulfillment_mode: str = "legacy_theatre_stock",
        assembly_flow_mode: str = "aggregate_line",
        periodic_release_mode: str = "completion_relative",
        operational_risk_initialization_mode: str = "deferred_first_cycle",
        risk_rng_mode: str = "shared",
        op9_dispatch_policy: str = "fixed_clock_daily",
        downstream_transport_capacity_mode: str = "parallel",
        op9_freight_offset_hours: float = 6.0,
        r24_attribution_window_hours: float = 0.0,
        demand_start_after_warmup: bool = False,
        material_lineage_mode: str = "off",
        op2_release_clock_mode: str = "inherit",
        cssu_topology_mode: str = "aggregate",
        cssu_allocation_a: float = 0.50,
        cssu_service_rule: str = "SPT_FULL",
        cssu_daily_capacity: Optional[float] = None,
        op8_dispatch_mode: str = "thesis_full_batch",
        op8_convoy_capacity: float = 5_000.0,
        op8_convoy_outbound_hours: float = 24.0,
        op8_convoy_return_hours: float = 24.0,
        serial_wip_capacity_rations: Optional[tuple[float, float]] = None,
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
        if material_lineage_mode not in {"off", "tagged_lots"}:
            raise ValueError(
                "material_lineage_mode must be 'off' or 'tagged_lots'."
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
        if procurement_contract_mode not in PROCUREMENT_CONTRACT_MODE_OPTIONS:
            valid = ", ".join(PROCUREMENT_CONTRACT_MODE_OPTIONS)
            raise ValueError(
                "Invalid procurement_contract_mode="
                f"{procurement_contract_mode!r}. Expected one of: {valid}."
            )
        if order_fulfillment_mode not in ORDER_FULFILLMENT_MODE_OPTIONS:
            valid = ", ".join(ORDER_FULFILLMENT_MODE_OPTIONS)
            raise ValueError(
                "Invalid order_fulfillment_mode="
                f"{order_fulfillment_mode!r}. Expected one of: {valid}."
            )
        if ret_recovery_period_mode not in RET_RECOVERY_PERIOD_MODE_OPTIONS:
            valid = ", ".join(RET_RECOVERY_PERIOD_MODE_OPTIONS)
            raise ValueError(
                "Invalid ret_recovery_period_mode="
                f"{ret_recovery_period_mode!r}. Expected one of: {valid}."
            )
        if op9_dispatch_policy not in {"fixed_clock_daily", "ready_headway"}:
            raise ValueError(
                "Invalid op9_dispatch_policy="
                f"{op9_dispatch_policy!r}. Expected fixed_clock_daily or "
                "ready_headway."
            )
        if assembly_flow_mode not in {"aggregate_line", "serial_wip"}:
            raise ValueError(
                "Invalid assembly_flow_mode="
                f"{assembly_flow_mode!r}. Expected aggregate_line or serial_wip."
            )
        if serial_wip_capacity_rations is not None:
            if len(serial_wip_capacity_rations) != 2 or min(
                map(float, serial_wip_capacity_rations)
            ) <= 0.0:
                raise ValueError(
                    "serial_wip_capacity_rations must contain two positive capacities."
                )
        if periodic_release_mode not in {"completion_relative", "start_to_start"}:
            raise ValueError(
                "Invalid periodic_release_mode="
                f"{periodic_release_mode!r}. Expected completion_relative or "
                "start_to_start."
            )
        if op2_release_clock_mode not in {"inherit", "calendar_anchored"}:
            raise ValueError(
                "op2_release_clock_mode must be 'inherit' or 'calendar_anchored'."
            )
        if operational_risk_initialization_mode not in {
            "deferred_first_cycle",
            "include_initial_cycle",
        }:
            raise ValueError(
                "Invalid operational_risk_initialization_mode="
                f"{operational_risk_initialization_mode!r}. Expected "
                "deferred_first_cycle or include_initial_cycle."
            )
        if risk_rng_mode not in {"shared", "per_risk"}:
            raise ValueError(
                f"Invalid risk_rng_mode={risk_rng_mode!r}. Expected shared or per_risk."
            )
        if downstream_transport_capacity_mode not in {
            "parallel",
            "tandem_capacity_one",
        }:
            raise ValueError(
                "Invalid downstream_transport_capacity_mode="
                f"{downstream_transport_capacity_mode!r}. Expected parallel "
                "or tandem_capacity_one."
            )
        if cssu_topology_mode not in {"aggregate", "split_v1"}:
            raise ValueError(
                "cssu_topology_mode must be 'aggregate' or 'split_v1'."
            )
        if float(cssu_allocation_a) not in ALLOCATION_LEVELS:
            raise ValueError(
                f"cssu_allocation_a must be one of {ALLOCATION_LEVELS}."
            )
        if cssu_service_rule not in SERVICE_RULES:
            raise ValueError(f"cssu_service_rule must be one of {SERVICE_RULES}.")
        if cssu_daily_capacity is not None and float(cssu_daily_capacity) <= 0:
            raise ValueError("cssu_daily_capacity must be positive when provided.")
        if op8_dispatch_mode not in {"thesis_full_batch", "finite_convoy_v1"}:
            raise ValueError(
                "op8_dispatch_mode must be 'thesis_full_batch' or "
                "'finite_convoy_v1'."
            )
        if min(
            float(op8_convoy_capacity),
            float(op8_convoy_outbound_hours),
            float(op8_convoy_return_hours),
        ) <= 0.0:
            raise ValueError("Op8 convoy capacity and travel times must be positive.")
        if backorder_overflow_mode not in BACKORDER_OVERFLOW_MODE_OPTIONS:
            valid = ", ".join(BACKORDER_OVERFLOW_MODE_OPTIONS)
            raise ValueError(
                "Invalid backorder_overflow_mode="
                f"{backorder_overflow_mode!r}. Expected one of: {valid}."
            )
        if backorder_priority_rule not in BACKORDER_PRIORITY_RULE_OPTIONS:
            valid = ", ".join(BACKORDER_PRIORITY_RULE_OPTIONS)
            raise ValueError(
                "Invalid backorder_priority_rule="
                f"{backorder_priority_rule!r}. Expected one of: {valid}."
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
        # Track C campaign regime (2026-07-10): an exogenous calm/campaign
        # schedule sampled ONCE from the seed (dedicated stream) so CRN-paired
        # policies share identical campaign paths. State-dependent frequency is
        # realized by exact thinning: risks named in frequency_multipliers are
        # SAMPLED at their maximum (campaign) rate and each candidate event is
        # ACCEPTED with probability m(state)/m_max — calm keeps the native rate,
        # campaign multiplies it. Impact multipliers apply at event-fire time.
        self.campaign_config: Optional[dict[str, Any]] = (
            dict(campaign_config) if campaign_config else None
        )
        self.campaign_path: list[tuple[float, str]] = []
        self.replenishment_route_aware = bool(replenishment_route_aware)
        self.procurement_contract_mode = str(procurement_contract_mode)
        self.order_fulfillment_mode = str(order_fulfillment_mode)
        self.cssu_topology_mode = str(cssu_topology_mode)
        self.cssu_allocation_a = float(cssu_allocation_a)
        self.cssu_service_rule = str(cssu_service_rule)
        self.cssu_daily_capacity_override = (
            None if cssu_daily_capacity is None else float(cssu_daily_capacity)
        )
        self._pending_cssu_action: Optional[dict[str, Any]] = None
        self.cssu_action_events: list[dict[str, Any]] = []
        self.cssu_demand_events: list[tuple[float, str, float]] = []
        self.cssu_delivery_events: list[tuple[float, str, float]] = []
        self.op8_dispatch_mode = str(op8_dispatch_mode)
        self.op8_convoy_capacity = float(op8_convoy_capacity)
        self.op8_convoy_outbound_hours = float(op8_convoy_outbound_hours)
        self.op8_convoy_return_hours = float(op8_convoy_return_hours)
        self.assembly_flow_mode = str(assembly_flow_mode)
        self.serial_wip_capacity_rations = (
            None
            if serial_wip_capacity_rations is None
            else tuple(map(float, serial_wip_capacity_rations))
        )
        self.periodic_release_mode = str(periodic_release_mode)
        self.op2_release_clock_mode = str(op2_release_clock_mode)
        self.operational_risk_initialization_mode = str(
            operational_risk_initialization_mode
        )
        self.risk_rng_mode = str(risk_rng_mode)
        risk_salts = {
            "R11": 11,
            "R12": 12,
            "R13": 13,
            "R14": 14,
            "R21": 21,
            "R22": 22,
            "R23": 23,
            "R24": 24,
            "R3": 30,
        }
        self.risk_rng_by_id = {
            risk_id: np.random.default_rng(
                np.random.SeedSequence([int(seed or 0), 0xA17D17, salt])
            )
            for risk_id, salt in risk_salts.items()
        }
        self.op9_dispatch_policy = str(op9_dispatch_policy)
        self.downstream_transport_capacity_mode = str(
            downstream_transport_capacity_mode
        )
        self.op9_freight_offset_hours = float(op9_freight_offset_hours)
        self.r24_attribution_window_hours = max(
            0.0, float(r24_attribution_window_hours)
        )
        self.demand_start_after_warmup = bool(demand_start_after_warmup)
        self._route_wait_pending: set[str] = set()
        if self.campaign_config:
            self._build_campaign_path()
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
        self.material_lineage_mode = material_lineage_mode
        # Causal-exposure attribution state (lazy; only populated when used).
        self._queue_len_history: list[tuple[float, int]] = []
        self._exposure_end_cache: dict[int, float] = {}
        self._r24_causal_episodes: dict[str, dict[str, Any]] = {}
        self._r24_causal_sequence = 0
        self._upstream_scarcity_debts: list[RiskEvent] = []
        self._material_lineage_sequence = 0
        self._material_lineage: dict[str, list[MaterialLineageSlice]] = {
            node: []
            for node in (
                "raw_material_wdc",
                "raw_material_al",
                "rework_op6",
                "pending_batch",
                "rations_al",
                "rations_sb",
            )
        }
        self._pending_lineage_events_by_stage: dict[str, list[str]] = {
            stage: [] for stage in ("op2_output", "op4_output", "op7_output", "op8_output")
        }
        self._lineage_event_index: dict[str, RiskEvent] = {}
        self.ret_recovery_period_mode = ret_recovery_period_mode
        self.backorder_overflow_mode = backorder_overflow_mode
        self.backorder_priority_rule = backorder_priority_rule
        self.backorder_age_threshold_hours = float(backorder_age_threshold_hours)
        # Program D audit trail. Construction does not append an event, preserving
        # the frozen default trajectory bitwise; only explicit dynamic changes do.
        self.backorder_priority_rule_events: list[dict[str, Any]] = []
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
        wip_caps = self.serial_wip_capacity_rations or (INF, INF)
        self.wip_op5_op6 = simpy.Container(
            self.env, capacity=float(wip_caps[0]), init=0
        )
        self.wip_op6_op7 = simpy.Container(
            self.env, capacity=float(wip_caps[1]), init=0
        )
        self.rations_al = simpy.Container(self.env, capacity=INF, init=0)
        self.rations_sb = simpy.Container(self.env, capacity=INF, init=0)
        self.rations_sb_dispatch = simpy.Container(self.env, capacity=INF, init=0)
        self.rations_cssu = simpy.Container(self.env, capacity=INF, init=0)
        # Program D split ledgers are inert in the default aggregate lane.
        # They make destination-specific flow observable without changing the
        # shared physical stock or downstream capacity.
        self.cssu_in_transit = {"A": 0.0, "B": 0.0}
        self.cssu_inbound_in_transit = {"A": 0.0, "B": 0.0}
        self.cssu_inventory = {"A": 0.0, "B": 0.0}
        self.cssu_outbound_in_transit = {"A": 0.0, "B": 0.0}
        self.cssu_delivered = {"A": 0.0, "B": 0.0}
        self.cssu_demanded = {"A": 0.0, "B": 0.0}
        self.cssu_dispatched = {"A": 0.0, "B": 0.0}
        self.cssu_allocation_live_epochs = 0
        self.cssu_allocation_moot_epochs = 0
        self.cssu_local_down_count = {
            (op_id, cssu): 0
            for op_id in (10, 11, 12)
            for cssu in ("A", "B")
        }
        self.cssu_local_risk_events: list[dict[str, Any]] = []
        self.rations_theatre = simpy.Container(self.env, capacity=INF, init=0)
        # Program-v2 opt-in reserve.  It is deliberately separate from the
        # Garrido theatre stock: in ``op9_linked`` mode normal orders travel
        # through Op10--Op12, whereas this stock is already positioned behind
        # that corridor.  Defaults are inert so frozen Track A/B/L lanes retain
        # their exact physical contract.
        self.emergency_theatre_reserve = simpy.Container(
            self.env, capacity=INF, init=0
        )
        self.emergency_reserve_enabled = False
        self.emergency_reserve_capacity = 0.0
        self.emergency_reserve_target = 0.0
        self.emergency_reserve_replenishment_lead_time = 336.0
        self.emergency_reserve_transport_mode = "fixed_lead"
        self.emergency_reserve_issue_delay = 24.0
        self.emergency_reserve_route_ops: tuple[int, ...] = (10, 11, 12)
        self.emergency_reserve_in_transit = 0.0
        self.emergency_reserve_units_issued = 0.0
        self.emergency_reserve_units_replenished = 0.0
        self.emergency_reserve_replenishment_requests = 0
        self.emergency_reserve_target_changes = 0
        self.emergency_reserve_inventory_time = 0.0
        self._emergency_reserve_last_accounting_time = 0.0
        # Program F is opt-in.  The reserve stock remains the existing physical
        # container; these fields only authorize a bounded R24 fragment release.
        self.program_f_reserve_enabled = False
        self.program_f_r24_issue_remaining = 0.0
        self.program_f_reserve_fragments_issued = 0.0
        self.program_f_reserve_issue_events: list[dict[str, float]] = []
        # op9_linked mode: LOC convoys (Op10, Op12) are capacity-1 daily-rate
        # servers (Table 6.20 ROP=24h) — a tandem pipeline, not parallel legs.
        self.op10_convoy = simpy.Resource(self.env, capacity=1)
        self.op12_convoy = simpy.Resource(self.env, capacity=1)
        # DRA-2 opt-in Op7-Op8 convoy. Historical transport remains untouched
        # unless ``op8_dispatch_mode='finite_convoy_v1'``.
        self.op8_convoy_available = True
        self.op8_convoy_departures = 0
        self.op8_convoy_dispatched_rations = 0.0
        self.op8_convoy_capacity_committed = 0.0
        self.op8_convoy_vehicle_hours = 0.0
        self.op8_convoy_idle_hours = 0.0
        self.op8_convoy_route_wait_hours = 0.0
        self.op8_convoy_ration_hours_in_transit = 0.0
        self.op8_convoy_masked_dispatch_attempts = 0
        self.op8_convoy_hold_actions = 0
        self.op8_convoy_dispatch_actions = 0
        self.op8_convoy_last_departure_at: Optional[float] = None
        self.op8_convoy_nominal_return_at: Optional[float] = None
        self.op8_convoy_actual_return_at: Optional[float] = None
        self.op8_staging_first_ready_at: Optional[float] = None
        self.op8_last_action = "HOLD"
        self.op8_convoy_action_events: list[dict[str, Any]] = []
        self.op8_convoy_departure_events: list[dict[str, Any]] = []

        # Procurement contract state.  Op1 creates/renews the framework
        # contract that authorises Op2 supplier deliveries.  The previous
        # implementation ran Op1 and Op2 as independent clocks, which meant
        # R12 could change the risk ledger without changing material flow.
        self.contract_valid_until = 0.0
        self._contract_renewed_event = self.env.event()
        self.contract_completion_events: list[tuple[float, float]] = []
        self.supplier_delivery_events: list[tuple[float, float]] = []
        # Cumulative-availability event streams used by paired leave-one-risk-out
        # audits. They are observational only and never feed a process decision.
        self.material_availability_events: dict[str, list[tuple[float, float]]] = {
            node: []
            for node in (
                "raw_material_wdc",
                "raw_material_al",
                "rations_al",
                "rations_sb",
                "order_release",
            )
        }

        self.inventory_buffer_targets = self._normalize_inventory_buffer_targets(
            initial_buffers or {}
        )
        self.total_external_raw_material = 0.0
        self.total_strategic_raw_injected = 0.0
        self.total_strategic_rations_injected = 0.0
        self.total_rations_created_from_raw = 0.0
        self.total_rations_scrapped = 0.0
        self.total_raw_material_consumed = 0.0
        self.total_order_fulfilled = 0.0
        self.total_theatre_inflow = 0.0
        if self.inventory_buffer_targets:
            op3_rm = float(self.inventory_buffer_targets.get("op3_rm", 0))
            op5_rm = float(self.inventory_buffer_targets.get("op5_rm", 0))
            op9_rations = float(self.inventory_buffer_targets.get("op9_rations", 0))
            if op3_rm > 0:
                self.raw_material_wdc.put(op3_rm)
                self._record_material_availability("raw_material_wdc", op3_rm)
                self._lineage_put("raw_material_wdc", op3_rm, source_stage="strategic")
                self.total_strategic_raw_injected += op3_rm
            if op5_rm > 0:
                self.raw_material_al.put(op5_rm)
                self._record_material_availability("raw_material_al", op5_rm)
                self._lineage_put("raw_material_al", op5_rm, source_stage="strategic")
                self.total_strategic_raw_injected += op5_rm
            if op9_rations > 0:
                self.rations_sb.put(op9_rations)
                self._record_material_availability("rations_sb", op9_rations)
                self._lineage_put("rations_sb", op9_rations, source_stage="strategic")
                self.total_strategic_rations_injected += op9_rations
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
        self._raw_material_in_transit = 0.0

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
        self._contingent_cssu_destination_pending: Optional[str] = None
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
        self,
        order: OrderRecord,
        *,
        risk_ids: tuple[str, ...] = ("R14", "R24"),
    ) -> tuple[float, float | None]:
        """Apply queued quantity-risk attribution to an order.

        Returns ``(period_contribution, earliest_start)`` for the AP/RP/DP
        period logic.  The contribution is deliberately a small indicator-like
        value because the Excel columns behave as risk gates, not as ration
        quantities in the ReT formula.
        """
        contribution = 0.0
        earliest: float | None = None
        for risk_id in risk_ids:
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

    # Track C route physics: a strategic top-up for a node can only arrive while
    # the node's site and its inbound line-of-communication are operational.
    # Op3 (WDC) receives from suppliers at its own site; Op5 (AL) is fed through
    # LOC Op4; Op9 (SB) is fed through LOC Op8. R21 downs sites 3/5/9 directly;
    # R22 downs LOCs 4/8.
    _BUFFER_ROUTE_REQUIREMENTS: dict[str, tuple[int, ...]] = {
        "op3_rm": (3,),
        "op5_rm": (4, 5),
        "op9_rations": (8, 9),
    }

    def _buffer_route_open(self, key: str) -> bool:
        if not self.replenishment_route_aware:
            return True
        required = self._BUFFER_ROUTE_REQUIREMENTS.get(key, ())
        return all(self.op_down_count.get(op, 0) == 0 for op in required)

    def _route_aware_top_up_when_open(self, key: str):
        """Hold a blocked top-up until the route reopens, then deliver.

        Delivers to the LATEST target for `key` (order-up-to at arrival), so
        stacked requests during a long outage collapse into one delivery.
        """
        try:
            while not self._buffer_route_open(key):
                if self.env.now >= self.horizon:
                    return
                yield self.env.timeout(24.0)
            target = float(self.inventory_buffer_targets.get(key, 0.0))
            event = self._deliver_buffer_top_up(key, target)
            if event is not None:
                yield event
        finally:
            self._route_wait_pending.discard(key)

    def _deliver_buffer_top_up(self, key: str, target: float) -> Any:
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
            if key in {"op3_rm", "op5_rm"}:
                self.total_strategic_raw_injected += shortfall
            elif key == "op9_rations":
                self.total_strategic_rations_injected += shortfall
            lineage_node = {
                "op3_rm": "raw_material_wdc",
                "op5_rm": "raw_material_al",
                "op9_rations": "rations_sb",
            }[key]
            self._record_material_availability(lineage_node, shortfall)
            self._lineage_put(lineage_node, shortfall, source_stage="strategic_top_up")
            return container.put(shortfall)
        return None

    def _top_up_inventory_buffer(self, key: str, target: float) -> Any:
        if self.replenishment_route_aware and not self._buffer_route_open(key):
            if key not in self._route_wait_pending:
                self._route_wait_pending.add(key)
                self.env.process(self._route_aware_top_up_when_open(key))
            return None
        return self._deliver_buffer_top_up(key, target)

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
        if self.assembly_flow_mode == "serial_wip":
            self.env.process(self._assembly_serial_wip_hourly())
        else:
            self.env.process(self._assembly_hourly())  # HOURLY granularity
        if self.op8_dispatch_mode == "finite_convoy_v1":
            self.env.process(self._op8_finite_convoy_warmup_controller())
            self.env.process(self._op8_convoy_accounting())
        else:
            self.env.process(self._op8_transport_to_sb())
        if self.order_fulfillment_mode == "legacy_theatre_stock":
            self.env.process(self._op9_sb_dispatch())
            self.env.process(self._op10_transport_to_cssu())
            self.env.process(self._op12_transport_to_theatre())
        else:
            # op9_linked: outbound service is rate-limited to ONE order-batch
            # per day (Fig 6.2 / Table 6.20: Op9 'daily freight rate', ROP=24h).
            self.env.process(self._op9_daily_freight_dispatch())
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
                    affected_cssu=(
                        str(row["affected_cssu"])
                        if row.get("affected_cssu") in {"A", "B"}
                        else None
                    ),
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
            if self.cssu_topology_mode == "split_v1":
                self._contingent_cssu_destination_pending = event.affected_cssu
            replayed = RiskEvent(
                risk_id,
                start,
                start + duration,
                duration,
                affected_ops or [13],
                event.description,
                magnitude=surge,
                unit=event.unit or "rations",
                affected_cssu=event.affected_cssu,
            )
            self.risk_events.append(replayed)
            self._add_ret_quantity_risk(replayed)
            return

        if risk_id == "R14":
            # Exogenous CRN replay: apply the recorded defect quantity to the
            # physical pending batch, not merely to the ReT attribution ledger.
            # The tape is materialized once from the reference campaign so every
            # policy faces the same defect count at the same post-warmup time.
            defects = min(max(0, int(round(float(event.magnitude)))), int(self._pending_batch))
            if defects > 0:
                self._pending_batch -= defects
                self.total_produced -= defects
                defect_lineage = self._lineage_take("pending_batch", defects)
                if self.r14_defect_mode == "thesis_strict_op6":
                    yield self.rework_op6.put(defects)
                    self._lineage_forward(
                        "rework_op6", defect_lineage, source_stage="r14_tape_rework"
                    )
                elif self.r14_defect_mode == "reprocess":
                    yield self.raw_material_al.put(defects)
                    self._lineage_forward(
                        "raw_material_al", defect_lineage, source_stage="r14_tape_reprocess"
                    )
                else:
                    self.total_rations_scrapped += float(defects)
            replayed = RiskEvent(
                risk_id,
                start,
                start,
                0.0,
                affected_ops or [7],
                event.description,
                magnitude=float(defects),
                unit=event.unit or "defective_products",
            )
            self.risk_events.append(replayed)
            self._add_ret_quantity_risk(replayed)
            return

        if duration > 0.0 and affected_ops:
            local = (
                self.cssu_topology_mode == "split_v1"
                and event.affected_cssu in {"A", "B"}
            )
            for op_id in affected_ops:
                if local and op_id in {10, 11, 12}:
                    self._take_down_cssu(op_id, event.affected_cssu)
                else:
                    self._take_down(op_id)
            yield self.env.timeout(duration)
            for op_id in affected_ops:
                if local and op_id in {10, 11, 12}:
                    self._bring_up_cssu(op_id, event.affected_cssu)
                else:
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
                    affected_cssu=event.affected_cssu,
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
                self.total_strategic_rations_injected += release_qty
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

    # =====================================================================
    # PROGRAM V2: CONSERVATIVE DOWNSTREAM EMERGENCY RESERVE (OPT-IN)
    # =====================================================================

    def _account_emergency_reserve_inventory_time(self) -> None:
        """Accumulate actual stored unit-hours since the last stock change."""
        now = float(self.env.now)
        elapsed = max(0.0, now - self._emergency_reserve_last_accounting_time)
        self.emergency_reserve_inventory_time += (
            elapsed * float(self.emergency_theatre_reserve.level)
        )
        self._emergency_reserve_last_accounting_time = now

    def configure_emergency_theatre_reserve(
        self,
        *,
        capacity: float,
        initial_stock: float = 0.0,
        target: float | None = None,
        replenishment_lead_time: float = 336.0,
        issue_delay: float = 24.0,
        route_ops: tuple[int, ...] = (10, 11, 12),
        transport_mode: str = "fixed_lead",
    ) -> None:
        """Enable a finite reserve positioned behind the downstream corridor.

        Initial stock is an explicitly costed strategic source.  Subsequent
        replenishment is stock-conserving: units are removed from Op9/SB,
        travel with a real lead, and cannot arrive while any required route
        operation is unavailable.
        """
        if self.emergency_reserve_enabled:
            raise RuntimeError("Emergency reserve may be configured only once.")
        capacity = float(capacity)
        initial_stock = float(initial_stock)
        if capacity <= 0.0:
            raise ValueError("Emergency reserve capacity must be positive.")
        if not 0.0 <= initial_stock <= capacity:
            raise ValueError("initial_stock must be within [0, capacity].")
        route_ops = tuple(int(op) for op in route_ops)
        if not route_ops or any(op < 1 or op > 13 for op in route_ops):
            raise ValueError("route_ops must contain valid operation ids.")
        if transport_mode not in {"fixed_lead", "physical_downstream"}:
            raise ValueError(
                "transport_mode must be fixed_lead or physical_downstream."
            )
        self.emergency_reserve_enabled = True
        self.emergency_reserve_capacity = capacity
        self.emergency_reserve_target = min(
            capacity, initial_stock if target is None else max(0.0, float(target))
        )
        self.emergency_reserve_replenishment_lead_time = max(
            0.0, float(replenishment_lead_time)
        )
        self.emergency_reserve_issue_delay = max(0.0, float(issue_delay))
        self.emergency_reserve_route_ops = route_ops
        self.emergency_reserve_transport_mode = str(transport_mode)
        self._emergency_reserve_last_accounting_time = float(self.env.now)
        if initial_stock > 0.0:
            self.emergency_theatre_reserve.put(initial_stock)
            self.total_strategic_rations_injected += initial_stock

    def _emergency_corridor_down(self) -> bool:
        return any(self._is_down(op) for op in self.emergency_reserve_route_ops)

    def request_emergency_reserve_target(self, target: float) -> None:
        """Set an order-up-to target and launch at most the missing transfer."""
        if not self.emergency_reserve_enabled:
            raise RuntimeError("Emergency reserve is not enabled.")
        clipped = min(self.emergency_reserve_capacity, max(0.0, float(target)))
        if abs(clipped - self.emergency_reserve_target) > 1e-9:
            self.emergency_reserve_target_changes += 1
        self.emergency_reserve_target = clipped
        positioned = (
            float(self.emergency_theatre_reserve.level)
            + float(self.emergency_reserve_in_transit)
        )
        missing = max(0.0, clipped - positioned)
        if missing > 1e-9:
            self.emergency_reserve_in_transit += missing
            self.emergency_reserve_replenishment_requests += 1
            self.env.process(self._replenish_emergency_reserve(missing))

    def enable_program_f_reserve(self) -> None:
        """Authorize stock-conserving partial response to contingent demand."""
        if not self.emergency_reserve_enabled:
            raise RuntimeError("Program F reserve requires configured physical stock.")
        self.program_f_reserve_enabled = True

    def set_program_f_r24_issue_quota(self, quantity: float) -> None:
        """Set the maximum reserve fragment available to the next R24 order."""
        if not self.program_f_reserve_enabled:
            raise RuntimeError("Program F reserve is not enabled.")
        self.program_f_r24_issue_remaining = max(0.0, float(quantity))

    def _deliver_program_f_reserve_fragment(self, order: OrderRecord, qty: float):
        yield self.env.timeout(self.emergency_reserve_issue_delay)
        self._in_transit -= qty
        self.total_delivered += qty
        self.total_order_fulfilled += qty
        self.emergency_reserve_units_issued += qty
        self.program_f_reserve_fragments_issued += qty
        self.delivery_events.append((self.env.now, qty))
        order.in_flight_qty = max(0.0, float(order.in_flight_qty) - qty)
        if order.remaining_qty <= 1e-9 and order.in_flight_qty <= 1e-9:
            self._remove_pending_backorder(order)
            self._finalize_pending_backorder(order)

    def _replenish_emergency_reserve(self, requested_qty: float):
        """Move existing SB stock across the threatened corridor conservatively."""
        qty = float(requested_qty)
        # Do not dispatch into a closed corridor, and never create source stock.
        while self._emergency_corridor_down() or self.rations_sb.level + 1e-9 < qty:
            yield self.env.timeout(1.0)
        yield self.rations_sb.get(qty)
        self._in_transit += qty
        if self.emergency_reserve_transport_mode == "physical_downstream":
            # Mirror the order path rather than inventing an administrative
            # lead: Op10 transit (24 h), Op11 availability, then Op12 transit
            # (24 h).  Any pre-leg outage delays that leg endogenously.
            while self._is_down(10):
                yield self.env.timeout(1.0)
            yield self.env.timeout(self._pt("op10_pt"))
            while self._is_down(11) or self._is_down(12):
                yield self.env.timeout(1.0)
            yield self.env.timeout(self._pt("op12_pt"))
        else:
            yield self.env.timeout(self.emergency_reserve_replenishment_lead_time)
            while self._emergency_corridor_down():
                yield self.env.timeout(1.0)
        self._in_transit -= qty
        self.emergency_reserve_in_transit = max(
            0.0, self.emergency_reserve_in_transit - qty
        )
        # A target may have been lowered during transit.  Excess is retained
        # physically and remains costed; it is never deleted from the model.
        self._account_emergency_reserve_inventory_time()
        yield self.emergency_theatre_reserve.put(qty)
        self.emergency_reserve_units_replenished += qty
        self.total_theatre_inflow += qty

    def _fulfill_from_emergency_reserve(self, order: OrderRecord, qty: float):
        """Issue already-positioned stock to theatre demand after a local delay."""
        yield self.env.timeout(self.emergency_reserve_issue_delay)
        self._in_transit -= qty
        self.total_delivered += qty
        self.total_order_fulfilled += qty
        self.emergency_reserve_units_issued += qty
        self.delivery_events.append((self.env.now, qty))
        self._finalize_pending_backorder(order)

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
            requested_priority_rule = action.get("backorder_priority_rule")
            if requested_priority_rule is not None:
                self.set_backorder_priority_rule(str(requested_priority_rule))
            for k, v in action.items():
                if k == "backorder_priority_rule":
                    continue
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
                    "new_theatre_inflow": 0.0,
                    "new_order_fulfilled": 0.0,
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
                    "flow_ledger": self.flow_ledger(),
                },
            )

        prev_backorders = self.total_backorders
        prev_delivered = self.total_delivered
        prev_order_fulfilled = self.total_order_fulfilled
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
        new_order_fulfilled = self.total_order_fulfilled - prev_order_fulfilled
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
            "new_theatre_inflow": new_delivered,
            "new_order_fulfilled": new_order_fulfilled,
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
            "campaign_state": (
                self.campaign_state_at(float(self.env.now))
                if self.campaign_config
                else None
            ),
            "flow_ledger": self.flow_ledger(),
        }
        return obs, reward, done, info

    def _inventory_detail(self) -> dict[str, float]:
        """Return a named snapshot of all tracked material buffers."""
        detail = {
            "raw_material_wdc": float(self.raw_material_wdc.level),
            "raw_material_al": float(self.raw_material_al.level),
            "rework_op6": float(self.rework_op6.level),
            "wip_op5_op6": float(self.wip_op5_op6.level),
            "wip_op6_op7": float(self.wip_op6_op7.level),
            "rations_al": float(self.rations_al.level),
            "rations_sb": float(self.rations_sb.level),
            "rations_sb_dispatch": float(self.rations_sb_dispatch.level),
            "rations_cssu": float(self.rations_cssu.level),
            "rations_theatre": float(self.rations_theatre.level),
        }
        if self.emergency_reserve_enabled:
            detail["emergency_theatre_reserve"] = float(
                self.emergency_theatre_reserve.level
            )
        if self.cssu_topology_mode == "split_v1":
            detail.update(
                {
                    "cssu_A_in_transit": float(self.cssu_in_transit["A"]),
                    "cssu_B_in_transit": float(self.cssu_in_transit["B"]),
                    "cssu_A_inventory": float(self.cssu_inventory["A"]),
                    "cssu_B_inventory": float(self.cssu_inventory["B"]),
                    "cssu_A_inbound": float(self.cssu_inbound_in_transit["A"]),
                    "cssu_B_inbound": float(self.cssu_inbound_in_transit["B"]),
                    "cssu_A_outbound": float(self.cssu_outbound_in_transit["A"]),
                    "cssu_B_outbound": float(self.cssu_outbound_in_transit["B"]),
                }
            )
        return detail

    def flow_ledger(self) -> dict[str, float]:
        """Return auditable raw-material and ration conservation residuals.

        Positive residuals mean unaccounted source material; negative values
        mean the modeled stocks/sinks exceed recorded sources.  The default
        thesis-strict R14 mode should remain at numerical zero apart from
        floating-point noise.
        """
        raw_sources = (
            float(self.total_external_raw_material)
            + float(self.total_strategic_raw_injected)
        )
        raw_stock = (
            float(self.raw_material_wdc.level)
            + float(self.raw_material_al.level)
            + float(self._raw_material_in_transit)
        )
        raw_sinks = float(self.total_raw_material_consumed) + raw_stock

        ration_sources = (
            float(self.total_rations_created_from_raw)
            + float(self.total_strategic_rations_injected)
        )
        ration_stock = (
            float(self.rework_op6.level)
            + float(self.wip_op5_op6.level)
            + float(self.wip_op6_op7.level)
            + float(self._pending_batch)
            + float(self.rations_al.level)
            + float(self.rations_sb.level)
            + float(self.rations_sb_dispatch.level)
            + float(self.rations_cssu.level)
            + float(self.rations_theatre.level)
            + float(self.emergency_theatre_reserve.level)
            + float(self._in_transit)
        )
        ration_sinks = (
            float(self.total_order_fulfilled)
            + float(self.total_rations_scrapped)
            + ration_stock
        )
        return {
            "raw_sources": raw_sources,
            "raw_consumed": float(self.total_raw_material_consumed),
            "raw_stock_and_transit": raw_stock,
            "raw_residual": raw_sources - raw_sinks,
            "ration_sources": ration_sources,
            "ration_order_fulfilled": float(self.total_order_fulfilled),
            "ration_theatre_inflow": float(self.total_theatre_inflow),
            "ration_scrapped": float(self.total_rations_scrapped),
            "ration_stock_and_transit": ration_stock,
            "ration_residual": ration_sources - ration_sinks,
        }

    def emergency_reserve_metrics(self) -> dict[str, float]:
        """Return resource metrics without mutating the physical state."""
        self._account_emergency_reserve_inventory_time()
        return {
            "emergency_reserve_level": float(self.emergency_theatre_reserve.level),
            "emergency_reserve_target": float(self.emergency_reserve_target),
            "emergency_reserve_in_transit": float(
                self.emergency_reserve_in_transit
            ),
            "emergency_reserve_capacity": float(self.emergency_reserve_capacity),
            "emergency_reserve_inventory_time": float(
                self.emergency_reserve_inventory_time
            ),
            "emergency_reserve_units_issued": float(
                self.emergency_reserve_units_issued
            ),
            "emergency_reserve_units_replenished": float(
                self.emergency_reserve_units_replenished
            ),
            "emergency_reserve_replenishment_requests": float(
                self.emergency_reserve_replenishment_requests
            ),
            "emergency_reserve_target_changes": float(
                self.emergency_reserve_target_changes
            ),
            "emergency_reserve_transport_mode_physical": float(
                self.emergency_reserve_transport_mode == "physical_downstream"
            ),
        }

    def _backorder_priority_key(
        self, order: OrderRecord
    ) -> tuple[int, float, float, int]:
        """
        Return the backlog priority key for the active rationing rule.

        The default rule ``spt_contingent`` reproduces Garrido's thesis rule
        bitwise: contingent demand takes precedence over regular demand, and
        within each priority class delayed orders are sorted in increasing
        order of size (an SPT proxy). ``backorder_priority_rule`` (Program D,
        lever D1) swaps this key among alternative rationing rules to probe
        whether the *sequencing* decision — who is served today and who is
        shed on cap-60 overflow — carries state-contingent value. Every rule
        is a pure re-ordering of the same standing queue; no physics changes.
        """
        rule = self.backorder_priority_rule
        contingent_class = 0 if order.contingent else 1
        qty = float(order.remaining_qty)
        age_key = float(order.OPTj)  # earlier OPTj == older == served first (FIFO)
        tiebreak = int(order.j)
        if rule == "spt_contingent":
            return (contingent_class, qty, age_key, tiebreak)
        if rule == "fifo_contingent":
            return (contingent_class, age_key, qty, tiebreak)
        if rule == "lpt_contingent":
            return (contingent_class, -qty, age_key, tiebreak)
        if rule == "spt_flat":
            return (0, qty, age_key, tiebreak)
        if rule == "fifo_flat":
            return (0, age_key, qty, tiebreak)
        if rule == "age_threshold":
            # Orders that have waited past the threshold jump ahead of the SPT
            # ordering (still contingent-first); an explicit age-priority rule.
            aged = 0 if (float(self.env.now) - age_key) >= self.backorder_age_threshold_hours else 1
            return (contingent_class, aged, qty, age_key)
        raise ValueError(f"Unhandled backorder_priority_rule={rule!r}.")

    def _backorder_queue_hash(self) -> str:
        """Stable hash of queue identity/order for Program D provenance."""
        payload = "\n".join(
            f"{int(order.j)}|{float(order.OPTj):.12g}|"
            f"{float(order.remaining_qty):.12g}|{int(bool(order.contingent))}|"
            f"{int(bool(order.lost))}"
            for order in self.pending_backorders
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def set_backorder_priority_rule(self, rule: str) -> dict[str, Any]:
        """Atomically change only the Op9 queue sequencing rule.

        The standing queue is re-sorted immediately. No order is created,
        removed, fulfilled, or marked lost, and no random generator is touched.
        """
        if rule not in BACKORDER_PRIORITY_RULE_OPTIONS:
            valid = ", ".join(BACKORDER_PRIORITY_RULE_OPTIONS)
            raise ValueError(
                f"Invalid backorder_priority_rule={rule!r}. Expected: {valid}."
            )
        previous = self.backorder_priority_rule
        before_ids = tuple(int(order.j) for order in self.pending_backorders)
        before_hash = self._backorder_queue_hash()
        self.backorder_priority_rule = rule
        self.pending_backorders.sort(key=self._backorder_priority_key)
        self._refresh_pending_backorder_qty()
        after_ids = tuple(int(order.j) for order in self.pending_backorders)
        event = {
            "time": float(self.env.now),
            "previous_rule": previous,
            "new_rule": rule,
            "queue_hash_before": before_hash,
            "queue_hash_after": self._backorder_queue_hash(),
            "queue_size": len(after_ids),
            "queue_membership_unchanged": sorted(before_ids) == sorted(after_ids),
        }
        if not event["queue_membership_unchanged"]:
            raise AssertionError("Priority-rule change altered queue membership.")
        self.backorder_priority_rule_events.append(event)
        return event

    def _refresh_pending_backorder_qty(self) -> None:
        """Recompute the outstanding delayed-demand quantity."""
        self.pending_backorder_qty = float(
            sum(
                max(0.0, float(order.remaining_qty))
                for order in self.pending_backorders
            )
        )

    # ------------------------------------------------------- causal exposure
    def _record_material_availability(self, node: str, quantity: float) -> None:
        """Record when quantity first becomes physically available at a node."""
        qty = float(quantity)
        if qty > 1e-9:
            self.material_availability_events[node].append((float(self.env.now), qty))

    def _lineage_event_ref(self, event: RiskEvent) -> str:
        return f"{event.risk_id}@{float(event.start_time):.9f}"

    def _lineage_put(
        self,
        node: str,
        quantity: float,
        *,
        risk_event_refs: Iterable[str] = (),
        source_stage: str,
        lot_id: str | None = None,
    ) -> None:
        """Mirror a physical put in the opt-in FIFO genealogy ledger."""
        if self.material_lineage_mode != "tagged_lots" or quantity <= 1e-9:
            return
        self._material_lineage_sequence += 1
        self._material_lineage[node].append(
            MaterialLineageSlice(
                quantity=float(quantity),
                lot_id=lot_id or f"L{self._material_lineage_sequence:09d}",
                risk_event_refs=tuple(sorted(set(risk_event_refs))),
                source_stage=str(source_stage),
                created_at=float(self.env.now),
            )
        )

    def _lineage_take(
        self, node: str, quantity: float, *, output_scale: float = 1.0
    ) -> list[MaterialLineageSlice]:
        """FIFO-split lineage exactly as the corresponding container is drawn."""
        if self.material_lineage_mode != "tagged_lots" or quantity <= 1e-9:
            return []
        remaining = float(quantity)
        consumed: list[MaterialLineageSlice] = []
        queue = self._material_lineage[node]
        while remaining > 1e-8 and queue:
            head = queue[0]
            used = min(remaining, float(head.quantity))
            consumed.append(
                MaterialLineageSlice(
                    quantity=used * float(output_scale),
                    lot_id=head.lot_id,
                    risk_event_refs=head.risk_event_refs,
                    source_stage=head.source_stage,
                    created_at=head.created_at,
                )
            )
            head.quantity -= used
            remaining -= used
            if head.quantity <= 1e-8:
                queue.pop(0)
        if remaining > 1e-6:
            raise AssertionError(
                f"Lineage underflow at {node}: missing {remaining} of {quantity}."
            )
        return consumed

    def _lineage_forward(
        self,
        node: str,
        slices: Iterable[MaterialLineageSlice],
        *,
        source_stage: str,
        extra_refs: Iterable[str] = (),
    ) -> None:
        extras = set(extra_refs)
        for item in slices:
            self._lineage_put(
                node,
                item.quantity,
                risk_event_refs=set(item.risk_event_refs) | extras,
                source_stage=source_stage,
                lot_id=item.lot_id,
            )

    def _consume_pending_stage_refs(self, stage: str) -> tuple[str, ...]:
        if self.material_lineage_mode != "tagged_lots":
            return ()
        refs = tuple(self._pending_lineage_events_by_stage[stage])
        self._pending_lineage_events_by_stage[stage].clear()
        return refs

    def _lineage_snapshot(self) -> dict[str, float]:
        """Diagnostic quantity totals for conservation tests and audit artifacts."""
        return {
            node: float(sum(item.quantity for item in slices))
            for node, slices in self._material_lineage.items()
        }

    def _record_queue_len(self) -> None:
        """Append (time, backlog length) to the queue history (monotone time)."""
        history = self._queue_len_history
        now = float(self.env.now)
        n = len(self.pending_backorders)
        if history and history[-1][0] == now:
            history[-1] = (now, n)
        else:
            history.append((now, n))
        # An exposure cached while open must be recomputed after the queue
        # changes; otherwise its first observed ``env.now`` becomes a false
        # permanent closing time.
        self._exposure_end_cache.clear()

    def _record_causal_block(
        self,
        order: OrderRecord,
        *,
        op_id: int,
        start_time: float,
        end_time: float,
        reason: str,
        event: RiskEvent | None = None,
    ) -> None:
        """Record an order-specific physical blocking interval."""
        start = float(start_time)
        end = float(end_time)
        if end <= start:
            return
        block = {
            "op_id": int(op_id),
            "start_time": start,
            "end_time": end,
            "reason": str(reason),
        }
        if event is not None:
            block.update(
                {
                    "risk_id": str(event.risk_id),
                    "risk_event_start": float(event.start_time),
                    "risk_event_end": float(event.end_time),
                    "propagation_source": "upstream_scarcity_debt",
                }
            )
        if order.causal_block_intervals and order.causal_block_intervals[-1] == block:
            return
        order.causal_block_intervals.append(block)

    def _record_active_upstream_stockout_causes(
        self, order: OrderRecord, *, duration: float
    ) -> None:
        """Record only operations physically down while the head lacks stock."""
        start = float(self.env.now)
        end = start + max(0.0, float(duration))
        for op_id in range(1, 10):
            if self._is_down(op_id):
                self._record_causal_block(
                    order,
                    op_id=op_id,
                    start_time=start,
                    end_time=end,
                    reason="risk_induced_stockout",
                )
        for event in self._upstream_scarcity_debts:
            upstream_ops = [
                int(op_id) for op_id in event.affected_ops if int(op_id) <= 8
            ]
            if not upstream_ops:
                continue
            self._record_causal_block(
                order,
                op_id=upstream_ops[0],
                start_time=start,
                end_time=end,
                reason="risk_induced_stockout",
                event=event,
            )
            ref = self._lineage_event_ref(event)
            if not any(row.get("event_ref") == ref for row in order.lineage_shortage_refs):
                order.lineage_shortage_refs.append(
                    {
                        "event_ref": ref,
                        "risk_id": event.risk_id,
                        "observed_at": start,
                        "blocked_hours": end - start,
                    }
                )

    def _register_upstream_scarcity_debt(self, event: RiskEvent) -> None:
        if self.risk_attribution_source != "causal_exposure":
            return
        if not any(int(op_id) <= 8 for op_id in event.affected_ops):
            return
        self._upstream_scarcity_debts.append(event)
        if self.material_lineage_mode != "tagged_lots":
            return
        ref = self._lineage_event_ref(event)
        self._lineage_event_index[ref] = event
        earliest_op = min(int(op_id) for op_id in event.affected_ops)
        if earliest_op <= 2:
            stage = "op2_output"
        elif earliest_op <= 4:
            stage = "op4_output"
        elif earliest_op <= 7:
            stage = "op7_output"
        else:
            stage = "op8_output"
        pending = self._pending_lineage_events_by_stage[stage]
        if ref not in pending:
            pending.append(ref)

    def _match_block_to_events(
        self, block: dict[str, Any]
    ) -> list[tuple[RiskEvent, float]]:
        """Match a physical block only to events affecting that operation."""
        op_id = int(block["op_id"])
        block_start = float(block["start_time"])
        block_end = float(block["end_time"])
        matches: list[tuple[RiskEvent, float]] = []
        explicit_risk = block.get("risk_id")
        explicit_start = block.get("risk_event_start")
        for event in self.risk_events:
            if explicit_risk is not None:
                if str(event.risk_id) != str(explicit_risk) or not math.isclose(
                    float(event.start_time), float(explicit_start), abs_tol=1e-9
                ):
                    continue
                matches.append((event, block_end - block_start))
                continue
            if op_id not in {int(value) for value in event.affected_ops}:
                continue
            event_start = float(event.start_time)
            event_end = max(event_start, float(event.end_time))
            overlap_start = max(block_start, event_start)
            overlap_end = min(block_end, event_end)
            if overlap_end > overlap_start:
                matches.append((event, overlap_end - overlap_start))
        return matches

    def _refresh_r24_episode_closure(self) -> None:
        if self.risk_attribution_source != "causal_exposure":
            return
        unresolved = [
            order
            for order in self.orders
            if order.OATj is None and not bool(order.lost)
        ] + list(self.pending_backorders)
        for episode_id, episode in self._r24_causal_episodes.items():
            if episode.get("closed_at") is not None:
                continue
            if episode.get("assigned_order_j") is None:
                continue
            if any(
                episode_id in order.causal_r24_event_ids
                for order in unresolved
            ):
                continue
            episode["closed_at"] = float(self.env.now)

    def _queue_len_at(self, t: float) -> int:
        """Backlog length just before time t (0 before the first sample)."""
        history = self._queue_len_history
        lo, hi = 0, len(history)
        while lo < hi:
            mid = (lo + hi) // 2
            if history[mid][0] < float(t):
                lo = mid + 1
            else:
                hi = mid
        return history[lo - 1][1] if lo > 0 else 0

    def _exposure_end_for(self, event: RiskEvent) -> float:
        """Endogenous exposure end: the disruption's effect persists until the
        order backlog returns to its pre-event level.

        Garrido's per-order risk columns mark ~3x more orders per event than
        the raw outage duration explains (CF11: R22 share 0.291 vs 0.108 with
        matching event counts) — his exposure covers the induced-backlog
        recovery period, not just the outage. The same rule closes an R24
        surge exposure when the contingent demand and the queue it displaced
        have been absorbed. Computed lazily from the queue-length history and
        cached per event id.
        """
        key = id(event)
        cached = self._exposure_end_cache.get(key)
        if cached is not None:
            return cached
        baseline = self._queue_len_at(float(event.start_time))
        end = float(event.end_time)
        history = self._queue_len_history
        # First history point at/after the raw event end with qlen <= baseline.
        lo, hi = 0, len(history)
        t0 = end
        while lo < hi:
            mid = (lo + hi) // 2
            if history[mid][0] < t0:
                lo = mid + 1
            else:
                hi = mid
        exposure_end = float(self.env.now)  # still open if never relieved
        resolved = False
        for i in range(lo, len(history)):
            if history[i][1] <= baseline:
                exposure_end = max(end, history[i][0])
                resolved = True
                break
        if resolved:
            self._exposure_end_cache[key] = exposure_end
        return exposure_end

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
        self._record_queue_len()
        self._refresh_r24_episode_closure()

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
        if self.risk_attribution_source == "causal_exposure":
            prior_episode_ids: set[str] = set()
            for queued_order in self.pending_backorders:
                if prior_episode_ids:
                    queued_order.causal_r24_event_ids.update(prior_episode_ids)
                prior_episode_ids.update(queued_order.causal_r24_event_ids)
        self._record_queue_len()
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
            dropped.lost_time = float(self.env.now)
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
        self._refresh_r24_episode_closure()

    def _delayed_backorder_check(self, order: OrderRecord):
        """Wait LTj hours, then classify as backorder if still unfulfilled.

        Per thesis Sec. 6.8.2: backorder = order not delivered within the
        pre-set lead time of 48 hours. Orders fulfilled within LTj are
        on-time (not backorders).
        """
        yield self.env.timeout(order.LTj)
        still_late = (
            order.OATj is None
            if self.order_fulfillment_mode == "op9_linked"
            else order.remaining_qty > 0
        )
        if still_late:
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

        In op9_linked mode this is a no-op: the ONLY server is the daily
        freight departure (`_op9_daily_freight_dispatch`), per the Table 6.20
        'daily freight rate' constraint.
        """
        if self.order_fulfillment_mode == "op9_linked":
            return
        while self.pending_backorders:
            next_order = self.pending_backorders[0]
            requested_qty = float(next_order.remaining_qty)
            if requested_qty <= 1e-9:
                # Order already fully served or numerically empty.
                self._finalize_pending_backorder(next_order)
                self._remove_pending_backorder(next_order)
                continue
            source = (
                self.rations_sb
                if self.order_fulfillment_mode == "op9_linked"
                else self.rations_theatre
            )
            if source.level + 1e-9 < requested_qty:
                break
            yield source.get(requested_qty)
            if self.order_fulfillment_mode == "op9_linked":
                next_order.remaining_qty = 0.0
                self.env.process(self._deliver_order_from_op9(next_order, requested_qty))
            else:
                self.total_order_fulfilled += requested_qty
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

    def get_cssu_observation(self) -> dict[str, float]:
        """Return deployable current/past split-state features for DRA-1.

        The mapping intentionally exposes no risk schedule, future repair,
        latent regime, or future demand field. A later environment wrapper can
        freeze ordering/normalization without changing this information set.
        """
        if self.cssu_topology_mode != "split_v1":
            raise RuntimeError("CSSU observations require cssu_topology_mode='split_v1'.")
        now = float(self.env.now)
        obs: dict[str, float] = {
            "sb_inventory": float(self.rations_sb.level),
            "allocation_a": float(self.cssu_allocation_a),
            "service_rule_spt_full": float(self.cssu_service_rule == "SPT_FULL"),
            "service_rule_fifo_partial": float(
                self.cssu_service_rule == "FIFO_PARTIAL"
            ),
            "service_rule_r24_age_partial": float(
                self.cssu_service_rule == "R24_AGE_PARTIAL"
            ),
            "day_phase": float((now % HOURS_PER_WEEK) / HOURS_PER_WEEK),
        }
        for cssu in ("A", "B"):
            orders = [
                order
                for order in self.pending_backorders
                if order.cssu_destination == cssu
            ]
            backlog = sum(float(order.remaining_qty) for order in orders)
            ages = [max(0.0, now - float(order.OPTj)) for order in orders]
            contingent = sum(
                float(order.remaining_qty) for order in orders if order.contingent
            )
            demand_1d = sum(
                qty
                for time, destination, qty in self.cssu_demand_events
                if destination == cssu and time >= now - HOURS_PER_DAY
            )
            demand_7d = sum(
                qty
                for time, destination, qty in self.cssu_demand_events
                if destination == cssu and time >= now - HOURS_PER_WEEK
            )
            delivered_7d = sum(
                qty
                for time, destination, qty in self.cssu_delivery_events
                if destination == cssu and time >= now - HOURS_PER_WEEK
            )
            obs.update(
                {
                    f"cssu_{cssu}_inventory": float(self.cssu_inventory[cssu]),
                    f"cssu_{cssu}_inbound": float(
                        self.cssu_inbound_in_transit[cssu]
                    ),
                    f"cssu_{cssu}_outbound": float(
                        self.cssu_outbound_in_transit[cssu]
                    ),
                    f"cssu_{cssu}_backlog_qty": float(backlog),
                    f"cssu_{cssu}_backlog_count": float(len(orders)),
                    f"cssu_{cssu}_max_age": float(max(ages) if ages else 0.0),
                    f"cssu_{cssu}_r24_share": float(
                        contingent / backlog if backlog > 0 else 0.0
                    ),
                    f"cssu_{cssu}_demand_1d": float(demand_1d),
                    f"cssu_{cssu}_demand_7d": float(demand_7d),
                    f"cssu_{cssu}_delivered_7d": float(delivered_7d),
                    f"cssu_{cssu}_op10_up": float(
                        not self._is_cssu_path_down(10, cssu)
                    ),
                    f"cssu_{cssu}_op11_up": float(
                        not self._is_cssu_path_down(11, cssu)
                    ),
                    f"cssu_{cssu}_op12_up": float(
                        not self._is_cssu_path_down(12, cssu)
                    ),
                }
            )
        return obs

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

    def _is_cssu_path_down(self, op_id: int, cssu: Optional[str]) -> bool:
        if self._is_down(op_id):
            return True
        if self.cssu_topology_mode != "split_v1" or cssu not in {"A", "B"}:
            return False
        return self.cssu_local_down_count.get((int(op_id), cssu), 0) > 0

    def _take_down_cssu(self, op_id: int, cssu: str) -> None:
        key = (int(op_id), str(cssu))
        self.cssu_local_down_count[key] += 1

    def _bring_up_cssu(self, op_id: int, cssu: str) -> None:
        key = (int(op_id), str(cssu))
        self.cssu_local_down_count[key] = max(
            0, self.cssu_local_down_count[key] - 1
        )

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
        """Create and renew the framework contract consumed by Op2.

        The first contract starts at t=0 and completes after Op1 PT=672 h,
        which is required by the thesis warm-up. Each completion authorises
        supplier deliveries for the next Op1 ROP window. R12 can therefore
        delay a renewal and physically block Op2.
        """
        if self.procurement_contract_mode == "legacy_independent":
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

        next_cycle_start = 0.0
        while True:
            if self.env.now < next_cycle_start:
                yield self.env.timeout(next_cycle_start - self.env.now)
            pt_remaining = self._pt("op1_pt")
            while pt_remaining > 0:
                while self._is_down(1):
                    yield self.env.timeout(1)
                yield self.env.timeout(1)
                pt_remaining -= 1
            valid_from = float(self.env.now)
            self.contract_valid_until = max(
                float(self.contract_valid_until),
                valid_from + float(self.params["op1_rop"]),
            )
            self.contract_completion_events.append(
                (valid_from, float(self.contract_valid_until))
            )
            if not self._contract_renewed_event.triggered:
                self._contract_renewed_event.succeed(valid_from)
            self._contract_renewed_event = self.env.event()
            next_cycle_start += float(self.params["op1_rop"])

    def _op2_supplier_delivery(self):
        next_eligible_start = float(self.params["op2_rop"])
        while True:
            clock_mode = (
                self.periodic_release_mode
                if self.op2_release_clock_mode == "inherit"
                else self.op2_release_clock_mode
            )
            if clock_mode in {"start_to_start", "calendar_anchored"}:
                wait = max(0.0, next_eligible_start - float(self.env.now))
                if wait > 0.0:
                    yield self.env.timeout(wait)
            else:
                yield self.env.timeout(self.params["op2_rop"])
            if self.procurement_contract_mode == "causal_coupled":
                while float(self.env.now) >= float(self.contract_valid_until):
                    yield self._contract_renewed_event
            while self._is_down(2):
                yield self.env.timeout(1)
            actual_start = float(self.env.now)
            if clock_mode == "start_to_start":
                next_eligible_start = actual_start + float(self.params["op2_rop"])
            elif clock_mode == "calendar_anchored":
                next_eligible_start += float(self.params["op2_rop"])
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
            self._record_material_availability("raw_material_wdc", total_delivery)
            self._lineage_put(
                "raw_material_wdc",
                total_delivery,
                risk_event_refs=self._consume_pending_stage_refs("op2_output"),
                source_stage="op2_output",
            )
            self.total_external_raw_material += float(total_delivery)
            self.supplier_delivery_events.append(
                (float(self.env.now), float(total_delivery))
            )

    def _op3_wdc_dispatch(self):
        next_eligible_start = 0.0
        while True:
            if self.periodic_release_mode == "start_to_start":
                while (
                    float(self.env.now) < next_eligible_start
                    or float(self.raw_material_wdc.level) <= 0.0
                ):
                    remaining = max(
                        0.0, next_eligible_start - float(self.env.now)
                    )
                    yield self.env.timeout(min(1.0, remaining) if remaining else 1.0)
            else:
                yield self.env.timeout(self.params["op3_rop"])
            while self._is_down(3):
                yield self.env.timeout(1)
            actual_start = float(self.env.now)
            total_dispatch = self.params["op3_q"] * NUM_RAW_MATERIALS
            available = self.raw_material_wdc.level
            if self.raw_material_flow_mode == "bom_total_units_order_up_to":
                target = self._target_for_raw_node("op5_rm", total_dispatch)
                total_dispatch = max(0.0, target - float(self.raw_material_al.level))
            dispatch = min(total_dispatch, available)
            if dispatch > 0:
                yield self.raw_material_wdc.get(dispatch)
                lineage = self._lineage_take("raw_material_wdc", dispatch)
                self._raw_material_in_transit += dispatch
                yield self.env.timeout(self._pt("op3_pt"))
                # Op4: transport WDC → AL (separate operation per thesis)
                while self._is_down(4):
                    yield self.env.timeout(1)
                yield self.env.timeout(self._pt("op4_pt"))
                self._raw_material_in_transit -= dispatch
                yield self.raw_material_al.put(dispatch)
                self._record_material_availability("raw_material_al", dispatch)
                self._lineage_forward(
                    "raw_material_al",
                    lineage,
                    source_stage="op4_output",
                    extra_refs=self._consume_pending_stage_refs("op4_output"),
                )
            if self.periodic_release_mode == "start_to_start":
                next_eligible_start = actual_start + float(self.params["op3_rop"])

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
                    rework_lineage = self._lineage_take("rework_op6", rework_qty)
                    self._lineage_forward(
                        "pending_batch", rework_lineage, source_stage="op6_rework"
                    )
                if raw_units_qty > 0:
                    yield self.raw_material_al.get(raw_units_qty)
                    raw_lineage = self._lineage_take(
                        "raw_material_al",
                        raw_units_qty,
                        output_scale=1.0 / max(self._raw_units_per_ration, 1.0),
                    )
                    self._lineage_forward(
                        "pending_batch",
                        raw_lineage,
                        source_stage="op7_work",
                        extra_refs=self._consume_pending_stage_refs("op7_output"),
                    )
                    self.total_raw_material_consumed += raw_units_qty
                    self.total_rations_created_from_raw += raw_produced_qty
                self._pending_batch += can_produce
                self._today_produced += can_produce
                self.total_produced += can_produce

                # DRA-2 exposes finished Op7 output continuously in staging;
                # the convoy action, not production, decides when it leaves.
                batch_size = self.params["batch_size"]
                if self.op8_dispatch_mode == "finite_convoy_v1":
                    self._pending_batch -= can_produce
                    yield from self._stage_finished_rations(can_produce)
                else:
                    # Historical thesis lane: only complete batches leave Op7.
                    while self._pending_batch >= batch_size:
                        self._pending_batch -= batch_size
                        yield from self._stage_finished_rations(batch_size)

                if (
                    self.warmup_trigger == "production"
                    and self.total_produced >= batch_size
                ):
                    self._mark_warmup_complete()

    def _assembly_serial_wip_hourly(self):
        """Op5 -> Op6 -> Op7 as a balanced line with explicit WIP.

        This is the thesis-faithful *candidate* for testing station-specific
        outage propagation.  Each station can drain its existing upstream WIP
        while another station is down.  Transfers use the inventory available
        at the start of the hourly tick, preventing zero-time passage through
        all three stations.  The aggregate implementation remains the default
        until held-out CF validation supports promotion.
        """
        week_hours = 7 * HOURS_PER_DAY
        while True:
            yield self.env.timeout(1.0)
            hour_in_week = int(self.env.now) % week_hours
            day_of_week = hour_in_week // HOURS_PER_DAY
            hour_of_day = hour_in_week % HOURS_PER_DAY
            shifts = int(self.params["assembly_shifts"])
            if day_of_week >= 6 or hour_of_day >= HOURS_PER_SHIFT * shifts:
                continue

            rate = float(RATIONS_PER_HOUR)
            if self.adaptive_benchmark_enabled:
                penalty = float(
                    ADAPTIVE_BENCHMARK_MAINTENANCE["throughput_penalty_max"]
                )
                rate *= max(0.0, 1.0 - penalty * self.maintenance_debt)

            start_wip56 = float(self.wip_op5_op6.level)
            start_wip67 = float(self.wip_op6_op7.level)
            start_rework = float(self.rework_op6.level)

            op7_qty = 0.0
            if not self._is_down(7):
                op7_qty = min(rate, start_wip67)

            op6_rework = 0.0
            op6_new = 0.0
            if not self._is_down(6):
                free67 = max(
                    0.0, float(self.wip_op6_op7.capacity) - start_wip67
                )
                op6_rework = min(rate, start_rework, free67)
                op6_new = min(
                    max(0.0, rate - op6_rework),
                    start_wip56,
                    max(0.0, free67 - op6_rework),
                )

            op5_qty = 0.0
            if not self._is_down(5):
                raw_ration_capacity = float(self.raw_material_al.level) / max(
                    self._raw_units_per_ration, 1.0
                )
                free56 = max(
                    0.0, float(self.wip_op5_op6.capacity) - start_wip56
                )
                op5_qty = min(rate, raw_ration_capacity, free56)

            if not any(self._is_down(op_id) for op_id in (5, 6, 7)):
                self._cumulative_available_assembly_hours += 1.0
                if self.adaptive_benchmark_enabled:
                    self._apply_maintenance_debt(shifts)

            if op7_qty > 0.0:
                yield self.wip_op6_op7.get(op7_qty)
                self._pending_batch += op7_qty
                self._today_produced += op7_qty
                self.total_produced += op7_qty

            if op6_rework > 0.0:
                yield self.rework_op6.get(op6_rework)
            if op6_new > 0.0:
                yield self.wip_op5_op6.get(op6_new)
            op6_total = op6_rework + op6_new
            if op6_total > 0.0:
                yield self.wip_op6_op7.put(op6_total)

            if op5_qty > 0.0:
                raw_units = op5_qty * self._raw_units_per_ration
                yield self.raw_material_al.get(raw_units)
                raw_lineage = self._lineage_take(
                    "raw_material_al",
                    raw_units,
                    output_scale=1.0 / max(self._raw_units_per_ration, 1.0),
                )
                self._lineage_forward(
                    "pending_batch",
                    raw_lineage,
                    source_stage="op5_output",
                    extra_refs=self._consume_pending_stage_refs("op7_output"),
                )
                self.total_raw_material_consumed += raw_units
                self.total_rations_created_from_raw += op5_qty
                yield self.wip_op5_op6.put(op5_qty)

            batch_size = float(self.params["batch_size"])
            if self.op8_dispatch_mode == "finite_convoy_v1" and op7_qty > 0.0:
                self._pending_batch -= op7_qty
                yield from self._stage_finished_rations(op7_qty)
            else:
                while self._pending_batch >= batch_size:
                    self._pending_batch -= batch_size
                    yield from self._stage_finished_rations(batch_size)

            if (
                self.warmup_trigger == "production"
                and self.total_produced >= batch_size
            ):
                self._mark_warmup_complete()

    # =====================================================================
    # DOWNSTREAM: Distribution (Op8-Op12)
    # =====================================================================

    def _stage_finished_rations(self, qty: float):
        """Move completed Op7 rations into the auditable Op8 staging stock."""
        qty = float(qty)
        was_empty = float(self.rations_al.level) <= 1e-9
        yield self.rations_al.put(qty)
        self._record_material_availability("rations_al", qty)
        batch_lineage = self._lineage_take("pending_batch", qty)
        self._lineage_forward(
            "rations_al", batch_lineage, source_stage="op7_output"
        )
        if was_empty and qty > 0.0:
            self.op8_staging_first_ready_at = float(self.env.now)

    def op8_convoy_dispatch_feasible(self) -> bool:
        """Whether DISPATCH_NOW can create a physical departure this epoch."""
        return bool(
            self.op8_dispatch_mode == "finite_convoy_v1"
            and self.op8_convoy_available
            and not self._is_down(8)
            and float(self.rations_al.level) > 1e-9
        )

    def apply_op8_convoy_action(
        self, action: str | int, *, source: str = "policy"
    ) -> dict[str, Any]:
        """Apply HOLD or DISPATCH_NOW at the current decision epoch."""
        if self.op8_dispatch_mode != "finite_convoy_v1":
            raise RuntimeError("Finite-convoy actions require finite_convoy_v1 mode.")
        if isinstance(action, (int, np.integer)):
            action = "HOLD" if int(action) == 0 else "DISPATCH_NOW" if int(action) == 1 else str(action)
        action = str(action).upper()
        if action not in {"HOLD", "DISPATCH_NOW"}:
            raise ValueError("Op8 convoy action must be HOLD or DISPATCH_NOW.")
        feasible_before = self.op8_convoy_dispatch_feasible()
        event: dict[str, Any] = {
            "time": float(self.env.now),
            "action": action,
            "source": str(source),
            "feasible_before": feasible_before,
            "staged_before": float(self.rations_al.level),
            "convoy_available_before": bool(self.op8_convoy_available),
            "route_up_before": not self._is_down(8),
            "departed": False,
            "quantity": 0.0,
            "mask_reason": "",
        }
        self.op8_last_action = action
        if action == "HOLD":
            self.op8_convoy_hold_actions += 1
            event["mask_reason"] = "intentional_hold"
        else:
            self.op8_convoy_dispatch_actions += 1
            if not feasible_before:
                self.op8_convoy_masked_dispatch_attempts += 1
                if not self.op8_convoy_available:
                    event["mask_reason"] = "convoy_away"
                elif self._is_down(8):
                    event["mask_reason"] = "route_down"
                else:
                    event["mask_reason"] = "empty_staging"
            else:
                qty = min(float(self.rations_al.level), self.op8_convoy_capacity)
                self.op8_convoy_available = False
                self.op8_convoy_last_departure_at = float(self.env.now)
                self.op8_convoy_nominal_return_at = float(self.env.now) + (
                    self.op8_convoy_outbound_hours + self.op8_convoy_return_hours
                )
                self.op8_convoy_departures += 1
                self.op8_convoy_dispatched_rations += qty
                self.op8_convoy_capacity_committed += self.op8_convoy_capacity
                event.update({"departed": True, "quantity": qty})
                self.op8_convoy_departure_events.append(dict(event))
                self.env.process(self._op8_finite_convoy_trip(qty))
        self.op8_convoy_action_events.append(event)
        return event

    def _op8_route_progress(self, hours: float, *, ration_qty: float = 0.0):
        remaining = float(hours)
        while remaining > 1e-9:
            yield self.env.timeout(1.0)
            if self._is_down(8):
                self.op8_convoy_route_wait_hours += 1.0
            else:
                remaining -= 1.0
            if ration_qty > 0.0:
                self.op8_convoy_ration_hours_in_transit += float(ration_qty)

    def _op8_finite_convoy_trip(self, qty: float):
        """Consume one persistent vehicle through outbound and return legs."""
        yield self.rations_al.get(qty)
        lineage = self._lineage_take("rations_al", qty)
        if float(self.rations_al.level) <= 1e-9:
            self.op8_staging_first_ready_at = None
        departed_at = float(self.env.now)
        self._in_transit += qty
        yield from self._op8_route_progress(
            self.op8_convoy_outbound_hours, ration_qty=qty
        )
        self._in_transit -= qty
        yield from self._op8_arrive_sb(qty, lineage, departed_at)
        yield from self._op8_route_progress(self.op8_convoy_return_hours)
        self.op8_convoy_available = True
        self.op8_convoy_actual_return_at = float(self.env.now)
        self.op8_convoy_nominal_return_at = None

    def _op8_convoy_accounting(self):
        while True:
            yield self.env.timeout(1.0)
            if self.op8_convoy_available:
                self.op8_convoy_idle_hours += 1.0
            else:
                self.op8_convoy_vehicle_hours += 1.0

    def _op8_finite_convoy_warmup_controller(self):
        """Use thesis full-load doctrine only until the common warm-up anchor."""
        while not self.warmup_complete:
            if (
                self.op8_convoy_dispatch_feasible()
                and float(self.rations_al.level) + 1e-9 >= self.op8_convoy_capacity
            ):
                self.apply_op8_convoy_action("DISPATCH_NOW", source="warmup")
            yield self.env.timeout(1.0)

    def _op8_arrive_sb(
        self, qty: float, lineage: list[MaterialLineageSlice], departed_at: float
    ):
        yield self.rations_sb.put(qty)
        self._record_material_availability("rations_sb", qty)
        op8_refs = self._consume_pending_stage_refs("op8_output")
        self._lineage_forward(
            "rations_sb", lineage, source_stage="op8_output", extra_refs=op8_refs
        )
        if self.risk_attribution_source == "causal_exposure":
            if self.material_lineage_mode == "tagged_lots":
                arrived_refs = set(op8_refs)
                for item in lineage:
                    arrived_refs.update(item.risk_event_refs)
                self._upstream_scarcity_debts = [
                    event for event in self._upstream_scarcity_debts
                    if self._lineage_event_ref(event) not in arrived_refs
                ]
            else:
                self._upstream_scarcity_debts = [
                    event for event in self._upstream_scarcity_debts
                    if float(event.end_time) > departed_at
                ]
        if self.warmup_trigger == "op9_arrival":
            self._mark_warmup_complete()
        if self.order_fulfillment_mode == "op9_linked" and self.pending_backorders:
            yield from self._serve_pending_backorders()

    def op8_convoy_metrics(self) -> dict[str, float]:
        committed = max(float(self.op8_convoy_capacity_committed), 1.0)
        return {
            "op8_convoy_available": float(self.op8_convoy_available),
            "op8_convoy_departures": float(self.op8_convoy_departures),
            "op8_convoy_dispatched_rations": float(self.op8_convoy_dispatched_rations),
            "op8_convoy_capacity_committed": float(self.op8_convoy_capacity_committed),
            "op8_convoy_load_factor": float(self.op8_convoy_dispatched_rations / committed),
            "op8_convoy_vehicle_hours": float(self.op8_convoy_vehicle_hours),
            # Explicit resource name used by the DRA-2 estimand. This includes
            # outbound, return, and any R22 route wait while the asset is away.
            "op8_convoy_unavailable_hours": float(self.op8_convoy_vehicle_hours),
            "op8_convoy_idle_hours": float(self.op8_convoy_idle_hours),
            "op8_convoy_route_wait_hours": float(self.op8_convoy_route_wait_hours),
            "op8_convoy_ration_hours_in_transit": float(self.op8_convoy_ration_hours_in_transit),
            "op8_convoy_masked_dispatch_attempts": float(self.op8_convoy_masked_dispatch_attempts),
            "op8_convoy_resource_residual": float(
                int(self.op8_convoy_available) + int(not self.op8_convoy_available) - 1
            ),
        }

    def get_op8_convoy_observation(self) -> dict[str, float]:
        if self.op8_dispatch_mode != "finite_convoy_v1":
            raise RuntimeError("DRA-2 observation requires finite_convoy_v1 mode.")
        now = float(self.env.now)
        ages = [
            max(0.0, now - float(order.OPTj))
            for order in self.pending_backorders
        ]
        recent_start = now - HOURS_PER_WEEK
        recent_demand = sum(q for t, q in self.daily_demand if t >= recent_start)
        recent_production = sum(q for t, q in self.daily_production if t >= recent_start)
        recent_delivery = sum(q for t, q in self.delivery_events if t >= recent_start)
        staging_age = (
            max(0.0, now - self.op8_staging_first_ready_at)
            if self.op8_staging_first_ready_at is not None else 0.0
        )
        return {
            "op7_staged_inventory": float(self.rations_al.level),
            "pending_assembly_quantity": float(self._pending_batch),
            "sb_inventory": float(self.rations_sb.level),
            "downstream_backlog_qty": float(self.pending_backorder_qty),
            "downstream_backlog_count": float(len(self.pending_backorders)),
            "oldest_backlog_age": float(max(ages) if ages else 0.0),
            "recent_7d_demand": float(recent_demand),
            "recent_7d_production": float(recent_production),
            "recent_7d_delivery": float(recent_delivery),
            "convoy_available": float(self.op8_convoy_available),
            "convoy_return_eta": float(
                max(0.0, self.op8_convoy_nominal_return_at - now)
                if self.op8_convoy_nominal_return_at is not None else 0.0
            ),
            "time_since_departure": float(
                max(0.0, now - self.op8_convoy_last_departure_at)
                if self.op8_convoy_last_departure_at is not None else 0.0
            ),
            "staging_age": float(staging_age),
            "op8_route_up": float(not self._is_down(8)),
            "previous_action_dispatch": float(self.op8_last_action == "DISPATCH_NOW"),
            "day_phase": float((now % HOURS_PER_WEEK) / HOURS_PER_WEEK),
        }

    def _op8_transport_to_sb(self):
        while True:
            batch_size = self.params["batch_size"]
            yield self.rations_al.get(batch_size)
            lineage = self._lineage_take("rations_al", batch_size)
            departed_at = float(self.env.now)
            self._in_transit += batch_size
            while self._is_down(8):
                yield self.env.timeout(1)
            yield self.env.timeout(self._pt("op8_pt"))
            self._in_transit -= batch_size
            yield self.rations_sb.put(batch_size)
            self._record_material_availability("rations_sb", batch_size)
            op8_refs = self._consume_pending_stage_refs("op8_output")
            self._lineage_forward(
                "rations_sb",
                lineage,
                source_stage="op8_output",
                extra_refs=op8_refs,
            )
            if self.risk_attribution_source == "causal_exposure":
                if self.material_lineage_mode == "tagged_lots":
                    arrived_refs = set(op8_refs)
                    for item in lineage:
                        arrived_refs.update(item.risk_event_refs)
                    self._upstream_scarcity_debts = [
                        event
                        for event in self._upstream_scarcity_debts
                        if self._lineage_event_ref(event) not in arrived_refs
                    ]
                else:
                    self._upstream_scarcity_debts = [
                        event
                        for event in self._upstream_scarcity_debts
                        if float(event.end_time) > departed_at
                    ]
            if self.warmup_trigger == "op9_arrival":
                self._mark_warmup_complete()
            if (
                self.order_fulfillment_mode == "op9_linked"
                and self.pending_backorders
            ):
                yield from self._serve_pending_backorders()

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
        self.total_theatre_inflow += qty
        # Backward-compatible legacy name. This is theatre inflow, not final
        # order consumption; use total_order_fulfilled for customer service.
        self.total_delivered += qty
        self.delivery_events.append((self.env.now, qty))
        yield from self._serve_pending_backorders()

    def _op9_daily_freight_dispatch(self):
        """Op9 outbound: at most ONE order-batch per day (op9_linked mode).

        Fig 6.2 / Table 6.20: Op9 ships Q = 2,400-2,600 rations 'at a daily
        freight rate' (ROP = 24 h). This serving-rate constraint — not an
        artificial delay — is what produces Garrido's congested standing queue
        (final ΣBt ≈ 60), strictly positive departure waits (no CTj = 48.0
        exactly, hence zero on-time orders in the workbooks), the ~2,300 h CT
        p95 (cap-60 queue drained at one order/day), and bounded attended-order
        tails (stragglers are evicted as Ut before aging for years).

        Departures happen once per day at `op9_freight_offset_hours` after
        midnight (orders are placed just after midnight in the thesis tapes;
        the 12% of attended orders with CTj <= 54 h are the queue-empty days
        where the same-morning departure adds <= 6 h of wait). A departure is
        skipped when Op9 is down, the queue is empty, or on-hand SB stock does
        not cover the SPT-head order.
        """
        if self.op9_dispatch_policy == "fixed_clock_daily":
            offset = float(self.op9_freight_offset_hours) % 24.0
            first = math.floor(self.env.now / 24.0) * 24.0 + offset
            if first <= self.env.now:
                first += 24.0
            yield self.env.timeout(first - self.env.now)
            while True:
                yield from self._dispatch_one_op9_order_if_ready(
                    next_check_hours=24.0
                )
                yield self.env.timeout(24.0)

        # Thesis-grounded alternative: a daily freight *headway*, without an
        # estimated absolute clock phase.  The first ready order leaves as
        # soon as stock and Op9 are available; the next release cannot occur
        # for 24 h.  Hourly polling is only a liveness mechanism while blocked.
        while True:
            dispatched = yield from self._dispatch_one_op9_order_if_ready(
                next_check_hours=1.0
            )
            yield self.env.timeout(24.0 if dispatched else 1.0)

    def _dispatch_one_op9_order_if_ready(self, *, next_check_hours: float = 24.0):
        """Release at most the SPT-head order; return whether it departed."""
        if self.cssu_topology_mode == "split_v1":
            return (yield from self._dispatch_split_cssu_day(
                next_check_hours=next_check_hours
            ))
        if not self.pending_backorders:
            return False
        head = self.pending_backorders[0]
        if self._is_down(9):
            self._record_causal_block(
                head,
                op_id=9,
                start_time=float(self.env.now),
                end_time=float(self.env.now) + float(next_check_hours),
                reason="operation_down",
            )
            return False
        qty = float(head.remaining_qty)
        if qty <= 1e-9:
            self._finalize_pending_backorder(head)
            self._remove_pending_backorder(head)
            return False
        if self.rations_sb.level + 1e-9 < qty:
            self._record_active_upstream_stockout_causes(
                head, duration=float(next_check_hours)
            )
            return False
        yield self.rations_sb.get(qty)
        self._record_material_availability("order_release", qty)
        consumed_lineage = self._lineage_take("rations_sb", qty)
        for item in consumed_lineage:
            if not item.risk_event_refs:
                continue
            head.consumed_material_lineage.append(
                {
                    "lot_id": item.lot_id,
                    "quantity": float(item.quantity),
                    "risk_event_refs": list(item.risk_event_refs),
                    "source_stage": item.source_stage,
                }
            )
        head.remaining_qty = 0.0
        head.in_flight_qty += qty
        head.op9_release_time = float(self.env.now)
        head.causal_wait_hours["op9_release"] = max(
            0.0, float(self.env.now) - float(head.OPTj)
        )
        self.env.process(self._deliver_order_from_op9(head, qty))
        self._remove_pending_backorder(head)
        return True

    def set_cssu_allocation_action(
        self,
        allocation_a: float,
        service_rule: str,
        *,
        activation_delay_hours: float = 24.0,
    ) -> dict[str, Any]:
        """Schedule a split-CSSU action without changing total capacity.

        Dynamic actions take effect after the preregistered one-day latency.
        Static frontier policies are configured in the constructor and require
        no runtime mutation.
        """
        allocation_a = float(allocation_a)
        if allocation_a not in ALLOCATION_LEVELS:
            raise ValueError(f"allocation_a must be one of {ALLOCATION_LEVELS}")
        if service_rule not in SERVICE_RULES:
            raise ValueError(f"service_rule must be one of {SERVICE_RULES}")
        if activation_delay_hours < 0:
            raise ValueError("activation_delay_hours must be non-negative")
        event = {
            "requested_at": float(self.env.now),
            "effective_at": float(self.env.now) + float(activation_delay_hours),
            "allocation_a": allocation_a,
            "service_rule": str(service_rule),
            "previous_allocation_a": float(self.cssu_allocation_a),
            "previous_service_rule": str(self.cssu_service_rule),
        }
        self._pending_cssu_action = event
        self.cssu_action_events.append(dict(event, status="scheduled"))
        return event

    def _activate_due_cssu_action(self) -> None:
        pending = self._pending_cssu_action
        if pending is None or float(self.env.now) + 1e-9 < pending["effective_at"]:
            return
        self.cssu_allocation_a = float(pending["allocation_a"])
        self.cssu_service_rule = str(pending["service_rule"])
        self.cssu_action_events.append(
            dict(pending, activated_at=float(self.env.now), status="activated")
        )
        self._pending_cssu_action = None

    def _cssu_daily_capacity(self) -> float:
        if self.cssu_daily_capacity_override is not None:
            return float(self.cssu_daily_capacity_override)
        # Event-keyed daily quantity tape: reproduces the selected thesis range
        # without consuming a simulator RNG or drifting when policies induce a
        # different event-call order.
        q_min = int(round(float(self.params["op10_q_min"])))
        q_max = int(round(float(self.params["op10_q_max"])))
        day = int(float(self.env.now) // HOURS_PER_DAY)
        digest = hashlib.sha256(
            f"dra1-capacity-v1:{int(self.seed or 0)}:{day}".encode()
        ).digest()
        return float(q_min + int.from_bytes(digest[:8], "big") % (q_max - q_min + 1))

    def _cssu_daily_allocation_draw(self) -> float:
        """Policy-independent daily draw used for indivisible full orders."""
        day = int(float(self.env.now) // HOURS_PER_DAY)
        digest = hashlib.sha256(
            f"dra1-allocation-v1:{int(self.seed or 0)}:{day}".encode()
        ).digest()
        return int.from_bytes(digest[:8], "big") / float(2**64)

    def _cssu_order_key(self, order: OrderRecord) -> tuple[Any, ...]:
        if self.cssu_service_rule == "SPT_FULL":
            return (
                0 if order.contingent else 1,
                float(order.remaining_qty),
                float(order.OPTj),
                int(order.j),
            )
        if self.cssu_service_rule == "FIFO_PARTIAL":
            return (float(order.OPTj), int(order.j))
        return (
            0 if order.contingent else 1,
            float(order.OPTj),
            int(order.j),
        )

    def cssu_allocation_is_live(self) -> bool:
        """Whether changing the allocation share can affect this dispatch epoch."""
        if self.cssu_topology_mode != "split_v1" or self._is_down(9):
            return False
        available = min(float(self.rations_sb.level), self._cssu_daily_capacity())
        if available <= 1e-9:
            return False
        queues = {
            cssu: sorted(
                [o for o in self.pending_backorders
                 if o.cssu_destination == cssu and float(o.remaining_qty) > 1e-9],
                key=self._cssu_order_key,
            )
            for cssu in ("A", "B")
        }
        if self.cssu_service_rule == "SPT_FULL":
            both_feasible = all(
                queues[cssu]
                and float(queues[cssu][0].remaining_qty) <= available + 1e-9
                for cssu in ("A", "B")
            )
            # The three allowed shares select different destinations only for
            # draws between the extreme thresholds.
            draw = self._cssu_daily_allocation_draw()
            return bool(both_feasible and 0.25 <= draw < 0.75)
        requested = {
            cssu: sum(float(order.remaining_qty) for order in queues[cssu])
            for cssu in ("A", "B")
        }
        return (
            requested["A"] > available * self.cssu_allocation_a + 1e-9
            and requested["B"] > available * (1 - self.cssu_allocation_a) + 1e-9
        )

    def _dispatch_split_cssu_day(self, *, next_check_hours: float = 24.0):
        """Dispatch one conserved daily capacity pool across CSSU A/B."""
        self._activate_due_cssu_action()
        if not self.pending_backorders:
            return False
        if self._is_down(9):
            for order in self.pending_backorders:
                self._record_causal_block(
                    order,
                    op_id=9,
                    start_time=float(self.env.now),
                    end_time=float(self.env.now) + float(next_check_hours),
                    reason="operation_down",
                )
            return False

        queues = {
            cssu: sorted(
                [
                    order
                    for order in self.pending_backorders
                    if order.cssu_destination == cssu
                    and float(order.remaining_qty) > 1e-9
                ],
                key=self._cssu_order_key,
            )
            for cssu in ("A", "B")
        }
        requested = {
            cssu: sum(float(order.remaining_qty) for order in queues[cssu])
            for cssu in ("A", "B")
        }
        available = min(float(self.rations_sb.level), self._cssu_daily_capacity())
        if available <= 1e-9:
            for order in self.pending_backorders:
                self._record_active_upstream_stockout_causes(
                    order, duration=float(next_check_hours)
                )
            return False
        if self.cssu_service_rule == "SPT_FULL":
            # A daily lane carries roughly one thesis-sized full order. The
            # share is implemented as a deterministic CRN-safe frequency over
            # days: A is preferred when the daily draw is below alpha. If that
            # destination cannot use the convoy, capacity is reallocated.
            candidates = [
                cssu
                for cssu in ("A", "B")
                if queues[cssu]
                and float(queues[cssu][0].remaining_qty) <= available + 1e-9
            ]
            jointly_constrained = self.cssu_allocation_is_live()
            if candidates:
                preferred = (
                    "A" if self._cssu_daily_allocation_draw() < self.cssu_allocation_a
                    else "B"
                )
                selected = preferred if preferred in candidates else candidates[0]
                selected_qty = float(queues[selected][0].remaining_qty)
                budgets = {"A": 0.0, "B": 0.0}
                budgets[selected] = selected_qty
            else:
                budgets = {"A": 0.0, "B": 0.0}
        else:
            allocation = allocate_shared_capacity(
                stock=float(self.rations_sb.level),
                daily_capacity=self._cssu_daily_capacity(),
                allocation_a=self.cssu_allocation_a,
                requested=requested,
                reallocate_unused=True,
            )
            # V1-A audit: the split is live only when both destinations exhaust
            # their nominal shares. Preserve this diagnostic for state sampling.
            nominal_a = allocation.available * self.cssu_allocation_a
            nominal_b = allocation.available - nominal_a
            jointly_constrained = self.cssu_allocation_is_live()
            budgets = {"A": allocation.dispatched_a, "B": allocation.dispatched_b}
        if jointly_constrained:
            self.cssu_allocation_live_epochs += 1
        else:
            self.cssu_allocation_moot_epochs += 1

        dispatched_any = False
        for cssu in ("A", "B"):
            budget = float(budgets[cssu])
            for order in queues[cssu]:
                remaining = float(order.remaining_qty)
                if budget <= 1e-9:
                    break
                if self.cssu_service_rule == "SPT_FULL" and remaining > budget + 1e-9:
                    break
                qty = min(remaining, budget)
                yield self.rations_sb.get(qty)
                self._record_material_availability("order_release", qty)
                order.remaining_qty -= qty
                order.in_flight_qty += qty
                if order.op9_release_time is None:
                    order.op9_release_time = float(self.env.now)
                    order.causal_wait_hours["op9_release"] = max(
                        0.0, float(self.env.now) - float(order.OPTj)
                    )
                self.cssu_dispatched[cssu] += qty
                self.env.process(self._deliver_order_from_op9(order, qty))
                budget -= qty
                dispatched_any = True
                if order.remaining_qty <= 1e-9:
                    order.remaining_qty = 0.0
                    self._remove_pending_backorder(order)
            self._refresh_pending_backorder_qty()
        return dispatched_any

    def _deliver_order_from_op9(self, order: OrderRecord, qty: float):
        """Move a reserved order through the physical Op10/Op12 service path.

        Fig 6.2 / Table 6.20: Op10 and Op12 are LOCs shipping 'at a daily
        freight rate' (ROP = 24 h) — each stage is a SINGLE convoy doing a
        24 h leg, i.e. a capacity-1 server with 24 h service, not an
        unlimited parallel pipeline. This tandem structure is what carries
        Garrido's standing mid-band congestion (CT p75 ≈ 650 h spread across
        stage queues) and holds orders 'in flight' beyond the Op9 backlog.
        Outages (R22 on the LOCs, R23 on Op11) block the convoy IN the slot,
        so downstream attacks propagate backpressure through the pipeline.
        """
        self._in_transit += qty
        destination = order.cssu_destination
        if self.cssu_topology_mode == "split_v1":
            if destination not in {"A", "B"}:
                raise AssertionError("split_v1 order lacks a valid CSSU destination")
            self.cssu_in_transit[destination] += qty
            self.cssu_inbound_in_transit[destination] += qty
        if self.downstream_transport_capacity_mode == "tandem_capacity_one":
            queue_start = float(self.env.now)
            with self.op10_convoy.request() as req:
                yield req
                order.causal_wait_hours["op10_resource_queue"] = max(
                    0.0, float(self.env.now) - queue_start
                )
                yield from self._wait_order_for_cssu_operation(order, 10)
                yield self.env.timeout(self._pt("op10_pt"))
        else:
            yield from self._wait_order_for_cssu_operation(order, 10)
            yield self.env.timeout(self._pt("op10_pt"))
        if self.cssu_topology_mode == "split_v1":
            self.cssu_inbound_in_transit[destination] -= qty
            self.cssu_inventory[destination] += qty
        yield from self._wait_order_for_cssu_operation(order, 11)
        if self.cssu_topology_mode == "split_v1":
            self.cssu_inventory[destination] -= qty
            self.cssu_outbound_in_transit[destination] += qty
        if self.downstream_transport_capacity_mode == "tandem_capacity_one":
            queue_start = float(self.env.now)
            with self.op12_convoy.request() as req:
                yield req
                order.causal_wait_hours["op12_resource_queue"] = max(
                    0.0, float(self.env.now) - queue_start
                )
                yield from self._wait_order_for_cssu_operation(order, 12)
                yield self.env.timeout(self._pt("op12_pt"))
        else:
            yield from self._wait_order_for_cssu_operation(order, 12)
            yield self.env.timeout(self._pt("op12_pt"))
        self._in_transit -= qty
        if self.cssu_topology_mode == "split_v1":
            self.cssu_in_transit[destination] -= qty
            self.cssu_outbound_in_transit[destination] -= qty
            self.cssu_delivered[destination] += qty
            self.cssu_delivery_events.append(
                (float(self.env.now), destination, float(qty))
            )
        self.total_theatre_inflow += qty
        self.total_delivered += qty
        self.total_order_fulfilled += qty
        self.delivery_events.append((self.env.now, qty))
        order.in_flight_qty = max(0.0, float(order.in_flight_qty) - qty)
        if order.remaining_qty <= 1e-9 and order.in_flight_qty <= 1e-9:
            self._finalize_pending_backorder(order)

    def _wait_order_for_operation(self, order: OrderRecord, op_id: int):
        """Wait for an operation and retain the exact order-specific block."""
        start = float(self.env.now)
        while self._is_down(op_id):
            yield self.env.timeout(1.0)
        end = float(self.env.now)
        if end > start:
            key = f"op{int(op_id)}_down"
            order.causal_wait_hours[key] = (
                order.causal_wait_hours.get(key, 0.0) + end - start
            )
            self._record_causal_block(
                order,
                op_id=op_id,
                start_time=start,
                end_time=end,
                reason="operation_down",
            )

    def _wait_order_for_cssu_operation(self, order: OrderRecord, op_id: int):
        """Wait on aggregate or destination-local downstream availability."""
        if self.cssu_topology_mode != "split_v1":
            yield from self._wait_order_for_operation(order, op_id)
            return
        start = float(self.env.now)
        while self._is_cssu_path_down(op_id, order.cssu_destination):
            yield self.env.timeout(1.0)
        end = float(self.env.now)
        if end > start:
            key = f"op{int(op_id)}_{order.cssu_destination}_down"
            order.causal_wait_hours[key] = (
                order.causal_wait_hours.get(key, 0.0) + end - start
            )
            self._record_causal_block(
                order,
                op_id=op_id,
                start_time=start,
                end_time=end,
                reason=f"cssu_{order.cssu_destination}_path_down",
            )

    # =====================================================================
    # DEMAND SINK: Op13
    # =====================================================================

    def _place_demand_order(self, order: OrderRecord):
        if self.cssu_topology_mode == "split_v1":
            if order.contingent and self._contingent_cssu_destination_pending:
                order.cssu_destination = self._contingent_cssu_destination_pending
                self._contingent_cssu_destination_pending = None
            elif order.cssu_destination is None:
                order.cssu_destination = stable_cssu_destination(
                    simulation_seed=int(self.seed or 0), order_id=int(order.j)
                )
            if order.cssu_destination not in {"A", "B"}:
                raise ValueError("split_v1 orders require CSSU destination A or B")
            self.cssu_demanded[order.cssu_destination] += float(order.quantity)
            self.cssu_demand_events.append(
                (float(self.env.now), order.cssu_destination, float(order.quantity))
            )
        for episode_id in order.causal_r24_event_ids:
            episode = self._r24_causal_episodes.get(episode_id)
            if episode is not None and episode.get("assigned_order_j") is None:
                episode["assigned_order_j"] = int(order.j)
        program_f_qty = 0.0
        if (
            self.program_f_reserve_enabled
            and order.contingent
            and self.program_f_r24_issue_remaining > 1e-9
            and self.emergency_theatre_reserve.level > 1e-9
        ):
            program_f_qty = min(
                float(order.remaining_qty),
                float(self.program_f_r24_issue_remaining),
                float(self.emergency_theatre_reserve.level),
            )
            if program_f_qty > 1e-9:
                self._account_emergency_reserve_inventory_time()
                yield self.emergency_theatre_reserve.get(program_f_qty)
                self._in_transit += program_f_qty
                order.remaining_qty -= program_f_qty
                order.in_flight_qty += program_f_qty
                self.program_f_r24_issue_remaining -= program_f_qty
                self.program_f_reserve_issue_events.append({
                    "time": float(self.env.now),
                    "order_j": float(order.j),
                    "quantity": float(program_f_qty),
                })
                self.env.process(
                    self._deliver_program_f_reserve_fragment(order, program_f_qty)
                )
                if order.remaining_qty <= 1e-9:
                    order.remaining_qty = 0.0
                    self.orders.append(order)
                    self.daily_demand.append((self.env.now, order.quantity))
                    return
        emergency_available = (
            self.emergency_reserve_enabled
            and self.order_fulfillment_mode == "op9_linked"
            and self._emergency_corridor_down()
            and self.emergency_theatre_reserve.level + 1e-9 >= order.quantity
        )
        if emergency_available:
            # The reserve is positioned behind Op10--Op12 and is released only
            # while that corridor is physically unavailable.  It therefore
            # cannot improve calm-period service or manufacture a forecasting
            # advantage in the absence of an actual downstream outage.
            self._account_emergency_reserve_inventory_time()
            yield self.emergency_theatre_reserve.get(order.quantity)
            # Preserve mass conservation during the local issue delay.
            self._in_transit += float(order.quantity)
            order.remaining_qty = 0.0
            self.env.process(
                self._fulfill_from_emergency_reserve(order, float(order.quantity))
            )
            self.orders.append(order)
            self.daily_demand.append((self.env.now, order.quantity))
            return
        source = (
            self.rations_sb
            if self.order_fulfillment_mode == "op9_linked"
            else self.rations_theatre
        )
        available = source.level
        if (
            self.order_fulfillment_mode != "op9_linked"
            and not self.pending_backorders
            and available >= order.quantity
        ):
            yield source.get(order.quantity)
            self.total_order_fulfilled += float(order.quantity)
            self._finalize_order_after_fulfillment_delay(order)
        else:
            # Enqueue for future delivery. Per thesis Sec. 6.8.2,
            # backorder classification deferred until LTj=48h elapses.
            # In op9_linked mode EVERY order queues for the daily freight
            # departure (no instant-service path exists in Fig 6.2).
            self._enqueue_backorder(order)
            if self.order_fulfillment_mode != "op9_linked":
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

    def _sample_calendar_demand_quantity(self) -> tuple[float, bool, set[str]]:
        demand_qty = float(
            self.demand_rng.integers(int(DEMAND["a"]), int(DEMAND["b"]) + 1)
        )
        demand_qty *= self.demand_mean_multiplier
        if self.adaptive_benchmark_enabled:
            demand_scale = float(self._adaptive_regime_params()["demand_scale"])
            demand_qty *= demand_scale
        contingent_qty = float(self._contingent_demand_pending)
        causal_episode_ids: set[str] = set()
        if contingent_qty > 0:
            demand_qty += contingent_qty
            self._contingent_demand_pending = 0
            if self.risk_attribution_source == "causal_exposure":
                for episode_id, episode in self._r24_causal_episodes.items():
                    if episode.get("assigned_order_j") is None and float(
                        episode.get("surge_qty", 0.0)
                    ) > 0.0:
                        causal_episode_ids.add(episode_id)
        return demand_qty, contingent_qty > 0, causal_episode_ids

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

            (
                demand_qty,
                is_contingent,
                causal_r24_event_ids,
            ) = self._sample_calendar_demand_quantity()
            self.total_demanded += demand_qty
            order_num += 1
            order = OrderRecord(
                j=order_num,
                OPTj=self.env.now,
                quantity=demand_qty,
                remaining_qty=demand_qty,
                contingent=is_contingent,
                causal_r24_event_ids=causal_r24_event_ids,
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

        if self.demand_start_after_warmup:
            while not self.warmup_complete and self.env.now < self.horizon:
                yield self.env.timeout(1.0)

        order_num = 0
        hour_of_week = int(self.env.now) % (7 * HOURS_PER_DAY)
        while True:
            yield self.env.timeout(DEMAND["frequency_hrs"])
            hour_of_week = (hour_of_week + HOURS_PER_DAY) % (7 * HOURS_PER_DAY)
            day_of_week = hour_of_week // HOURS_PER_DAY
            if day_of_week >= 6:
                continue

            (
                demand_qty,
                is_contingent,
                causal_r24_event_ids,
            ) = self._sample_calendar_demand_quantity()

            self.total_demanded += demand_qty
            order_num += 1
            order = OrderRecord(
                j=order_num,
                OPTj=self.env.now,
                quantity=demand_qty,
                remaining_qty=demand_qty,
                contingent=is_contingent,
                causal_r24_event_ids=causal_r24_event_ids,
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
        return min(
            0.98,
            float(base_p)
            * self._risk_frequency_multiplier_for(risk_id, campaign_mode="now"),
        )

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

    def _risk_frequency_multiplier_for(
        self, risk_id: str, campaign_mode: str = "max"
    ) -> float:
        base = float(
            self.risk_frequency_multipliers_by_id.get(
                str(risk_id), self.risk_frequency_multiplier
            )
        )
        if campaign_mode == "now":
            # Binomial risks (R12/R13) re-evaluate p every cycle, so the
            # state-CURRENT multiplier is exact — no thinning needed.
            return base * self._campaign_multiplier_now(risk_id, "frequency")
        # Uniform-window risks: sample at the MAX (campaign) rate; the
        # state-dependent acceptance gate in the risk loop restores the
        # native rate during calm phases (exact thinning).
        return base * self._campaign_freq_max(risk_id)

    def _risk_impact_multiplier_for(self, risk_id: str) -> float:
        base = float(
            self.risk_impact_multipliers_by_id.get(
                str(risk_id), self.risk_impact_multiplier
            )
        )
        return base * self._campaign_impact_now(risk_id)

    # ------------------------------------------------------------ Track C campaign
    def _build_campaign_path(self) -> None:
        """Sample the episode's regime schedule (seed-deterministic).

        Dedicated SeedSequence stream: identical seeds give identical campaign
        paths regardless of policy actions or other RNG consumption, and no
        other stream's draw sequence is perturbed.

        Two config formats:
          legacy 2-state: dwell_calm_weeks_mean / dwell_campaign_weeks_mean +
            frequency_multipliers / impact_multipliers (campaign-state only);
          cycle (v2): {"cycle": [{"name", "dwell_mean_weeks",
            "frequency_multipliers", "impact_multipliers"}, ...]} — states
            visited in order, looping (e.g. calm -> pre_campaign -> campaign).
        """
        cfg = self.campaign_config or {}
        week = 168.0
        min_dwell = max(week, float(cfg.get("dwell_min_weeks", 2.0)) * week)
        rng = np.random.default_rng(
            np.random.SeedSequence([int(self.seed or 0), 0xC4A9])
        )
        cycle = cfg.get("cycle")
        if cycle:
            states = [
                (str(s["name"]), max(week, float(s["dwell_mean_weeks"]) * week))
                for s in cycle
            ]
        else:
            states = [
                ("calm", max(week, float(cfg.get("dwell_calm_weeks_mean", 8.0)) * week)),
                ("campaign", max(week, float(cfg.get("dwell_campaign_weeks_mean", 5.0)) * week)),
            ]
        start_idx = 0
        initial = str(cfg.get("initial_state", states[0][0]))
        for i, (name, _d) in enumerate(states):
            if name == initial:
                start_idx = i
                break
        t = 0.0
        idx = start_idx
        path: list[tuple[float, str]] = []
        while t < float(self.horizon):
            name, dwell_mean = states[idx]
            path.append((t, name))
            dwell = max(min_dwell, float(rng.exponential(dwell_mean)))
            t += dwell
            idx = (idx + 1) % len(states)
        self.campaign_path = path

    def _campaign_state_tables(self) -> dict[str, dict[str, dict[str, float]]]:
        """Per-state {freq, impact} multiplier tables (both config formats)."""
        cfg = self.campaign_config or {}
        cycle = cfg.get("cycle")
        if cycle:
            return {
                str(s["name"]): {
                    "frequency": dict(s.get("frequency_multipliers") or {}),
                    "impact": dict(s.get("impact_multipliers") or {}),
                }
                for s in cycle
            }
        return {
            "campaign": {
                "frequency": dict(cfg.get("frequency_multipliers") or {}),
                "impact": dict(cfg.get("impact_multipliers") or {}),
            }
        }

    def campaign_state_at(self, t: float) -> str:
        if not self.campaign_path:
            return "calm"
        state = self.campaign_path[0][1]
        for start, s in self.campaign_path:
            if float(t) >= start:
                state = s
            else:
                break
        return state

    def _campaign_freq_max(self, risk_id: str) -> float:
        if not self.campaign_config:
            return 1.0
        best = 1.0
        for tables in self._campaign_state_tables().values():
            best = max(best, float(tables["frequency"].get(str(risk_id), 1.0)))
        return best

    def _campaign_multiplier_now(self, risk_id: str, kind: str) -> float:
        if not self.campaign_config:
            return 1.0
        state = self.campaign_state_at(float(self.env.now))
        tables = self._campaign_state_tables().get(state)
        if not tables:
            return 1.0
        return float(tables[kind].get(str(risk_id), 1.0))

    def _campaign_impact_now(self, risk_id: str) -> float:
        return self._campaign_multiplier_now(risk_id, "impact")

    def _campaign_accept_event(self, risk_id: str) -> bool:
        """Thinning acceptance: keep a max-rate candidate event with p=m(state)/m_max."""
        m_max = self._campaign_freq_max(risk_id)
        if m_max <= 1.0:
            return True
        m_now = self._campaign_multiplier_now(risk_id, "frequency")
        return bool(self._risk_rng_for(risk_id).random() < (m_now / m_max))

    def _risk_rng_for(self, risk_id: str) -> np.random.Generator:
        """Return a stable family-specific stream for causal risk ablations."""
        if self.risk_rng_mode == "per_risk":
            return self.risk_rng_by_id[str(risk_id)]
        return self.risk_rng

    def _sample_uniform_risk_window(self, risk_id: str) -> tuple[float, float]:
        """Sample the event offset for a thesis uniform-occurrence window."""
        a = int(RISKS_CURRENT[risk_id]["occurrence"]["a"])
        b_val = max(a, int(round(self._get_risk_b(risk_id))))
        delay = float(self._risk_rng_for(risk_id).integers(a, b_val + 1))
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
            if not self._campaign_accept_event("R11"):
                tail = self._tail_after_uniform_occurrence(delay, window)
                if tail > 0:
                    yield self.env.timeout(tail)
                continue
            if self.campaign_config:
                beta = self._get_risk_recovery_mean("R11")
            if self.risk_occurrence_mode == "thesis_window":
                self.env.process(self._risk_R11_event(beta))
                yield self.env.timeout(self._tail_after_uniform_occurrence(delay, window))
            else:
                yield from self._risk_R11_event(beta)

    def _risk_R11_event(self, beta: float):
        rng = self._risk_rng_for("R11")
        target = int(rng.choice(RISKS_CURRENT["R11"]["affected_ops"]))
        start = self.env.now
        self._take_down(target)
        repair = max(1, rng.exponential(beta))
        yield self.env.timeout(repair)
        self._bring_up(target)
        event = RiskEvent(
            "R11", start, self.env.now, self.env.now - start, [target]
        )
        self.risk_events.append(event)
        self._register_upstream_scarcity_debt(event)

    def _risk_R12(self):
        n = RISKS_CURRENT["R12"]["occurrence"]["n"]
        p = self._get_risk_p("R12")
        first_cycle = True
        while True:
            if not (
                first_cycle
                and self.operational_risk_initialization_mode
                == "include_initial_cycle"
            ):
                yield self.env.timeout(self.params["op1_rop"])
            first_cycle = False
            delayed = self._risk_rng_for("R12").binomial(n, p)
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
        event = RiskEvent(
            "R12",
            start,
            self.env.now,
            delay,
            [1],
            magnitude=float(delayed),
            unit="delayed_contracts",
        )
        self.risk_events.append(event)
        self._register_upstream_scarcity_debt(event)

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
            delayed = self._risk_rng_for("R13").binomial(n, p)
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
        event = RiskEvent(
            "R13",
            start,
            self.env.now,
            delay,
            [2],
            magnitude=float(delayed),
            unit="delayed_deliveries",
        )
        self.risk_events.append(event)
        self._register_upstream_scarcity_debt(event)

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
                defects = self._risk_rng_for("R14").binomial(produced, p)
                if defects > 0:
                    # Cap at available pending to maintain mass balance
                    defects = min(defects, int(self._pending_batch))
                    if defects > 0:
                        self._pending_batch -= defects
                        self.total_produced -= defects
                        defect_lineage = self._lineage_take("pending_batch", defects)
                        if self.r14_defect_mode == "thesis_strict_op6":
                            yield self.rework_op6.put(defects)
                            self._lineage_forward(
                                "rework_op6", defect_lineage, source_stage="r14_rework"
                            )
                            description = f"{defects} defective (returned to Op6)"
                        elif self.r14_defect_mode == "reprocess":
                            # Thesis Table 6.6b: defects returned to Op6 for
                            # re-processing. Model by feeding back to raw material
                            # so they re-enter the assembly pipeline later.
                            yield self.raw_material_al.put(defects)
                            self._lineage_forward(
                                "raw_material_al", defect_lineage, source_stage="r14_reprocess"
                            )
                            description = (
                                f"{defects} defective (returned to raw_material_al)"
                            )
                        else:
                            self.total_rations_scrapped += float(defects)
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
            if not self._campaign_accept_event("R21"):
                tail = self._tail_after_uniform_occurrence(delay, window)
                if tail > 0:
                    yield self.env.timeout(tail)
                continue
            if self.campaign_config:
                # Recovery mean at FIRE time so campaign impact applies;
                # without campaign_config keep the legacy pre-loop constant
                # (bitwise identity for all existing lanes).
                beta = self._get_risk_recovery_mean("R21")
            self.env.process(self._r21_event(affected, beta))
            tail = self._tail_after_uniform_occurrence(delay, window)
            if tail > 0:
                yield self.env.timeout(tail)

    def _r21_event(self, affected: list[int], beta: float):
        rng = self._risk_rng_for("R21")
        start = self.env.now
        for op_id in affected:
            self._take_down(op_id)
        recovery_times = {}
        for op_id in affected:
            rt = max(1, rng.exponential(beta))
            recovery_times[op_id] = rt
            self.env.process(self._delayed_bring_up(op_id, rt))
        max_rt = max(recovery_times.values())
        yield self.env.timeout(max_rt)
        event = RiskEvent(
            "R21", start, self.env.now, max_rt, list(affected)
        )
        self.risk_events.append(event)
        self._register_upstream_scarcity_debt(event)

    def _risk_R22(self):
        beta = self._get_risk_recovery_mean("R22")
        loc_ops = RISKS_CURRENT["R22"]["affected_ops"]
        while True:
            delay, window = self._sample_uniform_risk_window("R22")
            yield self.env.timeout(delay)
            if not self._campaign_accept_event("R22"):
                tail = self._tail_after_uniform_occurrence(delay, window)
                if tail > 0:
                    yield self.env.timeout(tail)
                continue
            if self.campaign_config:
                beta = self._get_risk_recovery_mean("R22")
            if self.risk_occurrence_mode == "thesis_window":
                self.env.process(self._risk_R22_event(beta, loc_ops))
                yield self.env.timeout(self._tail_after_uniform_occurrence(delay, window))
            else:
                yield from self._risk_R22_event(beta, loc_ops)

    def _risk_R22_event(self, beta: float, loc_ops: list[int]):
        rng = self._risk_rng_for("R22")
        target = int(rng.choice(loc_ops))
        target_cssu = (
            str(rng.choice(("A", "B")))
            if self.cssu_topology_mode == "split_v1" and target in {10, 12}
            else None
        )
        start = self.env.now
        if target_cssu is None:
            self._take_down(target)
        else:
            self._take_down_cssu(target, target_cssu)
        recovery = max(1, rng.exponential(beta))
        yield self.env.timeout(recovery)
        if target_cssu is None:
            self._bring_up(target)
        else:
            self._bring_up_cssu(target, target_cssu)
        event = RiskEvent(
            "R22", start, self.env.now, recovery, [target],
            affected_cssu=target_cssu,
        )
        self.risk_events.append(event)
        if target_cssu is not None:
            self.cssu_local_risk_events.append(
                {
                    "risk_id": "R22",
                    "op_id": target,
                    "cssu": target_cssu,
                    "start_time": float(start),
                    "end_time": float(self.env.now),
                }
            )
        self._register_upstream_scarcity_debt(event)

    def _risk_R23(self):
        beta = self._get_risk_recovery_mean("R23")
        while True:
            delay, window = self._sample_uniform_risk_window("R23")
            yield self.env.timeout(delay)
            if not self._campaign_accept_event("R23"):
                tail = self._tail_after_uniform_occurrence(delay, window)
                if tail > 0:
                    yield self.env.timeout(tail)
                continue
            if self.campaign_config:
                beta = self._get_risk_recovery_mean("R23")
            if self.risk_occurrence_mode == "thesis_window":
                self.env.process(self._risk_R23_event(beta))
                yield self.env.timeout(self._tail_after_uniform_occurrence(delay, window))
            else:
                yield from self._risk_R23_event(beta)

    def _risk_R23_event(self, beta: float):
        rng = self._risk_rng_for("R23")
        start = self.env.now
        target_cssu = (
            str(rng.choice(("A", "B")))
            if self.cssu_topology_mode == "split_v1"
            else None
        )
        if target_cssu is None:
            self._take_down(11)
        else:
            self._take_down_cssu(11, target_cssu)
        recovery = max(1, rng.exponential(beta))
        yield self.env.timeout(recovery)
        if target_cssu is None:
            self._bring_up(11)
        else:
            self._bring_up_cssu(11, target_cssu)
        self.risk_events.append(
            RiskEvent(
                "R23", start, self.env.now, recovery, [11],
                affected_cssu=target_cssu,
            )
        )
        if target_cssu is not None:
            self.cssu_local_risk_events.append(
                {
                    "risk_id": "R23",
                    "op_id": 11,
                    "cssu": target_cssu,
                    "start_time": float(start),
                    "end_time": float(self.env.now),
                }
            )

    def _risk_R24(self):
        while True:
            delay, window = self._sample_uniform_risk_window("R24")
            yield self.env.timeout(delay)
            if self._campaign_accept_event("R24"):
                self._apply_risk_R24_event()
            tail = self._tail_after_uniform_occurrence(delay, window)
            if tail > 0:
                yield self.env.timeout(tail)

    def _apply_risk_R24_event(self) -> None:
        surge_lo, surge_hi = self._get_risk_surge()
        r24_rng = self._risk_rng_for("R24")
        surge = r24_rng.integers(surge_lo, surge_hi + 1)
        target_cssu = (
            str(r24_rng.choice(("A", "B")))
            if self.cssu_topology_mode == "split_v1"
            else None
        )
        if target_cssu is not None:
            self._contingent_cssu_destination_pending = target_cssu
        pending_before = float(self._contingent_demand_pending)
        self._contingent_demand_pending += surge
        # Cap accumulated contingent demand to prevent unbounded obs[14]
        # spikes when multiple R24 events fire before demand is consumed.
        # 5×2600 = 13000 ≈ 5 regular demand cycles, well above any
        # realistic surge accumulation.
        max_contingent = 5 * 2600
        self._contingent_demand_pending = min(
            self._contingent_demand_pending, max_contingent
        )
        accepted_surge = max(
            0.0, float(self._contingent_demand_pending) - pending_before
        )
        # Attribution window: Garrido's per-order R24 column marks every order
        # whose cycle overlaps the surge STRESS PERIOD, not just the instant.
        # CF11 evidence: 75% of attended orders carry R24>0 at increased
        # frequency ≈ 6.8 orders marked per event ≈ one week of placements
        # (168 h, the thesis's weekly cadence). A point event (duration 0)
        # under-marks by an order of magnitude, pushes orders into the
        # fill-rate branch, and inflates mean ReT in the R2 configurations.
        # Default 0.0 preserves the legacy point event bitwise for all frozen
        # Track A/B lanes; the garrido_reference gate opts in with 168 h.
        window = self.r24_attribution_window_hours
        event = RiskEvent(
            "R24",
            self.env.now,
            self.env.now + window,
            window,
            [13],
            f"+{surge}",
            magnitude=float(surge),
            unit="rations",
            affected_cssu=target_cssu,
        )
        self.risk_events.append(event)
        self._add_ret_quantity_risk(event)
        if self.risk_attribution_source == "causal_exposure":
            self._r24_causal_sequence += 1
            episode_id = f"R24-{self._r24_causal_sequence:06d}"
            self._r24_causal_episodes[episode_id] = {
                "episode_id": episode_id,
                "event": event,
                "start_time": float(self.env.now),
                "surge_qty": accepted_surge,
                "baseline_pending_count": len(self.pending_backorders),
                "baseline_pending_qty": float(self.pending_backorder_qty),
                "assigned_order_j": None,
                "closed_at": None,
            }

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

        causal = self.risk_attribution_source == "causal_exposure"
        if causal:
            # A duration event is eligible only when an order-specific physical
            # block identifies the affected operation and interval.  This is
            # the key distinction from retrospective event-window overlap.
            seen_pairs: set[tuple[int, float, float]] = set()
            for block in order.causal_block_intervals:
                for event, contribution in self._match_block_to_events(block):
                    pair = (
                        id(event),
                        float(block["start_time"]),
                        float(block["end_time"]),
                    )
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    total_disruption_hours += float(contribution)
                    mark_event(event, float(contribution))

            # R24 propagates through explicit queue precedence ids, never a
            # fixed or global-backlog time window.
            for episode_id in sorted(order.causal_r24_event_ids):
                episode = self._r24_causal_episodes.get(episode_id)
                if not episode:
                    continue
                event = episode["event"]
                contribution = max(
                    1.0,
                    min(
                        float(order.OATj) - max(
                            float(order.OPTj), float(event.start_time)
                        ),
                        float(order.CTj),
                    ),
                )
                total_disruption_hours += contribution
                mark_event(event, contribution)
        else:
            for event in self.risk_events:
                if float(event.duration) <= 0.0:
                    if float(order.OPTj) <= float(event.start_time) <= float(order.OATj):
                        mark_event(event, float(event.magnitude))
                    continue

                raw_start = max(float(event.start_time), float(order.OPTj))
                raw_end = min(float(event.end_time), float(order.OATj))
                if raw_start < raw_end:
                    overlap = raw_end - raw_start
                    total_disruption_hours += overlap
                    mark_event(event, overlap)

            # Legacy ongoing-disruption attribution remains bitwise isolated.
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
            order, risk_ids=(("R14",) if causal else ("R14", "R24"))
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
        if (
            not causal
            and bool(getattr(order, "contingent", False))
            and "R24" not in order.ret_risk_indicators
        ):
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
