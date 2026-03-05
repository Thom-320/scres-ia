"""
config.py — Single source of truth for the Military Food Supply Chain (MFSC).

All parameters extracted from:
  Garrido-Rios (2017). "A simulation-based methodology for analysing the
  relationship between risks and resilience in a military food supply chain."
  PhD thesis, University of Warwick.

  Primary sources: Tables 6.4, 6.10, 6.12, 6.16, 6.20, 6.25; Figure 6.2;
  Sections 6.3–6.9.

Notation follows the thesis exactly:
  - Opk,j : Operation k of order j
  - PT    : Processing time (hours)
  - Q     : Order/batch quantity
  - ROP   : Reorder point (hours between triggers)
  - S     : Number of work shifts (1, 2, or 3)
  - It,S  : On-hand inventory buffer for period t and S shifts
  - Rcr   : Risk type r of category c
  - λ     : Assembly rate = 320.5 rations/hour
  - Dt    : Demand at time t (rations)
"""

# =============================================================================
# GLOBAL CONSTANTS
# =============================================================================

ASSEMBLY_RATE = 320.5  # λ: rations per hour
HOURS_PER_SHIFT = 8  # Hours per work shift
DAYS_PER_WEEK = 6  # Operating days per week (Mon–Sat)
HOURS_PER_DAY = 24
HOURS_PER_WEEK = 168  # 7 × 24
HOURS_PER_MONTH = 672  # 28 × 24 (thesis convention)
HOURS_PER_YEAR_THESIS = 8_064  # Thesis basis: 336 days × 24
HOURS_PER_YEAR_GREGORIAN = 8_760  # Gregorian basis: 365 days × 24
DEFAULT_YEAR_BASIS = "thesis"
YEAR_BASIS_OPTIONS = ("thesis", "gregorian")
# Backward-compatible alias (defaults to thesis basis).
HOURS_PER_YEAR = HOURS_PER_YEAR_THESIS
SIMULATION_HORIZON = 161_280  # 20 years in hours (thesis Section 6.8.1)
MAX_ORDERS = 6_000  # j = 1…6,000 (thesis notation)
NUM_RAW_MATERIALS = 12  # rm1…rm12
NUM_SUPPLIERS = 12  # cntr1…cntr12
RATIONS_PER_BATCH = 5_000  # Batch size from Op7 → Op8
RATIONS_PER_SHIFT = int(ASSEMBLY_RATE * HOURS_PER_SHIFT)  # 2,564

# Lead time: time from Op1 to Op13 under deterministic conditions (no risks).
# Sum of processing times Op1–Op12 ≈ 672 + 24×9 + 0 + 0.00312×3×5000 ≈ ~935 hrs
# But thesis uses LT = 48 hours for the last-mile (Op9→Op13) delivery promise.
LEAD_TIME_PROMISE = 48  # Hours — thesis Section 6.3.4

# =============================================================================
# OPERATION DEFINITIONS — Cf0 Baseline (S=1, It,1=0)
# =============================================================================
# From Table 6.20, Figure 6.2, and Sections 6.3.1–6.3.3
#
# Each operation is defined by:
#   name         : Human-readable name
#   pt           : Processing time in hours
#   q            : Order/batch quantity (for S=1)
#   rop          : Reorder point in hours (time between reorder checks/triggers)
#   init_inv     : Initial inventory (It,1 for Cf0)
#   risks        : List of risk types that can affect this operation
#   description  : What this operation does
#   is_assembly  : Whether this is part of the assembly line (Op5/6/7)
#   num_units    : Number of parallel units (e.g., Op2 has 12 suppliers)

OPERATIONS = {
    1: {
        "name": "Military Logistics Agency",
        "description": "Contracting of suppliers (biannual procurement cycle)",
        "pt": 672,  # 1 month to process contracts
        "q": NUM_SUPPLIERS,  # 12 supplier contracts
        "rop": 4_032,  # Biannual = 6 months × 672 hrs/month
        "init_inv": 0,
        "risks": ["R12"],
        "is_assembly": False,
        "num_units": 1,
    },
    2: {
        "name": "Suppliers",
        "description": "Preparation and shipping of raw materials to WDC",
        "pt": 24,  # 1 day to prepare and ship
        "q": 190_000,  # 190,000 units of each rm per month
        "rop": 672,  # Monthly reorder
        "init_inv": 0,
        "risks": ["R13"],
        "is_assembly": False,
        "num_units": 12,  # 12 distinct suppliers
    },
    3: {
        "name": "Warehouse & Distribution Centre",
        "description": "Reception, verification, and storage of raw materials",
        "pt": 24,  # 1 day to process
        "q": 15_500,  # Weekly quantity of each rm (S=1)
        "rop": 168,  # Weekly reorder
        "init_inv": 0,  # It,1 = 0 for Cf0
        "risks": ["R21"],
        "is_assembly": False,
        "num_units": 1,
    },
    4: {
        "name": "Line of Communication (WDC → AL)",
        "description": "Transport and delivery of raw materials to assembly line",
        "pt": 24,  # 1 day transit
        "q": 15_500,  # Weekly quantity of each rm (S=1)
        "rop": 168,  # Weekly
        "init_inv": 0,
        "risks": ["R22"],
        "is_assembly": False,
        "num_units": 1,
    },
    5: {
        "name": "Assembly Line — Pre-assembly",
        "description": "Pre-assembly of high-calorie products for combat rations",
        "pt": 1 / ASSEMBLY_RATE,  # 0.003120 hrs/ration
        "q": 1,  # Continuous flow, 1 at a time
        "rop": 1 / ASSEMBLY_RATE,  # Immediate (continuous)
        "init_inv": 0,  # It,1 = 0 for Cf0
        "risks": ["R11", "R21", "R3"],
        "is_assembly": True,
        "num_units": 1,
    },
    6: {
        "name": "Assembly Line — Assembly",
        "description": "Assembly of combat rations",
        "pt": 1 / ASSEMBLY_RATE,  # 0.003120 hrs/ration
        "q": 1,  # Continuous flow
        "rop": 1 / ASSEMBLY_RATE,  # Immediate (continuous)
        "init_inv": 0,
        "risks": ["R11", "R21", "R3"],
        "is_assembly": True,
        "num_units": 1,
    },
    7: {
        "name": "Assembly Line — QC & Shipping",
        "description": "Quality control and shipping of combat rations",
        "pt": 1 / ASSEMBLY_RATE,  # 0.003120 hrs/ration
        "q": RATIONS_PER_BATCH,  # Ships in batches of 5,000
        "rop": 48,  # Every 2 days (S=1)
        "init_inv": 0,
        "risks": ["R14", "R21", "R3"],
        "is_assembly": True,
        "num_units": 1,
    },
    8: {
        "name": "Line of Communication (AL → SB)",
        "description": "Transport and delivery of combat rations to supply battalion",
        "pt": 24,  # 1 day transit
        "q": RATIONS_PER_BATCH,  # 5,000 rations per shipment
        "rop": 48,  # Every 2 days (S=1)
        "init_inv": 0,
        "risks": ["R22"],
        "is_assembly": False,
        "num_units": 1,
    },
    9: {
        "name": "Supply Battalion",
        "description": "Receipt, classification, and storage of combat rations (SPT rule)",
        "pt": 24,  # 1 day to process
        "q": (2_400, 2_600),  # Range per thesis Figure 6.2: U(2400, 2600)
        "rop": 24,  # Daily
        "init_inv": 0,  # It,1 = 0 for Cf0
        "risks": ["R21", "R3"],
        "is_assembly": False,
        "num_units": 1,
    },
    10: {
        "name": "Line of Communication (SB → CSSUs)",
        "description": "Transport and delivery of combat rations to CSSUs",
        "pt": 24,  # 1 day transit
        "q": (2_400, 2_600),  # per thesis Figure 6.2
        "rop": 24,  # Daily
        "init_inv": 0,
        "risks": ["R22"],
        "is_assembly": False,
        "num_units": 1,
    },
    11: {
        "name": "Combat Service Support Units",
        "description": "Receipt and delivery of combat rations (2 units)",
        "pt": 0,  # Instantaneous (cross-docking)
        "q": (2_400, 2_600),  # per thesis Figure 6.2
        "rop": 24,  # Daily
        "init_inv": 0,
        "risks": ["R23"],
        "is_assembly": False,
        "num_units": 2,  # 2 CSSUs
    },
    12: {
        "name": "Line of Communication (CSSUs → Theatre)",
        "description": "Transport and delivery of combat rations to military personnel",
        "pt": 24,  # 1 day transit
        "q": (2_400, 2_600),  # per thesis Figure 6.2
        "rop": 24,  # Daily
        "init_inv": 0,
        "risks": ["R22"],
        "is_assembly": False,
        "num_units": 1,
    },
    13: {
        "name": "Theatre of Operations",
        "description": "Military personnel consuming combat rations",
        "pt": 0,  # Consumption is instantaneous
        "q": 0,
        "rop": 0,
        "init_inv": 0,
        "risks": ["R24"],
        "is_assembly": False,
        "num_units": 1,
    },
}


# =============================================================================
# DEMAND FUNCTION — Table 6.4
# =============================================================================
# Regular demand (Drg): Uniform distribution U(2400, 2600) rations per day,
# 6 days per week. Placed every 24 hours during operating days.

DEMAND = {
    "distribution": "uniform_discrete",
    "a": 2_400,  # Minimum daily demand (rations)
    "b": 2_600,  # Maximum daily demand (rations)
    "frequency_hrs": 24,  # Demand placed every 24 hours
    "operating_days_per_week": 6,  # Mon–Sat, no demand on Sunday
}


# =============================================================================
# RISK DISTRIBUTIONS — Table 6.12 (current level: '−')
# =============================================================================
# Each risk has: distribution, parameters, affected operations, and recovery.

RISKS_CURRENT = {
    # --- Category 1: Operational risks (R1r) ---
    "R11": {
        "name": "Workstation breakdowns",
        "category": 1,
        "occurrence": {"dist": "uniform", "a": 1, "b": 168},  # hours between events
        "recovery": {"dist": "exponential", "mean": 2},  # hours to repair
        "affected_ops": [5, 6],
    },
    "R12": {
        "name": "Contract delays with suppliers",
        "category": 1,
        "occurrence": {
            "dist": "binomial",
            "n": 12,
            "p": 1 / 11,
        },  # delayed contracts per cycle
        "affected_ops": [1],
    },
    "R13": {
        "name": "Shortages of raw materials",
        "category": 1,
        "occurrence": {
            "dist": "binomial",
            "n": 12,
            "p": 1 / 10,
        },  # delayed deliveries per cycle
        "affected_ops": [2],
    },
    "R14": {
        "name": "Defective products",
        "category": 1,
        "occurrence": {
            "dist": "binomial",
            "n": 2564,
            "p": 3 / 100,
        },  # defects per shift
        "affected_ops": [7],
    },
    # --- Category 2: Natural disasters & intentional attacks (R2r) ---
    "R21": {
        "name": "Natural disasters",
        "category": 2,
        "occurrence": {"dist": "uniform", "a": 1, "b": 16_128},  # hours between events
        "recovery": {"dist": "exponential", "mean": 120},  # hours to recover
        "affected_ops": [3, 5, 6, 7, 9],  # Simultaneous impact
    },
    "R22": {
        "name": "LOC destruction (terrorist attacks)",
        "category": 2,
        "occurrence": {"dist": "uniform", "a": 1, "b": 4_032},  # hours between events
        "recovery": {"dist": "exponential", "mean": 24},  # hours to recover
        "affected_ops": [4, 8, 10, 12],  # All LOCs
    },
    "R23": {
        "name": "Forward unit destruction",
        "category": 2,
        "occurrence": {"dist": "uniform", "a": 1, "b": 8_064},  # hours between events
        "recovery": {"dist": "exponential", "mean": 120},  # hours to recover
        "affected_ops": [11],
    },
    "R24": {
        "name": "Contingent demand surge",
        "category": 2,
        "occurrence": {"dist": "uniform", "a": 1, "b": 672},  # hours between events
        # Surge size: uniform distribution of additional rations (thesis doesn't
        # specify exact bounds for the surge size; to be confirmed with Garrido)
        "affected_ops": [13],
    },
    # --- Category 3: Black-swan events (R3) ---
    "R3": {
        "name": "Black-swan event",
        "category": 3,
        "occurrence": {"dist": "uniform", "a": 1, "b": 161_280},  # ~1 per 20 years
        "recovery": {"dist": "fixed", "duration": 672},  # 672 hrs (1 month) downtime
        "affected_ops": [5, 6, 7, 9],  # Assembly + Supply Battalion
    },
}

# Increased risk levels ('+' in Table 6.12) — for Scenario I configurations
RISKS_INCREASED = {
    "R11": {"dist": "uniform", "a": 1, "b": 42},
    "R12": {"dist": "binomial", "n": 12, "p": 4 / 11},
    "R13": {"dist": "binomial", "n": 12, "p": 4 / 10},
    "R14": {"dist": "binomial", "n": 2564, "p": 8 / 100},
    "R21": {"dist": "uniform", "a": 1, "b": 4_032},
    "R22": {"dist": "uniform", "a": 1, "b": 1_344},
    "R23": {"dist": "uniform", "a": 1, "b": 1_344},
    "R24": {"dist": "uniform", "a": 1, "b": 336},
    "R3": {"dist": "uniform", "a": 1, "b": 80_640},
}

# Severe risk levels ('++') — extrapolated from increased for DOE stress testing.
# Halves the inter-arrival window of each uniform risk (doubles frequency).
# Binomial risks: probability doubled from increased level.
RISKS_SEVERE = {
    "R11": {"dist": "uniform", "a": 1, "b": 21},
    "R12": {"dist": "binomial", "n": 12, "p": 8 / 11},
    "R13": {"dist": "binomial", "n": 12, "p": 8 / 10},
    "R14": {"dist": "binomial", "n": 2564, "p": 12 / 100},
    "R21": {"dist": "uniform", "a": 1, "b": 2_016},
    "R22": {"dist": "uniform", "a": 1, "b": 672},
    "R23": {"dist": "uniform", "a": 1, "b": 672},
    "R24": {"dist": "uniform", "a": 1, "b": 168},
    "R3": {"dist": "uniform", "a": 1, "b": 40_320},
}


# =============================================================================
# INVENTORY BUFFER LEVELS — Table 6.16 (Scenario II)
# =============================================================================
# On-hand inventory at critical storage points (Op3, Op5, Op9) for S=1.
# Key: period in hours → quantities at each operation.

INVENTORY_BUFFERS = {
    168: {"op3_rm": 15_360, "op5_rm": 15_360, "op9_rations": 15_750},
    336: {"op3_rm": 30_720, "op5_rm": 30_720, "op9_rations": 31_500},
    504: {"op3_rm": 46_080, "op5_rm": 46_080, "op9_rations": 47_250},
    672: {"op3_rm": 61_440, "op5_rm": 61_440, "op9_rations": 63_000},
    1344: {"op3_rm": 122_880, "op5_rm": 122_880, "op9_rations": 126_000},
}


# =============================================================================
# MANUFACTURING CAPACITY — Table 6.20 (Scenario III)
# =============================================================================
# Quantities change with number of shifts. Only Op3, Op4, Op7, Op8 change.

CAPACITY_BY_SHIFTS = {
    1: {
        "op3_q": 15_500,  # weekly raw material to AL
        "op4_q": 15_500,
        "op7_q": 5_000,  # batch size from AL
        "op7_rop": 48,  # every 2 days
        "op8_q": 5_000,
        "op8_rop": 48,
        "theoretical_capacity_hrs": HOURS_PER_SHIFT,  # 8 hrs/day
        "theoretical_capacity_rations": RATIONS_PER_SHIFT,  # 2,564/day
    },
    2: {
        "op3_q": 31_000,
        "op4_q": 31_000,
        "op7_q": 5_000,
        "op7_rop": 24,  # daily
        "op8_q": 5_000,
        "op8_rop": 24,
        "theoretical_capacity_hrs": 2 * HOURS_PER_SHIFT,  # 16 hrs/day
        "theoretical_capacity_rations": 2 * RATIONS_PER_SHIFT,  # 5,128/day
    },
    3: {
        "op3_q": 47_000,
        "op4_q": 47_000,
        "op7_q": 7_000,  # Larger batches at full capacity
        "op7_rop": 24,  # daily
        "op8_q": 7_000,
        "op8_rop": 24,
        "theoretical_capacity_hrs": 3 * HOURS_PER_SHIFT,  # 24 hrs/day
        "theoretical_capacity_rations": 3 * RATIONS_PER_SHIFT,  # 7,692/day
    },
}


# =============================================================================
# VALIDATION DATA — Table 6.10
# =============================================================================
# Historical delivered rations (Pt) vs. simulated effective capacity (ECS, S=1)
# Used to validate the deterministic baseline. RMSE = 87,918.

VALIDATION_TABLE_6_10 = {
    "years": [1, 2, 3, 4, 5, 6, 7, 8],
    "Pt_observed": [
        711_808,
        901_131,
        806_454,
        719_344,
        731_016,
        629_429,
        707_203,
        728_878,
    ],
    "ECS_simulated": [
        725_021,
        773_675,
        735_389,
        771_434,
        888_776,
        712_315,
        732_883,
        801_239,
    ],
    "RMSE": 87_918,
}


# =============================================================================
# WARM-UP CONFIGURATION — Section 6.8.2
# =============================================================================
# The warm-up period ends when the first batch of Q=5,000 rations arrives at Op9.
# Under deterministic conditions, this is approximately 838.8 hours (sum of
# processing times Op1–Op9). Data collection starts AFTER this point.

WARMUP = {
    "trigger_op": 9,
    "trigger_quantity": RATIONS_PER_BATCH,
    "estimated_deterministic_hrs": 838.8,
}


# =============================================================================
# RL / ReT APPROXIMATION DEFAULTS
# =============================================================================
# These parameters are repo defaults for the thesis-aligned experimental env.
# They are configurable approximations, not thesis-derived constants.

RET_CASE_THRESHOLDS = {
    "autotomy_fill_rate_threshold": 0.95,
    "nonrecovery_disruption_fraction_threshold": 0.5,
    "nonrecovery_fill_rate_threshold": 0.5,
}

# DOE-calibrated provisional default for the linear shift cost in ReT_thesis.
# Short calibration sweeps under increased risk showed a steep transition band
# between δ≈0.055 and δ≈0.060. We use 0.06 as a conservative default that still
# activates shift choice without collapsing immediately to S=3.
RET_SHIFT_COST_DELTA_DEFAULT = 0.06


# =============================================================================
# SIMULATION OUTPUT SCHEMA — Table 6.25
# =============================================================================
# Columns tracked per order j in the simulation data matrix (SDM).

OUTPUT_COLUMNS = [
    "Cfi",  # Configuration ID
    "j",  # Order number
    "OPTj",  # Order placement time
    "OATj",  # Order arrival time
    "CTj",  # Cycle time = OATj - OPTj
    "LTj",  # Lead time (fixed reference)
    "Bt",  # Cumulative backorders at time t
    "Ut",  # Cumulative unattended orders at time t
    "APj",  # Autotomy period
    "RPj",  # Recovery period
    "DPj",  # Disruption period
    "Rcr_Op",  # Risk type / affected operation
]
