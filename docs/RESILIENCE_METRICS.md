# Resilience Metrics (ReT) System

## Thesis Equations (Section 5.3, pp. 72-76)
Garrido defines the Total Resilience Indicator (ReT) at the **order level**:

- **Eq. 5.1 (Autotomy):** `Re(APj) = Re^max * (APj / LT)`
- **Eq. 5.2 (Recovery):** `Re(RPj) = Re * (1 / RPj)`
- **Eq. 5.3 (Non-recovery):** `Re(DPj, RPj) = Re^min * (DPj - RPj) / CTj = 0`
- **Eq. 5.4 (Fill Rate):** `Re(FRt) = 1 - (Bt + Ut) / Dt`
- **Eq. 5.5 (Total ReT):** Piecewise function selecting one of the above based on the disruption state of order `j`.

**Parameters (Figure 5.6):** `Re^max = 1.0`, `Re = 0.5`, `Re^min = 0.0`. `LT = 48` hours.

## Code Implementation Split

The repository now has two distinct ReT surfaces. Keep them separate:

1. **Training/step proxy** in `env_experimental_shifts.py`.
   This is a step-level approximation used inside RL transitions.
2. **Paper-facing order-level audit/evaluation** in `supply_chain/ret_thesis.py`
   and `MFSCSimulation.compute_order_level_ret()`.
   This computes Garrido-style order-level ReT using the thesis weights.

### Training Proxy (`env_experimental_shifts.py`)

The step-level proxy is not the exact order-level thesis metric. It exists to
provide dense feedback during RL training.

- **Case 2 (Autotomy proxy):** `1.0 - disruption_frac`
- **Case 3 (Recovery proxy):** `1.0 / (1.0 + disruption_frac)`
- **Case 4 (Non-recovery proxy):** `0.0`
- **Case 1 (Fill Rate proxy):** `max(0.0, 1.0 - backorder_qty / demanded)`

**Noted Differences:**
1. Code calculates ReT per simulation step (default 168h) rather than per order.
2. Code uses `Re=1` instead of `Re=0.5` based on reported personal communication with Garrido-Rios.
3. This proxy uses ration-flow quantities for the step fill term.

### Order-Level Evaluation (`supply_chain/ret_thesis.py`)

The order-level path is the current paper-facing evaluation surface:

- `DEFAULT_RET_WEIGHTS = {"max": 1.0, "mean": 0.5, "min": 0.0}`.
- `MFSCSimulation._order_level_fill_rate()` implements Eq. 5.4 with order
  counts: `1 - (Bt + Ut) / Dt`.
- `compute_order_level_ret()` reports the completed-order mean used by legacy
  summaries.
- `scripts/audit_garrido_metric_saturation.py` adds stricter diagnostics:
  all-order ReT with unfulfilled orders as zero, component contributions,
  `ret_p10_all`, `ret_p50_all`, `stockout_week_pct`, `flow_fill_rate`, and
  `period_weighted_ret_proxy`.

## Interpretation Rule

Do not judge policy victory from the dense training reward or from a saturated
mean alone. For Track A, compare against the same-panel best static baseline
using:

- `ret_mean_all_orders_zero_unfulfilled`
- `flow_fill_rate`
- `stockout_week_pct`
- `ret_p10_all`
- dynamic component contributions: `Re(APj)`, `Re(RPj)`, `Re(DPj,RPj)`

If mean ReT is high while `ret_p10_all`, stockout weeks, or dynamic components
are poor, the policy is not genuinely resilient.
