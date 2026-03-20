# Resilience Metrics (ReT) System

## Thesis Equations (Section 5.3, pp. 72-76)
Garrido defines the Total Resilience Indicator (ReT) at the **order level**:

- **Eq. 5.1 (Autotomy):** `Re(APj) = Re^max * (APj / LT)`
- **Eq. 5.2 (Recovery):** `Re(RPj) = Re * (1 / RPj)`
- **Eq. 5.3 (Non-recovery):** `Re(DPj, RPj) = Re^min * (DPj - RPj) / CTj = 0`
- **Eq. 5.4 (Fill Rate):** `Re(FRt) = 1 - (Bt + Ut) / Dt`
- **Eq. 5.5 (Total ReT):** Piecewise function selecting one of the above based on the disruption state of order `j`.

**Parameters (Figure 5.6):** `Re^max = 1.0`, `Re = 0.5`, `Re^min = 0.0`. `LT = 48` hours.

## Code Implementation (`env_experimental_shifts.py`)
The code currently implements a **step-level approximation** of ReT rather than the exact order-level evaluation, to better integrate with RL state transitions.

- **Case 2 (Autotomy proxy):** `1.0 - disruption_frac`
- **Case 3 (Recovery proxy):** `1.0 / (1.0 + disruption_frac)`
- **Case 4 (Non-recovery proxy):** `0.0`
- **Case 1 (Fill Rate proxy):** `max(0.0, 1.0 - backorder_qty / demanded)`

**Noted Differences:**
1. Code calculates ReT per simulation step (default 168h) rather than per order.
2. Code uses `Re=1` instead of `Re=0.5` based on reported personal communication with Garrido-Rios.
3. Fill rate uses ration quantities instead of order counts (discrepancy to be adjusted in future versions for exact alignment).
