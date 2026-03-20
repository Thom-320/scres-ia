# Risk Model Documentation

Based on Garrido-Rios (2017) Table 6.12 and Section 6.6.

## R11: Workstation Breakdowns
- **Definition:** Minor technical failures in the assembly line.
- **Thesis Parameters:** Occurrence ~ U(1, 168) hrs; Recovery ~ Exp(β=2) hrs. Affected: Op5, Op6.
- **Code Implementation:** `_risk_R11()` in `supply_chain.py`. Matches thesis exactly.

## R12: Bureaucratic Delays
- **Definition:** Delays in contract approvals.
- **Thesis Parameters:** B(n=12, p=1/11) delayed contracts. Penalty: +168 hrs per contract. Affected: Op1.
- **Code Implementation:** `_risk_R12()`. Matches thesis exactly.

## R13: Supplier Delays
- **Definition:** Delays in raw material delivery.
- **Thesis Parameters:** B(n=12, p=1/10) delayed deliveries. Penalty: +24 hrs per delivery. Affected: Op2.
- **Code Implementation:** `_risk_R13()`. Matches thesis exactly.

## R14: Quality Defects
- **Definition:** Defective products detected during QC.
- **Thesis Parameters:** B(n=2564, p=3/100) defects per shift (S=1). Affected: Op7. Items returned to Op6 for re-processing.
- **Code Implementation:** `_risk_R14()`. Defective items are removed from `_pending_batch` rather than explicitly re-processed (documented simplification).

## R21: Natural Disasters
- **Definition:** Severe environmental disruptions.
- **Thesis Parameters:** Occurrence ~ U(1, 16128) hrs; Recovery ~ Exp(β=120) hrs. Affected: Op3, Op5, Op6, Op7, Op9.
- **Code Implementation:** `_risk_R21()`. Recovery processes block the next occurrence event (conservative approximation).

## R22: Attacks on LOCs
- **Definition:** Enemy attacks on Lines of Communication.
- **Thesis Parameters:** Occurrence ~ U(1, 4032) hrs; Recovery ~ Exp(β=24) hrs. Affected: ONE random LOC from {Op4, Op8, Op10, Op12}.
- **Code Implementation:** `_risk_R22()`. Correctly targets exactly one random LOC.

## R23: Attacks on CSSUs
- **Definition:** Direct attacks on forward units.
- **Thesis Parameters:** Occurrence ~ U(1, 8064) hrs; Recovery ~ Exp(β=120) hrs. Affected: Op11.
- **Code Implementation:** `_risk_R23()`. Matches thesis exactly.

## R24: Contingency Demand (Surge)
- **Definition:** Unexpected surge in demand.
- **Thesis Parameters:** Occurrence ~ U(1, 672) hrs; Surge ~ U(2400, 2600) rations. Affected: Op13.
- **Code Implementation:** `_risk_R24()`. Configured in `config.py`. Matches thesis exactly.

## R3: Black Swan Event
- **Definition:** Massive, prolonged disruption.
- **Thesis Parameters:** Occurrence ~ U(1, 161280) hrs; Duration = 672 hrs fixed. Affected: Op5, Op6, Op7, Op9.
- **Code Implementation:** `_risk_R3()`. Correctly affects Op5, Op6, Op7, Op9. (Op3 is excluded, correctly aligning with thesis).
