# Preregistration — Program O relevant-risk sensitivity (v1)

**Date frozen:** 2026-07-17 · **Status:** FROZEN BEFORE IMPLEMENTATION AND ANY EXECUTION
**Contract:** `contracts/program_o_relevant_risk_sensitivity_v1.json`
**Contract SHA-256 (prefix):** `cdd9c6a0c397aeff`

## The idea (PI directive, Garrido's own method)

The wartime atlas died of compute because it swept every risk × every coupling × a 50,202-policy
comparator. Garrido never did that: **he activated only the risks relevant to the operations
under study** (thesis §6.4/§6.7, one-factor frequency escalation). We now know — from the
exhaustive search itself — which decision variable matters: **the Program O product mix on
Op5–Op7** (the only confirmed observable adaptive value on canonical ReT; D1's rationing is a
static win; buffers/shifts are constants). So the Garrido-style move is: activate exactly the
risks that hit the mechanism's operations, and map what happens to the mechanism's value.

**Sharpener:** all of Program O ran with `risks_enabled=False`. The one confirmed conversion has
never seen the thesis's own operational risks. This screen answers the robustness question the
manuscript needs either way.

## Design in one paragraph

Full-DES Program O physics; 18 frozen configs: risks-off baseline (identity anchor),
thesis-current (all five relevant risks at φ=ψ=1), one-at-a-time φ∈{1,2,4} for each of
R11/R14/R21/R22/R24 (the risks mapped to Op5–Op7 / Op8–Op12 / Op13), and one combined φ=2
profile. R12/R13/R23 excluded with reasons; **R3 frozen, untouched**. 12 burned development
tapes per config (7420001–12, already opened), CRN within config. Policies: the **byte-identical
frozen belief-MPC** (no retuning), constant k, and a restricted periodic-calendar family.
Primary: `H_dev(config)` = frozen controller − best restricted comparator on canonical
`ret_excel_request_snapshot_v2`. **Screening asymmetry frozen:** the restricted comparator biases
H upward, so `H_dev ≤ 0` kills a config *a fortiori*, while `H_dev > 0` is hypothesis-only and
can never promote. CVaR10 and anti-shed panels reported (never gates). Gates: G0 risks-off
identity, G1 risk-fires-on-mapped-ops fixture (R3 count = 0 everywhere), G2 all-cells-reported.
Compute ≈ 9,720 episodes ≈ 3–14 h local. No sealed tapes, no 747 seeds, no promotion, no learner.

## What this is and is not

**Is:** the robustness/generalization map for the paper's Program O chapter, and the evidence base
for risk-cell selection in any FUTURE learner contract (adjudicated separately, after M2 and the
CVaR instrument audit). **Is not:** a rescue of any closed record (`STOP_PROGRAM_O_AFTER_
CORRECTIVE_VALIDATION` and the risk screen's posture-invariance null are untouched — different
estimands), nor a reopening of the compute-infeasible atlas (whose caps and seeds stay frozen).

## Roles

Implementer: PI agent. Independent verifier: concurrent auditor — verifies G0 identity, the risk
fixture, and frozen-controller byte-identity BEFORE the map is read.
