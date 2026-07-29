# Garrido risk-headroom sensitivity freeze

Status: **FROZEN BEFORE DEVELOPMENT SEEDS 7450001-7450006**.

This screen follows Garrido-Rios (2017), Section 6.7.2. The thesis explicitly subjects the
MFSC to increased risk frequencies and gives exact current/increased levels in Table 6.12 plus
the Cf1-Cf20 designs in Tables 6.13-6.14. Risk editing is therefore scientifically authorized.

The black-swan risk R3 is not edited. It is excluded from the R1/R2 category-isolation screen so
that its rare occurrence cannot confound ranking, and no multiplier is ever applied to it.

The governing horizon is 520 weeks (10 thesis years), matching the horizon stated for Tables
6.13 and 6.14. A 52-week smoke may test plumbing but cannot support a risk-sensitivity claim,
because it is too short for the native R21-R23 recurrence rates.

The screen separates three outputs:

1. physical sensitivity: change in canonical ReT and guardrails as risks increase;
2. policy-ranking sensitivity: whether the best constant buffer/shift posture changes;
3. profile-tailoring headroom: the advantage of choosing a constant posture by declared risk
   profile over one robust constant, within each frozen resource cap.

Both raw and safe profile-tailoring values are reported. The safe oracle excludes, separately in
every profile, any posture that worsens ration ReT, CVaR10, lost orders, final backlog, maximum
backlog age, service-loss AUC, or realized resources relative to the robust constant. Guardrail
gains and harms may not cancel across profiles.

The third quantity is not H_PI or H_obs. It is a cheap pre-learner diagnostic. A positive result
can only license a new campaign-control contract with observable regime information and a full
open-loop/classical frontier.

All results, including nulls, will be retained. Cobb-Douglas and the proposed time-resolved
metric are outside this contract. The governing endpoint is `ret_excel_request_snapshot_v2`.
