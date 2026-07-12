# Program D D1-ReT — observable convertibility gate

Status: **FROZEN BEFORE TREE FITTING OR SEQUENTIAL EVALUATION** on 2026-07-11.
This is a new estimand requested after the terminal D1-v2 result. It does not
alter or supersede `STOP_NO_STATE_DEPENDENT_RATIONING_HEADROOM`, which remains
the answer to the original multi-endpoint question.

## Question and evidence boundary

Can a depth-3 policy tree using deployable observations convert the positive
28-day branching oracle for `ret_excel_visible_v1` into a sequential policy that
beats the frozen best constant `spt_flat`? This gate tests Garrido's narrower
criterion: dynamic policy ReT greater than static-policy ReT. It does not claim
service improvement, retained learning, prevention, or deployable RL value.

The only inputs are the already-open selection/validation tapes 810001-811030
and the frozen D1 visible-ledger branching artifacts. Virgin seeds
820001-820060 remain unopened. No PPO is authorized.

## Model and cross-fitting

- Labels: 28-day branch action with maximum visible ReT; frozen tie-breaks are
  service loss, lost orders, then enum order.
- Model: `DecisionTreeClassifier(max_depth=3, random_state=0)`.
- Five `GroupKFold` folds grouped by tape. No state from a held-out tape enters
  its fitted tree.
- Features: the frozen observable D1 state fields only: SB inventory; queue
  quantity/count/occupancy/contingent share; size and age quartiles; oldest age;
  in-transit stock; current Op9-Op12 downtime; recent demand, delivery and fill;
  prior rule; operational day. Family, risk level, future risk/demand/repair,
  tape identity and outcomes are excluded.
- Each fold's tree is executed sequentially, once per day, over its held-out
  tapes. Classification accuracy alone is never a promotion endpoint.

## Frozen comparator, endpoint and sensitivities

- Comparator: `spt_flat`, selected before this gate.
- Primary: paired tape-level delta in `ret_excel_visible_v1`.
- Required sensitivity: paired delta in `ret_excel_visible_clipped_0_1`.
- Guardrails: lost orders, service-loss AUC, backlog AUC, conservation and
  exogenous identity. The policy may not worsen lost orders, service loss, or
  backlog AUC by more than 2% at the upper CI95 bound.

## Promotion rule

Emit `PROMOTE_D1_RET_OBSERVABLE_POLICY` only if all hold:

1. primary delta CI95 lower bound is above zero;
2. clipped delta CI95 lower bound is above zero;
3. positive primary delta on at least 70% of tapes;
4. cross-fitted predictions capture at least 50% of the held-out branching
   oracle ReT headroom;
5. upper CI95 relative degradation is at most 2% for lost orders, service loss,
   and backlog AUC;
6. mass conservation and exogenous hashes pass on every held-out tape.

Otherwise emit `STOP_D1_RET_NOT_OBSERVABLY_CONVERTIBLE`. A pass is sufficient
for Garrido's narrow dynamic-greater-than-static ReT criterion, but not for the
original D1 service claim and not for retained learning.
