# Preventive Reserve v3 — physical-lead preregistration

**Status:** frozen before powered evaluation

**Parent:** `preventive_reserve_v2`

**Single authorized change:** replace fixed 336 h replenishment with the actual
downstream path: Op10 24 h, Op11 availability, Op12 24 h.

All other elements remain frozen from v2:

- reserve targets 0/15k/30k and capacity 30k;
- R22/R23 corridor events only;
- 336 h warning, FN 0.20, false-warning opportunity 0.20;
- shuffled-time placebo;
- daily decisions;
- service-loss AUC, Excel ReT, actual inventory-time, and physical liveness;
- static 0/15k/30k comparators;
- no PPO before the gate.

## Gate universe

```text
30 new paired tapes
seed_start = 773000
52 weeks
balanced R2/mixed and current/increased
```

## Promotion rules — all required

1. Perfect warning improves service-loss AUC by at least 5% over static zero,
   CI95 wholly above zero, uses no more than static-15k inventory-time +2%, and
   is Pareto non-dominated by the static frontier.
2. Imperfect warning improves service-loss AUC by at least 5% over its shuffled
   placebo, CI95 wholly above zero.
3. Reserve stock is physically issued on at least 50% of perfect-warning tapes.
4. Imperfect warning is Pareto non-dominated by the static frontier on service
   loss, actual inventory-time, and Excel ReT.

Failure is terminal for this proposed downstream-reserve/alert class. No lead,
capacity, target, warning-error, risk-frequency, or reward tuning follows.
