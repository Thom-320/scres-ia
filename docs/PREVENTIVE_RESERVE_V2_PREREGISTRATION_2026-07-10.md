# Preventive Reserve v2 — preregistration

**Status:** frozen before powered evaluation

**Contract:** `preventive_reserve_v2`

**Parent environment:** `garrido_proxy_v1`

**Relationship to prior work:** new decision class; it does not revise the
terminal Program L result or any Track A/B/C claim.

## 1. Scientific question

Can an imperfect, observable warning create deployable preventive value when a
finite emergency reserve is physically positioned behind the downstream
Op10–Op12 corridor, or is the apparent value reproduced by static stock and a
time-shuffled warning?

This is not a retained-learning test. PPO remains prohibited until the
non-learning headroom gate passes.

## 2. Physical intervention

- Target risks: R22/R23 events that affect Op10, Op11, or Op12.
- Reserve location: theatre, downstream of the threatened corridor.
- Capacity: 30,000 rations (two thesis-demand weeks).
- Daily order-up-to actions: 0, 15,000, or 30,000 rations.
- Replenishment source: existing Op9/SB inventory; no injection.
- Lead time: 336 h.
- Route rule: dispatch and arrival require Op10, Op11, and Op12 to be open.
- Issue rule: reserve serves new theatre demand only while that corridor is
  physically down.
- Local issue delay: 24 h.
- Costs: actual stored inventory-time plus target changes. Resource outcomes
  remain explicit; no post-result choice of a favorable scalar weight is
  permitted.

The reserve is a disclosed extension. It is not part of Garrido's original DES.

## 3. Warning contract

The true warning begins 336 h before the realized onset of an eligible event.
The imperfect warning has:

- false-negative probability 0.20 per eligible event;
- an independent false-warning opportunity probability 0.20 per eligible event;
- false warnings placed at event-free times;
- no disclosure of risk id, severity, duration, or affected operation.

The shuffled placebo preserves warning count and duration but shifts warning
times by half a campaign. It must be generated from the same tape and warning
seed as the imperfect arm.

## 4. Gate design

Powered calibration gate:

```text
30 paired tapes
52 weeks per tape
balanced R2/mixed and current/increased
arms: static 0 / static 15k / static 30k /
      perfect warning / imperfect warning / shuffled placebo
```

Primary endpoint: `service_loss_auc_ration_hours` (lower is better).  Key
co-outcomes: `ret_excel`, actual reserve inventory-time, reserve units issued,
backlog and fill.

### Promotion rules — all required

1. Perfect-warning relative service improvement over static zero is at least 5%
   and its paired tape-bootstrap CI95 is wholly above zero; its actual
   inventory-time is no more than 2% above static 15k; and it is Pareto
   non-dominated by the static frontier. This is a resource-efficiency upper
   bound, not an attempt to beat permanent maximum stock on service alone.
2. Imperfect-warning relative service improvement over the shuffled placebo is
   at least 5% and its paired CI95 is wholly above zero.
3. Perfect-warning reserve stock is physically issued on at least 50% of tapes.
4. The imperfect-warning arm is Pareto non-dominated by the three static arms
   on service loss, actual inventory-time, and Excel ReT.

Terminal outcomes:

- `STOP_NO_PREVENTIVE_HEADROOM`
- `STOP_WARNING_NOT_ACTIONABLE`
- `STOP_RESERVE_MECHANISM_NOT_LIVE`
- `STOP_ALERT_POLICY_STATIC_DOMINATED`
- `PROMOTE_TO_PPO_PILOT`

A screen run with `--screen-only` has no promotion authority.

## 5. Required falsification and V&V

- no reserve service when the downstream corridor is open;
- no replenishment stock creation;
- mass residual remains zero;
- same tape and actions reproduce the same trajectory;
- warning generation is deterministic by seed;
- placebo preserves alert count but destroys timing;
- only corridor-affecting R22/R23 events can generate true warnings;
- frozen Track A/B/L tests remain green.

## 6. Claim ladder

- Perfect warning fails: the new physical lever has no useful preventive
  headroom under this contract.
- Perfect passes, imperfect fails placebo: information is not actionable at the
  stated error rate.
- Imperfect passes placebo but is static-dominated: prediction has signal but no
  deployable resource advantage.
- Full gate passes: authorize only a PPO pilot. It does not yet prove retained
  learning, prevention in reality, or architectural superiority.

## 7. Prohibited changes after powered execution

No changes to reserve location, capacity, lead, action levels, warning error
rates, eligible risks, primary endpoint, thresholds, or tape universe after the
30-tape gate is opened. Any such change creates `preventive_reserve_v3` with a
new preregistration.

## 8. Pre-powered amendment after diagnostic screen

A non-promotional 6-tape × 26-week screen (`seed_start=772900`) exposed an
ill-posed draft rule: perfect warning had been required to improve service over
static 30k. Since static 30k holds the maximum reserve throughout the campaign,
that rule rewarded resource excess rather than preventive efficiency. Before
opening the powered universe (`seed_start=772000`), rule 1 was corrected to the
resource-matched formulation above. The screen did **not** support promotion:
imperfect warning versus shuffled placebo was +1.64% with CI95 crossing zero,
and static 15k dominated the imperfect arm. Rules 2–4 were not relaxed.
