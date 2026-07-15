# Restricted PI timing ceiling — execution freeze

This gate asks one narrow causal question: after a Garrido-authorized recurrent-risk
regime is selected without looking at timing outcomes, does physically constrained
daily intervention timing improve the canonical request-snapshot ReT over constants,
weekly open-loop timing, and a weekly clairvoyant comparator?

The quantity is named `restricted_PI_timing_ceiling`. It is not total perfect-information
headroom and it is not observable headroom. It selects only between two Track-A postures
already frozen by the preceding risk screen.

## Ordering

1. The ten-year risk screen must finish, pass custody, and select the first passing
   frequency regime under the contract order. A running job is not evidence.
2. If no frequency regime passes, seeds 7460001–7460048 remain unopened and this gate stops.
3. If a regime passes, freeze the low/high postures from the risk result at resource cap 0.5.
4. Only then may the timing seeds be claimed and executed once.

## Physics

- Daily review is an opportunity, not a mandatory rewrite.
- `HOLD` advances 24 hours with the previous request and creates no top-up.
- Shift requests ramp upward by at most one level per day and draw the existing surge budget.
- Strategic-buffer requests activate after 168 hours, cannot be cancelled or overwritten,
  and are limited to one new commitment per week.
- No arbitrary monetary churn cost is introduced.

The same-time convention is explicit: a buffer commitment due at a daily review boundary
activates immediately before that boundary's observation. Risk events at the boundary are
materialized before the observation and decision. Future event data are available only to
the restricted privileged oracle, never to the EWMA trigger.

## Metrics and comparators

Canonical `ret_excel_request_snapshot_v2` remains primary. The risk-cluster temporal panel
is report-only and cannot select or rescue a policy. The comparator set is all 18 constants,
all 256 eight-week periodic low/high calendars, the restricted weekly clairvoyant schedule,
an EWMA/hysteresis trigger, and shuffled/stale/cross-tape placebos.

The privileged schedule uses entry offsets `{-168,-72,-24,0,+24,+72}` hours around realized
R22/R24 onsets and returns to the low posture 72 hours after event end. Overlapping high
windows are unioned. Between schedule boundaries the action is `HOLD`.

Promotion requires an increment of at least 0.01 **on canonical
`ret_excel_request_snapshot_v2`** over the strongest comparator, paired
LCB95 above zero, at least 34/48 favorable tapes, multiple intervention times, no resource
increase, and every service/fairness guardrail noninferior. A failure closes this timing
family without changing cadence, offsets, postures, impacts, or metrics.

A difference confined to the temporal secondary panel can support only a recovery-rapidity
analysis. It cannot open observable validation, a learner, or Paper 2. The first augmented
regime is selected before timing results in the fixed order R24-increased, R22-increased,
then R22+R24-increased; current is the comparator rather than a selectable stress rescue.

The development result also reports `eta_timing`: the EWMA/hysteresis increment divided by
the restricted privileged increment against the same strongest nonobservable comparator.
This ratio diagnoses convertibility on burned tapes; it is not confirmed `H_obs`.

No learner, Paper 2 positive claim, or Paper 3 work is authorized by this freeze.
