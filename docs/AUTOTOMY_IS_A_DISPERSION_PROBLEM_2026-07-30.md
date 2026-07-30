# The autotomy share is not controllable by the delay — it needs dispersion

**Status:** `DEVELOPMENT_DIAGNOSTIC_NO_CONSTANT_CHANGED`. Fine sweep run on
`ovh-agent-lab`, retrieved immutably to
`results/metric_audit/fidelity_fine_48_49_v1_FROM_VPS/` (read-only), grid marked
`grid_is_the_frozen_contract_grid: false`.

## The measurement

Ten delays between 48.0 and 49.0, both transport modes, both families, 12 roots:

| delay | R1r autotomy share | R2r | reference R1r / R2r |
|---:|---:|---:|---|
| **48.000** | **0.65200** | **0.20009** | 0.00436 / 0.00064 |
| 48.005 | **0.00000** | 0.00000 | |
| 48.010 … 49.000 | 0.00000 | 0.00000 | |

**The transition is a discontinuity at exactly 48.0, not a gradient.** Five thousandths of
an hour — eighteen seconds — past the lead time, the share falls from 0.652 to exactly
zero. Both transport modes agree, so it is not a transport effect.

## Why: our `CTj` is a point mass

On the on-hand path `CTj = delay` exactly, for every order, with zero variance. So the
predicate `CTj <= LTj` is not a per-order property at all — it is a **single global
comparison, `delay <= 48`**. Every risk-touched order is autotomy, or none is. There is no
intermediate share to tune.

Garrido's share is 0.44% because **his `CTj` varies across orders**: 48.00744, 48.048, and
onward. The share is the fraction of his `CTj` distribution falling inside the tolerance
band. With a point mass that fraction can only be 0 or 1.

## What this settles

**The autotomy share cannot be calibrated by any delay.** The delay sweep is closed for
good: not because no value is close, but because the quantity is a step function of the
delay and takes only two values, neither of which is 0.44%.

**What is needed is dispersion in `CTj`, not a different constant.** That is exactly the
change I first proposed as "fix 1" — replace the flat constant with modelled transit — and
this says precisely why it is the right one: not because 48 is the wrong number, but
because *a constant has no distribution*, and the target moment is a property of a
distribution.

**It also gives the fix an acceptance test that did not exist before.** The transit
replacement is correct when the autotomy share lands near 0.44% (R1r) and 0.064% (R2r),
and it is wrong if the share stays at 0 or 1 regardless of the parameters — which would
mean the new transit is still degenerate.

## Corrections this carries forward

The earlier framing — "no delay reproduces autotomy because none can, this is structural"
— was right in its conclusion and wrong in its reasoning, and both errors are now visible.
The conclusion holds: no delay reproduces it. The reasoning was that risk implies lateness,
which was measured with the wrong proxy and is false. The true reason is degeneracy of the
`CTj` distribution, which is a *modelling* defect and is fixable.

Two of my claims in one day were right for the wrong reason. Recording that here because
the pattern matters more than either instance: a conclusion that survives its own
refuted argument still needs its reasoning replaced, not just its verdict kept.
