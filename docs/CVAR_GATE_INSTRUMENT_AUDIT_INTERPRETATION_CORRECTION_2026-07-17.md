# Correction to the interpretation of the Program O CVaR gate audit

## Status

`CORRECTED_INTERPRETATION_NUMERICAL_AUDIT_RETAINED`

This note does not alter the historical verdict
`STOP_PROGRAM_O_AFTER_CORRECTIVE_VALIDATION`, the corrective-validation data, or
the numerical output committed in `69cca7e`. It corrects one statistical claim
made from that output.

## What remains valid

- M0 reproduced the original simultaneous critical value and the two published
  CVaR10 lower bounds.
- The power curve is useful: under the frozen 48-tape, three-cell design, a
  genuine tail improvement of roughly `+0.079` was required for about 80% joint
  rejection probability.
- The privileged oracles passed the zero-margin gate.
- The observed Program O tail point estimates were positive, while their
  simultaneous lower bounds crossed zero in two cells. Thus the contract did
  not establish its preregistered joint tail-safety claim.

## What was wrong

The audit declared an instrument defect because a truly equivalent policy had
less than 0.5 probability of passing a one-sided 95% lower-bound test at a zero
margin. That rule is invalid. At the boundary `delta = 0`, such a test rejects
the null only at approximately its type-I error rate, not with high
probability. Requiring a lower confidence bound above zero is mathematically a
superiority requirement at the boundary, even if it is labelled
"non-inferiority with margin zero".

Consequently, low pass probability for a zero-effect control demonstrates low
power for small effects and the geometry of the zero-margin test; it does not
demonstrate a technical defect in the bootstrap or justify a corrective rerun.

## Correct claim boundary

The defensible conclusion is:

> The frozen zero-margin simultaneous CVaR10 gate was stringent and
> underpowered for small tail effects at 48 tapes. It remained the valid gate
> for the completed historical contract, which therefore stopped. The audit's
> numerical power analysis may inform a prospectively designed study, but it
> does not invalidate or reopen Program O.

Program O-R is a separately preregistered, ReT-primary successor. Its authority
does not derive from an alleged CVaR instrument defect.

