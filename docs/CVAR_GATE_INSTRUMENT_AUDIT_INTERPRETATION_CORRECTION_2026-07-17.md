# Correction to the interpretation of the Program O CVaR gate audit

## Status

`CORRECTED_INTERPRETATION_NUMERICAL_AUDIT_RETAINED`

This note preserves `STOP_PROGRAM_O_AFTER_CORRECTIVE_VALIDATION`, the data, and the numerical
power analysis committed in `69cca7e`. It retracts only the label that the instrument was
technically defective.

## Retained findings

- The audit reproduced the original simultaneous critical value and published CVaR10 bounds.
- Roughly `+0.079` genuine tail improvement was required for about 80% joint rejection probability
  in the frozen 48-tape, three-cell design.
- The privileged oracles passed.
- Program O had positive tail point estimates but did not establish the preregistered joint
  tail-safety claim in two cells.

## Corrected interpretation

At a zero noninferiority margin, requiring a one-sided 95% lower bound above zero is a superiority
test at the boundary. A true zero effect is expected to pass near the type-I error rate, not with
high probability. The low pass rate therefore demonstrates stringent geometry and low power for
small effects; it does not demonstrate a bootstrap defect or authorize a corrective rerun.

Program O-R derives authority exclusively from its separately frozen ReT-primary contract. M2 may
change external-validity and deployment language, but it is not the source of Program O-R's
internal research authorization.
