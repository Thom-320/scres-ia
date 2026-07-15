# Program O state-rich classical comparator fit freeze

Status: **post-fit development source frozen before evaluating the new family**.

This gate responds to a specific failure in the already-burned label-only fit:
the selected extreme HMM converted observable labels in the primary cell but
failed adjacent worst-product/backlog guardrails and used more loaded freight.
Supplying that policy with each cell's true HMM parameters changed zero of 192
action calendars, so auto-calibration is rejected.  No auto-calibrating HMM is
implemented and the sealed `7420049–7420096` block remains unopened.

The next question is narrower and more adversarial: can a finite, classical
controller using operational inventory, locked production, backlog, in-flight
service and a lagged-label belief beat the complete open-loop frontier without
using more actual transport?

## No new tapes

The only inputs are the custody-verified raw matrices and skeletons already
created for fit seeds `7420001–7420048`.  These tapes are burned development
data.  Any positive is a candidate-selection result only.  The source may not
read or generate a `7420049+` tape.

## Finite family

Exactly ten configurations are evaluated: two balanced base-stock covers, two
max-pressure hysteresis bands, two exact three-slot deficit allocations, MPC
horizons three and four, and approximate belief-DP horizons three and four.
There is no continuous optimization or post-result addition.  Every controller
uses the same fixed HMM parameters `(rho=0.75, share=0.90)` in every cell.

At a weekly decision the controller may observe only events strictly earlier
than the decision timestamp.  The whitelist is product-specific Op9 stock,
committed-but-not-arrived product targets, pending backlog quantity/count/age,
already released in-flight quantity, prior request labels, prior actions and
public phase.  Latent regimes, current/future demand, seed identity, skeleton
future realizations, oracle fields and scoring variables are forbidden.

The compact operational replay is not accepted merely because final metrics
match.  Before fit, emitted calendars are rerun in direct SimPy on burned
preseed `7400048`; on-hand inventory, locked pipeline, backlog quantity/count/
age and in-flight quantity must match at every pre-decision instant under the
strict half-open convention.  A mismatch invalidates the observation contract.

## Comparator and resources

The denominator is all `4^8 = 65,536` open-loop calendars.  The primary
resource contract continues to report identical reserved production and
fixed-clock freight capacity.  A stricter gate is added and is binding:
relative to the maximum-ReT static calendar, policy-minus-static mean loaded
departures, payload and realized vehicle-hours must each be non-positive.

The binding matched-throughput frontier is stricter.  One tape-independent
calendar must use at least as many loaded departures, payload and realized
vehicle-hours as the policy on every paired tape.  The controller must beat the
best mean-ReT calendar in that globally admissible subset.  A different static
may not be selected per tape.  If the subset is empty, the result is a resource
confound.  Therefore a null can identify that the apparent adaptive gain is
conversion of otherwise idle freight rather than intelligence value under
equal realized use.

## Promotion boundary

One configuration is selected on the primary cell only after all metric,
resource, trajectory and information-placebo gates.  Every configuration that
clears the non-placebo primary filter receives the four frozen placebo tests:
state stale by two epochs, no operational state, swapped product state, and a
shift-17 cross-tape state donor.  Real-state ReT must beat each with a paired
10,000-resample one-sided LCB95 strictly above zero.  The selected configuration
is then frozen across the four cells.  A connected three-cell component spanning
both persistence and share axes, including the same placebo gates, is required.

Primary selection also requires two direct state interventions.  Swapping all
product-labelled state channels must complement the action away from ties; and
adding one batch of backlog to the currently minority product must not increase
allocation to the previous majority product.

Even a pass establishes no H_obs.  It authorizes only a separately frozen
validation run on the still-sealed block; learners, Paper 2 confirmation and
Paper 3 remain prohibited.
