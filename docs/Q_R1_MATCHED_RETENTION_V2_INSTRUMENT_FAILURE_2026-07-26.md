# Q-R1 matched-retention v2: instrumental failure

## Verdict

`STOP_INSTRUMENT_INVALID_RHO_KAPPA_CONFLATION`

No learning, retention, architecture, or MPC claim may be computed from the v2
screen.

The confirmation block was not opened.

## Defect

The v2 runner used `REGIME_PERSISTENCE = 0.75` when its first development
workers started.

That constant is the within-campaign regime persistence \(\rho\).

The values 0.75 and 0.90 in the Q-R1 dose design are the cross-campaign
knowledge persistence \(\kappa\).

The canonical Q-R1 physical campaigns use \(\rho=0.90\).

Changing \(\rho\) changes the campaign skeleton and therefore invalidates an
exact-frontier lookup against campaigns built at another \(\rho\).

For history 7570801, campaign 6, the skeleton hash begins with `1ab4ec34` at
\(\rho=0.90\) and with `35e0d189` at \(\rho=0.75\).

The pre-correction smoke rows match the latter.

## Custody ruling

The contract did not explicitly freeze \(\rho\).

That omission does not make a post-opening code correction valid.

It makes the physical contract incomplete.

The development roots and the first three optimizer seeds are burned.

The original invalid smoke was restored instead of deleted.

The partial checkpoints and logs were preserved but are not eligible for
selection, aggregation, or a scientific result.

A second launch used the corrected constant but started after the v2 opening
and after the executable changed.

It was also stopped and classified as instrumental evidence only.

The reserved confirmation roots 7660201--7660264 remain unopened.

## Successor requirement

A successor must freeze \(\rho=0.90\) separately from the \(\kappa\) cells.

It must use fresh development and checkpoint-selection roots.

It must state whether it tests:

1. total retained information, including an explicit carried-prior feature;
2. recurrent hidden-state retention conditional on the same carried prior; or
3. both factors in a predeclared factorial.

Those estimands are not interchangeable.

The v2 contract and its results may not be repaired retrospectively.
