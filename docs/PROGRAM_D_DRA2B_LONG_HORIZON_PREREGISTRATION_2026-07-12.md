# Program D DRA-2b — long-horizon finite-convoy preregistration

Status: **FROZEN BEFORE DRA-2b TAPES OR POLICY FITTING**.

## Separation from DRA-2

DRA-2 remains terminal at `dra2-stop-2026-07-12`. DRA-2b is a new exploratory
study motivated by, but not confirmatory of, DRA-2's finding of material
resource-adjusted oracle headroom with failed 7d/10d horizon sufficiency. No DRA-2
tape contributes to a DRA-2b estimate, model fit, threshold or headline claim.

The physical contract is unchanged. It remains a stylized researcher-imposed
finite-convoy extension of the thesis-grounded MFSC, not a recovered operational
representation.

## Fixed decision horizon

The primary open-loop decision window is 14 days. The sensitivity window is 10
days. Outcomes are evaluated after both 28 and 56 days using the same frozen
static continuation.

The exact search enumerates canonical feasible strings only. Because a dispatch
makes the sole convoy unavailable at the next daily epoch, any nominal string
containing adjacent dispatch actions is physically equivalent to one in which the
second dispatch is HOLD/masked. Therefore all binary strings without adjacent
dispatches form an exact cover of physically distinct action requests:

```text
10 days: 144 canonical strings
14 days: 987 canonical strings
```

Route outages or empty staging may collapse further strings into the same realized
pattern; realized-pattern hashes, not nominal strings, are used for diversity.

## States and data

- 60 new calibration tapes, seeds 860001–860060, 15 per family.
- Two primary states per tape: 120 total.
- Four prefix policies rotate across tapes; state sampling is not conditioned on a
  calibration-winning policy.
- Twelve sensitivity states, exactly three per family and from distinct tapes.
- 40 new observable holdout tapes, seeds 870001–870040, remain closed until the
  oracle and cross-fitted tree gates pass.
- PPO-confirmatory tapes begin at 890001 and remain untouched until every earlier
  gate passes.

## Corrected sufficiency estimand

Total oracle headroom normally grows with the number of available decisions, so
DRA-2b does not demand equality of raw 10d and 14d totals. It requires:

1. first-action agreement at least 90%;
2. relative change in headroom per decision-day no more than 20%;
3. absolute change no more than 0.0005 ReT per decision-day;
4. no positive-to-negative reversal (zero-to-positive is reported but is not a
   negative reversal);
5. consistent service direction at 28d and 56d.

These rules are frozen before generating seeds 860001+.

## Resource and outcome gates

The primary comparator is selected once on calibration: the best same-contract
static policy using no more mean departures and no more mean unavailable-hours
than the candidate. Pareto status is secondary.

Promotion to an observable depth-3 tree requires all of:

- strong liveness at least 20%;
- HOLD and DISPATCH each supported in at least 15% of states;
- ReT improvement at least 0.01 with CI95 lower bound above zero;
- service-loss reduction at least 5% with CI95 lower bound above zero;
- no lost-order contradiction;
- horizon sufficiency;
- resource-envelope validity.

Promotion from tree to PPO additionally requires the cross-fitted tree to capture
at least 50% of oracle headroom, beat the frozen static comparator on the 40-tape
holdout, be favorable on at least 70% of holdout tapes and not worsen tail risk.

If any pre-tree gate fails, emit `STOP_DRA2B_PRE_TREE_GATE`. If the tree fails,
emit `STOP_DRA2B_NO_OBSERVABLE_CONVERTIBILITY`. PPO is trained only after an
explicit machine verdict `PROMOTE_DRA2B_TO_PPO`.
