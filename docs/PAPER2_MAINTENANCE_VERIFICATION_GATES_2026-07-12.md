# Paper 2 Maintenance Verification Gates

The implementation fails closed unless all of the following hold before any
learner or virgin tape is opened:

1. Mass conservation and finite-WIP blocking/starvation.
2. One crew, no impossible overlap, and exactly 24 scheduled maintenance hours per week.
3. Bitwise-identical consumed demand, wear, R11 candidate and R14 innovation ledgers across actions.
4. Observation keys exactly equal the contract whitelist; forbidden future or latent fields are absent.
5. PM5/PM6/PM7 each changes a future physical state from the same prefix.
6. At least two actions have support of 15%; no action exceeds 85%.
7. Oracle-static ReT delta at least 0.01 with positive CI95 lower bound.
8. Service-loss reduction at least 5%, with no lost-order or tail contradiction.
9. First-action agreement at least 90% between four- and eight-week horizons.
10. A deployable tree, hysteresis or belief-MPC captures at least 30% of oracle headroom and beats the frozen periodic comparator on fresh holdout.

Only `PROMOTE_MAINTENANCE_TO_LEARNER` opens learner training. Otherwise the
machine verdict is `STOP_NO_OBSERVABLE_MAINTENANCE_HEADROOM`.
