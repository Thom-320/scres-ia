# Program O dual-resource diagnostic — independent verdict

**Verdict:** `PASS_AS_POST_HOC_DIAGNOSTIC_ONLY__PROSPECTIVE_FIXED_CLOCK_VALIDATION_REQUIRED`

The new run finished normally and is reproducible under its frozen identity. It
establishes a stable *development signal* under the fixed-clock reserved-fleet
estimand and a null under pay-per-use transport accounting. It does not confirm
\(H_{obs}\), reopen the stopped source fit, authorize the sealed validation
block, authorize a learner, confirm Paper 2, or authorize Paper 3.

## Custody and provenance

- Scientific commit: `a9733d033b0ac0969ab9ccc994219bda2b66215e`.
- Result SHA-256: `e48606e79c4fcbcc7a1f2955476057d2cd9432b95a0bac7337f38f5b44cba535`.
- Source-result SHA-256: `d67ac97a359a307ca632b6a13493e3ff5940a97e9440a6bc4b7d77c08a147875`.
- Parent-result SHA-256: `2de51be686539ccc801566a2886d7b0ec27ff1e30a8694eb5a0a2f15fc7a9cd0`.
- The producer exited zero, the watcher observed an empty terminal process group,
  remote checksums passed before retrieval, and the retrieved files passed the
  same checksums locally.
- Exactly the 48 burned tapes `7420001`–`7420048` were used. The result and its
  source both report no validation access; `7420049`–`7420096` remain sealed.

## Independent numerical audit

The audit reconstructed all 360 controller-cell-placebo contrasts directly from
the immutable parent raw matrices and the recorded calendar indices. Every mean,
favorable-tape count, deterministic 10,000-resample paired LCB95, unique-sequence
count, and pass Boolean matched the result exactly.

It also independently recomputed all 40 controller-cell rows: the full metric
guardrail vector, reserved-resource equality, strict actual-use test, globally
matched transport frontier, complete action-trajectory statistics, recorded
state-counterfactual certificate, diagnostic gate logic, and connected-component
logic. No mismatch was found.

## What passed

Under the **fixed-clock reserved-fleet** estimand, both frozen controllers
`belief_mpc__3` and `belief_mpc__4` pass on the same connected three-cell region:

| Controller | Cell | ΔReT vs full open-loop frontier | Favorable tapes | Smallest placebo LCB95 |
|---|---|---:|---:|---:|
| belief_mpc__3 | rho75_share90 | 0.06481 | 39/48 | 0.00814 |
| belief_mpc__3 | rho90_share75 | 0.07173 | 43/48 | 0.00818 |
| belief_mpc__3 | rho90_share90 | 0.10125 | 39/48 | 0.01617 |
| belief_mpc__4 | rho75_share90 | 0.06613 | 40/48 | 0.01082 |
| belief_mpc__4 | rho90_share75 | 0.07014 | 44/48 | 0.00535 |
| belief_mpc__4 | rho90_share90 | 0.09935 | 40/48 | 0.01370 |

For these six rows:

- all total-information placebos pass;
- all operational-state-given-current-belief placebos pass;
- all belief-given-current-operational-state placebos pass;
- metric guardrails pass and reserved production/transport capacity is exactly
  equal;
- complete trajectories contain 44–48 unique eight-week sequences, three or
  four materially supported action levels, and modal fractions of only
  0.021–0.083;
- both state counterfactual families pass every tested non-tie state.

This corrects the earlier unsupported interpretation that the state-rich signal
collapsed to regime belief. In this diagnostic, operational state has positive
incremental value conditional on current belief for the two belief-MPC policies.

The fourth cell, `rho75_share75`, is not eligible: its mean ΔReT is positive, but
CVaR10 deteriorates by about 0.0061, so the simultaneous guardrail gate correctly
fails.

## What did not pass

No controller passes the **pay-per-use** estimand. In every otherwise passing
belief-MPC row, strict actual-use and the globally matched transport frontier
fail. Relative to the best-ReT open-loop schedule, the policies use approximately:

- 422–818 additional realized downstream vehicle-hours;
- 8.79–17.04 additional loaded departures;
- 22,213–42,719 additional payload units.

Those quantities remain inside the equal 5,376-hour reserved envelope, but they
are additional resources if freight is paid per realized use. Therefore the
resource-accounting interpretation is load-bearing and cannot be blurred in a
paper claim.

## Scientific boundary and next gate

This is post-hoc evidence on burned development tapes. The strongest valid
statement is:

> A frozen belief-MPC family shows stable observable and incremental operational-
> state value over the complete open-loop frontier on three adjacent Program O
> cells when downstream fleet capacity is treated as a fixed-clock reserved
> envelope; the same family does not pass when realized transport is charged as
> pay-per-use.

Before any sealed tape is opened, a separate prospective validation protocol must
freeze (1) the operationally justified fleet-cost estimand, (2) one primary
controller rather than two co-winners, (3) the three-cell component and all
placebos/guardrails, and (4) a rule that validation failure is terminal. Until
that protocol exists and passes on untouched tapes, `H_obs`, Paper 2, the learner,
and Paper 3 remain blocked.
