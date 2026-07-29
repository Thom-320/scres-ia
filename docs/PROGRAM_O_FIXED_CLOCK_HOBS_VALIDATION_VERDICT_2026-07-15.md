# Program O fixed-clock H_obs validation verdict

## Corrective status

**RETRACTED AS A TERMINAL ADJUDICATION.** The causal audit found that the executor selected the point comparator on the sealed validation tapes even though the contract froze the development-selected comparator. It also found an invalid unstandardized simultaneous critical value. See `PROGRAM_O_RHO90_SHARE90_CAUSAL_DIAGNOSTIC_2026-07-15.md`.

The raw trajectories, matrices, ledgers, actions, and custody remain valid burned evidence. The automatic PASS/STOP adjudication does not. H_obs remains unconfirmed pending one corrective validation; no learner is authorized.

## Original verdict (retracted)

`STOP_PROGRAM_O_CLASSICAL_HOBS_VALIDATION`

Program O does not confirm classical observable headroom and does not authorize a learner. The prospective validation opened seeds `7420049–7420096` exactly once and the frozen primary rule failed: `rho90_share90` was favorable on only 26 of 48 tapes, below the preregistered minimum of 34. The other two cells were favorable on 45/48 and 44/48 tapes.

This is a terminal failure under the frozen contract. There will be no new controller, hyperparameter change, cell removal, physics change, risk addition, inventory addition, metric substitution, or reuse of this validation block.

## What survived audit

- Mean canonical visible-ReT deltas against the full 65,536-calendar frontier were positive in all three cells: `0.08380`, `0.06879`, and `0.08289`.
- The physical replay completed 1,423 episodes with zero failures and exactly one scheduled-resource vector. Fixed-clock missions, vehicle-hours, crew-hours, payload capacity, production, and setup resources were equal by construction.
- The action audit passed in every cell, with 42, 48, and 34 unique sequences. State counterfactuals also passed.
- Every one of the 27 individually reported real-minus-placebo contrasts had a positive paired one-sided LCB. These are descriptive because the required familywise inference implementation was defective.

These findings show a strong but unstable adaptive signal. They do not override the prospective failure in tape-level consistency.

## Corrective statistical audit

The executor's simultaneous-inference implementation is invalid. It computed one raw-scale maximum deviation across 69 heterogeneous estimands, including ReT on a unit scale and service-loss AUC on a tens-of-millions scale. The resulting critical value was `23,651,319.39`, and that same number was subtracted from every estimand. Consequently, the generated simultaneous LCBs and the automatic `primary_pass`, `placebo_pass`, and `guardrail_pass` flags are not scientifically interpretable.

This defect does not rescue Program O. The 26/48 favorable-tape result is an exact, preregistered, inference-free failure. No corrected confidence procedure is being used post hoc to promote the result. The defect is recorded so the paper does not falsely claim that placebos or guardrails failed when the executor did not estimate their simultaneous bounds correctly.

## Custody

- Scientific source commit: `48062f8a52dda8417812cac828978906704a58fa`
- Custody harness commit: `4c686c080e34c6ba391a0f73cad8460b68ae989e`
- Executed repository commit: `de576d4fb1f1560ee0b61e780ff9503726a36a1d`
- Contract SHA-256: `d13508f6694a96dd219cf411a6cc14f1ef4031540805df34b55941289f14d9cc`
- Result SHA-256: `09ec3f1691d6e5da090d7afb62523124321de666d4441ea2f92f619cb8326362`
- Retrieved manifest SHA-256: `8b2bf7eae368e0c3b8c25e450fdfdb748ddfc13b931ccec0528944e19532e353`
- Raw calendar shards: 144 (48 tapes by 3 cells)
- Remote and retrieved checksum verification: pass
- Producer exit: 0; watcher reached `COMPLETE_PENDING_RETRIEVAL`

The seed claim names the frozen scientific source commit, while the launcher correctly records and verifies the later repository commit containing the finalized execution freeze. This is consistent with the two-commit freeze design and is not evidence of source drift.

## Claim boundary

- `H_obs` confirmed: no
- Learned advantage confirmed: no
- Learner authorized: no
- Paper 2 confirmed: no
- Paper 3 authorized: no
- Program O state: closed, no rescue

The publication-safe result is a boundary finding: the fixed-clock, nonfungible product-mix mechanism has material mean headroom and genuine state-dependent trajectories, but the frozen belief-MPC does not deliver the required robust tape-level advantage across the connected validation region.
