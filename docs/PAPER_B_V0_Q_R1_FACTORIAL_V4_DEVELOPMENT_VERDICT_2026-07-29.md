# Paper B v0 — Q-R1 factorial v4 development verdict

Date: 2026-07-29

## Terminal development status

`STOP_NO_NEURAL_PREMIUM_AND_IID_PLACEBO_GATE_FAILED`

The frozen screen completed with exact 8 × 3 coverage and advanced `s07` and
`s06`. The frozen full phase then completed with exact 2 × 5 coverage. All ten
workers passed custody checks: full checkpoint schedule `0..240000` by `24000`,
roots `7670101..7670116`, kappas `0.5/0.75/0.9`, exact arm counts, common
checkpoint hashes across neural arms, immutable shared static and structured
bars, and `confirmation_roots_opened=false`.

The full selector chose `s07`:

| quantity | development estimate |
|---|---:|
| mean selected-checkpoint `P1_H1` | 0.74526583 |
| total retained neural treatment, `P1_H1 - P0_H0` | +0.02270006 |
| descriptive seed-level 95% interval | [+0.02058239, +0.02481774] |
| recurrent residual given explicit context | +0.02023648 |
| explicit-context value | +0.00246358 |
| mean absolute iid effect | 0.00913120 |
| neural premium, `P1_H1 - structured_retained` | -0.02128824 |
| descriptive seed-level 95% interval for neural premium | [-0.02269393, -0.01988255] |

The retained treatment passed its point, descriptive interval, positive-seed,
and dose-response checks. It failed the frozen iid-placebo bound:
`0.00913120 > 0.005`. The neural premium failed decisively in all five seeds
and in every kappa cell (`-0.01190`, `-0.02640`, `-0.02556`).

Therefore:

- the factorial does **not** authorize confirmation roots `7670201..7670264`;
- it does **not** authorize a KAN/MLP/DMLPA architectural bakeoff;
- it does **not** establish retained-learning superiority over the strongest
  tested structured retained controller;
- the positive retained-versus-reset contrast remains development evidence
  confounded by a placebo effect larger than the frozen tolerance.

## Frozen H1-H4 mapping

### H1 — learning effect

The preregistered primary recovery-time endpoint was not recorded by this
runner, so the primary H1 estimand is not adjudicable. Both recorded secondary
directions are adverse:

- service-loss improvement: `-1,149,028.56`, with descriptive seed-level 95%
  interval `[-2,536,196.94, +238,139.82]`;
- worst-product-fill improvement: `-0.02341910`, with interval
  `[-0.04122702, -0.00561118]`.

Status: `PRIMARY_NOT_ADJUDICABLE_RECOVERY_TIME_NOT_RECORDED`.

### H2 — adaptation over successive campaigns

The campaign-index slope contrast `P1_H1 - P0_H0` is negative in 5/5 optimizer
seeds:

`-0.00220772`, descriptive 95% interval
`[-0.00276072, -0.00165473]`.

Status: `DIRECTION_NOT_SUPPORTED`.

### H3 — volatility reduction

The retained arm has lower dispersion:

- mean-absolute-deviation reduction: `+0.01100252`, descriptive 95% interval
  `[+0.00891607, +0.01308898]`;
- variance ratio `P1_H1 / P0_H0`: `0.90695474`, interval
  `[0.86864342, 0.94526606]`.

Status: `DIRECTIONAL_DEVELOPMENT_SUPPORT`. This is not confirmation and does
not rescue H1, H2, H4, or the neural-premium gate.

### H4 — path dependency / accumulated learning

The total retained treatment is positive in 5/5 seeds and increases with
kappa:

`-0.00913` at `0.50`, `+0.02542` at `0.75`, and `+0.05181` at `0.90`.

However, the iid cell itself is adverse and its absolute effect exceeds the
frozen `0.005` placebo tolerance. The composite development-retention gate
therefore fails.

Status: `FAIL_DEVELOPMENT_RETENTION_GATE`.

## Claim boundary

This is a completed **development** adjudication, not a confirmatory result.
It answers the retained-learning lane honestly: recurrent state improves the
reset neural ablation in persistent cells, but the effect is not clean under
the iid placebo and the neural policy is materially worse than the structured
retained controller. Paper B should retain H3 as a bounded development result
and report H1/H2/H4 as unsupported or non-adjudicable under this contract.

Cobb-Douglas remains sensitivity-only and did not select configurations or
checkpoints. Submission A was not modified.

## Custody

- Full selection:
  `results/q_r1/matched_retention_factorial_v4_development/full_selection.json`
  (`sha256:9c742ffffa4e84a8d4efd67966465b8465c56295bbe259c52125c4d844a084ec`)
- H1-H4 adjudication:
  `results/q_r1/matched_retention_factorial_v4_development/paper_b_v0_h1_h4_adjudication.json`
  (`sha256:33a35583525973dd5c97c4a0373f23cde1e97a9e0108d25f3d6ceaa82e6a70e5`)
- Hypothesis mapping:
  `contracts/paper_b_v0_hypothesis_mapping_v1.json`
  (`sha256:7a13decdd0e6d1ef95d3ca4cb11b8844bfa7d4be5f0e2fcc5c1134625dae0a79`)
- Confirmation roots opened: `false`
