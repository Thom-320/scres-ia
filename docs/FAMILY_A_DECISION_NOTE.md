# Family A Decision Note

Updated: 2026-03-31

## Decision

The repository will treat **Family A** as the current primary paper-facing benchmark family.

Family A is defined as:

- `reward_mode=ReT_seq_v1`
- `ret_seq_kappa in {0.10, 0.20}`
- `observation_version=v1`
- `frame_stack=1`
- `year_basis="thesis"`
- `step_size_hours=168`
- `risk_level="increased"` with cross-eval on `severe`
- `stochastic_pt=True`

This family is selected because it is the only matched, auditable, post-audit benchmark family that currently exists in the repo.

## Valid evidence

Primary auditable bundles:

- `outputs/paper_benchmarks/paper_ret_seq_k020_500k`
- `outputs/paper_benchmarks/paper_ret_seq_k010_500k`
- `outputs/paper_benchmarks/paper_control_v1_500k`

Secondary auditable comparator:

- `outputs/benchmarks/final_ret_seq_v1_500k`

Use caution with the secondary comparator:

- it is post-audit and auditable,
- but it uses `year_basis="gregorian"`,
- so it is not a matched replacement for the thesis-basis paper bundles.

## What is excluded

Do not use these as primary evidence for the current repository state:

- `docs/artifacts/control_reward/control_reward_500k_increased_stopt`
- `docs/artifacts/control_reward/control_reward_500k_severe_stopt`
- old seed-inference notes tied to the `*_stopt` family
- `section4_3_*` planned directories that do not yet exist as auditable bundles
- `outputs/paper_benchmarks/paper_trio_analysis/analysis.md` as a source of truth snapshot

Reason:

- these artifacts are either historical, invalid, unmatched, or superseded by the post-audit bundle review.

## Interpretation rule

Within Family A:

- compare `paper_ret_seq_k020_500k` and `paper_ret_seq_k010_500k` directly
- use `paper_control_v1_500k` as the operational comparator lane
- use `fill_rate`, `backorder_rate`, and `order_level_ret_mean` as the common cross-mode metrics

Do not:

- compare raw rewards across `control_v1` and `ReT_seq_v1`
- mix Family A (`v1/ReT_seq`) with Family B (`v4/control_v1/RecurrentPPO`) when making source-of-truth claims

## Current reading

- `ReT_seq_v1, kappa=0.20` is the leading Family A lane.
- `ReT_seq_v1, kappa=0.10` is the conservative ablation.
- `control_v1` remains a valid comparator, but not the leading paper-facing lane.
- No current valid RL lane clearly dominates `static_s2` on service in both increased and severe settings.

This is acceptable.

The contribution can be framed as:

- auditable DES+RL benchmark design,
- reward-resilience alignment,
- matched benchmark discipline,
- and an honest negative or near-negative result.

## Next-step policy

Until an explicit new decision note supersedes this one:

1. Do not launch new Family A variants that change the action contract.
2. Do not treat Family B runs as evidence for Family A claims.
3. Wait for the current `RecurrentPPO` run only as Family B evidence.
4. Base repo-facing writing and evidence tables on the Family A bundles listed above.

## Immediate next steps

1. Wait for `RecurrentPPO` to finish and classify it as Family B evidence only.
2. Regenerate any stale summaries that still quote the obsolete trio snapshot or historical `*_stopt` artifacts.
3. Build a final Family A comparison table from:
   - `paper_ret_seq_k020_500k`
   - `paper_ret_seq_k010_500k`
   - `paper_control_v1_500k`
   - `static_s2`
4. Freeze manuscript- or repo-facing claims around the Family A interpretation above.
