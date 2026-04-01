# Track B Source of Truth

This note freezes the current Track B contract, the artifact bundle that carries
the result, and the language discipline required to discuss it honestly.

## Frozen experimental contract

- Environment: `track_b_adaptive_control`
- Reward mode: `ReT_seq_v1`
- `ret_seq_kappa`: `0.20`
- Observation version: `v7`
- Action contract: `track_b_v1`
- Risk level: `adaptive_benchmark_v2`
- Year basis: `thesis`
- Step size: `168` hours
- `stochastic_pt=True`
- Training seeds: `11 22 33 44 55`
- Training timesteps: `500000`
- Evaluation episodes: `20`
- Static comparators: `s2_d1.00`, `s3_d1.00`, `s3_d2.00`

## Primary artifact bundle

- Run bundle:
  `outputs/track_b_benchmarks/track_b_ret_seq_k020_500k_rerun1`
- Posthoc resilience audit:
  `outputs/track_b_benchmarks/track_b_ret_seq_k020_500k_rerun1/posthoc_resilience_audit`
- Results package:
  `outputs/track_b_benchmarks/track_b_results_package`

Required files:

- `command.txt`
- `status.json`
- `heartbeat.json`
- `manifest.json`
- `summary.json`
- `policy_summary.csv`
- `comparison_table.csv`
- `posthoc_resilience_audit/summary.json`
- `track_b_results_package/results_discussion_package.md`

## Primary metrics

For the main Track B claim, use:

- `fill_rate`
- `backorder_rate`
- `order_level_ret_mean`
- `terminal_rolling_fill_rate_4w`

For resilience audit/reporting, use:

- `ret_thesis_corrected_total`
- `ret_unified_total`
- order-case shares:
  `fill_rate`, `autotomy`, `recovery`, `non_recovery`, `unfulfilled`

Interpretation rule:

- `order_level_ret_mean` is the strongest policy-level resilience discriminator in
  the current Track B result.
- `ret_unified_total` is comparable within Track B because all policies share the
  same reward family.
- `ret_thesis_corrected_total` is a reporting-only audit metric here; it does not
  show the same separation magnitude as `order_level_ret_mean`.

## Allowed claim

Allowed:

> Under the frozen Track B contract, PPO beats the strongest static baselines in
> service and order-level resilience metrics.

Also allowed:

> Track B provides strong evidence that exposing downstream control can unlock RL
> advantage in this MFSC benchmark.

## Prohibited claim

Do not claim:

- `Track B proves the causal mechanism is downstream control alone`
- `Track B and Track A are the same benchmark except for one fix`
- `ret_thesis_corrected_total by itself explains the full Track B gain`
- `the Garrido framework doubles under PPO`

## Causality: CLOSED via matched ablation

The matched ablation (same v7, same adaptive_benchmark_v2, same ReT_seq_v1, same 100k budget) isolates the action-contract factor:

- **5D (track_a actions):** PPO fill=0.607, S2 fill=0.616 → PPO LOSES by 0.9pp
- **7D (track_b actions):** PPO fill=0.964, S2 fill=0.616 → PPO WINS by +34.8pp

Artifact: `outputs/track_b_ablation_5d_vs_7d.json`

Conclusion: the downstream control dimensions (Op10/Op12), not richer observation or different risk, are materially responsible for the Track B gain.

**Remaining limitation:** The ablation is at 100k × 3 seeds (quick test), not 500k × 5 (production). It identifies the role of downstream control within the matched benchmark family; it does not claim universal optimality outside that contract.

## Current repo stance

- Track A remains the thesis-faithful negative family.
- Track B is the current positive paper lane.
- The matched ablation closes the action-contract causality question.
- The correct paper claim is:

  > A matched 5D-vs-7D ablation shows that the added downstream control
  > dimensions, not merely richer observation or a different risk profile,
  > are materially responsible for the Track B gain.
