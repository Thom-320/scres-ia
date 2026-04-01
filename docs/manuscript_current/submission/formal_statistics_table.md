# Formal Statistics

## Track B Pairwise Seed-Level Comparisons

| Comparator | Metric | PPO mean | Baseline mean | PPO - baseline | CI95 | Wilcoxon p | Sign-flip p | Cohen d | Wins |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| s2_d1.00 | fill_rate | 0.99996 | 0.96592 | +0.03404 | [+0.03359, +0.03447] | 0.0625 | 0.0625 | +57.76 | 5/5 |
| s2_d1.00 | reward_total | 254.21453 | 178.23807 | +75.97646 | [+75.61888, +76.32754] | 0.0625 | 0.0625 | +162.56 | 5/5 |
| s2_d1.00 | order_level_ret_mean | 0.95030 | 0.48860 | +0.46170 | [+0.45334, +0.46786] | 0.0625 | 0.0625 | +47.24 | 5/5 |
| s3_d2.00 | fill_rate | 0.99996 | 0.98765 | +0.01231 | [+0.01125, +0.01334] | 0.0625 | 0.0625 | +8.73 | 5/5 |
| s3_d2.00 | reward_total | 254.21453 | 171.05721 | +83.15731 | [+81.98245, +84.09576] | 0.0625 | 0.0625 | +62.15 | 5/5 |
| s3_d2.00 | order_level_ret_mean | 0.95030 | 0.45842 | +0.49188 | [+0.48478, +0.49760] | 0.0625 | 0.0625 | +58.36 | 5/5 |

## Matched Ablation

| Comparison | 7D fill | 5D fill | Difference | 7D reward | 5D reward | Note |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 7D vs 5D | 0.96376 | 0.60681 | +35.7 pp | 250.48 | 177.04 | Aggregate-only row. Per-seed paired tests require the underlying seed-level ablation export. |

## Interpretation Note

The paired seed-level tests are the defensible inferential unit for the frozen Track B bundle because PPO and the static comparators share the same evaluation seeds. With only five shared seeds, the exact two-sided p-values remain coarse even when PPO wins on every seed; this is why the table reports direction, effect size, and wins alongside p-values rather than overclaiming significance.
