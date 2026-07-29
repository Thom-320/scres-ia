# Garrido Cf1-Cf20 replication verdict

## Verdict

`FORMULA_RECONSTRUCTION_PASS__DES_GENERATIVE_REPLICATION_FAIL`

The two delivered thesis workbooks are internally reproducible: all 47,546
order-level `ReT` values are recovered exactly from their visible columns.
The current repository DES does **not**, however, numerically regenerate the
Cf1-Cf20 outputs from the published configuration plus workbook seed.

This is a historical-source validation exercise. It does not open Paper B
selection roots and does not resolve the future Paper B endpoint.

## Sources and custody

| source | sha256 |
|---|---|
| `Raw_data1+Re.xlsx` | `30b88c9b9fe68ef527dbfcc70d8e653ea7bd152ab891b3fc0ecf53cb6f043486` |
| `Raw_data2+Re.xlsx` | `4bd462771fefff16fc5666a851256b3780198d474832dec1423c0b6f94be86b0` |

The canonical run used commit `d4d937eaf27d3e8b1da391e7c22647826db63911`
from a clean worktree. Its result digest is
`dfaacf44ec13d5dc2344638f949615477390291134ee4a2c3e17925d5e15c8f2`.

`Rsult_1.xlsx` is not used as a validation target. Its configuration count,
row population and values do not match the two delivered raw thesis
workbooks, and no provenance link to the published tables has been established.

## What reproduced

The workbook formula

```text
IF(
  AVERAGE(risk_cols) > 0,
  IF(APj > 0, APj / LT, 0.5 * (1 / RPj)),
  1 - ((sumBt + sumUt) / j)
)
```

reproduced exactly:

| check | result |
|---|---:|
| workbook rows | 47,546 |
| formula mismatches | 0 |
| maximum absolute difference | 0 |
| seeds recovered | 20/20 |
| Cf2 seed | 91 |

This proves that the repository understands the spreadsheet calculation. It
does not prove that the DES regenerates the physical trajectories that supplied
the spreadsheet columns.

## Generative DES comparison

The generative run imported no Excel demand, timing or risk-attribution tape.
For each configuration it used the published design, the workbook seed and the
repository's thesis-faithful protocol.

| family | mean Excel ReT | mean DES ReT | bias | MAE | max abs gap | within-family Pearson r |
|---|---:|---:|---:|---:|---:|---:|
| Cf1-Cf10 / R1 | 0.006282 | 0.004286 | -0.001996 | 0.001996 | 0.004702 | 0.084 |
| Cf11-Cf20 / R2 | 0.200742 | 0.464936 | +0.264194 | 0.264194 | 0.380634 | 0.585 |

The pooled correlation (`r=0.917`) is not evidence of replication: it is driven
mainly by the large scale difference between the R1 and R2 families. The
within-family correlations are the relevant diagnostics.

Order counts are closer than ReT:

- mean relative error: +2.67% in R1 and +1.10% in R2;
- Cf5 is the main count outlier at +13.66%;
- R2 ReT remains grossly high despite order counts being within 1.23% in every
  configuration.

The discrepancy is therefore not reducible to a missing configuration table or
an incorrect seed. The next fidelity diagnosis must inspect R2 occurrence,
attribution, recovery-period and visible-ledger semantics, while separately
examining the Cf5 demand/order-count outlier.

No numerical acceptance tolerance was preregistered for this historical audit.
The verdict is descriptive: discrepancies of 0.15-0.38 in individual R2 ReT
means are too large to support a numerical-replication claim under any
plausible material tolerance. This is not a post-hoc Paper B selection gate.

## Cf21-Cf90

The published **input design** is regenerable. The independent matrix report
checks all 90 rows and reports 90 matches and zero mismatches across risk
patterns, buffers, shifts, horizons and source-row inheritance.

That does not make the missing outputs regenerable as thesis ground truth:

- Cf1-Cf20 seeds are known.
- Therefore the inherited seeds for Cf31-Cf50 and Cf61-Cf80 are known.
- The original seeds for Cf21-Cf30 were not delivered.
- Therefore the inherited seeds for Cf51-Cf60 and Cf81-Cf90 are also unknown.
- More importantly, the generative DES has not reproduced Cf1-Cf20 outputs.

Status:

`HOLD_FIDELITY_NOT_ESTABLISHED`

Cf21-Cf90 may be represented as a dry-run design manifest. They must not be
reported as regenerated thesis results, used to select Paper B mechanisms, or
interpreted as validation evidence until the fidelity gap is closed or a new
prospective contract explicitly treats them as new simulations.

## What “metric unresolved” means

It does **not** mean that the historical Excel metric is unknown. For Cf1-Cf20,
the 2017 `ReT` formula and its row population are resolved and reproduced
exactly.

It means the future Paper B primary endpoint remains unfrozen. The audit has
already shown that the historical metric can censor a policy-dependent
population and lose service ordering under risk. Paper B therefore still needs
a prospective decision on the primary endpoint, population, out-of-range
handling and Cobb-Douglas physical/cost semantics before any mechanism or
architecture is selected.

Historical fidelity and future metric validity are different questions:

1. Reproduce the old metric to validate lineage.
2. Do not automatically reuse it as the future optimization target.

## Reproduction

```bash
python scripts/reproduce_garrido_cf1_cf20.py \
  --output-dir results/garrido_reproduction/cf1_cf20_v1
```

Canonical artifacts:

- `results/garrido_reproduction/cf1_cf20_v1/result.json`
- `results/garrido_reproduction/cf1_cf20_v1/rows.csv`
- `results/garrido_reproduction/cf1_cf20_v1/completion_receipt.json`
- `results/garrido_reproduction/cf1_cf90_published_design_v1/`
