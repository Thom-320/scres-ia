# Garrido H2/H3 corrective execution — development result

**Verdict:** `PASS_DIRECTIONAL_TAPE_LEVEL` for H2 and H3 in R1r, R2r, and
R3. This is development evidence only. It is not an order-level inference,
confirmation result, or learner claim.

## Instrument identity

- Frozen code commit:
  `2e910b6fa5ba0e48614b8ebf4c13e007d2a1739d`.
- Frozen contract:
  `contracts/garrido_h2_h3_corrective_v1.json`.
- Contract SHA-256:
  `ea283088d528bb9bf9b1cd0e6ab1aac91ff9eca19bac558bbe96c78587dd0c09`.
- Aggregate result SHA-256:
  `319cee735a889a5f98f7fba9719b6e53f2e1ba5230d4a865ebdc43d77c8d6ccb`.
- Matrix: 90 configurations × 12 paired tape roots = 1,080 unique rows.
- Evaluation starts at hour 2,016 for every configuration.
- Confirmation roots opened: `false`.

The estimand is computed in two stages. Within each tape and risk family, it
averages the ten paired differences `Cf(b+30)-Cf(b)` for H2 or
`Cf(b+60)-Cf(b)` for H3. Inference then uses the 12 tape-level means, not the
thousands of correlated order rows.

## Primary results

The primary endpoint is `ret_excel`. Relative changes are descriptive ratios
to the paired baseline mean; confidence intervals are Student-t intervals over
the 12 tape-level means.

| Family | Contrast | Baseline | Treated | Mean delta | 95% CI | Positive tapes | Relative |
|---|---|---:|---:|---:|---:|---:|---:|
| R1r | H2 buffers | 0.004736 | 0.005256 | +0.000520 | [+0.000510, +0.000530] | 12/12 | +10.98% |
| R1r | H3 shifts | 0.004736 | 0.005089 | +0.000353 | [+0.000343, +0.000362] | 12/12 | +7.45% |
| R2r | H2 buffers | 0.544009 | 0.676685 | +0.132675 | [+0.127686, +0.137664] | 12/12 | +24.39% |
| R2r | H3 shifts | 0.544009 | 0.621213 | +0.077204 | [+0.073567, +0.080842] | 12/12 | +14.19% |
| R3 | H2 buffers | 0.955858 | 0.992631 | +0.036773 | [+0.035908, +0.037637] | 12/12 | +3.85% |
| R3 | H3 shifts | 0.955858 | 0.980023 | +0.024164 | [+0.023641, +0.024687] | 12/12 | +2.53% |

All six directional gates pass. H2 is larger than H3 in every family under
this design, and the effect magnitude remains strongly risk-family dependent.

## Population and physical checks

`generated_orders` has an exact mean paired delta of zero in all six panels.
The improvement therefore does not come from evaluating treatments on a larger
generated demand population.

| Family | Contrast | Full-ledger ReT delta | Fill-rate delta | Delivered-rations delta | Unresolved delta |
|---|---|---:|---:|---:|---:|
| R1r | H2 buffers | +0.000920 | +0.087504 | +614,271.5 | −237.11 |
| R1r | H3 shifts | +0.000651 | +0.067513 | +473,926.2 | −182.94 |
| R2r | H2 buffers | +0.147385 | +0.091884 | +687,037.8 | −265.54 |
| R2r | H3 shifts | +0.098000 | +0.062112 | +462,853.8 | −178.81 |
| R3 | H2 buffers | +0.023369 | +0.016837 | +239,424.9 | −92.16 |
| R3 | H3 shifts | +0.017795 | +0.012466 | +177,276.9 | −68.24 |

Thus the direction is corroborated by the uncensored ledger and physical
service endpoints: treatments deliver more rations, raise fill, and leave
fewer unresolved orders.

## Corrective requirements exercised

1. **Cf2 seed.** The source audit recovers and validates `Cf2=91`.
2. **Periodic buffers.** All 360 H2 rows show replenishment after the initial
   target; buffers are not one-time initial injections.
3. **Common evaluation origin.** All 1,080 rows use hour 2,016.
4. **Separate populations.** Every row records generated, scored/visible,
   omitted, served, unresolved, and lost counts separately.
5. **Workbook quarantine.** Cf1 and Cf2 are quarantined from source validation
   because their workbooks span about 20 years; Cf5 is quarantined because it
   ends near 8.94 years. The prospective generated matrix itself uses the
   common frozen horizon and evaluation origin.
6. **Table 6.20 traces.** Preflight passed exact Q/ROP behavior:
   Op3 weekly start gaps with total raw-material quantities 186,000, 372,000,
   and 564,000 for shifts 1–3; Op7 batch/gap pairs 5,000/48 h,
   5,000/24 h, and 7,000/24 h.
7. **Neutral identity.** All 108 H3 pairs whose shift setting remains at one
   are exactly identical across every mandatory endpoint.
8. **Tape-level inference.** Twelve paired roots are the inferential units;
   order rows are never treated as independent replicates.

## Custody

The six complete local shards have valid completion receipts, exact and
disjoint roots `9420000` through `9530000`, matching row hashes, the same code
commit and contract hash, and 1,080/1,080 unique `(tape_root, cf)` identities.

Two earlier VPS shards were stopped for inferior throughput before completing
any 90-configuration tape. Their directories remain preserved and are not
action-eligible; no row from them enters this result. The replacement
directories were new and never reused.

The canonical machine-readable evidence is under
`results/garrido_h2_h3_corrective_v1/`, including all source rows, per-shard
receipts, tape-level deltas, the trace preflight, source-workbook audit,
aggregate result, and aggregate receipt.

## Claim boundary and next decision

The corrected execution supports:

> Under the thesis-design development contract, periodic inventory buffers
> (H2) and added shift capacity (H3) improve the Garrido ReT endpoint in all
> three risk families, with concordant gains in uncensored service measures.

It does **not** yet support:

- causal generalization beyond this reconstructed DES;
- independent replication of the thesis workbooks order by order;
- an architectural or neural advantage;
- a confirmatory H2/H3 claim;
- treating `ret_excel` as the sole adequate resilience endpoint.

The next scientifically justified step is a separately frozen confirmation or
metric adjudication. These results do not authorize KAN, PPO, DDMRP, or the
full authority ladder by themselves.
