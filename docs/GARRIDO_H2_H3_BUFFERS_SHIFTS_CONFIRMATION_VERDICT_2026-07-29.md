# Garrido H2/H3 buffers and shifts — confirmation verdict

Date: 2026-07-29

## Verdict

`CONFIRM_H2_H3_ALL_SIX_PANELS`

The fresh confirmation executed all 90 configurations on 12 preregistered
paired tapes (1,080 rows). All four shards had identical frozen code, contract,
source-file hashes and execution commit. Development roots were not reopened.

This confirms the resource-intervention directions in the thesis-grounded
reconstructed DES:

- **H2 buffers:** periodic buffer replenishment improves the primary ReT
  endpoint in R1r, R2r and R3.
- **H3 shifts:** the intention-to-treat shift intervention improves the primary
  ReT endpoint in R1r, R2r and R3.

It does **not** establish dynamic feedback value, learned adaptation, retained
learning, MPC saturation or neural/KAN superiority.

## Primary tape-level results

| family | intervention | mean Δ `ret_excel` | 95% interval | positive tapes |
|---|---|---:|---:|---:|
| R1r | H2 buffer | +0.00053542 | [+0.00052081, +0.00055003] | 12/12 |
| R1r | H3 shift | +0.00036416 | [+0.00035067, +0.00037765] | 12/12 |
| R2r | H2 buffer | +0.12844155 | [+0.12384565, +0.13303746] | 12/12 |
| R2r | H3 shift | +0.07582819 | [+0.07334975, +0.07830663] | 12/12 |
| R3 | H2 buffer | +0.03623303 | [+0.03527370, +0.03719235] | 12/12 |
| R3 | H3 shift | +0.02359898 | [+0.02275944, +0.02443852] | 12/12 |

Holm step-down over the six one-sided primary tests passes all six panels.

## Mandatory physical concordance

Every panel also passes all preregistered concordance gates:

- `ret_excel_full_ledger` lower 95% bound above zero;
- flow fill lower 95% bound above zero;
- delivered rations lower 95% bound above zero;
- unresolved-orders upper 95% bound below zero;
- generated-orders paired delta exactly zero in all 12 tapes.

The 108 H3 pairs with `S=1` are exactly identical, which validates the neutral
intention-to-treat embedding.

## Required metric caveat

`ret_thesis` agrees in R1r and R2r but remains unresolved in R3:

| R3 sensitivity | mean Δ `ret_thesis` | 95% interval | positive tapes |
|---|---:|---:|---:|
| H2 buffer | +4.57e-7 | [-6.13e-7, +1.53e-6] | 8/12 |
| H3 shift | +3.65e-7 | [-2.93e-7, +1.02e-6] | 6/12 |

The earlier development block had slightly negative R3 point estimates. Fresh
confirmation changed the point-estimate sign but still did not exclude zero.
Therefore the correct conclusion is **metric-specific instability near the R3
ceiling**, not a confirmed `ret_thesis` benefit or harm. The primary
`ret_excel`, uncensored full-ledger, fill, delivered-rations and unresolved
panels remain positive.

## Scientific consequence

Buffers and productive capacity materially move resilience outcomes,
particularly under R2r. This establishes a resource-value surface worth
studying. The next scientific question is whether state-dependent control can
beat the strongest fixed and structured controllers under matched resources.
No learner should be trained until that observable residual is demonstrated.

This confirmation is distinct from Paper B v0 H2/H3:

- Paper B v0 H2 (successive-campaign adaptation) was contradicted by Q-R1.
- Paper B v0 H3 (volatility reduction) retained development-only support.

## Custody

- Contract:
  `contracts/garrido_h2_h3_confirmation_v1.json`
  (`sha256:1d3c80bd48feac4c71065ad3e432accfcc99cfe7c0cbf20a8166ea95103cfe98`)
- Freeze receipt:
  `contracts/garrido_h2_h3_confirmation_v1_freeze_receipt.json`
  (`sha256:352a4dcaa4635c4aeeafa582c783588af2e156a58ee73f3c45ec75f7ebae1a0f`)
- Result:
  `results/garrido_h2_h3_confirmation_v1/result.json`
  (`sha256:bc375d3021b64d1069f4111d5350294db77317c04e81ec222ac1ebdd17e8b195`)
- Tape-level deltas:
  `results/garrido_h2_h3_confirmation_v1/tape_level_deltas.json`
  (`sha256:e12f3cf944c7ac0f3ce66211ccd5c43fa5efb584aae3cdabe1a5946c28138f6f`)
- Aggregate receipt:
  `results/garrido_h2_h3_confirmation_v1/completion_receipt.json`
  (`sha256:d4305bcf6bf5209d52f0d44ff6238efb0b4a053bf1f3a9d5d3311e534faa61dc`)
