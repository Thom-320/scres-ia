# Paper 2 — master results table (machine-generated; do not edit)

| Level | Cell | Point | Bound | Integrity / notes | Source |
|---|---|---|---|---|---|
| L1 physical opportunity | all (safe oracle) | H_PI = 0.15151 | LCB95 = 0.11562 | fungible null = 0 (exact) | `results/program_o/full_des_hpi_translation_v1/validation_custody_verdict_v1.json@9dc23deda568` |
| L2 classical H_obs | rho75_share90 | LCB95 = +0.06595 | 44/48 favorable | placebos 27/27=True; CVaR10 LCB -0.0086 (pt +0.0350) | `results/program_o/fixed_clock_hobs_corrective_validation_v1/independent_audit_v1.json@7fffbc30aef4` |
| L2 classical H_obs | rho90_share75 | LCB95 = +0.04303 | 42/48 favorable | placebos 27/27=True; CVaR10 LCB -0.0155 (pt +0.0195) | `results/program_o/fixed_clock_hobs_corrective_validation_v1/independent_audit_v1.json@7fffbc30aef4` |
| L2 classical H_obs | rho90_share90 | LCB95 = +0.05860 | 46/48 favorable | placebos 27/27=True; CVaR10 gate met | `results/program_o/fixed_clock_hobs_corrective_validation_v1/independent_audit_v1.json@7fffbc30aef4` |
| L3 learned H_OL | rho75_share90 | est +0.07576 | LCB95 = +0.04323 | 41/48 vs 65,536 frontier | `result.json@dc55880f0dca` |
| L3 learned H_OL | rho90_share75 | est +0.06261 | LCB95 = +0.03659 | 42/48 vs 65,536 frontier | `result.json@dc55880f0dca` |
| L3 learned H_OL | rho90_share90 | est +0.10455 | LCB95 = +0.06630 | 44/48 vs 65,536 frontier | `result.json@dc55880f0dca` |
| L4 neural premium Δ_N | rho75_share90 | est -0.00165 | LCB95 = -0.00879 | 1/10 seeds beat both comparators | `result.json@dc55880f0dca` |
| L4 neural premium Δ_N | rho90_share75 | est -0.00273 | LCB95 = -0.01400 | 0/10 seeds beat both comparators | `result.json@dc55880f0dca` |
| L4 neural premium Δ_N | rho90_share90 | est -0.00150 | LCB95 = -0.00828 | 2/10 seeds beat both comparators | `result.json@dc55880f0dca` |
| Q prospective replication | all | PENDING | contract frozen: N=128/cell, block 7490001+ | outcomes: PASS_PREMIUM / PASS_EQUIVALENT / BOUND / STOP | `contract@frozen` |
