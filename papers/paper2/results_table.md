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
| Q replicated H_OL | rho75_share90 | est +0.07952 | LCB95 = +0.06608 | 84.8% of 256 tapes favorable; 10/10 seeds positive | `results/program_q/confirmation_v1_20260718/artifacts/confirmation/evaluation/result.json@62f6fd390471` |
| Q replicated H_OL | rho90_share75 | est +0.07255 | LCB95 = +0.06233 | 89.8% of 256 tapes favorable; 10/10 seeds positive | `results/program_q/confirmation_v1_20260718/artifacts/confirmation/evaluation/result.json@62f6fd390471` |
| Q replicated H_OL | rho90_share90 | est +0.11724 | LCB95 = +0.10614 | 95.7% of 256 tapes favorable; 10/10 seeds positive | `results/program_q/confirmation_v1_20260718/artifacts/confirmation/evaluation/result.json@62f6fd390471` |
| Q neural relation Δ_N | rho75_share90 | est -0.00159 | CI95 [-0.00627, +0.00310] | TOST equivalence bar: CI ⊂ [−0.01, +0.01] | `results/program_q/confirmation_v1_20260718/artifacts/confirmation/evaluation/result.json@62f6fd390471` |
| Q neural relation Δ_N | rho90_share75 | est -0.00072 | CI95 [-0.00552, +0.00408] | TOST equivalence bar: CI ⊂ [−0.01, +0.01] | `results/program_q/confirmation_v1_20260718/artifacts/confirmation/evaluation/result.json@62f6fd390471` |
| Q neural relation Δ_N | rho90_share90 | est -0.00041 | CI95 [-0.00268, +0.00186] | TOST equivalence bar: CI ⊂ [−0.01, +0.01] | `results/program_q/confirmation_v1_20260718/artifacts/confirmation/evaluation/result.json@62f6fd390471` |
| Q guardrail worst-product fill (vs classical) | rho75_share90 | est -0.01036 | LCB95 = -0.02266 | frozen Class-B margin −0.02 (binding) | `results/program_q/confirmation_v1_20260718/artifacts/confirmation/evaluation/result.json@62f6fd390471` |
| Q guardrail worst-product fill (vs classical) | rho90_share75 | est -0.01573 | LCB95 = -0.02566 | frozen Class-B margin −0.02 (binding) | `results/program_q/confirmation_v1_20260718/artifacts/confirmation/evaluation/result.json@62f6fd390471` |
| Q guardrail worst-product fill (vs classical) | rho90_share90 | est -0.00451 | LCB95 = -0.02632 | frozen Class-B margin −0.02 (binding) | `results/program_q/confirmation_v1_20260718/artifacts/confirmation/evaluation/result.json@62f6fd390471` |
| Q terminal adjudication | all | STOP_Q_NO_REPLICATED_LEARNED_ADAPTATION | N=256/cell, seeds 7490001–7490256 | compound label; components above reported separately as preregistered | `results/program_q/confirmation_v1_20260718/artifacts/confirmation/adjudication.json@e13e17f001a1` |
