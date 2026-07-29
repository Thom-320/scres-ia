# Program O corrective classical-H_obs validation verdict

## Terminal verdict

`STOP_PROGRAM_O_AFTER_CORRECTIVE_VALIDATION`

The corrective run repaired the two adjudication defects without changing the policy, cells, physics, metric, placebos, thresholds, or guardrails. It used the development-frozen full-frontier comparators and studentized one-sided max-t inference on a fresh 48-tape block.

The primary canonical-ReT result passed decisively:

| Cell | Mean ΔReT | Simultaneous LCB95 | Favorable tapes |
|---|---:|---:|---:|
| rho75/share90 | 0.09852 | 0.06595 | 44/48 |
| rho90/share75 | 0.07347 | 0.04303 | 42/48 |
| rho90/share90 | 0.09974 | 0.05860 | 46/48 |

All 27 information-placebo contrasts passed; the smallest simultaneous LCB was `0.00716`. Physical resource equality passed with one scheduled-resource vector across 1,451 replays and zero failures. Action trajectories and state counterfactuals passed in all cells.

The sole failing gate was tail-risk non-inferiority:

| Cell | Mean ΔCVaR10 | Simultaneous LCB95 |
|---|---:|---:|
| rho75/share90 | 0.03502 | **−0.00858** |
| rho90/share75 | 0.01954 | **−0.01551** |
| rho90/share90 | positive and non-inferior | pass |

The point estimates favor the MPC, but simultaneous uncertainty still permits tail deterioration in two cells. The preregistration required every guardrail LCB to be non-negative. Therefore the joint safe-H_obs contract fails even though mean canonical ReT, observability, state dependence, resources, and all placebos pass.

## Scientific interpretation

Program O establishes a strong boundary:

- observable state feedback has a reproducible mean canonical-ReT advantage over the strongest development-selected full-horizon open-loop schedule;
- the advantage is not a fixed calendar, information placebo, resource purchase, mass error, or comparator omission;
- the current frozen belief-MPC does not prove tail-risk non-inferiority with familywise 95% coverage.

It is therefore accurate to report **confirmed mean adaptive value with unresolved simultaneous tail safety**, but not the project's required safe classical H_obs positive and not Paper 2 learned value.

## No second rescue

The corrective contract explicitly forbids another controller, hyperparameter change, cell removal, comparator change, physics change, metric change, threshold change, or guardrail relaxation. Program O is closed. The CVaR condition cannot now be demoted or moved to sensitivity.

Consequently:

- learner authorized: no;
- Paper 2 confirmed: no;
- Paper 3 authorized: no.

A learner may be studied only under a genuinely new preregistered mechanism/contract, not as a rescue of Program O.

## Custody

- Executed commit: `7a05d448f4d788a19385a0c65c842b8663ed8391`
- Scientific source commit: `14b559cc0c88b8c186673077403d7c4253337cae`
- Contract SHA-256: `8cb665df8fad6bdfe9172d6c224c4b50da4ff14e77f4fbe3c04ff58616b4e278`
- Result SHA-256: `3d3ff5b37510a993582cfc82b5414868da5cec2f99eda5da1df58013af389877`
- Retrieved manifest SHA-256: `635f855228af404165484f8ca1732cd13114a01c16e8ce5e996efdebbe8b938e`
- Seed block: `7430001–7430048`, opened once and permanently burned
- Remote and retrieved checksum verification: pass
