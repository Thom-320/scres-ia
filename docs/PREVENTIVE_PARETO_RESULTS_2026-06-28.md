# Preventive Pareto Results (2026-06-28)

## Dense-CRN Update: Previous Pareto Claim Falsified

The later dense-frontier CRN Kaggle audit supersedes the earlier coarse-frontier result below.
It tested the same dynamic lane against a 21-fraction static grid (`f0.00` to `f1.00` by 0.05,
all S1/S2/S3) with common eval seeds.

- Kernel: `thomaschisica/scresia-preventive-pareto-dense-crn`
- Output: `outputs/kaggle/scresia-preventive-pareto-dense-crn_auto/scresia_preventive_pareto_dense_crn_outputs/decision.json`
- Config: `ReT_excel_delta`, `phi=4`, `psi=1.5`, 10 learner seeds, `60000` timesteps, `n_envs=4`,
  `eval_episodes=8`, `max_steps=104`, `n_fracs=21`, `crn_eval=true`, `eval_seed0=9000`.

Kaggle dense-CRN decision:

| metric | dynamic | best dense static at <= dynamic resource |
|---|---:|---:|
| Excel ReT | 0.0021291884 | 0.0022791952 (`f0.10_S1`) |
| CVaR95 service loss | 5.30992345e9 | 4.87786161e9 (`f0.10_S1`) |
| Resource composite | 0.1743905642 | 0.0500000007 (`f0.10_S1`) |
| Buffer action std | 0.1982366640 | 0.0 |

Verdict:

| gate | result |
|---|---|
| Excel Pareto win | false |
| CVaR Pareto win | false |
| Dominated by static on Excel/resource | true |
| Dominated by static on CVaR/resource | true |
| Primary win | false |

Interpretation: the learned policy is still adaptive (`frac_std≈0.20`) and uses realized-risk /
state signals, but the earlier Pareto win was an artifact of an insufficiently dense static
frontier. The cheap low-buffer static `f0.10_S1` dominates it on both resilience and resource.
This lane should now be treated as a useful mechanism/near-miss, not as a paper win.

## Claim Boundary

**Superseded by the dense-CRN update above.** The original boundary below was valid only relative
to the coarse charged static frontier used in the first final Kaggle run.

This is **not** a raw-resilience dominance claim against a free-spending static policy. The supported
claim is narrower and stronger:

> In the war-stress continuous `I_{t,S}` lane (`phi=4`, `psi=1.5`, h104), a dynamic preventive policy
> trained with `ReT_excel_delta` Pareto-dominates the charged static frontier on Excel ReT and CVaR,
> using lower average resource than the static policies required to match its resilience.

Static and dynamic policies are both charged with the same normalized resource measure:

`resource_composite = 0.5 * buffer_frac + 0.5 * ((shift - 1) / 2)`

## Final Kaggle Run

- Kernel: `thomaschisica/scresia-preventive-pareto-final`
- Dataset: `thomaschisica/scres-ia-payload`
- Output: `outputs/kaggle/scresia-preventive-pareto-final/scresia_preventive_pareto_final_outputs/decision.json`
- Seeds: `1,2,3,4,5,8501,8502,8503,8504,8505`
- Timesteps: `60000`
- Eval episodes: `8`
- Horizon: `104` weekly decisions
- Reward: `ReT_excel_delta`
- Environment: `continuous_its`, `risk_obs+hazard`, `risk_frequency_multiplier=4.0`, `risk_impact_multiplier=1.5`

Kaggle decision:

| metric | dynamic |
|---|---:|
| Excel ReT | 0.0021424619 |
| CVaR95 service loss | 5.217598302e9 |
| Resource composite | 0.2411771896 |
| Buffer action std | 0.2570250314 |
| Excel Pareto win | true |
| CVaR Pareto win | true |
| Primary win | true |

## Local Confirmation Ladder

| run | seeds | timesteps | Excel Pareto | CVaR Pareto | Excel | CVaR | resource |
|---|---:|---:|---|---|---:|---:|---:|
| `preventive_pareto_excel_delta_5seed_40k_2026-06-27` | 5 | 40000 | true | true | 0.0021157 | 5.4757e9 | 0.1479 |
| `preventive_pareto_excel_delta_5seed_60k_2026-06-28` | 5 | 60000 | true | true | 0.0021068 | 5.5384e9 | 0.0955 |
| `preventive_pareto_excel_delta_mixed10seed_40k_2026-06-28` | 10 | 40000 | true | true | 0.0021591 | 5.3433e9 | 0.2118 |
| `preventive_pareto_excel_delta_mixed10seed_60k_2026-06-28` | 10 | 60000 | true | true | 0.0021793 | 5.3571e9 | 0.2048 |
| Kaggle final | 10 | 60000 | true | true | 0.0021425 | 5.2176e9 | 0.2412 |

## Interpretation

The result survived:

- increasing training from `40k` to `60k`,
- moving from 5 seeds to a mixed 10-seed set,
- rerunning the final mixed 10-seed setup on Kaggle.

The policy is not static: `frac_std` remains nonzero across confirmations. Action traces show a mixed
mechanism: risk/hazard features (`active/recent_R12/R22/R23/R24`, `weeks_since_last_R1/R3`) matter, but
inventory and phase/time terms also contribute. The paper language should therefore say **preventive
resource-aware dynamic allocation**, not pure risk-forecast anticipation.

## Mechanism Audit

Follow-up audit:

- Script: `scripts/audit_prevention_anticipation.py`
- Output: `outputs/audits/prevention_anticipation_excel_delta_5seed_60k_2026-06-28/summary.json`
- Config: 5 seeds, 60k timesteps, h104, `risk_obs+hazard`, `ReT_excel_delta`,
  `phi=4.0`, `psi=1.5`

Summary:

| diagnostic | value | interpretation |
|---|---:|---|
| mean buffer std | 0.192 | adaptive, not constant |
| hazard strength | 0.074 | weak hazard conditioning |
| reactive strength | 0.130 | backlog/fill consequences dominate |
| lead sum (`k>0`) | -0.023 | not strong pre-shock build-up |
| lag sum (`k<0`) | -0.110 | weak relative lead signal, but both signs are negative |

The audit supports the Pareto claim but weakens any pure "anticipates the next disruption" claim.
The best current wording is:

> The learned controller uses dynamic buffer/shift allocation to preserve Excel/CVaR resilience at
> lower charged resource. It is adaptive and resource-aware, but its measured mechanism is more
> efficiency/reactive than clean pre-shock anticipation.

The next strongest untested mechanism is Δmemory (`retained-reset`) on the same continuous lane.

## Related Nulls / Non-Claims

- Old free-static `ReT_excel_delta` Kaggle lane is null: `primary_win=false`.
- `ReT_excel_plus_cvar alpha=0.2` won Excel Pareto locally but did not confirm CVaR Pareto.
- `h260` did not confirm the h104 signal.
- Raw fill remains downstream-bottlenecked; this is an efficiency/Pareto result, not a fill-rate breakthrough.
