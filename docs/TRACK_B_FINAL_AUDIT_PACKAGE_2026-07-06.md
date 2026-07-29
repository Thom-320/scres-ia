# Track B final audit package - 2026-07-06

## Purpose

This document consolidates the final July 6 Track B tuning and scenario
confirmation work into one paper-facing source of truth.

It does not replace the older 10-seed dense-CRN headline evidence. It answers a
different reviewer-facing question:

> If we remove explicit forecast channels and use a defensible two-year horizon,
> which Track B scenario and architecture package should be carried into the
> manuscript?

Primary metric throughout: Garrido Excel ReT, `order_ret_excel_mean`.

## Final Defaults

| Design choice | Final choice | Evidence |
|---|---|---|
| Observation | `v7_no_forecast` | Forecast channels are not needed for the Track B ReT result; see C16 and no-forecast fixed-RNG final. |
| Horizon | `104` weeks | Best compromise between realistic resilience horizon and visible adaptive headroom. |
| PPO batch size | `64` | Best/near-best in the PPO sensitivity screens; used in final A/B/C. |
| Real-KAN A/B batch size | `256` | `512` won a v7-full/adaptive-v2 sweep but failed to transfer to no-forecast A/B. |
| Real-KAN selected Case C batch size | `512` | Used in the final selected Case C confirmation after the Real-KAN sweep; improves cost-side comparison but remains below PPO on ReT. |
| Reward | `control_v1` | Gamma/GAE/reward-normalization screens did not justify changing the reward setup. |
| Main horizon claim | h104 / two years | h52 has more headroom but is too opportunistic; h260 dilutes PPO-vs-static margin. |

## Scenario Definitions

| Scenario | Risk setup | Intended role |
|---|---|---|
| Case A: all Garrido risks | Garrido `current`, all risks active | Main reviewer-safe scenario. |
| Case B: downstream risks | Garrido `current`, only `R22,R23,R24` active | Mechanism panel for branch shift and exposure reduction. |
| Case C: selected stress/adaptation | `R22,R23,R24` active; `R24` frequency x3; `R22/R23` impact x1.5 | Controlled headroom/stress panel, not the main headline. |

Case B and Case C absolute ReT values are not directly comparable to Case A
because the Garrido Excel branch mix changes. Compare them within scenario
against the best static baseline.

## Final Confirmatory Results

All final rows below use `v7_no_forecast`, h104, seeds `1..5`, 60k training
timesteps, and 12 evaluation episodes.

| Scenario | Architecture | ReT Excel | CVaR05 | Best static | Delta vs static | Relative delta | Cost |
|---|---|---:|---:|---|---:|---:|---:|
| Case A: all Garrido risks | PPO+MLP | 0.005900714 | 0.001911092 | `s2_d2.00` | +0.000256755 | +4.55% | 0.488 |
| Case A: all Garrido risks | Real-KAN bs256 | 0.005854707 | 0.001787318 | `s2_d2.00` | +0.000210748 | +3.73% | 0.378 |
| Case B: R22/R23/R24 only | PPO+MLP | 0.604785754 | 0.000000000 | `s2_d2.00` | +0.013894863 | +2.35% | 0.431 |
| Case B: R22/R23/R24 only | Real-KAN bs256 | 0.604782218 | 0.000000000 | `s2_d2.00` | +0.013891327 | +2.35% | 0.359 |
| Case C: selected stress | PPO+MLP | 0.481160397 | 0.000000000 | `s2_d2.00` | +0.046323705 | +10.65% | 0.719 |
| Case C: selected stress | Real-KAN bs512 | 0.470095597 | 0.000000000 | `s2_d2.00` | +0.035258905 | +8.11% | 0.387 |

Independent VPS PPO A/B corroboration:

| Scenario | Local PPO ReT | VPS PPO ReT | Difference |
|---|---:|---:|---:|
| Case A: all Garrido risks | 0.005900714 | 0.005903104 | +0.000002390 |
| Case B: R22/R23/R24 only | 0.604785754 | 0.604846242 | +0.000060489 |

## What This Supports

### Main Paper Spine

Use Case A PPO+MLP no-forecast h104 as the clean reviewer-safe spine.

Safe wording:

> Under the all-risk Garrido current scenario, with explicit forecast channels
> removed, PPO improves Garrido Excel ReT by +4.55% over the best static
> baseline at a two-year horizon.

This is the strongest single result for the manuscript because it uses all
risks, the conservative no-forecast observation, and a defensible horizon.

### Real-KAN Role

Real-KAN should be carried as an interpretability / efficiency sidecar, not as
the main spine.

Evidence:

- Case A: Real-KAN beats static (+3.73%) but is below PPO on ReT.
- Case B: Real-KAN is effectively tied with PPO on ReT and cheaper.
- Case C: Real-KAN captures 76.1% of PPO's static margin with far lower cost.
- Batch-size 512 does not improve Real-KAN under no-forecast A/B, so the earlier
  v7-full/adaptive-v2 sweep should not be overgeneralized.

Safe wording:

> Real-KAN provides a lower-cost, interpretable sidecar that recovers much of
> the adaptive gain, but PPO+MLP remains the stronger ReT optimizer under the
> all-risk no-forecast spine.

### Case B / Case C Mechanism

Case B and Case C show a real branch/exposure mechanism:

- Best static in Case B: 77.06% fill-rate branch.
- PPO in Case B: 80.32% fill-rate branch.
- Real-KAN in Case B: 80.34% fill-rate branch.
- Best static in Case C: 49.48% fill-rate branch.
- PPO in Case C: 55.88% fill-rate branch.
- Real-KAN in Case C: 54.66% fill-rate branch.

This is adaptive exposure reduction: learned policies dispatch quickly enough to
shorten order exposure windows, so fewer orders overlap active downstream risk
events.

Safe wording:

> The learned policies improve resilience partly by reducing exposure to
> disruption windows, moving more orders into the fill-rate branch and improving
> risk-conditional ReT under downstream stress.

## What This Does Not Support

Do not claim:

- Causal prevention or anticipation.
- That Case B / Case C absolute ReT values are comparable to all-risk Case A.
- That Real-KAN replaces PPO+MLP as the main optimizer.
- That batch size 512 is the general Real-KAN no-forecast setting.
- That CVaR chooses the downstream-only Case B/C winner; CVaR is zero in those
  branch mixes.

The prevention-audit evidence from the July 4 sequence remains negative:
counterfactual `R_full - R_reset(pre-risk)` tests did not show a robust
positive-pair rate. The current supported mechanism is adaptive exposure
reduction and recovery, not anticipatory preparation.

## Paper Figure / Table Plan

Recommended paper package:

1. Main result table: Case A, PPO vs Real-KAN vs best static.
2. Stress/adaptation table: Case C, PPO vs Real-KAN vs best static, with
   relative delta and cost.
3. Branch-shift mechanism panel: fill-rate/recovery/risk-no-recovery branch
   percentages for best static, PPO, and Real-KAN in Case B/C.
4. Architecture diagram: PPO+MLP spine and Real-KAN sidecar.
5. Claim-boundary paragraph: no-forecast spine, adaptive exposure reduction,
   no prevention claim.

## Source Artifacts

- `docs/TRACK_B_HORIZON_SCREEN_VERDICT_2026-07-06.md`
- `docs/TRACK_B_FINAL_AB_H104_CONFIRM_VERDICT_2026-07-06.md`
- `docs/TRACK_B_REALKAN_BATCH_SIZE_SWEEP_VERDICT_2026-07-06.md`
- `docs/TRACK_B_REALKAN_NO_FORECAST_BS512_AB_VERDICT_2026-07-06.md`
- `docs/TRACK_B_CASE_C_PER_RISK_HEADROOM_VERDICT_2026-07-06.md`
- `docs/TRACK_B_CASE_C_SELECTED_H104_CONFIRM_VERDICT_2026-07-06.md`

