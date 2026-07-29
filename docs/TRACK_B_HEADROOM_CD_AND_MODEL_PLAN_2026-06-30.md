# Track B Headroom + Cobb-Douglas + Model Plan (2026-06-30)

## Status

Track B is now the main positive lane. The confirmed claim is not only against the old 9-policy static frontier: it now survives a matched dense CRN static frontier.

Artifacts:

- Dense CRN static frontier: `outputs/experiments/track_b_dense_crn_static_2026-06-30`
- Garrido-style audit workbook: `outputs/audits/track_b_top_tier_confirm_dense_crn_2026-06-30/track_b_top_tier_audit.xlsx`
- Headroom gate: `outputs/experiments/track_b_headroom_matrix_risklevel_2026-06-30`

## Dense CRN Verdict

Matched setup: frozen 5-seed PPO confirmatory run vs 147 Track B statics, using the same evaluation seeds and episodes.

| Metric | PPO | Best Dense Static | Delta | Verdict |
|---|---:|---:|---:|---|
| Order-level ReT | 0.005666 | 0.005251 | +0.000415 | raw ReT win |
| Tail ReT CVaR05 | 0.005428 | 0.004922 | +0.000506 | tail win |
| Flow fill | 0.961 | 0.669 | +0.293 | service win |
| CD sigmoid mean | 0.939 | 0.754 | +0.185 | CD eval win |
| Cost index | 0.682 | 0.667 | +0.015 | not resource-efficient |

Paired seed CI95:

- ReT delta: `[+0.000401, +0.000428]`
- Flow-fill delta: `[+0.288, +0.297]`
- CD sigmoid delta: `[+0.177, +0.193]`

Claim boundary:

- `raw_ret_win`: yes.
- `tail_win`: yes.
- `pareto_ret_cost`: yes, no dense static has >= ReT and <= cost.
- `resource_efficient_win`: no, PPO uses slightly more cost than the best dense static by ReT.

## Cobb-Douglas

Evaluation already supports Track B under the balanced/variance-log CD lens: PPO beats the best dense static by `+0.185` in CD sigmoid mean, with CI95 fully positive.

Do not train first with full-cost CD. Track A showed that full-cost CD can become perverse when the cost term dominates. For Track B:

1. Use CD as a secondary evaluation lens first.
2. If training with CD, prefer `ReT_cd_balanced` or `ReT_garrido2024_train`, not full-cost `ReT_garrido2024` as the only reward.
3. Promote only if the learned policy keeps ReT/flow/CVaR while improving CD; a cheap-but-low-service policy is not a win.

Note: `scripts/audit_track_b_all_rewards.py` now accepts `--observation-version`, but the older frozen Kaggle model has a 48-dimensional VecNormalize contract while the current local `v7` has 52 dimensions. Do fixed-policy reward-lens re-evaluation either from the saved confirmatory metrics or with a compatibility wrapper/payload, not by silently mixing observation contracts.

## Headroom Gate

Moving only `phi/psi` inside `adaptive_benchmark_v2` was not enough, because the adaptive benchmark has its own internal regime machinery. The useful headroom appears when we use thesis-like controlled families plus `risk_level` and demand pressure.

Corrected static-only gate results:

| Risk level | Family | Oracle - best constant | Distinct best actions | Verdict |
|---|---|---:|---:|---|
| current | R2 | +0.015256 | 6 | PROMOTE |
| current | R24 | +0.009379 | 7 | PROMOTE |
| current | all | +0.000149 | 5 | weak promote |
| increased | R2 | +0.015993 | 6 | PROMOTE |
| increased | R24 | +0.010254 | 7 | PROMOTE |
| increased | all | +0.000077 | 4 | null |
| severe | R2 | +0.008618 | 5 | PROMOTE |
| severe | R24 | +0.003569 | 6 | PROMOTE |
| severe | all | +0.000026 | 4 | null |

Interpretation:

- The best amplification route is not all-risks severe stress. That often collapses service and creates little usable learning signal.
- The strongest controllable headroom is in R2 and R24 controlled campaigns, especially with demand pressure.
- A campaign runner should mix promoted R2/R24 cells so the optimal dispatch action genuinely changes over time.

## Track A Lessons To Transfer

- Always compare against a dense CRN frontier in the same action space.
- Export rich metrics before declaring a win.
- Keep raw ReT, Pareto, and resource-efficient verdicts separate.
- Do not trust in-sample checkpoints.
- Use low learning rate, `n_envs=4-8`, VecNormalize, and checkpoint selection for any new Track B training.
- Static gates come before PPO. If the oracle does not beat a best single static, PPO cannot invent headroom.

## Models With History

PPO MLP remains the baseline for the primary Track B win because the current observation exposes the Markov regime and downstream pressure. History models should be positioned as ablations, not replacements for the frontier validation.

Recommended order:

1. PPO MLP: primary confirmed baseline.
2. SAC: natural comparator for continuous dispatch actions; useful if PPO plateaus or action variance is too constrained.
3. RecurrentPPO/LSTM: test under `obs_hidden`, where regime and forecast indicators are masked. If it improves only there, that supports a memory/partial-observability claim.
4. DMLPA/Transformer: long-context ablation after the recurrent smoke. Useful for H4 or hidden-regime retention, not needed to prove H1.

H4 retained-reset remains separate:

- Claim it only if `retained - reset > 0` under cold transfer with CI.
- If null, the main paper can still stand on Track B H1 plus Track A boundary characterization.

## Next Concrete Runs

1. Build a Track B campaign runner over the promoted R2/R24 cells from the headroom gate.
2. Run one cheap PPO smoke: 1-2 seeds, 20k-30k, h52, dense campaign static frontier.
3. Run reward mini-sweep only after the campaign smoke survives:
   - `control_v1`
   - `ReT_garrido2024_train`
   - `ReT_cd_balanced`
   - optional `ReT_tail`/CVaR-style reward if tail metrics become the bottleneck.
4. Confirm one frozen config: 5-10 seeds, 60k, CRN, rich workbook, mechanism audit.

## Campaign Runner Results

Implemented: `scripts/run_track_b_campaign.py`.

The runner samples training episodes from promoted R2/R24 cells and evaluates PPO against a dense static campaign frontier under the same selected cells and CRN seeds.

Selected cells for the first campaign:

- `current_R2_phi1_psi1_dm1`
- `current_R24_phi1_psi2_dm1`
- `increased_R2_phi1_psi1_dm1`
- `increased_R24_phi1_psi2_dm1`
- `severe_R2_phi1_psi1_dm1`
- `severe_R24_phi1_psi1_dm1`

### Control reward probe

Run: `outputs/experiments/track_b_campaign_r2_r24_2seed_30k_2026-06-30`

| Metric | PPO | Best campaign static | Delta |
|---|---:|---:|---:|
| Excel ReT | 0.362647 | 0.338564 | +0.024083 |
| Flow fill | 0.969198 | 0.916614 | +0.052584 |
| Service loss/order | 1.02e6 | 1.91e6 | -0.89e6 |
| Cost index | 0.667 | 1.000 | -0.333 |

Verdicts:

- `raw_ret_win`: true
- `tail_service_win`: true
- `pareto_ret_cost`: true
- `resource_efficient_win`: true

Mechanism:

- Best static uses S3 and max Op10/Op12: `S3_op9x1_op10x2_op12x2`.
- PPO uses S2 with moderate downstream multipliers: Op10 about `1.27`, Op12 about `1.25`.
- PPO wins the aggregate campaign by improving R24 cells strongly; it still loses some R2 cells. This suggests the next amplification should either weight R24 more or add a reward/architecture mechanism that improves R2 without sacrificing R24.

### Reward mini-sweep

| Reward | Seeds/steps | Excel ReT delta | Flow delta | Service-loss signed win | Cost delta | Verdict |
|---|---:|---:|---:|---:|---:|---|
| `control_v1` | 2 seeds / 30k | +0.024083 | +0.052584 | +0.894e6 | -0.333 | promote |
| `ReT_excel_delta` | 2 seeds / 30k | +0.021513 | +0.053159 | +0.749e6 | -0.333 | promote |
| `ReT_excel_plus_cvar`, alpha=0.1 | 2 seeds / 30k | +0.030014 | +0.057628 | +0.831e6 | -0.333 | best reward so far |
| `ReT_excel_plus_cvar`, alpha=0.2 | 2 seeds / 30k | +0.023267 | +0.055686 | +0.783e6 | -0.333 | promote |
| `ReT_excel_plus_cvar`, alpha=0.4 | 2 seeds / 30k | +0.026189 | +0.057077 | +0.832e6 | -0.333 | promote |
| `ReT_garrido2024_train` | 2 seeds / 30k | +0.013853 | +0.043723 | +0.641e6 | -0.333 | promote as CD same-bar candidate |
| `ReT_cd_balanced` | 1 seed / 20k | -0.048944 | -0.061728 | -0.417e6 | -0.333 | do not scale |

Interpretation:

- The best campaign reward so far is `ReT_excel_plus_cvar` with alpha=0.1. It is aligned with the Excel/ReT target and adds a light tail term.
- `ReT_excel_delta` works, but alpha=0.1 improves raw ReT and flow without losing the resource advantage.
- `ReT_garrido2024_train` is viable if we need a Cobb-Douglas same-bar training result, but it is weaker than the Excel-aligned rewards on raw ReT.
- `ReT_cd_balanced` should stay an evaluation lens for now; as a reward it undertrained or moved the policy away from service.

Same-reward notes:

- `ReT_excel_plus_cvar` alpha=0.1: PPO reward total `140.77` vs best same-reward static `128.54`, delta `+12.23`.
- `ReT_excel_delta`: PPO reward total `139.06` vs best same-reward static `131.03`, delta `+8.02`.
- `ReT_garrido2024_train`: PPO reward total `191.18` vs best same-reward static `179.57`, delta `+11.61`.
- `ReT_cd_balanced`: PPO reward total `28.28` vs best same-reward static `28.76`, delta `-0.49`.

Excel headroom in the selected campaign:

- Oracle static by cell: `0.353468`.
- Best single static across cells: `0.338927`.
- Excel-ReT headroom: `+0.014540`.
- The PPO with `ReT_excel_plus_cvar` alpha=0.1 reaches `0.368578`, exceeding both the best single static and the static oracle estimate in this cheap 2-seed probe. This needs a denser confirmatory run before a final claim.

Next promotion:

- Scale `ReT_excel_plus_cvar` alpha=0.1 campaign to 5 seeds, 60k, h52/h104.
- Keep `control_v1` as the operational baseline and `ReT_garrido2024_train` as the CD same-bar secondary.
- Add per-cell weighting or curriculum only if R2 losses persist in the 5-seed confirm.

### Kaggle optimized campaign confirmatory

Run: `outputs/experiments/track_b_campaign_optimized_kaggle_v3/manual_output/track_b_campaign_optimized`

Profile:

- Reward: `ReT_excel_plus_cvar`, alpha `0.1`
- Seeds: `1,2,3,4,5`
- Timesteps: `60k`
- Horizon: `h104`
- Cells: the six promoted R2/R24 cells above
- Static frontier: `3 shifts x 1 op9 x 4 op10 x 4 op12`
- Training: BC warm-start, reward normalization, cosine LR, checkpoint selection

Verdict:

| Metric | PPO | Best campaign static | Delta | Winner |
|---|---:|---:|---:|---|
| Excel ReT | 0.443966 | 0.474556 | -0.030590 | static |
| Flow fill | 0.960550 | 0.955968 | +0.004582 | PPO |
| Service loss/order | 1.388e6 | 1.958e6 | -0.569e6 | PPO |
| CTj p99 | 4913h | 8340h | -3427h | PPO |
| RPj p99 | 629h | 978h | -349h | PPO |
| DPj p99 | 5934h | 10344h | -4411h | PPO |
| Cost index | 0.8375 | 1.0000 | -0.1625 | PPO |
| Lost rate | 0.0179 | 0.0098 | +0.0081 | static |

Best campaign static: `S3_op9x1_op10x2_op12x2`.

Per-cell Excel ReT deltas for PPO vs that best static:

| Cell | Delta |
|---|---:|
| current/R24 | -0.082875 |
| current/R2 | -0.014482 |
| increased/R24 | +0.018630 |
| increased/R2 | -0.152898 |
| severe/R24 | -0.016958 |
| severe/R2 | +0.065042 |

Interpretation:

- The optimized campaign does **not** confirm a raw Excel ReT win. The static max-dispatch policy becomes very strong at `h104`.
- PPO still learns a useful resource/service tradeoff: lower cost, much better tail queues, lower service-loss AUC, and lower cross-regime volatility (`CV 0.273` vs static `0.369`).
- The next campaign amplification should not repeat this exact mixed six-cell setup as a raw-ReT claim. It should either:
  1. Split R2 and R24 campaigns instead of mixing them, because PPO wins severe/R2 and increased/R24 but loses badly on increased/R2/current/R24 against max-dispatch static.
  2. Use the existing top-tier `adaptive_benchmark_v2` Track B confirm as the primary raw-ReT claim, where dense-CRN ReT already survives.
  3. Treat this controlled campaign as evidence for service/tail/cost robustness, not as the main ReT win.

Runner issue to fix before the next confirmatory run: `ret_ci95_low` in `scripts/run_track_b_kaggle.py` is currently computed as `ppo_ci_low - static_ci_high`, which can print a positive lower bound even when the mean delta is negative. Use paired/bootstrap deltas by seed/cell instead.
