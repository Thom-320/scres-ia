# Claims Registry and Q1 Reviewer Defense (2026-07-01)

This is the current source of truth for paper-facing claims. Older documents
may still mention retired dimension counts, perfect-fill headlines, old
training-budget claims, or coarse-frontier
wins; treat those as historical unless they are reconciled here.

Primary evaluation bar: Garrido/Excel ReT. CVaR/tail, Cobb-Douglas, service,
backlog/recovery, and resource are separate evidence columns, not substitutes.

## Claim Registry

| ID | Claim | Status | Current evidence | Before submission |
|---|---|---|---|---|
| C1 | Track B improves Garrido/Excel ReT when downstream dispatch is controllable. | **Supported** | 10-seed Track B confirm, dense CRN audit: Excel ReT PPO `0.005898` vs dense static `0.005460`, delta `+0.000438`, seed-clustered CI95 `[+0.000421,+0.000458]`; order-level mean ReT PPO `0.005673` vs static `0.005247`, delta `+0.000426`. See `docs/track_b_q1_stats_2026-07-02_final_10seed/`. | Keep paired dense-CRN table in final workbook; do not cite coarse frontier. |
| C2 | Track B improves lower-tail resilience / CVaR. | **Supported** | Tail ReT CVaR05 delta about `+0.000506`; recovery tail metrics improve strongly. | Report CVaR05, ReT p05/p10, CTj/RPj/DPj p95/p99 with CI/effect size. |
| C3 | Track B improves service and recovery, not just mean ReT. | **Supported** | Flow fill, service-loss AUC, backlog, CTj/RPj/DPj tails favor PPO in the rich audit. | Use Garrido-style workbook with PPO, best dense static, top statics, and order ledger. |
| C4 | Track B also wins under the balanced/variance-log Cobb-Douglas evaluation lens. | **Supported as evaluation; not as a training-reward claim** | Balanced CD sigmoid mean PPO about `0.939` vs dense static about `0.754`; CI positive. `variance_log` is already ported in `ret_garrido2024_calibration.json`. | Name the CD variant exactly; do not use full-cost CD as headline unless separately confirmed. |
| C5 | PPO is Pareto non-dominated in ReT/cost, but not strictly resource-efficient. | **Supported with wording constraint** | No dense static dominates PPO on ReT and cost; PPO cost is slightly higher than the best dense static by ReT. | Report `raw_ret_win`, `tail_win`, `pareto_ret_cost`, and `resource_efficient_win` separately. |
| C6 | The Track B mechanism is adaptive recovery / backlog control. | **Supported, mechanism audit still needed** | PPO uses balanced shifts and lower average Op10/Op12 than aggressive static, with high p95 dispatch when pressure rises; queues and service loss improve. | Run lead/lag and action-trace correlations; avoid "anticipatory" unless pre-event action rises. |
| C7 | RL value is frontier-dependent: action-space alignment with the downstream bottleneck explains Track A vs Track B. | **Supported** | 5-seed/60k final ablation (2026-07-02, `docs/E4_ABLATION_VERDICT_2026-07-02.md`). Two comparator definitions, same ordering: vs best STATIC in-arm, downstream_only +0.000566 > joint +0.000415 ≈ shift_only +0.000416; vs best EVALUATED comparator incl. heuristics (the stronger bar, used in the manuscript), downstream_only +0.000429 > joint +0.000367 ≈ shift_only +0.000377. Win is not explained by action-space size; concentrates in downstream dispatch access. | Manuscript uses the heuristic-inclusive definition (stronger). Keep joint 8D as the canonical result; ablation is mechanism evidence only. Ablation LR (1e-4) differs from canonical (3e-4) — within-ablation comparison valid, absolute numbers not comparable to canonical headline. |
| C8 | Track A is a boundary characterization: Garrido's buffer/shift family does not expose a publishable dynamic frontier. | **Supported with caveat** | Dense static frontiers, CRN, rich metrics, and repair attempts close current Track A claims. | Say "no publishable signal after dense frontier/CRN/rich metrics", not "mathematically impossible". |
| C9 | Reward choice is secondary once Track B controls the right bottleneck. | **Needs verification** | Older reward-sweep docs are stale; current v7/v8/v9 sweep is running. | Use only the current adaptive sweep results; retire old reward-insensitivity numbers if inconsistent. |
| C10 | v9 observation improves Track B headroom. | **Candidate only** | v9 implemented and smoke-checked: v7=`52`, v8=`79`, v9=`89` dimensions. No confirmatory win yet. | Promote only if v9 beats canonical v7 on Excel ReT delta or CVaR05 under dense CRN. |
| C11 | Track B transfers beyond one adaptive benchmark cell (cross-regime stress evaluation). | **Supported for current/increased; severe is the boundary regime** | CORRECTED 2026-07-02 (manuscript rev): an earlier count used a per-cell comparator selected by a secondary criterion, which flipped severe/h104 marginally positive and inflated h52 deltas. Against the conservative convention (best in-cell static by the primary metric, order-level ReT; `docs/track_b_q1_stats_2026-07-02_final/e3_per_cell_seed_ci.csv`), PPO wins 4/6 cells with fully positive seed-clustered CI95 (current h52 +0.000359, increased h52 +0.000537, current h104 +0.000209, increased h104 +0.000552; 5/5 seeds positive in each) and loses narrowly in BOTH severe cells (h52 −0.000060, h104 −0.000075; service-floor regime, fill ~0.1% for all policies). DENSE-FRONTIER UPGRADE 2026-07-02: current/h104 and increased/h104 re-validated against fully re-optimized per-cell 147-cell dense frontiers (+0.000209 CI[+0.000171,+0.000247]; +0.000542 CI[+0.000500,+0.000584]; 5/5 seeds; replicated by an independent duplicate run) — `docs/E3_H104_DENSE_FRONTIER_VERDICT_2026-07-02.md`. Fixed-9-cell caveat now applies only to h52 and severe cells. R2/R24/mixed families and h260 still not covered. | Report as "cross-regime stress evaluation," 4/6 with severe as the boundary regime; never cite superseded positive severe/h104 or inflated h52 numbers. |
| C15 | The `adaptive_benchmark_v2` regime alone (not the downstream-dispatch action space) explains the Track B win. | **Refuted** | Track A's original buffer/shift family, retrained fresh on `adaptive_benchmark_v2` itself (2026-07-02, `docs/E3_GENERALIZATION_VERDICT_2026-07-02.md`), loses to the best static comparator (Excel ReT 0.00530 vs 0.00539; CVaR also loses). The regime is not sufficient by itself; the win requires downstream-dispatch action-space access. | Cite as the direct rebuttal to the "regime was engineered to win" reviewer attack. |
| C12 | Retained learning / `L_{t-1}` improves future campaigns. | **Supported at small effect size, both observation arms** | RESOLVED 2026-07-02: `docs/H4_RETAINED_VS_RESET_VERDICT_2026-07-02.md`. 10 seeds, 8 online-adaptation cycles, `obs_full` retained-minus-reset CI95 [+0.0000095,+0.0000525]; `obs_hidden` (preferred, privileged fields masked) CI95 [+0.0000167,+0.0000819], 9/10 seeds positive. Effect is real but ~10x smaller than the Track B headline gain and ~100x smaller than the Track A oracle headroom. | State as a small, disclosed future-work extension in one sentence (see verdict doc for exact safe wording); do NOT reframe the paper's central claim around it or call it cross-campaign organizational learning. |
| C13 | PPO is enough; SAC/TD3 are unnecessary. | **Reviewer defense only** | RecurrentPPO was negative in Track A; PPO wins Track B. | If time allows, run SAC as robustness; otherwise state algorithm comparison is secondary to action-space mechanism. |
| C14 | A formal MDP/POMDP analysis supports the empirical result. | **Deferred appendix** | Not done. | Add a short appendix defining state/action/reward/partial observability and why empirical dense-frontier tests are primary. |
| C16 | PPO's Track B advantage does not depend on privileged simulator observation fields (true regime one-hot + true-transition-matrix forecasts), and the paper can use a no-forecast spine if needed. | **Supported** | Two complementary audits now exist. (1) Full privileged mask (`v7_no_regime_forecast`, 2026-07-02, `docs/E2_PRIVILEGED_OBSERVATION_VERDICT_2026-07-02.md`): with regime+forecast fields zeroed, PPO retains ~95% of the canonical delta (+0.000395 vs +0.000415) and still beats every static/heuristic comparator. (2) Final fixed-RNG no-forecast retrain, 15 seeds/60k (`docs/TRACK_B_NO_FORECAST_FIXED_RNG_FINAL_VERDICT_2026-07-05.md`): removing only `risk_forecast_48h_norm` and `risk_forecast_168h_norm` gives ReT Excel `0.00590391` vs full-v7 `0.00590177`, paired delta `+0.00000214` CI95 `[-0.00001255,+0.00001683]`, 9/15 favorable. This does not prove no-forecast is better; it proves forecast is not needed for the primary ReT result. | For reviewer-safe framing, use no-forecast as the conservative spine if desired. State clearly that no-forecast removes only forecast channels; the stronger no-regime+forecast mask is an ablation/defense, not the main 15-seed spine. |
| C17 | A non-learning policy given the same privileged regime signal PPO could exploit still cannot match PPO. | **Supported — GO** | E1 confirmatory go/no-go (2026-07-02, `docs/E1_GO_NO_GO_VERDICT_2026-07-02.md`): a regime-conditioned static lookup table, fitted directly on the true adaptive regime, gains only +0.0000277 over the best single static; PPO gains +0.000427 over the same comparator (15× larger). PPO beats all 75 static candidates, all 6 heuristics, and the fitted regime table individually, all CI95 > 0 (60/60 comparisons). Closes T3 from the other direction — together with C16, the privileged-observation attack is answered both ways. | Cite the gap-decomposition table (common static → regime table → heuristic → PPO) as the paper's central mechanism figure. |

## Retired Or Unsafe Claims

| Retired claim | Reason |
|---|---|
| "Track B is thesis-faithful." | Track B is an operational extension; Garrido fixed downstream dispatch. |
| "Track B uses the retired seven-dimensional label." | Current `track_b_v1` contract is 8D; old dimension-count docs are historical. |
| "Perfect-fill or fill=1.000 is the headline." | Current primary metric is Excel ReT; fill can be saturated or misleading. |
| "Strictly Pareto-dominates on all metrics." | Resource-efficient win is not confirmed; report verdicts separately. |
| "PPO anticipates risks." | Current evidence supports adaptive recovery; anticipation requires lead/lag proof. |
| "H4 is proven." | H4 is not proven on the winning Track B lane. |
| "Track A preventive/coarse-frontier wins are publishable." | Dense CRN and rich metrics falsified those lanes. |

## Q1 Verification Roadmap

### 1. Canonical Track B Audit

Goal: lock H1 under the paper's primary metric.

Use the canonical run:

```text
outputs/experiments/track_b_gain_2026-06-30/top_tier_confirm_v3_output/track_b_top_tier_confirm_5seed_60k_h104/
```

Required outputs:

- PPO vs dense static CRN table.
- Garrido-style workbook with policy summary, metric deltas, APj/RPj/DPj/CTj/ReTj ledger, CVaR, CD, cost, action traces.
- Bootstrap CI95 and Cohen's d for Excel ReT, CVaR05, flow fill, CD, service-loss AUC, cost.
- Pareto figures: ReT-cost, ReT-CVaR/tail, ReT-flow.

Current stats/figure bundle:

```text
outputs/audits/track_b_q1_stats_2026-07-01/
```

Primary dense-static comparator selected by Excel ReT:
`S2_op10_2.00_op12_1.50`.

Primary result:
`order_ret_excel` PPO `0.005898` vs dense static `0.005460`,
delta `+0.000438`, CI95 `[+0.000409, +0.000468]`,
seed-clustered CI95 `[+0.000421, +0.000458]`.

Pareto result:
PPO is non-dominated versus the dense static frontier on
Excel ReT, cost, CTj p99 tail, and flow fill.

Command skeleton:

```bash
.venv/bin/python scripts/run_track_b_dense_crn_static.py \
  --run-dir outputs/experiments/track_b_gain_2026-06-30/top_tier_confirm_v3_output/track_b_top_tier_confirm_5seed_60k_h104 \
  --output-dir outputs/experiments/track_b_dense_crn_static_final \
  --seeds 1,2,3,4,5 \
  --eval-episodes 12 \
  --reward-mode control_v1 \
  --risk-level adaptive_benchmark_v2 \
  --observation-version v7 \
  --max-steps 104 \
  --export-order-ledger
```

### 2. Mechanism Audit

Goal: prove the mechanism is recovery/backlog control and set the language boundary.

Required checks:

- Op10/Op12 mean, p90, p95, p99 vs backlog, rolling fill, regime, risk/hazard.
- Shift mix and switching frequency.
- CTj/RPj/DPj p95/p99 and service-loss AUC.
- Lead/lag windows: action before, during, after R22/R23/R24 or backlog spikes.

Command skeleton:

```bash
.venv/bin/python scripts/audit_track_b_mechanism.py \
  outputs/experiments/track_b_gain_2026-06-30/top_tier_confirm_v3_output/track_b_top_tier_confirm_5seed_60k_h104 \
  --output-dir outputs/audits/track_b_mechanism_final
```

Decision rule:

- If action rises before risk/backlog, allow limited "anticipatory" wording.
- If action rises only after pressure appears, use "adaptive recovery / backlog control".

### 3. Current-Contract Ablations

Goal: answer "Track B just gives the agent more power" with the current 8D contract.

Required arms:

- `joint`
- `downstream_only`
- `shift_only`
- fixed-shift variant if implemented or added as a small extension
- no-risk/no-hazard or shuffled-hazard observation variant if available

Command skeleton:

```bash
.venv/bin/python scripts/run_track_b_ablation.py \
  --output-dir outputs/experiments/track_b_ablation_8d_final \
  --ablation-configs joint downstream_only shift_only \
  --reward-mode control_v1 \
  --risk-level adaptive_benchmark_v2 \
  --observation-version v7 \
  --seeds 1 2 3 4 5 \
  --train-timesteps 60000 \
  --eval-episodes 12 \
  --max-steps 104 \
  --n-envs 4 \
  --learning-rate 0.0001 \
  --export-order-ledger
```

Important: this runner trains a learned policy for each ablation arm. It is not
a free frozen-policy evaluation. Use it after the mechanism audit unless a
lighter frozen-policy ablation runner is added.

Decision rule:

- If downstream-only keeps most of the win and shift-only loses, the bottleneck-alignment story is strong.
- If joint is the only winner, the claim becomes "downstream plus capacity coordination" rather than downstream sufficiency.

### 4. Generalization

Goal: show the result is not a single-cell artifact.

Minimum matrix:

- Risk levels: `current`, `increased`, `severe`, `adaptive_benchmark_v2`.
- Families: R2-only, R24-only, mixed/all-risks where supported.
- Horizons: h52, h104, h260.

Report:

- Excel ReT delta vs best static.
- CVaR05 delta.
- Flow/service/tail deltas.
- Cost delta.
- Win/loss by cell; do not average away failures without showing them.

Decision rule:

- Strong Q1 claim if PPO is positive or non-inferior in most cells and never catastrophically worse.
- If wins are concentrated in adaptive_benchmark_v2, frame as a stress-campaign result with generalization as future work.

### 5. H4 Retained-vs-Reset

Goal: optional upgrade from "adaptive control" to "retained operational learning".

Run only after H1 and mechanism are locked.

Required protocol:

- Conditions: `obs_full` and `obs_hidden`.
- Arms: frozen, reset, retained, retained-shuffled, no-update, wrong-history.
- Primary estimand: `Delta_memory = ReT_retained - ReT_reset`.
- Negative controls must be reported.
- One-seed-per-kernel or partial writes, CPU forced, live watcher mandatory.

Claim rule:

- H4 claim only if CI95 lower bound > 0, preferably in `obs_hidden`.
- If null, H4 is an open question, not a failed main paper.

## Reviewer #2 Defense Package

| Attack | Required response |
|---|---|
| Track B is just more powerful. | Show 8D ablations; emphasize action-space alignment, not dimension count. |
| Track B is not Garrido-faithful. | Label Track B as operational extension; Track A is thesis-variable boundary evidence. |
| Reward tuning/fishing. | Show pre-registered gates, held-out CRN, current sweep, and failed lanes in the registry. |
| PPO only. | Say action-space mechanism is the primary tested variable; RecurrentPPO negative; SAC is robustness/future work unless run. |
| No retained-adaptation evidence. | H4 is now positive at small effect size, but the paper's main claim remains adaptive closed-loop recovery/action-space alignment. |
| Single topology. | Defend benchmark depth and thesis validation; add cross-regime/horizon generalization table. |
| Excel ReT quirks. | Report full Garrido-style panel, tail/CVaR, service, cost, and order ledger; do not rely on one scalar. |
| Static frontier too weak. | Use dense CRN in the same action space; no coarse-frontier claims. |
| Mechanism is reactive, not anticipatory. | Agree unless lead/lag proves otherwise; use "adaptive recovery / backlog control". |

## Corrections to Recent Draft Plans

- The Track B observation audit is mostly correct, but the current code has
  `v7=52`, `v8=79`, and `v9=89`; v9 adds **10** dimensions, not 7.
- `variance_log` CD is already ported and active through
  `supply_chain/data/ret_garrido2024_calibration.json`; what remains is not a
  port, but final same-bar reporting and avoiding full-cost CD as the headline.
- "CD same-bar" is supported as an **evaluation** claim; `ReT_cd_balanced` as a
  training reward underperformed in the controlled campaign and should not be
  promoted without a new sweep win.
- The safest immediate work while Kaggle runs is mechanism analysis and
  statistical/figure generation. Current 8D ablations are valuable but are not
  zero-cost because the available runner trains each arm.

## Submission Readiness Checklist

- [ ] Final canonical Track B workbook rebuilt from current dense CRN and rich metrics.
- [x] Cohen's d and CI95 table added for all headline metrics.
- [x] Pareto figures generated.
- [ ] Current 8D ablations completed or the causal claim softened.
- [ ] Mechanism lead/lag audit completed or anticipation language removed.
- [ ] Generalization table completed or scope narrowed.
- [ ] H4 either confirmed and reported, or explicitly deferred as open.
- [ ] Manuscript/docs scrubbed for retired dimension labels, perfect-fill headlines, old training-budget-only claims, strict all-metric dominance claims, and Track-B-as-thesis-faithful wording.
| C18 | Dispatch-inclusive costing favors PPO: pricing dispatch expediting at λ≥0.025 makes PPO significantly cheaper in total than the best static (CI95<0). | **Supported** | `docs/track_b_q1_stats_2026-07-02_final_10seed/dispatch_cost_sensitivity.csv`: C(λ)=shift cost+λ[(m10−1)++(m12−1)+], 120 CRN pairs; PPO mean multipliers ≈1.30/1.27 vs static 2.00/1.50 constant. At λ=0 PPO is nominally cheaper but CI crosses zero; from λ=0.025 upward CI95 is fully negative. In manuscript Table (dispatch_cost) §4.3. | Do NOT claim strict cost dominance at λ=0; claim conditional cost advantage under positive dispatch charges. |
| C19 | PPO beats each of the top-12 statics individually (winner's-curse defense). | **Supported** | `docs/track_b_q1_stats_2026-07-02_final/top12_static_robustness.csv`: all 12 deltas +0.000426..+0.000465, CI95 all positive; delta grows down-ranking. Manuscript Appendix A. | — |
| C20 | Local upstream static bound does not overturn the Track B PPO advantage. | **Supported as a local bound, not an exhaustive eight-dimensional frontier** | `outputs/experiments/track_b_upstream_bound_3x3_2026-07-02/` and mirrored `docs/track_b_q1_stats_2026-07-02_final/upstream_bound_3x3_*`: 3x3 eval-only grid over Op3/Op9 quantity multipliers `{0.75,1.00,1.25}` at S2, Op10×2.00, Op12×1.50; best bound policy ReT `0.0056120`, PPO `0.0056660`, seed-paired delta `+0.0000540` CI95 `[+0.0000424,+0.0000656]`; CRN keys verified across 9 policies × 60 episodes. | Say "dense downstream-dispatch grid plus local upstream bound"; do NOT claim an exhaustive eight-dimensional static frontier. |
| C21 | Track B headline result holds at 10 training seeds, not just 5. | **Supported** | `docs/TRACK_B_SEED_EXPANSION_6_10_VERDICT_2026-07-02.md` + `docs/track_b_q1_stats_2026-07-02_final_10seed/`: seeds 6-10 CRN-paired against the exact canonical best static (S2_op10_2.00_op12_1.50, 147-cell convention), 120 total pairs. Authoritative Excel ReT delta +0.000438, CI95 [+0.000409,+0.000468], seed-clustered CI95 [+0.000421,+0.000458], 10/10 seeds positive; order-level ReT delta +0.000426. | Cited in manuscript sections/04_results.tex §4.3; Table 4 now uses the 10-seed headline comparison. |
| C22 | Track B (canonical, KAN sidecar, DMLPA sidecar, all `make_track_b_env`-based experiments) run on the *corrected* raw-material flow mode, not the stale gate that produced the June 28 fidelity numbers. | **Supported, with a disclosed fidelity nuance** | Verified 2026-07-03: `MFSCGymEnvShifts.__init__`'s own default (`env_experimental_shifts.py:390`) is `raw_material_flow_mode="kit_equivalent_order_up_to"`, and `make_track_b_env` never overrides it — so every Track B run this session (canonical headline, KAN-RBF sidecar, DMLPA sidecar, gae_lambda Track A retunes) already uses the corrected mode by construction, no explicit flag needed. BUT per `docs/E6_FIDELITY_MODE_RECONCILIATION_2026-07-02.md`, this corrected mode gives a WEAKER risk_r3 fidelity story than the old (wrong-mode) June 28 gate: H2 r3 8/10 (p=0.055, borderline) and H3 r3 6/10 (p=0.377, NOT a sign-test pass) vs the superseded legacy_validated gate's 9/10 and 7/10. r1/r2 remain strong (10/10 both gates). | The manuscript currently does not cite specific H2/H3 sign-test counts (checked 03_methodology.tex, 04_results.tex — only general "±15% validation threshold" language), so there is no active inconsistency to fix. But if/when a fidelity subsection or table is written, it MUST use the corrected numbers (8/10, 6/10 for r3) and describe this as a "ReT-sign fidelity/moderation gate," not broad fill-rate fidelity, per the E6 doc's own wording. Do not resurrect the legacy_validated r3 numbers. |
| C23 | A resilience/recovery-weighted training reward (`ReT_tail_v2`) improves Track B's tail-risk and time-to-recovery metrics over the canonical `control_v1` reward, at no cost penalty, without becoming causally preventive. | **NOT supported at n=15 — the n=10 signal regressed to noise on every metric; do not cite** | 15-seed/60k, paired per-seed vs. `control_v1` fixed-RNG canonical (seeds 1-5 `track_b_fixed_rng_confirm_5seed_60k_2026-07-03`, 6-10 `track_b_control_v1_fixed_rng_extension_6_10_60k_2026-07-05`, 11-15 `track_b_control_v1_fixed_rng_extension_11_15_60k_2026-07-05`; `ReT_tail_v2` same seed structure from `track_b_ret_tail_v2_confirm_5seed_60k_2026-07-04` / `_extension_6_10_60k_2026-07-05` / `_extension_11_15_60k_2026-07-05`), all `max_steps=104` verified. **Adding seeds 11-15 reversed the n=10 read: every CI95 now includes zero.** ReT Excel Δ+0.000008 CI95[-0.000004,+0.000021] (9/15 favorable, was 7/10); CVaR05 Δ+0.000045 CI95[-0.000025,+0.000115] (10/15); rolling_4w_min Δ+0.000052 CI95[-0.000013,+0.000117] (10/15, **no longer clears zero — cleared at n=10**); service_loss_auc/order Δ-5294 CI95[-15435,+4847] (9/15); ttr_mean Δ-0.306h CI95[-1.000,+0.388] (10/15); ttr_p95 Δ-2.616h CI95[-6.477,+1.244] (10/15, **no longer clears zero — cleared at n=10**); cost_index Δ-0.011 CI95[-0.075,+0.053] (8/15). Seeds 13,14,15 are unfavorable on nearly every metric, pulling all means toward zero. Causal counterfactual (unaffected by this) remains negative: R22 9.4% positive pairs, R24 10.4%. | This is a textbook early-look/winner's-curse pattern (n=10 showed 2/7 metrics clearing zero; n=15 shows 0/7) — do NOT cite `ReT_tail_v2` as an improvement over `control_v1` in any form. Keep as a documented negative-after-extension lesson, same category as the continuous_its `ReT_tail_v2` screens that also failed to confirm (`PROMISING_LANES_REGISTRY.md` ⭐1b-v8-tail, ⭐2, ⭐8). See `docs/PROMISING_LANES_REGISTRY.md` ⭐13 for the full seed-by-seed breakdown. |

**C23 update after n=15 extension (2026-07-05): keep Candidate only; do not promote.** Seeds 11-15
were added for both `control_v1` and `ReT_tail_v2`
(`track_b_control_v1_fixed_rng_extension_11_15_60k_2026-07-05`,
`track_b_ret_tail_v2_extension_11_15_60k_2026-07-05`) and merged in
`outputs/experiments/track_b_ret_tail_v2_15seed_merged_summary_2026-07-05`. With CI95 t(14), no
metric clears zero: ReT Excel Δ+0.000008 CI95[-0.000004,+0.000021] (9/15); CVaR05 Δ+0.000045
CI95[-0.000025,+0.000115] (10/15); rolling_4w_min Δ+0.000052 CI95[-0.000013,+0.000117] (10/15);
service_loss_auc/order Δ-5294 CI95[-15434,+4846] (9/15); ttr_mean Δ-0.306h
CI95[-1.000,+0.388] (10/15); ttr_p95 Δ-2.616h CI95[-6.476,+1.244] (10/15); cost_index Δ-0.011
CI95[-0.075,+0.053] (8/15). Final C23 stance: exploratory/candidate only, useful as future-work
reward-design evidence, not a manuscript claim.

| C24 | A reviewer-safe no-forecast Track B package is now available at h104, with PPO+MLP as the all-risk spine and Real-KAN as an interpretable efficiency sidecar; controlled downstream stress creates larger adaptive headroom but still not a prevention claim. | **Supported with claim boundary** | `docs/TRACK_B_FINAL_AUDIT_PACKAGE_2026-07-06.md`. Final A/B h104 no-forecast, 5 seeds x 60k x 12 eval: Case A all risks PPO ReT `0.005900714`, delta vs best static `+0.000256755` (+4.55%), CVaR05 `0.001911092`, cost `0.488`; Real-KAN bs256 ReT `0.005854707`, delta `+0.000210748` (+3.73%), cost `0.378`. Case B downstream-only PPO and Real-KAN tie on ReT (~`0.60478`) while Real-KAN is cheaper. Independent VPS PPO corroboration matches local PPO (Case A `0.005903104`, Case B `0.604846242`). Selected Case C (`R24` frequency x3, `R22/R23` impact x1.5) confirms adaptive headroom: PPO ReT `0.481160397`, delta `+0.046323705` (+10.65%); Real-KAN ReT `0.470095597`, delta `+0.035258905` (+8.11%) with much lower cost. Mechanism is branch/exposure reduction, not proven anticipation. | Use Case A PPO no-forecast h104 as reviewer-safe main result; use Real-KAN as novelty/interpretable sidecar; use Case C as stress/adaptation panel. Do not compare Case B/C absolute ReT to Case A; do not claim causal prevention without a new event-aligned counterfactual. |

| C25 | The old Ruta B pre-risk splice gate is not valid prevention evidence; preventive headroom is closed for the current Track B Case C DES/action contract, while Ruta B's surviving value is an architecture-driven efficiency effect. | **Supported as a negative/methodological boundary** | `docs/TRACK_B_RUTA_B_COUNTERFACTUAL_GATE_AUDIT_2026-07-07.md` retracts the 74.1% Ruta B positive-pair result because reactive PPO and permuted-label controls also pass the splice/calm-action gates. `docs/TRACK_B_RUTA_B_ESTABLISHED_GATE_FINAL_2026-07-07.md` reruns the trusted established gate in the correct Case C environment: reactive/Ruta B positive rates remain 3-19%, far below the preventive bar. `docs/TRACK_B_PREVENTIVE_HEADROOM_CEILING_VERDICT_2026-07-07.md` then closes the ceiling: forced-prep response surface is flat in Case C and exactly zero in R22-only clean physics (0/84 real, 0/96 placebo), while clairvoyant PPO with true future labels visible in train+eval does not improve over the reactive baseline and costs more (`ReT=0.485035`, cost `0.853`). `docs/PREVENTION_GATE_AUTOPSY_AND_CLOSURE_2026-07-07.md` is the consolidated closure. | Do not cite Ruta B as preventive. Keep the paper framed as adaptive recovery/exposure reduction/action-space alignment. Ruta B may be cited only, if needed, as an efficiency-via-architecture side result: true/permuted/constant/λ=0 variants all match ReT with low cost, so label content and auxiliary loss are not the mechanism. |

| C26 | The preventive-headroom boundary (C25) generalizes beyond R22 to the full mediable risk roster (R23, R12, R13, R21) and to the `surge_inertia` dispatch-lag lever, under `track_b_v1`. | **Supported, stronger than C25's R22-only scope** | `docs/TRACK_B_PREVENTION_HEADROOM_GENERALIZED_VERDICT_2026-07-08.md`. Same forced-prep response-surface design as the R22 ceiling test, run on the risks with an in-principle mediable channel (R23: theatre pre-positioning bridges an Op12 block; R12/R13: upstream buffer levers bridge contract/supplier stalls; R21: multi-op knockout, downstream-prepositioning-mediable in principle) plus `surge_inertia` (activation lag + finite budget on shift changes, the one lever built specifically to reward pre-positioning). Results: R23-only clean physics 0/45 real, 0/96 placebo, DiD exact 0.0; R12-only 0/65 real vs 0/96 placebo; R12+R13 background 0/66 real vs 2/96 placebo; R21-only 0/24 real, 0/96 placebo, DiD exact 0.0; R23-only+surge_inertia 0/45 real, 0/96 placebo, DiD exact 0.0. The one non-null cell (R23 evaluated inside full Case C, R22+R23+R24 all active) is 2/28 (7.1%) real vs 4/96 (4.2%) placebo — inside the same 3-19% noise band every negative result this investigation has produced, and directly contradicted by its own clean-physics R23-only tier's exact zero; read as the same busy-environment contamination pattern diagnosed in the original gate autopsy, not a real R23-specific channel. Independently corroborated by two previously-undigested 2026-07-04 runs found during a historical-runs audit: `track_b_oracle_resilience_metrics_2026-07-04` (fixed 8-week-lead boost oracle, ReT moves ~0.1-0.3% relative = flat) and Codex's independent event-tape Gate v2 (`docs/TRACK_B_PREVENTION_GATE_V2_IMPLEMENTATION_2026-07-07.md`, `prevention_headroom_found=false` on its own Case C screen). | Cite as the primary prevention-boundary claim in the manuscript's prevention subsection/appendix — "no preventive channel for this risk family," not "R22 specifically." Caveat to state explicitly: all tiers reused the Case C reactive PPO checkpoint as reference policy, off-distribution for single-risk rosters; the exact-zero results (4/6 tiers) make this unlikely to change the conclusion but it should be disclosed. Do not reopen Phase 2 (gate redesign) or resume prevention training without a pre-registered environment change (e.g. a real action lead time) per the stop rule. |

| C27 | The Track B win is not specific to PPO: SAC and TD3 replicate it under the identical canonical protocol. | **Supported** | 3 seeds x 30k, `adaptive_benchmark_v2` v7 h104, `control_v1` reward, identical CRN eval + static/heuristic comparator set as the canonical PPO screen (`scripts/run_track_b_smoke.py --algo sac\|td3`, new algorithm-scope screen added 2026-07-08). SAC: `order_ret_excel_mean=0.005911` vs best static/heuristic (`heur_disruption_aware`, `0.005418`), delta `+0.000493` (**+9.10%**), all 3 seeds individually positive (`0.005894`/`0.005933`/`0.005905`). TD3: `0.005893` vs the same comparator, delta `+0.000475` (**+8.77%**), all 3 seeds positive (`0.005870`/`0.005894`/`0.005914`). Both land in the same range as PPO's own screen-scale result. `outputs/experiments/track_b_{sac,td3}_screen_3seed_30k_2026-07-08/`. | Closes the algorithm-scope reviewer objection (previously only worded as a limitation in discussion §5.5/future-work §5.6, no experiment run). Cite in the limitations/robustness section: PPO's win is not an artifact of on-policy learning specifically. A Kaggle confirmatory scale-up (5 seeds x 60k) was built and locally smoke-tested but blocked from pushing by the harness's data-exfiltration classifier (embeds the repo as base64 for kernel upload) — script sits at `kaggle/track_b_sac_td3_confirm/`, untracked, pending either a manual out-of-session push or a stripped-down payload that the classifier accepts. |

| C28 | Under a pre-registered commitment contract (`track_bp_v1`: track_b_v1 + lagged strategic-buffer targets at Op3/Op5/Op9, Garrido's I_{t,S} made dynamic), preventive headroom EXISTS — but only for risks whose outages stock can physically bridge, and only in regimes that actually drain working stock. | **Supported (Gates 0/1); Gate 2 conversion screen in progress** | `docs/TRACK_BP_PREREGISTRATION_2026-07-08.md` (rules/falsifiers fixed before results) + `docs/TRACK_BP_GATES_0_1_VERDICT_2026-07-09.md`. Gate 0 (forced-prep causal DiD, constant-neutral reference, no checkpoint caveat): R11 rare-long (freq×0.125, impact×8) real anchors `+0.002856` (31+/4−/27·, n=62) vs placebo `+0.000182` (15.7× smaller), DiD `+0.002674` bootstrap CI95 `[+0.001061,+0.004622]` — the FIRST causal forced-prep positive of the program. Gate 1 (clock-policy oracle, n=24 CRN): R21 compound starvation (freq×8, impact×4) always−never `+0.0299` episode ReT CI95 `[+0.0142,+0.0478]` (+13.9% rel., 14+/10·/0−). Nulls preserved everywhere else: R21-isolated exact 0/16, R23, R24, R13 null; R22 negative control in band. All numbers recomputed from raw CSVs. `outputs/experiments/track_bp_g{0,1}_*_2026-07-08/`. | This does NOT contradict C25/C26: those closed prevention under `track_b_v1`'s instant levers; C28 shows the boundary is contract-dependent, which STRENGTHENS the paper's action-space-alignment thesis (the preventive frontier, like the adaptive one, appears exactly when the action space carries the right physics — here temporal commitment). Keep out of the current manuscript (paper-2/extension lane). CORRECTION 2026-07-09: the R11 Gate-0 signal is RETRACTED as prevention evidence — a buffers-only forced posture reproduces it at exact zero (0/62 real, 0/40 placebo), and the original 3-point surface shows calm < medium ≈ max_prep, i.e. the delta was de-preparation harm in the calm arm, not anticipatory gain. The C28 claim now rests solely on the R21-compounding cell, where it is STRONGER than first stated: contract ablation isolates a pure preventive increment PPO_11D−PPO_8D = +0.0533 pooled (all seeds CI95>0, 70/72 episodes), 178% of the static oracle, surviving holding charges to λ_h=0.2; regime-breadth grid maps a clean monotone frontier (null for freq≤2 or impact≤2; turns on at freq×4/impact×4). docs/TRACK_BP_GATE2_SCREEN_VERDICT_2026-07-09.md. |
