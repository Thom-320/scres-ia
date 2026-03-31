# Paper Findings Registry

All empirical findings identified during the experimental campaign (2026-03-27 to 2026-03-30).
Each finding includes the evidence source, whether it's confirmed, and how it could appear in the paper.

---

## F1. Reward Misspecification: Resilience Metrics Resist RL Optimization

**Status:** CONFIRMED across 5+ reward variants

**Evidence:**
- ReT_thesis (piecewise Eq. 5.5) → S1 collapse (99.99% S1, fill_rate 0.845)
- ReT_seq_v1 (C-D geometric) → S1-leaning at 500k (61.5% S1, 24.3% S2, 14.2% S3) [source: final_ret_seq_v1_500k/summary.json]
- ReT_garrido2024_raw (5-var C-D) → S1 collapse (n_kappa dominates)
- ReT_garrido2024_train (4-var, no cost) → S3 collapse (88% S3)
- ReT_unified_v1 (gated cost) → competitive but no clear advantage over control_v1
- control_v1 (linear, w_bo/w_cost=200) → PPO fill=0.782, shift mix 46/28/27, does NOT beat S2 [source: paper_control_v1_500k, 5 seeds, POST-AUDIT DES]
- **NOTE:** Earlier claim of fill_rate=0.838 came from control_reward_500k_increased_stopt, which used the PRE-AUDIT DES (known bugs). That run is now classified as `historical_artifact`.

**Finding:** Theoretically grounded resilience metrics (Cobb-Douglas, piecewise) systematically fail as RL training rewards because agents exploit the weakest dimension. However, the simpler linear control_v1 also fails to beat S2 on the corrected DES, suggesting the problem is structural (F11), not purely reward-related. The reward alignment finding remains valid: C-D rewards produce WORSE policies than linear rewards, but neither beats static S2 on service metrics.

**Paper section:** Section 4.2 or dedicated subsection on reward alignment
**Strength:** STRONG — reproducible, multi-variant, directly addresses a gap in the literature

---

## F2. Inventory Action Sensitivity is Asymmetric, and PPO Stays Near the Neutral Region

**Status:** CONFIRMED (DOE) + SUGGESTIVE (action trace)

**Evidence (DOE, 10 seeds, `action_sensitivity_track_a_2026-03-30`):**
- `shift_only_random`: reward=-2,154, fill_rate=0.821
- `q_max_s2`: reward=-2,156, fill_rate=0.815
- `s2_fixed`: reward=-2,175, fill_rate=0.814
- `q_min_s2`: reward=-3,579, fill_rate=0.273
- `rop_max_s2`: reward=-3,137, fill_rate=0.518
- Positive headroom above `s2_fixed` is small: the best simple policy is only about **1%** better in reward.
- Downside is extreme: poor quantity/reorder settings reduce reward by roughly **44-65%** and can collapse fill rate.

**Evidence (short control-reward action trace, 50k training, `w_bo=4.0`, `w_cost=0.02`):**
- Global PPO action mean = `[-0.484, +0.059, +0.057, -0.093, -0.050]`
- Shift remains near the `S2` region.
- Inventory dimensions stay near-neutral or mixed rather than converging to a strong `q_max` / `rop_min` heuristic.

**Interpretation:**
- The 5D action space is **not** flat: destructive settings are easy to identify empirically.
- However, under moderate stress the upside of "better-than-neutral" inventory control is small relative to the downside of bad settings.
- PPO appears to learn the high-impact binary capacity choice (`S2`) while remaining close to the neutral region on inventory actions.
- This pattern is consistent with a conservative response to asymmetric downside risk and delayed pipeline effects, but it does **not** by itself prove a specific credit-assignment mechanism.

**Finding:** Under the thesis-faithful moderate-stress regime, inventory control has strongly **asymmetric sensitivity**: bad settings are highly destructive, while the positive headroom above `S2`-neutral is small. This helps explain why PPO often looks close to static `S2`: the benchmark offers limited upside for aggressive inventory tuning, while penalizing poor settings heavily.

**Paper section:** Section 4.3 + Section 5 (Discussion)
**Strength:** STRONG for the asymmetric-sensitivity pattern; MODERATE for the proposed mechanism

---

## F3. Cumulative Fill Rate is a Lagging Indicator That Masks Policy Quality

**Status:** CONFIRMED

**Evidence:**
- After warmup+priming (2351h), fill_rate starts at 0.42, pending_bo = 124,125
- System never fully recovers from warmup backlog — pending_bo grows from 124k to 145k over 260 steps
- Fill rate climbs from 0.42 to 0.79 regardless of policy (dominated by cumulative history)
- NOTE: the "14 of 946 steps" and "98.5% service_loss" figures come from a diagnostic run WITHOUT max_steps=260, using the full ~946-step horizon. Under the paper-facing 260-step contract, the qualitative finding holds but the specific step counts differ.

**Source:** Diagnostic from Claude instance (step-by-step trajectory analysis). Specific step counts need re-verification under 260-step contract.

**Finding:** The cumulative fill rate metric used for evaluation is dominated by inherited warmup backlog, not by the agent's decisions. This creates a floor effect where all policies converge to similar fill rates (~0.79) regardless of shift strategy, compressing the observable performance gap between learned and static policies.

**Paper section:** Section 4.1 (DES analysis) or Discussion/Limitations
**Strength:** STRONG — explains a structural limitation of the experimental setup

---

## F4. RL Advantage Grows with Disruption Severity

**Status:** PARTIALLY CONFIRMED (directional, not statistically significant)

**Evidence (valid, post-audit DES):**
- paper_control_v1_500k (increased): PPO fill=0.782 vs S2=0.792 → PPO LOSES by 1.0pp
- paper_ret_seq_k020_500k (increased): PPO fill=0.788 vs S2=0.792 → PPO LOSES by 0.4pp
- paper_ret_seq_k020_500k (severe cross-eval): PPO fill=0.484 vs S2=0.495 → PPO LOSES by 1.1pp
- **NOTE:** Earlier evidence from control_reward_500k_*_stopt is INVALID (pre-audit DES). The "PPO marginally better under severe" finding was based on those invalid runs.

**Evidence (invalid, pre-audit DES — for historical context only):**
- control_reward_500k_increased_stopt: PPO vs static_s2 = -1.95 (pre-audit DES, historical_artifact)
- control_reward_500k_severe_stopt: PPO vs static_s3 = +4.61 (pre-audit DES, historical_artifact)

**Finding (REVISED):** On the corrected, thesis-aligned DES, PPO does NOT show a clear advantage under any stress level tested so far. The earlier "severity-dependent gains" finding was an artifact of comparing against the pre-audit DES. This finding needs re-evaluation with valid severe-stress runs on the current DES.

**Paper section:** Section 4.2 (main results table)
**Strength:** WEAK — the directional pattern may still hold but is not confirmed on the corrected DES. Needs new severe-only runs to re-test.

---

## F5. Memory (RecurrentPPO) Provides Modest Improvement

**Status:** PARTIALLY CONFIRMED (from 100k smoke, awaiting 500k)

**Evidence:**
- smoke_unified_v4_recurrent_168h_100k: RecurrentPPO reward=140.02 vs PPO+MLP reward=137.78 (same v4/168h setup)
- RecurrentPPO beats static_s2 (140.02 vs 139.40)
- PPO+MLP does not beat static_s2 (137.78 vs 139.40)
- Shift mix: RecurrentPPO 46/46/8 vs PPO+MLP 52/35/13

**Evidence (500k × 5 seeds, COMPLETED):**
- final_recurrent_ppo_v4_control_500k (increased): RecPPO fill=0.751 vs S2=0.794 → **LOSES by 4.3pp**
- final_recurrent_ppo_v4_control_500k (severe): RecPPO fill=0.422 vs S2=0.499 → **LOSES by 7.7pp**
- Shifts: 51% S1, 22% S2, 28% S3 (S1-dominant)

**Finding (REVISED):** RecurrentPPO (LSTM) does NOT improve over static S2 at 500k × 5 seeds. The earlier 100k smoke suggested modest improvement, but the production run shows clear degradation. Memory alone does not solve the structural MDP limitation (F11). The POMDP hypothesis is not supported as a sufficient explanation.

**Paper section:** Section 4.3 (algorithm comparison)
**Strength:** STRONG as negative result — definitively closes the "memory helps" hypothesis for Track A.

---

## F6. Anticipatory Cycle-Phase Signals Improve Policy Quality

**Status:** PARTIALLY CONFIRMED (from 100k smoke)

**Evidence:**
- PPO+MLP v5/168h: reward=139.19, FR=0.796, shifts=26/69/6
- PPO+MLP v4/168h: reward=137.78, FR=0.791, shifts=52/35/13
- v5 adds sin/cos cycle-phase signals for Op1 (biannual) and Op2 (monthly) procurement cycles

**Finding:** Adding thesis-faithful cycle-phase signals to the observation space improves PPO's shift mixing (S2-dominant 69% vs 35%) and fill rate (+0.5pp), suggesting the agent can learn to anticipate procurement risk windows.

**Paper section:** Section 4.3 (observation ablation)
**Strength:** WEAK-MODERATE — only 100k, 3 seeds. The shift mix improvement is the most notable signal.

---

## F7. 48h Decision Cadence Does NOT Clearly Help

**Status:** CONFIRMED NEGATIVE

**Evidence:**
- smoke_unified_v5_48h_100k: PPO reward=455.36, Random=453.24, gap=0.47%
- Statistical tests: PPO vs random p=1.000, PPO vs garrido_cf_s3 p=0.700
- Random achieves 99.5% of PPO's performance at 48h

**Finding:** Reducing decision cadence from 168h to 48h does NOT improve the RL advantage. Instead, it reduces action sensitivity because random shift alternation at 48h accidentally produces reasonable average capacity. The reward function was not calibrated for 48h step size.

**Paper section:** Discussion/Limitations or Future Work
**Strength:** STRONG as a negative finding — prevents false claims about cadence effects

---

## F8. The Cobb-Douglas Form is Mathematically Valid but Operationally Inferior for RL

**Status:** CONFIRMED

**Evidence:**
- ReT_seq_v1 IS a proper C-D function (verified 8 mathematical properties: constant returns to scale, non-compensability, diminishing marginal returns, log-linear equivalence, unit elasticity, output elasticities match weights)
- Garrido et al. (2024) independently uses C-D for factory resilience
- But C-D with intermediate variables (ζ, φ) biases toward S3 (φ ∝ R always)
- C-D with outcome variables (SC, BC, AE) biases toward S1 (AE exploitable)
- No C-D formulation produced better policies than linear control_v1

**Finding:** The Cobb-Douglas functional form is theoretically sound for resilience measurement (non-compensatory, smooth, grounded in Garrido 2024) but structurally inferior to linear rewards for RL training. The multiplicative structure creates exploitable shortcuts that gradient-based optimization finds faster than the intended service improvement.

**Paper section:** Section 3.3 (reward design) + Section 4.2 (empirical comparison)
**Strength:** STRONG — this is the core methodological contribution. No other paper has documented this specific failure mode.

---

## F9. Garrido's Best Static Policy (S=2) is Near-Optimal Under Increased Risk

**Status:** CONFIRMED

**Evidence:**
- garrido_cf_s2 fill_rate = 0.787 and static_s2 fill_rate = 0.792 in `smoke_unified_v5_168h_100k`
- PPO best fill_rate = 0.793 (with ReT_seq_v1 κ=0.20, final_ret_seq_v1_500k) — still below S2's 0.794
- **NOTE:** Earlier claim of PPO fill=0.838 was from pre-audit DES (invalid)
- Under increased risk, S=2 provides enough capacity (5,128 rations/day vs ~2,500 demand)

**Finding:** Under the thesis's "increased" risk parameters, static S=2 operation already provides sufficient buffering capacity (2× demand rate), leaving limited room for adaptive improvement. This explains why RL's advantage is modest under moderate stress but emerges under severe stress where S=2 is no longer sufficient.

**Paper section:** Section 4.1 (baseline analysis) or Discussion
**Strength:** STRONG — connects to Garrido's thesis finding and explains the RL results

---

## F10. The Warmup Backlog Problem is Structural, Not a Bug

**Status:** CONFIRMED

**Evidence:**
- Warmup runs S=1 for 838h with no initial inventory
- This creates ~80k backorder queue before RL starts
- Priming (S=2 for ~1500h) helps but doesn't clear the backlog
- pending_bo grows from 124k to 145k even under S=2 at 260 steps

**Finding:** The inherited warmup backlog is a structural property of the thesis DES, not a bug. It represents the realistic scenario where a supply chain starts operating from scratch under disruption. However, it creates a "floor effect" on cumulative metrics that compresses policy differences.

**Paper section:** Section 3.2 (DES description) + Discussion/Limitations
**Strength:** MODERATE — important for methodology transparency

---

## F11. Downstream Distribution Limits the Marginal Value of Extra Assembly Capacity

**Status:** CONFIRMED (strong structural finding, some claims need caution)

**Evidence:**
- S1 weekly production: ~12,000 → clearly insufficient (fill_rate ~0.62)
- S2 weekly production: ~24,000 → delivered ~3.9M over episode
- S3 weekly production: ~36,000 → delivered ~4.0M (only marginally more than S2 despite 44% more production)
- Op9 dispatches max ~2,500 rations/day (U(2400,2600)), creating a downstream throughput ceiling
- Excess production under S3 accumulates in intermediate buffers (rations_sb reaches ~2.1M)

**Source:** Deep diagnostic analyzing DES production vs delivery flows across shift levels

**Finding:** The MFSC operates in a regime where increasing assembly capacity beyond S=2 yields sharply diminishing returns because downstream distribution absorbs only a small fraction of the extra production. This helps explain why:

1. **Static S=2 is already strong** under moderate stress — it produces enough to keep the downstream pipeline near-saturated
2. **S2 ≈ S3 in delivered rations** — more assembly capacity doesn't translate to proportionally more deliveries
3. **RL's adaptive control gains remain modest** because the agent's primary lever (shifts) has diminishing marginal returns beyond S=2

**Caution:** Assembly capacity DOES matter — S1 is clearly worse than S2. The finding is about diminishing returns beyond S=2, not that assembly is irrelevant. The severe-stress hypothesis (that disruptions bring the constraint back into the agent's zone) is plausible but not causally confirmed.

**Implications for Track B:** To create genuine RL advantage, either expand the action space to include downstream control, or create conditions where the downstream constraint becomes dynamic.

**Paper section:** Section 4.1 (DES bottleneck analysis) + Discussion
**Strength:** STRONG — explains the main pattern of results, connects to diminishing returns / bottleneck theory. Partially confirmed (the mechanism is clear; the severe-stress explanation is hypothesis).

---

## Summary: Findings by Paper Section

| Section | Findings | Combined Strength |
|---------|----------|-------------------|
| 3.2 DES Description | F10 (warmup structural) | Moderate |
| 3.3 Reward Design | F1 (misspecification), F8 (C-D vs linear) | **STRONG** |
| 4.1 DES Results | F3 (lagging indicator), F9 (S2 near-optimal), **F11 (downstream bottleneck)** | **VERY STRONG** |
| 4.2 Main Results | F4 (severity-dependent gains) | Moderate |
| 4.3 Algorithm Comparison | F2 (asymmetric action sensitivity), F5 (memory helps), F6 (cycle signals) | Moderate |
| Discussion | F7 (48h negative), all limitations | Strong |

**The strongest publishable story (updated after audit):**

> "We present a validated DES benchmark for adaptive operational resilience control. On the thesis-aligned simulation, NO RL configuration (across 7 reward modes, 3 observation versions, 2 algorithms) beats static S=2 on fill_rate. We explain why through structural analysis: (1) downstream distribution limits the value of extra assembly capacity (F11), (2) inventory actions have extreme asymmetric sensitivity with only ~1% upside (F2), and (3) resilience metrics systematically fail as RL rewards (F1/F8). This is a benchmark contribution with honest negative results and mechanistic explanations."

**CRITICAL AUDIT NOTE (2026-03-30):** Earlier versions of this registry cited evidence from pre-audit DES runs (control_reward_500k_*_stopt) that showed PPO competitive or marginally better than S2. Those runs are now classified as `historical_artifact` because they used a DES with known bugs (missing Op4 delay, R12/R13 bypass, R21 non-blocking). All paper-facing evidence must use post-audit bundles only (paper_benchmarks/ or benchmarks/ from March 27+).
