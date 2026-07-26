# Clairvoyant-headroom diagnostic and training-progress pilot
*(the explicit learning measure requested by Garrido, meeting 2026-07-22)*

> **Corrected 2026-07-26 after an adversarial internal review.** Three defects were confirmed and
> fixed: a transcribed mean (0.7994 -> 0.7990), a wrong description of the pooled ratio that had
> produced a materially wrong claim about the belief-reset arm, and an undisclosed
> information-rights asymmetry between the neural pilot and the model-predictive arms. The
> corrections are recorded in section 6.

**What Garrido asked for.** An explicit learning metric, measured over time: run the episodes,
compute the clairvoyant maximum resilience for each *already-run* episode (valid post-hoc only,
since it needs the future), then measure what percentage of that theoretical ceiling the model
captures. Learning is confirmed only if the model beats the best static policy.

**Status.** Implemented and computed. `BURNED_DEVELOPMENT_NO_CLAIM_METHODOLOGICAL` — the metric
is an instrument, evaluated on the burned development campaigns (roots 7570801–24); it opens no
sealed block and asserts no new confirmatory claim.

- Metric module: `supply_chain/oracle_capture.py`
- Stage 1 (all existing controllers): `scripts/report_oracle_capture_metric.py`
  → `results/oracle_capture_v1/oracle_capture_metric.json`
- Stage 2 (learning curve): `scripts/run_oracle_learning_curve.py`
  → `results/oracle_capture_v1/learning_curve_<arch>.json`
- Figure: `deliverables/figures/fig7_oracle_metric.{pdf,png}`

## 1. Definition

For campaign *i*, let `L_i[k]` be the exact terminal resilience of calendar *k*, enumerated over
all **4⁸ = 65,536** weekly allocation calendars (objective `early_ret_complete_cohort`, the
canonical ReT cohort):

- `C_i = max_k L_i[k]` — the **clairvoyant ceiling**. Exact, not estimated, and a valid upper
  bound for *any* policy in this action space, including one with privileged information.
- `B_i` — a **static bar** (below).
- `V_i = L_i[k(policy)]` — the policy's value, obtained by **table lookup**: no re-simulation,
  so the grading introduces no numerical noise of its own.
- `η_i = (V_i − B_i) / (C_i − B_i)` — the **capture ratio**. η = 1 means the policy matched a
  decision-maker who knew the entire future; η = 0 means it did no better than the static bar;
  η < 0 means worse.

Two aggregations are reported, because they answer different questions:

- **per-campaign η**, clustered on `history_root` with one-sided LCB95. Undefined where the bar
  already sits at the ceiling, so it is conditional on the campaigns that have headroom.
- **pooled capture** `Σ(V_i − B_i) / Σ(C_i − B_i)` over **all** 48 campaigns. A zero-headroom
  campaign contributes 0 to both sums, so nothing is discarded. This is the headline number.

## 2. The bars

- **Best static open-loop** (the bar Garrido named): the single calendar maximizing the mean
  exact label across the 48 campaigns — `[0,0,3,3,3,3,3,3]`, mean ReT **0.7376**. It is allowed
  full knowledge of the campaign *distribution* but none of the individual campaign, so beating
  it is evidence of state-dependent decision-making rather than of a lucky fixed guess. This is
  a deliberately hard bar.
- **Uninformed expectation** (Baseline-0 analogue): the mean label over all 65,536 calendars,
  i.e. an arbitrary discretionary calendar in expectation. This is the honest stand-in for
  Garrido's Baseline 0, whose own policies are discretionary and, in his words, "clearly
  suboptimal".
- **Constant allocations** (0, 1, 2, 3 P_C batches every week) as interpretable anchors.

## 3. Stage 1 — where every controller stands

Ceiling mean **0.8203** (range 0.7487–0.8768) over the 48 campaigns. Best static **0.7376**.

| controller | mean ReT | pooled capture vs static (LCB95) | per-campaign η (n=21) | exact optima |
|---|---|---|---|---|
| Clairvoyant ceiling | 0.8203 | 1.0000 | 1.0000 | 48/48 |
| **Retained belief-MPC** | **0.7990** | **+0.743 (+0.554)** | +0.789 | **42/48** |
| Service-aware variants (9) | 0.796–0.799 | +0.692 … +0.726 | +0.764 … +0.789 | 39–41/48 |
| Best static calendar | 0.7376 | 0.000 | 0.000 | 27/48 |
| Belief-reset MPC | 0.7293 | −0.099 (−0.617) | +0.326 | 0/48 |
| Constant 1:2 | 0.6582 | −0.957 | −0.679 | 0/48 |
| Constant 2:1 | 0.6076 | −1.565 | −0.594 | 0/48 |

Four findings worth reporting to Garrido:

1. **The retained belief-MPC captures 74% of the clairvoyant headroom** over a hard static bar,
   LCB95 +0.554, and hits the *exact* optimum in 42 of 48 campaigns. On his criterion, this
   controller demonstrably learns: it beats the best static policy with the lower bound well
   clear of zero.
2. **Retention's contribution is mostly about restraint, not about acting better.** The
   belief-reset MPC — same machinery, same horizon, no knowledge carried between campaigns —
   scores −0.099 pooled, but that single number is misleading and the decomposition matters:
   on the 21 campaigns that actually offered headroom it still captures **+0.546**, while on the
   27 campaigns where the static calendar was already optimal it is charged **−2.562** for
   breaking them. The retained arm captures more where headroom exists (+0.799) and, far more
   decisively, does much less damage where none exists (−0.221). Retention buys knowing *when
   not to act*.
3. **In 27 of 48 campaigns a single static calendar is already exactly optimal.** There is
   literally nothing to learn in more than half the population. This is not a defect of the
   controllers; it is a property of the decision problem, and it is the quantitative core of the
   "when *not* to train" argument. Averaging over these campaigns without saying so would
   understate every controller and hide the real structure.
4. **The nine service-aware variants move within exact value ties.** They select a calendar
   different from the retained arm in 16–46 campaigns, but in 39–45 of those the objective value
   is *identical to machine precision*. The ReT objective cannot distinguish those calendars —
   only the service ledger can. That is the same tie-plateau structure the Fig-5 surrogate study
   documented, and it explains mechanically why the Track A screen changed service statistics
   without changing ReT.

## 4. Stage 2 — the learning curve

A learner is trained on campaigns built from a **fresh, previously unused root block
(7650001–7650040, 440 campaigns)**, disjoint from the 48 evaluation campaigns and built through
the same `rebuild_campaign` path, so train and test distributions match. Every 3,000 environment
timesteps the deterministic policy is rolled out on the 48 evaluation campaigns and graded by
lookup, giving capture-versus-experience — exactly the curve Garrido asked for.

The learner is rewarded on `early_ret_complete_cohort`, the same scalar the ceiling and the MPC
arms are graded on. The frozen environment rewards `ret_visible`, so the objective is recomputed
by a thin wrapper rather than by editing the frozen environment.

Two architectures were run at 5 seeds each, 48,000 timesteps (6,000 episodes), 17 evaluation
checkpoints per seed: PPO with an MLP policy, and RecurrentPPO with an LSTM policy (the
incumbent in the three-model comparison).

### Result: learning is visible, and it does not reach the static bar

| checkpoint | PPO+MLP capture (mean of 5) | RecurrentPPO capture (mean of 5) |
|---|---|---|
| 0 (untrained) | −1.298 | −1.959 |
| 6,000 | −0.987 | −2.881 |
| 15,000 | **−0.466** (best) | −2.280 |
| 30,000 | −0.872 | −2.337 |
| 48,000 | −0.630 | −1.190 |
| seeds above the static bar at the end | **0 / 5** | **0 / 5** |

Two things are simultaneously true, and both belong in the paper:

1. **The learners do learn.** PPO+MLP improves from −1.30 to −0.47 within 15,000 timesteps, and
   the number of *distinct* calendars it produces across the 48 campaigns rises from 8 to about
   17–21 — it is conditioning its allocation on the observed state, not settling on a constant.
   RecurrentPPO improves too, later and less (−1.96 → −1.19). The metric registers learning
   cleanly, which is exactly what it was built to do.
2. **Neither crosses the bar.** At Garrido's criterion — beat the best static policy — learning
   is **not confirmed** for either learner at this budget: 0 of 10 seeds ends above zero, while
   the structured retained belief-MPC sits at **+0.743**. PPO+MLP also plateaus by ~15,000
   timesteps and then oscillates without further progress, and one RecurrentPPO seed stays
   collapsed on a constant allocation for the whole run (capture −3.65).

**What this does and does not license.** It licenses: "on this decision problem, at this budget,
the trained policies improve steadily but remain below a hard static bar that a structured
belief controller clears by a wide margin". It does **not** license any ranking of MLP versus
LSTM (library-default hyperparameters, no tuning, small budget), nor a general claim that RL
cannot succeed here. The general claim is already established far more strongly and
architecture-independently by the exact ceiling analysis: the C6 gate enumerated the finer
8⁸ = 16,777,216 per-batch action space and found no stratum where the ceiling itself clears the
gate over the frozen controller. That bounds every policy **within that action space, on those
48 burned campaigns, at the preregistered decision threshold** — it is a statement about this
decision problem, not about learning methods in general, and it must not be quoted as "RL cannot
work".

## 5. Why this metric is worth the paper's space

An absolute ReT number answers "how good is this policy?" only against whatever comparator
happens to be at hand. The capture ratio answers "how much of what was *achievable* did the
policy actually get?", against a ceiling that is exact rather than estimated and that no policy
can exceed. It converts an open-ended search for a better controller into a bounded efficiency
statement, and it makes a null result *informative*: when the ceiling itself sits at the static
bar, the correct engineering conclusion is not to train a network, and the metric says so with
a number instead of a shrug.

It also gives Garrido what he pressed for in a form that survives review: learning is confirmed
by a lower confidence bound above a hard static bar, measured over training time, on episodes
whose optimum is known exactly.

## 6. Corrections applied 2026-07-26 (adversarial internal review)

**C1 — transcribed mean.** The retained arm's mean ReT is 0.799035, which rounds to 0.7990; the
table had 0.7994. The DOCX was unaffected because it reads the JSON directly. Fixed.

**C2 — the pooled ratio was described wrongly, and the description hid a real result.** The
documentation claimed a zero-headroom campaign "contributes 0 to both sums, so neither helps nor
hurts". False: it adds 0 to the denominator but still adds (V − B) to the numerator, which is
negative whenever a controller breaks an already-optimal static calendar. The calculation is
kept — charging regressions is deliberate — but it is now described correctly and reported as
three numbers (pooled over 48, conditional on the 21 with headroom, penalty on the 27 without).
This overturned the earlier claim that feedback without retention "captures nothing": it
captures +0.546 where there is headroom, and its negative pooled figure is almost entirely the
−2.562 penalty for damaging campaigns that needed no action.

**C3 — unequal information rights in the pilot.** The retained MPC is initialized with
`fixed_theta_belief(retained_prior)`, a posterior carried across campaigns. The learner's
environment has no mechanism to accept a carried prior, and the rollout passes
`episode_start=True` at every campaign boundary, which resets the recurrent state. The learner
was therefore structurally incapable of the accumulation that produces the retained arm's
advantage, and the head-to-head reading conflated architecture with information rights. The
pilot is now labelled as such in the document, in the DOCX and on the figure panel, and the
matched-rights design is preregistered in
`contracts/oracle_retained_learning_curve_v2.json` before any further training.

**C4 — scope of the ceiling claim.** "Bounds every policy, trained or not" is now stated with its
three qualifiers: that action space, those campaigns, that threshold.

**Not changed:** the ceiling is exact, the lookup grading is exact, the retained arm's 42/48
exact optima stand, and the tie-plateau finding stands. What changed is what may be *concluded*
from the pilot, and one substantive claim about the reset arm that was wrong in the reader's
favour and is now corrected against us.
