# Garrido Meeting Script — Track B, Reward Design, and Paper Framing

Date: 2026-04-03
Audience: Alexander Garrido, David Pro, Thom Chisica
Goal: present the new benchmark results without overselling novelty or architecture

## Core message

The main result is not that we are the first, and not yet that DKANA is already superior.
The main result is that RL becomes effective once the validated DES benchmark gives the agent control over the active operational bottleneck, and that resilience-aligned rewards can then train strong policies.

## What to show

1. The problem with the old framing
2. The Track A diagnostic
3. The minimal Track B extension
4. The resilience reward results
5. The paper framing decision

## 10-minute script

### 1. Opening (1 minute)

Suggested wording:

Thank you both. I wanted to use this meeting to show where the benchmark stands now and what I think is the most defensible paper framing.

The short version is that we now have three things:

1. a validated resilience audit,
2. a clear explanation of why the earlier benchmark underperformed, and
3. a new benchmark lane in which RL does work once the agent controls the active bottleneck.

### 2. Reset the framing (1 minute)

Suggested wording:

After the Ding discussion, I think the safest framing is no longer "first RL for SCRES" or "first recursive improvement analysis."

The stronger position is:

we have a validated DES benchmark for operational resilience control in a military food supply chain, and we can now show more precisely when RL helps and when it does not.

### 3. Track A: honest negative result (2 minutes)

Suggested wording:

The first important result is actually negative, but useful.

In the thesis-faithful Track A setting, PPO and RecurrentPPO do not beat the best static benchmark. The reason is not simply the algorithm. The problem is that the action contract does not give the agent enough control over the active downstream bottleneck.

So Track A gives us an honest diagnostic:

- the DES is valid,
- the benchmark is auditable,
- but RL does not help if the agent cannot act on the main operational constraint.

Evidence to cite verbally:

- PPO + control_v1: fill around 0.782 vs S2 around 0.792
- PPO + ReT_seq_v1: fill around 0.788 vs S2 around 0.792
- RecurrentPPO also loses

Transition sentence:

That negative result led to the next question: is RL failing because of the reward or because of the control contract?

### 4. Track B: minimal benchmark extension (2 minutes)

Suggested wording:

To test that diagnosis, I created a minimal new lane, Track B.

Track B does not replace Track A. It is a separate benchmark lane.

What changed is minimal but important:

- the action space goes from 5D to 7D,
- the agent now controls downstream dispatch at Op10 and Op12,
- the observation contract moves to v7 so the bottleneck is more visible,
- and the rest of the DES backbone remains matched and auditable.

The key point is: Track B gives the agent control over the active bottleneck that Track A kept effectively fixed.

### 5. Main result table (2 minutes)

Show:

- `/Users/thom/Desktop/Universidad_Codigo/proyecto_grarrido_scres+ia/outputs/track_b_benchmarks/track_b_all_reward_audit_20260403T155227Z/paper_main_table_manuscript.md`

Suggested wording:

Under this corrected benchmark, PPO now clearly outperforms the static policies.

The main numbers are:

- PPO: fill rate 1.000, order-level Garrido 2017 ReT 0.950, autotomy 96.8%
- RecurrentPPO: essentially the same
- S2: fill 0.958, order-level ReT 0.469
- S3(d=2.0): fill 0.985, order-level ReT 0.449

So the important conclusion is:

the gain is not coming from recurrence alone, because PPO and RecurrentPPO are nearly identical here.
The gain comes mainly from action-space alignment with the operational bottleneck.

### 6. Resilience reward question (1.5 minutes)

Suggested wording:

I also tested whether resilience itself can be used as the training reward.

The answer is yes, but not with the 2017 thesis function in its original online form.

What we see is:

- ReT_thesis and ReT_corrected train poor policies
- ReT_seq_v1 trains an excellent policy
- ReT_garrido2024_train also trains an excellent policy

So the defensible statement is:

the 2017 thesis metric remains valuable as an order-level audit metric,
while the 2024 Garrido resilience family can be operationalized as a training reward for RL.

### 7. Paper framing recommendation (0.5 minutes)

Suggested wording:

My recommendation is:

- Track A stays in the paper as the honest negative benchmark,
- Track B becomes the positive benchmark showing when RL works,
- Garrido 2017 is kept as the thesis-faithful audit metric,
- and the training reward is a resilience-aligned stepwise function, with ReT_seq_v1 as the cleanest current primary candidate and Garrido 2024 as a strong robustness lane.

### 8. Close / decision request (0.5 minutes)

Suggested wording:

What I would like to align today is not every final wording detail, but the overall paper logic:

1. Track A as diagnosis,
2. Track B as corrected benchmark,
3. reward-resilience alignment as the contribution,
4. architecture claims deferred until DKANA is benchmarked against PPO and RecurrentPPO.

## If Garrido asks: "Why not use my resilience function as the main reward?"

Suggested answer:

We can say that your 2024 resilience family is viable as a training reward, because it already produces near-best results.

The only reason I still keep ReT_seq_v1 as the current primary lane is methodological simplicity: it is easier to interpret through SC, BC, and AE, and right now it is slightly stronger numerically.

So I do not see this as a contradiction. I see it as:

- Garrido 2024 proves resilience-as-reward is viable,
- ReT_seq_v1 is the reduced operational variant,
- and both support the same benchmark story.

## If David pushes DKANA

Suggested answer:

I agree that DKANA may still become an architectural contribution.

But at the moment, the strongest empirical result is benchmark-level: PPO and RecurrentPPO already show that action-space alignment matters more than recurrence alone.

So the next fair architectural test is DKANA against both PPO and RecurrentPPO under the same Track B benchmark.

## What not to say

Avoid these phrases:

- "we are the first"
- "DKANA is already superior"
- "Track B fixes Track A"
- "the thesis function was wrong"

Prefer:

- "Track B is a separate benchmark lane"
- "Track A revealed a structural limitation"
- "the thesis metric is valuable as an order-level audit metric"
- "the 2024 resilience family is trainable"

## Files to have open during the meeting

- `/Users/thom/Desktop/Universidad_Codigo/proyecto_grarrido_scres+ia/outputs/track_b_benchmarks/track_b_all_reward_audit_20260403T155227Z/paper_main_table_manuscript.md`
- `/Users/thom/Desktop/Universidad_Codigo/proyecto_grarrido_scres+ia/outputs/track_b_benchmarks/track_b_all_reward_audit_20260403T155227Z/paper_mechanism_table.csv`
- `/Users/thom/Desktop/Universidad_Codigo/proyecto_grarrido_scres+ia/docs/TRACK_B_MINIMAL_SPEC.md`
- `/Users/thom/Desktop/Universidad_Codigo/proyecto_grarrido_scres+ia/outputs/track_b_benchmarks/reward_sweeps/night_20260403T050823Z/ppo/runs/track_b_reward_sweep_ppo_ret_thesis_500k/summary.json`
- `/Users/thom/Desktop/Universidad_Codigo/proyecto_grarrido_scres+ia/outputs/track_b_benchmarks/reward_sweeps/night_20260403T050823Z/ppo/runs/track_b_reward_sweep_ppo_ret_seq_v1_500k/summary.json`
- `/Users/thom/Desktop/Universidad_Codigo/proyecto_grarrido_scres+ia/outputs/track_b_benchmarks/reward_sweeps/night_20260403T050823Z/ppo/runs/track_b_reward_sweep_ppo_ret_garrido2024_train_500k/summary.json`

