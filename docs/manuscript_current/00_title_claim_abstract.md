# Title, Claim, Contributions, and Abstract

Status: working draft for the current manuscript cycle.

## Working title

**When Does Reinforcement Learning Work for Supply Chain Resilience? Action-Space Alignment in a Military DES Benchmark**

Longer alternative if the journal prefers a more explicit title:

**Action Space Alignment with Operational Constraints Determines Reinforcement Learning Effectiveness for Supply Chain Resilience: A DES Benchmark on a Military Food Supply Chain**

## One-line claim

In this thesis-grounded MFSC benchmark, reinforcement learning does not beat strong static baselines under the original control contract, but it does after a minimal action-space extension exposes the active downstream constraint; therefore, RL effectiveness depends critically on action-space alignment with the operational bottleneck.

## Contributions

**C1. Empirical-negative benchmark finding.** Under the thesis-faithful `Track A` contract, learned policies do not beat the strongest static baseline. In the frozen `ReT_seq_v1` lane, PPO reaches fill rate `0.788` versus `0.792` for `static_s2`, and RecurrentPPO does not recover the gap in the corresponding recurrent lane.

**C2. Empirical-positive benchmark finding.** Under the extended `Track B` contract, PPO reaches fill rate `1.000`, zero backorders, and order-level resilience `0.950`, beating `s2_d1.00` by `+3.40 pp` in fill rate and the best static policy by `+1.23 pp`.

**C3. Structural diagnosis via matched ablation.** A matched `5D vs 7D` ablation with the same observation space, reward, risk profile, and training budget indicates that the added downstream control dimensions are materially responsible for the Track B gain.

**C4. Reproducible benchmark contribution.** The paper contributes a thesis-grounded DES+RL benchmark for military supply chain resilience control, together with auditable artifacts, frozen benchmark contracts, and a resilience-aware reward/reporting pipeline.

## What we do not claim

- We do not claim to be the first paper on RL for supply chains.
- We do not claim to be the first paper on DES+RL for supply chains.
- We do not claim architectural novelty.
- We do not claim that RL always beats static baselines.
- We do not claim that the matched ablation proves universal causal optimality outside the frozen benchmark family.

## Abstract

Supply chain resilience research increasingly uses reinforcement learning and hybrid simulation workflows, but much of the literature still emphasizes inventory cost optimization, abstract network reconfiguration, or generic benchmark environments rather than operational control on validated, domain-specific models. We present a reproducible discrete-event simulation plus reinforcement learning benchmark for resilience control in a 13-operation military food supply chain rebuilt from Garrido-Rios' (2017) thesis and exposed through a Gymnasium-compatible interface. We first report a negative result: under the thesis-faithful control contract, PPO trained with a Cobb--Douglas resilience reward reaches fill rate 0.788 versus 0.792 for the strongest static baseline, and RecurrentPPO does not recover the gap. We then introduce a minimal action-space extension that adds downstream dispatch control at Op10 and Op12 while keeping the DES backbone intact. Under this extended contract, PPO reaches fill rate 1.000, zero backorders, and order-level resilience 0.950, exceeding the baseline by 3.40 percentage points in fill rate and the best static policy by 1.23 percentage points. A matched 5D-versus-7D ablation with identical observation space, reward family, and risk profile further indicates that the added downstream control dimensions are materially responsible for the gain. The contribution is not architectural novelty but a benchmark-centered diagnosis of when reinforcement learning fails and when it works in supply chain resilience control.
