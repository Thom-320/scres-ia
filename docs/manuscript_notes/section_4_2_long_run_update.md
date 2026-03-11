**Section 4.2 update for the 500k stochastic-PT runs**

Use the following paragraph logic in the revised hybrid results section:

1. Static shift-control baselines establish that `S1` is clearly suboptimal and that `S2/S3` define the high-service region under the operational reward.
2. The purpose of risk escalation is not to force a PPO win, but to test whether adaptive control becomes valuable when fixed policies are no longer sufficient.
3. Under `increased + stochastic_pt`, PPO is competitive with `static_s2`: service is effectively matched, but control reward is slightly lower.
4. Under `severe + stochastic_pt`, PPO outperforms the best fixed baseline (`static_s3`) in control reward while maintaining comparable service.
5. Therefore, adaptive-control value emerges under strong stress rather than uniformly across all operational regimes.

**Suggested replacement paragraph for Section 4.2.5**

> Long-run stochastic-processing-time benchmarks clarified the stress dependence of the adaptive-control signal. Under `increased` risk, PPO matched the service level of the best fixed baseline (`static_s2`) but did not improve control reward, indicating competitiveness rather than superiority in a moderate-stress regime. Under `severe` risk, however, PPO outperformed the best fixed baseline (`static_s3`) in control reward while maintaining effectively equivalent service. The resulting interpretation is not that adaptive RL dominates fixed policies uniformly, but rather that its value emerges under sufficiently strong stress, which is precisely the regime in which resilience-oriented adaptation matters most.

**Suggested reward-discussion sentence**

> In this benchmark lane, `ReT_thesis` was retained as a reporting metric, while `control_v1` was used as the training objective because direct control learning required an explicit operational signal based on service loss and cost.

**Suggested limitation sentence**

> These results should still be treated as preliminary and stress-regime specific: the current evidence does not justify a claim of universal superiority across all risk conditions, nor a formal claim of exact Markov sufficiency for the exposed observation.
