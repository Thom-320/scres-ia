**Control-Reward 500k Source of Truth**

This note freezes the current paper-facing interpretation of the long-run control-reward benchmark. These statements should be treated as the source of truth for meeting materials and manuscript revisions unless a new benchmark family is explicitly introduced later.

**Scenarios locked for the current paper revision**

- `increased + stochastic_pt`
- `severe + stochastic_pt`

Both runs use:

- `w_bo=4.0`
- `w_cost=0.02`
- `w_disr=0.0`
- `500,000` PPO timesteps
- `5` seeds

**Locked interpretation**

- Under `increased + stochastic_pt`, PPO matches the service level of the best fixed baseline (`static_s2`) but does not improve control reward.
- Under `severe + stochastic_pt`, PPO outperforms the best fixed baseline (`static_s3`) in control reward while maintaining effectively equivalent service.
- `ReT_thesis` remains useful as a reporting metric, but it is not the training objective for this benchmark lane and should not be used as the main discriminator of adaptive-control value under the severe regime.

**Do not revise the story this way**

- Do not say PPO wins uniformly across all stress regimes.
- Do not say the current `severe` profile is a fully realistic military campaign model.
- Do not move the benchmark goalposts by redefining `severe` inside the same manuscript iteration.

**Preferred language**

> We escalated risk not to force a positive result, but to evaluate resilience in the regime where fixed policies become insufficient. Under moderate stress, PPO remains competitive with the best fixed baseline. Under severe stress, adaptive switching yields a reward advantage while maintaining comparable service.

**Primary auditable sources**

- `/Users/thom/Desktop/Universidad_Codigo/proyecto_grarrido_scres+ia/docs/artifacts/control_reward/control_reward_500k_increased_stopt`
- `/Users/thom/Desktop/Universidad_Codigo/proyecto_grarrido_scres+ia/docs/artifacts/control_reward/control_reward_500k_severe_stopt`
