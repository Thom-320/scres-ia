**Control-Reward 500k Source of Truth**

This note freezes the current repo-facing interpretation of the long-run `control_v1` benchmark lane after the March 2026 DES audit/alignment fixes. These statements should be treated as the source of truth for the `control_v1` comparator lane unless a new matched benchmark family is explicitly introduced later.

**Valid scenarios currently available**

- `increased + stochastic_pt` via `outputs/paper_benchmarks/paper_control_v1_500k`

The current auditable `control_v1` comparator uses:

- `w_bo=4.0`
- `w_cost=0.02`
- `w_disr=0.0`
- `500,000` PPO timesteps
- `5` seeds
- `observation_version=v1`
- `year_basis="thesis"`

**Locked interpretation**

- Under `increased + stochastic_pt`, `control_v1` PPO is competitive but trails `static_s2` on the paper-facing service metrics in the current auditable bundle.
- The old `severe + stochastic_pt` `control_v1` bundles are historical artifacts and should not be used as primary evidence for the current repository state.
- A valid `control_v1 + v4 + RecurrentPPO` long run is still pending completion under the current codebase.
- `ReT_thesis` remains useful as a reporting metric, but it is not the training objective for this benchmark lane and should not be used as the main discriminator of adaptive-control value under the severe regime.

**Do not revise the story this way**

- Do not say PPO wins uniformly across all stress regimes.
- Do not say the current `severe` profile is a fully realistic military campaign model.
- Do not move the benchmark goalposts by redefining `severe` inside the same manuscript iteration.

**Preferred language**

> In the post-audit codebase, `control_v1` remains the operational comparator lane rather than the leading paper-facing reward. The current auditable `increased` run shows competitive but not dominant behavior relative to `static_s2`, and any `severe` claim must wait for a matched rerun under the current DES.

**Primary auditable sources**

- `/Users/thom/Desktop/Universidad_Codigo/proyecto_grarrido_scres+ia/outputs/paper_benchmarks/paper_control_v1_500k`

**Historical-only sources**

The old `control_reward_500k_*_stopt` bundles remain legacy context only and must not be cited as the primary evidence for the current repo state.
