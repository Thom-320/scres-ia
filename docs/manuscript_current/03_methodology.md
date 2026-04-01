# 3. Methodology

Status: draft prose base aligned to the frozen paper contracts.

## 3.1 MFSC discrete-event simulation

The experimental platform is a 13-operation military food supply chain reconstructed in Python/SimPy from Garrido-Rios' thesis. The model preserves the original operational structure rather than replacing it with a stylized benchmark environment. This matters for the paper's positioning: the contribution is not a generic RL sandbox, but a thesis-grounded operational model whose buffers, transport stages, and disruption processes already carry domain meaning.

The DES retains hourly granularity where the original supply chain requires sub-day behavior, especially around the assembly-line operations. This avoids collapsing operational effects that would disappear under daily aggregation, including short breakdowns and shift-sensitive capacity changes. The warm-up regime also follows the validated thesis lineage, with the first full batch reaching the downstream chain after roughly `838.8` simulated hours. The RL layer therefore sits on top of a model that is already operationally constrained before learning begins.

The simulation backbone is not treated as a moving target. Deterministic and stochastic checks remain anchored to the thesis reference tables, and the benchmark story is built on the audited post-alignment DES rather than on earlier historical artifacts. This allows the paper to claim benchmark fidelity without claiming that the model is a universal representation of all supply chains.

## 3.2 Control formulation

The learning problem is defined as weekly operational control over the DES. At each decision epoch, the agent observes a state vector derived from the simulation, selects a continuous action, and receives a resilience-oriented reward. Decisions are taken every `168` simulated hours, which matches the weekly planning cadence already embedded in procurement, disruption, and operating rhythms. For manuscript language, the problem should be described as an augmented observed-state control problem, or equivalently a practical `POMDP`, rather than as a fully sufficient MDP.

The paper uses two control contracts. `Track A` is the thesis-faithful benchmark family. It uses observation version `v1` with `15` dimensions and a `5D` action contract covering upstream quantity and reorder multipliers together with assembly-shift selection. This is the negative paper lane: it asks whether RL can beat strong static policies when control remains limited to the original scope.

`Track B` is a separate research lane rather than a retrofit of `Track A`. It uses observation version `v7` with `46` dimensions and a `7D` action contract. The first five actions keep the Track A controls, while the sixth and seventh actions add downstream dispatch multipliers at `Op10` and `Op12`. The paper's central question is whether this minimal extension changes the benchmark outcome by exposing the operational bottleneck that Track A leaves mostly outside the controllable interface.

## 3.3 Reward and reporting metrics

The main paper lane uses `ReT_seq_v1` with `ret_seq_kappa = 0.20` as the training reward. The reason for centering this reward is not that it is the only reward ever tested in the repository, but that it is the best-aligned resilience-index family that remains smooth enough for policy-gradient learning while still preserving a visible link to the thesis resilience framing. In the current paper story, reward design and control-contract design are discussed jointly: Track A shows that a reasonable resilience-aware reward is not enough when the controllable action scope misses the binding downstream constraint, whereas Track B shows that the same reward can support strong learning once the action contract is repaired.

The paper-facing comparison should therefore prioritize service and resilience outcomes rather than raw reward totals across incompatible families. The primary metrics for the main claim are `fill_rate`, `backorder_rate`, `order_level_ret_mean`, and `terminal_rolling_fill_rate_4w`. Secondary audit metrics such as `ret_thesis_corrected_total`, `ret_unified_total`, and the order-case shares remain useful for interpretation, but they should not replace the main service/resilience metrics when the paper compares policies across benchmark families.

One language rule should stay explicit in the final manuscript: `reward_total` is only comparable within a shared reward family. Whenever the paper compares Track A and Track B, or different reward families inside Track A, the argument should rely on service and resilience metrics rather than on raw cumulative reward.

## 3.4 Experimental protocol

The experimental design contains two benchmark families and one matched ablation. `Track A` is the thesis-faithful benchmark lane. It retains the validated DES backbone, the original operational scope, and strong static baselines as the main comparators. Learned policies are reported honestly as negative evidence when they remain below the strongest static baseline; this includes PPO in the frozen `ReT_seq_v1` lane and RecurrentPPO in the corresponding recurrent comparator lane.

`Track B` is the positive paper lane. Its frozen contract is `env_variant="track_b_adaptive_control"`, `reward_mode="ReT_seq_v1"`, `ret_seq_kappa=0.20`, `observation_version="v7"`, `action_contract="track_b_v1"`, `risk_level="adaptive_benchmark_v2"`, `year_basis="thesis"`, `step_size_hours=168`, and `stochastic_pt=True`. The production benchmark is trained for `500000` timesteps with seeds `11 22 33 44 55` and evaluated over `20` episodes per seed. Static comparators are `s2_d1.00`, `s3_d1.00`, and `s3_d2.00`.

To separate the action-contract argument from broader environment changes, the paper also uses a matched `5D vs 7D` ablation. Within that ablation family, the observation version, reward family, risk profile, and training budget are held fixed while only the action contract changes. This is the cleanest evidence available inside the current repo for the claim that downstream control coverage is materially responsible for the Track B gain. The manuscript should still state the limitation clearly: the ablation is supportive causal evidence inside the frozen benchmark family, not universal proof outside it.
