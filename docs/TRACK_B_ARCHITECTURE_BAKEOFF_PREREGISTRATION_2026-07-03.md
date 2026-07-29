# Track B Architecture Bakeoff Preregistration -- 2026-07-03

## Purpose

Garrido's architecture concern is legitimate: if the manuscript reads as
"PPO+MLP applied to a supply chain", the novelty is weak. The paper's defensible
claim is instead that the Garrido-grounded DES reveals when learned control adds
SCRES value: the action surface must reach the downstream recovery bottleneck.

This preregistration fixes the architecture comparison before additional
architecture runs are promoted. The goal is not to defend PPO+MLP by assertion.
The goal is to test whether richer architectures materially improve the same
Track B decision problem under the same evaluation bar.

## Scope

The architecture bakeoff is a sidecar to Paper 1. It can change the architecture
reported in the manuscript only if a richer architecture beats canonical PPO+MLP
under the same protocol and primary metric. Otherwise, the manuscript keeps the
PPO+MLP headline and reports architecture alternatives as robustness evidence or
future work.

## Canonical Evaluation Contract

All architecture claims must use the same Track B recovery-control contract:

- Environment: `make_track_b_env`
- Risk regime: `adaptive_benchmark_v2`
- Observation: `v7`
- Action contract: `track_b_v1`
- Reward for training: `control_v1`
- Horizon: `max_steps=104`
- Evaluation episodes: `12` per seed
- Primary metric: Garrido Excel/order-level ReT (`order_level_ret_mean`)
- Secondary metrics: flow fill rate, backlog/tail metrics, shift-utilization
  cost index, dispatch-inclusive cost sensitivity where available
- Comparator: the same dense downstream-dispatch static family and heuristics
  used in the Track B Q1 bundle
- CRN rule: every architecture must share identical `(seed, episode, eval_seed)`
  keys within a comparison table

The primary architecture bakeoff uses the full 8D Track B action contract,
because that is the manuscript's headline setting. The stress variant for
Garrido's "CDC onward" question is `post_cdc_only`: Op3/CDC authority is
frozen, and the learned controller acts only through post-CDC/downstream
recovery levers plus shift authority. A result that survives `post_cdc_only`
cannot be dismissed as a CDC/top-up artifact, but it does not replace the full
8D architecture comparison.

## Architectures

### A0. PPO+MLP

Canonical learner and current paper baseline. It is deliberately conservative:
a small MLP minimizes architectural novelty so that the experiment isolates the
action-surface question.

Promotion status before this bakeoff:

- Track B headline: positive at 10 seeds.
- Track A: negative under repeated clean attempts.
- Interpretation: the strongest verified architecture today.

### A1. PPO+DMLPA

David's DMLPA is a Transformer-over-history feature extractor for PPO. In the
repo it is implemented as `scripts/dmlpa_extractor.py` and matches the pasted
architecture: stacked observations are split into a temporal sequence, projected
with `Linear -> GELU -> Linear`, enriched with sinusoidal positional encoding,
passed through a multi-layer `TransformerEncoder`, and pooled on the last token.

Hypothesis:

- If hidden temporal structure matters, DMLPA should outperform PPO+MLP,
  especially under masked/partially observable or severe-stress settings.

Risk:

- A DMLPA win is only meaningful if it remains adaptive and does not collapse to
  a near-constant static policy.

Required reporting:

- Frame-stack factor, feature dimension, heads/layers, and trainable parameter
  count.
- Action variability diagnostics, not only mean ReT.

### A2. PPO+KAN-Style Feature Extractor

The existing sidecar in `scripts/kan_extractor.py` is not an official pyKAN
model. It is a dependency-free RBF/KAN-inspired feature extractor with a linear
skip path. It can test whether a KAN-like nonlinear basis preserves the Track B
result, but it must not be sold as a pure Kolmogorov-Arnold Network.

Existing evidence:

- `docs/KAN_SIDECAR_SMOKE_2026-07-02.md` reports a positive five-seed sidecar.
- Safe label: "KAN-inspired RBF feature extractor", not "KAN".

Promotion condition:

- It can become an appendix robustness result unless reimplemented as official
  pyKAN or materially beats PPO+MLP under the same CRN protocol.

### A3. Official pyKAN Surrogate

The official pyKAN demo in `docs/KAN_REAL_DEMO_2026-07-02.md` fits the static
decision-to-ReT surface extremely well. This is valuable for Garrido because it
implements the decision-variables-to-SCRES map he sketched, but it is supervised
regression, not an online control policy.

Use:

- Demonstration artifact and interpretability figure.
- Not a substitute for PPO unless wrapped into a true policy/value network and
  trained/evaluated online.

### A4. DKANA

DKANA, as implemented in this repo, is not simply DMLPA and not a standard
external architecture name. It is a project-specific structured temporal policy
pipeline:

- `build_mfsc_relational_state`: encodes observations and state constraints as
  row triplets `[variable_id, relation_id, value]`.
- Relation mode can be equality or temporal delta (`=`, `<`, `>`).
- `build_dkana_windows`: builds causal history windows.
- `DKANAPolicy`: row encoder, config encoder, local self-attention over rows,
  global self-attention over time, and Gaussian action decoder.
- `DKANAOnlinePolicyAdapter`: keeps an online rolling context window for
  evaluation.

Important boundary:

- Current DKANA training is behavior cloning via
  `scripts/train_dkana_behavior_clone.py`, not PPO/RL. Therefore DKANA can be
  evaluated fairly as an offline learned policy, but it should not be described
  as "PPO with a different feature extractor" unless an RL wrapper is actually
  implemented.

Promotion condition:

- DKANA may replace or complement PPO+MLP only if it beats PPO+MLP on the same
  Track B CRN evaluation and does not rely on a weaker comparator or a different
  metric.

## Decision Rules

An architecture may be promoted into the manuscript spine only if all are true:

1. Mean Excel/order-level ReT exceeds canonical PPO+MLP on the same protocol.
2. At least 4/5 seeds are positive in a five-seed screen; for headline
   replacement, use a 10-seed confirmation or a seed-clustered CI95 whose lower
   bound is positive.
3. The win does not come from changing the action contract, reward, horizon,
   risk regime, or comparator family.
4. Service/tail metrics do not deteriorate enough to undermine the SCRES claim.
5. The policy remains meaningfully adaptive; action-collapse diagnostics are
   disclosed.

If no richer architecture clears this bar, the paper's architecture
justification is:

> PPO+MLP is used deliberately as the conservative learner. Richer temporal or
> KAN-style architectures did not materially improve the same Track B recovery
> task, so the manuscript's mechanism is not architectural novelty; it is
> bottleneck-aligned action authority in a Garrido-grounded DES.

## Recommended Run Order

1. PPO+DMLPA five-seed Track B screen on the full 8D contract, using the same
   seeds, horizon, reward, and CRN protocol as the KAN-inspired sidecar.
2. If needed, a matched five-seed table with PPO+MLP, KAN-inspired RBF, and
   DMLPA side by side. Existing MLP/KAN outputs may be reused only if their
   `(seed, episode, eval_seed)` keys and comparator family match exactly.
3. `post_cdc_only` 10-seed confirmation for PPO+MLP, because this directly
   answers the CDC challenge but is not the primary architecture bakeoff.
4. KAN-inspired sidecar 10-seed confirmation only if Garrido needs a stronger
   KAN robustness artifact after the five-seed architecture screen.
5. DKANA behavior-cloning or RL evaluation after a frozen dataset/export and
   training protocol is declared. Do not compare DKANA to PPO unless both are
   evaluated on identical eval seeds and the DKANA training source is stated.

## Manuscript Language If Results Stay As They Are

Use:

> The architecture comparison supports a conservative interpretation: replacing
> the default MLP with recurrent, transformer-over-history, or KAN-inspired
> feature extraction did not overturn the Track A null nor materially exceed the
> Track B PPO+MLP result. The contribution is therefore diagnostic rather than
> architectural: a learned controller improves SCRES when its action surface
> reaches the downstream recovery bottleneck.

Avoid:

- "KAN proves the result."
- "DKANA is superior."
- "The agent anticipates disruptions."
- "PPO+MLP is best" unless a same-protocol architecture bakeoff shows it.
