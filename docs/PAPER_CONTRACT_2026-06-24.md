# Paper Contract 2026-06-24

> **PARTIALLY SUPERSEDED (2026-06-26):** the **reward** clause (frozen `ReT_cd`, §"Reward,
> Observation, and Outcomes") and the implied **algorithm** are superseded by
> `docs/EXPERIMENT_CONTRACT_V2_2026-06-26.md` (primary reward `control_v1`, DQN, ReT as outcome).
> The research position, fidelity gates, observation spec, and resource/cost rules below STILL HOLD.

## Research Position

This paper uses the Garrido-Rios military food supply chain as a
thesis-anchored experimental platform. Exact thesis reproduction is no longer
the paper claim. The thesis supplies the DES structure, parameter targets,
resilience metric, and interpretable decision levers; the learning experiment is
a justified extension.

The reference DES should preserve the interpretation that best reproduces the
thesis numbers, especially Table 6.10 production and Table 6.11 risk-frequency
targets. When multiple thesis interpretations remain plausible, choose one
primary interpretation before training, document why it best matches the thesis
or operational realism, and keep the alternatives as named sensitivity lanes.

## Primary Hypothesis

We test whether retained learning state improves future resilience:

```text
H1: E[ReT_retained - ReT_reset] > 0
```

Dose-response (strongest evidence against a "single lucky win"):

```text
H2: d/d_rho  E[ReT_retained - ReT_reset] > 0
```

where `rho` is the disruption-persistence parameter of the frozen learning
regime. A monotone gradient across `rho` is more convincing than any single
positive contrast.

The null result is admissible. After the learning-extension contract is frozen,
no reward, observation, algorithm, resource constraint, risk process, or
scenario generator may be changed using test-set results. If the retained-minus-
reset contrast is null, the manuscript reports a null or control-only result
instead of changing the environment until a positive result appears.

## Experimental Lanes

- `des_reference_v1`: stationary thesis-anchored DES reference. It uses the
  implementation choices that best reproduce Garrido-Rios tables and metrics.
- `learning_extension_v1`: retained/reset learning protocol on the
  thesis-derived `6 x 3` inventory-shift action surface.
- `stress_extension_v1`: preregistered robustness regime with stronger or more
  heterogeneous disruptions, stochastic processing times, or other realistic
  thesis extensions when they are operationally justified.

Non-stationarity and stronger risks are moderators of learning value, not proof
requirements. They may be used when justified by operational realism and chosen
from training/calibration evidence only. They must be evaluated separately from
the stationary reference regime.

Track B downstream-control experiments are outside the primary learning paper
and require a separate contract.

## DES Reference Contract

- Year basis: thesis, 8,064 hours/year.
- Horizon: 161,280 hours for thesis reference audits; shorter horizons are
  smoke tests only.
- Warm-up trigger: `op9_arrival`.
- Downstream quantity source: `figure_6_2` for the primary reference lane;
  `table_6_20` remains a named sensitivity lane.
- R14 defect mode: `thesis_strict_op6`.
- Risk occurrence mode: `thesis_window` for the primary reference lane because
  it best reproduces Table 6.11. `legacy_renewal` is a historical negative
  control and sensitivity lane.
- R11 topology: one incident clock; each incident affects one selected
  workstation among Op5 and Op6. The simultaneous Op5+Op6 interpretation is a
  sensitivity lane, not the primary reference interpretation.
- Raw-material flow: use the implementation that best matches Table 6.10,
  Table 6.1, and strategic-buffer behavior. Current implementation:
  `kit_equivalent_order_up_to`, canonicalized internally to
  `bom_total_units_order_up_to`, multiplier `2.0`.

The raw-material multiplier is an implementation calibration, not thesis text.
It must be reported as such unless original model evidence later confirms it.

## Learning Extension Contract

- Primary action surface: `MultiDiscrete([6, 3])`, with a `Discrete(18)` DQN
  view over the same choices.
- Inventory levels: `I0`, `I168`, `I336`, `I504`, `I672`, `I1344`, applied as a
  common strategic-buffer level to Op3, Op5, and Op9.
- Shift levels: `S1`, `S2`, `S3`.
- Default decision cadence: `block`. The policy chooses one of the 18
  configurations at the start of a disruption block and holds it for the block.
  Weekly reselection is a sensitivity lane, not the primary learning claim.
- Online update timing (CORRECTED 2026-06-24 after audit --- TRANSFER estimand): the
  previous "symmetric adapt-then-evaluate" design was WRONG. It let both arms re-learn
  the current block before evaluation, measuring the asymptotic point where the
  reset/no-memory arm catches up, where the effect is ~0 by construction. The correct
  estimand is the COLD head-start: at the start of block `k`, evaluate the retained
  learner (theta accumulated from blocks 0..k-1) and the frozen baseline (theta_0)
  on block `k` BEFORE any block-`k` training. `ΔR_k = R_retained - R_frozen` is the
  transfer that accumulated learning gives on an UNSEEN shock (H1/H4); its growth with
  `k` is the learning curve (H2). The retained learner then trains incrementally on
  block `k` and carries theta forward. Budget is expressed in BLOCKS (weekly cadence,
  no block-collapsing wrapper), not in wrapper timesteps. (frozen = theta_0 evaluated
  cold on an unseen block means "more training" cannot trivially explain a positive
  `ΔR` --- it must generalise.)
- Outcome is the TREATMENT-WINDOW order-level ReT (`supply_chain.clean_metrics`):
  orders placed before the policy acts (`OPTj < warmup_time`) are excluded. The
  warm-up backlog under (I0,S1) shifts ReT by ~0.08, an order of magnitude above the
  retained-learning signal; the previous all-orders metric was contaminated.
- Implementation: `scripts/retention_transfer.py`. Supersedes the adapt-then-eval
  pilots, which are now historical.

This design keeps Track A close to Garrido's static-configuration semantics
while testing whether retained policy state, `L_{k-1}`, changes future
resilience.

## Frozen Learning Regime (decided 2026-06-24)

`learning_extension_v1` is fixed to TWO coupled, individually-justified realism
elements. Selection criterion on the record is operational realism, never "RL
wins":

1. **Persistent disruption regime** — disruption intensity follows a
   Markov-switching / campaign-phase process with persistence `rho` (e.g.
   buildup / high-intensity / lull phases, clustered attacks). Justification:
   military theaters operate in phases; recent disruption history is
   probabilistically predictive of near-future intensity. This is the element
   that makes retained state `L_{k-1}` valuable and supplies the H2 gradient.
2. **Stochastic ration demand** — demand driven by operational tempo rather than
   a fixed daily uniform draw. Justification: food demand tracks force size and
   operations tempo. This makes a single static configuration suboptimal, so the
   frozen-PPO-vs-static contrast separates neural-control value from retention
   value.

Anti-fishing protocol (decided: pre-register + pilot on training tapes only):
write the realism justification and regime parameters first; pilot ONLY on
training scenario tapes to confirm there is learnable structure (not
dead-on-arrival); then freeze. The held-out retained-vs-reset contrast is run
once. The regime is never iterated against held-out results.

The stationary thesis reference (`des_reference_v1`) remains the primary fidelity
regime and is evaluated separately; an out-of-distribution severity tape is the
stress regime. Persistence and stochastic demand are moderators of learning
value, reported as such.

## Other Extensions Allowed Before Freeze

Beyond the decided regime above, further extensions are allowed only when they
are realistic, thesis-anchored, and predeclared before held-out evaluation.
Candidate extensions:

- stochastic processing times, preferably mean-preserving;
- stronger `increased` or `severe` risk regimes when reported as stress tests;
- piecewise-stationary or regime-persistent risk tapes if operationally
  justified;
- stochastic demand beyond the existing daily uniform demand only if the rule is
  specified before evaluation;
- finite physical resource limits for surge, such as shift-hour budgets, if
  parameterized defensibly.

Do not claim monetary cost savings unless defensible cost parameters are added.
Report physical resource use: inventory targets, shift-hours, surge-hours, and
unattended orders.

## Reward, Observation, and Outcomes

Reward and observation must be frozen before confirmatory training. Reward
selection may use reward-surface audits and small smoke runs on training seeds
only.

**Frozen training reward (decided 2026-06-24): `ReT_cd`.** The 2026-06-24
downstream-Q reward-surface audit found that no reward keeps a stable *rank* across
the Figure 6.2 and Table 6.20 interpretations. `ReT_cd` is selected because it is
the only candidate whose *best static policy* is identical under both sources
(`L1a_uniform_I336_S3`); its rank moves (7 -> 1), which we treat as a selection
meta-artifact, not an experiment property. Rationale: the paper's estimand is the
within-architecture `retained - reset` contrast, not the static-policy ranking the
rank-stability rule was guarding. Therefore:

- `ReT_cd` is frozen under the Figure 6.2 paper-facing downstream-Q source.
- The `retained - reset` contrast is reported under **both** Figure 6.2 and
  Table 6.20 as a robustness panel — robustness of the *estimand* is tested, not
  gated. The learning claim is only made for downstream-Q conditions where it holds.
- The Table 6.20 reward-*ranking* instability is reported as a stated limitation.
- Audit evidence: `docs/manuscript_notes/track_a_reward_gate_2026-06-24.md`.

Recommended primary observation for `learning_extension_v1`:

- observable inventories at Op3, Op5, and Op9;
- backorders and unattended/lost orders;
- recent demand;
- active disruption indicators and current downtime;
- current shift level;
- previous action;
- time within block;
- remaining physical resource budget if a budgeted extension is active.

Excluded from the primary observation: future repair duration, future risk
labels, hidden regime labels, Track B downstream controls, and large exploratory
observation versions.

Primary outcome: order-level Garrido ReT. Secondary outcomes: service-loss area,
fill rate, time to recovery where available, pending backlog, lost/unattended
orders, shift-hours, inventory use, and CVaR or 95th percentile service loss.

## Policy Conditions

- Robust static policy: best of the 18 thesis-derived configurations selected on
  training scenarios only.
- Threshold heuristic: state-dependent non-neural comparator.
- Frozen neural policy: common initial trained checkpoint, no evaluation-time
  updates.
- Retained online learner: keeps policy state across blocks.
- Reset-learning learner: same learner and update budget, restored to the common
  checkpoint after each block.

Per-scenario oracle static policies may be reported only as upper-bound
diagnostics, never as the operational baseline.

## Gates and Stop Rules

- G1: DES reference gates pass or are documented as unresolved thesis
  ambiguities: Table 6.10, Table 6.11, risk tables, decision tables, operations,
  raw-material semantics, and order-level ReT schema.
- G2: Reward-surface audit runs on common seeds under `current`, `increased`,
  and `severe`, with `figure_6_2` primary and `table_6_20` sensitivity. Reward
  eligibility = Figure-6.2-shortlisted AND best-static-policy stable across
  downstream-Q (the rank-position rule is retired as too strict — it guarded
  static-policy ranking, not the `retained - reset` estimand). PASS: `ReT_cd`.
  Downstream-Q robustness of the estimand is tested in G4, not gated here.
- G3: Learning-extension contract freezes reward, observation, action cadence,
  update timing, checkpoint reset behavior, train/evaluation seeds, and allowed
  extensions.
- G4: Confirmatory retained-vs-reset evaluation uses held-out common scenario
  tapes or common random seeds, and is reported under BOTH `figure_6_2` and
  `table_6_20` downstream-Q sources (estimand robustness panel).

No outcome direction is required. No held-out evaluation result can be used to
retune the environment and rerun the same claim.

## Allowed Manuscript Artifacts

- `docs/environment_spec.md`
- this contract
- Table 6.10 and Table 6.11 gate outputs generated after this contract
- downstream-Q reward-surface comparison outputs generated after this contract
- retained/reset evaluation outputs generated after this contract

Pre-contract PPO, Track B, continuous-action, and legacy-renewal artifacts are
historical context, not paper evidence for endogenous retained learning.
