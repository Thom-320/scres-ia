# Program D DRA-1-v3 — authoritative preregistration

Status: **AUTHORITATIVE AND FROZEN BEFORE THE CSSU-A/B IMPLEMENTATION**.

This document supersedes the endpoint and sequencing language in the historical
DRA-1 preregistration, metric addendum, and v2 adjustment. Those documents
remain provenance only. DRA-1-v3 implements the agreed publication target:

> a PPO policy must beat the best same-contract static policy on Garrido Excel
> ReT out of sample; retained learning is tested only after that result exists.

## 1. Evidence boundary and environment

The environment is a thesis-grounded reconstruction with audited physical
causality and a disclosed attribution proxy, not an exact Simulink replica.
Figure 6.2 documents two CSSUs at Op11, but their separate demand, inventory,
capacity and localized-risk processes are our structural extension.

The R22/R23 preflight freezes the following boundary:

- physical localization and duration: passed;
- R22/R23 order-path liveness: passed;
- R2 aggregate ReT absolute mean relative gap: approximately 31%;
- R22 order-attribution share gap: material and disclosed.

Therefore DRA-1 may test relative same-tape policy effects under a physically
causal localized-threat extension. It may not claim exact reproduction of the
CF11-CF20 distributions.

## 2. Physical contract

- Opt-in topology `cssu_split_v1`; all historical lanes retain the aggregate
  CSSU bitwise default.
- Separate CSSU-A/B inventories, backlogs, inbound/outbound in-transit ledgers,
  lane state and destination-tagged orders.
- Aggregate demand is conserved exactly; symmetric expected destination share
  in the base lane. R24 can create temporary asymmetric mission demand.
- Total daily downstream capacity is fixed across actions:

  \[
  q_{A,t}+q_{B,t}\leq\min\{C_t,I_{SB,t}\}.
  \]

- Primary downstream quantity source: Figure 6.2/text, 2,400-2,600. Table 6.20,
  2,000-2,500, is a named sensitivity and is never averaged with the primary.
- R22 strikes one explicit lane; R23 strikes one CSSU. Total event intensity is
  held constant across policies. Shuffled localization is a placebo arm.
- No action creates inventory, transport, demand or risk; no eviction action is
  available. Residual partial orders remain in the ledger.

## 3. Actions, observations and timing

Discovery actions are enumerable:

- allocation to A: 0.25, 0.50 or 0.75; B receives the complement;
- service rule: `SPT_FULL`, `FIFO_PARTIAL`, or `R24_AGE_PARTIAL`.

Decision epoch: 24 h immediately before dispatch; one-day activation latency
and minimum duration. PPO later receives the identical nine-action contract.

Allowed observations: SB and CSSU A/B inventories; days of cover; backlog
quantity/count/age/R24 share by destination; in-transit quantities and ETA;
current lane/CSSU availability; recent 1/7-day demand, delivery and fill;
previous action; operational-day phase. Future shocks, repairs, demand, latent
regime and retrospective outcomes are prohibited.

## 4. Endpoint hierarchy

Primary: `ret_excel_visible_v1`.

Mandatory sensitivity: `ret_excel_visible_clipped_0_1`.

Binding no-artifact guardrails:

- lost-order relative degradation CI95 upper at most 2%;
- service-loss AUC relative degradation CI95 upper at most 2%;
- backlog-AUC relative degradation CI95 upper at most 2%;
- worst-CSSU ReT no lower by more than 0.01 unless mission priority was frozen
  prospectively;
- mass and total transport capacity conserved;
- no benefit from disappearing demand, privileged information, greater total
  transport or a short-horizon gain that becomes 28-day damage.

ReT is an evaluation outcome, not the PPO reward. The PPO reward is a dense,
calibration-frozen operational loss; reward design may not use final tapes.

## 5. Data universes

1. Discovery/calibration: 60 tapes, balanced across nominal, localized R22,
   localized R23 and localized R24/mixed threats.
2. Observable-policy holdout: 40 distinct tapes. These are not reused for PPO
   selection or its final claim.
3. PPO training/probes: separate materialized tapes and fixed no-gradient probes.
4. PPO confirmatory: 60 completely virgin tapes, opened once after policy,
   hyperparameters, comparator and analysis are frozen.

Every split has true-localization and shuffled-localization counterparts with
identical total threat intensity.

## 6. Pre-RL discovery gates

### Gate A — liveness and invariance

Require exact aggregate-demand and mass conservation; targeted A/B degradation;
allocation moves only existing stock; localized risks change only the declared
lane/node; aggregate topology remains bitwise identical when split mode is off.

### Gate B — same-contract static frontier

Enumerate all nine constant actions under paired CRN. Select the best admissible
constant using calibration ReT subject to the binding guardrails. Freeze it
before branching and all holdouts.

### Gate C — exact branching

Replay identical prefixes; branch all nine actions for one epoch; use a common
continuation; evaluate 72 h and 28 d. Require:

- at least two allocation levels branch-optimal in at least 15% of states each;
- no action above 85%;
- oracle-minus-best-static ReT CI95 lower above zero;
- clipped result co-directional;
- no guardrail breach;
- benefit at 72 h does not reverse at 28 d.

### Gate D — observable signal, tree and heuristic

Fit a depth-3 tree and a prespecified inventory/backlog-imbalance heuristic using
five folds grouped by tape. Execute both sequentially on held-out tapes.

The tree and heuristic remain mandatory baselines. PPO may proceed when Gates
A-C pass and deployable observations show positive out-of-fold branch-value
capture. The shallow tree need not itself beat the static policy if the action
surface is demonstrably state-contingent and observable but nonlinear. If all
observable models capture zero or negative oracle value, emit
`STOP_NO_OBSERVABLE_SPATIAL_HEADROOM` and do not train PPO.

## 7. PPO versus best static

Initial algorithm: categorical PPO with MLP 2x64 tanh. A bounded pilot freezes
the reward weights, learning budget and one hyperparameter configuration using
training/probe tapes only. No KAN, RNN, SAC, TD3 or MARL in the primary study.

Primary confirmatory contrast:

\[
\Delta_{RL}=E[ReT(\pi_{PPO})-ReT(a^*_{static})].
\]

Promote `ADAPTIVE_CONTROL_SUPPORTED` only if all hold on the 60 virgin tapes:

- two-way seed-by-tape CI95 lower for ReT is above zero;
- clipped ReT CI95 lower is above zero;
- at least 8/10 learner seeds and 70% of tapes are positive;
- true-localization result exceeds shuffled-localization placebo;
- all binding guardrails pass;
- PPO is compared with the best static, thesis policy, heuristic and tree under
  the identical action/resource contract.

If PPO fails, retain DRA-1 as a decision-right boundary result. No architecture
sweep is authorized to rescue it.

## 8. Retained learning and path dependency

Only after `ADAPTIVE_CONTROL_SUPPORTED`:

- frozen PPO versus static identifies learned adaptive policy value;
- persistent weights versus reset identifies retained learning;
- persistent full versus weights-only is an optimizer-state sensitivity;
- different orders of the same campaign multiset identify path dependency.

Promote retained learning only if persistent-minus-reset ReT CI95 is above zero
on common no-gradient probes with matched compute and identical physical reset.
Promote path dependency only if history order creates reproducible differences
on identical final probes. Otherwise remove the corresponding claim.

## 9. Claim ladder

| Highest passed stage | Allowed claim |
|---|---|
| Physical only | Spatial split is live; no adaptive value claim |
| Oracle only | Latent headroom; not deployable |
| Observable tree/heuristic | Interpretable adaptive decision aid |
| PPO > best static | Neural adaptive control improves simulated ReT |
| Persistent > reset | Retained policy learning adds resilience value |
| History order differs | Simulated resilience is history-dependent |

No stage supports predictive accuracy or real-world organizational learning.
