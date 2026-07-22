# Q-R1 reconciliation with the Codex comparator review (2026-07-21)

The Codex reconciliation review is **accepted**. Its epistemic corrections are correct and are
applied here. This document records what was reclassified, one factual correction to the review, and
the converged plan. All of my Gate 0–2 evidence is preserved as `EXPLORATORY_NO_CLAIM`; nothing is
deleted.

## What Codex got right (accepted, verified against the code)

1. **My `mpc_star_frozen_v1` is a prototype, not the strongest comparator.** Verified: the module
   integrates the 6 latent states exactly but MC-approximates demand (8 tapes/state); its CRN is
   keyed on `observation_sha256` (policy-dependent, so not common across divergent arms); its
   fail-closed fallback can still return an infeasible action (it flags it, but is not
   guaranteed-safe); it optimizes `ret_visible`, not the full-cohort endpoint. → **Reclassified
   `EXPLORATORY_COMPARATOR_PROTOTYPE`** (contract + module docstring updated).

2. **My `STOP_NO_DEPLOYABLE_NEURAL_PREMIUM_STATIONARY_ENV` was premature.** The sufficient-statistic
   argument proves the retained Bayesian belief is optimal for *inference* of the regime, but NOT
   that an approximate H3 MPC is the optimal *control* policy for the full-DES POMDP. A stronger
   controller (deeper horizon) with the same belief could beat the retained H3 MPC — deployable
   control headroom this gate never measured. → **Reclassified
   `HYPOTHESIS_NO_DEPLOYABLE_PREMIUM_PENDING_COMPARATOR_V2`** (Gate 2 doc updated). The
   parametric-carrier NULL stands (a direct measurement); the regime-carrier conclusion is a
   hypothesis.

3. **My product-coupled risk contract is an incomplete draft, not a preregistration.** Its
   unresolved margin, risk grid, exact roots, and threshold justifications are placeholders. →
   **Reclassified `INCOMPLETE_DRAFT_NOT_EXECUTABLE`**; carries no authority to open seeds; gated
   behind the successor confirmatory + a learner-blind screen of the existing risks + written
   Garrido validation.

4. **Submission A stays independent.** No Q-R1 evidence is merged into it and no "five independent
   confirmations" is claimed; it lives on `codex/submission-a-program-q` (a full manuscript already
   exists there) and proceeds on its own.

## One factual correction to the review

The review states the confirmatory "ya abrió 32 histories prospectivas … Esperar los cuatro shards."
**Verified state: `q_r1_successor_replication_v1` is `FROZEN_PROSPECTIVE_UNOPENED`** — the seed block
is not opened, there are no shard results yet, nothing is running. It is frozen (A/B/C estimands,
`selected_universal_planner = ret_proxy_scenario_h4_p16_stratified`, `primary_metrics =
[early_ret_visible, early_ret_complete_cohort]`, 4 shards over roots 7572001–7572032, gates frozen).
So there is nothing to "wait for" yet; running it is the next prospective step, and it is Codex's
lane (Codex froze it at commit `983c197`).

## Convergence (what the two lanes independently agree on)

- My Gate 0 full-cohort endpoint and Codex's `early_ret_complete_cohort` primary metric are the same
  fix — the metric-population hole is closed the same way in both lanes.
- My C1 A/B/C burned diagnostic and Codex's frozen successor A/B/C estimands are the same design
  (A prefix+natural replan, B sustained control, C splice-diagnostic-never-rescuer).
- My Gate 1 finding (the legacy MC-MPC is noise-dominated) motivates exactly Codex's move to a
  calibrated stratified planner; Codex chose deeper H4 where I stopped at H3, which is precisely the
  control-optimality axis my STOP failed to test.

## What of my Gate work survives (as EXPLORATORY_NO_CLAIM precursors)

- **Gate 0 C1 full-cohort diagnostic** (`+0.02303`, LCB `+0.01933`, service-clean, action-vs-belief
  recorded): a valid burned precursor to the frozen prospective successor. Stands.
- **Gate 1 action-stability certificate** (legacy p16-vs-p64 0/12; exact-state integration 12/12
  first-action): a valid finding that the legacy comparator was noise; motivates comparator v2.
- **Parametric-carrier NULL vs the prototype** (0.000000, 24 pairs): a direct measurement; stands.
- **`ExactJointBelief.enumerate_states`**: a correct, reusable exact-enumeration primitive.

## Converged plan (supersedes my earlier gate sequencing)

1. **Codex's lane:** run the frozen `q_r1_successor_replication_v1` intact (4 shards, roots
   7572001–7572032), verify receipts, adjudicate A/B/C on the frozen gates → one of
   `PASS_REPAIRED_RETAINED_INFORMATION_VALUE` / `STOP_…` / explicit instrument failure. A PASS
   supports retained *structured* value vs the frozen H4/p16 comparator only — it authorizes no net
   and asserts no MPC optimality.
2. **Comparator v2** (Codex's spec): universal, exact/nested-stratified integration, history-keyed
   CRN, full-cohort objective, guaranteed-safe-or-abstain fallback, family H1/H3/H4/H6 (+H8 only
   after DP/caching preflight), convergence-gated (≥95% first-action agreement, mean value error
   <0.005), calibrated learner-blind. Only then re-measure `Retained − Reset` and
   `BestDeployableHistoryPolicy − RetainedStructured`. My prototype is a cross-check, not the answer.
3. **Neural premium** only if comparator v2 leaves a deployable residual: M0–M4 ladder + RecurrentPPO
   ablation, each with independent holdout, ranking-change + placebo + ledger/service gates, final
   `LCB95[RetainedHybrid − BestRetainedStructured] ≥ 0.01`. If v2 absorbs the residual → learner-blind
   sensitivity over existing risks R11/R14/R21/R22/R24 first; product-coupled only if they are
   decision-inert AND Garrido validates the physics.
4. **Submission A** proceeds independently on its own branch; Q-R1 does not gate its ≤8-week target.

## Frozen assumptions carried forward
Publication-first is binding. `+0.02303` is burned evidence until the prospective replication closes.
The parametric null does not close the regime carrier or terminal-value learning. A Bayesian posterior
optimal for inference does not make the MPC optimal for control. No STOP closes more than the
mechanism and comparator actually tested.
