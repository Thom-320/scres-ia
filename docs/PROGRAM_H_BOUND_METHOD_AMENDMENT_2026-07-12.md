# Program H — observable-value method: REVIEW + frozen amendment (2026-07-12)

Frozen BEFORE seeds 1060001+ open, satisfying `program_h_belief_audit_v1.json ->
observable_upper_bound.optional_secondary_method` ("only if its algorithm and penalty are frozen in a
hashed amendment before 1060001 is opened"). Compute the SHA of this file and record it before opening
tapes.

## Review of the edited contract (endorsed, with one fix)
- **ENDORSED — augmented belief state.** `(Z_A, dwell_age_A, Z_B, dwell_age_B)` is correct: a
  semi-Markov tempo is NOT Markov in the 9 tempo labels alone; the filter needs dwell age / residual
  phase. Verified the augmented space is tiny: 3 tempo × 8 dwell-ages = **24 per CSSU, 576 joint**.
- **ENDORSED — O1 blocked-by-default** until a signed Garrido domain record; O0 (Program-G-exact) is
  the primary contract and does not depend on O1.
- **FLAGGED (the substantive gap this amendment fixes).** The contract's stated *primary* rigorous
  upper bound — the "exact full-tape 81-sequence information-relaxation ceiling" — is just the
  perfect-information value `J_PI`. We ALREADY know `J_PI − J_static ≈ +0.018 > 0` (Program G). A bound
  that is known to exceed `J_static` **cannot produce Case A** (`UB(J*_obs) − J_ABAB < δ_min`). It is a
  trivial sanity ceiling, not the central deliverable. QMDP was (correctly) demoted to diagnostic —
  QMDP's upper-bound property is not free for the episode-aggregate `ret_order` objective. So neither
  currently-named method can settle policy-class-vs-information. This amendment pins a valid, tight one.

## Frozen method (primary): EXACT observable optimum, not a bound
Because the horizon is 4 weekly decisions, actions are 3, and the augmented latent space is 576, the
best NON-ANTICIPATIVE (observable) value `J*_obs` is computed DIRECTLY by finite-horizon backward
induction over the belief-MDP — no bound needed:

1. Belief `b_t` over the 576 augmented states; deterministic Bayes update from the frozen augmented
   semi-Markov transition + the O0 observations (imperfect signal + observable physical trajectory).
2. Observation integration: the signal is discrete; realized demand enters only through its tempo
   likelihood `P(demand | Z)`. Discretize the per-week observation into a frozen finite set (signal ×
   demand-bin), so the reachable belief tree over 4 steps is finite (≤ |O|⁴ leaves). Backward-induct
   `V_t(b) = max_a E_{o}[ r_t + V_{t+1}(b') ]` on `ret_order` per-order rewards.
3. `J*_obs` = value at the root belief, averaged over the tape distribution under CRN. This is the
   EXACT optimal observable value on the frozen discretized-observation POMDP (an ε-approximation of
   the continuous-demand POMDP; ε reported from the demand-bin granularity by a refinement check).

**Decision:** if `J*_obs − J_ABAB < δ_min` (=0.01 ret_order) → **Case A, information-limited** (no
observable policy materially beats ABAB — an EXACT statement, stronger than a bound). If
`J*_obs − J_ABAB ≥ δ_min` with LCB95>0 → **Case B, policy-class-limited**; the learner gate opens.

## Frozen method (rigorous fallback): information-relaxation dual bound
If the discretized-observation ε cannot be driven below `δ_min/2` by refinement, certify with a
Brown–Smith–Sun information-relaxation dual: relax to perfect information, subtract a dual penalty
`z_t` that is a martingale difference w.r.t. the observable filtration, frozen before 1060001. The dual
value is a rigorous upper bound `≥ J*_obs`; if it is `< J_static + δ_min`, Case A holds rigorously.
QMDP is reported alongside as a diagnostic only (not a certified bound).

## Discipline
Method, discretization grid, penalty form, δ_min, tapes and horizon are frozen here BEFORE 1060001.
Nothing is selected by learner performance. The exact solve and the dual bound use the SAME tapes/CRN
as ABAB and the perfect-information oracle. Belief filter validated (log-loss beats the marginal prior)
before any belief policy or solve is trusted; if the filter is uninformative, the study reports that
and Case A follows a fortiori.
