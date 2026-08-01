# C&IE manuscript rewrite map — Garrido–WRAP/v0

**Paper lane:** Garrido–WRAP/v0
**Target:** *Computers & Industrial Engineering*
**Status:** `HOLD_RESULTS_BEFORE_MANUSCRIPT_CLAIMS`

This map separates Garrido et al. (2024)'s methodological questions from the
derived hypotheses in the v0 draft. It is a writing contract, not a license to
promote pending or retired results.

## Central questions

| source question | operational answer currently allowed |
|---|---|
| Q1. Which AI family best represents learning in the supply chain? | In the current WRAP panel, a linear surrogate is already very strong; backprop/KAN do not exceed the preregistered SESOI. `NO_GO_NEURAL_PREMIUM_IN_WRAP_PANEL`. |
| Q2. How can learning be inserted between DES design and SCRES measurement? | As a between-campaign state update and configuration search (`theta_k`/`rho_k`), not as an intra-episode controller. The 90-cell result is replay-only; the corrected 288 DES rerun is pending. |

## Evidence order for the manuscript

1. **Problem and contribution:** DES-based SCRES measures can be structurally
   insensitive or vulnerable to abandonment; the paper tests where the proposed
   learning loop is identifiable.
2. **WRAP substrate:** preserve the thesis-faithful lane, source coverage, risks,
   buffers, capacity, horizon, warm-up and ReT boundary. Do not call the DES a
   one-to-one reproduction while the behavioral gate is blocked.
3. **Q1:** reject `drivers -> ReT` as leakage/identity; report the held-out task
   and the linear/backprop/KAN comparison with grouped splits and the SESOI.
4. **Q2:** define the interface explicitly: DES episode -> observable outcome ->
   state update -> next configuration. Compare retained and reset under common
   budgets, virgin tapes, and no future-event access.
5. **Derived v0 hypotheses:** report H1/H2/H3/H4 with their construct changes in
   a separate subsection. H1 original is not evaluable; H1' is service-loss AUC,
   not system recovery time; H3' is search-cost dispersion, not WRAP cost
   volatility.
6. **Boundary experiment:** report CSSU Gate A as interface liveness only and
   Gate B as held. The E1 screen observes zero leading headroom but remains held
   because its required uninformed placebo was not opened; it does not justify
   neural training.
7. **Prospective neural environment:** include E2 only as a preregistered future
   test unless its new contract and gates are actually executed. A positive neural
   result must beat the strongest classical controller on service-safe virgin
   tapes, not merely a constant policy.

## Claim language currently permitted

- “The present WRAP panel does not demonstrate a neural premium.”
- “The CSSU reassignment interface is computationally live; finite physical Op11
  handling has not been validated.”
- “The 90-configuration result validates replay/search logic, not independent DES
  behavior.”
- “A neural premium is a conditional question: it requires observable headroom,
  state-dependent action value, and a classical-control gap.”

## Claim language prohibited until gates close

- “Garrido's Figure 5 has been replicated as a valid planner.”
- “The retained learner improves Q2 in the corrected DES.”
- “H1 is a time-to-recovery result.”
- “WRAP proves cost volatility or that RL is useful.”
- “Op11 has validated finite handling physics.”
- “KAN/MLP/PPO is superior” or any neural-premium claim.

## Required figures/tables after the 288 rerun

- source/Cf coverage and thesis-vs-extension boundary;
- Q1 held-out performance with grouped confidence intervals;
- Q2 retained/reset search curve and common-budget audit;
- service, queue, and abandonment diagnostic panel;
- CSSU liveness/physical-gate diagram;
- claim-status table separating valid, retired, pending, and prospective results.

The v0 DOCX should be rewritten only after the pending H3' and DES-288 artifacts
are sealed. Until then, this file is the authoritative outline for §4.2 and §4.3.
