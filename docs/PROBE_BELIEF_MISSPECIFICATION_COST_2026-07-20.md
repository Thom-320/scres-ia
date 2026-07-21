# Probe spec — belief-misspecification cost + fair MPC variants (EXPLORATORY_NO_CLAIM)

**One-page discovery spec. Not a frozen contract. Burned tapes only, direct-SimPy, no
training.** Purpose: size the *only* documented crack for "RL beats the MPC" and make any
future win fair. Every arm is a planner run; nothing is learned, no sealed block is opened.

## Why this exists (the crack)

T0 (reselection bootstrap) shows the Q learner already edges the best *reinforced* MPC in point
estimate in all 3 cells (+0.0047 / +0.0093 / +0.0147) but not at the +0.015 safe bar. Verified
in `supply_chain/program_t_full_des_mpc.py:36-37`: **every** reinforced MPC plans with a frozen
belief `(regime_persistence=0.75, dominant_share=0.90)` — which is *correct only in
`rho75_share90`* and misspecified in the other two cells. The T0 winner was `nominal_h8` (full
episode horizon), so planning truncation is ruled out: the residual gap is a **belief-inference**
gap, and the RL edge scales with how wrong the MPC's belief is (largest in `rho90_share90`,
smallest in the correctly-specified `rho75_share90`). This probe measures the size of that gap
before anyone trains a Q2 agent.

## Design

- **Physics:** Program O risk-off, `fixed_clock_physical_v1`, the 3 confirmed cells
  `{rho75_share90, rho90_share75, rho90_share90}`. Identical to Q.
- **Tapes:** BURNED exploratory, **24 tapes/cell**, CRN across all arms (same tape drives every
  arm). Candidate block **7570301–7570372** (72 seeds) — `PENDING_COLLISION_SCAN`: the
  authoritative global scanner (registry owner) must clear it before opening; do not reuse the
  war-stress span 7580020+ or the retained-plan reservations 7570001–7570124 / 757100001+.
- **Planner held fixed at the T0 winners** so only the belief axis moves: run both
  `ret_proxy_nominal_h8` and `ret_proxy_scenario_h3` (the two T0-selected configs). Optional
  robustness: sweep `H ∈ {3,8}`. No new planner modes.

### Belief arms (the whole point — vary only the belief the planner plans with)

| Arm | Belief `(persistence, share)` | Deployable? |
|---|---|---|
| `mpc_frozen` | `(0.75, 0.90)` fixed — the current Q/T0 comparator | yes (baseline) |
| `mpc_oracle_belief` | true `(ρ, s)` per cell | **no** — privileged upper bound |
| `mpc_adaptive_belief` | online estimate of `(ρ, s)` from observed demand (particle filter over regime params, or online MLE/EM); starts at the frozen prior | yes — the honest strong comparator |

## Estimands (per cell; point estimates + coarse bootstrap CIs — exploratory, no max-t ceremony)

- `C_missp = V(mpc_oracle_belief) − V(mpc_frozen)` — the crack size (cost of misspecification).
- `C_adaptive = V(mpc_adaptive_belief) − V(mpc_frozen)` — how much a *deployable* controller
  recovers on its own.
- `Gap_remaining = V(mpc_oracle_belief) − V(mpc_adaptive_belief)` — what is left for a learned
  controller to capture that an adaptive MPC cannot.
- Report `worst_product_fill` for every arm too: does a correct belief also fix the worst-product
  problem Q failed, or is that a separate defect?
- Descriptive context row: the frozen Q learner's already-evaluated ReT on these cells (no rerun).

## Invariants that survive even in exploration mode (non-negotiable)

1. **Exact scheduled-resource equality** across all arms (max abs deviation = 0). A belief change
   must not buy resources.
2. **No privileged leakage into deployable arms.** `mpc_oracle_belief` is explicitly labelled
   privileged/non-deployable; `mpc_adaptive_belief` may see only realized demand, never the true
   `(ρ, s)`, latent regime, tape seed, or future.
3. CRN across arms; direct-SimPy replay for `mpc_adaptive_belief` (belief update changes the
   plan, so the transducer's fixed-calendar parity does not cover it — use it only as a risk-off
   accelerator for the two fixed-belief arms, with a ≥10% direct-SimPy parity spot-check).
4. `EXPLORATORY_NO_CLAIM` on every output; minimal result JSON (config, seeds, estimands,
   resource check, parity, routing verdict). No manuscript text.
5. **Compute preflight** (amendment 2) before running: measured s/episode × (3 beliefs × 2 configs
   × 24 tapes × 3 cells ≈ 432 planner rollouts) must project well under one session.

## Routing verdicts (decisions, not scientific claims)

- **`CRACK_MATERIAL`** — `C_missp` point ≥ 0.02 in ≥2 cells: beating the MPC is plausible.
  Proceed to design the Q2 win arm (rich-history RL vs `mpc_adaptive_belief`, same information
  both sides), and predict RL should win most in the misspecified cells. The honest target
  becomes: does RL match/beat the *adaptive-belief* MPC without knowing `(ρ, s)`.
- **`CRACK_NEGLIGIBLE`** — `C_missp` point < 0.01 in all cells: the MPC is robust to its own
  misspecification (observed demand corrects it fast within an episode). "Beat the MPC in this
  stationary environment" is likely unreachable via inference; route to the nonstationary
  envelope (blocked on the direct-SimPy instrument) or accept Q's certified equivalence.
- **`CRACK_INTERMEDIATE`** — otherwise: report `Gap_remaining`; if `mpc_adaptive_belief` already
  closes most of `C_missp`, the deployable win is small and a Q2 campaign is low-yield.
- **Mechanism check (all cases):** does `C_missp` track the T0 residual pattern (largest in
  `rho90_share90`, ≈0 in `rho75_share90`)? A match confirms belief-inference as the mechanism and
  is itself a publishable diagnostic for the Q paper's discussion.

## Relationship to the two live plans

- Does **not** touch anything frozen (Q, O, O-R, T0, U1, sealed S) and does **not** collide with
  Codex's retained-learning-discovery lane (that lane owns cross-campaign R0; this probe is
  within-campaign belief cost). Fills the gap both plans left open: neither measures what the
  MPC's frozen belief actually costs.
- Feeds the U5 graded outcome set (amendment 1): if the win is "match the adaptive-belief MPC at
  equal information + recover worst-product," that is `PASS_HYBRID_SAFE_EQUIVALENT`, already a
  primary result.
