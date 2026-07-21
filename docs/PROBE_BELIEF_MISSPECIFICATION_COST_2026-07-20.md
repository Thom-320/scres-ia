# Probe spec — belief-misspecification cost + fair MPC variants (EXPLORATORY_NO_CLAIM)

**v2 — 2026-07-20.** Incorporates the auditor's six identification corrections to v1 (`fc92dfa`).
Two of them fix over-claims v1 made; verified against the canonical full-DES split before folding
in. **One-page discovery spec, not a frozen contract. Burned tapes only, no training.** Purpose:
*decide* whether the Q/T0 structured MPC's frozen belief is what leaves a residual for a learner,
and make any future win fair (MPC gets the same information).

## The question (stated correctly — belief gap is a CANDIDATE, not a finding)

Verified in `supply_chain/program_t_full_des_mpc.py:36-37` and
`contracts/program_o_ret_only_learner_v1.json`: every reinforced MPC plans with a frozen belief
`(ρ,s)=(0.75,0.90)`, correct only in `rho75_share90` and misspecified in the other two cells.
T0 leaves a small positive learner-over-MPC residual. **But the two providers disagree on its
shape, so the mechanism is unconfirmed:**

| Cell | belief vs truth | transducer reselection Δ | **canonical full-DES split Δ** | split winner |
|---|---|---:|---:|---|
| `rho75_share90` | correct | 0.00470 | **0.00310** | `constraint_aware_h3_p32` |
| `rho90_share75` | wrong (both) | 0.00927 | **0.00572** | `nominal_h1_p1` |
| `rho90_share90` | wrong (persistence) | 0.01465 | **0.00283** | `nominal_h1_p1` |

The transducer numbers *look* like "edge scales with misspecification" (v1 asserted this). The
**canonical full-DES split does not reproduce it** — the largest Δ is the middle cell and the
smallest is the most-persistence-misspecified cell. Two of three canonical winners are
**horizon 1**, so "truncation ruled out at H=8" (v1) was a proxy-provider artifact. Correct
framing: *belief misspecification is one candidate explanation of the residual; the probe must
decide it, not assume it.* Other live candidates the probe cannot ignore: proxy-objective error,
rounded nominal demand, distribution error, proxy-vs-canonical ReT mismatch, horizon.

## Design

- **Physics:** Program O risk-off, `fixed_clock_physical_v1`, the 3 confirmed cells. As in Q.
- **Tapes:** BURNED exploratory, 24 tapes/cell, CRN across all arms. Block **7570301–7570372**
  (72 seeds) — clean under two independent tree greps (between R0-robustness ≤7570124 and Q2
  training ≥757100001); register as `BURNED_EXPLORATORY_BELIEF_PROBE` and still clear it through
  the registry owner's authoritative global scanner before opening.
- **Planner, canonical provider only, do not mix providers:** primary `ret_proxy_scenario_h3_p4`
  (canonical ReT-transducer finalist); safety `ret_proxy_constraint_aware_h3_p4`; auxiliary
  fast screen `compact_nominal_h8_p1` labelled as such. Never present `nominal_h8` as evidence
  that truncation was eliminated.

### Belief arms (vary only the belief the planner plans with)

| Arm | Belief | Deployable? |
|---|---|---|
| `mpc_frozen` | point mass `(0.75,0.90)` — the current Q/T0 comparator | yes (baseline) |
| `mpc_oracle_params` | true `(ρ,s)` for the cell, **but still infers the latent regime Zₜ causally from observed demand** — it does NOT receive Zₜ | no — privileged on parameters only |
| `mpc_adaptive_bayes` | exact joint Bayesian filter (below) | yes — honest strong comparator |

`mpc_oracle_params` is deliberately *not* given the current regime state; handing it Zₜ would
conflate parameter cost with privileged state information — the whole point is to isolate the
parameter cost.

### The adaptive comparator is an exact small Bayesian filter (not particle/EM)

The parameter space is exactly the three candidate models
`Θ = {(0.75,0.90), (0.90,0.75), (0.90,0.90)}`. Maintain the exact joint posterior
`p(θ, Zₜ | y₁:ₜ)` over the **6 joint states** (3 params × 2 regimes) by Bayesian update — it is
deterministic, cheap, auditable, and better calibrated than EM on ~48 orders/episode with no
particle noise. Two non-negotiables:

1. The MPC must **plan over the full posterior mixture**, never substitute `(E[ρ], E[s])` — a
   mixture of HMMs is not the HMM of averaged parameters.
2. The prior over θ must have **support on all three models**; "start at the frozen prior" cannot
   mean a point mass at `(0.75,0.90)`, or it can never learn the other parameters.

## Estimands (per cell; point + coarse bootstrap CIs — exploratory)

- `C_missp = V(mpc_oracle_params) − V(mpc_frozen)` — cost of the frozen misspecification.
- `C_adaptive = V(mpc_adaptive_bayes) − V(mpc_frozen)` — what a deployable filter recovers alone.
- **`G_remaining = V(mpc_oracle_params) − V(mpc_adaptive_bayes)` — the decisive one: value the
  best structured filter leaves on the table, i.e. what a learner could still capture.**
- Extra readouts: action divergence between arms; first week of divergence; belief
  log-loss/calibration; per-campaign ReT; early (first two decisions) service-loss; worst-product
  fill for every arm (does a correct belief also fix the Q worst-product defect, or is it separate?).

## Invariants that survive exploration (non-negotiable)

Exact scheduled-resource equality across arms (max abs dev = 0); no privileged leak into
deployable arms (`oracle_params` privileged on parameters only, never Zₜ; `adaptive_bayes` sees
only realized demand); CRN; direct-SimPy replay for the adaptive arm (belief update changes the
plan, so transducer fixed-calendar parity does not cover it — transducer only accelerates the two
fixed-belief arms with a ≥10% direct-SimPy spot-check); `EXPLORATORY_NO_CLAIM` on all outputs;
minimal result JSON only.

## Compute preflight (amendment 2)

Measured planning times: canonical `scenario_h3_p4` ≈ 0.64 s/plan; compact `nominal_h8` ≈ 0.14 s;
compact `scenario_h3_p32` ≈ 0.011 s. ~432 planner rollouts ≈ ~5 min of canonical planning plus
SimPy and the adaptive filter. Run a 1-tape × 3-arm smoke, project from measured time, **hard cap
30 min, stop if the projection exceeds it.**

## Graded routing verdicts (decisions, not claims) — the 0.02 bar for a big campaign stays

- `CRACK_MATERIAL` — `C_missp` ≥ 0.02 in ≥2 cells.
- `CRACK_USABLE` — `C_missp` ≥ 0.01 **and** materially different actions between arms.
- `ADAPTIVE_MPC_CLOSES_CRACK` — the deployable filter recovers ≥80% of the oracle gap.
- **`LEARNABLE_RESIDUAL` — `G_remaining` ≥ 0.01: structured value remains uncaptured → authorizes
  learned belief / terminal value / hybrid, and the retained-parameter R1 lane.**
- `CRACK_NEGLIGIBLE` — `C_missp` < 0.01 in all cells → this mechanism closes (not the north star).

A large `C_missp` fully closed by the adaptive filter is **not** a neural crack. Only
`G_remaining ≥ 0.01` authorizes a learning campaign. To confirm a final ~1% premium we normally
need materially more than 1% of diagnostic headroom — so the 0.02 launch bar is not lowered.

## Execution order (cheap-first, fail-fast)

1. `oracle_params` vs `frozen` only — size `C_missp`.
2. Only if a crack exists, run `adaptive_bayes` — compute `C_adaptive`, `G_remaining`.
3. Only if `G_remaining ≥ 0.01`, authorize: learned belief residual → terminal full-DES value →
   short-horizon constrained MPC → RL fine-tuning (search the residual over the teacher, never
   from scratch); and open the retained-parameter **R1** lane (persist `p(ρ,s,Z_e | H_e)`, full
   physical reset, primary endpoint = first-two-decision ReT/service-loss, iid-parameter null).
4. If `adaptive_bayes` closes the crack, use it as the teacher and seek only terminal/residual
   value. If there is not even an oracle gap, this mechanism is closed.

## Fair-win principle (frozen)

The opportunity is **not** more information for the learner than the MPC — both get identical
causal history. A legitimate 1% win over structured control has exactly three sources: (i) better
regime inference than the parametric filter, (ii) better full-DES continuation than the MPC
horizon/proxy, (iii) amortization buying more quality at equal online budget. Never via extra
information, true parameters, an environment chosen by PPO return, a weaker comparator, or extra
resources.

## Relationship to the two live plans

Touches nothing frozen (Q/O/O-R/T0/U1/sealed S); does not collide with Codex's
retained-learning-discovery lane (that lane owns cross-campaign R0/R1; this probe is within-
campaign belief cost and *gates* R1). Feeds the U5 graded outcome set (amendment 1).
