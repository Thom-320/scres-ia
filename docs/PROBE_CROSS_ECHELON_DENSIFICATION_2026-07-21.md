# Probe spec — CROSS_ECHELON_SURGE densification (EXPLORATORY_NO_CLAIM)

**One-page discovery spec, not a frozen contract. Burned tapes only, direct-SimPy only, no
training.** This is the **go/no-go gate** for the "beat the MPC under nonstationarity" campaign —
the only place in the whole search with large measured headroom. Every stationary mechanism is
dead (belief crack 0.006, belief-insensitive decisions <5% action change, David's transformer
ties + fails worst-product, terminal value fails, action-regret dies on holdout). The one live
lead is nonstationary.

## The lead (verified, causal-residual U1 direct discovery, `results/program_u1/direct_bounded_discovery_v1/result.json`)

Under mask **`CROSS_ECHELON_SURGE` = risks (R21, R23, R24)** — cross-echelon line-of-communication
disruption + demand surge — the direct-SimPy discovery found a strong isolated point:

- `h_pi_sampled_mean` **0.091** (rho90_share90), 0.077 (rho90_share75); best isolated classical
  **H_obs +0.1185, LCB95 +0.0273** — ~10× the stationary headroom, in **both** the clairvoyant
  (H_PI) and the observable-classical (H_obs) sense;
- **`ranking_reversal: true`** and `oracle_calendar_indices [0,31,0]` — the optimal calendar
  **switches mid-episode**: adaptation is decision-relevant here, unlike the stationary env
  (<5% action change);
- static policy sacrifices worst-product (`worst_product_fill_mean_at_static` 0.426) → both a ReT
  headroom AND a worst-product-recovery opportunity;
- `resource_range 0.0`, `lost_orders_max 0.0`, `selection_uses_learner_returns: False` (selection
  was on H_PI/H_obs, per the frozen rule).

**It STOPPED only on an underpowered connectivity check** — 36-point sparse Morris grid, 3 burned
tapes (7430001–3), 12-tape expansion, 1/5 connected → `STOP_U1_NO_CONNECTED_CLASSICAL_CONVERSION_REGION`.
That is not "no region"; it is "the sparse grid found the peak but never sampled its neighborhood
densely enough to prove a region." This probe decides: **real connected region, or isolated
artifact?**

## Design

- **Physics:** Program O extension, `CROSS_ECHELON_SURGE` (R21/R23/R24) enabled, **direct-SimPy
  MANDATORY** — this mask is action-dependent and FAILS transducer exactness (0.027 > 1e-10); no
  transducer anywhere in this probe. Cells `rho90_share90` (primary, strongest) and
  `rho90_share75`.
- **Anchor + dense local grid.** Reconstruct the factor vector at the recorded strong candidate
  (Morris group 2 / trajectory 3 / points 1–2) — the multipliers on R21/R23/R24 (frequency φ,
  impact ψ), the coupling/concurrency (independent / coincident / lead–lag 72 h), and the timing
  relative to demand and to the irreversible batch commitment. Then a **dense grid** (fine full
  factorial or ±2 steps per axis at ≥5 levels) over a local neighborhood of that vector, so
  adjacency in factor space is testable.
- **Tapes:** BURNED exploratory, **24 tapes/point** (expand promoted neighborhoods to 48), CRN —
  the SAME tapes drive every grid point and every arm. Candidate block **7570401–7570448** (48
  seeds), `PENDING_COLLISION_SCAN` by the registry owner; above the retained-plan reservation
  (≤7570124) and below the war-stress span (≥7580020).
- **Matched stationary control (equal risk mass).** Every point carries a stationary comparator
  with the SAME total R21/R23/R24 mass but no within-episode regime structure, so a pass measures
  *nonstationary structure creating headroom*, not merely "risks lower ReT."
- **Selection axis, frozen:** H_PI^safe, H_obs^classical, ranking reversals, guardrails —
  **NEVER learner return** (`selection_uses_learner_returns` must stay False; the discipline line
  that keeps a future win real).

## Estimands (per grid point; point + coarse bootstrap CIs — exploratory)

- `H_PI_safe` — clairvoyant safe-oracle vs best static calendar (full 65,536, mean-selected).
- `H_obs_classical` — best frozen non-privileged classical controller vs best static.
- `ranking_reversal_fraction` — share of tapes where the mean-optimal calendar switches within
  the episode (the decision-relevance signal).
- `worst_product_fill` vs classical and vs static (recovery opportunity).
- `resource_range` (must be 0.0) and matched-stationary-control delta.

## Connectivity gate (the crux — what the sparse grid could not test)

A **connected region** requires a contiguous set of **≥3 adjacent grid points** (neighbors in
factor space) that ALL satisfy, simultaneously:

- `LCB95(H_PI_safe) ≥ 0.02` AND `LCB95(H_obs_classical) ≥ 0.015`;
- `ranking_reversal_fraction ≥ 0.5` (≥2 materially-distinct optimal actions across the region);
- worst-product and resource guardrails pass;
- headroom exceeds the matched stationary control (nonstationarity, not risk mass alone).

## Verdicts (routing, not claims)

- **`REGION_CONFIRMED`** — a connected region exists → authorizes designing the nonstationary
  hybrid campaign in this envelope (Max-Obs RL vs reinforced belief-MPC, both same info; graded
  U5 outcomes incl. `PASS_HYBRID_SAFE_EQUIVALENT`; and R1 retention becomes meaningful because
  now experience has value to accumulate). This is the environment where beating the MPC is
  physically possible AND where the north star (accumulated experience) lives.
- **`ISOLATED_ARTIFACT`** — the peak does not extend to any connected region → this mechanism
  closes honestly; the beat-the-MPC route in this DES family is exhausted and Program Q (with its
  mechanistic "why") is the paper. Not empty-handed.
- **`INCONCLUSIVE_NEEDS_WIDER`** — signal at the grid boundary → one wider grid, then decide.

## Invariants (non-negotiable, even in exploration)

Direct-SimPy only; exact resource equality (0.0) across all arms and the matched control; no
privileged info in deployable arms; CRN; burned tapes only, no sealed seed opened;
`EXPLORATORY_NO_CLAIM`; selection never on learner return; minimal result JSON.

## Compute preflight (amendment 2)

The 36-point sparse discovery ran in 48 s at 0.0137 s/episode. A dense grid of ~200 points × 24
tapes × (H_PI enumeration + classical + matched control) is minutes-to-~1 h. Run a 1-point ×
all-arms smoke, project from measured time, **hard cap 60 min, stop if the projection exceeds
it.** No contract freezes without a passing `compute_preflight.json`.

## Relationship to the live plans

Does not touch anything frozen (Q/O/O-R/T0/U1-stationary/sealed S); does not collide with
Codex's Q2-A/B/C stationary bakeoff (that hardens Q's null for defensibility; this hunts the
nonstationary win) or the retained-learning lane. `REGION_CONFIRMED` is the precondition that
makes the nonstationary hybrid, the U5 graded outcomes, and R1 retention all executable.
