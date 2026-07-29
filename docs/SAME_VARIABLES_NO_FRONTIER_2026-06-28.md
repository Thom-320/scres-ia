# Characterization result: no decision frontier with Garrido's exact variables (2026-06-28)

**Central empirical finding (the paper's spine).** With Garrido's exact decision variables — on-hand
inventory buffer `I_LS` at Op3/Op5/Op9 × short-term manufacturing capacity (shifts S) — this thesis-faithful
MFSC DES has **no time-varying optimal action a dynamic policy can exploit**. A constant (base-stock) buffer
is optimal across stationary and non-stationary regimes. This is the frontier-dependent-learning theory
demonstrated, not a failed experiment.

## Four independent lines of evidence
1. **Dense-CRN falsification (Kaggle, Codex).** The dense static `f0.10_S1` (Excel 0.002279, resource 0.05)
   **dominates** the trained dynamic (0.002129, resource 0.174) on Excel ReT, CVaR, AND resource. The earlier
   "Pareto win" was an artifact of a coarse (5×3) static frontier that skipped the 0.10 sweet spot.
2. **Structural audit (thesis Figure 6.2, verbatim).** The chain is balanced at S1: manufacturing 2564 ≈
   each downstream dispatch U(2400,2600) ≈ demand 2500/day. Downstream LOC distribution (op9–12) is a FIXED
   cap. **Shifts scale only manufacturing → inert when distribution binds, and harmful via R14 (defects ∝
   production).** Buffer has authority only under R2 (distribution) disruptions. `SHIFT_DOWNSTREAM_BOTTLENECK_AUDIT_2026-06-28.md`.
3. **A0 headroom gate (5-seed CRN).** Under R2 intensities {φ2,φ4,φ6}, buffer **0.15 is ReT-optimal for ALL
   intensities** (single-buffer and per-op Op9). Headroom (oracle − best constant) = +0.002, **CI95
   [−0.000, +0.009]** — not significant. The 2-seed "+5% non-stationary opening" was seed noise.
   `scripts/run_headroom_gate.py`.
4. **Lane A closeout (ReT + service-loss tail, across R2 / R24-demand-surge / R1-manufacturing / mixed).**
   ReT rises with buffer then **saturates flat** (no cost → more buffer never worse). The "best" buffer is just
   the saturation threshold (0.10–0.15); a **constant b ≥ 0.15 ties the optimum in every regime** (R2 0.217,
   R24 0.389, R1 0.004, mixed 0.002). `scripts/run_lane_a_closeout.py`.

## The dilemma that closes the win-search (either branch → constant dominates)
- **No resource cost:** high buffer is free insurance → a constant high buffer is optimal/tied everywhere →
  no timing decision → no dynamic advantage.
- **With resource cost (charged Pareto):** a LOW constant buffer (0.10–0.15) at low resource Pareto-dominates →
  the dense frontier falsifies any dynamic policy.
Either way **a constant buffer dominates** → no dynamic win is possible with Garrido's exact variables in this
DES. Lane A (same-variables win-search) is exhausted.

## Why (the mechanism, in one line)
The disruption processes are renewal/Poisson-like (no persistent, predictable time-structure that the optimal
buffer should track), and the binding constraint (downstream LOC dispatch) is FIXED and unreachable by the
buffer/shift levers. So the optimal policy is a constant safety stock — exactly base-stock optimality from
inventory theory.

## Implications for the paper (this is the contribution, not a gap)
- **Theory:** *SCRES learning value is frontier-dependent.* A learned policy improves resilience only when the
  action space reaches the binding constraint AND the optimal action varies in time under a resource charge.
  Here neither holds with Garrido's variables → the rigorous negative result that delimits the theory.
- **Lead claim:** characterize WHEN learning helps, demonstrated by the boundary (this null) + the mechanism.
- **The two ways to MAKE a frontier (future work / extensions):**
  (a) **Track B** — make downstream LOC dispatch a decision variable (the lever ON the binding cap) → can raise
      raw ReT; deferred per user but the highest-value extension.
  (b) **L_{t-1} retained learning (H4)** — cross-campaign path-dependency on the same lane; the one remaining
      positive-result candidate, independent of within-episode timing.
- **Honest boundary (report, don't hide):** falsified stationary Pareto win, inert shift lever, DQN Δmemory
  null, h260 non-confirm. These DEFINE the theory.

## Status (corrected 2026-06-28 — do not overclaim)
Lane A: **structurally PREDICTED-null, NOT yet empirically exhausted.** The 4 lines above are a strong
prediction (constant base-stock dominates), but the **reward sweet spot `α × holding_cost` under the exact
current contract** (dense-CRN, charged, h104, φ4/ψ1.5, risk_obs/hazard, `ReT_excel_plus_cvar`) has NOT been
exhaustively searched. **Open caveat:** `α` alone with `holding_cost=0` CANNOT win the charged Pareto — with
no resource penalty the dynamic sits at high resource (α=0.1 → resource 0.610) and the lean static dominates
*by construction*. The only knob that forces economizing is the **holding cost**; the untested cell is the
JOINT `α × holding_cost` grid. **Win-rule (hard):** counts only if the dynamic matches `f0.10_S1`'s Excel/CVaR
at **≤ its resource (0.05)** or beats its tail — NOT if it imitates `f0.10_S1`.
- **Action:** complete the disciplined sequential CRN reward sweep (1-seed/20k screen → 2–3-seed/40–60k → winner
  to Kaggle), varying `holding_cost` (not just α). If it nulls → Lane A genuinely exhausted, characterization
  stands. If it finds a real charged-Pareto edge → we have a lane. Lane B (theory + H4) proceeds in parallel.
Track B remains the deferred high-value extension.
