# Q-R1 successor — prospective corrections to self-imposed limitations (2026-07-21)

**DISCIPLINE PREAMBLE (binding).** These corrections modify NOTHING in the replication currently
running on the opened block `7570801–7570824`. That run's seeds are already opened and its
estimand (SESOI +0.01, persistent-0.90 primary) is frozen; it must complete UNMODIFIED — changing
it now, after D0 replicated at +0.0226 but heterogeneous, would be a post-hoc modification the
program forbids. These corrections apply ONLY to a **successor contract with FRESH, unopened
seeds**, frozen before those seeds are opened. The running replication's result stands as its own
prospective study.

**Why this exists.** An adversarial self-audit of Q-R1 found that the caught 0.02 heuristic was not
an isolated error — it is a *pattern*. There is a cluster of heuristic thresholds plus two
structural over-conservatisms that may be shrinking or blocking a legitimate accumulated-learning
result. Each fix below is a prospective design change, not a retroactive relaxation.

## What the running replication already established (context, not to be changed)
D0 on 7570801–7570824: retained binary context, κ=.90 early_ret_2w **+0.0226 (LCB95 +0.019)**,
κ=.75 +0.0138, iid null exactly 0, shuffled −0.005, wrong −0.065 — a genuine prospective
replication of the cold-start accumulated-resilience signal, comfortably above the +0.01 SESOI.
Caveats: **41.7% favorable (heterogeneous — a concentrated average benefit, not universal)** and
up to +3 end-of-campaign unresolved orders in some campaigns (safe label not yet). The successor
must address these, not paper over them.

## Corrections (successor, fresh seeds, frozen before opening)

### C1 — measure the FULL-campaign natural continuation, not only the 2-week common slice (highest impact)
Current design (`q_r1_retained_learning.py`: arms differ only in the first 2 decisions, then a
single common reset-MPC continuation) isolates cold-start but **erases any compounding value over
weeks 3–8**. "2 weeks" and "common reset continuation" are both arbitrary. Successor: run each arm
with its OWN controller for the full campaign and report BOTH (i) the isolated 2-week cold-start
slice (current estimand) AND (ii) full-campaign ReT under natural continuation, plus the decay
curve week-by-week. If retained knowledge compounds, the current design is hiding most of it; if it
decays fast, the two agree and the isolation is confirmed. Do not pre-commit to only the slice.

### C2 — power-derive ALL remaining heuristic thresholds (the 0.02 was not the only one)
Still hard-coded in the D-runners: `≥0.02` oracle (superseded by the +0.01 SESOI but still in
code), `≥0.10` action divergence, `≤0.005` iid/shuffled/wrong tolerances, `≥−0.005` no-adverse-cell
margin, `−0.02` worst-product, **`70%` favorable** (which D0 fails at 41.7%). Successor: derive each
from a pilot power/variance calc (as the 0.02→0.01 fix did) and freeze before seeds. **Reconsider
the 70%-favorable bar specifically:** for a genuinely heterogeneous accumulated-learning effect, a
per-pair-favorable gate is the wrong instrument — a mean-with-clustered-LCB (which D0 passes) is the
correct one; the favorable fraction should be reported as heterogeneity, not used as a hard gate
(the current replication contract already softens this — carry it forward).

### C3 — recovery/tail resilience sub-metrics as CO-PRIMARY, not secondary
The oracle-prevention finding showed foresight helps TTR / tail-CVaR / worst-4-week even when ~0 on
mean ReT; Garrido's ReT is itself a composite (autotomy/recovery/disruption periods + fill rate).
Fixating on the mean order-level scalar may hide the resilience story. Successor: elevate TTR,
tail-CVaR, and worst-4-week-window to co-primary endpoints (reported with the same rigor as
early_ret_2w), consistent with "resilience is recovery dynamics, not just the mean." (This is
reporting scope, not a metric change — canonical ReT stays thesis-faithful.)

### C4 — extend the persistence grid to the high-κ regime
`PERSISTENCE_MODES = {0.5, 0.75, 0.90}` caps at 0.90; retained knowledge matters MOST at high
persistence. Successor: add κ ∈ {0.95, 0.99} — the regime where the effect should be strongest and
cleanest, and where dose-response is most diagnostic. 0.5 stays the iid null.

### C5 — use Garrido's researcher-defined-risk license (stop over-fidelity to Table 6.12)
`RISK_LEVELS` uses only the exact Garrido Table 6.12 current/increased (R22 ×1/×3, R24 ×1/×2). But
Garrido explicitly licensed researcher-defined risks (frequency/impact/timing chosen to create
decision-relevant headroom). "More risk ≠ more premium" was concluded from Garrido's exact levels
only. Successor: a small direct-SimPy researcher-defined risk regime, selected by H_PI / ranking
reversals / observable classical conversion — NEVER by learner return — designed for
decision-relevance (frequency × impact × timing × concurrency), disclosed as a doctrine-grounded
assumption. This is the war-stress-atlas idea at feasible scale.

### C6 — the environment-size trade-off (STRATEGIC USER DECISION, not auto-included)
The deepest self-imposed limitation is that the exact 4⁸=65,536 frontier FORCED a decision space so
small the decision is belief-insensitive (<5% action change) — this is *why* RL≈MPC. A richer
action space (Discrete(8), per-batch 24-binary, or a longer horizon) gives adaptation more room but
loses the exact answer key (degrades to bounds/BnB at confirmatory scale). This is not a bug to
silently fix — it is a rigor-vs-headroom trade-off the PI should decide explicitly before any
successor commits to an action space. Left OPEN pending that decision.

## What stays (NOT arbitrary — do not relax)
Same information for learner and MPC; selection never on learner returns; the RETAINED-MPC
comparator; sealed one-shot seeds + custody; direct-SimPy for action-dependent masks; the
thesis-faithful ReT metric. Relaxing these would forfeit attributability.

## Sequencing
1. The running replication (7570801–824) finishes unmodified; its D0/D3 result is adjudicated as-is.
2. THEN a successor contract folds in C1–C5 (C6 pending the PI's environment decision), with fresh
   collision-scanned seeds, frozen before opening.
3. C&IE manuscript of the current package proceeds in parallel, independent of all of this.
