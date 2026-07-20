# PI analysis — RL-as-optimizer ladder, risk-headroom design, and the S1 successor cascade

**Date:** 2026-07-20 · **Author:** PI-root agent (Fable), responding to PI directives of 2026-07-20
**Status:** ANALYSIS + DRAFT DESIGN. Nothing here opens seeds, freezes a contract, or reopens a
closed verdict. The S implementation lane remains the auditor's; the draft contract in §5 is
input to their freeze process.

---

## 1. The semi-objection (enumeration vs RL) — and its resolution

The PI's objection: *the point of RL is precisely not having to enumerate all policies; if the
space were 45!-sized, evaluating against "all of them" would be impossible.*

This is correct, and the program's answer is that the 65,536-calendar enumeration was never a
deployment strategy — it is the **measuring instrument**. In this deliberately small action
space we can compute the certified optimum, which converts "the learner looks good" into "the
learner is provably above every static policy" (Program Q: simultaneous LCB95 +0.062..+0.106,
replicated). In a 45!-sized space no such instrument can exist; results on this benchmark are
what calibrate how much to trust RL where no gold standard can be built. The cost asymmetry the
PI names is real and now measured: the learned policy is one forward pass at decision time,
while the audit consumed hundreds of millions of trajectories. **Certification, not learning,
is the expensive part** — that sentence is a finding of this program (Paper 2 §4.3), not a flaw.

## 2. The three-stage ladder the PI proposes, mapped to program state

**Stage 1 — agent restricted to ONE static policy (RL as optimizer).** Open question, cheap,
and uniquely powerful *here* because the enumerated frontier exists: for any calendar the agent
converges to, we can report (a) exact certified regret vs σ*_OL and (b) the exact percentile of
that calendar among all 65,536 on the same tapes — numbers no large-space study can produce.
Design sketch (draft name `program_col_static_policy_convergence_v1`): the agent commits the
8 weekly decisions blind (no intermediate observations — structurally a 4⁸-arm selection
problem); baselines at equal sample budget: random search, CEM, and enumerate-then-evaluate;
metrics: regret and percentile-vs-episodes curves. Uses only burned development tapes and the
already-custodied frontier matrices — **no virgin seeds needed for the development version**.
This is the direct answer to "can RL find the best static policy without enumeration?"

**Stage 2 — adaptation on a fixed weekly cadence.** Already executed and replicated: this is
exactly Program O-R/Q. The weekly-adaptive learner beats *every* one of the 65,536 static
calendars (H_OL replicated in all three cells, 10/10 seeds). So the ladder's stage-2 answer is
known and positive; stage 1 would retroactively complete the ladder's first rung.

**Stage 3 — the agent chooses WHEN to act (the "manager" / endogenous timing).** This is
impulse control. Current state: in the thesis-native envelope, timing seeds are CLOSED by the
frozen exhaustion rule, and the 45-profile invariance result undercuts *regime*-timing (the
optimal action does not vary with regime, so better-timed switching between regimes has nothing
to switch to) — though within-event response timing was never strictly refuted. On the Program
O product-mix mechanism **with risks active**, endogenous timing is untested and is the natural
successor question if the S lane finds risk×product headroom. The frozen discipline applies:
**oracle gate first** — measure clairvoyant impulse-control headroom over the weekly-cadence
adaptive policy before any training; if the oracle shows nothing beyond weekly cadence, do not
train. Instrument note: the branch `thesis-native-timing-oracle` already carries a
thesis-faithful time-resolved resilience metric layer (commits 27ad4af, d30ffc5) usable for
that oracle; running it requires a new prospective contract consistent with the exhaustion
certificate's reopener terms.

## 3. "¿Ya optimizamos el entorno / los riesgos?" — the honest inventory

- **Thesis-native action space (buffer/shift postures): YES, terminally.** The 45-profile risk
  escalation screen (thesis's own Cf risks, one-at-a-time R11–R24, impacts 1/1.5/2×) showed
  risks devastate physical resilience (ReT 0.53→0.20) while creating **zero regime-tailoring
  headroom**: the optimal constant posture is invariant across all 45 profiles; max
  H_profile_safe = 6.9e-05 vs the 0.01 bar. Certified, closed.
- **Program O product-mix mechanism: NO — this is exactly the open S lane.** All of Program O
  (H_PI, belief-MPC, the learner, Program Q) ran `risks_enabled=False`. Program S1 was the
  risk×product headroom screen; its *instrument* died fail-closed on 2026-07-18 (transducer
  skeleton not action-independent under PRODUCTION_QUALITY_SURGE, error 1.36e-05 vs 1e-10) —
  the *question* remains open. The PI's "optimize risks to create headroom" is the S lane's
  charter, and §5 below is the successor design.

## 4. What Garrido actually does (verified against thesis text), and what we may adopt

1. **Two risk levels per risk, explicitly designed:** Table 6.12 codes every risk with a
   − (current) and + (increased) level — e.g. R14 defect probability B(n, 3/100) → B(n, 8/100);
   R24 sudden-order inter-arrival U(1,672) → U(1,336). The escalation is a *researcher-imposed
   stress instrument*, justified exactly as the PI argues: in a digital environment you impose
   risks to test resilience because you cannot wait for real ones.
2. **Risk groups matched to hypotheses/echelons:** R1r (production: R11–R14), R2r (LOC:
   R21–R24), R3 (catastrophic) are escalated as *groups*, one category at a time (Tables
   6.13–6.15), never all at once.
3. **Levers are tested UNDER escalated risks:** Scenario II (inventory, H2) and Scenario III
   (shifts, H3) "start from the configurations described in Tables 6.13, 6.14 and 6.15" — the
   increased-risk designs. Garrido creates the adverse conditions first, then measures whether
   the lever moderates the risk→resilience relationship. This is precisely the license for our
   "select a risk envelope with headroom, then test the adaptive lever inside it".
4. **One-factor-at-a-time, decision variables frozen during risk characterization** (ceteris
   paribus, S=1, I=0): his risk sensitivity is a property of the *chain*, not of the decision.
   Our screen must go one step further (which his design cannot): measure headroom *of the
   decision* under each envelope — that is the H_PI^safe oracle per risk cell.
5. **Stochastic elements already in the thesis:** annual demand is stochastic (uniform), R24 is
   a stochastic sudden-order process, and processing times "may be affected by stochastic
   events (risks)". Adding stochastic PT or richer demand is thesis-compatible *physics* if
   parameterized justifiably — but it is NEW physics and needs its own contract and disclosure.
6. **The stationarity limitation (§8.5.1) is the license for changing risks:** he flags that
   buffering effectiveness "is guaranteed as long as demand is steady". Non-stationary
   (regime-switching) **risks** are the risk-side analog of our demand-side regime switching —
   the environment class where static policies are structurally disadvantaged and adaptation
   should matter.

**Adoption rules (so headroom-by-design stays honest):**

- We may define our own risk levels/frequencies (Garrido is the base, not the cage), but the
  envelope must be **disclosed as a researcher-defined stress envelope**, mapped per-risk to
  the mechanism's ops (the frozen relevant-risk mapping: R11/R14→Op5–7, R21/R22→LOCs,
  R24→demand; R12/R13/R23 excluded with reasons; R3 frozen).
- **Selection is prospective and oracle-gated:** envelopes are chosen by clairvoyant headroom
  (H_PI^safe) on burned tapes; any envelope with H≤0 dies a fortiori; promoted envelopes are
  hypotheses only, and conversion (classical, then learner) is tested on fresh sealed seeds
  under a new frozen contract. No post-hoc envelope shopping on scientific tapes — this is the
  winner's-curse control, the K→K2 lesson.
- **Changing risks require a matched stationary control:** to claim "adaptation is needed under
  non-stationary risks", the same total risk mass must be delivered in a stationary version;
  the estimand is the *difference* in adaptive headroom. Otherwise the claim is circular.

## 5. S1 successor — the efficiency cascade (execute)

The S1 failure was not RL cost; it was an exhaustive audit (hundreds of millions of
trajectories) that checked exactness too late. The successor (`program_s_s1b`) inverts the
order. Draft contract skeleton committed alongside this document
(`contracts/program_s_s1b_cascade_screen_draft_v0.json`, status DRAFT_NOT_FROZEN, seeds TBD by
auditor, fresh block — the 7510001–12 seeds are burned and no 751x reuse is permitted):

- **G0 — adversarial exactness first.** Every mask × extreme calendars (all-C, all-H, static,
  oracle, modal) × all 12 dev tapes × critical points, transducer vs direct SimPy at 1e-10,
  BEFORE any matrix generation. This gate would have caught the R14 action-dependent skeleton
  on day zero.
- **Per-mask instrument routing.** Masks whose physics are action-dependent (defect/rework
  class) route to direct SimPy (or a corrected transducer that admits action-dependent
  skeletons, if built and re-certified); action-independent masks keep transducer speed.
  Early-exit of a failing mask stops that mask only, under the new contract's prospective rule
  — it does not block LOC/CROSS masks and does not contaminate them.
- **Incremental screen.** 1 tape per Morris point first; only points that are exact AND show
  preliminary oracle headroom receive the remaining 11 tapes.
- **Worker batching + caching.** Persistent worker processes evaluating many points each (no
  per-shard interpreter launch); risk tapes, demand streams, and common episode prefixes cached
  across points.
- **Certified pruning instead of blind enumeration.** Where the transducer is valid, exact
  frontier values are cheap; where SimPy-direct is required, use certified bounds/B&B or a
  cheap surrogate to *locate* candidate regions only.
- **Search vs confirmation split.** The screen's job is to nominate ≤6–12 (mask, φ, cell)
  cells; the full 65,536-calendar enumeration and all Class-B guardrails run only in promoted
  cells, on fresh seeds, under the confirmation phase of the frozen successor contract.

Estimated effect: from ~10⁸ trajectories to low 10⁶–10⁷ for the screen, with the exhaustive
instrument reserved for the handful of cells where it earns its cost.

## 6. Recommended sequencing (no new compute authorized by this document)

1. Auditor: review this cascade design; freeze `program_s_s1b` with fresh seeds; run G0.
2. In parallel (cheap, burned tapes only): Stage-1 `program_col` development probe — RL-as-
   optimizer with certified regret. This is publishable regardless of sign and directly answers
   the PI's question.
3. If S1b promotes envelopes with oracle headroom → classical conversion → learner, each on
   fresh seeds (the O-R/Q pattern).
4. Endogenous-timing (stage 3) only after S1b, and only through an oracle gate on the
   risk-active mechanism; thesis-native timing stays closed per the frozen rule.
