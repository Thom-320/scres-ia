# DRA-2 pre-calibration DECISIONS under PI autonomy (frozen 2026-07-12)

The PI granted autonomy ("Garrido nos da autonomía; toma tus decisiones para
continuar y aplicar RL"). This replaces the Garrido-sign-off gate with an explicit
PI-autonomy authorization + honest disclosure. It also LOCKS the corrections that both
independent reviews converged on. These are binding before any calibration tape opens.

## Decision 0 — physical contract accepted AS A DISCLOSED EXTENSION (not thesis reproduction)
Authorized under PI autonomy. Provenance is labeled honestly (both reviews were right
that "thesis-grounded convoy capacity" overclaimed):
- `outbound_24h` = THESIS-SUPPORTED (Op8 PT = 24 h, AL→SB).
- `capacity_5000` = thesis SHIPMENT BATCH, **reinterpreted** as one indivisible convoy
  slot — a Garrido-informed STRUCTURAL EXTENSION, NOT a demonstrated vehicle capacity.
- `one_convoy`, `return_24h`, `full_48h_unavailability`, `partial_load_allowed`,
  `partial_load_consumes_full_slot` = RESEARCHER-IMPOSED EXTENSIONS.
- `R22` = **temporary bidirectional LOC closure** (Interpretation 1: convoy paused &
  preserved, not destroyed) — the minimal extension; stated as such in the paper.
Paper language (binding): "a stylized finite-convoy logistics extension of the
thesis-grounded MFSC," never "validated operational reproduction." The DRA-2 result
is conditional on this disclosed contract.

## Decision 1 — REQUIRED code corrections before calibration (both reviews)
Opening 60 tapes without these repeats DRA-1's confound / manufactures a resource-
purchase win. All are prerequisites:

1. **Prefix-balanced state sampling (anti-DRA-1-confound, CRITICAL).** Branch states
   must be generated from ≥3 distinct static prefixes, not the single best-static.
   Frozen: aggressive `1000/24`, middle `2500/48`, thesis `5000/48`, conservative
   `5000/72`. 60 tapes × 4 prefix-states (+ balanced recovery/high-backlog) → ~240
   states; cluster inference BY TAPE. (This is the exact confound I caught in DRA-1
   V3–V5; it must not recur.)
2. **Resource-envelope comparator EXECUTED in the oracle (CRITICAL).** The dynamic
   candidate is compared to the best static within its OWN resource envelope:
   `D_static ≤ D_dynamic AND H^away_static ≤ H^away_dynamic` (departures and
   convoy-unavailable-hours incl. route-wait). Comparator selected ONCE on calibration,
   frozen for holdout. Rename to **"best static within the dynamic candidate's resource
   envelope"** (not "equal vehicle-hours"). The oracle's current `max(long_ret,
   -long_service)` must apply this envelope. Add explicit metric
   `op8_convoy_unavailable_hours`.
3. **Strong-liveness measured, not feasibility.** Report `dispatch_feasible_fraction`
   AND `strong_live_fraction` where strong-live = HOLD & DISPATCH both admissible AND
   next-state signatures differ (availability/ETA/staging/in-transit/departures/
   vehicle-hours/next-feasible-action). The 49.8% is feasibility smoke, NOT G-B.
   G-B = PENDING until strong_live_fraction is computed on the calibration regime.
4. **Diversity on REALIZED departure patterns + tie rules.** Count realized patterns
   (128 nominal → 34 realized in smoke), not nominal action strings. Freeze a numeric
   tie tolerance; an exactly-zero improvement is NOT "optimal"; no action gets
   diversity support unless it beats the comparator by a strictly positive margin.
5. **7d primary + 10d sufficiency sensitivity.** Keep 2^7 as the primary restricted
   sequence oracle (rename **"restricted seven-day sequence oracle"** — it is an exact
   bound only for {7-day open-loop + frozen static continuation}, NOT a 28-day dynamic
   upper bound). Before reading calibration, freeze a 2^10 check on 12 pre-selected
   states (3/family): promote-as-sufficient only if first optimal action agrees ≥90%
   between H=7 and H=10, headroom changes <20% and <0.002 ReT, no family flips sign,
   no late (14/28d) service damage. Else widen the primary horizon.
6. **Authorization artifact, not a CLI flag.** Replace `--face-validation-accepted`
   with `--authorization-record docs/PROGRAM_D_DRA2_AUTONOMY_AUTHORIZATION_2026-07-12.json`
   carrying: decision, authorized_by (PI, autonomy grant), date, contract_sha256,
   disclosure labels (Decision 0). The runner verifies the contract hash matches.

## Decision 2 — corrected adaptive-headroom criterion (theory, for the paper)
Both reviews correctly rejected my earlier "reachability" condition (`A(X_{t+1})`
depends on `A_t`) as the general law (a contextual bandit is a counterexample — it has
adaptive value with no next-state coupling). The correct, resource-constrained object:
- Perfect-information headroom: `H_PI = E_X[max_a Q_H(X,a)] − max_a E_X[Q_H(X,a)] ≥ 0`,
  under a common continuation and a fixed resource envelope.
- **Zero-headroom proposition:** `H_PI = 0` iff a single constant action a* is optimal
  a.s. over the reachable state distribution. This is the "law" our 6 boundary results
  actually exhibit.
- Observable version: `H_obs = E_O[max_a Q̄_H(O,a)] − max_a E_O[Q̄_H(O,a)]`.
- Reachability/intertemporal coupling is property #5 of six RL-eligibility conditions
  (physical authority; resource trade-off; ranking heterogeneity; observable
  convertibility; intertemporal coupling; persistent learnability), NOT the law.
Paper framing: **"a resource-constrained adaptive-headroom criterion and preregistered
diagnostic hierarchy for deciding whether a DES warrants sequential learning"** — a
proposition + framework motivated by 6 boundary results, not a claimed universal theorem.

## Decision 3 — the RL gate (how we "apply RL" honestly)
PPO is trained ONLY after the corrected pipeline shows, on the resource-enveloped,
prefix-balanced, realized-pattern oracle: ≥2 actions each optimal in ≥15% of states;
oracle service-loss ≥5% CI95>0; ReT co-directional; a depth-3 observable tree recovers
≥50% and beats the enveloped comparator on holdout; ≥70% holdout tapes positive; tail
not worsened; 7d/10d sufficiency holds. If it passes → PPO vs enveloped-best-static, and
only then persistent/reset. If it fails → 7th boundary result, and the "Before RL"
decision-rights paper is complete. Either outcome is pre-committed; neither is a rescue.

## Sequence (frozen)
provenance-narrow → corrected runners (1–5) → autonomy-authorization artifact (6) →
pre-calibration freeze tag → open 60 tapes ONCE → static frontier → prefix-balanced
resource-enveloped branching + 7d/10d check → observable tree → holdout → PPO iff gate.
No PPO / no virgin tapes before the gate. DRA-1 stays closed (`dra1-stop-2026-07-11`).
