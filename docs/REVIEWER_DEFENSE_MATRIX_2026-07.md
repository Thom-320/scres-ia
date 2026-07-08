# Reviewer Defense Matrix (2026-07-08)

Prepared answers to the six anticipated Q1 review attacks identified in the
strategy plan (`docs/CLAIMS_REGISTRY_Q1_DEFENSE_2026-07-01.md` Reviewer #2
package, extended after the prevention-boundary and SAC/TD3 work closed
this session). Each entry gives: the attack in the reviewer's own likely
words, our prepared response, the exact numbers to cite, and the artifact
path a rebuttal letter should link back to. Cross-reference the Claims
Registry (`docs/CLAIMS_REGISTRY_Q1_DEFENSE_2026-07-01.md`) for the
canonical claim IDs (C-numbers) behind each answer.

---

## 1. Baseline strength — "You beat a weak/undertuned baseline"

**Attack.** "The static comparator is probably not well-tuned; a stronger
heuristic or a properly-optimized static policy would close the gap."

**Response.** The comparator is not a single hand-picked static policy but
the best of a **147-cell dense grid** (5 inventory-buffer levels x 3 shift
levels x downstream dispatch multipliers at Op10/Op12), evaluated under
common random numbers (CRN) paired against PPO episode-for-episode. We
additionally show:

- The delta is **robust to which of the top-12 static comparators is
  used** — every one of the twelve strongest static policies loses to PPO
  individually, with CI95 excluding zero, and the delta *grows* down the
  ranking (the opposite of what winner's-curse selection bias would
  produce). (C19, Appendix A of the manuscript, Table `tab:top12_robustness`.)
- A **local 3x3 upstream bound** at the best downstream cell (Op3/Op9
  quantity multipliers) does not overturn PPO: best bound policy ReT
  `0.005612` vs PPO `0.005666`, seed-paired delta `+0.0000540` CI95
  `[+0.0000424, +0.0000656]`. (C20.)
- Six **reactive heuristic policies** (not just static constants) are
  included in the comparator pool and all lose to PPO (Table
  `tab:privileged_defense`, "Best E1 heuristic" row: `0.005436` Excel ReT
  vs PPO `0.005893`).
- The dense grid is not an exhaustive 8-dimensional static frontier (we
  say so explicitly), but it is far denser than any comparator used by
  the closest rival paper (Ding et al., IJPE 2026, uses only QMIX and
  MADDPG as baselines — no static/heuristic frontier at all).

**Cite:** C1, C19, C20; `docs/track_b_q1_stats_2026-07-02_final_10seed/`;
`docs/track_b_q1_stats_2026-07-02_final/top12_static_robustness.csv`;
`outputs/experiments/track_b_upstream_bound_3x3_2026-07-02/`.

---

## 2. Privileged observation — "PPO is exploiting information a real
   operator wouldn't have"

**Attack.** "The observation vector includes the true disruption regime
and forecast fields derived from the ground-truth transition matrix. A
static policy doesn't get this. The comparison is unfair."

**Response.** Two independent, complementary audits close this from both
directions:

- **Give a non-learning policy the same privileged signal.** A
  regime-conditioned lookup table, fitted directly on the true adaptive
  regime PPO could exploit, gains only `+0.0000277` over the best static —
  PPO's gain is `+0.000427` over the same comparator, **15x larger**. If
  privileged regime access alone explained the win, the regime table
  should have closed most of the gap; it closes almost none of it. (C17,
  E1 go/no-go audit.)
- **Take the privileged signal away from PPO.** Retraining with the seven
  privileged fields (5 true-regime one-hot + 2 true-transition-matrix
  forecast fields) masked to zero, PPO retains **~95% of the canonical
  delta** (`+0.000395` vs `+0.000415` full-observation). (C16, E2 audit.)
- A **third, most conservative check**: a 15-seed fixed-RNG retrain
  removing only the two forecast fields (leaving regime one-hots intact)
  shows the forecast channel carries essentially zero marginal value:
  paired delta `+0.00000214`, CI95 `[-0.00001255, +0.00001683]`, 9/15
  favorable — statistically indistinguishable from zero, meaning deleting
  the forecast signal outright "changes nothing measurable." This result
  is now in the manuscript (§4.5, Figure `fig11_no_forecast_defense`).

**Cite:** C16, C17; `docs/E1_GO_NO_GO_VERDICT_2026-07-02.md`;
`docs/E2_PRIVILEGED_OBSERVATION_VERDICT_2026-07-02.md`;
`docs/TRACK_B_NO_FORECAST_FIXED_RNG_FINAL_VERDICT_2026-07-05.md`;
manuscript Table `tab:privileged_defense`, Figure `fig:no_forecast_defense`.

**Residual honest limitation:** the ablation removes the seven explicitly
privileged fields, but the remaining observation still contains rich
operational feedback (backlog, queues, rolling fill). We do not claim
full field-deployability of every remaining state variable — stated as
Limitation #3 in §5.5.

---

## 3. Single topology — "One supply chain, one benchmark; how do you know
   this generalizes?"

**Attack.** "This is a single validated DES of one military food supply
chain. The result could be an artifact of this specific topology."

**Response.** We do not claim topology-general RL superiority. The paper's
claim is conditional: *when the controllable action space reaches the
system's binding operational constraint, RL captures resilience value
that a fixed action space cannot* — a design principle (Theory of
Constraints applied to RL action-space design), not a universal law about
RL and supply chains. Within this one topology we show the principle is
not a single-cell artifact:

- **Cross-regime stress evaluation**: frozen canonical checkpoints (no
  retraining) win in 4 of 6 tested cells (current/increased risk x
  h52/h104 horizons) with fully positive seed-clustered CI95, and lose
  narrowly in both `severe` cells — a disclosed boundary regime, not a
  hidden failure. (C11.)
- **Regime-sufficiency rebuttal**: retraining Track A's original
  buffer/shift-only action family fresh on the exact `adaptive_benchmark_v2`
  regime that Track B wins on, it *still loses* to the best static
  (`0.00530` vs `0.00539`) — the regime itself is not sufficient; the win
  requires downstream-dispatch action-space access. This directly answers
  "you engineered the regime to make Track B win." (C15.)
- **Action-space decomposition** shows the effect concentrates specifically
  in downstream-dispatch access (`downstream_only` +0.000429 > `joint`
  +0.000367 ≈ `shift_only` +0.000377 vs best evaluated comparator), which
  is a mechanism claim, not a topology claim — the same reasoning should
  transfer to any system with an analogous binding downstream bottleneck.
  (C7.)
- We depth-validated the DES itself against the thesis reference before
  any RL claim (Section 4.1 / Figure `fig12_des_validation`), which the
  closest rival paper (Ding et al.) does not do — their simulator is
  unvalidated synthetic.

**Cite:** C7, C11, C15; `docs/E3_H104_DENSE_FRONTIER_VERDICT_2026-07-02.md`;
`docs/E3_GENERALIZATION_VERDICT_2026-07-02.md`;
`docs/E4_ABLATION_VERDICT_2026-07-02.md`.

**Residual honest limitation** (§5.5, Limitation #1): generalization to
other topologies, demand patterns, or organizational structures requires
further investigation — stated plainly, not hidden.

---

## 4. Effect size — "+0.000438 Excel ReT sounds small; is this actually
   operationally meaningful?"

**Attack.** "The headline delta is a fraction of a percent on an opaque
composite metric. Translate this into something a reviewer or a
practitioner can evaluate."

**Response.** Two independent translations, both already in the
manuscript:

- **Ceiling-fraction framing** (§4.3 footnote): the recovery branch of the
  ReT formula is empirically capped at `0.5/72 ≈ 0.006944` per order in
  this stress regime. The measured gain moves the system from **78.6% to
  84.9%** of the attainable metric ceiling — about 6.3 percentage points
  of the operative recovery-branch range, not a marginal decimal move.
- **Operational-significance paragraph** (§5, newly added this session):
  translating the recovery-tail metrics (CTj p99) into calendar terms —
  worst-case cycle time drops from an equivalent of roughly 350 days to
  roughly 50 days at the tail. The same policy produces larger-magnitude,
  more legible improvements in flow fill rate, service-loss AUC, and
  backlog than the scalar ReT delta alone conveys.
- The gain is **consistent across dispatch-cost pricing**: from
  `lambda_d = 0.025` upward, PPO is simultaneously resilience-dominant
  *and* cheaper in total cost than the best static (Figure
  `fig14_dispatch_cost_sensitivity`) — the effect is not a rounding
  artifact traded against unpriced flexibility.
- Effect size is **stable across algorithm family** (see §5 below) and
  **stable across 18 independently-trained reward/observation
  configurations** (range `+0.000195` to `+0.000452`, all positive;
  Figure `fig16_reward_sensitivity`) — a small-looking number that
  reproduces reliably is a different epistemic claim than a small number
  that appears once.

**Cite:** manuscript §3.3 (ceiling-fraction derivation), §4.3
(`tab:dispatch_cost`, `fig:dispatch_cost_sensitivity`), §4.6
(`fig:reward_sensitivity`), §5.1 (operational-significance paragraph).

---

## 5. Algorithm scope — "This is a PPO paper wearing a supply-chain
   costume; would any algorithm show this?"

**Attack.** "You only tested PPO (plus RecurrentPPO as a weak comparator).
Maybe this is a PPO-specific artifact, not evidence about action-space
alignment."

**Response.** We explicitly do **not** claim algorithmic superiority — the
claim is action-space alignment, and algorithm choice is not the tested
mechanism. To close the objection empirically rather than only by
disclaimer, we ran a **screening-scale SAC/TD3 confirmation** under the
identical canonical protocol (`control_v1` reward, observation v7,
`track_b_v1` 8D action contract, 3 seeds x 30,000 timesteps, same CRN
eval harness and static/heuristic comparator set as the PPO screen):

- **SAC**: Excel ReT `0.005911` vs best static/heuristic comparator
  (`heur_disruption_aware`, `0.005418`), delta `+0.000493`
  (**+9.10%**), all 3 seeds individually positive
  (`0.005894` / `0.005933` / `0.005905`).
- **TD3**: Excel ReT `0.005893` vs the same comparator, delta
  `+0.000475` (**+8.77%**), all 3 seeds positive
  (`0.005870` / `0.005894` / `0.005914`).
- Both land in the same range as PPO's own screen-scale result — the win
  is not an artifact of on-policy learning specifically.

We report this as a **scope check, not a confirmatory-scale claim**: it
was not run at the 10-seed budget used for the headline number, and we
say so explicitly in §5.5. A 5-seed x 60k confirmatory scale-up kernel
was built and locally smoke-tested (both SAC and TD3 ran end-to-end
through the full pipeline) but is not yet available as a result — future
work (§5.6).

**Cite:** C27; `outputs/experiments/track_b_{sac,td3}_screen_3seed_30k_2026-07-08/`;
manuscript §5.5 (updated this session), §5.6 Future Work.

---

## 6. Prevention absence — "Why doesn't the policy anticipate disruption
   before it happens? Isn't that the more valuable claim?"

**Attack.** "A truly intelligent policy should prepare *before* a
disruption, not just react well afterward. The absence of an anticipatory
result is a weakness, or worse, suggests the method has a ceiling."

**Response.** This is the strongest and most unusual part of the paper's
defense: we do not merely fail to find prevention, we **actively looked
for it, found a false positive, retracted it under adversarial controls,
and then closed the boundary with a designed ceiling-test methodology**
— itself a contribution, not just a limitation.

- An initial auxiliary-loss architecture (the "Ruta B" splice-gate
  design) appeared to show 74.1% positive preventive pairs. Adversarial
  controls (permuted labels, a reactive-PPO negative control that should
  fail the gate and didn't) exposed the gate as an artifact of the
  evaluation design, not a real preventive signal. We retracted the
  claim and rebuilt the trusted gate, which showed reactive/Ruta B
  positive rates of only 3-19% — far below a genuine preventive bar.
- We then ran a **generalized ceiling test** (forced-prep response
  surface + a clairvoyant-PPO upper bound with true future risk labels
  visible in both training and evaluation, no auxiliary head) across the
  **full mediable risk roster** — not just the one risk originally
  tested. This covers every risk family with an in-principle causal
  pre-positioning channel: R23 (theatre pre-positioning bridges an Op12
  block), R12/R13 (upstream buffer levers bridge contract/supplier
  stalls), R21 (multi-op knockout), and the `surge_inertia` lever (the
  one mechanism in the action contract purpose-built to reward
  anticipatory timing). Every tier is null: 0/45 to 0/66 real-anchor
  positive rate against 0-2/96 placebo, with four of six tiers showing
  an *exact* zero-vs-zero difference-in-differences. The clairvoyant PPO
  ceiling — genuine perfect foreknowledge, the best any architecture
  could do — does not beat the reactive baseline and costs more.
- This closes the objection more strongly than a single-risk negative
  result would: the claim is not "we didn't find anticipation in R22,"
  it is "under this action contract, no mediable risk family in the
  roster offers a preventive channel, including the one lever designed
  specifically to reward pre-positioning." That is a designed,
  adversarially-tested boundary result, cited alongside the retraction
  methodology as a contribution in its own right (§4.8, Figure
  `fig9_prevention_ceiling`).
- We are explicit that this is a scope statement about *this* action
  contract and risk roster, not a claim that preventive RL is impossible
  in principle — a pre-registered extension (action lead-time queuing,
  strategic buffer replenishment lag) is named as the concrete next step
  if prevention is wanted, and is out of scope for this paper.

**Cite:** C25, C26; `docs/PREVENTION_GATE_AUTOPSY_AND_CLOSURE_2026-07-07.md`;
`docs/TRACK_B_RUTA_B_COUNTERFACTUAL_GATE_AUDIT_2026-07-07.md`;
`docs/TRACK_B_PREVENTIVE_HEADROOM_CEILING_VERDICT_2026-07-07.md`;
`docs/TRACK_B_PREVENTION_HEADROOM_GENERALIZED_VERDICT_2026-07-08.md`;
manuscript §4.8, Appendix B (`fig10_efficiency_architecture` — the one
surviving secondary result from this line of investigation, an
architecture-driven resource-efficiency effect explicitly *not* claimed
as anticipatory).

---

## Rival-paper positioning (supporting context for all six)

**Ding et al. (IJPE 2026)** — the closest published competitor — uses
MAPPO for strategic-level topology reconfiguration (filling/repairing/
recruiting links in an abstract supply-chain-of-systems network).
Relative to this paper:

| Dimension | Ding et al. 2026 | This paper |
|---|---|---|
| Baselines | QMIX, MADDPG only | 147-cell dense static + 6 heuristics + regime-conditioned oracle table |
| Simulator validation | Unvalidated synthetic | DES validated against thesis reference (Table 6.10), within accepted threshold |
| Statistics | 10 unseeded repetitions, no CIs | Seed-clustered bootstrap CI95 throughout, 10-seed confirmatory scale |
| Ablations | None | Action-space decomposition (joint/downstream-only/shift-only), privileged-observation masking, reward-sensitivity screen (18 cells), algorithm-scope screen (SAC/TD3) |
| Prevention claim | None (explicitly reactive) | Actively tested, initially false-positive, retracted, and closed as an adversarially-tested boundary result |
| Negative results reported | None found | Track A boundary, prevention boundary, `ReT_tail_v2` reward-design negative — all disclosed |

Cite as complementary (macro-topology reconfiguration vs. operational
control within a validated fixed topology), not as a direct competitor to
be "beaten" — the two papers ask different questions at different
organizational levels. The differentiators that matter for a Q1 outlet
(validated simulator, dense frontier, causal ablations, honest negatives,
a genuine prevention-boundary contribution) are all absent from Ding et
al. and present here.

---

## Open items not yet closed (disclose, do not hide)

- Cross-regime generalization matrix (§4.7) uses a fixed dense-comparable
  static set per cell, not a fully reoptimized 147-cell frontier in every
  cell — stated as Limitation #6.
  R2/R24/mixed-family risk rosters and h260 horizon remain untested.
- SAC/TD3 confirmed only at screening scale (3 seeds x 30k); a 10-seed
  confirmatory extension is future work (§5.6), kernel built but not run.
- The prevention-boundary generalization reused the reactive Case C PPO
  checkpoint as the reference policy for all six ceiling-test tiers —
  off-distribution for single-risk rosters. Disclosed in the
  generalized-verdict doc; the four exact-zero tiers make this caveat
  unlikely to change the conclusion, but it is not hidden.
- CRediT authorship, funding statement, and acknowledgments sections
  remain `[TO COMPLETE]` placeholders pending author-list finalization.
