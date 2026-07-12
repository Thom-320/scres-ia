# Program H — Information Sufficiency and Belief-State Audit (preregistration, frozen 2026-07-12)

Status: **FROZEN BEFORE ANY PROGRAM H TAPE OR LEARNER.** Non-rescue boundary:
`docs/PROGRAM_H_NON_RESCUE_BOUNDARY_2026-07-12.md`. Contract: `contracts/program_h_belief_audit_v1.json`.

## The one open question (why Program H exists)
Program G (closed, `006b41c`) established, under the corrected clock + canonical ret ledger:
`J_PI > J_static` (clairvoyant spatial headroom) but `J_π < J_static` for cover, MPC, a depth-3 tree,
and MaskablePPO. That does **not** establish `J*_obs ≤ J_static`, where `J*_obs` is the best possible
NON-ANTICIPATIVE policy. The hierarchy is `J_static ≤ J*_obs ≤ J_PI`; Program G pinned the ends and
tested a few interior points. Program H resolves the residual ambiguity:

> Is observable conversion failing because the tested policies are insufficient (policy-class-limited),
> or because the available information is fundamentally insufficient (information-limited)?

Second correction Program H bakes in: the depth-3 tree was fit by **action accuracy** — the wrong
loss (Program E: 82.6% accuracy, materially worse rollout). Program H trains the observable policy on
**regret** `L = Q(s,a*) − Q(s,â)`, not `1{â≠a*}`.

## The central deliverable: an observable UPPER BOUND
The study needs more than "another policy also lost." It computes a **QMDP (and/or strengthened /
relaxed-information) upper bound** on `J*_obs`. QMDP is optimistic: `J_QMDP ≥ J*_obs`. Therefore

- if **`J_QMDP − J_ABAB < δ_min`** → `J*_obs − J_ABAB < δ_min` too → **NO non-anticipative policy can
  materially beat ABAB** (Case A, information-limited — a formal insufficiency result, the strongest
  possible close for the spatial contract);
- else a belief-aware policy that beats ABAB is possible (Case B, policy-class-limited).

This is computable because the belief space is tiny (9 joint latent tempo states, frozen semi-Markov
transitions, 4-week horizon). The bound is computed on `ret_order`, same tapes, CRN.

## Belief state (POMDP/MOMDP)
Physical state `X_t` (inventories, backlogs, convoy/route state) is observed; the latent joint tempo
`Z_t=(Z_A,Z_B)∈{low,routine,surge}²` (9 states) is filtered into a belief `b_t(z)=P(Z_t=z|H_t)` from
the frozen transition + imperfect signal + realized demand + inventory/service history. The latent
label is never revealed — the belief is an operational estimate, not a cheat. The filter is validated
(must beat the marginal prior in log-loss) before any belief policy is trusted.

## Two frozen information contracts
- **O0 — Program G exact:** only the observations already available in Program G. Settles
  policy-class-vs-information for the SAME information set (isolates the accuracy-vs-regret / class issue).
- **O1 — minimal operational ledger:** adds only data an operator would plausibly have (prev-week
  realized demand per CSSU, backlog qty + max age, recent deliveries, convoy ETA, current route). NO
  future demand/tempo/risk/closure/oracle labels. **O1 needs Garrido sign-off before the prereg locks;
  if the data is not operationally accessible, O1 is dropped.**

## Policy ladder (frozen)
ABAB (frozen) · cover/MPC (references only) · regret-weighted observable policy · belief-MPC · POMCP/
point-based · perfect-information oracle (ceiling only). Estimands: `H_PI=J_PI−J_static`,
`H_belief=J_belief−J_static`, `η_information=H_belief/H_PI`, `UB_gap=J_QMDP−J_static`.

## Universes (new; nothing from D–G used for selection)
development/calibration 1060001–1060199 · locked belief-policy test 1070001–1070399 · virgin RL test
1080001–1080399 (opened only if the learner gate passes).

## Learner-authorization gate (on 1070001+, all required)
`ΔReT_order ≥ 0.01` with `LCB95 > 0`; ret_quantity non-inferior; attended non-inferior; worst-CSSU
fill not worse by >0.02; unfulfilled rations non-inferior; resources equal; ≥70% favorable tapes;
perfect-info conversion ≥30%; **and it must beat the best NON-neural belief policy too.** If it fails,
1080001+ stays closed and the spatial contract is definitively terminal.

## Single learner if the gate passes
PPO with the filtered belief in the observation `A_t = π_θ(X_t, b_t)`; one RecurrentPPO ablation over
raw history. Frozen 2×64, 10 seeds, no sweep. persistent/reset/frozen only after beating ABAB AND the
best non-neural belief policy.

## Outcomes (all pre-committed, all publishable)
- **Case A** (UB gap < δ_min): information-insufficiency — the collection of nulls becomes a formal
  explanation. Write and submit.
- **Case B** (belief policy beats ABAB): headroom IS observably convertible; the learner gate opens.
- Either way, and even if RL later wins: **write and submit.**

## Sequence + stop
Manuscript writing starts NOW, in parallel (not after). Program H is the LAST computational extension
on the stylized contract; the full-DES Op1–Op13 port is a separate paper and must not start before the
current manuscript is submitted. "Not giving up" = finishing the paper, not opening Programs I/J/K until
a network wins.
