# Paper 2 — Introduction (draft v1, 2026-07-18)

**Working title:** Learning Adaptive Control Without a Neural Premium: Exhaustive Open-Loop,
Belief-MPC, and Recurrent-RL Comparisons in a Full-DES Military Supply Chain
**Status:** replication-grade — Program Q executed 2026-07-18 (N=256/cell; E1 replicated, E2
TOST-equivalent, compound verdict STOP on the worst-product-fill guardrail; see §3.5). Every
number below traces to a custodied artifact; the claim ladder is frozen in
`docs/PAPER2_CLAIM_LADDER_2026-07-18.md`.

---

Reinforcement learning is increasingly proposed for supply-chain resilience, yet most published
gains are measured against weak, non-adaptive baselines — a fixed reorder policy, a single
heuristic schedule — leaving open whether the learner acquired *adaptation* or merely rediscovered
a better constant. We argue that adaptive value must be decomposed into four separately falsifiable
levels, and we measure all four in a validated discrete-event simulation of a military food supply
chain (Garrido-Ríos, 2017), extended with two non-fungible ration classes competing for shared
assembly capacity.

**Level 1 — physical opportunity.** With perfect information, does *any* adaptive policy beat the
best fixed schedule? In our extension the clairvoyant ceiling is material: a resource-conserving
resilience gap of 0.152 (simultaneous LCB95 0.116) against the best of all 65,536 production
calendars, collapsing to exactly zero under a fungible-product null — the mechanism, not the
noise, carries the value.

**Level 2 — observable conversion.** A belief-MPC controller using only non-privileged
observations converts a material share of that ceiling out of sample (mean canonical-resilience
advantage, lower confidence bounds +0.043 to +0.066 across three demand-regime cells, with 27/27
information placebos beaten and exactly equal physical resources). A preregistered joint
tail-safety gate at zero margin was not met; we show by instrument audit that a zero-margin
"non-inferiority" bound is mathematically a superiority test, and we report tail behaviour
descriptively rather than claiming deployment-grade safety.

**Level 3 — learned adaptation.** Ten independently seeded recurrent policies, trained on 250,240
distinct stochastic realizations with the latent regime, tape identity, and true cell parameters
withheld, beat the *complete* open-loop frontier on held-out tapes in all three cells (lower
confidence bounds +0.037 to +0.066; 41–44 of 48 tapes favorable). Action-trajectory audits,
modal/phase/frequency replacement policies, exact resource ledgers, and 990 bit-exact physical
replays certify that the advantage is genuine state feedback, not schedule discovery or resource
purchase.

**Level 4 — the neural premium.** Against the strongest structured comparator — a frozen
belief-MPC selected per cell from ten classical controllers — the learned policies show no
incremental value (mean differences −0.0015 to −0.0027; per-cell lower bounds −0.008 to −0.014).
The network learns, in effect, an amortized approximation of the structured controller: one
forward pass instead of an online planning loop, at parity of outcome.

This four-level separation is the contribution. It explains simultaneously why naive RL-in-DES
studies over-claim (they stop at Level 3 against under-specified baselines, or never reach it) and
why "RL failed" narratives under-claim (a compound success gate can label genuine learned
adaptation as failure because a well-specified operations-research controller already captures the
observable value). In our study the honest sentence is precise: *recurrent reinforcement learning
acquired genuine adaptive product-mix control, outperformed every admissible open-loop schedule,
and matched — but did not surpass — structured belief-state control under identical physical
resources.*

We contribute: (i) a fully audited full-DES multiproduct decision benchmark with an exact
65,536-calendar frontier and an exact fungibility null; (ii) the four-level decomposition with
preregistered, hash-frozen contracts and adversarial custody (sealed one-shot validations,
information placebos, resource conservation, direct physical replay); (iii) a negative result on
the neural premium with a prospective frozen-policy replication protocol (superiority over
open-loop plus TOST equivalence to belief-MPC); and (iv) a computational-amortization analysis:
where the learned policy matches MPC at a fraction of the per-decision cost, its value is
operational, not architectural.

*(Methods, results and replication sections follow the frozen contracts; the executed Program Q
outcome — endpoints replicated, compound guardrail STOP — is reported component-wise in §3.5
and does not change this framing.)*
