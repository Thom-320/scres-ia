# Track B preventive-headroom ceiling verdict — 2026-07-07

## Question

Before any further gate redesign or architecture work: does this DES, under the
`track_b_v1` 8D action contract, have ANY preventive headroom at all for R22 —
i.e., can pre-event preparation causally change the outcome of orders exposed
to an R22 disruption? This is the go/no-go test recommended unanimously by
three independent external reviews after the Ruta B causal-prevention claim
was retracted (`docs/TRACK_B_RUTA_B_COUNTERFACTUAL_GATE_AUDIT_2026-07-07.md`,
`docs/TRACK_B_RUTA_B_ESTABLISHED_GATE_FINAL_2026-07-07.md`).

Two independent, complementary tests, both eval-only or cheap-train (no
dependence on the previously-broken splice gate):

## Test 1a — Forced-prep response surface (no RL, pure branch rollouts)

For isolated real R22 anchors and matched event-free placebo anchors, force
the whole 8D action to a fixed posture (`calm=-1`, `medium=0`, `max_prep=+1`)
for the 4 weeks before the anchor, then let the reference policy run normally
from the event onward. Outcome is LOCAL: Garrido Excel ReT restricted to
orders whose `[OPTj, OATj-or-now]` window overlaps `[t_e, t_e+6w]` — the
orders actually exposed to that event, not episode-wide ReT.

Two environment tiers:

| Tier | Enabled risks | Real anchors | Placebo anchors | DiD (max_prep − calm) | Real paired positive | Placebo paired positive |
|---|---|---:|---:|---:|---:|---:|
| Case C (R22/R23/R24 background) | R22,R23,R24 | 67 | 96 | +0.0000185 | 1/67 (1.5%) | 4/96 (4.2%) |
| **R22-only (clean physics)** | R22 | 84 | 96 | **0.0** | **0/84 (0%)** | **0/96 (0%)** |

The R22-only tier is not just small — it is **bit-identical across all three
postures for every single anchor** (verified: `cost_index` DOES differ across
postures — 0.913/0.917/0.933 for one anchor — confirming forced actions are
genuinely reaching `env.step()`; `local_ret_excel_mean` is exactly
0.020058354570804664 regardless). This rules out a plumbing bug and points to
a mechanistic explanation: R22 ("LOC destruction/terrorist attack") has an
**exogenous, action-independent recovery duration** (`exponential(mean=24h)`)
and knocks out specific operations (`affected_ops=[4,8,10,12]`) directly — the
`track_b_v1` action contract's levers (assembly shift level, downstream
dispatch multipliers) have no causal path to that damage channel at all. The
small non-zero effect in the Case C tier is attributable to R23/R24's
background congestion interacting with the same downstream capacity dispatch
controls the policy uses generally, not to any R22-specific preventive value.

## Test 1b — Clairvoyant PPO (upper bound via visible ground truth, not reward shaping)

Plain PPO (default SB3 extractor, no auxiliary head) trained AND evaluated
with the TRUE future R22/R24 event label directly visible in its observation
every step (discovery-pass under fixed RNG, same mechanism as Ruta B's label
wrapper, but consumed directly rather than through an aux loss). This is a
genuine upper bound: if perfect foreknowledge doesn't help, no learned
predictor ever will.

| Policy | ReT Excel | vs static | Cost index |
|---|---:|---:|---:|
| Reactive PPO baseline (5×60k reference) | 0.481160 | +10.65% | 0.719 |
| **Clairvoyant PPO (true future visible, 3×30k)** | 0.485035 | +9.83% | **0.853** |
| Ruta B true-label | 0.481086 | +10.05% | 0.396 |

Clairvoyant does not exceed the reactive baseline's ReT and its resource cost
is the **highest of any variant tested** — perfect foreknowledge bought no
resilience gain and made the policy less efficient, not more. (Its printed
counterfactual positive-rate, 88.3%, comes from Ruta B's inlined naive gate
and is not evidence of anything per the retraction — the ReT/cost comparison
is the only load-bearing number here.)

## Stop-rule verdict

**Both ceiling tests are null. Preventive headroom ≈ 0 for R22 under the
current DES physics and `track_b_v1` action contract.** Per the pre-committed
stop rule, prevention is closed for this environment as a boundary result;
Phase 2 (gate redesign) is skipped. Recommended framing for the paper:

> "R22's damage channel (exogenous recovery duration, direct operation
> knockout) is not mediated by the downstream-dispatch/shift levers available
> to the policy; forced pre-event posture has no causal effect on
> event-exposed order outcomes (clean-physics tier: exact null, 0/84 and 0/96
> anchors), and a policy with perfect foreknowledge of the event calendar
> gains no resilience advantage over the reactive baseline while spending
> more resources. This DES rewards fast adaptive recovery, not anticipatory
> preparation, under the studied action contract."

## Ruta B efficiency-claim attribution (Phase 3, run in parallel)

The one positive result surviving all controls — same ReT as reactive at much
lower cost — needed mechanism attribution:

| Arm | ReT Excel | vs static | Cost index |
|---|---:|---:|---:|
| Reactive PPO (reference) | 0.481160 | +10.65% | 0.719 |
| Ruta B true-label (λ=0.25) | 0.481086 | +10.05% | 0.396 |
| Ruta B permuted-label | 0.485139 | +9.85% | 0.426 |
| **Ruta B λ=0 (head present, zero gradient)** | 0.484962 | +9.81% | **0.418** |
| Ruta B constant-label | 0.484651 | +9.74% | 0.419 |
| Clairvoyant (extra feature, default extractor) | 0.485035 | +9.83% | 0.853 |

True-label, permuted-label, λ=0, and constant-label are statistically
indistinguishable on both ReT (0.481–0.485) and cost (0.396–0.426) — critically,
**λ=0 (the auxiliary loss contributes literally zero gradient) reproduces the
same low-cost/high-ReT pattern as λ=0.25 with a real label.** Clairvoyant,
which adds an extra observation feature but uses the *default* PPO extractor
rather than `RutaBAuxFeaturesExtractor`, does NOT show the low-cost pattern
(0.853, matching reactive's high-cost profile).

**Attribution: the efficiency effect is architectural, not from the auxiliary
loss or its label content.** It traces to `RutaBAuxFeaturesExtractor`'s
specific trunk (features_dim=64, hidden_width=64, separate net_arch heads)
versus plain PPO's default extractor — not to prediction, not to
regularization from a live gradient. Publishable claim: "An alternative
feature-extractor architecture induces resource-efficient policies at
equivalent ReT; the auxiliary predictive task contributes nothing beyond the
architecture change itself" (falls under the plan's "λ=0 also cheap →
architecture effect" branch, not the regularization or predictive-content
branches).

## Artifacts

- `outputs/experiments/track_b_headroom_sweep_case_c_2026-07-07/`
- `outputs/experiments/track_b_headroom_sweep_r22_only_2026-07-07/`
- `outputs/experiments/track_b_clairvoyant_ceiling_3seed_30k_2026-07-07/`
- `outputs/experiments/track_b_ruta_b_lambda0_head_3seed_30k_2026-07-07/`
- `outputs/experiments/track_b_ruta_b_constant_label_3seed_30k_2026-07-07/`
- Scripts: `scripts/audit_prevention_headroom_sweep.py` (new),
  `scripts/run_track_b_ruta_b_sidecar.py` (`--clairvoyant`, `--aux-label-mode constant`, additive)

## Next steps

Per the post-retraction roadmap plan: Phase 2 (gate redesign) is skipped.
Phase 4 remains: write the gate-autopsy protocol doc, update paper framing
(adaptive recovery as the main claim, prevention explicitly not claimed,
efficiency-via-architecture as a secondary result), update
`docs/PROMISING_LANES_REGISTRY.md` and memory, commit and push.
