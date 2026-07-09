# Paper 2 Research Program: Strategic Buffering Under Commitment Physics

Date: 2026-07-09

## Working thesis

The preventive value of reinforcement learning is not an architectural property alone. It is
jointly determined by (i) whether the action contract exposes a physically effective reserve
lever, (ii) whether replenishment is committed through time, and (iii) whether the disruption
regime is severe enough to drain working stock. In the current R21 stress assay, the confirmed
11D advantage is real, but the identified mechanism is a heterogeneous reserve **posture**, not
event anticipation or state-contingent timing.

Working title:

> When Does Strategic Buffering Create Resilience? Contract-Regime Alignment and Learned
> Reserve Postures in a Thesis-Grounded Supply-Chain DES

Paper 1 remains unchanged: under `track_b_v1`, RL improves adaptive recovery but has no
preventive channel. Paper 2 studies the distinct `track_bp_v1` extension.

## Source idea bank

The durable proposal inventory is `docs/RESEARCH_PROPOSALS_REGISTRY_2026-06-28.md`. The ideas
selected for this paper are:

- P3, two-stage policy: separate strategic posture from weekly adaptive control;
- P7, strong fixed/dense buffer baselines;
- P1, holding-cost and constrained-resource sensitivity;
- P10, regime/lead/horizon generalization;
- P11, full resilience, tail, service, and resource panel.

Auxiliary-head prediction, recurrent policies, Transformers, H4 retention, and additional
reward searches are not part of this paper unless the posture baselines first leave genuine
dynamic headroom.

## Evidence already established

1. **Physical headroom is conditional.** Under Garrido-native and moderate R21 intensities,
   always-prepared and never-prepared are equal. Headroom appears only in the compounding
   starvation region (frequency at least x4 and impact x4; strongest at x8/x4).
2. **The 11D contract improves over 8D.** At 5 seeds x 60k, PPO-11D minus PPO-8D is
   `+0.028488`, seed-clustered CI95 `[+0.015813,+0.041163]`, 5/5 seeds positive.
3. **Timing is not established.** The original 8D-plus-common-scalar graft was confounded and
   is superseded.
4. **Within-checkpoint timing is null and cross-fitted.** Per-episode per-op clamping changes
   ReT by only `+0.000153`, CI95 `[-0.000899,+0.001206]`; pre-event and post-event blocking
   are null. A disjoint 12-episode calibration then froze the global posture before opening
   evaluation: frozen `0.340321` versus dynamic `0.340164`, self-minus-frozen CI95
   `[-0.000746,+0.000433]`. Weekly buffer variation is unnecessary.
5. **Architecture sidecar.** Real-KAN-11D matches PPO-11D at screen scale, but KAN-8D has not
   been run; this is architecture parity, not an architecture-matched causal decomposition.
6. **Cost scope.** Confirmatory significance survives a modest holding charge near `lambda=0.05`;
   it does not survive `lambda=0.2`. The current holding proxy is target fraction, not actual
   inventory exposure.

## Research questions and hypotheses

### RQ1: Where does strategic-buffer headroom exist?

H1: reserve headroom is positive only when disruption frequency and recovery duration jointly
drain working stock. Test with the existing frequency x impact surface, extended across lead time.

### RQ2: Does learning add value beyond a strong fixed reserve posture?

H2: PPO-11D must beat an 8D PPO trained from scratch under a frozen, calibration-selected
per-operation reserve vector. If it does not, the result is optimal reserve design plus adaptive
recovery, not dynamic preventive control.

### RQ3: Does temporal scheduling add value?

H3: the same-checkpoint policy must beat calibration-frozen per-op constants and temporally
permuted schedules under seed-clustered inference. Current evidence predicts a null.

### RQ4: Is the result economically and physically robust?

H4: the reserve posture remains non-dominated under actual inventory exposure, modest holding
charges, and a route-aware replenishment sensitivity that cannot inject stock through a disabled
supply path.

### RQ5: Is the mechanism architecture-specific?

H5: PPO and Real-KAN converge to comparable reserve postures and ReT under matched 8D and 11D
decompositions. KAN is interpretability/novelty evidence, not the causal spine.

## Experiment gates

### Gate A: frozen-posture identification, eval-only

- Estimate one `(Op3, Op5, Op9)` vector per checkpoint from disjoint calibration episodes.
- Freeze it before opening canonical evaluation episodes.
- Compare the same 11D checkpoint as-is vs per-seed frozen vs globally frozen posture.
- Primary inference: t-CI over five training-seed mean deltas.
- Promotion: scheduling is supported only if the lower CI bound is positive.

Status: **complete, null**. Global calibrated posture `(0.1531, 0.2480, 0.2068)`;
dynamic-minus-frozen `−0.000156`, CI95 `[−0.000746,+0.000433]`.

### Gate B: retrained strong posture baseline

- Train canonical 8D PPO under the globally calibrated fixed reserve vector.
- Screen: 3 seeds x 30k; confirm only if 11D retains material headroom.
- Confirm: 5 seeds x 60k, 24 CRN episodes, identical h104/R21 x8/x4 protocol.
- Primary contrast: PPO-11D minus PPO-8D-fixed-posture.
- If null, the paper reports a two-stage result: learned reserve design plus adaptive control;
  it does not claim dynamic preventive RL.

### Gate C: fixed reserve frontier

- Calibration-only search over per-op reserve vectors; evaluation remains held out.
- Start with coordinate levels `{0, 0.1, 0.2, 0.3, 0.5}` around the learned posture, then refine
  locally. Do not multiply this by every weekly static action; weekly control is supplied by the
  trained 8D policy.
- Include no-buffer, blanket-buffer, common-scalar, learned-per-op, and optimized-per-op rows.

### Gate D: physical robustness

- Record actual time-weighted inventory in each buffer, replenishment arrivals, and requested
  target fractions separately.
- Add a sensitivity where each replenishment request captures its target at order time.
- Add a route-aware arrival sensitivity: replenishment is delayed or blocked when its supplying
  path is unavailable. Keep this separate from the original Track B-P result.
- Repeat the strongest fixed and learned postures at lead times 168/336/504 h.

### Gate E: economics and generalization

- Holding-price frontier using actual inventory exposure, not action fractions.
- Optional constrained PPO only after the static posture frontier is known.
- Evaluate R21 cells x4/x4, x8/x3, x8/x4; horizons h52/h104/h260.
- Report Excel ReT, CVaR05, CT/RP tails, service-loss AUC, backlog clearance, fill, actual
  inventory exposure, and total resource cost.

## Claim ladder

1. **Already supported:** contract-regime interaction creates strategic-buffer headroom.
2. **Already supported:** 11D access improves over separately trained 8D control in R21 x8/x4.
3. **Pending Gate A/B:** learning selects a superior heterogeneous fixed reserve posture.
4. **Currently unsupported:** state-contingent preventive scheduling.
5. **Affirmatively unsupported:** hazard-clock anticipation.
6. **Never claim without Gate D:** operational deployability of the replenishment mechanism.

## Paper structure

1. Introduction: prevention requires both information and a mediable commitment channel.
2. Related work: proactive buffering, inventory-control RL, action-space design, resilience.
3. DES and contracts: Track B boundary vs Track B-P extension; replenishment semantics.
4. Pre-registered gates: physical ceiling, regime surface, contract ablation, posture controls.
5. Results: headroom map; 11D vs 8D; fixed-posture decomposition; cost/physics robustness;
   architecture sidecar.
6. Discussion: prevention as reserve posture rather than prediction; managerial reserve sizing;
   limits of engineered stress and direct-replenishment abstraction.
7. Conclusion: prevention is contract-regime conditional, and dynamic timing is not automatic.

## Compute allocation

- Local Mac: smoke tests, eval-only posture audits, figures, manuscript builds.
- VPS: 3-seed screens and fixed-posture PPO training in tmux with durable logs.
- Kaggle: confirmatory 5-seed scale or lead/horizon matrix only after a screen passes.

No additional architecture or reward grid should run before Gates A and B close the central
mechanism question.
