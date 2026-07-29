# Claim and identification contract — L(e−1) program (frozen 2026-07-11)

Status: frozen BEFORE any experiment of this program. Runtime frozen at the
versions all Tier-1 audits actually ran on (gymnasium 1.3.0, stable_baselines3
2.9.0, numpy 2.4.6, torch 2.12.1, simpy 4.1.2; `requirements-pinned.txt`
regenerated 2026-07-11 — the old 1.2.1/2.7.1 pins predated this week's audited
evidence and were stale, not authoritative).

## Primary claim (the only headline this program may produce)

> Retaining actor-critic weights across disruption campaigns improves
> held-out resilience relative to reset and frozen controls, under identical
> physical initial states, disruption tapes, observations, reward,
> per-campaign update budgets, and resource constraints.

Theoretical claim (conditional on the above):
> These results operationalize retained policy learning as an endogenous
> mechanism through which simulated resilience can become history-dependent
> (Garrido et al. 2024's L(t−1), corrected to L(e−1) at campaign timescale).

## Unit of analysis
Training seed × campaign checkpoint × common probe tape.

## Treatment
Persistence of actor-critic weights (θ) across campaigns. Optimizer state is
a TECHNICAL carrier (ablation only); normalization statistics are estimated on
the calibration panel, frozen, and identical across all arms — never part of L.

## Counterfactual
Same initialization θ₀ (pretrained on calibration only), same DES physical
reset distribution, same shock tapes (CRN), same observations, same reward,
same per-campaign updates and resource constraints — but weights reset to θ₀
before each campaign.

## Environment and decision surface (garrido_learning_v1)
- Physics: `garrido_proxy_v1` (frozen, Tier-1 audited; route-aware
  replenishment DECLARED as an operational extension; a periodic-unblocked
  replenishment sensitivity is scheduled; neither is ever called
  "thesis-faithful").
- Strategic buffer B_e ∈ {0, I168, I336, I504, I672, I1344}: chosen at
  campaign start, FIXED for the whole campaign (thesis Scenario II semantics).
  One policy trained per buffer level — cross-buffer transfer is NOT studied
  in v1 (it would conflate physical-configuration transfer with disruption
  learning).
- Tactical action A ∈ {S1, S2, S3} weekly (Discrete(3)); orders issued at week
  k take effect at k+1; minimum one-week dwell; switching logged and priced in
  the reward; no artificial internal surge budget in the primary treatment.
- Physical reset between campaigns: fresh DES + same warm-up procedure with
  campaign-specific initial_state_seed; NO inventory/WIP/backlog/RNG carry-over.
  Only L persists (in the arms where it should).
- Observation (bespoke, NOT v6–v10): buffer level, effective+pending shift,
  physical inventories (7), pending batch and in-transit material, backlog qty/
  count/max-age, rolling demand/production/deliveries, rolling fill rate and
  utilization, fraction of ops currently down, observed downtime last week,
  week sin/cos. EXCLUDED: R1/R2 labels, latent regime, true intensities,
  forecasts, future events, retrospective fields.
- Campaign tapes frozen as artifacts: {campaign_id, family: R1|R2|mixed|R3,
  risk level, base seed, realized risk-event tape, horizon, hash}. R3 is
  OOD/stress evaluation ONLY — never in training.

## Reward (training) vs outcomes (evaluation)
- Reward: −[ServiceLoss̃ + BacklogAUC̃ + λ·ExtraShiftHours̃ + 0.10·Switching̃],
  normalizers = calibration p95, frozen. Three λ values piloted; ONE selected
  on calibration only, by a predefined resource-constraint rule.
- ReT is an OUTCOME, never the reward (avoids circularity and the known
  ReT-reward collapse-to-S1 failure).
- Confirmatory outcomes: co-primary = Garrido Excel ReT (`ret_excel`, never
  `order_level_ret_mean`) and service-loss AUC. Key secondary: TTR (contract
  below), backlog AUC, fill rate, extra shift-hours, switches, CVaR10 (not
  CVaR05 at 60-tape scale).
- TTR contract: disruption clusters merge events separated by <1 week; clock
  starts at cluster onset; recovery declared after 2 consecutive weeks with
  fill rate ≥95% of baseline AND backlog ≤105% of baseline; unrecovered
  clusters right-censored and reported as such. The legacy RPj-based
  `ttr_mean` is renamed to avoid collision.

## Arms (all initialized from the SAME θ₀ pretrained on calibration tapes only)
| Arm | Definition | Identifies |
|---|---|---|
| static | best fixed configuration per buffer (calibration-selected) | non-adaptive reference |
| heuristic | observable backlog/fill/utilization rule (calibration-tuned) | deployable benchmark |
| frozen | θ₀, no further updates | adaptive control without new learning |
| persistent_weights (PRIMARY) | keeps actor+critic across campaigns; optimizer re-initialized each campaign | retained knowledge |
| reset_local | returns to θ₀ each campaign; learns within-campaign only | value of prior experience |
| persistent_full | keeps weights + optimizer (ablation) | technical-carrier sensitivity |
| scratch_balanced | from θ₀, same TOTAL steps/updates over the same campaign multiset in balanced order, fresh on-policy rollouts (PPO cannot replay old trajectories) | sequential-trajectory vs more-compute |

Key contrasts: frozen−static (adaptive value), persistent−frozen (continued
learning), persistent−reset_local (retention, PRIMARY), persistent−scratch_balanced
(sequential order vs compute), history_A−history_B (path dependence).

## Hypotheses
- H1 retained learning: persistent_weights − reset_local > 0 on common
  held-out probes (co-primary outcomes).
- H2 learning curve: performance on FIXED update-free probes improves with
  campaign index for persistent, absent/weaker in reset and frozen
  (Persistent×CampaignIndex interaction).
- H3 downside robustness: CVaR10, p95 TTR, Brown–Forsythe dispersion + mean
  non-inferiority (never "lower variance" alone).
- H4 path dependence (non-directional): same campaign multiset in different
  orders (histories A/B/C; same tapes, counts, total intensity, init, steps,
  updates, final probes) produces reproducibly different outcomes on
  identical probes. persistent>reset does NOT establish H4 by itself.
- M1 managerial (separate from H1–H4): persistent vs best-static/heuristic
  under comparable resource use + Pareto frontier. The static frontier is
  disclosed CONTEXT, not the H1–H4 bar.

## Promotion rules (frozen)
Retained-learning claim promoted ONLY if, on virgin confirmatory probes:
1. persistent_weights − reset_local CI95 wholly positive on Excel ReT;
2. service-loss AUC improves ≥5% with CI95;
3. policy is Pareto non-dominated OR holds shift-hours within ±2% of the
   comparator;
4. the effect does NOT depend on retaining optimizer state or normalization
   (ablation check).

## Claim ladder (pre-specified)
| Result | Allowed claim |
|---|---|
| No observable headroom (Gate 2 fails) | decision space does not justify adaptive control (boundary paper) |
| PPO > statics but persistent ≈ reset | adaptive control; NO L(e−1), path dependency, or organizational learning |
| persistent > reset, not > statics | retained learning with limited managerial value |
| persistent > reset AND Pareto vs static/heuristic | retained learning with operational value |
| reproducible order effects (H4) | path dependence |
| beneficial ordering + no backward interference | robust cumulative learning |

## Non-claims (binding)
No predictive accuracy. No real-world organizational learning (computational
analogue only). No KAN/RNN superiority. No claim that the learned policy
dominates every static policy. No "first DES+RL". No prevention/anticipation.

## Failure conditions (any one kills the headline)
1. No deployable adaptive headroom at Gate 2 (branching test).
2. persistent−reset CI includes the practical-relevance threshold δ_min
   (fixed with the PI before any confirmatory look).
3. Gains disappear under equal resource use.
4. Gains explained by optimizer state or normalization.
5. Path-order effects not reproducible (H4 reported as failed).

## Mandatory acceptance tests (before any experiment counts)
buffer immutable within campaign; symmetric one-step shift delay; identical
frozen normalizer across arms; identical physical reset under same seed;
zero-updates ⇒ persistent == reset bitwise; save/load preserves weights (and
optimizer only in persistent_full); prefix replay reproduces observations and
metrics exactly; identical tapes give identical exogenous events across
policies; R3 never appears in training; metrics use ret_excel; a SYNTHETIC
retention scenario (planted learnable regularity) demonstrates the protocol
detects retention when it truly exists.
