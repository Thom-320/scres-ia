# Program L — corrective audit of alternate-route recourse

**Date:** 2026-07-13
**Corrected status:** `STOP_PROGRAM_L_HEURISTIC_CONTRACT_ONLY__ROUTE_FAMILY_OPEN_PENDING_DOMAIN_FACTS_AND_CERTIFIED_BOUND`
**Learner trained:** none. **Virgin tapes opened:** none. **Paper 3 authorized:** no.

## What the full-DES development screen actually establishes

The implementation adds a disclosed alternate Op8 route to the canonical finite-convoy DES. The full-DES runner uses the repository `compute_episode_metrics` path and therefore evaluates canonical order-level ReT. Focused tests verify that route recourse disabled is bitwise identical to Program E on two tapes and that an alternate-route trip bypasses Op8 without creating mass.

On 40 development tapes per cell, the frozen signal heuristic failed to beat the small comparator set reliably. Its two positive mean deltas were below 0.01, their confidence intervals crossed zero, and they used more departures:

| Controlled R22 cell | Mean ΔReT | LCB95 | Mean extra departures |
|---|---:|---:|---:|
| 6 × 120 h | +0.00316 | -0.00967 | +2.40 |
| 8 × 72 h | +0.00402 | -0.00799 | +1.38 |

This is a valid no-go for that heuristic and those cells. It is not a closure of the route-recourse family.

## Why the earlier terminal interpretation is retracted

1. The quantity formerly labelled `H_PI` came from a myopic true-state routing rule with a fixed state-responsive dispatch trigger. It did not optimize the full trajectory. Negative values in two cells prove it was not a perfect-information upper bound.
2. The comparator set contained only route 1, route 2, and an alternating route rule. These policies shared a state-responsive dispatch trigger, so they were not a complete full-horizon open-loop frontier. MPC, DP/belief policy, hysteresis, and exact or bounded calendars were absent.
3. The stylized headline cell explicitly has `resource_ok: false`; the report's former statement of equal resources was incorrect.
4. The generated controlled tapes replaced `risk_events` without refreshing their stored hash. The runner now recomputes the hash, but the v1 JSON is retrospective development evidence and cannot be promoted.
5. No Program L preregistration, immutable commit/contract hashes, seed/command manifest, complete per-tape ledgers, action-trajectory audit, modal-calendar replacement, or confirmatory split exists.
6. The “downstream buffer intercept” explanation was not established by an ablation or certified bound. It remains a hypothesis, not a structural theorem.

Therefore the modification to `results/paper2_search/boundary_certificate.json` that says the terminal boundary was “reaffirmed” by Program L is not supported by these artifacts.

## Correct scientific state

- **Implemented heuristic contract:** falsified as promotable.
- **Thesis-native route recourse:** exact null because the thesis-native transition kernel has no alternate-route action; its route-choice liveness, H_PI, and H_obs are zero.
- **Researcher-introduced alternate-route extension:** unresolved and blocked on domain facts. It cannot enter the Garrido-defensible numerical envelope until its route, fleet, signal, and degradation semantics are face-validated.
- **Family-level quantitative closure after validation:** still requires a resource-restricted full-horizon oracle or certified upper bound, complete same-contract open-loop/classical comparators, componentwise resource matching, canonical guardrails, and the mandatory action-trajectory replacement audit.

## Exact Garrido question

Does the MFSC operator actually choose among at least two routes from Op8 toward the same downstream demand, using one finite shared fleet? If yes, what are each route's payload, outbound/return time distributions, R22 exposure, booking or departure commitment, cancellation/reassignment rules, degradation persistence, and the timestamped warning available before dispatch? Are the current experimental values—36 h each way, +24 h when degraded, persistence 0.85, degraded prevalence 0.25, and signal accuracy 0.85—operationally plausible?

## Evidence

- `supply_chain/program_l_route_recourse_env.py`
- `research/paper2_exhaustive_search/program_l_full_des_gate.py`
- `results/paper2_search/program_l_full_des_gate.json` (v1 retrospective development output; labels corrected by this audit)
- `results/paper2_search/program_l_route_recourse_screen.json` (stylized, resource-inferior headline)
- `tests/test_program_l_route_recourse.py`
