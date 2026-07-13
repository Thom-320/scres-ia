# Action-trajectory audit

Date: 2026-07-13

## K3 decisive audit

The K3 learner emits one tape-independent eight-week action sequence on every evaluated tape:

`(1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.0, 0.0) * D0`.

- unique PPO seed-0 test sequences: `1`;
- PPO minus fixed replacement ReT: exactly `[0,0,0]`;
- fixed minus MPC resource delta (`ordered_D0`): exactly `[0,0,0]`;
- fixed minus MPC learner-test ReT: `+0.017708`, CI95 `[+0.010417,+0.026042]`.

Result: open-loop schedule discovery, not feedback. `RETRACT_K3_ADAPTIVE_AND_NEURAL_CLAIMS_STATIC_PERIOD8_CONFOUND` is the effective verdict.

## Integrated M/T/R team corrective audit

The original confirmation stored no action trajectories in its verdict. The corrective audit re-ran already-burned calibration/locked tapes and retained them long enough to compute replacements.

- locked signal-policy unique active sequences: `120/120`;
- calibration complete-sequence mode frequency: `1/60`;
- calibration modal sequence: `MMTTMMRRRRRRTTMMTTRRTTRR`;
- calibration per-week calendar: `MRTTMRRRTTTRRMMMMMMRRTRR`;
- calibration phase-only actions: `M`, `T`, `R`.

Locked replacement contrasts versus constant M:

| Policy | Delta canonical ReT, mean [CI95] | Service-loss reduction, mean [CI95] |
|---|---:|---:|
| signal adaptive | `-0.001309 [-0.006384,+0.003093]` | `-0.03035 [-0.03877,-0.02246]` |
| calibration modal complete sequence | `-0.004274 [-0.008721,-0.000506]` | `-0.04330 [-0.05225,-0.03454]` |
| calibration per-week calendar | `-0.003685 [-0.008698,+0.001149]` | `-0.04838 [-0.05755,-0.03992]` |
| calibration phase-only | `-0.001894 [-0.004597,+0.000796]` | `-0.03346 [-0.03968,-0.02774]` |
| feasible next-context clairvoyant diagnostic | `+0.002336 [-0.000993,+0.005841]` | `-0.02390 [-0.03297,-0.01562]` |

The 120 unique sequences demonstrate behavioral dependence, but neither signal policy nor its fixed/phase replacements establishes positive adaptive value. The feasible clairvoyant policy is a lower bound, not an oracle ceiling, and worsens service loss.

### Exact effect-quotient collision

On burned locked tape `1110001`, calendars `MTTMMMMMMMMMMMMMMMMMMMMM` and `MRRMMMMMMMMMMMMMMMMMMMMM` differ in weeks 1–2, which contain only R11 events. Their selected canonical, guardrail, mass, reserve and exogenous-hash digest is bit-identical; only allocation-destination hours differ. This validates one collision of the exact tape-effect quotient. Across all 180 calibration/locked tapes, subset determinization counts 88,684,583 distinct effect executions instead of 2,013,265,980 calendar-tape runs. No quotient table outcomes, frontier or PI ceiling have yet been computed.

### Exact signal-mapping screen

All `3^3=27` deterministic mappings from the observed equipment/interdiction/mission signal label to M/T/R were evaluated on the 60 burned calibration tapes. The calibration winner is `MMM`, exactly constant M. On the 120 burned locked tapes its delta versus constant M is identically zero for ReT, service loss and lost orders. This retires the complete memoryless label-to-action mapping class; it remains a tested-policy-class null, not the maximized H_obs or a full-calendar ceiling.

The null physics cell (`r11_factors=1`, `transport_factors=1`, `reserve_issue=0`) produced exact equality of canonical ReT, quantity ReT, CVaR05, service AUC, lost orders, and CRN hashes for all three constants and the signal policy on three disposable tapes. It is retrospective and therefore not a confirmatory null, but it verifies implementation of the falsification mechanism.

## Other learner routes

- Program L: no action-trajectory certificate exists. The development runner did not persist complete sequences or evaluate a modal/best fixed full-horizon replacement; variation across tapes is not proof of feedback value. Its family-level claim remains blocked.

- Track B: full-contract constant beats canonical PPO, so a feedback certificate cannot rescue the primary claim.
- Program E: PPO is a validation null and does not convert DRA-2b EVPI.
- Program J: PPO beats its static comparator in `0/6` seeds and the observable gate is negative.
- Paper 3: no trajectory/retention audit is authorized because no learned Paper 2 value exists.
