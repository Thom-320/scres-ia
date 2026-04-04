# Track B Critique Audit

Date: 2026-04-03

## Bottom line

The critique is directionally right that Track B creates much more adaptive headroom
than Track A, but it overstates the result by claiming that the learned policy
simply "puts everything at the maximum."

The new canonical audit shows:

- PPO does **not** keep downstream controls at `d=2.0` all the time.
- PPO beats every expanded static baseline under the common audit contract.
- The strongest static comparator is now `s2_d1.50`, not `s3_d2.00`.
- Track B remains a **research extension**, not a thesis benchmark.

## New evidence

Source bundle:

- `/Users/thom/Desktop/Universidad_Codigo/proyecto_grarrido_scres+ia/outputs/track_b_benchmarks/track_b_all_reward_audit_20260403T203800Z`

### Learned downstream control usage

From `paper_control_table.csv` under the canonical Track B audit:

- PPO `Op10` downstream multiplier mean: `1.355`
- PPO `Op12` downstream multiplier mean: `1.501`
- PPO `% steps with Op10 >= 1.90`: `27.7%`
- PPO `% steps with Op12 >= 1.90`: `40.5%`
- PPO `% steps with both Op10 and Op12 >= 1.90`: `16.8%`

Interpretation:

- PPO uses high downstream dispatch at times, but it is **not** a pure
  "always maxed" policy.
- RecurrentPPO is even less aggressive on downstream dispatch and still ends up
  nearly tied with PPO on outcomes.

### Expanded static DOE

The static comparison now includes:

- `s1_d1.00`
- `s1_d1.50`
- `s1_d2.00`
- `s2_d1.00`
- `s2_d1.50`
- `s2_d2.00`
- `s3_d1.00`
- `s3_d1.50`
- `s3_d2.00`

Under the reference audit lane (`ReT_seq_v1`):

- PPO: `fill=1.0000`, `order-level ReT=0.9504`
- RecurrentPPO: `fill=1.0000`, `order-level ReT=0.9488`
- Best static by fill: `s2_d1.50` with `fill=0.9896`, `order-level ReT=0.4787`
- `s3_d2.00` is no longer the strongest static comparator

Interpretation:

- The static frontier moved upward once intermediate downstream settings were added.
- Even against that wider DOE, PPO still dominates by a large margin.

## What this means for the paper

### Claims supported

- Track B improves materially over static policies once the action space touches
  the active downstream bottleneck.
- Multiple trainable resilience rewards converge to a strong policy region.
- PPO does not need recurrence to reach that region.

### Claims not yet supported

- "The learned policy is trivial because it always maxes the controls."
- "Track B is just the thesis benchmark solved by brute force."

## Recommended manuscript framing

Use this wording:

> Track B is a minimal benchmark extension that grants the agent direct control
> over the active downstream bottleneck. The resulting PPO policy does not merely
> saturate all controls; instead, it uses selective downstream intensification and
> a predominantly low-shift operating mix, while still outperforming an expanded
> static DOE.

## Next hardening steps

1. Add a dynamic heuristic baseline, not only static baselines.
2. Evaluate the learned policy under harder or out-of-distribution risk profiles.
3. Keep Track A and Track B explicitly separated in the paper:
   - Track A: thesis-faithful negative benchmark
   - Track B: minimal adaptive-control extension
