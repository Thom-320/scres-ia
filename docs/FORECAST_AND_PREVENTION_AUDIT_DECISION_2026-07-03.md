# Forecast and Prevention Audit Decision -- 2026-07-03

## Paper framing

If the no-forecast Track B 8D run performs nearly as well as the forecast-enabled
run, use the no-forecast variant as the more defensible main paper lane.

Rationale:

- It avoids the privileged-information critique.
- It keeps the result focused on resilience and observable supply-chain state.
- It leaves forecast as an explicit early-warning probe, not as a hidden crutch.

The forecast-enabled model should be retained as a prevention probe only. It
earns a headline role only if the prevention audit shows that forecast-driven
pre-risk actions improve Garrido Excel ReT before the risk materializes.

## Audit correction

The preventive/reactive audit should not use one fixed reset action
(`s2_d1.50`) for every policy. That confounds the reset:

- For a policy whose calm behavior is S1, resetting to S2 changes the normal
  operating mode, not just the intervention being tested.
- For a policy whose alert behavior is already close to S2 and 1.5x dispatch,
  the reset may erase almost nothing.

Use each policy's own calm/baseline action as the reset reference.

## Classification design

Do not rely on one event anchor for all mechanisms.

- Prevention should be anchored to forecast/regime warning and measured in a
  pre-risk window.
- Reaction should be anchored to realized service or congestion deterioration:
  backlog growth, rolling fill-rate drop, Op10/Op12 queue pressure, or similar.
- Recovery should be measured by the ReT value of restoring actions after the
  service/congestion signal appears.

Keep Garrido Excel ReT as the value metric. The question is not whether a signal
changes an auxiliary metric; it is whether the action window improves ReT.

## Current status

At the time of this note, the local no-forecast confirmatory run was still in
progress:

`outputs/experiments/track_b_e2_no_forecast_confirm_2026-07-03/`

Do not report the no-forecast result as final until `summary.json` is verified.
