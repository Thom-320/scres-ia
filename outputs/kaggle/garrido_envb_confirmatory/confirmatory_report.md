# Garrido Env-B Confirmatory Report

Generated: 2026-06-27T14:10:23.366638+00:00

## Decision

```json
{
  "complete_win_labels": [],
  "partial_win_a_labels": [],
  "partial_win_b_labels": [],
  "s3_pareto_labels": [
    "envb_aggr_g24_raw_ppo",
    "envb_aggr_g24_raw_recurrent",
    "envb_cons_control_v2_ppo"
  ]
}
```

## Target Summary

| label | target | Excel Δ | Excel CI95 | CD Δ | Resource Δ | strict | pareto |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| envb_aggr_g24_raw_ppo | frozen_efficient | +0.000004 | [-0.000032, +0.000039] | +0.001778 | +5854874 | False | False |
| envb_aggr_g24_raw_ppo | static_S3_I1344 | -0.000008 | [-0.000040, +0.000023] | -0.012199 | -11068942 | True | True |
| envb_aggr_g24_raw_ppo | original_S1_I0 | +0.000341 | [+0.000316, +0.000365] | +0.234610 | +8280050 | False | False |
| envb_aggr_g24_raw_recurrent | frozen_efficient | -0.000069 | [-0.000118, -0.000020] | +0.001626 | +3700343 | False | False |
| envb_aggr_g24_raw_recurrent | static_S3_I1344 | -0.000081 | [-0.000126, -0.000036] | -0.012351 | -13223473 | True | True |
| envb_aggr_g24_raw_recurrent | original_S1_I0 | +0.000268 | [+0.000225, +0.000311] | +0.234458 | +6125519 | False | False |
| envb_cons_control_v2_ppo | frozen_efficient | -0.000171 | [-0.000312, -0.000030] | -0.001890 | +752116 | False | False |
| envb_cons_control_v2_ppo | static_S3_I1344 | -0.000330 | [-0.000479, -0.000180] | +0.044099 | -16171700 | False | True |
| envb_cons_control_v2_ppo | original_S1_I0 | +0.001086 | [+0.000949, +0.001222] | +0.056817 | +3177292 | False | False |

## All-Static Checks

| label | beats all Excel | beats all CD | resource lower than all | min Excel Δ | min CD Δ | max resource Δ |
| --- | --- | --- | --- | ---: | ---: | ---: |
| envb_aggr_g24_raw_ppo | False | False | False | -0.000008 | -0.012199 | +8280050 |
| envb_aggr_g24_raw_recurrent | False | False | False | -0.000081 | -0.012351 | +6125519 |
| envb_cons_control_v2_ppo | False | False | False | -0.000331 | -0.007513 | +3177292 |
