# Program D DRA-2b long-horizon final verdict

Status: **CLOSED — `STOP_DRA2B_PRE_TREE_GATE`**.

DRA-2b was preregistered and tagged before generating any of its tapes. It is a
new exploratory study and does not reopen or revise the terminal DRA-2 result.

## Executed design

- 60 new calibration tapes, seeds 860001–860060, 15 per disruption family.
- 120 states, two per tape, with rotating policy-prefixes.
- Exact 14-day canonical open-loop search: 987 sequences per state.
- Exact 10-day sensitivity: 144 sequences on 12 states, three per family.
- Outcomes evaluated at 28 and 56 days.
- Nine same-contract static continuations from every state.
- Resource-dominance comparator selected once on calibration.
- No DRA-2 tapes reused for estimation or fitting.
- No holdout or virgin tapes opened; no tree or PPO fitted.

## Passed gates

| Gate | Result |
|---|---|
| Strong liveness | 100% — PASS |
| HOLD diversity | 65/120 = 54.2% — PASS |
| DISPATCH diversity | 40/120 = 33.3% — PASS |
| Tie/zero | 15/120 = 12.5% |
| Resource envelope | PASS |
| ReT improvement | +0.02212, CI95 [+0.01725, +0.02734] — PASS |
| Lost-order change | 0.0 — PASS |

The resource-dominance static comparator was `threshold_5000__wait_72h`:

| Resource | Long-horizon oracle | Static comparator |
|---|---:|---:|
| Departures | 24.917 | 24.483 |
| Unavailable hours | 1193.18 | 1169.53 |

The comparator remains inside the candidate's joint resource envelope.

## Failed gates

### Service magnitude

Service-loss reduction was 3.075%, CI95 [1.987%, 4.033%]. Its direction is
favorable and the CI excludes zero, but it does not reach the preregistered 5%
practical threshold.

### Horizon sufficiency

First-action agreement between 10 and 14 days was 100%, and neither service
direction nor action direction reversed. However, normalized headroom remained
horizon-sensitive:

- relative-change threshold passed in 5/12 sensitivity states;
- absolute-change threshold passed in 2/12 states;
- maximum relative change: 81.95%;
- maximum absolute change: 0.003487 ReT per decision-day.

Therefore the expanded horizon did not solve the specific instability that
motivated DRA-2b.

## Binding conclusion

```text
observable_tree_authorized = false
holdout_opened = false
ppo_authorized = false
ppo_trained = false
virgin_tapes_opened = 0
```

Allowed claim:

> Across independent tapes, finite-convoy timing repeatedly exposes material
> resource-adjusted ReT headroom and stable first-action direction. Nevertheless,
> its estimated magnitude continues to grow with the planning window and its
> service-loss improvement does not meet the preregistered practical threshold.
> The evidence therefore does not establish deployable observable control or
> justify PPO.

DRA-2b should not be extended again by selecting a still longer horizon from the
same decision family. Any future sequential-control study requires a different
estimand or physical decision family and a new paper-level justification.

Machine verdict:
`results/program_d/dra2b_long_horizon_calibration/verdict.json`.
