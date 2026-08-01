# Preregistro de métrica — `service_first_resilience_v1`

**Status:** `DESIGN_FROZEN_NOT_RUN`
**Purpose:** prevent an abandonment policy from winning the extended DES
before the expedition or the corrected Garrido configuration search is
interpreted.

## Why a new endpoint is needed

The audit in `docs/RESULTADO_RET_PREMIA_EL_ABANDONO_2026-07-31.md` found that
visible ReT, full-ledger ReT and clipped ReT can all prefer a policy that serves
roughly half of the demand. Clipping the tail is therefore not a sufficient
repair. Cobb–Douglas remains a labelled secondary factory-level construct; it
does not replace the order/service endpoint.

## Frozen estimand

`service_first_resilience_v1` is an ordered four-component endpoint, not a
weighted scalar:

```text
(
  no_lost_orders,
  flow_fill_rate,
  -backorder_qty_final,
  ret_excel_visible_clipped_0_1
)
```

The components are compared lexicographically, in that order. Thus any policy
with one or more lost orders loses to a no-loss policy, regardless of its ReT.
If the abandonment status is tied, service quantity wins; then final queue;
only a remaining tie is resolved by clipped visible ReT. All four components
must be reported. No weighted sum, hidden service penalty or post-hoc floor is
allowed under this contract.

## Acceptance and claim boundary

The negative control is fixed before execution: the abandonment audit's extreme
allocation must lose to the balanced/service allocation under this key even if
its clipped ReT is higher. If that falsifier fails, the endpoint implementation
is invalid and no expedition or learner result may use it.

For headroom experiments, the endpoint is adjudicated componentwise. A timing
policy can be called service-compatible only if it does not increase lost
orders, does not decrease flow fill, and does not worsen final backorder under
the paired comparison. ReT improvements outside that boundary are descriptive
and cannot authorize MLP or PPO.

This endpoint is an operational extension. It is not claimed to be Garrido's
thesis ReT, and it does not repair or relabel the thesis metric.

## Required artifacts

- implementation: `supply_chain/service_first_metric.py`;
- unit falsifiers: `tests/test_service_first_metric.py`;
- expedition runner must report the four components per arm, budget and seed;
- `ret_excel*` and Cobb–Douglas remain separate sensitivity columns;
- no MLP/PPO authorization before the endpoint falsifier and the relevant DES
  liveness gates pass.
