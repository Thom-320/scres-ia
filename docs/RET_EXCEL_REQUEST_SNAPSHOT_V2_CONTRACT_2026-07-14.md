# `ret_excel_request_snapshot_v2` — canonical source-aligned contract

Date: 2026-07-14  
Status: **implemented as the frozen researcher-defined primary; Garrido confirmation controls
source-faithfulness language, and a dual same-time sensitivity is mandatory before Program Q**.

The primary per-order formula is unchanged from the Garrido workbooks. The
contract change is the provenance and timing of `Bt` and `Ut`: they are frozen
on each order when request `j` is generated (`OPTj`), before the request is
inserted into the Op9 queue. They are not reconstructed from completion time
`OATj` and later events cannot rewrite them.

The four auditable request fields are:

- `ret_bt_at_request`
- `ret_ut_at_request`
- `ret_ledger_snapshot_time`
- `ret_ledger_event_sequence`

Completed, non-lost rows are emitted in original `j`/`OPTj` order; original `j`
is preserved and the formula is not clipped. Lost and horizon-unresolved orders
do not emit a primary row, but lost orders, backlog, service, quantity ReT,
worst-node outcomes, tail risk and resource conservation remain simultaneous
guardrails.

The provisional deterministic same-time convention is: execute events already
scheduled before the request callback, snapshot `Bt/Ut`, enqueue request `j`,
and record the event sequence. The alternative `snapshot_before_events` convention is available
only for explicit sensitivity rescoring with `force_reconstruct=True`; it cannot silently replace
captured native fields. Garrido/Simulink confirmation determines whether the primary may be
described as original-model semantics rather than a disclosed researcher convention.

## Evidence and claim boundary

The canonical aggregator reproduces all 47,546 workbook formula cells exactly
when source `Bt/Ut` snapshots are injected: zero mismatches and maximum absolute
error `0.0`. This establishes formula and snapshot-interface fidelity. It does
not establish endogenous DES ledger fidelity, adaptive headroom, a comparator
ceiling or a Paper-2 result.

`ret_excel_visible_v1` and every result scored with its OAT-derived ledger are
quarantined as metric-development only. This explicitly includes Program H,
Program J and all local/VPS M/T/R switch-frontier outputs. Passing execution
custody, deep replay or a confidence interval cannot restore them. Restoration
requires a new v2 re-score on the identical eligible tapes, rebuilt
same-contract comparators and all guardrails, followed by a new versioned
verdict.

The governing machine records are:

- `research/paper2_exhaustive_search/metric_governance_audit.json`
- `research/paper2_exhaustive_search/ret_excel_visible_v1_source_semantics_audit_20260714.json`
- `research/paper2_exhaustive_search/ret_excel_request_snapshot_v2_implementation_audit_20260714.json`

Current authorization is therefore fail-closed: no Paper-2 positive, null or
ceiling is confirmed; no learner or Paper 3 work is authorized.
