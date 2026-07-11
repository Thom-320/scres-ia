# `ret_excel_visible_v1` — frozen metric contract (2026-07-11)

Status: **canonical for new confirmatory work**. Historical artifacts are not
rewritten; their `ret_excel` is relabelled conceptually as full-ledger unless
their runner explicitly used the visible-ledger function.

## Primary endpoint

Apply Garrido's workbook formula without clipping to completed, non-lost rows,
preserve original order index `j`, and reconstruct the time-varying capped `Bt`
and cumulative `Ut` ledgers from the complete order history. Lost/unresolved
orders do not emit ReT rows but remain causal through `Bt`/`Ut`.

Canonical key: `ret_excel` = `ret_excel_visible`.

## Required companions

- `ret_excel_visible_clipped_0_1`: sensitivity to the thesis's stated range.
- `ret_excel_full_ledger`: historical secondary metric that emits every generated
  order and scores unfulfilled orders zero.
- service-loss AUC, lost orders, backlog AUC, fill, CT and mass conservation:
  binding physical guardrails.

## Claim discipline

The un-clipped visible metric preserves workbook lineage even though isolated
recovery observations can exceed one. A result that disappears after clipping
or violates a physical guardrail cannot be a headline. `Rsult_1.xlsx` is a
secondary transformed analysis artifact, not the primary ReT ledger.

## D1 reanalysis

Reuse selection/validation tapes 810001–811030 and the frozen D1 branching
protocol. Do not open seeds 820001–820060 and do not train PPO. The service-loss
stop remains binding independently of the revised ReT estimate.
