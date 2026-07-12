# Garrido material-lineage gate verdict (2026-07-10)

## Decision

**Implemented, physically neutral, and not promoted.** The opt-in
`material_lineage_mode="tagged_lots"` transports FIFO quantity slices and stable
risk-event references through Op2 output, WDC stock, Op3/Op4 transfer, assembly,
Op7 batches, Op8 transport, SB stock, and order release. It does not alter the
frozen lanes or the canonical `garrido_reference_v2` gate.

The mechanism deliberately records two different facts:

1. `consumed_material_lineage`: the order consumed units from a lot carrying a
   prior risk reference. This is descriptive provenance, not proof that the risk
   delayed the order.
2. `lineage_shortage_refs`: the order was physically unable to release while the
   corresponding upstream debt was still open. This is the stricter candidate,
   but it is not by itself counterfactual proof that the debt caused the shortage.

## Falsification design

- Split: odd Cf calibration cases only (Cf1, 3, ..., 19).
- Even Cf validation cases were not inspected or modified.
- Four arms used identical seeds and physical contracts:
  raw overlap, fixed R24 168 h, prior order-causal ledger, and material lineage.
- Promotion required material lineage to improve mean absolute ReT gap,
  per-risk active-share gap, and per-risk RP95 log error over both comparators,
  with zero change to physical CT.

## Result

| Arm | ReT gap | Per-risk share gap | Per-risk RP95 log error | Physical CT delta |
|---|---:|---:|---:|---:|
| Raw overlap | 0.069955 | 0.146908 | 1.551361 | 0 |
| Fixed R24 168 h | **0.025844** | **0.120695** | 1.519918 | 0 |
| Prior causal order ledger | 0.101258 | 0.189001 | **0.628660** | 0 |
| Material lineage | 0.099962 | 0.193274 | 0.961354 | 0 |

Across the ten odd Cf cases, 13,964 visible orders consumed at least one marked
material slice, whereas 4,822 orders experienced a stockout while a matching
upstream debt remained open. Treating the former as affected would therefore
turn provenance into label expansion. The promotion rule fails.

## Interpretation

The implementation closes the software blocker: material identifiers now reach
the order rather than disappearing at inventory aggregation. It also exposes a
more precise scientific blocker. Attaching a completed disruption to the whole
next supplier or production lot is too coarse: the marked quantity is not yet a
counterfactual estimate of the quantity whose availability time changed because
of that event. The genealogy can answer *which marked lot was consumed*, but it
cannot yet answer *which units would have arrived earlier absent the event*.

Therefore:

- do not promote material consumption as a ReT risk indicator;
- do not replace the bounded R24 arm in the current reference gate;
- retain `tagged_lots` as an audit-only lane;
- the next valid refinement, if pursued, is event-specific delayed-quantity
  accounting at each release clock, not a wider exposure window or another
  attribution constant.

## Verification

- 93 focal, thesis-lane, Track-B, interface, and preflight tests passed.
- Cf1 lineage off/on: placed orders, visible orders, CT p50, and CT p95 were
  identical; only attribution changed.
- FIFO split, mixed-lot conservation, BOM conversion, stage routing, debt
  closure, unrelated-inventory negative control, and lineage-off no-op have
  dedicated tests in `tests/test_material_lineage.py`.

Primary artifact:
`outputs/audits/garrido_causal_attribution_odd_cfs/verdict.json`.
