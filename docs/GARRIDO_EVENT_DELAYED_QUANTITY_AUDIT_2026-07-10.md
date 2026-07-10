# Event-specific counterfactual availability audit (2026-07-10)

## Purpose

The previous material-lineage lane could show which marked lot an order consumed,
but it marked an entire next lot after a disruption. This audit estimates the
quantity whose availability date actually changes because of each event.

For a frozen event calendar, each event is removed one at a time while every
other event, seed, demand stream, policy, and physical contract remains fixed.
At each material node the estimand is

`cumulative availability without event - cumulative availability with event`.

Its positive peak is the counterfactually delayed quantity; its positive area is
delayed unit-hours; its final return to zero is debt recovery. A positive terminal
value means the release was foregone within the horizon rather than merely late.

## Implementation

- Observational arrival streams were added for raw material at WDC, raw material
  at AL, completed Op7 batches, Op8 arrivals at SB, and order releases.
- Recording does not feed any DES decision and therefore cannot change physics.
- `scripts/audit_garrido_event_delayed_quantity.py` first freezes the complete
  replayable calendar. Risk filters select events to audit, not events retained
  in the factual calendar.
- Duration events upstream of Op9 are currently eligible. R14/R24 quantity events
  require a different quantity-counterfactual and are intentionally excluded.

## Cf1 R13 pilot

- Horizon: 20,000 h.
- Complete frozen calendar: 238 upstream duration events.
- First 15 R13 events audited leave-one-out.
- Events with any positive material-availability effect: 4/15.

Examples:

| Event | Node | Peak delayed quantity | Debt recovered |
|---|---|---:|---:|
| R13@840 | WDC raw | 2,280,000 | 4,632 h |
| R13@840 | AL raw | 186,000 | 19,993 h |
| R13@840 | Op7/Op8 rations | 5,000 | 16,541/16,565 h |
| R13@840 | Order release | 2,593 | 19,158 h |
| R13@1008 | WDC raw | 2,280,000 | 3,288 h |
| R13@1008 | AL raw | 186,000 | 2,857 h |
| R13@1008 | Op7/Op8 rations | 10,000 | 1,732/1,756 h |
| R13@1008 | Order release | 5,076 | 2,214 h |

Eleven of the first fifteen recorded R13 events have zero counterfactual material
effect. This directly falsifies the rule “a recorded R13 event marks the whole
next lot.” Event occurrence, temporal overlap, lot consumption, and material
causation are different quantities.

## Decision

The blocker is closed at the measurement level: the environment can now calculate
event-specific delayed quantity and recovery time without widening attribution
windows or tuning constants. It is not yet promoted into ReT. Before using it for
order attribution, the audit must be run on the predeclared odd-Cf calibration
set and delayed quantities must be matched to FIFO consumption opportunities.
Even Cf validation cases remain untouched.

The result is an in-model paired counterfactual, not evidence of a real-world
causal effect and not proof that Garrido's workbook columns use this semantics.

Primary local artifact:
`outputs/audits/garrido_event_delayed_quantity_r13/verdict.json`.
