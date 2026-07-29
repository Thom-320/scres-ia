# Garrido causal exposure contract (2026-07-10)

## Purpose

Replace retrospective event-window overlap with a falsifiable event-to-order
ledger. This contract governs only the opt-in `causal_exposure` attribution
lane. It must not change the physical DES, the frozen Track A/B defaults, or
the workbook-tape forensic lane.

## Non-negotiable semantics

An order may carry risk `Rcr` only if the event caused at least one recorded
blocking interval on the order's actual service path. Temporal coexistence is
not sufficient.

For a completed delayed order:

`RPj = OATj - min(causal_event.start_time)`

where the minimum is taken only over causal events. `DPj = CTj`. If no causal
event exists, the order uses the no-risk workbook branch even if unrelated
events overlap `[OPTj, OATj]`.

## Required ledger records

Each causal record must expose:

- stable event identifier;
- risk identifier;
- event start and end;
- affected operation;
- order `j`;
- blocking reason;
- blocking interval start/end and duration;
- propagation source, if inherited through queue or material scarcity;
- direct versus propagated attribution.

Allowed blocking reasons:

- `operation_down`: the order or its material cannot advance because an
  affected operation is down;
- `route_down`: Op10/Op12 transport or Op11 distribution is physically blocked;
- `risk_induced_stockout`: an upstream event delayed material that the order
  requires;
- `priority_propagation`: a causally affected earlier/contingent order occupies
  the release opportunity;
- `r24_pressure`: unresolved contingent quantity or its induced backlog.

Disallowed reasons:

- `time_overlap` alone;
- a fixed post-event window;
- any risk whose affected operations are outside the blocking path;
- queue waiting caused only by the normal daily headway.

## R24 lifecycle

Each R24 event opens a pressure episode with:

- event id and start time;
- surge quantity;
- baseline pending-order count and quantity immediately before the surge;
- the contingent order(s) consuming that surge;
- outstanding surge quantity;
- induced backlog above baseline.

The episode remains open while either:

1. surge quantity remains unfulfilled; or
2. the causally induced backlog remains above its pre-event baseline.

It closes at the first timestamp where both conditions are false. No fixed
168-hour expiry is allowed. An order receives R24 only if its own waiting or
service interval intersects an open pressure episode and the episode either
contains that contingent order or occupies a release/stock resource that would
otherwise serve it.

Overlapping R24 episodes must retain separate ids. Closing one must not close
another.

## R11-R23 lifecycle

Duration events are registered when they fire, not only when recovery ends.
Operation-down state must retain the responsible event ids. A process that
polls an affected operation records the exact blocked interval and responsible
ids.

For upstream operations without an order object in flight, causal influence
must propagate through an explicit scarcity/debt record. It may not be inferred
solely because an old order window overlaps a later event. If material lineage
cannot be demonstrated, label the R1/R21 attribution incomplete rather than
fall back to overlap.

## Mandatory falsification tests

1. **Unrelated overlap:** an R23 event overlaps a long order but the order never
   waits at Op11. Expected: no R23 attribution.
2. **Direct route block:** an order waits while Op11 is down due to R23.
   Expected: R23 present; event id and wait duration match.
3. **Wrong operation:** an R22 event affects Op4 while an order is already
   downstream with available stock. Expected: no direct R22 attribution.
4. **R24 immediate resolution:** a surge is fulfilled without inducing backlog.
   Expected: episode closes with the contingent order; later orders unmarked.
5. **R24 propagated backlog:** a surge occupies release capacity and delays later
   orders. Expected: those delayed orders inherit the episode until recovery to
   baseline.
6. **Overlapping R24:** two episodes overlap. Expected: independent closure and
   references.
7. **Mode identity:** with `des_events`, all frozen baseline metrics remain
   bitwise identical.
8. **Negative-control removal:** removing an attributed event from an otherwise
   identical deterministic tape must weakly improve the attributed order's OAT;
   removing an unattributed event must not.

## Calibration and promotion

Development and selection use odd CFs only. Compare:

- current `des_events` overlap lane;
- fixed-168-hour R24 screen;
- causal exposure lane.

Primary promotion endpoints:

- mean absolute risk-active-share gap;
- per-risk share gap for R11-R14 and R21-R24;
- RP p50/p95 log error by family;
- ReT mean absolute gap.

Guardrails:

- CT, warm-up, placed, visible, lost, and conservation must remain unchanged
  within numerical tolerance because attribution must not change physics;
- no risk-family share may improve by relabelling unrelated orders;
- R24 must close endogenously in every finite smoke scenario.

Promote to the even-CF gate only if odd-CF causal exposure improves aggregate
risk-share and RP errors over both comparators, does not worsen either R1 or R2
family materially, and passes every falsification test. Execute the even-CF
gate once after freezing the implementation.

