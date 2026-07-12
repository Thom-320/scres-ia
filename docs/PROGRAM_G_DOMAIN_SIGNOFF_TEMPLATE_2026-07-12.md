# Program G — Garrido domain sign-off template (2026-07-12)

> **V1.2 UPDATE (decisive items first).** After verifying that convoy 5000/48h = 2500 rations/day ≈
> thesis daily demand (2400–2600) and that S2 (5128/day) cannot be moved by a 2500/day convoy, three
> questions now gate the build ABOVE all others:
>
> - **[ ] Convoy ontology (THE decisive question).** Is the shared convoy the *primary scarce
>   downstream transport* SB→CSSU at thesis scale (**Option A**, physics-favored), or an *extra
>   emergency asset on top of a complete normal flow* (**Option B**, non-binding by the throughput
>   math)? **A / B / OTHER:** ____
> - **[ ] Dead shift dimension.** Do you accept that with transport binding at 2500/day, dynamic S1/S2
>   control does not change final delivery and is therefore removed from the primary action space
>   (production fixed at S1)? **CONFIRM / REJECT:** ____
> - **[ ] Surge magnitude.** Is a *sustained* 4–8 week surge realistically **1.25×–1.50×** routine
>   demand (v1.2), rather than 2×–3× (which suits a one-day spike, not a two-month regime)? **____**

Status: **UNSIGNED. G0 build stays blocked until this is returned with defensible ranges.**
Editing authorization is not enough; Program G needs **defensible ranges**, not invented ones. This
template lists exactly what must be confirmed. Each item: Garrido writes CONFIRMED / ADJUST(→value) /
REJECT, with a one-line operational justification. Anything left REJECT is removed from v1.1, not
worked around.

## 0. Overall framing
- [ ] Program G is a *stylized emergency-reserve-overlay extension* of the thesis MFSC, not a claimed
  reproduction of two-theatre transport doctrine. **CONFIRM / ADJUST / REJECT:** ____

## 1. Spatial structure
- [ ] Two competing forward destinations (CSSU-A, CSSU-B) with separate inventory, backlog, demand
  and route is operationally meaningful. **____**
- [ ] Realistic number of simultaneously-competing destinations (is 2 right, or more?). **____**

## 2. Shared lift (convoy)
- [ ] A single finite lift shared between A and B, where committing it to one denies the other, is
  realistic. **____**
- [ ] Outbound 24 h + return 24 h, re-decidable every 48 h when available. **____**
- [ ] One indivisible 5,000-ration load (batch anchor), no partial loads. **____**

## 3. Emergency reserve
- [ ] 10,000-ration reserve as two 5,000 loads, held at SB, no in-episode replenishment, over 56
  days is a defensible stress framing. **____**
- [ ] Delivered reserve remains and is consumed at the destination. **____**

## 4. R22 semantics
- [ ] R22 as a bidirectional LOC pause (no vehicle/cargo destruction) for v1 is acceptable; a
  destruction variant is deferred. **____**

## 5. Operational tempo (demand)
- [ ] Latent tempo regimes low / routine / surge, persisting ~4–6 weeks, are realistic. **____**
- [ ] Surge demand multiplier range 2×–3× routine. **____**
- [ ] Tempo shifts the A/B split while total demand is conserved (primary study). **____**
- [ ] Realistic A/B surge **commonality** — how often do both theatres surge together within a
  1–2 week window? (localized 0.10 vs partially-common 0.35, or another value) **____**

## 6. Advance information — THE decisive item
For each candidate signal, confirm it exists, its realistic **lead**, **sensitivity** and
**false-positive rate** (NOT a scalar accuracy):
- [ ] `planned_troop_level` — exists? lead ____ days; sensitivity ____; FPR ____
- [ ] `deployment_calendar_pressure` — exists? lead ____ days; sensitivity ____; FPR ____
- [ ] `route_threat_score` — exists? lead ____ days; sensitivity ____; FPR ____
- [ ] Do the two proposed tiers (moderate 0.70/0.20, high 0.85/0.10) bracket reality? **____**
- [ ] Does a real alert identify WHICH theatre (A vs B), or only that "a surge is coming"? **____**
- [ ] Are signal errors temporally correlated (a bad week stays bad)? **____**

## 7. Costs / limits that forbid "max everything"
- [ ] What physical cost or constraint prevents always maximizing reserve at both theatres
  (holding, expiry, lift contention)? **____**
- [ ] Is a reserve holding or expiry cost realistic for the TRSC arm? **____**

## 8. Falsification comfort
- [ ] Garrido accepts that if no realistic cell shows observable adaptive headroom, Program G STOPs
  and is published as a boundary result (no parameter is tuned until it passes). **____**

---

### Decision line
G0 build (tapes 980001+) is authorized only when items 1–7 are CONFIRMED or ADJUSTED to defensible
values, recorded back into `contracts/program_g_domain_envelope_v1_1.json` and the intervention
ledger, and item 8 is accepted. Signed: __________________  Date: __________
