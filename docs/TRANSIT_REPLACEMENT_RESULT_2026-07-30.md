# The transit replacement gives dispersion — and does not make autotomy meaningful

**Status:** `DEVELOPMENT_NEGATIVE_NO_CONSTANT_CHANGED`. `on_hand_transit_mode` and
`op11_handling_hours` added to `MFSCSimulation`; both default to the historical behaviour
and `modelled_legs` with `op11_handling_hours = 0` is byte-identical to it.

## What was implemented

The flat constant is replaced by the legs it was standing in for:

    transit = _pt("op10_pt") + _pt("op12_pt") + U(0, op11_handling_hours)
            = 24 + 24 + handling

The handling term is thesis-sourced rather than invented: Op11 (CSSU receipt and
distribution) carries `PT = 0` in our model, while §6.3.3 describes it as *"in less than
1 hour"*. That is a bounded positive quantity we dropped, and it is the only continuous
component the thesis offers on this path.

## It does produce dispersion

`CTj` goes from **43 distinct values to 193**, and it is no longer a point mass. The
acceptance test from the previous document — "wrong if the share stays at 0 or 1 regardless
of parameters" — is passed.

## And the share is still not a measurement

| `op11_handling_hours` | autotomy share | `tol/h` | reference |
|---:|---:|---:|---:|
| 0.02 | 0.520 | 1.000 | 0.00436 |
| 0.05 | 0.498 | 0.960 | 0.00436 |
| 0.20 | 0.124 | 0.240 | 0.00436 |
| 1.00 | 0.029 | 0.048 | 0.00436 |

The share tracks `risk_fraction × min(1, tol/h)` — the probability that a uniform draw
lands under the tolerance, scaled by how many orders a risk touched. **Both `tol` and `h`
are parameters we declare.** The autotomy share has become tunable, which is not the same
as becoming right: it now carries information about our two choices and none about the
supply chain.

Reaching 0.00436 would need `h ≈ 5.7 h`, which **violates the thesis's own "less than
1 hour"** by nearly sixfold. So the one value that hits the target is also the one the
thesis excludes.

**`op11_handling_hours` is therefore not tuned, and must not be.** Fitting a free parameter
until a moment matches is exactly the procedure that produced `delay = 54`, and this
document exists so that the next reader does not repeat it with a new knob.

## What the data says is actually needed

In Garrido's model the tolerance is not a parameter at all. His classification is
`CTj = LTj`, and his on-schedule orders simply *are* at 48.00744–48.048 — the excess is
what his simulation produces for an order that met its window, not a band someone chose.
His 0.44% is then the fraction of orders that were **touched by a risk and still met the
window**, which is a property of how often a disruption fails to propagate.

Ours has no such property: in our model a risk that touches an order always consumes real
time on that order's path. So the share cannot emerge; it can only be imposed.

Closing this properly means giving risks a way to not propagate — the tail-autotomy
behaviour of his Figure 5.2, where the chain sheds the affected part and the remainder
keeps moving. That is a larger change than a transit term, it needs its own preregistration,
and this document is the baseline it would be measured against.

## What is kept

The dispersion is a genuine improvement and stays available behind the flag: `CTj` can now
have a distribution, which every downstream moment needs and which the flat constant made
impossible. The default is unchanged, so no frozen figure moves.
