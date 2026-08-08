# Three slices where macOS and Linux do not agree, and why the base surface could not have found them

## What was measured

The base surface reproduced **bit-exactly** across two architectures: 103,680 cells produced on
macOS arm64 / Python 3.11.15 and verified on Linux x86_64 / glibc 2.43 / Python 3.14.4, `max|Δ| =
0.0`. That result stands. What it does not do is generalise, and these three slices are why.

Of 55 extended-surface slices the VPS verified, **52 reproduce exactly**. Three do not, and all
three sit in the same context, `R1r|esc`:

| slice | cells differing | max abs delta |
|---|---:|---:|
| `ext__R1r_esc__8200011` | 39 of 4,608 | 10006.0 |
| `ext__R1r_esc__8200054` | 18 of 4,608 | 5111.0 |
| `ext__R1r_esc__8200053` | 72 of 4,608 | 5.3e-08 |

The first is not floating-point noise. Re-evaluated on the VPS for cell index 2873
(`buffer_hours=504, shifts=3, op9_rop=12, op12_rop=48, op3_rm=70000, op5_rm=17500`):

| panel key | cache (macOS) | VPS (Linux) | difference |
|---|---:|---:|---:|
| `delivered_rations` | 440,123 | 430,117 | **10,006 rations** |
| `flow_fill_rate` | 0.5701 | 0.5571 | 1.30 pp |
| `lost_orders` | 70 | 74 | 4 |
| `ret_excel_risk_conditional` | 0.0069458 | 0.0070251 | 7.9e-05 |

Same configuration, same context, same seed, same horizon: a materially different simulation
trajectory, not a different rounding.

## Why the base surface was blind to it

**Every configuration in the 288 grid has `op3_rm = op5_rm = 0`.** The extended grid is exactly the
addition of those two raw-material factors. The divergent cell sets both above zero. So the
cross-architecture agreement established on the base surface was measured on a subspace in which
two of the six factors are pinned at their null level, and the code path the extended grid adds was
never exercised across platforms until now.

That is the correction the headline needs. "103,680 cells reproduce bit-exactly across two
architectures" is true and must not be read as "the simulator is platform-independent".

## What this does and does not decide

It does not say which platform is right. The cache was produced on macOS and reproduces there: a
full 4,608-cell sweep of `8200011` on the producing platform found **zero** differences, and its
worst cell returned the cached value exactly three times out of three. So the artifact is
reproducible on its own environment and disagrees with another.

Consequently the authoritative forward-equivalence verdict is computed on the **producing**
platform, and these three slices are recomputed locally. The 52 exact VPS slices are kept in
`shards/`: a slice that reproduces exactly on a different architecture is stronger evidence than one
that reproduces on the same, and nothing is gained by discarding it.

## Open, and it belongs in the manuscript

The mechanism is not identified. A 10,006-ration gap from an identical seed points at event ordering
under ties rather than at arithmetic, which is the kind of thing a discrete-event simulator can
carry for years without anyone noticing, because nobody runs the same seed on two libcs. The
confinement to one context out of six and to 129 cells out of 18,432 says it is rare and specific,
not pervasive.
