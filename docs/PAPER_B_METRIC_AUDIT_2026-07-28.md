# Paper B metric audit

Date: 2026-07-28

Verdict: `STOP_PAPER_B_METRIC_UNRESOLVED`

This audit changes Paper B's preregistration, not Program Q. It opened no fresh
root, tape, or optimizer seed and produced no learner or policy result.

## What is confirmed

Garrido-Rios (2017), Equation 5.5, defines an order-level conditional `ReT`
from autotomy, recovery, non-recovery, and fill-rate states. Section 5.6.3
states that `ReT` is normalized on a 0-to-1 scale.

Garrido, Pongutá, and García-Reyes (2024), Equations 3--6, define a different
five-variable factory-resilience index:

```text
R = sigmoid(
      0.024 ln(zeta)
    - 0.026 ln(epsilon)
    + 0.040 ln(phi)
    - 0.060 ln(tau)
    - 0.1771 ln(kappa_dot)
)
```

The 2024 paper describes the coefficients as offset factors that make the five
arguments comparable. It obtained each coefficient by setting the term at its
observed Monte Carlo maximum equal to `1/5`, for example
`a ln(zeta_max) = 0.20`.

The two metrics must not be conflated. The inspected Excel files contain the
2017-style order-level `ReT`; they do not document the 2024 Cobb--Douglas
calculation.

## Workbook audit

Source hashes are frozen in
`contracts/paper_b_metric_gate_v1.json`. `Rsult_1.xlsx` has:

- twelve processed sheets `Cf1`--`Cf12`;
- an aggregate `Re` sheet with 2,518 data values per configuration;
- varying populations in the individual `Cf` sheets (2,429--2,510 numeric
  values in column `I`);
- pasted `ReT` values in the `Cf` sheets rather than formulas;
- values up to `1.000155`, outside the range declared in the thesis.

The out-of-range values are real, but the proposed causal interpretation in
the prior review was too strong.

Their mechanism is visible in the same rows. The autotomy branch implements
`Re(APj) = APj/LT`; the repeated workbook values satisfy:

```text
48.00744 / 48 = 1.000155
48.00120 / 48 = 1.000025
```

Thus the excess comes from a continuous event time slightly beyond the nominal
48-hour lead time. The prospective contract must decide whether to bound
`APj` by `LT` or clip final `ReT`. Neither choice licenses deletion of the
order.

For `Cf1`:

| treatment | mean |
|---|---:|
| observed values | 0.014330072 |
| clip values to `[0,1]` | 0.014329478 |
| delete values `>1` | 0.007481746 |
| delete the full tail `>=0.5` | 0.005653348 |

Clipping changes the mean by only `5.95e-7` and preserves the complete
`Cf1`--`Cf12` mean ranking. The ranking reversal appears only after deleting
whole high-resilience observations. Deletion is not equivalent to enforcing
the declared bound and is not an authorized correction.

The rare high values do contribute materially to the arithmetic mean. That is
a distributional property worth reporting, but it does not establish that a
published conclusion is erroneous:

1. the thesis used distributional and rank-based comparisons, not the
   twelve-sheet mean ranking constructed here;
2. `Rsult_1.xlsx` covers only a subset of the thesis configurations;
3. the aggregate `Re` sheet is not identical to the individual `Cf`
   populations: it adds mainly zero-valued observations and a small number of
   boundary values;
4. no provenance artifact currently proves that this workbook version was the
   exact source of a specific published table.

The allowed wording is:

> The inspected workbook contains small upper-bound violations and strongly
> non-Gaussian order-level ReT distributions. Its aggregate and per-sheet
> populations differ. These facts motivate a prospective metric audit.

The forbidden wording is:

> Garrido's published ranking is caused by erroneous out-of-range orders.

## Why the 2024 index remains promising

The Cobb--Douglas index directly represents average inventory, backorders,
spare capacity, fulfillment time, and cost. It therefore offers a principled
candidate for testing whether the 2017 state classification and
policy-dependent order population suppress decision discrimination.

It is not yet an authorized endpoint. The largest numerical coefficient is
the cost term, while the MFSC DES has no Garrido-validated cost contract.
Moreover:

- `zeta` needs a frozen container registry and a decision on raw,
  finished-goods, and in-transit inventory;
- `epsilon` needs one common backorder population;
- `phi` needs installed and realized capacity at native time resolution;
- `tau` must implement the published net-requirements ratio or be declared an
  amended proxy;
- `kappa_dot` depends on a frozen comparison set;
- maxima used for scale calibration must come from a frozen development
  library, never evaluation or confirmation data;
- zero values need a preregistered log-domain rule.

## Existing implementation is not authority

The repository contains a historical implementation labelled
“paper-faithful.” It is useful exploratory code, but its current documentation
already reveals authority gaps:

- `tau` is a DES-compatible proxy;
- costs are repository-assigned rather than Garrido-validated;
- `kappa_dot` is normalized by a Monte Carlo reference mean rather than by
  the frozen set of compared strategy-level mean costs;
- calibration cycles through a small policy set and computes maxima on
  episode rows.

Paper B must not inherit that implementation as an endpoint merely because it
runs and has tests.

## Binding decision

Paper B remains prospective. Before scientific execution, the metric gate
must:

1. freeze a primary endpoint and companion endpoints;
2. resolve every null in
   `contracts/paper_b_metric_gate_v1.json`;
3. obtain Garrido's written approval for the operational semantics and cost
   coefficients;
4. pass the negative checks in that contract;
5. emit `PASS_PAPER_B_METRIC_PRE_FREEZE`.

Until then, `STOP_PAPER_B_METRIC_UNRESOLVED` is binding.
