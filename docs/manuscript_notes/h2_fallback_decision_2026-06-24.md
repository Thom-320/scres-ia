# H2 Fallback Decision - 2026-06-24

## Question

If the powered retained-vs-reset run shows that H2 is flat or absent, what is the
fallback?

H2 is the persistence dose-response claim:

```text
d/d_rho E[ReT_retained - ReT_reset] > 0
```

## Decision

Precommit the fallback as:

```text
H1-primary, H2 exploratory.
```

If retained learning beats reset/static but the persistence gradient is flat, the
headline remains path-dependent learning:

```text
retained online learner > reset-learning ablation
```

The persistence dependence is then reported as exploratory, heterogeneous, or
unsupported rather than being forced into the main claim.

## Why

This is the cleanest and most defensible fallback. It still answers Garrido's
learning gap: the model tests whether retaining policy state across disruption
blocks matters. It avoids changing the mechanism after seeing the data, and it
keeps the manuscript honest if persistence is not the variable that unlocks
retention value.

## Exploratory Follow-Up

If H2 is absent, investigate whether retention value is driven by disruption
heterogeneity or diversity rather than persistence. The first `ReT_cd` pilot
hinted that the retained-minus-reset gap was largest near the memoryless setting,
so diversity of disruption experience may be a better mechanism than sticky
campaign phases.

This remains exploratory unless pre-registered in a later experiment.

## Rejected Fallback

Do not simply decide later from the data. The powered run can inform discussion
and future mechanisms, but the paper framing should not be chosen post hoc from
whichever estimate looks best.

