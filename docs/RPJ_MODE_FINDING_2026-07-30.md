# `RPj` uses a formula the thesis does not specify, and it is the worse one

**Status:** `DEVELOPMENT_FINDING_NO_DEFAULT_CHANGED`. Measured on roots 2,000,001–3, both
families, against `results/metric_audit/fidelity_reference_v1/`.

## The two modes

`ret_recovery_period_mode` selects between:

| mode | formula | source |
|---|---|---|
| `disruption` (**current default**) | sum of risk-event overlaps with `[OPTj, OATj]` | ours |
| `elapsed` | `OATj − first risk detection` | **thesis Algorithm 2, p.69** |

The thesis is explicit: *"RPj ← (OATj − first-R⁰ₒᵣ)"*. That is `elapsed`. The shipped
default is `disruption`, which the thesis does not specify anywhere.

## Measured against the reference

| family | mode | `rpj_mean` | × ref | `rpj_p95` | × ref |
|---|---|---:|---:|---:|---:|
| R1r | disruption | 531.6 | 2.74 | 3385.3 | 7.42 |
| R1r | **elapsed** | **469.5** | **2.42** | **3246.0** | **7.11** |
| R2r | disruption | 182.8 | **0.29** | 930.1 | **0.31** |
| R2r | **elapsed** | **804.7** | **1.28** | **4401.3** | **1.45** |

**`elapsed` is closer on all four measurements**, and the R2r difference is not marginal:
`disruption` sits at 0.29× and 0.31× of the reference — three times too *low* — while
`elapsed` is at 1.28× and 1.45×.

## Why the wrong one was chosen

The code comment reads: *"The elapsed mode lets plain queue wait inflate RPj up to CTj,
which diverges from the bounded workbook RPj distribution."*

That concern is real **for R1r**, where `elapsed` gives a p95 of 3,246 against a reference
456 and close to our own CT p95. But it was acted on without checking R2r, where the same
choice pushes `RPj` to a third of the reference.

It is the same shape as `delay = 54`: **a parameter fitted on one observable that broke
another**, chosen before the multi-moment reference existed. The reference now exists, and
it says the thesis formula is better on every measurement we have.

## What is still open after switching

`elapsed` does not close R1r: 2.42× on the mean and 7.11× on the p95 remain. So the mode
is a real improvement and not a solution. The residual R1r gap is the next question, and
it is now isolated from the mode choice.

## Why the default is NOT flipped here

Changing it moves every `ReT` figure in the project, exactly like the delay. It goes
through the same discipline: declared, swept alongside the other axes, and adopted only
with the historical line kept and labelled. The measurement is what this document
contributes; the switch is a separate, preregistered decision.

What can be said now without any change: **the shipped default is not the thesis's
formula, and the thesis's formula fits better.** That is a defect independent of which
default we end up running.
