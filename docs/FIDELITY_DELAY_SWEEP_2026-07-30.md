# The calibration grid, scored against Garrido's moments — and what it found

> **RETRACTADO 2026-07-30 — el hallazgo central de este documento es falso.**
>
> Usé `RPj > 0` como proxy de «tocada por riesgo». **Es el proxy equivocado**: una orden
> clasificada como autotomía tiene `RPj = 0` y `APj > 0` **por construcción**. Así que la
> intersección «en horario ∧ `RPj>0`» está vacía por definición de la clasificación, no
> porque el fenómeno falte. Medí la ausencia de un conjunto que no puede existir.
>
> Con el proxy correcto (`APj > 0`), la autotomía **sí dispara**, y el problema es el
> opuesto al que reporté — sobra, no falta:
>
> | familia | delay ≤ 48 | delay ≥ 49 | referencia |
> |---|---:|---:|---:|
> | R1r | **0,4994–0,5006** (≈115× la referencia) | 0,000 | 0,00436 |
> | R2r | **0,1480–0,1504** (≈236× la referencia) | 0,000 | 0,00064 |
>
> No hay «partición estricta» ni «ninguna celda produce un solo caso». Hay un salto brusco
> entre delay 48 y 49: por debajo del lead time la rama dispara para la mitad de las
> órdenes en R1r, por encima no dispara nunca. Ninguno de los dos lados se acerca a la
> referencia. El sucesor correcto es
> `docs/AUTOTOMY_PROXY_CORRECTION_2026-07-30.md`.



**Status:** `DEVELOPMENT_FIDELITY_SWEEP_NO_CONSTANT_CHANGED`. Runner
`scripts/run_fidelity_delay_sweep.py`, artifact
`results/metric_audit/fidelity_delay_sweep_v1/result.json`, reference
`results/metric_audit/fidelity_reference_v1/result.json`, contract
`contracts/paper_b_independent_calibration_v2.json`. Roots 2,000,001–2,000,012, `LT`
fixed at 48 (thesis §6.8.2 p.111), `S = 1`, no strategic buffers.

Executed as preregistered: 24 cells per family (6 delays × 2 semantics × 3 tolerances),
scored by epsilon-dominance over six moments, no winner selected.

## The grid does not contain a cell that reproduces autotomy

**Not one of the 24 cells produces a single autotomy case, in either family.** Not at
delay 42 or 47, six and one hours *below* the lead time. The reference is 0.436% of
orders in R1r and 0.064% in R2r; every cell returns exactly 0.000%.

The reason is not the delay. It is a strict partition, verified over the whole grid and
both families on three roots each:

| family | delay | served | on time (CTj ≤ 48) | touched by risk | **both** |
|---|---:|---:|---:|---:|---:|
| R1r | 42 | 647 | 415 | 232 | **0** |
| R1r | 47 | 647 | 415 | 232 | **0** |
| R1r | 48 | 646 | 414 | 232 | **0** |
| R1r | 49–60 | 646 | 0 | 646 | **0** |
| R2r | 42 | 645 | 395 | 194 | **0** |
| R2r | 47 | 645 | 395 | 194 | **0** |
| R2r | 48 | 643 | 393 | 194 | **0** |
| R2r | 49–60 | 643 | 0 | 318–330 | **0** |

**In our DES, a risk touching an order always makes it late.** On-time and risk-touched
are disjoint sets at every delay. Autotomy — a disruption occurs and the order still
arrives on schedule — is the thesis's central absorption mechanism, and our model has no
mechanism that can produce it.

That is a **structural gap, not a calibration gap**, and no value of a constant can close
it. In Garrido's data 0.44% of R1r orders are hit by a risk and still arrive on schedule;
in ours the conditional probability of arriving on schedule given a risk is exactly zero.

## What this settles

**The delay sweep is answered, negatively and cleanly.** The question was whether some
delay in the declared grid reproduces Garrido's moments better than 54. On the autotomy
moment the answer is that none does, because none can. The remaining five moments still
discriminate — the grid is not degenerate — but a cell that is at 11 combined standard
errors on autotomy is not a fidelity improvement whatever it does elsewhere.

**The migration question is moot for now.** There is no winner to migrate to. The
historical lane keeps delay 54, labelled, and the prospective lane has no candidate until
the structural gap is addressed.

**And the earlier diagnosis was incomplete.** `RET_METRIC_DEFECTS_2026-07-29.md` §2 said
the autotomy branch is unreachable because the minimum cycle time is 54 h against a 48 h
promise. True but not the cause: at delay 42 the cycle time is 42 h, 415 of 647 orders are
on time, and autotomy still never fires. The 54 h floor was a symptom sitting on top of
the real gap.

## What it does not settle

Why the partition holds. Two candidates, both testable and neither tested here: our risk
processes may have no sub-lead-time impact scale — every disruption is long enough to push
an order past the promise — or the last-mile pipeline may carry no slack that could absorb
a short disruption. Garrido's own autotomy rows are consistent with the second: his
autotomy orders exceed LT by 0.00744 to 0.048 h, which is absorption of a *tiny* overshoot,
not of a long outage.

Nothing here changes a constant, relabels a frozen result, or authorises a learner. The
six moments, their reference spreads and the full 48-cell grid are in the artifact.

## Epsilon and dominance

Both families report `grid_discriminates: true` with 12 of 24 (R1r) and 16 of 24 (R2r)
cells non-dominated, and `epsilon_stable: false` — the non-dominated set moves across
ε ∈ {0.25, 0.5, 1.0, 2.0}. Per the contract that set is reported as unstable rather than
shown as a result. It does not matter much here: every cell fails the autotomy moment
identically, so the dominance ordering is decided by the other five.
