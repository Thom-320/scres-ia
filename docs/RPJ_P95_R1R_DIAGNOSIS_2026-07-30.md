# `rpj_p95` in R1r: three of four risks have no recovery duration, so RPj degenerates to CTj

> **CORRECCIÓN 2026-07-30 (`8a6aa16`).** Todos los `d_k` de este documento están
> **inflados**: se calcularon con un script ad-hoc que dividía solo por el error estándar
> de la referencia y omitía el nuestro, mientras `fidelity_moments.discrepancies` divide
> por `sqrt(s²/n_ref + se²)`. Los valores correctos son **`rpj_p95` 14,6 (no 249,8)**,
> `rpj_mean` 11,0 (no 19,3), `ret_mean` 1,7 (no 1,6); `autotomy_share` queda en 11,2
> porque nuestra SE ahí es exactamente cero. La brecha es real; la magnitud estaba
> sobrestimada.
>
> **Además**, la afirmación central de la §4 —que R12/R13 «no tienen distribución de
> duración»— es **falsa**: la Tabla 6.6b(3) fija 168 h y 24 h y el código ya las tenía.
> Y el mecanismo propuesto fue **refutado** (`a1485a0`, `a0912bd`).

**Status:** `DEVELOPMENT_DIAGNOSIS_NOTHING_CHANGED`. Roots 2,200,001–3, one shift, no
strategic buffers, escalated R1r. Reference `fidelity_reference_v3`.

## 1. It is not the mode migration

First check, because I had just switched the default:

| momento | `disruption` | `d_k` | `elapsed` | `d_k` | referencia |
|---|---:|---:|---:|---:|---:|
| `rpj_p95` R1r | 1761.5 | 230.6 | 1869.6 | 249.8 | 456.5 |
| suma `d_k` R2r | 47.7 | | **19.6** | | |

Both modes are broken on this moment; `disruption` is marginally less bad and much worse
everywhere else. **The migration to `elapsed` stands** and did not cause this.

## 2. The reference is a ceiling, not a tail

Garrido's nine R1r sheets, 21,561 rows with `RPj > 0`:

| | p50 | p90 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|
| **Garrido R1r** | 99 | 424 | **456** | 816 | **1156** |
| **ours** | 54 | 941 | **2078** | — | **7176** |

His p95 is 456.5 with an SD of **17.0 across nine sheets** (range 421–480). That is a
structural bound, and his all-sheet maximum is 1,156 h. **Our maximum is 7,176 h — 6.2×
beyond anything his model ever produced.**

Note our p50 is *below* his (54 against 99). We are not uniformly slower. We are bimodal:
most orders too fast, a minority catastrophically slow.

## 3. The mechanism: `RPj ≡ CTj`

**638 of our 639 orders have `RPj` exactly equal to `CTj`.** In Garrido's CF1 it is 730 of
4,217 — about 16%.

`supply_chain.py:5933`:

```python
eff_risk_start = max(earliest_risk_start, order.OPTj)
order.RPj = max(0.0, order.OATj - eff_risk_start)
```

When a risk is already active as the order is placed, `eff_risk_start` collapses to `OPTj`
and `RPj = OATj − OPTj = CTj` identically. Measured, **66.7% of risk-touched orders are
placed with a risk already ongoing**; for the rest the first onset lands essentially at
placement anyway.

So RPj stops measuring a recovery period and measures the order's whole lifetime. Two
consequences follow, and both are visible in the numbers:

* our **median RPj is exactly 54.0** — that is `GARRIDO_FULFILLMENT_DELAY_HOURS`, a fitted
  constant. Every risk-touched-but-unblocked order reports the calibration constant as its
  "recovery period". Garrido's median is 99, a real duration, and his values are continuous
  (76.97, 69.98, 71.44 …), not pinned to a grid;
* our **tail becomes the blocked-order lifetime**, unbounded by anything.

## 4. The root cause, and why R2r is fine

The tail is entirely procurement. Every order above his p95 carries **R13**, and the extreme
ones carry **R12**:

| riesgos atribuidos | n | p50 | p95 | max |
|---|---:|---:|---:|---:|
| R11+R14 (+ongoing) | 310 | 54 | 96 | 168 |
| R11+R13+R14 (+ongoing) | 278 | 54–192 | 1258–2359 | 3288 |
| **R11+R12**+R13+R14 (+ongoing) | 30 | **3000** | **6451** | **7176** |

And the reason is in `config.py`:

| riesgo | ops | duración de recuperación |
|---|---|---|
| R11 workstation breakdowns | 5,6 | exponencial, media **2 h** |
| **R12** contract delays | **1** | **ninguna** |
| **R13** raw-material shortages | **2** | **ninguna** |
| **R14** defective products | 7 | **ninguna** |
| R21 natural disasters | 3,5,6,7,9 | exponencial, media 120 h |
| R22 LOC destruction | 4,8,10,12 | exponencial, media 24 h |
| R23 forward-unit destruction | 11 | exponencial, media 120 h |

**Three of the four R1r risks carry no recovery-duration distribution at all** — they are
counts (delayed contracts per cycle, delayed deliveries per cycle, defects per shift) — and
the fourth recovers in 2 h. There is no principled recovery period to attribute, so both
modes improvise, and both improvisations resolve to the order's blocked time.

R2r reproduces (`rpj_p95` at **0.7 SD**, `rpj_mean` 4.2) because R21/R22/R23 all carry real
durations. **The family whose risks have durations reproduces; the family whose risks are
counts does not.** That is the diagnosis, and R2r is its control.

## 5. What this is and is not

**It is not** a metric defect, a mode defect, a population defect, or a horizon defect —
those were checked and each is ruled out above or in
`docs/RETRACTACION_POBLACION_PUNTUADA_2026-07-30.md`.

**It is** a missing model parameter: R12 and R13 need a delay-duration distribution, since
a "delayed contract" and a "delayed delivery" are delays and a delay has a length. Ours has
none, so the length becomes whatever the queue does.

## 6. What I am not doing, and why

I am **not** fitting durations to close `rpj_p95`. That is exactly the failure mode this
project has hit six times: fit one observable, break another. Garrido's own R1r `RPj`
distribution (p50 99, p95 456, max 1156, 21,561 rows) is a legitimate target, but any
proposal must be preregistered with **multi-moment** acceptance — at minimum `ret_mean`,
which is currently good at 1.6 SD in R1r and must not regress — and must declare the
duration family before seeing its fit.

Two facts are worth carrying into that preregistration:

1. `ret_mean`, the endpoint the paper reports, is **already at 1.6 SD (R1r) and 1.5 SD
   (R2r)**. This defect does not block the manuscript's headline.
2. `autotomy_share` is **0.000 in both families** against a nonzero reference. Autotomy
   never fires anywhere, which is older and independent of everything here, and is likely
   the same missing-duration problem seen from the other side: with `RPj = CTj > LTj`
   always, the `CTj <= LTj` autotomy branch is unreachable.

Point 2 means these are probably **one defect, not two**.
