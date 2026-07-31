# Provenance repair — corrected audit

**Status:** `AUDIT_CORRECTED_BOTH_ORIGINAL_BLOCKERS_WITHDRAWN`. No manuscript number changed.

> **Retractación, y es sustancial.** La primera versión de esta auditoría (commit `86efe03`)
> declaró **dos contradicciones bloqueantes**. **Las dos eran mías.** Busqué en `sections/*.tex`
> y en `results/` y concluí que las cifras no trazaban — sin mirar
> `scripts/build_manuscript_figures.py`, que **sí cita sus fuentes**, ni
> `outputs/experiments/`, donde viven las corridas. La procedencia existía; yo no la busqué
> donde estaba.
>
> Peor: **llegué a editar la tabla** para «corregir» el valor que resultó ser el correcto y
> trazable. Revertido antes de commitear el manuscrito. Habría debilitado un resultado
> correcto sobre la base de una lectura equivocada.

---

## 1. «Contradicción A» — RETIRADA. No había contradicción

Afirmé que la celda `increased/h104` llevaba dos valores irreconciliables y que la tabla
reportaba el más favorable en contra de su propia convención.

**El artefacto existe y respalda la tabla.**
`docs/track_b_q1_stats_2026-07-02_final/e3_per_cell_seed_ci.csv`, fila `h104,increased`:

| campo | artefacto | tabla |
|---|---|---|
| `delta_mean` | 0.0005515742695782133 | **+0.000552** ✓ |
| `ci_lo` | 0.0005225507855504733 | **+0.000523** ✓ |
| `ci_hi` | 0.0005805977536059532 | **+0.000581** ✓ |
| `n_seeds` / `seeds_positive` | 5 / 5 | **5/5** ✓ |

Los dos valores son **dos comparadores distintos y legítimos**, exactamente como dice la
prosa: la tabla es el comparador **fijo** de E3 (`s3_d2.00`), y el `+0.000542` es la
re-optimización posterior sobre la **frontera densa de 147 políticas**.
`docs/E3_GENERALIZATION_VERDICT_2026-07-02.md:12` lo declara: E3 es *«not a re-optimized
dense per-regime dense-CRN spec»*.

## 2. «Contradicción B» — RETIRADA. Son dos corridas distintas

Afirmé que Track A llevaba dos conteos de semillas irreconciliables. **Ambos son correctos y
ambas corridas existen:**

| dónde | corrida | verificado |
|---|---|---|
| tabla `:103`, «3 seeds» | `outputs/experiments/per_op_conflict_campaign_..._3seed_richmetrics_2026-06-29` | **existe** |
| figura 13, «5 seeds» | `outputs/experiments/track_a_repair_continuous_5seed_2026-06-30` | **existe**, citada en `build_manuscript_figures.py:941` |

## 3. Lo que SÍ queda, y es menor de lo que dije

**(a) Ambigüedad de lectura, no contradicción.** La tabla y la figura 13 muestran **dos
aprendices distintos** —PPO BC-warm-started a 3 semillas contra PPO continuo a 5— y **el texto
no lo dice**. Un revisor asumirá que es el mismo y verá un conflicto donde no lo hay,
exactamente como me pasó a mí. **Una frase lo cierra.**

**(b) El valor de frontera densa no tiene artefacto localizado.** `+0.000542` / `0.003118` /
CI `[+0.000500, +0.000584]` aparecen solo en prosa. El bundle de E3 no los contiene y no
encontré el de la frontera densa. **Ése es el único hueco de procedencia real que confirmé.**

**(c) La procedencia no llega al lector.** 9 de 17 funciones de `build_manuscript_figures.py`
declaran `Source:`, pero **0 de 20 captions** lo hacen. La trazabilidad existe en el pipeline
y no en el papel. Para C&IE, una línea por caption la haría auditable sin tocar ningún número.

**(d) 12 de 20 captions no declaran conteo de semillas.**

**(e) El `±15%` inventado no existe** — verificado en todas sus formas. Ese sí estaba cerrado.

## 4. La lección, que es la del día entero

Declaré «cero procedencia» tras buscar en dos de los cuatro sitios donde vive. El defecto no
estaba en el manuscrito: estaba en el alcance de mi búsqueda. **Y la señal de alarma la tenía
delante** — el propio script de figuras llevaba un comentario fechado
`2026-07-09 provenance fix` explicando el valor que yo estaba a punto de «corregir».

Antes de declarar que una cifra no traza, hay que agotar: `sections/*.tex`,
`scripts/build_manuscript_figures.py`, `docs/track_b_q1_stats_*`, y `outputs/experiments/`.

## 5. Acciones reales que quedan

1. **Una frase** distinguiendo los dos aprendices de Track A (§3a) — mía, inmediata.
2. **Localizar o re-generar** el bundle de frontera densa (§3b) — único bloqueo de procedencia.
3. **`Source:` en cada caption** (§3c) y conteo de semillas en los 12 que faltan (§3d).
4. Los 4 placeholders siguen dependiendo del PI y de Garrido.
