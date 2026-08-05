# Resultado E\*-C (v1_1) — con `f6` implementado de verdad, el barrido se detiene

**Sustituye a `docs/RESULTADO_ESTAR_CAPACIDAD_2026-08-03.md`, que no es adjudicable.** Mismo
contrato (`docs/PREREGISTRO_ESTAR_CAPACIDAD_BARRIDO_2026-08-03.md`), mismas semillas quemadas
`5.200.001–16`, réplica declarada. Ninguna semilla nueva.

**Estado: `HALTED_FALSIFIER_FAILED`.** El veredicto descriptivo sigue siendo
`ARGMAX_MOVES_WITHOUT_VALUE`, pero **no se adjudica**, porque un guardarraíl falla.

## 1. Lo que estaba roto en el instrumento

Dos de los seis falsadores **no podían fallar** — el patrón que este proyecto ya lleva un mes
cazando:

| falsador | cómo estaba | por qué no probaba nada |
|---|---|---|
| `f2` | `len({demanda redondeada}) >= 1` | verdadero por construcción; sin comparador |
| `f6` | `"passed": True` literal | no calculaba ni UCB ni daño |

Reparados **contra el brazo nulo sin capacidad sobre la misma cinta**: `f2` compara las 576
celdas capadas con su propia corrida sin tope, total **y por reclamante**; `f6` bootstrapea el daño
en el reparto **seleccionado** y exige `UCB95(daño) ≤ δ`. Ambos verificados por control: `f2`
devuelve `False` ante una demanda cuyo total coincide pero cuyo reparto A/B se movió, y `f6` **falla
en la corrida real**.

## 2. Los números

| presupuesto | `H_regime` | LCB95 | `argmax` base → freq3 | dispersión | binding |
|---|---:|---:|---|---:|---:|
| 600 | +0,00025 | +0,00000 | 0,6 → 0,5 | 0,104 | 0,77 |
| 1.200 | +0,00000 | +0,00000 | 0,6 → 0,6 | 0,043 | 0,59 |

Idénticos a la corrida anterior: **la reparación no tocó la física**, sólo los falsadores. `f2`
pasa limpio — 576 comprobaciones, 0 discrepancias, delta máximo **exactamente 0**.

## 3. Por qué falla `f6`, que es el hallazgo

En el régimen escalado la capacidad **sí destruye**:

| celda | `lost_orders` (daño medio) | UCB95 | δ | `worst_claimant_fill` UCB95 | δ |
|---|---:|---:|---:|---:|---:|
| 600 · freq3_imp2 | **+0,500** | **+1,500** | 0,50 | **+0,0300** | 0,010 |
| 1.200 · freq3_imp2 | +0,375 | +1,125 | 0,50 | +0,0305 | 0,010 |
| ambos · base | +0,000 | +0,000 | 0,50 | +0,0026 | 0,010 |

**«La capacidad retrasa; no destruye» era una generalización de la sonda, y es falsa.** La sonda se
corrió en el régimen base, donde efectivamente `lost_orders = 0`. Bajo `freq3_imp2` se pierden
pedidos y el reclamante peor servido cae por encima del margen firmado. **La afirmación defendible
es la restringida al régimen base.**

Igual de restringida debe quedar la otra: `worst_claimant_fill` **no respondió en el contrato
probado**, que no es lo mismo que ser estructuralmente ciego.

## 4. Qué queda dicho, y qué no

* **No se cierra el eje de capacidad.** Un barrido detenido por guardarraíl no adjudica.
* Lo descriptivo se mantiene: la palanca **tiene autoridad** (dispersión 0,104, binding 0,77) y el
  `argmax` **casi no depende del régimen** — pero es descripción, no veredicto.
* **Alcance real:** sólo **CSSU A/B** están cableados al DES. `wdc`, `al` y `sb` existen en el
  módulo con capacidad admitida y **sin conectar**, así que **esto no es todavía la expansión por
  nodo que pidió Garrido**.
* **`flow_fill_rate` es un cociente terminal censurado**, no una métrica temporal pura: ve el
  retraso sólo a través de lo que sigue sin entregar al horizonte. Para hablar de retraso hacen
  falta AUC de pérdida de servicio, backorder temporal o panel semanal.
* **Ninguna semilla nueva, y no autoriza entrenar nada.**
