# Preflight G3c sobre tapes quemados — `f2` falla, y la rejilla congelada tiene un nivel muerto

**Autoridad:** `docs/SUPERSESION_CIENTIFICA_G3C_2026-08-05.md` (supersede la regla terminal de
G3-obs §5) + `docs/ENMIENDA_G3C_PREFLIGHT_BURNED_2026-08-05.md` (congela celdas, potencia y
presupuesto). **Semillas:** bloque quemado `5.200.001–16`, réplica declarada. **Ninguna nueva.**

**Estado: `PREFLIGHT_HALTED_FALSIFIER_FAILED`.** No adjudica G3c, no reserva nada, no entrena nada.

## 1. El hallazgo: dos de los tres niveles no son tratamiento

`f2` exige que `min_dwell` **ate** en 3 y 7 días, y que **no ate nunca** en 1. Falla, y la razón es
física, medida directamente sobre el simulador **bajo presión de conmutación máxima** —alternando
el reparto pedido en cada paso, que es lo más duro que puede pedirle cualquier política:

| `min_dwell_days` | acciones retenidas | conmutaciones realizadas |
|---:|---:|---:|
| 1 (nulo) | **0** | 121 |
| 2 | **0** | 121 |
| 3 | **0** | 121 |
| 4 | **0** | 121 |
| 7 | **210** | 61 |

**El espaciado natural de la re-decisión es de unos tres días** —latencia de activación de 24 h más
cadencia diaria—, así que **cualquier permanencia de hasta cuatro días es inerte**. La rejilla
congelada `{1, 3, 7}` contiene **un nulo, una celda muerta y un solo tratamiento real**.

Eso no es un defecto del runner: es el contrato describiendo un factor con menos niveles efectivos
de los que declara. Y **una potencia calculada sobre tres niveles sobreestima lo que el diseño
puede aprender**.

## 2. Lo que sí midió el preflight

Sobre el contraste primario `histéresis − miope`, pareado por semilla, en `worst_claimant_fill`:

| celda | contraste | LCB95 | MDE@16 | `n*` para el SESOI |
|---|---:|---:|---:|---:|
| dwell=1 · base | −0,00073 | −0,00643 | 0,0115 | 22 |
| dwell=1 · freq3 | −0,00290 | −0,00904 | 0,0117 | 23 |
| dwell=3 · base | −0,00122 | −0,00676 | 0,0115 | 22 |
| dwell=3 · freq3 | −0,00306 | −0,00910 | 0,0116 | 22 |
| dwell=7 · base | −0,00157 | −0,00973 | 0,0168 | 46 |
| dwell=7 · freq3 | −0,00139 | −0,01051 | 0,0172 | 48 |

**El candidato pierde en las seis celdas**, incluso donde el dwell ata de verdad. `n*` peor celda
= 48, dentro del presupuesto de 96 — es decir, **la potencia no es el problema**: el problema es
que el signo es negativo.

Los demás falsadores pasan, incluidos los que podían tumbarlo: el nulo explícito reproduce el
legacy **por hash científico canónico**, el incumbente **bate a la mejor constante** (no es hombre
de paja), y el placebo y el reclamante equivocado **pierden** — que en op12 no fue el caso.

## 3. Qué significa, dicho con cuidado

**No cierra G3c.** Un preflight detenido por falsador no adjudica, y el `f2` que falla es un
defecto de **diseño de niveles**, no una medición de la hipótesis.

Pero la dirección es informativa y conviene no maquillarla: **allí donde el dwell realmente ata, la
histéresis sigue perdiendo contra la regla miope**. La conjetura que justificaba reabrir G3c —que
la permanencia mínima deja al incumbente miope fuera de su clase de optimalidad— **no encuentra
apoyo en los tapes quemados**.

## 4. Qué hace falta antes de volver a correr esto

1. **Re-derivar los niveles.** `{1, 7, 14}` o `{1, 7, 21}`, con la inercia medida y no supuesta.
   Requiere enmienda: la rejilla actual está congelada.
2. **Volver a calcular la potencia** sobre la rejilla nueva, otra vez sobre quemados.
3. **Y sólo entonces** el recibo de Submission A o una supersesión de autoridad, antes de tocar un
   bloque virgen.

## 5. Nota de custodia

Este resultado se produjo con un runner que **fue reemplazado en el árbol el mismo día** por una
implementación paralela. El artefacto queda sellado con su propio `module_manifest`, y los tests que
sostienen el hallazgo se reescribieron **contra el simulador y no contra ningún runner**, de modo
que la tabla de la §1 es reproducible con independencia de cuál de los dos sobreviva.
