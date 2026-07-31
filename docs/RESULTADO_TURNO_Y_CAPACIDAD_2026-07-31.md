# Result — turno y capacidad: HALTED, y el mecanismo está refutado por conteo

**Status:** `HALTED_FALSIFIER_FAILED`. Ejecuta
`docs/PREREGISTRO_TURNO_Y_CAPACIDAD_2026-07-31.md` sobre `supply_chain/arm_runner.py`.
Artefacto `results/metric_audit/shift_capacity_arms_v1/result.json`, sellado.
Raíces 2.800.001–12. **Ningún momento se reporta**, per §5.

## 1. Falsadores

| falsador | resultado |
|---|---|
| f2 soporte: `min(CTj) ≥ 48` | **PASA** |
| f5 el turno no toca aguas arriba (`risk_events` bit-idéntico A↔S) | **PASA** |
| f8 conjunto `epsilon`-estable | **PASA** |
| **f3.1 `δ ~ U(0,8)`** | **FALLA** |
| **f3.2 bandas con huecos vacíos** | **FALLA** |
| **f3.3 SC reconstruye `p25`/`p50`** | **FALLA** |
| **f6 `CTj` deja de ser masa puntual** | **FALLA** |

| brazo | min | distintos | p25 | p50 | δ p25 | δ p50 | δ p75 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Garrido** | 48,01 | — | **75,00** | **101,45** | **2,00** | **4,02** | **6,00** |
| A statu quo | 54,00 | 228 | 54,00 | 54,00 | 0,00 | 6,00 | 6,00 |
| S turno | 48,00 | 228 | 48,00 | 48,00 | **0,00** | **0,00** | **0,00** |
| C capacidad | 48,00 | 228 | 48,00 | 48,00 | **0,00** | **0,00** | **0,00** |
| SC ambos | 48,00 | 228 | 48,00 | 48,00 | **0,00** | **0,00** | **0,00** |

S, C y SC salen **idénticos entre sí**. Ninguno de los dos mecanismos dispara.

## 2. Por qué, medido

| | |
|---|---|
| órdenes servidas por día (mediana) | **1,0** |
| `q` mediana | 2.480 raciones |
| lo que consume del turno, `q/λ` | **7,74 h de 8** |
| capacidad diaria de flete | 2.600, y un pedido de 2.480 **cabe** |

**Con un pedido por día, no hay competencia entre pedidos.** El turno nunca se llena, así
que `used` al entrar siempre vale 0 y `δ ≡ 0`. El flete nunca se satura, así que `k` nunca
sube por capacidad. **Los dos mecanismos son estructuralmente incapaces de disparar en
nuestro régimen de demanda.**

## 3. Y esto refuta la hipótesis, no solo mi implementación

El punto no es que el código esté mal: es que **el conteo lo prohíbe en los dos modelos**.
Su tasa de órdenes puntuadas es ~213/año, la nuestra ~213/año — están emparejadas. Con menos
de un pedido por día, **`δ` no puede venir de una cola entre pedidos en su modelo tampoco**.

Así que la parte «espera por capacidad» de mi lectura está refutada por aritmética, y la
parte «turno» necesita un generador que no sea encolamiento. Candidatos que **no** puedo
distinguir con lo que tengo:

* `δ` intra-pedido — dónde dentro del propio handover se registra `OATj`. Pero si fuera el
  final del handover, `δ = q/λ ∈ [7,49; 8,10]`, y lo observado es `p50 = 4,02`. No encaja.
* algo exógeno del solver de Simulink. Su `OPTj` **sí** deriva de fase (`corr = 0,9999`, de
  0,85 h a 5,53 h a lo largo de la corrida), aunque ya verifiqué que `δ` **no** sigue esa
  deriva (`corr(δ, OPTj) = −0,01`).

## 4. Lo que sobrevive, y no es poco

**El techo de 8 h sigue siendo exacto.** `δ` observado tiene p25/p50/p75 = 2,005 / 4,020 /
6,000 contra `U(0,8)` teórico 2,000 / 4,000 / 6,000, con 98,5% dentro de `[0,8]`. Que el
límite superior coincida al 0,1% con `HOURS_PER_SHIFT` a `S = 1` es demasiado preciso para
ser casualidad — **pero coincidencia numérica no es mecanismo**, y este contrato existía
para distinguir esas dos cosas. Lo distinguió.

**La descomposición sigue en pie.** `CTj = 48 + k·24 + δ` reconstruye `p25 = 75,00` y
`p50 = 101,45` exactos, y la estructura de bandas con huecos vacíos es un hecho de sus datos
(`f1dfd2f`). Lo que no sabemos es **qué genera `δ`**.

**f5 pasó**, así que el calendario de turno quedó confinado a la pierna de cumplimiento y no
se filtró aguas arriba — el aislamiento que el contrato pedía funciona.

## 5. Lo que no hago

**No re-implemento `δ` como un sorteo `U(0,8)`.** Sería ajustar la forma observada y
volvería tautológico el falsador 3.1 — exactamente el defecto que este proyecto pasó un día
eliminando. Si `δ` se implementa alguna vez como sorteo, tiene que ser bajo un contrato que
lo declare **como supuesto, no como predicción**, y que lo puntúe solo por lo que arrastre en
los otros momentos.

**No sigo iterando implementaciones del mecanismo.** Probar variantes hasta que una produzca
`U(0,8)` es ajuste por implementación, que no deja rastro en ningún parámetro y por eso es
peor, no mejor, que ajustar un número.

## 6. Estado

`ret_mean` bajo los defaults embarcados no lo toca nada de esto. La brecha abierta con causa
nombrada sigue siendo `δ`, y ahora sabemos **qué no es**: no es encolamiento entre pedidos,
ni en su modelo ni en el nuestro.
