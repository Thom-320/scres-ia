# Preregistro v4 — el borde declarado antes, y el piso de señal convertido en curva

**Escrito y commiteado ANTES de correr.** Runner: `scripts/run_monotone_transform_family_v4.py`.
Predecesor: `results/monotone_transform_family_v3/result.json`
(`A_MONOTONE_RESCALING_SURVIVES_LCB_MULTIPLICITY_AND_SIGNAL`, `f6` FALLÓ).
Semillas: bloque quemado `garrido_q2_des288`, réplica declarada. **Ninguna nueva.**

## 1. Los dos defectos de v3, y por qué el segundo es el grave

**`f6` preguntaba lo que no decidía.** Escribí el chequeo de borde contra la **mejor por LCB**
—`power(γ=100)`, en el borde— cuando la que decide el veredicto es la **mejor que califica**
—`power(γ=21,5)`, interior—. No lo redefiní después de ver el resultado, así que la corrida quedó
con un falsador fallido que no afectaba al veredicto. **Aquí se declara antes, contra la que
decide.**

**Y el piso de señal es un número que elegí yo.** `0,90` no sale de ningún sitio. Con `H_regime`
creciendo monótona en `γ` **sin máximo interior** —0,0195 en la identidad, 0,293 en `γ=21,5`,
0,632 en `γ=100`—, **el piso es lo único que acota la respuesta**. Reportar un solo número
condicionado a una constante arbitraria mía es exactamente la clase de grado de libertad que este
proyecto persigue en otros.

**La reparación no es elegir mejor el piso: es dejar de elegirlo.** Se reporta `H*(piso)` como
**curva** sobre `piso ∈ {0,80 · 0,85 · 0,90 · 0,95 · 0,99}` y **la curva entera entra al
manuscrito**.

## 2. El ancla que faltaba: la curvatura que Garrido sí declaró

La regla de adopción dice que una transformación sólo se adopta por **mecanismo declarado**. Hay
exactamente una curvatura declarada en toda esta discusión, y es **suya**: su índice publicado es
`σ(Σ signo·aₓ·ln x)`, es decir **nuestra identidad**. `γ = 1`.

Así que la pregunta del manuscrito tiene una respuesta que no depende de ningún piso:

> **Bajo la curvatura que Garrido declaró, `H_regime` vale `0,0195`, por debajo del umbral.** El
> headroom sólo aparece imponiendo curvatura **adicional que él no declaró**.

Se mide y se reporta explícitamente como `garrido_declared_curvature`. **Falla si la identidad
deja de coincidir con su parametrización publicada.**

## 3. La familia, con la rejilla lo bastante ancha para que el borde signifique algo

| subfamilia | parámetros | n |
|---|---|---:|
| identidad (= la curvatura de Garrido) | — | 1 |
| logística | 25 umbrales × 20 nitideces (`β` 0,05–500) | 500 |
| potencia | `γ` de **0,001 a 1.000** (v3: 0,01–100) | 61 |
| escalón | 99 umbrales por cuantil | 99 |

**`K = 661`.** Holm sobre las 661.

Con la rejilla tres órdenes de magnitud más ancha, que la **que califica** siga siendo interior deja
de ser trivial: **si el óptimo que decide vuelve a caer en el borde, la familia sigue truncada**.

## 4. Reglas de lectura, fijadas antes de mirar

`GATE = 0,05`. Califica si cumple las tres, **al piso de referencia 0,90**:

```
LCB95 >= 0,05    y    Holm sobre K=661 < 0,05    y    resolubles >= piso x resolubles(identidad)
```

* potencia insuficiente → **`UNDERPOWERED_NO_VERDICT`**
* el proxy no cae en el escalón → **`SIGNAL_CRITERION_VOID`**
* alguna califica al piso 0,90 → **`A_MONOTONE_RESCALING_SURVIVES_ALL_THREE`**
* ninguna al 0,90 pero alguna pasa LCB+Holm → **`SURVIVES_LCB_AND_MULTIPLICITY_BUT_COSTS_SIGNAL`**
* ninguna pasa LCB+Holm → **`NO_MONOTONE_RESCALING_SURVIVES`**

**Y la regla que decide qué se puede decir con esto**, fijada aquí: el veredicto describe **la
familia**, no autoriza **ninguna** transformación. La única curvatura con mecanismo declarado es la
de Garrido, y es la identidad. Cualquier otra exige un preregistro propio que declare **por qué**
esa curvatura, antes de ver su `H`, más confirmación en bloque virgen.

## 5. Falsadores

| falsador | por qué puede fallar |
|---|---|
| `f1_identity_reproduces_the_sealed_scalar` | ancla externa contra el escalar sellado, a 1e-9 |
| `f2_the_signal_proxy_can_actually_fall` | escalón con `resolubles < 0,5 ×` identidad **en las dos rejillas** |
| `f3_the_instrument_has_power` | óptimo plantado a `H = 0,10` → `LCB95 ≥ 0,05`; si no, no hay veredicto |
| `f4_multiplicity_over_the_declared_family` | `K` debe ser exactamente 661 |
| `f5_the_base_grid_stays_at_zero` | control negativo: 0 en las 661 sobre la rejilla de 288 |
| **`f6_the_DECIDING_transform_is_interior`** | **la mejor que CALIFICA** no puede estar en un borde de la rejilla de parámetros. Declarado aquí contra la que decide, no contra la mejor global. **Falla si la familia sigue truncada donde importa** |
| `f7_the_signal_floor_actually_binds` | debe existir alguna transformación rechazada **sólo** por señal. Falla si ninguna lo es, porque entonces el piso es decoración |
| `f8_the_answer_depends_on_the_floor` | `H*(0,80)` debe diferir de `H*(0,99)`. **Falla si no dependen**, y entonces mi objeción sobre la arbitrariedad del piso era infundada y debe retirarse |
| `f9_no_fresh_seeds` | custodia central, réplica declarada |

**Alcance:** desarrollo sobre tapes quemados. No abre semillas, no adjudica, **no adopta ninguna
transformación**, no cambia la primaria del contrato.
