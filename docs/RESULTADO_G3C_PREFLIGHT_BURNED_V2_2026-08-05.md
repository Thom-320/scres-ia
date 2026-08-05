# Resultado G3c burned-only v2 — `STOP_G3C_GUARDRAIL`

**Artefacto:** `results/headroom/g3c_preflight_grid_v2/result.json`
**Contrato:** `contracts/g3c_burned_preflight_v2.json`
**Rol:** `BURNED_PREFLIGHT` · `replay_of=contention_headroom`
**Semillas:** `5.200.001–5.200.016`, todas quemadas; ninguna semilla nueva.

## Veredicto

```text
STOP_G3C_GUARDRAIL
```

El preflight v2 no autoriza reservar ni abrir un bloque virgen. No es una confirmación de G3c,
no entrena ningún learner y no modifica Program Q, Program O ni Submission A.

## Diseño ejecutado

La caracterización burned-only derivó la grilla físicamente viva:

```text
min_dwell_days ∈ {1, 6, 12}
```

`1` es el nulo legacy; `6` es el primer nivel que retiene acciones en todas las cintas y suprime
al menos 10 % de las conmutaciones del incumbente; `12` es su doble. El primario es
`hysteresis − myopic_equivariant` en `worst_claimant_fill`, con SESOI `+0,010`.

## Contraste primario

| régimen | dwell | media | LCB95 | UCB95 |
|---|---:|---:|---:|---:|
| base | 6 | −0,00157 | −0,00571 | +0,00282 |
| base | 12 | −0,00469 | −0,00868 | −0,00097 |
| freq3_imp2 | 6 | +0,00084 | −0,00305 | +0,00589 |
| freq3_imp2 | 12 | −0,00121 | −0,00656 | +0,00498 |

Ningún contraste muestra una prima material; todos los LCB quedan por debajo del SESOI.

## Guardarraíl que detuvo el preflight

El daño se calculó pareado contra el incumbente, con `UCB95(daño) ≤ δ`:

| régimen | dwell | guardarraíl | UCB95 | δ | estado |
|---|---:|---|---:|---:|---|
| base | 6 | backorder relativo | 0,01164 | 0,010 | **falla** |
| base | 12 | backorder relativo | 0,01028 | 0,010 | **falla** |

Los demás guardarraíles de esas celdas, y todos los guardarraíles del régimen `freq3_imp2`, pasan.
No se relajan los márgenes después de observar el resultado.

## Falsadores y potencia

Pasan `f1`, `f2`, `f3`, `f4`, `f5`, `f6` y `f7`; `f9` falla. `f8` queda `NOT_APPLICABLE` porque es una réplica declarada
de semillas quemadas, no una prueba de virginidad.

La potencia observada está dentro del presupuesto fijado: el `required_n_max` por celda es
10, 11, 12 o 18 frente al máximo de 96. Esto no rescata el candidato: la detención se debe al
guardarraíl y el contraste tampoco supera el SESOI.

## Alcance de la conclusión

El resultado cierra este preflight v2 bajo sus niveles, política de histéresis, comparador,
observaciones, márgenes y tapes. No demuestra que ningún acoplamiento temporal pueda producir
valor en general, pero tampoco deja base para abrir una confirmación o entrenar una red. Cualquier
nueva física requeriría una nueva hipótesis, enmienda y autoridad explícitas; no otro ajuste de
niveles o márgenes para rescatar este screen.
