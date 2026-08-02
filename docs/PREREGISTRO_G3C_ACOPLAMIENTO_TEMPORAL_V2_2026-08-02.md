# Preregistro G3c v2 — permanencia mínima con dos reclamantes

**Estado:** `DESIGN_ONLY_NOT_AUTHORIZED_UNTIL_SUBMISSION_A_RECEIPT`
**Contrato machine-readable:** `contracts/g3c_temporal_coupling_v2.json`
**Sustituye al preregistro anterior sólo para los bloqueadores 1 y 3.** La enmienda de márgenes
operacionales de 2026-08-02 sigue siendo la fuente de los márgenes del bloqueador 2.

## 1. Una sola física

G3c v2 estudia únicamente **permanencia mínima**. El parámetro es:

```text
cssu_min_dwell_days ∈ {1, 3, 7}
```

`1` es el nulo/regresión del modelo legacy: con la latencia de activación de 24 horas y la
cadencia diaria, no introduce una restricción adicional. `3` y `7` son los dos niveles de
acoplamiento temporal fijados antes de cualquier semilla.

`switch_cost_rations` queda **fuera de este contrato**. El código puede conservar una implementación
experimental no registrada, pero no se ejecutará ni se mezclará con G3c. Si se quiere estudiar
coste de cambio, deberá existir otro preregistro con otra familia física, niveles y potencia.

La permanencia retrasa una acción solicitada hasta que venza el dwell; no la cancela. La acción
mantiene la latencia física de 24 horas. El contrato conserva `N=2`, `split_v1` y la acción
`cssu_allocation_a`.

## 2. Estimando y comparador

El endpoint primario es `worst_claimant_fill`, con SESOI absoluto `+0,010`. El incumbente no es
la mejor constante: es la **mejor regla miope equivariante** obtenida bajo el contrato. La
constante se conserva sólo como contexto.

La escalera es:

```text
constante → regla miope equivariante → histéresis → tabular → DP/rollout → MPC
```

Si la histéresis, el DP o el MPC capturan el residual, el resultado terminal es estructurado y
no se entrena ninguna red.

## 3. Márgenes

Se heredan de [ENMIENDA_G3C_MARGENES_OPERACIONALES_2026-08-02.md](ENMIENDA_G3C_MARGENES_OPERACIONALES_2026-08-02.md):

| guardarraíl | margen |
|---|---:|
| `flow_fill_rate` | `0,005` |
| `lost_orders` | `0,50` pedidos/episodio |
| `backorder_qty_final` | `1,0 %` relativo |
| masa, capacidad, recursos algebraicos | `0,0` exacto |

La potencia se calcula sobre tapes burned antes de abrir cualquier bloque nuevo, con corrección
simultánea sobre las celdas. Si la potencia requerida excede el presupuesto, el estado terminal es
`STOP_G3C_UNDERPOWERED`.

## 4. Identidad del nulo: ahora es un test ejecutable

El brazo `cssu_min_dwell_days=1` debe reproducir el modelo legacy explícito. La comparación no
usa `self_sha256`, porque éste contiene timestamps y provenance. Usa
`canonical_scientific_payload_sha256` sobre:

```text
orders · risk_events · actions · ledgers · metrics
```

El hash canónico excluye sólo metadatos de envelope: `created_at`, duración, provenance, manifiesto,
contrato, referencia y `self_sha256`. `f1` falla si una sola cantidad científica cambia.

Los tests unitarios ejecutan el mismo tape con el default legacy y con el nulo explícito. También
verifican que una mutación científica cambia el hash y que `min_dwell=7` realmente bloquea cambios.

## 5. Falsadores y mutantes

Los nueve falsadores del contrato quedan congelados en el JSON. En particular:

- ignorar `min_dwell` debe hacer fallar el falsador de acoplamiento;
- sustituir la regla miope por una constante debe hacer fallar el falsador del incumbente;
- poner margen cero en un guardarraíl estocástico debe hacer fallar el falsador de márgenes.

Ninguno autoriza semillas ni entrenamiento. G3c sigue en diseño hasta superar la frontera de
autoridad vigente.

## Estado

```text
BLOCKER_1: RESOLVED — una sola física, niveles {1,3,7}
BLOCKER_2: RESOLVED — márgenes operacionales en enmienda separada
BLOCKER_3: RESOLVED — payload científico canónico y f1 ejecutable
G3c: DESIGN_ONLY_NOT_AUTHORIZED
```
