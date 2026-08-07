# Enmienda — qué dice y qué NO dice el cierre del Programa O, bajo el objetivo declarado

Fuente: `results/program_o/fixed_clock_hobs_corrective_validation_v1/independent_audit_v1.json`
(`STOP_PROGRAM_O_AFTER_CORRECTIVE_VALIDATION`). **No se edita**, no se re-corre, y **no hay segundo
rescate** — el contrato lo prohíbe y esta enmienda no lo toca.

## 1. Lo que el propio artefacto declara confirmado

```
claim_boundary.classical_primary_ret_advantage_confirmed = true
claim_boundary.safe_joint_h_obs_contract_confirmed       = false
```

**Primaria: PASA en las tres celdas.**

| celda | Δ medio | LCB95 simultáneo | cintas a favor |
|---|---:|---:|---:|
| `rho75_share90` | +0,0985 | **+0,0660** | 44 / 48 |
| `rho90_share75` | +0,0735 | **+0,0430** | 42 / 48 |
| `rho90_share90` | +0,0997 | **+0,0586** | 46 / 48 |

Con **27 de 27 placebos batidos** (LCB95 mínimo +0,00716), **1.451 replays físicos con 0 fallos**,
`unique_scheduled_resource_vectors = 1` —los recursos son literalmente iguales— y las compuertas de
acción y de contrafactual de estado pasando en todas las celdas.

## 2. Lo que falló, con precisión

**Sólo `ret_visible_cvar10`, en 2 de 3 celdas.** Y los dos estimados son **POSITIVOS**:

| celda | estimado | LCB95 simultáneo |
|---|---:|---:|
| `rho75_share90::ret_visible_cvar10` | **+0,0350** | −0,00858 |
| `rho90_share75::ret_visible_cvar10` | **+0,0195** | −0,01551 |

Inferencia: **max-t studentizado de una cola sobre 69 estimandos**, valor crítico **2,836**.

> **El guardarraíl no detectó daño. No consiguió certificar la ausencia de daño** a un nivel
> simultáneo del 95 % repartido entre 69 cantidades. Son dos afirmaciones distintas, y el
> artefacto ya las separa: la ventaja media está confirmada, el **contrato conjunto** no.

## 3. Un caveat de medición que no existía cuando se cerró

`ret_visible_cvar10` se calcula sobre `ret_visible` — **la población visible del workbook, que
censura**. Después de aquel cierre quedó **medido** que esa métrica premia el abandono: el reparto
que la maximiza entrega el 50 % de las raciones y el que la minimiza entrega el 80 %.

Eso **no** rescata nada por sí solo, y hay que decir en qué dirección corta: **no sabemos si el
censurado infla o desinfla la cola**. Lo honesto es que el criterio que cerró el programa se
calculó sobre una métrica que hoy no usaríamos como primaria.

## 4. Lo que la decisión del PI cambia, y lo que no

**Decisión registrada (2026-08-07):** la medida es la **resiliencia**, y la lectura operativa es la
**media**.

**Cambia** — cómo se cita:

> **La conversión observable media del Programa O está confirmada** (LCB95 simultáneo +0,043…+0,066,
> 27/27 placebos batidos, recursos idénticos, 1.451 replays sin fallos). **La seguridad conjunta de
> cola no quedó establecida**: los dos estimados de CVaR10 son positivos y lo que falla es la
> certificación simultánea sobre 69 estimandos.

**NO cambia:**

* **El guardarraíl no se retira retroactivamente.** Se congeló antes de ver los resultados;
  quitarlo ahora convertiría la validación de dominio en herramienta de selección. Se **reporta**.
* **No hay segundo rescate.** El contrato lo prohíbe y sigue prohibido.
* `safe_joint_h_obs_contract_confirmed` **sigue en `false`** y así entra al manuscrito.
* La decisión del PI es **prospectiva**: define el endpoint de campañas futuras. No promueve un
  artefacto pasado.

## 5. Por qué esto es más fuerte que borrar el guardarraíl

Un revisor que vea «quitamos el criterio que fallaba» deja de leer. Un revisor que vea

> *dos estimados positivos, un fallo de certificación simultánea entre 69 cantidades, y el criterio
> calculado sobre una métrica que después medimos censurada*

tiene delante un resultado con su límite dicho por sus autores. **El claim se sostiene por ser
estrecho, no por ser amplio**, y ésa es la única forma en que sobrevive.

## 6. La frase para el manuscrito

> Bajo contención no fungible con reloj fijo, la política de creencia convierte headroom observable
> en ventaja media de resiliencia sobre el mejor comparador clásico, confirmada con recursos
> físicamente iguales y batiendo los 27 placebos. La no-inferioridad **conjunta** en la cola
> (CVaR10) **no quedó establecida** bajo corrección simultánea sobre 69 estimandos, con ambos
> estimados puntuales positivos. Reportamos la conversión media como confirmada y la seguridad de
> cola como abierta.
