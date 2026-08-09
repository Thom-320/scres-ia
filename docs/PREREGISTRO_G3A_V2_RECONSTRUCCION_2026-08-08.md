# Preregistro — reconstrucción de G3a con custodia, sobre semillas nuevas

**Fecha:** 2026-08-08. **Congelado antes de escribir el runner.**

## 1. Por qué se reconstruye en vez de citarse

El paquete externo `Garrido_CIE_development_package_v1_1` describe una campaña G3a de 18.360
episodios cuyo hallazgo es el más fuerte del portafolio: bajo **cuotas rígidas** la adaptación
observable vale `H_obs = 0,0963` [0,0682, 0,1245], y **desaparece** al permitir reasignación de
sobrante (0,00126) y es **exactamente 0** bajo pooling FIFO global.

Su propio manifiesto declara `g3a_code_and_raw_results_in_remote_head: false`: el runner, el
contrato y los resultados crudos se borraron antes de empujarse. **Existe sólo como prosa.** No se
puede replicar, auditar ni superseder, así que por la regla de este repositorio **no es evidencia
citable**. Y sus semillas `8701001–8701060` no estaban en el registro; ya quedan marcadas
`ATTEMPTED_NO_SEALED_ARTIFACT`.

**Los números del CSV son objetivos a reproducir, no evidencia.** Si esta reconstrucción no los
reproduce, eso es el resultado, y es mejor descubrirlo aquí que en revisión.

## 2. Excepción del PI para semillas nuevas

El registro está en `new_seed_opening: False`. El PI autorizó explícitamente esta apertura
(«reconstruye G3a con semillas nuevas y córrelo»). Bloque **`8800001–8800060`**, verificado libre
antes de escribir. 30 de selección (`8800001–8800030`) y 30 de evaluación (`8800031–8800060`).

Son semillas de **desarrollo**, no de confirmación: la partición 30/30 es un holdout de desarrollo
y no otorga grado confirmatorio a nada.

## 3. El mecanismo, declarado como extensión nuestra

Dos CSSU (A y B) comparten una capacidad diaria fija de despacho — física que **ya existe** en el
simulador (`cssu_topology_mode="split_v1"`). Lo que se añade:

* **Régimen latente** de tres estados `{A_presión, N, B_presión}` con auto-transición **0,78** en las
  celdas persistentes; la celda `iid` sortea el estado cada semana de forma independiente. El régimen
  **no se observa**.
* **Mezcla de demanda A/B dirigida por el régimen**, aplicada a través de
  `stable_cssu_destination_weighted`, que transforma un uniforme **con clave de evento** y por tanto
  **no consume ningún RNG del simulador** ni desplaza ninguna extracción posterior. Cuota del
  reclamante presionado **0,75**; `N` reparte 0,5/0,5.
* **Aviso observable** con exactitud **0,72** sobre el régimen, disponible **antes** de decidir.
* **Contratos de capacidad**, tres: cuotas rígidas (`reallocate_unused=False`), reasignación de
  sobrante (`True`), y **pooling FIFO global**, que atiende una sola cola por antigüedad y es
  **invariante a la acción de asignación por construcción**.

Nada de esto viene de Garrido-Ríos (2017): la tesis modela un CSSU agregado. Es extensión nuestra,
declarada, y ningún resultado se presenta como reproducción de la fuente.

## 4. Endpoint

`worst_claimant_late_exposure_service_v1`, el del manuscrito, reimplementado desde su definición:
para cada pedido elegible se integra la cantidad pendiente desde su fecha límite hasta cada
fragmento de entrega o el cierre de medición; se normaliza por la exposición tardía máxima posible;
se convierte a servicio como `1 − exposición normalizada`; y se toma **el peor** de A y B. Mayor es
mejor.

**Se declara post-hoc, como en el original**: nació después de descubrir que el contrato de 48 h es
incompatible con el retardo de 54 h registrado, que hace el fill puntual idénticamente cero bajo
toda política. No es una medida SCRES validada y no puede sostener un claim confirmatorio.

## 5. Diseño: 9 celdas × 14 controladores × 60 tapes

**Celdas:** `{iid/uniforme, persistente/uniforme, persistente/estacional}` × `{cuota rígida,
reasignación de sobrante, pooling global}`.

**Biblioteca de controladores, enumerada aquí y cerrada.** Se reduce de los 34 del original a **14**
declarados, cubriendo las mismas familias, porque el propósito es la frontera entre familias y no
el barrido dentro de cada una:

* 5 constantes (`allocation_a ∈ {0,3, 0,4, 0,5, 0,6, 0,7}`);
* 2 de aviso (lookup directo e invertido);
* 2 placebos (**aviso barajado** y **aviso retrasado**), que conservan el mecanismo y destruyen la
  información;
* 2 de creencia (con estado y su ablación **reset**), ambos con **un único modelo de transición
  común a todas las celdas** — nunca la matriz generadora de la celda;
* 1 de demanda rezagada, 1 de umbral de backlog;
* 1 **privilegiado** de estado verdadero, marcado como diagnóstico: **no es desplegable, no es
  óptimo y no es cota**.

**Selección en el bloque de entrenamiento, congelación y evaluación en el de test.** Ésta es la
corrección que el auditor interno del paquete impuso sobre su primer análisis, que usaba un máximo
retrospectivo por tape de test.

## 6. Falsadores, y por qué cada uno puede fallar

| id | exige | por qué puede fallar |
|---|---|---|
| `f1_pooling_is_action_invariant` | bajo pooling global, el servicio no cambia con `allocation_a` | si cambia, el pooling no está implementado como se declara y la celda nula no es nula |
| `f2_action_is_live_under_hard_quota` | cambiar la asignación mueve al menos un lado | una acción inerte haría de todo lo demás ruido |
| `f3_mass_conserves` | residual de flujo relativo < 1e-6 | ya destruimos inventario en silencio una vez hoy |
| `f4_demand_tape_is_identical_across_policies` | dispersión de demanda dentro de tape = 0 | sin CRN, el ruido supera al efecto |
| `f5_belief_uses_one_common_model` | el modelo de creencia desplegable no depende de la celda | usar la matriz generadora es fuga de la celda experimental |
| `f6_placebos_lose_under_hard_quota` | el aviso barajado pierde contra el aviso real | si empata, lo medido es cadencia y no información |
| `f7_hard_quota_shows_observable_headroom` | `LCB95(H_obs) > 0` bajo cuota rígida | **puede fallar**: sería no reproducir el hallazgo del paquete |
| `f8_work_conserving_kills_it` | `H_obs` bajo pooling global < 0,01 | si el pooling también muestra headroom, el mecanismo causal propuesto es falso |
| `f9_forfeiture_is_measured` | se reporta capacidad desperdiciada por contrato | sin ella, una prima por no desperdiciar el camión se vende como adaptación |

## 7. Reglas de lectura, en orden

1. **Primero la invariancia y la física.** Si `f1`, `f3` o `f4` fallan → `BLOCKED_INSTRUMENT` y
   nada más se lee.
2. `f7` **y** `f8` juntos definen el hallazgo: adaptación observable **dentro** de la cuota rígida
   que **desaparece** al conservar trabajo. Uno sin el otro no es el resultado.
3. Si `f7` falla, el veredicto es `G3A_DID_NOT_REPRODUCE`, y el hallazgo del paquete queda sin
   respaldo reproducible.
4. **Nada de esto autoriza entrenar.** Un `H_obs` positivo bajo un contrato dominado no es una
   puerta a un aprendiz: es una razón para arreglar el contrato.
