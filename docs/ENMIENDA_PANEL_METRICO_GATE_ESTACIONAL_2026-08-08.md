# Enmienda — panel métrico y eje de riesgo escalado, como familia sucesora

**Escrita ANTES de correr el sucesor.** Contrato madre:
`docs/PREREGISTRO_GATE_HEADROOM_ESTACIONAL_R2_2026-08-08.md`. Runner:
`scripts/run_seasonal_r2_headroom_gate_v2.py`. Misma custodia: réplica declarada de
`reconciled_8600001` (`8600001–8600012`).

## 0. Esto NO reabre el gate cerrado

`results/seasonal_r2_headroom_gate/result.json` (sello `5bb556d3…`) devolvió
**`STOP_NO_HEADROOM_UNDER_GARRIDO_PHYSICS`** con `flow_fill_rate` en **0,00000 exacto** en las
cuatro celdas, ocho falsadores en verde. **Ese resultado queda como está y no se retira.** Si el
panel de aquí abajo encontrara algo, no borra aquel cero: lo sitúa.

El §7 del contrato madre dice que cualquier variante posterior de la física **es una familia nueva,
declarada y con su multiplicidad pagada**. Esto es esa familia, y paga.

## 1. Por qué se añaden endpoints, y por qué eso no es metric shopping

La distinción importa y la nombró un auditor independiente en la misma revisión que propone la
métrica: *«cambiar el endpoint después de observarlas sería metric shopping»*. De acuerdo. Por eso
la razón para añadir éste **no es que los otros fallaran**, sino un **defecto de mecanismo que los
tres endpoints anteriores comparten**, y que está medido:

* **`ret_excel` premia el abandono.** Su reparto óptimo entrega **50,7 %** de fill y renuncia a
  **318.621** raciones, contra el reparto 0,5 que entrega 79,5 % y no renuncia a ninguna.
* **Cobb-Douglas pasa el contraste de abandono pero no cierra el mecanismo.** `ε` es la media de
  backorders **pendientes**: cuando un pedido se pierde sale de la cola *y* deja de generar coste,
  así que ni `ε` ni `κ` lo penalizan. Es el mismo defecto un nivel más abajo, y el crédito de
  haberlo visto es del auditor.
* **`flow_fill_rate` es inmune al abandono pero es un ratio de nivel**: no distingue un pedido
  servido con dos semanas de retraso de uno servido a tiempo, y ahí es donde vive el riesgo.

**El déficit operacional no tiene ninguno de los tres problemas, y ya está implementado.**
`service_loss_auc_ration_hours` recorre **todos** los pedidos y para uno no servido toma
`end = horizonte`, ponderando por cantidad — verificado en `episode_metrics.py:206-214`. Un pedido
abandonado acumula por tanto **la penalización máxima posible**. Abandonar nunca puede mejorar el
score, que es la propiedad que ninguna de las otras tres tiene.

## 2. El panel — cinco endpoints, y sólo dos deciden

| papel | endpoint | dirección | por qué |
|---|---|---|---|
| **decide** | `service_deficit` = `service_loss_auc_ration_hours / demanded_rations` | menor mejor | inmune al abandono, ponderado por cantidad, denominador fijo sobre la demanda generada |
| **decide** | `service_deficit_es10` — Expected Shortfall del decil peor entre semillas | menor mejor | la cola, que es donde el riesgo actúa; **con 12 semillas la cola son 1–2 tapes y eso se declara aquí, no después** |
| reporta | `flow_fill_rate` | mayor mejor | el primario del gate madre; su cero entra a la tabla |
| reporta | `R_cobb_douglas` | mayor mejor | el secundario que el gate madre declaró **no computado**; ahora se computa, con su diagnóstico de independencia de κ̇ |
| reporta | `ret_excel` | mayor mejor | comparabilidad histórica; **nunca decide** |

## 3. El eje de riesgo se amplía a tres niveles

El contrato madre tenía `R_fixed` y `R_draw`. Se añade **`R_esc`**: los riesgos de la familia
puestos en `increased`, que es la escalada de la propia tesis. Queda `2 demandas × 3 riesgos = 6
celdas`.

Motivo declarado: el screen de perfiles midió escalada **sin** demanda estacional, y el gate madre
midió demanda estacional **sin** escalada. La celda `D1 × R_esc` no la ha visto nadie, y si la
contención es el mecanismo que produce headroom —lo único que lo ha producido en este proyecto,
`H_PI = 0,1515` bajo recurso no fungible— es la celda donde más apretado está el sistema.

## 4. Multiplicidad

**`K = 5 endpoints × 6 celdas = 30`.** Holm-Bonferroni sobre las 30. Barra por test:
`LCB95 ≥ 0,01` **y** batir el placebo p95, igual que el gate madre.

Los dos endpoints que deciden no obtienen trato preferente en la corrección: se corrigen contra las
30, incluidos los tres que sólo se reportan. Corregir sólo los que deciden ocultaría que se miraron
cinco.

## 5. Falsadores nuevos

| falsador | qué exige | por qué puede fallar |
|---|---|---|
| `f10_deficit_penalises_abandonment` | entre posturas de la misma celda, el déficit debe ser **mayor** donde el fill es **menor**; correlación de rangos negativa y material | si el déficit no se moviera en contra del fill, la propiedad de anti-abandono sería teórica y no medida, y el endpoint entero se retira |
| `f11_es10_tail_is_declared_thin` | se reporta el número de tapes en la cola del decil | no puede fallar como test; es una **divulgación obligatoria** y se marca como tal en vez de disfrazarse de falsador |
| `f12_kappa_independence_reported` | por celda, `corr(ln κ̇, ln ζ)` y `corr(ln κ̇, ln ε)` | hoy quedó medido que bajo `c = 1` κ̇ es un duplicado de ζ+ε con `corr = 0,999993`. Si vuelve a pasar aquí, el endpoint Cobb-Douglas se lee **con esa advertencia pegada**, nunca solo |

Siguen vigentes `f1`–`f9` del gate madre, incluido **`f9`**, que existe porque una rejilla de
posturas muerta ya se leyó una vez como un nulo medido.

## 6. Reglas de lectura, fijadas de antemano

* **Ningún endpoint que decide cruza** → `STOP_NO_HEADROOM_ACROSS_THE_METRIC_PANEL`. El cero de
  `flow_fill_rate` deja de ser un endpoint y pasa a ser un panel de cinco, sobre seis celdas de
  física, con multiplicidad pagada. Es el cierre más fuerte que este proyecto puede escribir sobre
  las dos peticiones de Garrido.
* **Cruza `service_deficit` o `service_deficit_es10`, batiendo placebo** →
  `CEILING_OPEN_ON_OPERATIONAL_DEFICIT`. Se reporta la celda, la descomposición demanda × riesgo y
  el panel entero al lado. **Autoriza diseñar una confirmación. No autoriza entrenar.**
* **Cruza pero no bate al placebo** → `PERIOD_VARYING_NOT_STATE_VARYING`, igual que en el madre.
* **Cruza sólo un endpoint que se reporta y no uno que decide** → se reporta como **discrepancia
  entre métricas** y no como headroom, porque los que reportan son precisamente los que tienen el
  defecto de mecanismo documentado en §1.

## 7. Corrección de una etiqueta previa

El auditor tiene razón en que `NO_HEADROOM_EVEN_UNDER_A_SOUND_METRIC` es demasiado fuerte:
Cobb-Douglas pasa el contraste de abandono **probado**, no es «sano» en general, y su port impone
un suelo `x = 1` que la fuente no especifica —material para `τ`, que es exactamente 0 en 88 de 108
episodios de calibración—. La etiqueta pasa a:

> **`NO_MATERIAL_REGIME_HEADROOM_UNDER_A_SOURCE_FAITHFUL_PORT_THAT_PASSES_THE_TESTED_ABANDONMENT_CONTRAST`**

Se corrige en el lock y en el manuscrito, con el motivo, y el número no cambia.

## 8. Alcance

`DEVELOPMENT_ON_DECLARED_REPLAY`. No abre semillas, no adjudica, no autoriza aprendices, y no
retira el `STOP` del gate madre.
