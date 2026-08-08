# Preregistro — headroom con presupuesto físico congelado y actuador alineado

**Escrito y commiteado ANTES de escribir el runner y ANTES de correr nada.** Familia nueva; no
enmienda ni reabre a `seasonal_r2_headroom_gate` ni a su `v2`, cuyos STOP quedan como están con el
alcance corregido en `docs/ENMIENDA_ALCANCE_GATE_ESTACIONAL_2026-08-08.md`.

**Custodia:** no hay bloques vírgenes (`ENMIENDA_4`, `new_seed_opening: false`). Esto es
**desarrollo sobre réplica declarada** por construcción y **no puede adjudicar**. El bloque exacto
se nombra en la invocación contra el registro reconciliado; el runner falla si no se declara.

## 0. Por qué existe esta familia

El gate anterior midió `H_regime = 0` en todos los endpoints y en todas las celdas, y el cero era
**del endpoint, no del entorno**: `flow_fill_rate` no cobra recursos, y la misma meseta de servicio
de 0,8404 se compra con **4.368 horas-turno o con 13.104**. Con el recurso gratis, la postura
maximalista domina con independencia del estado y `H = 0` sale por construcción.

De ahí las tres condiciones que el headroom necesita **juntas**, y que esta familia impone en vez
de esperar:

1. **el recurso es escaso** → presupuesto congelado, idéntico para todas las políticas;
2. **su valor marginal cambia en el tiempo** → riesgos con incidencia temporal dentro del episodio;
3. **la política tiene una señal observable para asignarlo** → un actuador alineado con el riesgo.

Program O tenía las tres y midió `H_PI = 0,1515` con el nulo fungible en exactamente 0. Nada más
en este proyecto las ha tenido a la vez.

## 1. Un solo actuador, y por qué

**Sólo turnos.** El inventario queda para una familia posterior y **no se combina** con turnos en
esta. Dos recursos a la vez hacen que un resultado nulo no se pueda atribuir y que uno positivo no
se pueda descomponer.

**El presupuesto** es `B_S = Σ_t (S_t − 1)·Δt`, que ya existe como `extra_shift_hours`
(`episode_metrics.py:314, 321, 328`). No hace falta física nueva: hace falta **no gastarla gratis**.

Referencia: un `S2` constante sobre el horizonte de 26 semanas consume `1 × 168 × 26 = 4.368`
horas-turno extra. Los tres presupuestos congelados son **25 %, 50 % y 75 %** de eso:

| presupuesto | horas-turno extra | semanas de `S2` equivalentes |
|---|---:|---:|
| `B25` | 1.092 | 6,5 |
| `B50` | 2.184 | 13 |
| `B75` | 3.276 | 19,5 |

Con `B25` una política **no puede** estar en `S2` todo el tiempo. Tiene que decidir **cuándo**
gastar la reserva, y ésa es exactamente la pregunta que el gate anterior no podía plantear.

## 2. Clases de política — cinco, todas con el mismo presupuesto

Cada una recibe **exactamente** el mismo `B_S`. Que sean iguales no se asume: lo verifica `f2`.

| clase | qué ve | papel |
|---|---|---|
| `best_feasible_openloop_constant` | nada; reparte el presupuesto uniformemente | referencia baja |
| `best_seasonal_openloop` | sólo `t`; gasta alineado al perfil de demanda | **el comparador que hay que batir** |
| `hazard_backlog_rule` | backlog realizado y riesgo observable | la política observable |
| `uninformed_placebo` | nada; mismo número de semanas de surge, **calendario permutado** | el placebo obligatorio |
| `clairvoyant_schedule` | la tape entera | **techo, y sólo techo** |

**El techo es clairvoyant y por eso sobrestima**, igual que antes. Eso hace fuerte un STOP y débil
un GO: un techo abierto autoriza **diseñar** una confirmación, nunca entrenar.

Y a diferencia del gate anterior, **la clase incluye políticas que varían en el tiempo**, así que
el veredicto que produzca sí habla de calendarios y no sólo de constantes. Ésa era la frase que
tuve que retirar.

## 3. Riesgos con fundamento en la fuente, y dos controles negativos

Verificado en `supply_chain/config.py:469-499`:

| riesgo | qué es | dónde golpea | papel |
|---|---|---|---|
| **R24** | *Contingent demand surge*, 2.400–2.600 raciones, `U(1, 672)` h | op13, teatro | **primario** — un pico de demanda en el punto de consumo; el actuador alineado es reservar capacidad |
| **R21** | *Natural disasters*, recuperación `exp(120 h)`, `U(1, 16.128)` h | ops 3, 5, 6, 7, 9 **simultáneas** | **primario preventivo** — tumba producción aguas arriba |
| **R22** | *LOC destruction* | ops 4, 8, 10, 12, todas las LOC | **control negativo** |
| **R23** | *Forward unit destruction* | op 11 | **control negativo** |

**Niveles discretos de la fuente — `current` e `increased` — y no sorteos continuos de
multiplicadores.** La incertidumbre ya viene de los eventos dentro del episodio; sortear encima el
multiplicador promedia sobre perfiles y diluye justo lo que se quiere ver.

**Los controles negativos son la mitad del diseño.** Si subir turnos «resuelve» R22 o R23 —riesgos
que golpean transporte y unidad avanzada, donde el actuador no llega— hay confusión, y el resultado
en R21/R24 no se puede leer. `f6` lo convierte en criterio.

Demanda: `D0` heredada y `D1` = `researcher_defined_periodic_demand_v1`, con ese nombre y **no**
atribuida a Garrido — la senda realizada es `U(2400,2600) × nuestro perfil de 12 semanas`, y `α, γ`
sólo alimentan el pronóstico (`supply_chain.py:5494-5503`).

Diseño: `3 presupuestos × 2 demandas × 4 regímenes de riesgo` (R24↑, R21↑, R22↑, R23↑) `= 24`
celdas, con 5 clases de política en cada una.

## 4. Endpoints — el primario cobra el retraso y no se puede engañar abandonando

**Primario, menor es mejor:**

```
L_s(π) = ∫ U_s^π(t) dt  /  ∫ D_s(t) dt
```

con `U(t)` = demanda no servida acumulada: backlog, demanda perdida y demanda pendiente. Numerador
implementado como `service_loss_auc_ration_hours`, que recorre **todos** los pedidos y toma
`end = horizonte` para uno no servido, ponderado por cantidad (`episode_metrics.py:206-214`).
Abandonar **nunca** puede mejorarlo, y eso ya está **medido**, no supuesto: `corr(fill, déficit) =
−1,0` en las seis celdas del panel v2.

**Secundarios reportados:** `ES10(L_s)` · raciones entregadas / demandadas · cantidad de demanda
perdida · `shift_hours` · `strategic_buffer_units`.

**`ret_excel` no entra ni como reporte.** Su óptimo de reparto abandona 318.621 raciones, y en el
panel v2 dos de sus celdas quedaron por debajo de dos errores estándar de discriminación.

## 5. Los gates, en orden, y qué autoriza cada uno

**G1 — ¿existe valor de timing?**
`LCB95[ L(best_seasonal_openloop) − L(clairvoyant_schedule) ] ≥ 0,01`
Si no pasa: **STOP, y no se abre ninguna política observable.** Con presupuesto congelado y sin
valor en el timing clarividente, no hay nada que una regla, un MPC o una red puedan capturar.

**G2 — ¿convierte a observable?**
`LCB95[ L(best_seasonal_openloop) − L(hazard_backlog_rule) ] > 0` **y** la regla debe batir al
placebo. Si pasa G1 y falla G2, el veredicto es `TIMING_VALUE_EXISTS_BUT_DOES_NOT_CONVERT`, que es
un resultado y no un fracaso.

**G3 — ¿queda residual para una red?** **Fuera de esta familia.** Se declara aquí para que nadie lo
dé por incluido: exige `LCB95[ L(hazard_rule) − L(belief_MPC) ] > 0`, y sólo entonces se preregistra
un aprendiz. Si belief-MPC captura todo el valor, la ruta neuronal viable es **amortización
computacional** —misma calidad de servicio con menor latencia y menos llamadas al DES— y ésa es
otra familia con sus dos gates propios.

## 6. Falsadores — todos pueden fallar y todos pueden pasar

| falsador | qué exige | por qué puede fallar |
|---|---|---|
| `f1_budget_binds` | en cada celda, al menos una clase agota su presupuesto y ninguna lo excede | si nadie lo agota, el presupuesto no ata y volvemos al recurso gratis — el defecto que originó esta familia |
| `f2_budgets_are_equal_across_policies` | `extra_shift_hours` idéntico entre clases hasta `1e-9` | si difieren, se compara política **y** recurso, que es exactamente lo que invalidó el gate anterior |
| `f3_at_least_three_schedules_are_non_dominated` | ≥ 3 calendarios presupuestados en el frente no dominado | **el falsador que la auditoría exigió**: `f9` del gate anterior pasaba porque existía **un** rincón malo, y un rincón no es una frontera decisional |
| `f4_not_explained_by_one_corner` | quitando el peor calendario, el spread restante debe seguir superando 2 errores estándar | si todo el spread lo aporta un único calendario de baja capacidad, no hay superficie de decisión |
| `f5_placebo_does_not_match_the_rule` | la regla debe batir al placebo con calendario permutado | **el decisivo, y ya falló en op12**: mismo presupuesto y mismo número de semanas de surge, sólo cambia *cuándo*. Si el placebo iguala, el valor está en gastar, no en gastar bien |
| `f6_negative_controls_stay_negative` | en R22 y R23 la regla **no** debe batir al open-loop de forma material | si «resuelve» un riesgo que su actuador no toca, hay confusión y **R21/R24 no se pueden leer** |
| `f7_clairvoyant_dominates` | el techo domina débilmente a toda clase observable, por construcción | un techo que pierde está mal implementado; control de integridad |
| `f8_endpoint_discriminates` | el spread del primario entre calendarios supera 2 errores estándar | heredado, y existe porque una rejilla muerta ya se leyó una vez como nulo medido |
| `f9_no_fresh_seeds` | réplica declarada contra el registro reconciliado | custodia central |

## 7. Multiplicidad y reglas de lectura

**`K = 24 celdas × 2 contrastes que deciden (G1, G2) = 48`.** Holm-Bonferroni sobre las 48. Los
secundarios se reportan y **no** se corrigen contra los primarios, porque no deciden.

* **G1 no pasa en ninguna celda** → `STOP_NO_TIMING_VALUE_UNDER_A_BINDING_BUDGET`. Es el cierre más
  fuerte que este proyecto puede escribir: recurso escaso, riesgos con incidencia temporal,
  actuador alineado, políticas que varían en el tiempo, y aun así el timing clarividente no compra
  nada. **Y no se entrena nada.**
* **G1 pasa y G2 falla** → `TIMING_VALUE_EXISTS_BUT_DOES_NOT_CONVERT`. Se reporta la celda, la
  brecha y el placebo.
* **G1 y G2 pasan, controles negativos limpios** → `OBSERVABLE_TIMING_VALUE_UNDER_EQUAL_BUDGET`.
  Autoriza **diseñar** la confirmación y el contraste contra belief-MPC. **No autoriza entrenar.**
* **G2 pasa pero un control negativo también** → `CONFOUNDED_NO_ADJUDICATION`. No se lee nada de
  R21/R24 hasta explicar por qué el actuador afecta a un riesgo que no toca.

## 8. Lo que queda prohibido, y por escrito

**Cero retuning tras ver el resultado.** Presupuestos, riesgos, endpoint, clases de política,
falsadores y barra quedan fijados aquí. Si cierra en negativo **no se ajusta la física y se vuelve
a correr**: subir el ruido o bajar la capacidad hasta que una arquitectura gane produce un hallazgo
sobre el entorno que construimos, no sobre la cadena.

**No se combina el presupuesto de inventario con el de turnos** en esta familia.

**Y el veredicto nombra la clase que se buscó.** El error de ayer no fue el cero: fue llamarlo
techo general cuando la clase eran posturas constantes. Aquí la clase incluye calendarios, y el
nombre lo dirá con esa precisión y ni un grado más.
