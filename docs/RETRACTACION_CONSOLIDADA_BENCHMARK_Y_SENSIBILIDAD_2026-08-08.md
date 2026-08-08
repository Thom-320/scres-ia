# Retractación consolidada — el benchmark KAN–MLP v2 y la sensibilidad riesgo por riesgo

Dos auditorías externas. Verifiqué **todas** sus afirmaciones comprobables contra el código y los
artefactos antes de escribir esto. **Ninguna resultó falsa.** Los dos artefactos se conservan; lo
que se retira son sus etiquetas y varias frases mías.

## 1. `results/kan_mlp_r2_benchmark_v2/result.json` (sello `627448cf`)

**Se retira** `EQUIVALENT_BY_TOST_CHOOSE_MLP_BY_PARSIMONY`. Queda:

```text
NEITHER_ARCHITECTURE_BEATS_TRAIN_SELECTED_CALENDAR
ARCHITECTURE_EQUIVALENCE_NOT_ESTABLISHED
retained: ZERO_DOWNSTREAM_DIFFERENCE_ON_FOUR_DECLARED_REPLAY_TAPES
```

**El espacio de decisión colapsó a un bit.** Once calendarios contienen la semana 4 y son
simultáneamente óptimos en 22 de 24 tapes; el calendario fijo 0 es óptimo en 7 de 8 tapes de test y
en la excepción pierde 0,001012. El hueco open-loop–clarividente es 0,000253 bajo uniforme y
**exactamente cero** bajo exponencial. La causa está en el contrato: el buffer se repone al
instante, apagarlo después no revierte el inventario ya añadido, y `holding_cost = 0`. La pregunta
efectiva es *«¿activa el buffer en la primera semana elegible?»*. **Mi `f7` sólo verificaba que
existiera spread entre algún calendario temprano y alguno tardío; no detecta que la dimensión
decisional efectiva sea uno.**

**El orden del veredicto estaba invertido.** El runner evalúa equivalencia **antes** que la
ausencia de ventaja sobre el open-loop, así que una igualdad arquitectónica en una tarea sin
headroom desplazó al veredicto causalmente relevante. Debe ser: primero *¿hay headroom sobre el
comparador estático?*; sólo si lo hay, *¿difieren las arquitecturas?*. Aquí la primera respuesta es
no, y por tanto **la segunda no es identificable**.

**No es un TOST.** No ejecuta las dos pruebas unilaterales: es una regla de inclusión de intervalo
sobre cuatro tapes, con las diez inicializaciones **promediadas** de antemano —su incertidumbre
desaparece en vez de entrar por inferencia jerárquica— y con un bootstrap que degenera a `[0, 0]`
porque los cuatro valores son exactamente cero. Remuestrear cuatro ceros no cuantifica
incertidumbre sobre tapes no observadas. Lo admisible es *«en cuatro tapes de replay las pérdidas
observadas fueron idénticas»*, no *«se demostró equivalencia»*.

**«Nueve falsadores computados y aprobados» es falso**: `f6` estaba hardcodeado a `True`, `f9`
devuelve `not_applicable` con `passed = null` —no «pasa»—, `f4` sólo comprueba que una constante
sea ≥ 10 y `f5` sólo que tres listas midan tres.

**Faltan piezas del preregistro**: la regla simple o belief-MPC, la memoria pico, y los `picks`.
Sin los `picks`, ni siquiera puede verificarse mi frase *«ambas redes encuentran el óptimo»* — sólo
que obtienen una pérdida perteneciente a la meseta.

## 2. `results/per_risk_sensitivity/result.json` (sello `0e050e3d`)

**Se retira** `PER_RISK_SENSITIVITY_MEASURED_WITH_EXPOSURE_DISCLOSED`. Queda:

```text
PARTIAL_DIAGNOSTIC_ONLY
TERMINAL_LEAVE_ONE_RISK_OUT_EFFECTS_MEASURED_ON_12_REPLAY_SEEDS
TEMPORAL_RISK_SENSITIVITY_NOT_ESTABLISHED
```

**La curva de eventos está mal etiquetada.** `traj["events"]` usa `len(sim.risk_events)`, que cuenta
**todos** los riesgos. La curva rotulada `R24` no son eventos R24: son eventos totales con R24 en
×4. La figura hay que retirarla.

**Y los conteos no son comparables entre riesgos.** R24 se registra al ocurrir; R11/R21/R22/R23 al
terminar su recuperación; R12/R13 al completar el retraso; R14 instantáneamente y con una magnitud
de defectos por evento. **Mi frase «R14 dispara 157 veces y aporta menos que R23, que dispara 0,75»
es inválida**: un evento R14 es un día con una cantidad variable de defectos y uno R23 es una
interrupción completa de Op11. Hacen falta horas-operación afectadas, cantidad afectada y severidad
acumulada, no conteos crudos.

**`×4` tampoco es una escalera uniforme**, y no empieza a romperse en ×16: en R11 y R21–R24 el
multiplicador mueve el intervalo entre eventos; en R12–R14 mueve la probabilidad binomial, y con
ella incidencia, número de contratos afectados y duración.

**El leave-one-out no parte del mismo estado inicial.** Encender o apagar el riesgo ocurre **antes
del warm-up**, así que el tratamiento cambia la inicialización. La diferencia atribuida a un riesgo
puede incluir inventario y backlog heredados. Y `L(todos) − L(todos menos r)` es un marginal
**condicional al portafolio completo**, con interacciones incluidas — no la «contribución propia»
que yo dije.

**`R13|x16` es un fallo de reset, no una región difícil.** Lo verifiqué: tras `reset()` y **antes
del primer `step()`**, `env.now = 161.280 h` contra un horizonte de 4.368 —**37 veces**—,
`warmup_complete = False`, y 2.254 eventos ya disparados. Los 958 eventos R13 pertenecen a una
inicialización fallida. Es una celda **inelegible**, y describirla como «demasiado difícil» fue
incorrecto.

**Y `"inert_even_at_x16": ["R12"]` es incorrecto.** Cero eventos *registrados* no es cero
ocurrencias: R12 se registra al completarse, y puede haber interrupciones largas todavía activas al
cierre del horizonte.

**El escenario no es el de v2**: la sensibilidad volvió en silencio a `thesis_uniform` mientras v2
usaba `garrido_seasonal_v1`. No es la sensibilidad del mismo experimento.

**«Siete falsadores pasan» vuelve a estar inflado**: `f3` y `f4` hardcodeados a `True`, `f7`
`not_applicable`, `f5` y `f6` sólo miran `trajectory_seed0`. Y **retirar ×16 del claim después de
ver que hacía fallar los gates es una enmienda post-resultado**, no una validación independiente.

## 3. Lo que sí sobrevive, con su alcance

> En doce semillas de desarrollo reutilizadas y bajo una política fija, retirar **R24** del
> portafolio produjo la mayor reducción observada de `L*`: **+0,029963**, positiva en **12 de 12**
> semillas. R23 es positiva en 4, cero en 7 y negativa en 1. **R21 es compatible con cero**, no con
> un efecto negativo interpretable.

Identifica a **R24 como el principal candidato a confirmación**. No establece dominancia general ni
sensibilidad temporal.

## 4. El estado real de las peticiones de Garrido

```text
#1  Q1 (qué CATEGORÍA de IA)            OPEN
#2  R1 quieto / R2 modificado           R2_FAMILY_CHANGE_IMPLEMENTED
                                        ARCHITECTURE_EQUIVALENCE_NOT_ESTABLISHED
#7  sensibilidad riesgo por riesgo      PARTIAL_TERMINAL_DIAGNOSTIC
                                        TEMPORAL_REQUEST_NOT_COMPLETED
```

## 5. Por qué se repiten estos errores, y el arreglo mecánico

Los diecinueve defectos de hoy tienen **cuatro causas**, no diecinueve:

1. **Escribo un falsador que atrapa una clase de error y no lo llevo al siguiente runner.** El de
   degeneración del espacio de decisión se añadió dos veces y se omitió la tercera, justo donde
   hacía falta.
2. **Hardcodeo `passed: True` y lo llamo «divulgación»**, existiendo una memoria titulada
   `falsifier-must-be-seen-to-fail` escrita porque eso mismo dejó pasar una fuga de datos. Y luego
   cuento esos falsadores en el total que reporto.
3. **Narro números de memoria en lugar de releer el artefacto** (15 contra 16; 0,11 contra 0,09465).
4. **Lanzo y luego descubro**, sin pre-vuelo. El grid de posturas muerto, el contrato estacional no
   cableado y el viaje en el tiempo habrían caído en treinta segundos de comprobación previa.

La causa raíz es una: **optimicé por producir un artefacto sellado por turno en vez de uno
correcto.**

El arreglo no es prometer más cuidado. Es mecánico y va antes de cualquier corrida nueva:

* **`supply_chain/falsifiers.py`**, módulo compartido donde cada comprobación se define **una vez** y
  se hereda, de modo que un falsador aprendido no se pueda dejar fuera del siguiente runner;
* **`passed` debe computarse de datos** — un literal `True` es un error de construcción y el sellador
  lo rechaza;
* **el recuento reportado excluye** `not_applicable`, y las divulgaciones van en su propio campo,
  nunca en el total de falsadores;
* **bloque de pre-vuelo obligatorio** antes de toda corrida cara: el endpoint responde a la acción,
  el espacio de decisión tiene más de una dimensión efectiva, el reset deja `env.now` dentro del
  horizonte, y el escenario es el declarado.

## 6. Sobre la Q1, y por qué no la corro todavía

Comparar familias distintas —surrogate supervisado contra búsqueda contra control basado en
modelo— es lo correcto para su pregunta. Pero **ninguna comparación de familias es identificable
mientras el mejor calendario fijo iguale al clarividente**: eso es exactamente lo que acaba de
pasar, y correrla ahora repetiría el error por cuarta vez.

El orden es: **gate de elegibilidad primero** —`oracle − open_loop_elegido_en_train ≥ SESOI`, más un
falsador de complejidad efectiva que detecte cuándo un solo bit del calendario predice la clase
óptima—; y sólo si pasa, la comparación de familias. Este benchmark habría fallado los dos.
