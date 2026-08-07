# Enmienda — DDMRP sale del dominio compartido, y la asimetría se declara

**Escrita ANTES de correr.** Runner: `scripts/run_ddmrp_unprojected_v1.py`.
Corrida que corrige: `results/step3_pooled/result.json` (`f6` FALLÓ).
Tapes: **los ya materializados** en `results/step3_s*/full/<familia>_actual_tapes.json`, bloque
`1.420.001+`. **No se abre ninguna semilla.**

## 1. Lo que midió la corrida anterior, y no era DDMRP

`f6` falló: el brazo emite **una sola postura**, `(1344, 1344, 504)`, en los 78 puntos de decisión.
El diagnóstico inicial —*«corre sobre un número mágico»*— **era falso** y queda retractado aquí: el
ADU de reserva (30.000) sólo actúa en las tres primeras épocas; después hay medición real y el
`top_of_green` de `op3_rm` toma 13 valores distintos entre 62.532 y 1.890.000.

**La causa es una incompatibilidad de escala:**

| nodo | on-hand medio | objetivo DDMRP medio | techo de la escalera |
|---|---:|---:|---:|
| `op3_rm` | **3.412.581** | 3.703.351 | **122.880** |
| `op5_rm` | 201.862 | 764.574 | **122.880** |
| `op9_rations` | 42.066 | 45.768 | 126.000 |

El inventario real de materia prima corre en millones y la escalera de la Tabla 6.16 tapa **28×
por debajo**, así que `nearest_posture` aplasta el objetivo contra el peldaño superior siempre.
`op9_rations` es la única coordenada donde las escalas casan — y es la única que no está pegada
arriba (504, no 1344).

## 2. Por qué NO se amplía la escalera

Ampliarla hasta que quepan 3,7 millones exige peldaños de ~40.000 horas: **más de cuatro años de
buffer**, que no es una postura de cadena de suministro. Y obligaría a re-enumerar las 216
estáticas sobre el dominio nuevo, invalidando el incumbente contra el que ya se midió el MPC.

Hay además un hecho medido que dice que esa vía no paga: **la materia prima mueve 4,56 M de
unidades para exactamente cero ReT** (`results/authority_ladder/`).

## 3. Lo que se hace, y la asimetría que crea

DDMRP escribe sus **objetivos continuos** directamente en `inventory_buffer_targets`, sin
proyectar. Es el método real, sin mutilar.

**Esto le da a DDMRP estrictamente MÁS derechos de decisión que a los demás brazos**, y esa
asimetría es la que gobierna cómo se lee el resultado:

* **Si DDMRP gana** → **no es evidencia de que el método sea superior.** Está confundido con un
  conjunto de acciones más amplio, y se reporta así: *«gana con derechos que los demás no
  tienen»*. Para convertirlo en un claim de método haría falta darle el mismo dominio ampliado a
  las estáticas y al MPC.
* **Si DDMRP pierde** → **es evidencia MÁS fuerte que bajo derechos iguales**, porque tuvo más
  libertad y aun así no batió a la mejor postura fija.

La asimetría entra al artefacto como campo, no como nota al pie.

## 4. Reglas de lectura, fijadas antes de mirar

Contraste pareado por tape contra **la misma mejor de las 216 posturas estáticas** ya calculada.
Métrica `ret_excel_full_ledger`.

* `LCB95 > 0` → **`DDMRP_WINS_WITH_WIDER_RIGHTS`** (no «DDMRP converts»).
* IC cruza cero → **`DDMRP_INDISTINGUISHABLE_UNDER_WIDER_RIGHTS`**.
* `UCB95 < 0` → **`DDMRP_LOSES_EVEN_WITH_WIDER_RIGHTS`**.

## 5. Falsadores

| falsador | por qué puede fallar |
|---|---|
| `f1_the_targets_actually_vary` | el objetivo continuo debe tomar **más de un valor por nodo**. Es `f6` de la corrida anterior, ahora sin la proyección que lo aplastaba. **Falla si el método era constante por sí mismo**, en cuyo caso la proyección no era la causa |
| `f2_the_targets_leave_the_old_ceiling` | al menos un objetivo debe superar el techo de la escalera. Falla si nunca lo hace, y entonces sacarlo del dominio no cambiaba nada |
| `f3_the_tapes_are_the_same_ones` | los seeds deben coincidir exactamente con los de `step3_pooled`; el contraste es pareado contra el incumbente ya calculado |
| `f4_the_asymmetry_is_recorded` | el artefacto debe llevar el campo de asimetría de derechos. Falla si se reporta como comparación simétrica |
| `f5_no_fresh_seeds` | tapes materializados, ninguna semilla nueva |

**Alcance:** desarrollo sobre los tapes ya abiertos. No adjudica el paso 4 y **no autoriza llamar
a esto una comparación de métodos con derechos iguales**.
