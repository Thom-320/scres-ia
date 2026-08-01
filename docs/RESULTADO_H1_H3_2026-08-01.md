# Resultado — H1 y H3 **no son evaluables** en este entorno, y la razón es el hallazgo

**Artefacto:** `results/manuscript/h1_h3_v1/result.json` (sello `5a01aa2e47aff29c…`,
`HALTED_FALSIFIER_FAILED`) · preregistro `docs/PREREGISTRO_H1_H3_2026-07-31.md`, commiteado antes
de correr · brazos tomados de la corrida **sin fuga** `garrido_meta_learner_v2`.

**Dos falsadores fallan, y cada uno bloquea una hipótesis por una razón distinta.**

## 1. `f1` — el híbrido y el estático despliegan la MISMA configuración

| brazo | configuración desplegada |
|---|---|
| **híbrido** (`neuron_memory`) | `buffer 1344 h · turnos 2 · op9_rop 12 · op12_rop 12` |
| **estático** (`ofat`, el diseño de la tesis) | **idéntica** |
| reinicio (`neuron_reset`) | igual salvo **turnos 3** |

El preregistro fijó qué significaba esto: *«si eligen la MISMA configuración, `H1` y `H3` son
vacías por construcción»*. Los `+0,00` con IC `[0,00, 0,00]` **no son una medición de que no haya
diferencia — son la tautología de comparar algo consigo mismo.**

> **Y eso ES el hallazgo:** la ventaja del aprendiz está en **cuánto tarda en encontrar** el
> óptimo, no en **qué** encuentra. Ambos convergen al mismo punto; la neurona llega en **6,99**
> corridas y OFAT en **12,42**.

Encaja exactamente con el resultado central de la campaña: la superficie tiene **un óptimo
dominante que no se mueve** (`H_regime = 0` bajo las tres métricas, seis regímenes). Con un óptimo
invariante, **ningún buscador puede desplegar algo distinto** — sólo puede llegar antes.

**Consecuencia para el borrador:** `H1` y `H3`, tal como están redactadas, **presuponen que el
aprendiz despliega algo diferente**. Aquí no lo hace. No es que fallen: es que **su premisa no se
cumple en este entorno**, y hay que reescribirlas o declarar el ámbito.

## 2. `f3` — `system_ttr` está censurado al **100 %**

    fracción censurada:  híbrido 1,000   estático 1,000   reinicio 1,000

**Ningún clúster de recuperación se cierra jamás**, así que `system_ttr_mean` vale `0,00` en los
tres brazos **por vacuidad**. `H1` no tiene estimando, con configuraciones idénticas o sin ellas.

**Y aquí endurecí mi propio falsador a mitad de camino, que es lo que corresponde decir.** La
primera versión sólo comprobaba que la censura fuese **comparable** entre brazos (`gap < 0,10`) —
y con los tres al 1,000 el hueco es cero, así que **PASABA**. Comparable no es lo mismo que
utilizable: una medición **totalmente** censurada es idéntica entre brazos precisamente porque no
mide nada. `f3` ahora exige además `censura < 0,999`, y **falla**.

Es el segundo falsador que arreglo hoy por el mismo motivo de fondo: comprobaba la propiedad
adyacente en vez de la que da sentido al número.

## 3. Lo que sí quedó medido

El único contraste con configuraciones realmente distintas es **híbrido vs reinicio** (turnos 2 vs
3):

| `H3` — varianza entre intensidades ×1…×4 | diferencia | IC95 |
|---|---:|---|
| `flow_fill_rate` | +5,48e-05 | **[−1,57e-05, +1,42e-04]** |

El IC **cruza el cero**: `H3` **no sostenida** ni siquiera en el único contraste legítimo. La
varianza de Cobb-Douglas apunta igual (híbrido 2,18e-05 vs reinicio 3,09e-05) pero sin intervalo
sobre esa diferencia.

`f2` **PASA**: la escalera de intensidad sí escala. El instrumento funcionaba; el estimando no
existía.

## 4. Estado de las cuatro hipótesis del borrador

| | estado |
|---|---|
| **H2** adaptación / curva de aprendizaje | **medida** (`garrido_meta_learner_v2`) |
| **H4** dependencia de trayectoria (`L_{t−1}`) | **medida** — el contraste memoria vs reinicio |
| **H1** tiempos de recuperación | **NO evaluable**: `system_ttr` 100 % censurado **y** brazos idénticos |
| **H3** reducción de varianza | **NO sostenida**, y sólo comprobable en un contraste de tres |

## 5. Lo que haría a continuación, y no lo hago sin decidirlo contigo

`H1` necesita **otro estimando de recuperación**, no otro experimento. Candidatos, en orden de
honestidad:

1. **`temporal_maximum_service_drop` + tiempo hasta volver al 95 % del nivel pre-evento**, que se
   define sin depender de que un clúster «cierre»;
2. **`service_loss_auc_ration_hours`**, ya en el panel: integral del servicio perdido, sin
   censura por construcción;
3. arreglar `system_ttr` para que cierre clústeres — es cambiar la definición de un instrumento
   **después** de ver que da 1,000, y por eso lo pongo el último.

`H3` necesita **brazos que difieran**, lo que en este entorno sólo ocurre si el óptimo se mueve —
y hoy quedó medido que no se mueve. Reescribir `H3` como *«el aprendiz reduce la varianza del
COSTE DE BÚSQUEDA entre contextos»* sería fiel al espíritu del borrador y **sí** es medible con lo
que ya tenemos.

---

## Apéndice — reproducción cruzada entre máquinas (2026-08-01)

La misma corrida corregida se ejecutó **en paralelo** en dos arquitecturas distintas: el M1 Pro
local (arm64) y el VPS `ovh-agent-lab` (Intel Haswell, x86-64).

**Coinciden exactamente:**

| | local | VPS |
|---|---:|---:|
| memoria | 6,986111111111111 | 6,986111111111111 |
| reinicio | 14,888888888888891 | 14,888888888888891 |
| OFAT | 12,416666666666666 | 12,416666666666666 |
| aleatoria | 19,541666666666668 | 19,541666666666668 |
| efecto Alzheimer | +7,902777777777778 [6,875 · 8,930555555555555] | idéntico |

Cero diferencias en las seis claves numéricas comparadas, mismo veredicto, y las **secuencias de
configuraciones visitadas son idénticas índice a índice**. El DES, la búsqueda y el bootstrap son
deterministas e independientes de plataforma.

**Los sellos SÍ difieren** (`efb3f067…` local vs `c4381a71…` VPS) y la razón es mundana y hay que
declararla: el VPS ejecutó el fichero **anterior al reencuadre del docstring**, y el sello incluye
el hash de la fuente. Números idénticos, procedencia distinta. El artefacto de registro es el
local, producido por la fuente commiteada; el del VPS se conserva en
`results/garrido_meta_learner_v2_vps_crosscheck/` como verificación independiente.
