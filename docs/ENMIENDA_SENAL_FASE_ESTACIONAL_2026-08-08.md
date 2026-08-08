# Enmienda — la fase estacional como señal, y por qué sola no puede convertir

**Escrita ANTES de correr.** Runner: `scripts/run_phase_signal_conversion_v1.py`. Custodia: réplica
declarada. Sucede a `results/observable_conversion/result.json` (sello `5e5d29c3`,
`BLOCKED_INSTRUMENT`), que se conserva.

## 1. Lo que medí antes de diseñar

La fase es `semana mod 12` y es **idéntica en todas las tapes**:

```
fase por semana : 0 1 2 3 4 5 6 7 8 9 10 11 0 1 ...
escala          : 1,059 ... 1,059  0,35  1,059 ...     (el valle está en la fase 11)
```

Verificado en las semillas 8600001, 8600007 y 8600012: las tres dan la misma secuencia.

## 2. La consecuencia, y es estructural

**Una política que lee sólo la fase es una función determinista del tiempo: es open-loop.** El
hueco clarividente es, por construcción, `mejor-por-tape − mejor-fijo`, es decir **la parte que
exige conocer la tape**. Una regla determinista **es** un calendario fijo, así que su aportación a
ese hueco es **cero por definición**.

De modo que una regla de fase no puede convertir el techo. Lo que sí puede hacer es ser **un
calendario fijo mejor** que los que enumeré — y eso responde a otra pregunta, legítima y distinta:
**¿era demasiado estrecha mi clase de bloques contiguos?**

Decirlo después de ver el número sería racionalizar. Va aquí.

## 3. Los dos brazos, y qué puede afirmar cada uno

| brazo | señal | naturaleza | qué puede afirmar |
|---|---|---|---|
| **A `phase_only`** | fase (determinista en `t`) | **open-loop** | si la clase de bloques contiguos era demasiado estrecha. **No puede convertir** |
| **B `phase_plus_state`** | fase **y** desviación de la demanda realizada respecto de su expectativa estacional | **lee estado** | si un control observable convierte el techo |

El brazo B es el único con la propiedad que hace falta: su señal **varía entre tapes al mismo `t`**,
que es exactamente lo que el backlog no consiguió aprovechar.

**Familia de políticas, declarada:** ventana de fases contigua de anchura `w ∈ {2,4,6,8}` y offset
`0..11`; en el brazo B, además, sostener sólo si la demanda realizada de la semana anterior superó
su expectativa estacional. Ventana y offset se seleccionan **sólo en tapes de entrenamiento**.

## 4. Falsadores nuevos

| falsador | por qué puede fallar |
|---|---|
| `f7_phase_is_deterministic_across_tapes` | si la fase variara entre tapes, el brazo A no sería open-loop y todo el encuadre de §2 caería. Se comprueba, no se supone |
| `f8_arm_A_cannot_convert_per_tape_headroom` | el calendario realizado del brazo A debe ser **idéntico en todas las tapes**. Si difiere, o la fase no es determinista o el runner filtró estado |
| `f9_arm_B_signal_varies_across_tapes` | si la desviación de demanda resultara constante entre tapes, el brazo B sería open-loop disfrazado y no podría convertir tampoco |

Siguen vigentes los del contrato anterior: la regla es causal, umbral y comparador se eligen en
entrenamiento, el placebo conserva la libertad y destruye la información, y la política **no puede
superar al techo**.

## 5. Lo que no cambia

`λ = 0,35` sigue siendo el titular y sigue siendo **un pico seleccionado sobre estas mismas tapes**;
la banda 0,275–0,500 va al lado. El techo de `d5e0b9bf` —0,045103 [LCB95 +0,028482]— no se toca.
Y `OBSERVABLE_POLICY_IS_WORSE_THAN_THE_FIXED_SCHEDULE` sigue siendo un veredicto admisible: la regla
de backlog ya perdió por −0,019549 con el intervalo entero bajo cero, y nada obliga a que ésta gane.
