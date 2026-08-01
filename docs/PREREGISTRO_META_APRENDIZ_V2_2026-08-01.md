# Preregistro v2 — Fase 4 corregida: aprendizaje entre configuraciones

**Estado:** escrito antes de los reruns v2. Este documento sustituye la lectura operativa del
runner v1; no reabre ni rescata los contrastes retirados del runner contaminado.

## Propósito

Responder las dos preguntas operativas de Garrido et al. (2024):

1. qué familia de aprendizaje puede cerrar el enlace entre variables de decisión y la métrica
   SCRES; y
2. cómo cambia el resultado cuando el estado del aprendiz (`rho`) se conserva entre corridas.

La unidad de aprendizaje es una **configuración completa**, no una decisión intraepisodio. Los
drivers son observaciones post-episodio: pueden actualizar el aprendiz después de una corrida,
pero no pueden ser entradas para rankear una configuración que aún no se ha ejecutado.

## Dos superficies, dos estatus

### A. Thesis-native: Cf1–Cf90

`scripts/run_meta_learner_thesis90_v1.py` usa la tabla 90-celda ya generada en
`results/garrido_drivers_per_configuration/result.json`, valida sus coordenadas contra
`supply_chain/garrido_thesis_design.py` y la recorre como nueve bloques publicados:

`H1a, H1b, H1c, H2a, H2b, H2c, H3a, H3b, H3c`.

Este brazo es un **replay de superficie**: no vuelve a ejecutar el DES. Por ello sus
repeticiones son replays algorítmicos sobre una misma tabla, no réplicas físicas independientes.
El baseline se llama `thesis_order`, porque el orden de las tablas de riesgo no es literalmente
OFAT en todos los bloques; no se lo rebautiza como OFAT para ganar una comparación.

El endpoint que se rerunea es `ret_excel` por continuidad con Fig. 5, pero su estado de métrica es
`HOLD_METRIC_PROVISIONAL`. El resultado no puede promover una política ni una prima neural: el
endpoint service-first requiere también la cola final, que no forma parte de esta tabla histórica.

### B. Extensión: 288 configuraciones

`scripts/run_meta_learner_over_configs_v1.py` mantiene la superficie DES extendida de
`buffer_hours × shifts × op9_rop × op12_rop`, los seis contextos de riesgo, CRN por
contexto-réplica y el presupuesto común. El rerun confirmado usa el bloque de semillas
`5_910_001…5_910_012`, separado de los bloques históricos y del proceso VPS v1 que se abrió con
`5_300_001…`.

La extensión sigue reportando ReT como diagnóstico de continuidad de Garrido y no autoriza por sí
sola una decisión. Cualquier lectura de servicio debe conservar por separado los componentes del
endpoint `service_first_resilience_v1`; no se permite sustituirlos por una suma ponderada.

## Brazos

En ambas superficies:

* `thesis_order`/`ofat` es el comparador abierto declarado por la superficie;
* `random` es el nulo no informado;
* `neuron_memory` actualiza el modelo con la configuración que acaba de correr y conserva `rho`
  al cambiar de contexto;
* `neuron_reset` usa el mismo código, superficie, orden, presupuesto y semillas/replays, pero
  reinicia `rho` en cada contexto.

El contraste memory–reset es el estimando del efecto Alzheimer. No es un resultado de control
adaptativo dentro del episodio y no es RL.

## Falsadores que deben poder fallar

1. La superficie tiene variación real entre configuraciones.
2. El comparador conserva el orden declarado y no se presenta como OFAT cuando no lo es.
3. Memory y reset comparten superficie, orden, presupuesto y streams; una diferencia deliberada
   de presupuesto debe ser detectada por el checker.
4. Random produce la misma secuencia cuando se sustituyen todos los valores de la superficie;
   consulta el resultado sólo después de sortear el índice.
5. Permutar los drivers sin tocar los valores no cambia ninguna secuencia; si cambia una sola,
   el buscador leyó una respuesta post-hoc al rankear una configuración no corrida.
6. La superficie y el bloque de replays son los declarados y no se reutilizan como semillas DES
   de confirmación.

Un fallo detiene la lectura del resultado. El número de corridas ahorradas sólo se describe si
todos los falsadores pasan.

## Regla de promoción

La comparación puede llenar el panel exploratorio de §4.2/§4.3. No autoriza MLP, PPO ni PPO
recurrente. Para una prima neural se exigiría además un endpoint de servicio cerrado, una
comparación contra el lineal/nulo y el gate WRAP de fidelidad conductual; ese gate sigue en HOLD.

El resultado thesis-native se rotula `SURFACE_REPLAY_ONLY`; el resultado 288 se rotula
`DES_SURFACE_CONFIRMATION` sólo si termina con todos los falsadores y semillas vírgenes.
