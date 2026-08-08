# Enmienda v2 — auditoría estructural del comparador reparado

**Estado:** protocolo correctivo emitido después del primer replay `v1`; no eleva el grado ni
reemplaza la confirmación original.

El artefacto `results/comparator_repair/result.json` queda preservado como `v1` histórico. Su
resultado científico se reproduce en la versión siguiente, pero su falsador de “deriva del prior”
usaba correlación del AUC agregado con el orden de semilla. Eso no prueba inmutabilidad: cada semilla
posee su propio entrenamiento base y, por tanto, su propio prior.

La corrida v2 corrige únicamente la adjudicación instrumental:

- `frozen_factor_prior` es una compresión independiente por nivel, con pseudocuenta Laplace
  `alpha=1` y factores nuevos uniformes; no se afirma que sea la misma extensión interna de cada
  carrier.
- La inmutabilidad se verifica mediante digest antes y después de cada contexto objetivo.
- `causal_prefix` se valida por masa exacta: antes del caso `q+1` contiene
  `4608 + 24q` cuentas y sólo se actualiza después del replay.
- La contaminación del caso actual se valida mediante distancia de variación total, no mediante una
  cota AUC arbitraria.
- El replay acumulativo recibe trayectoria descriptiva, no intervalo bootstrap i.i.d.
- Las listas `transfer`, `cold` y replay original se comparan con igualdad exacta de valores, no con
  redondeo a doce decimales.

La versión v2 usa exclusivamente el bloque quemado `8200001–8200060`, los seis contextos y los
cachés ya sellados. El resultado sólo puede ser `REPLAY/DEVELOPMENT`; cualquier falsador de
integridad produce `HALTED_FALSIFIER_FAILED` y no autoriza otra variante.

La etiqueta `state-blind marginal replay` queda retirada. El control histórico se denomina
`online_cumulative_frequency_replay`; el control sin contaminación actual se denomina
`causal_prefix_replay`.
