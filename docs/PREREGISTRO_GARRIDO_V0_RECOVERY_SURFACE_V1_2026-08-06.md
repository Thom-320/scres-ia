# Prerregistro — superficie recovery-first para el bucle repetido de Garrido/v.0

**Estado:** escrito antes de construir la superficie de 216 posturas.  Desarrollo sobre
semillas ya consumidas; no abre ni reclama un bloque virgen.

## Pregunta que habilita

El gate v2 (`GO_BUILD_V0_RECOVERY_SURFACE`) demostró que seis riesgos aislados producen una
perturbación incremental de servicio y que la postura de buffers cambia su recuperación. Eso no
demuestra todavía que haya una tarea de aprendizaje. Antes de entrenar cualquier política hay que
probar dos condiciones:

1. la superficie de recuperación no debe reducirse a efectos aditivos por nodo; y
2. elegir una postura distinta según el riesgo debe ahorrar al menos un día de tiempo de
   recuperación restringido frente a la mejor postura común.

Si cualquiera falla, no se entrenará PPO, MLP ni KAN para esta vía. El resultado será que el DES
admite control, pero no aporta el headroom contextual necesario para el claim de aprendizaje.

## Física congelada

- DES: `MFSCSimulation`, `strict_exogenous_crn=True`.
- Demanda: natural; no se usa `excel_order_tape`, porque esa ruta anulaba R24.
- Horizonte: 36 semanas.
- Inicio del evento: 4.031 h, una hora antes del ciclo Op1 de 4.032 h y alineado con el ciclo Op2
  de 672 h.
- Ventana de recuperación: `tau = 1.344 h` (ocho semanas).
- Recuperación: siete días consecutivos con servicio y backlog dentro del criterio congelado en
  `supply_chain/garrido_v0_recovery.py`.
- Endpoint primario por celda: `restricted_ttr = min(TTR, tau)`. Si el shock es absorbido, vale
  cero; si impacta y no recupera, vale `tau`. La censura nunca puede parecer recuperación rápida.
- Escalar de búsqueda: `recovery_utility`, con TTR dominante y AUC/fill sólo como desempates. Una
  mejora de un día en TTR excede la contribución conjunta máxima de ambos desempates.

Los riesgos, duraciones y magnitudes son exactamente los del gate v2. No se ajustan después de ver
la superficie.

## Dominio de decisión

Tres buffers estratégicos: `op3_rm`, `op5_rm`, `op9_rations`. Cada uno toma la escalera de la tesis
`(0, 168, 336, 504, 672, 1344)` horas, convertida a las unidades físicas ya cableadas por
`posture_targets`. La rejilla completa tiene `6^3 = 216` posturas. Ningún otro parámetro cambia.

## Contextos

Orden contractual completo: `R11, R12, R13, R14, R21, R22, R23, R24`.

- Contextos físicamente vivos, fijados por el gate v2: `R11, R14, R21, R22, R23, R24`.
- Controles nulos: `R12, R13`. Se conservan en la caché y deben seguir sin impacto incremental; no
  contribuyen a los gates positivos ni a H1--H4.

## Semillas y separación

Bloque de replay declarado: `5.300.001--5.300.012`, ya consumido por `garrido_q2_des288`.

- Desarrollo y gates: `5.300.001--5.300.006`.
- Evaluación retenida: `5.300.007--5.300.012`.

Primero se construyen y analizan únicamente las seis semillas de desarrollo. No se leerá ni
construirá la mitad retenida hasta congelar por escrito el algoritmo, hiperparámetros, secuencia de
campañas, estimandos y placebos. Esta separación es un holdout de desarrollo, no virginidad.

## Caché

Un artefacto sellado por semilla contiene:

- el placebo completo para las 216 posturas;
- los ocho riesgos completos para las 216 posturas;
- panel agregado, panel temporal, hash de demanda realizada, eventos, endpoint derivado y utilidad;
- `module_manifest`, hash científico y sello del sobre.

La caché almacena paneles, nunca sólo el escalar optimizado. Un consumidor falla cerrado si falta
una celda, postura, contexto, sello o si cambia el hash científico.

## Gates de desarrollo

### G0 — integridad física y de caché

Todas las semillas deben contener 216 posturas únicas por contexto, un único cluster temporal por
celda, TTR restringido en `[0, tau]`, demanda CRN idéntica entre posturas dentro de cada
`(semilla, contexto)` y sellos válidos.

### G1 — controles nulos

R12 y R13 deben conservar `impact_fraction = 0` y rango de TTR restringido igual a cero. Si dejan
de ser nulos, se detiene: la física ya no es la que aprobó el gate v2.

### G2 — no separabilidad fuera de semilla

Para cada contexto vivo se comparan dos regresiones categóricas:

- aditiva: efectos principales de los tres factores;
- completa de orden dos: efectos principales más interacciones pareadas.

Se usa leave-one-seed-out: se ajusta en cinco semillas y se puntúa la sexta. El estimando es
`Delta_CV_R2 = R2_interacciones - R2_aditivo`, bloqueado por semilla. Deben existir al menos cuatro
de seis contextos con media `>= 0.02` y `LCB95 > 0`. El R2 en muestra no cuenta.

### G3 — valor del óptimo contextual

En cada fold se eligen, usando sólo las otras cinco semillas:

- una postura por contexto que maximiza la utilidad media; y
- una única postura común que maximiza la utilidad media sobre los seis contextos vivos.

Ambas se evalúan en la semilla dejada fuera. El estimando primario es

`H_regime_TTR = mean_context(TTR_common - TTR_contextual)`.

Debe cumplir `LCB95 >= 24 h`. Se reportan además utilidad, posturas elegidas y número de óptimos
distintos. Distintos argmax sin valor operativo no pasan el gate.

## Regla de salida

- `GO_FREEZE_REPEATED_CAMPAIGN`: G0--G3 pasan.
- `STOP_NO_RECOVERY_LEARNING_HEADROOM`: cualquier gate falla.

Sólo el primer veredicto autoriza desarrollar comparadores sobre las seis semillas de desarrollo.
No autoriza abrir semillas nuevas ni mirar el holdout.

