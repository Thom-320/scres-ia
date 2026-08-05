# Supersesión científica — por qué G3c puede seguir viva después de que G3-obs se cerrara

**Estado que sustituye:** la regla terminal de
`docs/RESULTADO_G3_OBS_V2_2026-08-02.md` §5, que dice literalmente que
`STRUCTURED_CONTROL_SUFFICES_G3_OBS` **NO abre G3c**, porque sólo un residual material sobre el
umbral lo habría hecho.

**Estado nuevo:** `G3C_REOPENED_AS_ORTHOGONAL_EXTENSION_PREFLIGHT_ONLY`.

Esta supersesión **no reabre G3-obs, no toca su artefacto sellado y no revierte su veredicto**. El
residual observable sobre la regla estructurada sigue siendo cero-equivalente, y eso sigue siendo
el resultado.

## 1. Qué era la regla, y por qué era correcta entonces

La regla existía para impedir exactamente un movimiento: que un carril derrotado por un `if` de dos
ramas se rescatara añadiendo física hasta que la red volviera a tener trabajo. Ese movimiento es
p-hacking con pasos de simulador, y la regla lo bloquea bien.

**Sigue bloqueándolo.** Nada de lo que sigue autoriza entrenar.

## 2. Por qué G3c no es ese movimiento

G3-obs preguntaba: **¿hay valor de decisión dependiente del estado que sea observable?** Respuesta
medida: sí, y **una regla miope de dos ramas lo agota**.

G3c pregunta otra cosa: **¿qué pasa cuando la decisión no se puede revisar cada día?** El mecanismo
es `cssu_min_dwell_days`, permanencia mínima — **compromiso intertemporal**. Y la ortogonalidad es
estructural, no retórica:

> **Una regla miope es óptima sólo si puede reoptimizar en cada instante de decisión.** La
> permanencia mínima elimina precisamente esa condición. El incumbente de G3-obs no es que sea
> peor bajo dwell: es que **deja de ser la clase correcta de política**, porque su argumento de
> optimalidad no sobrevive a la restricción.

Por eso G3c **no es un rescate**: no añade holgura para que gane un aprendiz, **añade una
restricción que el ganador anterior no puede honrar**. Si bajo dwell la regla miope sigue ganando,
G3c muere, y muere más fuerte que si nunca se hubiera corrido.

## 3. Lo que esta supersesión autoriza, y lo que no

**Autoriza:** escribir el runner, sus tests, y ejecutar un **preflight de potencia exclusivamente
sobre tapes quemados** (`5.200.001–16`, réplica declarada). Autorización firmada por el PI en
sesión de 2026-08-05, registrada aquí porque el texto del contrato no se autoriza a sí mismo.

**NO autoriza:**

* abrir un solo bloque virgen — el contrato sigue en
  `DESIGN_ONLY_NOT_AUTHORIZED_UNTIL_SUBMISSION_A_RECEIPT` para semillas frescas;
* adjudicar G3c: un preflight sobre tapes quemados **no es evidencia confirmatoria**, y su salida
  lleva el prefijo `PREFLIGHT_`;
* entrenar ninguna red, bajo ningún resultado;
* tocar Program Q ni Program O, que siguen inmutables.

**La semilla `5.200.001` es fixture de test y potencia de desarrollo, nunca potencia científica.**

## 4. Y si el preflight sale bien, tampoco basta

Si la potencia alcanza, el estado es `PREFLIGHT_POWERED_PENDING_AUTHORITY`: se congela el runner y
se espera **el recibo de Submission A o una supersesión de autoridad explícita** antes de reservar
un bloque virgen. Si no alcanza, el estado terminal es `STOP_G3C_UNDERPOWERED` y G3c se cierra sin
gastar una semilla — que es el ahorro por el que este preflight existe.
