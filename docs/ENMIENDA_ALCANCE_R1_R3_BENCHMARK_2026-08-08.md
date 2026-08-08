# Enmienda — qué significa «R1 quieto», y qué no ha hecho nunca el análisis de sensibilidad

**Enmienda a** `docs/PREREGISTRO_GARRIDO_R2_RANDOMIZED_BENCHMARK_V1.md`, escrita antes de que
exista su runner. Corrige una restricción que puse **más estrecha de lo que el PI autorizó**, y
revisa el análisis de sensibilidad de riesgos a la luz de la aclaración.

## 1. La aclaración del PI

Para **R1 y R3**, «quieto» significa exactamente tres cosas:

* **se pueden encender y apagar**;
* **se pueden editar sus parámetros** (impacto y frecuencia);
* **no se les cambia la distribución de probabilidad**.

**La licencia que distingue a R2 es la familia de distribución en sí.** Ésa es la diferencia
operativa, y es lo que hace de R2 «conocido-desconocido» frente a R1/R3.

## 2. Lo que decía mal el preregistro

Escribí *«R1r fijo en `current`»*, *«R1r idéntico al baseline»* y *«R3 apagado»*. Eso es **más
restrictivo** que lo autorizado: confundí *«no lo movemos en este contraste»* con *«no se puede
mover»*. Sustituido por:

> **R1 y R3:** encendido/apagado y escalado de frecuencia e impacto son **admisibles** como ejes de
> estrés declarados. **Su familia de distribución queda congelada.** En el contraste primario
> `baseline` ↔ `R2 modificado` se mantienen **idénticos entre brazos**, porque de lo contrario la
> interacción no aislaría a R2 — pero eso es una restricción **del contraste**, no del espacio de
> diseño.
>
> **R2:** además de encendido/apagado y parámetros, **puede cambiar la familia de distribución**, y
> ése es el eje que Garrido pidió y que ningún artefacto nuestro ha tocado.

## 3. Revisión del análisis de sensibilidad de riesgos

Verificado en `supply_chain/config.py:511-537` y en
`scripts/run_garrido_risk_headroom_sensitivity.py:107-115`.

**Las escaleras `current → increased → severe` conservan la familia de distribución de todos los
riesgos.** R11 sigue uniforme (`b` 168 → 42 → 21); R12, R13 y R14 siguen binomiales (cambia `p`);
R21–R24 siguen uniformes (cambia `b`). Sólo se mueven los **parámetros**.

Y el screen de 4.860 evaluaciones usó **exclusivamente** `risk_overrides` (intercambios de nivel) y
`risk_impact_multipliers_by_id`, con `risk_frequency_multiplier = 1.0`.

De ahí salen dos conclusiones, y la segunda es la que importa:

**(a) El screen es conforme con la restricción de R1/R3.** Nunca cambió una distribución, así que
no violó nada. Bien.

**(b) Y por la misma razón, nunca hizo con R2 lo que Garrido pidió.** Varió niveles **dentro de la
misma familia uniforme**; jamás la familia. La petición era *«hacerlos más aleatorios y complejos
modificando distribuciones»*, y eso **no está hecho en ningún artefacto de este repositorio**.

Esto refuerza —por una vía distinta y con evidencia de código— la enmienda
`docs/ENMIENDA_RESPUESTA_R2_ALEATORIZADO_2026-08-08.md`, que ya había retirado la inferencia de que
escalar perfiles predecía el resultado de aleatorizarlos. No es sólo que la inferencia no se
siguiera: es que **el eje pedido nunca se movió**.

```text
R2_LEVEL_LADDER_WITHIN_UNIFORM_FAMILY      SCREENED_DEVELOPMENT
R2_DISTRIBUTION_FAMILY_CHANGE              NEVER_RUN
```

## 4. Lo que cambia en el diseño del entorno modificado

El brazo `R2 modificado` deja de ser «niveles sorteados» y pasa a ser lo que se pidió: **la familia
de distribución de R21–R24 es un eje declarado**, anclado en la fuente y congelado antes de correr.
Los niveles `off / current / increased / severe` siguen disponibles como **parámetros dentro** de
cada familia.

Sigue vigente todo lo demás del preregistro: la interacción `(KAN−MLP)_R2mod − (KAN−MLP)_baseline`
como contraste primario, SESOI 5 %, los cuatro comparadores, los tres presupuestos paramétricos
emparejados, el endpoint adimensional, la ausencia de rama `STOP`, y **la divulgación de §1 —que el
entorno no contiene un problema de asignación entre sus dos palancas— colocada antes del resultado
y no después**.
