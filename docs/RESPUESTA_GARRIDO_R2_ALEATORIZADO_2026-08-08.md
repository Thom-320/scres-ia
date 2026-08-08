# Respuesta a la petición de R2 aleatorizado — ya está contestada, y con una corrección nuestra

**Fecha:** 2026-08-08 · **Petición:** Garrido, reunión 2026-08-07 — *«dejar los riesgos R1
(conocido-conocidos) fijos y aleatorizar los R2 (conocido-desconocidos)… combinaciones aleatorias de
frecuencia e impacto para que ninguna política estática sea óptima»*.
**Artefacto:** `results/garrido_risk_headroom_sensitivity_v1/result.json`

---

## La pregunta detrás de la petición

Garrido no pide aleatoriedad por sí misma. La justifica: *«un patrón determinístico lo aprende la red
rápidamente»*, y quiere **que ninguna política estática sea óptima**. Es decir, la pregunta real es:

> **¿La acción óptima varía con el régimen de riesgo?**

Esa pregunta ya se midió, y con un instrumento **más severo** que aleatorizar.

## Lo que se corrió

| | |
|---|---|
| perfiles de riesgo | **45**, escalando los riesgos recurrentes de la propia tesis |
| grupos | `R1_frequency` · `R1_one_at_a_time` · `R2_frequency` · `R2_one_at_a_time` · `impact_R11` · `impact_R21` · `impact_R22` · `impact_R23` · `impact_R24` |
| posturas constantes | 18 |
| semillas | 6 (`7450001–7450006`) |
| evaluaciones | **4.860** |
| filas de resumen | 63 (grupo × tope de presupuesto) |

**R2 está cubierto en las dos dimensiones que él nombra**: frecuencia (`R2_frequency`,
`R2_one_at_a_time`) e impacto (`impact_R21`…`impact_R24`).

## El resultado

```
status         DEVELOPMENT_NO_DOOR_UNDER_TESTED_FRONTIER
passing_doors  []            (0 de 63 filas con door_pass)
max H_profile_safe   6,931e−05     CI95 [0,0 , 2,079e−04]
barra preregistrada  0,01          →  144× por encima del máximo observado
```

Con `H_profile_safe` = el valor de **adaptar la postura al perfil de riesgo** en vez de usar la mejor
constante robusta. El intervalo del máximo **incluye cero**.

## Una corrección nuestra, antes de que circule más

Veníamos diciendo —yo el primero, y está en nuestras notas— que **«la postura óptima es invariante en
los 45 perfiles»**. **No es exacto.** El campo `unique_profile_optima` toma valores 1, 2 y 3 según la
fila: en la fila del máximo son **dos** (`f0.125_S1` y `f0.25_S1`).

La afirmación correcta, que además es más fuerte porque sobrevive a un revisor:

> **El óptimo varía entre a lo sumo tres posturas, y el valor medido de adaptarse al perfil es
> 6,93e−05 [0 , 2,08e−04] contra una barra de 0,01 — 144× por debajo, con el intervalo incluyendo
> cero.** No es que el óptimo no se mueva: es que moverse con él no compra nada.

## Por qué escalar es más severo que aleatorizar, para esta pregunta

Aleatorizar frecuencia e impacto **promedia** sobre perfiles: si el óptimo se moviera en algunos, el
sorteo diluiría el efecto. Escalar cada riesgo uno a uno **aísla** el perfil y pregunta directamente
si el óptimo se mueve con él. Si no se mueve —o si moverse no paga— bajo escalada sistemática de cada
riesgo de la familia, no lo hará bajo sorteos de esa misma familia.

## Lo que el screen NO cubre, y hay que decirlo

Varió **perfiles de riesgo**, no la **realización estocástica dentro del episodio**. Los perfiles son
intercambios de nivel (`overrides: {R21: "increased", R22: "current", …}`), fijos durante la corrida.
Si la petición es que la frecuencia y el impacto de R2 se **sorteen por episodio** —de modo que la
política no pueda anticipar el régimen ni siquiera al principio— **el screen no lo responde**.

Y hay una diferencia real entre las dos cosas: bajo perfil fijo, una constante puede ser óptima
porque el régimen es conocible; bajo sorteo por episodio, podría existir valor en **inferir** el
régimen sobre la marcha. Eso es un mecanismo distinto y no está medido.

## La pregunta para Garrido

> **¿Tu petición es variar el perfil de riesgo, o la realización dentro del episodio?**

* **Perfil** → está hecho, con 4.860 evaluaciones, y se cita.
* **Realización por episodio** → `S2` se preregistra y se corre, aislado del panel estacional.

## Estado

```
R2_PROFILE_VARIATION          ANSWERED_BY_EXISTING_SCREEN (development)
R2_WITHIN_EPISODE_REALISATION NOT_MEASURED — pending Garrido's disambiguation
```

Esto **no es aplazar por silencio**: es responder con evidencia y nombrar con precisión el hueco. Y
libera las 72 h de la Fase 2 para el panel estacional, que sí no tiene respuesta previa.

## Custodia

El screen es **desarrollo** (`claim_boundary`: `H_PI_established: false`,
`learner_authorized: false`), sobre semillas `7450001–7450006`. Se cita con ese grado, nunca como
confirmación. Y con cero bloques vírgenes disponibles (`ENMIENDA_4`), un eventual `S2` tampoco podría
serlo.
