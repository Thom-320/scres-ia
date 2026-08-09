# Incidente de reproducibilidad G3a y plan de reconstrucción

**Fecha:** 2026-08-08
**Estado:** `ORIGINAL_G3A_RUNNER_AND_RAW_ROWS_NOT_RECOVERABLE`
**Alcance:** evidencia de desarrollo citada en el manuscrito v1.1; no afecta artefactos sellados
de otras campañas.

## Hallazgo forense

El manuscrito y su tabla agregada sobrevivieron, pero el productor, el contrato efectivo, las
pruebas y las 18.360 filas crudas del G3a ejecutado no entraron en Git. Se comprobaron:

1. el árbol de trabajo, todas las ramas y referencias remotas disponibles;
2. objetos inalcanzables mediante `git fsck --full --no-reflogs --unreachable`;
3. blobs huérfanos buscando `G3a`, los nombres de los contratos y las cifras publicadas;
4. los registros de agente conservados en `docs/agent_runs/`.

Los objetos huérfanos no contienen el productor ni las matrices. Un registro contemporáneo
(`docs/agent_runs/2026-08-09T013120Z__subagentstop__run.md`) confirma que el mapa, las 18.360
ejecuciones y el manuscrito estaban fuera del repositorio en ese momento. El manuscrito fue
recuperado después; el runner y los resultados crudos no.

## Inconsistencia que impide llamar «exacta» a una recreación

El contrato versionado `contracts/g3a_asymmetric_claimants_v2.json` seguía en estado
`DESIGN_ONLY_NOT_AUTHORIZED_UNTIL_SUBMISSION_A_RECEIPT`, reservaba `7700001–7700120` y definía un
factorial distinto. El manuscrito v1.1 describe en cambio semillas `8701001–8701060`, 16 semanas,
tres procesos de riesgo-demanda, tres contratos de capacidad y 34 controladores. Por tanto, el
contrato que gobernó la ejecución tampoco sobrevivió.

## Regla de evidencia

Las cifras G3a preservadas se mantienen como **agregados de desarrollo no reproducibles**. No se
borran ni se renombran confirmatorias. Tampoco se aceptará una reproducción posterior porque
redondee a los mismos decimales: calibrar hasta concordar sería circular.

## Sucesor permitido

`g3a_forensic_reconstruction_v1` reconstruye únicamente lo que el manuscrito especifica de forma
trazable:

- 16 semanas y estados latentes `B/N/A`;
- persistencia 0,78 e iid como control;
- warning con exactitud 0,72;
- R24 localizado con probabilidad 0,55 y magnitud 2.500;
- demanda uniforme o extensión estacional declarada;
- cuotas rígidas, reasignación de sobrantes y FIFO global;
- nueve constantes y 25 comparadores declarados, 34 en total;
- semillas quemadas `8701001–8701060`, 30 selección / 30 evaluación;
- endpoint de exposición tardía descrito en el manuscrito.

La reconstrucción debe producir y commitear:

1. un runner autocontenido;
2. pruebas del conteo, conservación e invariancia FIFO;
3. las 18.360 filas crudas comprimidas;
4. un resultado sellado con hashes de código, contrato y filas;
5. una comparación explícita contra los agregados supervivientes, sin criterio de «éxito por
   parecido».

Su rol es `FORENSIC_DEVELOPMENT_REPLAY`. No abre semillas, no restaura virginidad, no autoriza
aprendiz, no reemplaza el experimento perdido y no puede citarse como confirmación.

## Resultado posterior más informativo

Mientras se completaba esta custodia se publicó en la misma rama una reconstrucción independiente
sobre el DES del repositorio, preregistrada antes de escribir su runner y ejecutada con el bloque de
desarrollo `8800001–8800060`. El artefacto final restaura los 34 controladores descritos por el
manuscrito: `results/g3a_boundary_v2/result.full34.json`.

Su veredicto es `G3A_DID_NOT_REPRODUCE`. En la celda persistente/uniforme con cuota rígida,
`H_obs=+0,002789` con IC95 `[−0,007635,+0,012373]`; además el mejor placebo no pierde contra la
señal real. La invariancia FIFO, acción viva, CRN y medición de capacidad desperdiciada sí pasan.

Ese resultado tampoco recupera el productor original: declara extensiones propias, un endpoint
post-hoc y semillas nuevas de desarrollo. Sin embargo, es evidencia operativa más fuerte que esta
reproducción forense sobre si el titular G3a vuelve en el DES actual. Regla conjunta:

- la tabla G3a del manuscrito continúa como agregado histórico no reproducible;
- el replay forense documenta exactamente qué parte de la descripción pudo reconstruirse;
- `g3a_boundary_v2/full34` es la prueba de desarrollo vigente y negativa;
- ninguna de las tres piezas autoriza un learner sobre asignación A/B entre CSSU.
