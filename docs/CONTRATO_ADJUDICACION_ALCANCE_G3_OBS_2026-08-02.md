# Contrato de adjudicación de alcance — G3-obs

**Estado:** `SCOPE_ADJUDICATION_DESIGN_ONLY`

**Fecha:** 2026-08-02

**Tipo:** recibo de custodia y alcance contractual; no es una ejecución científica,
una confirmación ni una re-selladura retroactiva.

## Propósito

Resolver formalmente el hecho de que
`results/headroom/g3_obs_conversion_v2/result.json` fue sellado con el contrato fuente
de 2026-08-01, mientras que sus parámetros de ejecución corresponden al bloque y a los
márgenes definidos prospectivamente en la v2 de 2026-08-02.

## Regla de no reescritura

El artefacto fuente, su JSON, sus hashes, su sello, sus semillas y su `claim_status` no
se editan. Este recibo puede clasificar su alcance, pero no puede convertirlo en una
ejecución sellada bajo otro contrato.

## Adjudicación que debe aplicar el runner

1. Comprobar la integridad del artefacto fuente y de la auditoría suplementaria de `f2`.
2. Comparar por separado el sello contractual, el bloque de semillas, los márgenes y el
   esquema del artefacto.
3. Conservar el artefacto fuente como evidencia histórica/de desarrollo con el alcance
   que efectivamente puede demostrar.
4. Registrar que no es una ejecución plenamente conforme al contrato fuente anterior si
   sus campos de ejecución contradicen el bloque o los márgenes declarados allí.
5. Registrar que tampoco es una ejecución v2: su sello y su runner pertenecen al contrato
   fuente, y la auditoría de `f2` no puede re-sellar la corrida.
6. Permitir que la auditoría suplementaria informe que el orden completo de `f2` aparece
   en los resúmenes almacenados, sin promover por ello la corrida a confirmación v2.

## Claims prohibidos por este recibo

- “La corrida fue ejecutada bajo el contrato v2”.
- “La corrida es una confirmación virgen o confirmación de potencia v2”.
- “La corrida es plenamente conforme al contrato fuente anterior” cuando los campos de
  ejecución no coinciden con ese contrato.
- “El runner original ejecutó el `f2` completo de la v2”.
- “Existe una prima neural confirmada”.

Una futura ejecución v2, si se autoriza, necesitará su propio artefacto sellado en el
momento de ejecución. Este recibo no abre semillas ni solicita un nuevo DES.
