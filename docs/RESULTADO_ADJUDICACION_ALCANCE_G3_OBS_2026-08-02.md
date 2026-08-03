# Resultado — adjudicación formal del alcance contractual de G3-obs

**Tipo:** adjudicación de custodia y alcance; no es una nueva ejecución.

**Contrato de adjudicación:**
`docs/CONTRATO_ADJUDICACION_ALCANCE_G3_OBS_2026-08-02.md`

**Recibo sellado:**
`results/headroom/g3_obs_conversion_v2/contract_scope_adjudication.json`
· sello `eee480febd401873082deca61042c3636d1ffc05a5947e56335c506461ec1e64`

## Decisión

El artefacto fuente se conserva exactamente como fue sellado. **No se re-sella como v2 y
tampoco se presenta como plenamente confirmatorio bajo el contrato anterior**, porque sus
campos de ejecución no coinciden con el bloque ni con el margen declarados en ese contrato.

La clasificación final es:

```text
claim_status:     SOURCE_ARTIFACT_PRESERVED_SCOPE_MISMATCH_NOT_PROMOTABLE
promotion_status: BLOCKED_NO_RETROACTIVE_RESEAL_AND_NO_CONTRACT_CONFORMITY
audit_status:     CONTRACT_SCOPE_ADJUDICATION_NO_NEW_SEEDS_NO_DES_RERUN
```

## Qué se verificó

| comprobación | resultado |
|---|---|
| sello propio del artefacto fuente | válido · `317daf920579ec6…` |
| sello del contrato fuente declarado | válido · `70f2e8adffa8f…` |
| sello del contrato v2 pretendido | no coincide con la fuente · `ad0395b5a0f5…` |
| bloque de semillas frente al contrato anterior | no coincide: el viejo declara `5.200.001–5.200.016` |
| bloque de semillas frente a los campos v2 | coincide: `7.800.001–7.800.140` |
| margen `lost_orders` frente al contrato anterior | no coincide: `0,25` frente a `0,50` |
| orden completo de `f2` en resúmenes almacenados | pasa en ambas celdas |
| auditoría `f2` vinculada al `self_sha256` fuente | sí |
| DES re-ejecutado por esta adjudicación | no |
| semillas nuevas abiertas por esta adjudicación | no |
| artefacto fuente modificado | no |

La fuente está, por tanto, **sellada históricamente con el contrato anterior**, pero la
ejecución que describe usa el bloque y el margen de la v2. Esa combinación no debe
convertirse en una ficción de conformidad contractual.

## Qué sí puede afirmarse

- El artefacto fuente y su sello permanecen íntegros.
- Los resúmenes almacenados satisfacen el orden completo de `f2`, según la auditoría
  suplementaria `f2_audit_result.json` (sello
  `f6ad2119e21510c0bb15ae351cc67c11981df14ab0abbad824b9a82833cc79e6`).
- El resultado puede conservarse como evidencia de desarrollo con esta limitación de
  alcance.
- Una futura ejecución v2 necesitaría su propio sello v2 en el momento de ejecución.

## Qué queda prohibido

- Llamar a la corrida una ejecución o confirmación v2.
- Llamarla plenamente confirmatoria bajo el contrato anterior.
- Llamarla confirmación virgen o independiente.
- Afirmar que el runner original ejecutó el `f2` completo de la v2.
- Presentar este recibo como prueba de una prima neural.

La adjudicación resuelve el bloqueo formal **sin alterar los datos**. El coste de la decisión
es explícito: el resultado no obtiene promoción confirmatoria bajo ninguno de los dos
contratos. Para lograrla habría que ejecutar prospectivamente el contrato correcto, si la
autoridad del proyecto lo permite.
