# Resultado — auditoría sellada de `f2` de G3-obs

**Tipo:** auditoría determinista sobre un artefacto existente; no se reejecutó el DES y no se
abrieron semillas.

**Auditor:** `scripts/audit_g3_obs_f2.py`

**Fuente:**
`results/headroom/g3_obs_conversion_v2/result.json`

**Recibo:**
`results/headroom/g3_obs_conversion_v2/f2_audit_result.json`
· sello `f6ad2119e21510c0…`

## 1. Resultado de `f2`

El preregistro exige el orden de medias:

```text
threshold_windowed > threshold_delayed > uninformed_placebo > wrong_claimant
```

| celda | ventana | retardo | placebo | equivocado | resultado |
|---|---:|---:|---:|---:|---|
| `base` | +0,02075 | +0,01826 | −0,00487 | −0,24490 | **PASA** |
| `freq3_imp2` | +0,01288 | +0,01124 | −0,00853 | −0,19779 | **PASA** |

El orden completo se cumple en ambas celdas. La auditoría conserva también los IC95 almacenados
por brazo; no vuelve a calcular ningún resultado del DES.

## 2. Límite de contrato

El artefacto fuente declara como contrato el preregistro anterior:

```text
fuente:    70f2e8ad…  PREREGISTRO_G3_OBS_CONVERSION_OBSERVABLE_2026-08-01.md
intendido: ad0395b5…  PREREGISTRO_G3_OBS_V2_POTENCIA_2026-08-02.md
```

Por ello el recibo queda como:

```text
claim_status:     F2_ORDER_HOLDS_SOURCE_CONTRACT_MISMATCH
promotion_status: BLOCKED_SOURCE_ARTIFACT_NOT_SEALED_UNDER_INTENDED_CONTRACT
```

La auditoría confirma que los datos guardados satisfacen el orden de `f2`; **no convierte
retroactivamente la corrida fuente en una ejecución bajo el contrato v2**. El resultado
`STRUCTURED_CONTROL_SUFFICES_G3_OBS` queda, por tanto, respaldado numéricamente pero pendiente de
resolver ese alcance de contrato.

## 3. Custodia y alcance

* `source_artifact_self_sha256` es válido.
* La auditoría usa las 140 semillas ya contenidas en el artefacto, sin semillas nuevas.
* `des_rerun = false`.
* `new_seeds_opened = false`.
* La auditoría no adjudica equivalencia específica del residual: el MDE publicado en la corrida
  fuente corresponde a `H_obs` frente a la constante, no a `residual_over_simple`.

El siguiente paso correcto es corregir el alcance de contrato mediante un recibo/adjudicación
explícito, no modificar el resultado fuente ni relanzar semillas por este defecto de auditoría.
