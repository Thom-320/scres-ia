# Resultado — auditoría de fuente H3′: reproducción exacta, y aun así **no hay merge**

> **SUPERSEDIDO 2026-08-02** por `docs/RESULTADO_H3_AMBAS_REBANADAS_CONTRATADAS_2026-08-02.md`.
> Este documento cubre **un solo lado** del experimento y está atado al artefacto replay antiguo.
> Ambas rebanadas se re-sellaron después bajo el contrato H3′ con manifiesto de módulos, lo que
> hace **decidibles por primera vez** los tres falsadores de fusión. Su §5 pedía exactamente eso.
> Se conserva sin tocar el cuerpo; las conclusiones vigentes están en el sucesor.

**Artefacto:** `results/garrido_meta_learner_h3power_vps_local_replay/result.json`
(sello `dbab2f27d42f0638…`) · contrato **H3′** `docs/PREREGISTRO_H3_POTENCIA_2026-08-01.md`
(`576d02b5…`) · semillas `6.000.091–120`, **bloque quemado, ninguna nueva** ·
`f6_seed_custody = NO APLICA` (`DECLARED_REPLAY`, sellado **dentro** del artefacto) ·
**manifiesto de 7 módulos** incluido.

## 1. La reproducción es exacta

Re-ejecución local de las semillas del VPS con el checkout actual, contra el artefacto original:

| cantidad | VPS | réplica local |
|---|---:|---:|
| `runs_to_within_1pct.neuron_memory` | 7,694444444444445 | **idéntico** |
| `runs_to_within_1pct.neuron_reset` | 15,305555555555559 | **idéntico** |
| `runs_to_within_1pct.ofat` | 12,616666666666667 | **idéntico** |
| `runs_to_within_1pct.random` | 19,383333333333333 | **idéntico** |
| `memory_vs_ofat` (media, LCB95, UCB95) | 4,922… / 3,905… / 5,916… | **idénticos** |
| `memory_vs_random` (media, LCB95, UCB95) | 11,688… / 10,100… / 13,233… | **idénticos** |
| `final_regret` (4 estrategias) | — | **idénticos** |

> **14 de 14 cantidades idénticas al último decimal. Diferencia máxima `0,000e+00`.**

Y los cinco falsadores científicos pasan, incluido `f5`, el que **valida el arreglo de la fuga**
que obligó a retractar las cifras de julio.

## 2. Lo que esto NO establece, y es la mitad importante

**Veredicto: `BEHAVIORAL_REPRODUCIBILITY_FOR_H3_ESTIMAND`.** No `MERGE_VALID`.

El contrato H3′ exige `f_merge_source_is_identical`. **Una réplica conductual no puede
satisfacerlo por construcción**: demuestra que el estimando se reproduce **bajo el checkout
actual**, no que la fuente del VPS fuera la misma. La deriva conocida —`supply_chain.py` distinto,
`service_first_metric.py` ausente en el snapshot— pudo ser inmaterial **para este estimando** sin
serlo para otros.

**Y hay un impedimento más básico, que yo mismo había pasado por alto:**

| rebanada | contrato sellado |
|---|---|
| local, 90 réplicas | `PREREGISTRO_META_APRENDIZ_2026-07-31.md` · `a24b164d…` |
| VPS, 30 réplicas | `PREREGISTRO_META_APRENDIZ_2026-07-31.md` · `a24b164d…` |
| **contrato H3′** | `PREREGISTRO_H3_POTENCIA_2026-08-01.md` · **`576d02b5…`** |

**Ninguna de las dos rebanadas es un artefacto H3′ contratado.** Yo había supuesto que el problema
era sólo el VPS; es de las dos. Por tanto **ambas permanecen no promovibles**. Aquí las dos cifras
son estimandos distintos: `+7,61` es el efecto Alzheimer (`reset − memory`) y `+4,92` es la
comparación `memory − OFAT`; ninguno entra al manuscrito. El estado canónico sigue siendo
**`ARTIFACTS_PRESENT_MERGE_PENDING`**.

## 3. Un defecto que introduje y que obligó a regenerar

La primera corrida selló `claim_status = "NO APLICA"` **con los seis falsadores en orden**. Causa:
en el bucle de impresión usé `verdict` como variable local y **pisé el veredicto científico antes
de sellarlo**. Las cifras no se vieron afectadas —se calculan antes del bucle— pero un
`claim_status` sellado erróneo es un artefacto inválido, así que **se regeneró en vez de
editarse**.

Es el tercer defecto de instrumento del día en mi propio código —la latencia de activación que
mataba el actuador, el falso «label swap», y éste— y **los tres tienen la misma forma: el código
hacía algo distinto de lo que su nombre prometía.**

## 4. Lo que la auditoría sí deja reparado, y sirve para todo lo que venga

* **`f6` ya no miente.** Antes comprobaba una lista interna que nunca supo de los bloques
  `6.000.0xx`: para estas mismas semillas devolvía **`passed = True`**, llamándolas vírgenes
  mientras el registro central las marcaba usadas. Ahora lee el registro **y** escanea artefactos
  sellados, en un solo módulo en vez de seis copias.
* **La réplica se declara en la EJECUCIÓN**, no en un documento. `--replay-of garrido_h3_vps`
  produce `DECLARED_REPLAY` y `not_applicable=True`, **contado en ninguna de las dos columnas**.
* **`module_manifest()`** sella los hashes de los módulos declarados y del entry point, con rutas
  relativas al repositorio. Hace decidible la comparación de ese conjunto declarado para rebanadas
  **futuras**; no cubre el intérprete, paquetes de terceros ni todo el entorno, y no rescata
  retroactivamente las rebanadas selladas sin él.

## 5. Qué haría falta para un H3′ citable

Nada de esto es ciencia nueva ni abre semillas:

1. re-ejecutar **ambas** rebanadas bajo el contrato H3′ y con manifiesto de módulos — las 90
   locales son también un replay declarado, y el bloque está quemado;
2. entonces `f_merge_source_is_identical` es decidible y el merge de 120 se puede adjudicar;
3. hasta entonces, el precio del efecto Alzheimer permanece **medido en desarrollo y pendiente de
   custodia**.
