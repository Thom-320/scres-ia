# Enmienda — dos artefactos sellados discrepan en el signo de una cota, y la corrección la debo yo

## 1. La discrepancia

`neuron_memory` menos `ofat_transfer`, AUC de arrepentimiento normalizado:

| artefacto sellado | media | `lcb95` |
|---|---:|---:|
| `results/search_ladder_v2_ordered/result.json` | 0,010710319493365220 | **−2,761e−05** |
| `results/search_ladder_v5/result.json` | 0,010710319493365220 | **+3,565e−05** |

**La media coincide hasta el último dígito.** No es el dato: es el sorteo del bootstrap. Un
documento dice que el intervalo excluye el cero y el otro que lo incluye, en el quinto decimal.

## 2. La corrección que debo

Una auditoría externa citó **−0,0000276** y yo le dije al PI que esa auditoría **se equivocaba**,
apoyándome en `search_ladder_v5`. No se equivocaba: **estaba citando el otro artefacto sellado**,
que es real aunque esté superado. El equivocado fui yo, y queda escrito aquí y en la tabla canónica.

Lo que sí sobrevive de mi lectura es la conclusión práctica —una diferencia demasiado pequeña para
importar— pero eso hay que decirlo por el camino correcto, no reclamando una cota que el propio
repositorio contradice.

## 3. Qué mide el runner, antes de correrlo

`scripts/run_ofat_lcb_reconciliation_v1.py`:

1. recalcula el contraste pareado desde los arreglos `per_arm` sellados con **B = 50.000** y
   semilla de RNG declarada;
2. repite el bootstrap entero bajo **40 semillas de remuestreo independientes** y cuenta en qué
   fracción la cota inferior cae por encima de cero.

**Una cota cuyo signo es cara o cruz no es una cota**, y el reporte honesto es la cara o cruz.

## 4. Reglas de lectura, fijadas ahora

| fracción de semillas con `LCB95 > 0` | veredicto |
|---|---|
| entre 0,05 y 0,95 | **`OFAT_LCB_SIGN_IS_RESAMPLING_UNSTABLE`** — se reporta como indistinguible de cero, citando **las dos** cotas selladas |
| ≥ 0,95 | `OFAT_LCB_IS_STABLY_POSITIVE` — y entonces se nombra el signo sin hedging |
| ≤ 0,05 | `OFAT_LCB_IS_STABLY_NON_POSITIVE` |

## 5. Falsadores

| falsador | por qué puede fallar |
|---|---|
| `f1_the_two_ladders_score_the_same_replicates` | compara los arreglos `per_arm` de las dos escaleras. **Si difieren, no es un artefacto del bootstrap: son dos experimentos**, y toda la comparación cambia de naturaleza |
| `f2_the_instability_is_measured_not_asserted` | declarar inestable una cota desde un solo remuestreo es una opinión; esto repite el bootstrap bajo semillas independientes y cuenta |
| `f3_a_stable_bound_would_be_reported_as_stable` | si la fracción fuese 0 o 1, el veredicto **debe** nombrar el signo y no refugiarse en «indistinguible» |

## 6. Cómo se cita a partir de ahora

> La neurona con memoria y OFAT con transferencia son **indistinguibles** en AUC de arrepentimiento
> (media +0,01071; la cota inferior del bootstrap cae a ambos lados del cero según el remuestreo:
> −2,76e−05 en `search_ladder_v2_ordered`, +3,56e−05 en `search_ladder_v5`).

**Prohibido escribir «excluye el cero»** para este contraste.

**Alcance:** re-análisis de artefactos sellados, sin simulación ni semillas. No cambia el orden de
la escalera ni ningún otro contraste.
