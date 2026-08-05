# Enmienda G3c burned-only v2 — grilla físicamente viva

**Estado:** `BURNED_PREFLIGHT_AUTHORIZED_NO_FRESH_SEEDS`

Esta enmienda supersede únicamente la grilla y el contrato del preflight v1. No reabre
G3-obs, no adjudica G3c y no autoriza semillas nuevas.

## Motivo de la supersesión

El preflight v1 mostró, con el simulador y bajo presión de conmutación máxima, que `dwell=3`
no retiene ninguna acción: la latencia de activación y la cadencia diaria dejan inertes los
niveles de hasta cuatro días. Por tanto, `{1,3,7}` no era una grilla de tres niveles efectivos.
El resultado v1 se conserva intacto como `PREFLIGHT_HALTED_FALSIFIER_FAILED`; no se reinterpreta
ni se sobreescribe.

La caracterización burned-only `results/headroom/g3c_dwell_inertia/result.json`, con manifiesto
de módulos y `replay_of=contention_headroom`, midió la supresión de conmutaciones día por día.
El primer nivel material bajo el criterio preregistrado fue `6`; la caracterización propone su
doble como segundo tratamiento. La nueva grilla se fija antes de esta ejecución:

```text
min_dwell_days ∈ {1, 6, 12}
1  = nulo legacy
6  = primer tratamiento material
12 = tratamiento fuerte separado por factor 2
```

La selección de 6 y 12 se basa exclusivamente en liveness y supresión de conmutaciones observadas,
no en retornos de una celda. La unidad de resampling sigue siendo la semilla y el contraste sigue
siendo pareado.

## Diseño congelado

Se conservan los dos regímenes ya usados:

```text
R1r+R2r|base
R1r+R2r|freq3_imp2
```

El incumbente es la regla miope equivariante:

```text
unmet_A > unmet_B → 0.9
unmet_B > unmet_A → 0.1
empate            → 0.5
```

El candidato es una política de dos estados con histéresis, con umbrales normalizados
`tau_in=0.10` y `tau_out=0.02`, fijados antes de correr. También se ejecutan placebo no
informado, reclamante equivocado y la rejilla de constantes `[0.1,...,0.9]` como controles.

El primario es:

```text
hysteresis − myopic_equivariant
```

en `worst_claimant_fill`, con SESOI absoluto `+0.010`. Los guardarraíles y sus denominadores
son los del contrato machine-readable v2: `flow_fill_rate`, `lost_orders`, backorder relativo
y las identidades algebraicas.

La corrección simultánea es Bonferroni sobre cuatro contrastes (2 tratamientos × 2 regímenes),
potencia objetivo 90 % y máximo prospectivo de 96 semillas por celda. Este preflight sólo estima
potencia sobre las 16 semillas quemadas; aunque salga potenciado, no abre raíces frescas.

## Custodia y procedencia

La salida se escribe en un directorio nuevo:

```text
results/headroom/g3c_preflight_burned_v2/result.json
```

Debe sellar `run_role=BURNED_PREFLIGHT`, `replay_of=contention_headroom`, contrato v2, comando
efectivo, módulo de entrada y manifiesto de dependencias. El artefacto v1 y su runner paralelo
quedan sólo como antecedente; no se mezclan sus cifras con v2.

No se autoriza learner, confirmación, Program Q/O ni Submission A adicional.
