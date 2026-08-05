# Resultado E*-C v2 — replay de reparación instrumental

**Estado:** `HALTED_FALSIFIER_FAILED`

**Alcance:** replay declarado sobre el bloque ya consumido `5.200.001–5.200.016`.
No se abrieron semillas nuevas, no se cambió la física y no se autoriza `G1`, una
confirmación independiente ni entrenamiento neuronal.

**Contrato ejecutado:**
`docs/ENMIENDA_ESTAR_CAPACIDAD_BARRIDO_V2_REPLAY_2026-08-05.md`

**Contrato SHA-256:**
`05ce14200a0c786bab2321f5b205f4aba24adc4fee84a30927a58b19fc84961b`

**Custodia:** `run_role=REPLAY`, `replay_of=contention_headroom`, manifiesto de
módulos completo (`missing=[]`). El artefacto canónico es
`results/headroom/estar_capacity_sweep_v2_replay_20260805/result.json`.

## 1. Resultado de los falsadores

| falsador | resultado | lectura |
|---|---|---|
| `f1_capacity_actually_binds` | pasa | la capacidad sí ata en los presupuestos probados |
| `f2_mass_and_demand_are_untouched` | pasa | 576 comparaciones, 0 discrepancias; demanda total y A/B preservadas |
| `f3_the_endpoint_responds_to_the_lever` | pasa | `flow_fill_rate` responde a la palanca |
| `f4_argmax_moves_with_regime` | diagnóstico pasa | `0,6→0,5` sólo con presupuesto 600; no es un falsador de software |
| `f5_no_fresh_seeds` | no aplica | replay explícito de semillas quemadas |
| `f6_no_gain_by_abandonment` | **falla** | daño de servicio por encima de los márgenes firmados |

La salida de la ejecución es `claim_status=
HALTED_FALSIFIER_FAILED`. El `f4` descriptivo no puede rescatar un contrato cuyo
guardarraíl `f6` falla.

## 2. Estimando descriptivo

El cálculo usa el promedio sobre semillas antes del máximo por acción:

```text
H_regime = mean_r[ max_a( mean_s(Y[r,a,s]) ) ]
           - max_a( mean_r( mean_s(Y[r,a,s]) ) )
```

| presupuesto | `H_regime` | `argmax` base → `freq3_imp2` | dispersión | binding |
|---:|---:|---|---:|---:|
| 600 | `+0,000248` | `0,6 → 0,5` | `0,10374` | `0,766` |
| 1200 | `+0,000000` | `0,6 → 0,6` | `0,04306` | `0,584` |

Estos valores son descriptivos del screen. No constituyen headroom adjudicado:
`f6` impide promoverlos.

## 3. Por qué falla `f6`

La acción seleccionada en cada celda se compara contra el brazo sin capacidad en
la misma cinta. El bootstrap pareado usa 5.000 remuestras sobre las 16 semillas.

| celda | share | daño `lost_orders` | UCB95 | daño `worst_claimant_fill` | UCB95 |
|---|---:|---:|---:|---:|---:|
| 600 · `base` | 0,6 | 0,000 | 0,000 | 0,000566 | 0,002637 |
| 600 · `freq3_imp2` | 0,5 | **0,500** | **1,500** | 0,007678 | **0,030000** |
| 1200 · `base` | 0,6 | 0,000 | 0,000 | 0,000566 | 0,002579 |
| 1200 · `freq3_imp2` | 0,6 | **0,375** | **1,125** | 0,007678 | **0,030549** |

Márgenes preregistrados:

```text
UCB95(harm_worst_claimant_fill) <= 0,010
UCB95(harm_lost_orders)         <= 0,50
```

El régimen `freq3_imp2` no pasa ninguno de los dos guardarraíles. Por tanto, no
es defendible afirmar que la capacidad sólo retrasa o que el efecto es seguro.
La afirmación máxima permitida es más estrecha: en este contrato, la capacidad
ata y puede producir daño terminal de servicio bajo el régimen escalado.

## 4. Qué reparaciones quedan demostradas

* `f2` ya no prueba que un conjunto no esté vacío: compara cada brazo capado con
  su propio brazo sin capacidad, por semilla y régimen, en total y por reclamante.
* `f6` ya no está cableado: calcula daño pareado y UCB95 para `lost_orders` y
  `worst_claimant_fill`.
* El presupuesto se valida mediante `budgeted_ledger`; el error máximo de suma
  de capacidades CSSU es `0,0`.
* El payload incluye órdenes, eventos, resultados, manifiesto de módulos,
  `run_role`, `replay_of`, contrato y hashes.
* La suite focalizada de reparación pasa: **337 tests**.

## 5. Límites de interpretación

Este screen conecta capacidad únicamente en CSSU A/B. Aunque el helper admite
`wdc`, `al` y `sb`, esos nodos no están conectados al DES; por ello todavía no es
el E* ampliado por nodo que Garrido pidió.

`flow_fill_rate` es el endpoint primario de la intervención, pero es una razón
terminal de raciones servidas sobre demanda total. Es sensible a retrasos sólo
por lo que queda sin entregar al final del horizonte; no es una medida temporal
pura. AUC de pérdida de servicio, backorder temporal o panel semanal exigirían
una extensión contractual separada.

El resultado previo con falsadores inválidos permanece retractado. Este replay
repara el instrumento y conserva la evidencia, pero no convierte el bloque
quemado en confirmación nueva. No se abre G3c, no se abre E* confirmatorio y no
se entrena ninguna política neuronal.
