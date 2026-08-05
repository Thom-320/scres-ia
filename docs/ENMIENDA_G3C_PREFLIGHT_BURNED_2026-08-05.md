# Enmienda G3c — celdas, potencia y autorización `BURNED_PREFLIGHT`

**Estado:** `BURNED_PREFLIGHT_AUTHORIZED_NO_FRESH_SEEDS`
**Contrato padre:** `contracts/g3c_temporal_coupling_v2.json` (sigue en
`DESIGN_ONLY_NOT_AUTHORIZED_UNTIL_SUBMISSION_A_RECEIPT` para semillas frescas)
**Supersesión científica del gate:** `docs/SUPERSESION_CIENTIFICA_G3C_2026-08-05.md`

Escrita **antes** del runner y **antes** de cualquier ejecución. Congela lo que el preregistro dejó
sin fijar: `{1, 3, 7}` son niveles de `min_dwell`, **no una definición de celdas**.

## 1. Autorización, y sus límites

El PI autorizó explícitamente en sesión de 2026-08-05 **sólo** `BURNED_PREFLIGHT`: ejecución sobre
el bloque quemado `5.200.001–5.200.016`, réplica declarada. **Ninguna semilla nueva, ningún
learner, ninguna adjudicación.** Toda salida lleva `claim_status` con prefijo `PREFLIGHT_`.

## 2. Celdas, congeladas

```text
celda = (min_dwell_days, regimen)
min_dwell_days ∈ {1, 3, 7}          # 1 = nulo de regresión legacy
regimen        ∈ {R1r+R2r|base, R1r+R2r|freq3_imp2}
=> 6 celdas
```

Horizonte 52 semanas, `step_hours = 24`, `split_v1`, `N = 2`, `FIFO_PARTIAL`,
`op9_dispatch_policy = fixed_clock_daily`, `strict_exogenous_crn = True`. Idénticos a G3-obs v2, a
propósito: **el preflight debe ser comparable con el carril que lo precede.**

## 3. Brazos

| brazo | papel |
|---|---|
| `constant_best` | contexto, y **cota inferior del incumbente** |
| `myopic_equivariant` | **el incumbente**: umbral de dos ramas, simétrico en A/B |
| `hysteresis` | el candidato: dos umbrales `τ_in > τ_out`, la política que **sí** puede honrar el compromiso |
| `uninformed_placebo` | debe perder |
| `wrong_claimant` | control de reclamante equivocado, debe perder |

**Contraste primario:** `hysteresis − myopic_equivariant`, **pareado por semilla**, en
`worst_claimant_fill`.

## 4. Potencia, fijada de antemano

* **Unidad de resampling: la semilla.** Los brazos comparten cinta, así que el contraste es
  pareado y la varianza relevante es la de la diferencia, no la de los niveles.
* **Corrección simultánea: Bonferroni sobre las 6 celdas**, test de una cola. `z = Φ⁻¹(1 − 0,05/6)`.
* **Potencia objetivo: 90 %**, una cola.
* **SESOI: `+0,010` absoluto** en `worst_claimant_fill`, heredado del contrato.
* **MDE** con `n = 16` tapes quemados: `MDE = (z_power + z_corr)·SD_pareada/√n`.
* **`n` requerida** para el SESOI: `n* = ⌈((z_power + z_corr)·SD_pareada / SESOI)²⌉`.
* **Presupuesto máximo: `n* ≤ 96` semillas frescas.** Es el mayor bloque virgen que este proyecto
  ha abierto de una vez, y por encima de eso el coste supera el valor esperado de un carril cuyo
  predecesor cerró en cero.

**Regla terminal:** `n* > 96` en cualquier celda ⇒ `STOP_G3C_UNDERPOWERED`, y G3c se cierra **sin
gastar una semilla**. `n* ≤ 96` ⇒ `PREFLIGHT_POWERED_PENDING_AUTHORITY`, que **no autoriza abrir
nada**.

## 5. Guardarraíles y sus denominadores

| guardarraíl | δ | denominador |
|---|---:|---|
| `flow_fill_rate` | 0,005 | raciones entregadas / demandadas, absoluto |
| `lost_orders` | 0,50 | pedidos **por episodio**, absoluto |
| `backorder_qty_final` | 1,0 % | **relativo al valor del incumbente** en la misma celda |
| masa, capacidad, recursos | 0,0 exacto | identidad algebraica |

Daño medido **contra el incumbente sobre la misma cinta**, con `UCB95(daño) ≤ δ`. Ésta es la
lección de E\*-C: un `f6` que no se calcula contra un comparador **no puede fallar**, y el que sí
se calculaba falló.

## 6. Nueve falsadores, y por qué cada uno puede fallar

| falsador | por qué puede fallar |
|---|---|
| `f1_null_arm_payload_identity` | `min_dwell=1` explícito debe reproducir el legacy por hash científico canónico; una física que se cuele lo rompe |
| `f2_min_dwell_actually_binds` | `cssu_blocked_by_dwell_count > 0` en 3 y 7 y **exactamente 0** en 1; un dwell que nunca ata hace vacuo todo |
| `f3_incumbent_is_myopic_equivariant` | el incumbente debe **batir a la mejor constante**; si no, es un hombre de paja y el residual no significa nada |
| `f4_uninformed_placebo_fails` | si el placebo iguala al candidato, el valor está en variar, no en el estado |
| `f5_wrong_claimant_fails` | si leer el reclamante equivocado empata, la señal no es la que se declara |
| `f6_guardrails_use_signed_margins` | δ estocásticos estrictamente positivos y δ algebraico exactamente 0, verificados contra el JSON del contrato |
| `f7_power_frozen_before_execution` | SESOI, margen y celdas del runner deben coincidir con el contrato **leído del disco**, no con constantes propias |
| `f8_no_fresh_seeds_before_authority` | custodia central; una semilla fuera del bloque quemado lo rompe |
| `f9_no_gain_by_abandonment` | daño pareado contra el incumbente con `UCB95 ≤ δ` |

**Mutantes obligatorios** (del contrato): ignorar `min_dwell` ⇒ `f2` falla; comparar contra la
constante en vez del miope ⇒ `f3` falla; poner a cero un δ estocástico ⇒ `f6` falla. Cada uno se
verifica en test, porque **un falsador que nunca se ha visto fallar no es evidencia**.
