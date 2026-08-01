# Preregistro — H1 (recuperación) y H3 (volatilidad) del borrador

**Escrito y commiteado ANTES de correr.** Runner: `scripts/run_h1_h3_recovery_volatility_v1.py`.
Cierra las dos hipótesis del borrador `v.0_neuralNet-scres` que nadie ha medido nunca.

## Qué exigen, literalmente

> **H1 – Learning Effect.** *«Hybrid simulation–neural models achieve significantly shorter
> recovery times compared to static simulation models.»*
>
> **H3 – Volatility Reduction.** *«Learning-enabled models reduce performance variance across
> heterogeneous disruption intensities.»*

`H2` y `H4` ya están medidas (`docs/RESULTADO_META_APRENDIZ_2026-07-31.md`: +6,31 corridas
[+5,18, +7,49], y la curva +0,00 → +10,00). Faltan éstas dos.

## Qué es «el híbrido» y qué es «el estático» aquí

No se inventan brazos nuevos. Son **las configuraciones que cada estrategia efectivamente
desplegaría**, ya elegidas en la Fase 4 y ahora registradas en el artefacto:

* **híbrido** = la configuración elegida por `neuron_memory` (la neurona que conserva `ρ`);
* **estático** = la elegida por `ofat`, que es **el diseño de la propia tesis** de Garrido.

Se añade `neuron_reset` como tercer brazo, porque el contraste memoria-vs-reinicio es el que
aísla el aprendizaje y ya sostiene `H4`.

## Diseño

* **Escalera de intensidad (4)**: multiplicador de frecuencia de riesgo `×1, ×2, ×3, ×4` sobre
  `R1r+R2r`. «Heterogeneous disruption intensities» del borrador, operacionalizado con el permiso
  explícito de Garrido sobre frecuencia.
* **Emparejamiento**: cada brazo se evalúa en **las mismas semillas** y los mismos escalones.
* **`H1`** — `system_ttr_mean` y `system_ttr_p95` del panel temporal
  (`include_temporal_panel=True`), **más bajo es mejor**.
* **`H3`** — **varianza entre escalones de intensidad** de `flow_fill_rate` (primaria) y de
  `R_cobb_douglas`, **más baja es mejor**.
* **Semillas**: `5 700 001…` vírgenes, CRN entre brazos.

**Por qué el servicio es la primaria de `H3` y no ReT.** Hoy quedó medido que `ret_excel` prefiere
abandonar a un reclamante, y que la preferencia sobrevive a quitar la censura y a acotar la cola
(`docs/RESULTADO_RET_PREMIA_EL_ABANDONO_2026-07-31.md`). Una «reducción de varianza» leída sobre
una métrica que premia el abandono no significaría nada. ReT se reporta al lado, con la
advertencia.

## Falsadores

| falsador | por qué puede fallar |
|---|---|
| `f1_the_arms_are_actually_different_configurations` | si `neuron_memory` y `ofat` eligen la MISMA configuración, `H1` y `H3` son vacías por construcción: dos brazos idénticos dan resultados idénticos |
| `f2_the_intensity_ladder_actually_escalates` | sin más eventos de riesgo al subir el multiplicador, «heterogeneous intensities» no se ha probado |
| `f3_ttr_censoring_is_disclosed_and_comparable` | `system_ttr` está **censurado por la derecha** por construcción, así que su media es optimista; si además la **fracción censurada difiere** entre brazos, la comparación está confundida y el número no vale |
| `f4_arms_share_seeds_and_ladder` | brazos con semillas distintas medirían suerte, no política |
| `f5_variance_is_across_intensities_not_within` | `H3` habla de varianza **entre intensidades**; calcularla dentro de una sola sería otra hipótesis |
| `f6_seeds_are_virgin` | reutilizar semillas invalidaría la confirmación |

## Regla de lectura, fijada de antemano

* **`H1`**: el híbrido tiene `system_ttr_mean` **menor** que el estático, con `LCB95 > 0` en la
  diferencia pareada → **H1 sostenida**. En cualquier otro caso, **no sostenida**, y se dice así.
* **`H3`**: la varianza entre intensidades de `flow_fill_rate` del híbrido es **menor**, con
  `LCB95 > 0` en la diferencia → **H3 sostenida**.
* **Si `f3` falla** (censura no comparable), **`H1` no se reporta como medida**: se reporta el
  defecto y se propone el estimando alternativo. No voy a publicar un tiempo de recuperación que
  sé confundido.

**Y una advertencia que me impongo:** el estático aquí es **el diseño de Garrido, bien
implementado**, no un hombre de paja. Si gana, lo digo.
