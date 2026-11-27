# Validez de la cota penalizada BSS (revisión de Opus, 2026-11-26)

Revisé `scripts/bss_inner.py`, `bss_lib.py` y `bss_bound.py` para responder a las dudas de Gemini. Los valores son provisionales, de M=8 rep0 (`full_M8_rep0.json`). La cifra final (M=64) sigue pendiente.

## Dudas de Gemini y respuesta

1. **¿Las muestras internas comparten semilla con la cinta externa?**
   - No. Usan su propio `SeedSequence([20261126, celda, cinta, M, rep])` (`bss_inner.py:48`), que nunca es una semilla del contrato.
   - El futuro se muestrea de la ley condicional a la historia observada H_t:
     - El filtro exacto coincide con la enumeración: diferencia máxima 4,4e-15 en 1.152 comprobaciones (`step1b_filter_exact.json`).
     - El muestreador coincide con el rechazo dentro de ~1 se (`step1_validate.json`).
   - Solo varían entre semillas `order_products` y `order_quantities`: 144/144.
2. **¿V̂ se ajustó sobre las mismas cintas?**
   - No hay ajuste. V̂ es el valor de continuación de un controlador clásico fijo, κ, estimado por simulación anidada.
   - La cota es válida para cualquier κ. La elección de κ solo afecta lo apretada que queda.
3. **¿La penalización tiene media cero?**
   - Sí, por construcción. J = Σ_t [Q_t(p_{t+1}) − V_t(p_t)], que telescopa a R − Σ z_t con z_t = W_{t+1} − E[W_{t+1} | F_t].
   - Con estimadores internos ruidosos se sigue cumpliendo E[z̃_t | F_t] = 0, porque W̃_{t+1} es insesgado dado F_{t+1}. Así que la cota con M finito es **válida y más floja**: el máximo de un objetivo ruidoso solo puede sesgar hacia arriba.
   - Prueba empírica: R − J ≈ 0 para 17 políticas (open-loop, 5 fijas, 10 clásicas, aprendiz). Todos los |z| ≤ 1,56 en las tres celdas.
4. **¿La media del control variate es exacta?**
   - Sí. A lo largo de κ, J(κ) = F̄_0[κ_0], es decir, la estimación interna de E[R | κ], así que E[J(κ)] = V(κ) sin sesgo.
   - Sanidad: J(κ) − V_CL = +0,002 / +0,0006 / −0,004 (se ≈ 0,005).
   - U_pen ≥ J(κ) en todas las cintas.

## Salvedades
- El hueco con control variate, U_pen − J(κ), mide el margen sobre κ. No se ve afectado por el ruido externo de R(κ).
- κ se escogió entre 10 clásicos (`BEST_K`). Si se escogió sobre las mismas cintas, V_CL está levemente inflado. Eso afecta el hueco crudo, no el del control variate.
- Todo usa cintas quemadas de calibración (7480001–048). Es una cota de desarrollo, no una confirmación.

## Cifras provisionales (M=8, rep0; hueco con control variate, IC95 bootstrap)
| celda | U0 − V_CL (sin penalización) | U_pen − J(κ) | IC95 |
|---|---|---|---|
| rho75_share90 | 0,081 | 0,0143 | [0,0104, 0,0187] |
| rho90_share75 | 0,049 | 0,0073 | [0,0054, 0,0098] |
| rho90_share90 | 0,074 | 0,0159 | [0,0118, 0,0205] |

La penalización elimina el 80–84 % del hueco de clarividencia.
- Con M=8, rho90_share75 ya queda por debajo de la regla de parada de 0,01: UCB95 = 0,0098.
- Las otras dos celdas quedan en ~0,015.
- Como el sesgo por M finito es hacia arriba, M=64 y la extrapolación (`bss_extrap.py`) solo pueden bajar estas cifras.

## Cifras intermedias M=32 (reps 0–3 del VPS; `full_M32.json`, `extrap_M32.json`), verificadas por Opus 10:20Z
| celda | U0 − V_CL | U_pen − J(κ) | IC95 | U_pen − V_CL crudo [IC95] | extrapolación 1/M (M→∞) |
|---|---|---|---|---|---|
| rho75_share90 | 0,081 | 0,0063 | [0,0048, 0,0079] | 0,0072 [−0,0024, 0,0170] | 0,0043 |
| rho90_share75 | 0,049 | 0,0038 | [0,0029, 0,0049] | 0,0041 [−0,0046, 0,0130] | 0,0025 |
| rho90_share90 | 0,074 | 0,0092 | [0,0065, 0,0121] | 0,0055 [−0,0045, 0,0156] | 0,0031 |

- La penalización ya elimina el 88–92 % del hueco de clarividencia.
- Regla de parada 0,01 (UCB95 del hueco CV): se cumple en rho75_share90 y rho90_share75; rho90_share90 queda en 0,0121 (punto 0,0092).
- Media cero: máx |z| = 1,83 en 51 pruebas (17 políticas × 3 celdas). U_pen ≥ J(κ) en todas las cintas.
- El hueco crudo contra V_CL ya incluye el 0 en las tres celdas: con 48 cintas no se distingue el techo de la política clásica.
- La extrapolación es diagnóstico (3 puntos M=8/16/32), no cota. La cota válida es la de M finito.
- Pendiente: reps 4–7 de Kaggle → M=64.

## Cifra final M=64 (reps 0–3 VPS + 4–7 Kaggle; `penalizada/full_M64.json`, `extrap_M64.json`), verificada por Opus
| celda | U0 − V_CL (clarividencia) | U_pen − J(κ) (control variate) | IC95 | U_pen − V_CL crudo [IC95] | hueco eliminado |
|---|---|---|---|---|---|
| rho75_share90 | 0,081 | **0,0045** | [0,0035, 0,0056] | 0,0053 [−0,0041, 0,0149] | 93 % |
| rho90_share75 | 0,049 | **0,0030** | [0,0022, 0,0038] | 0,0027 [−0,0058, 0,0115] | 94 % |
| rho90_share90 | 0,074 | **0,0068** | [0,0045, 0,0094] | 0,0045 [−0,0053, 0,0147] | 94 % |

**Veredicto:** se cumple la regla de parada (UCB95 del hueco < 0,01) en las tres celdas. Ninguna política no anticipativa (RL incluido) puede superar al controlador clásico κ en más de 0,0056–0,0094 de valor esperado (escala 0–1, UCB95) sobre estas cintas; el hueco crudo contra V_CL no se distingue de 0.

- La cota es válida a M finito y solo puede bajar con más M: la secuencia M=8/16/32/64 es monótona (p. ej. 0,0160 → 0,0095 → 0,0065 → 0,0053).
- Extrapolación M→∞ (diagnóstico, no cota): 0,002–0,004 con 1/M; el ajuste 1/√M da valores negativos, señal de que no es la forma correcta.
- Media cero: máx |z| = 1,80 en 51 pruebas. U_pen ≥ J(κ) en todas las cintas. Réplicas 4–7 distintas de 0–3 (comprobado en las mismas cintas).
- Salvedades vigentes: cintas de calibración quemadas (7480001–048), 48 cintas, κ elegido entre 10 clásicos. Es un resultado de desarrollo; una confirmación pide cintas nuevas preregistradas.
