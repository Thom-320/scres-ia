# Veredicto final overnight: ¿se logró aprendizaje preventivo en Track B? — 2026-07-04

## Respuesta corta

**No.** Ninguna de las variantes probadas esta noche (ni las de sesiones anteriores) muestra
evidencia causal robusta de que PPO+MLP o Real-KAN aprendan a anticipar riesgos por *timing* —
es decir, a preparar la cadena *antes* de un evento de forma que se pueda atribuir valor causal en
ReT Excel a esa anticipación específica. Esto se sostiene con el método ahora válido (fix de RNG),
a través de seis líneas de evidencia independientes, incluyendo un gate oráculo posterior. Se cierra el trabajo de esta noche conforme
a la regla de parada acordada: cuando una rama pasa la señal de entrenamiento pero falla la prueba
causal, no se escala; cuando ni siquiera pasa la señal de entrenamiento, no se corre la prueba
causal.

## Las seis líneas de evidencia, todas ya verificadas número por número

### 1. Interpretabilidad de Real-KAN — sí aprende estructura real, pero no timing

`docs/REAL_KAN_INTERPRETABILITY_VERDICT_2026-07-04.md`. Atribución 2.5x sobre uniforme,
concentrada en variables operacionalmente sensatas (presión de demanda, backorder, tiempo caído,
régimen, probabilidad de defecto); splines aprendidos no triviales para variables de alta
atribución, vacíos para las de atribución cero. **Conclusión**: KAN construye una representación
interna genuina, no ruido — pero esto es aprendizaje de *estructura*, no evidencia de timing
preventivo.

### 2. Contrafactual base (políticas sin belief), escala completa — sin señal causal

`docs/TRACK_B_PREVENTIVE_VS_REACTIVE_FIXED_RNG_VERDICT_2026-07-04.md`. 5 seeds × 12 episodios,
7 riesgos discretos válidos, hasta 205 pares por riesgo. PPO+MLP: deltas medios pequeños y
positivos en 6/7 riesgos pero tasas de pares positivos de 4-32% (muy por debajo del 67%
necesario). Real-KAN: plano o negativo en casi todo, porque su "calma" (cuartil de menor
intensidad) ya está cerca del techo (shift=0.88, op10=0.80, op12=0.97) — casi no hay margen
bajo-vs-alto para que el contrafactual detecte nada.

### 3. Belief-escalar (2 probabilidades congeladas pegadas al vector v10)

`docs/TRACK_B_RISK_BELIEF_RECALIBRATION_VERDICT_2026-07-04.md` +
`docs/TRACK_B_BELIEF_R24_COUNTERFACTUAL_VERDICT_2026-07-04.md`.

- **PPO+MLP**: con el batch_size corregido a 64 (protocolo canónico), el resultado mejora sobre
  las corridas con batch_size=256 mal configurado, pero sigue por debajo de v10 crudo y de la
  referencia v7 al mismo presupuesto (0.005799-0.005820 vs. v10 crudo 0.005811 vs. v7-30k
  0.005843). Señal de entrenamiento débil/nula — no se escaló a contrafactual causal.
- **Real-KAN**: sí mostró una mejora de entrenamiento pequeña pero consistente en las 3 semillas
  (0.005935-0.005937 vs. v10 crudo 0.005915), con batch_size=256 correcto en ambas calibraciones.
  Pero el contrafactual `R_full - R_reset(pre-R24)` sobre esos checkpoints da tasas de pares
  positivos de 2.5%-6.3% — sin señal causal. La "baja de costo" que parecía prometedora con la
  calibración mal hecha (`class_weight="balanced"`) desapareció por completo al corregir la
  calibración (volvió a costo=1.000) — confirmando que era un artefacto de la probabilidad
  distorsionada, no un beneficio real.

### 4. Ruta A — PPO+MLP (encoder pre-entrenado trasplantado al `features_extractor`)

`docs/TRACK_B_BELIEF_ENCODER_RUTA_A_SMOKE_VERDICT_2026-07-04.md`, verificado número por número.

| Variante (30k, mismo protocolo) | ReT Excel | Costo |
|---|---:|---:|
| v10 crudo | 0.005811 | 0.763 |
| belief-escalar (bs64, ambas calibraciones) | 0.005799 / 0.005820 | 0.747 / 0.689 |
| v7 a 30k (control de presupuesto) | 0.005843 | 0.734 |
| **Ruta A (encoder trasplantado)** | **0.005902** | **0.747** |

Ruta A gana limpiamente a todos los comparadores, con CI95 [0.005865, 0.005940] — incluso el
límite inferior queda por encima de todos los demás. **Esta es la señal de entrenamiento más
fuerte de toda la noche.** Pero el contrafactual `R_full - R_reset(pre-R24)` sobre ese mismo
checkpoint (158 pares) da solo **14/158 positivos (8.9%)**, delta medio +0.0000015 —
estadísticamente indistinguible de ruido. **La mejora de Ruta A para PPO+MLP no viene de anticipar
R24 con timing preciso.** Es una representación mejor en algún otro sentido (robustez general,
mejor lectura del estado operativo), no prevención demostrada.

### 5. Ruta A — Real-KAN

`outputs/experiments/track_b_belief_encoder_real_kan_3seed_30k_2026-07-04_v4/`, batch_size=256
(protocolo correcto para Real-KAN), verificado directamente:

| Semilla | v10 crudo | Ruta A | Delta |
|---:|---:|---:|---:|
| 1 | 0.005914 | 0.005945 | +0.000031 |
| 2 | 0.005907 | 0.005895 | **-0.000013** |
| 3 | 0.005923 | 0.005933 | +0.000010 |

Agregado: 0.005924 (Ruta A) vs. 0.005915 (v10 crudo) — una mejora pequeña en promedio, pero **no
consistente entre semillas** (2/3 positivas, 1/3 negativa), y por debajo de lo que ya lograba el
belief-escalar más simple (0.005935-0.005937). A diferencia de PPO+MLP, Ruta A **no** pasa el
primer filtro (señal de entrenamiento clara y consistente) para Real-KAN — por lo tanto, siguiendo
la regla de parada acordada, **no se corrió el contrafactual causal para esta rama**: no tiene
sentido gastar cómputo en probar causalidad sobre una mejora que ni siquiera es un resultado de
entrenamiento sólido.

## Tabla resumen de las primeras cinco líneas

| Rama | Señal de entrenamiento | Señal causal (contrafactual) | Veredicto |
|---|---|---|---|
| Políticas base (sin belief) | — (ya es el resultado headline) | No (4-32% positivos) | No preventivo |
| Belief-escalar PPO+MLP | Débil/nula (bs64 corregido) | No corrida (no ameritaba) | No preventivo |
| Belief-escalar Real-KAN | Sí, pequeña y consistente | No (2.5-6.3% positivos) | No preventivo |
| Ruta A PPO+MLP | **Sí, fuerte y consistente** | No (8.9% positivos) | No preventivo |
| Ruta A Real-KAN | No consistente (2/3 semillas) | No corrida (no pasó el primer filtro) | No preventivo |

### 6. Gate oráculo de techo preventivo — ni el futuro perfecto compra ReT

`docs/TRACK_B_ORACLE_PREVENTION_CEILING_VERDICT_2026-07-04.md`.

Después de las cinco ramas anteriores, se corrió un gate más fuerte: replay del checkpoint
canónico PPO+MLP `v7` fixed-RNG con conocimiento perfecto del calendario real de R22/R24. El
oráculo inyecta boost privilegiado en `shift`, `op10_q` y `op12_q` durante ventanas de preparación
de `L ∈ {1,2,4,8}` semanas antes de cada evento, con `boost ∈ {0.25,0.5,0.75,1.0}`.

La línea base del replay reproduce el checkpoint canónico:
`0.005920543949378895` vs. `0.005920640377903427` conocido (diferencia `9.6e-8`).

| Conjunto | Mejor L | Mejor boost | Δ vs. línea base | Costo |
|---|---:|---:|---:|---:|
| R22 | 8 | 0.75 | +0.000006 | 0.836 |
| R24 | 8 | 1.00 | +0.000021 | 0.997 |
| R22+R24 | 8 | 1.00 | +0.000021 | 0.997 |

El umbral de señal era `+0.0004`; el mejor punto no llega ni al 6% de ese umbral y exige costo
casi máximo. Esto cambia la lectura: el problema ya no parece arquitectura ni crédito de gradiente.
Bajo `control_v1`, reaccionar rápido ya recupera casi todo el ReT que se podía recuperar; no hay
techo preventivo material que Ruta B, RL restringido o una política de dos niveles puedan capturar
sin cambiar la economía del entorno.

## Por qué se detiene aquí

Se probaron, en una sola noche, las dos vías principales que la literatura y el propio diseño
del proyecto sugerían para inducir comportamiento preventivo: (a) pegar una señal de creencia
aprendida como observación adicional, y (b) trasplantar una representación pre-entrenada al
extractor de features de la política. Ambas fueron probadas en ambas arquitecturas. En los dos
casos donde hubo una mejora de entrenamiento real y consistente (Real-KAN con belief-escalar,
PPO+MLP con Ruta A), el contrafactual causal — ahora metodológicamente válido gracias al fix de
RNG de esta sesión — no encontró que esa mejora viniera de actuar antes de los eventos de riesgo.
Y en el único caso restante (Ruta A para Real-KAN), la señal de entrenamiento en sí no fue lo
bastante consistente como para justificar gastar más cómputo en la prueba causal. El gate oráculo
posterior cierra la duda estructural: incluso regalando futuro perfecto y capacidad anticipatoria
sin presupuesto, la ganancia de ReT Excel es prácticamente nula.

Siguiendo la instrucción explícita del usuario ("continúa únicamente si hay señal... si no ves
camino, te detienes"), **no se lanza ningún experimento nuevo esta noche.** Se detuvieron además
tres corridas duplicadas que un agente paralelo (Codex) había relanzado innecesariamente
(`v5`, `v6` de PPO+MLP Ruta A) sobre una pregunta ya respondida, para no desperdiciar cómputo.

## Lo que sí queda como base sólida para el futuro

- El fix de RNG (`strict_exogenous_crn=True` + `regime_rng`) es una contribución metodológica real
  y duradera — sin él, ninguna de las pruebas causales de esta sesión habría sido válida.
- El pipeline de Ruta A (dataset v10 completo, `MLPBeliefExtractor`/`RealKANBeliefExtractor`,
  preentrenamiento BCE, trasplante de pesos) está construido, probado y funcionando — reutilizable
  si se quiere intentar un objetivo de predicción distinto a R24, o mayor escala de entrenamiento,
  en el futuro.
- El hallazgo de que Ruta A mejora PPO+MLP en ReT Excel de forma sólida (aunque no
  preventivamente) es en sí mismo interesante y podría ameritar una mención separada en el paper
  como una mejora de representación, sin la etiqueta "preventivo".
- Después del gate oráculo, Ruta B, RL restringido y política de dos niveles dejan de ser
  prioridades bajo el entorno/reward actual: atacarían arquitectura o crédito temporal, pero el
  techo preventivo material no aparece. Para retomar prevención habría que cambiar explícitamente
  la economía del entorno (por ejemplo, hacer la reacción tardía más lenta/costosa o introducir
  una métrica que valore preparación antes del riesgo). Eso es una línea nueva de investigación,
  no una optimización adicional del spine Q1.
