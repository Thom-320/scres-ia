# ¿Está aprendiendo Real-KAN? — auditoría de interpretabilidad — 2026-07-04

## Qué se hizo

`scripts/audit_real_kan_interpretability.py`: carga el checkpoint Real-KAN ya entrenado
(fixed-RNG, seed 1, `outputs/experiments/track_b_real_kan_fixed_rng_confirm_5seed_60k_2026-07-03`),
sin reentrenar nada. Reactiva `save_act`/`symbolic_enabled` de pykan (desactivados solo por
velocidad durante el entrenamiento, per el docstring de `scripts/real_kan_extractor.py` — no
afectan a los pesos ya aprendidos). Corre 3 episodios reales de evaluación para recolectar
observaciones auténticas (no sintéticas), hace un forward pass sobre ellas, y usa la API propia
de pykan: `model.attribute()` (puntaje de atribución por variable de entrada) y `model.plot()`
(curvas spline aprendidas por conexión).

## Resultado: atribución por variable

Sobre 52 variables de observación v7, el top-10 concentra **47.7%** de la atribución total,
frente a un **19.2%** que correspondería a una red que no discrimina nada (atribución uniforme).
Es decir, la red concentra más de 2.5x el peso esperado por azar en un subconjunto de variables.

Top-10 variables atribuidas (de mayor a menor):

| Variable | Atribución |
|---|---:|
| `rations_theatre_norm` (presión de demanda) | 1.585 |
| `backorder_rate` | 0.627 |
| `cum_downhours_fraction` | 0.622 |
| `regime_recovery` | 0.371 |
| `assembly_line_down` | 0.369 |
| `regime_disrupted` | 0.355 |
| `r14_defect_prob` | 0.306 |
| `op9_down` | 0.294 |
| `prev_step_disruption_hours_norm` | 0.288 |
| `backlog_age_norm` | 0.287 |

Estas son exactamente las variables que uno esperaría que importen para resiliencia: presión de
demanda, tasa de backorder, fracción de tiempo caído, estado del régimen adaptativo, probabilidad
de defecto, antigüedad del backlog. **No son variables arbitrarias ni ruido** — es una lectura
operacionalmente sensata.

En el otro extremo, `op2_down`, `op10_down`, `op12_down`, `ewma_downstream_risk` tienen atribución
exactamente **0.0** — la red aprendió a ignorarlas por completo (o esas señales no varían lo
suficiente en esta política/observación para importar).

## Resultado: forma de las curvas aprendidas (splines)

Para la variable más atribuida (`rations_theatre_norm`, índice 5), la curva aprendida
(`figures/sp_0_5_0.png`) muestra una forma de **umbral-y-rampa**: plana en valores bajos, luego
una subida marcada y sostenida — consistente con "la presión de demanda no importa hasta que cruza
un umbral, y a partir de ahí domina la respuesta". Esto es una función no lineal real, no ruido.

Para una variable de atribución cero (`op2_down`, índice 23), la curva (`sp_0_23_0.png`) está
completamente **vacía/plana** — sin trazo visible. Este contraste es la prueba más directa: el
método no está inventando estructura en todas partes por igual; diferencia lo que la red realmente
usa de lo que ignora.

## Veredicto

**Sí, Real-KAN está aprendiendo algo real y no trivial**, no solo maximizando la métrica por fuerza
bruta ni operando de forma uniforme/aleatoria sobre sus entradas:

1. La atribución se concentra 2.5x por encima de lo esperado por azar en variables
   operacionalmente sensatas.
2. Las curvas aprendidas para variables de alta atribución son formas no lineales interpretables
   (umbral-rampa), no ruido.
3. Las variables de atribución cero muestran curvas literalmente vacías — confirma que el método
   discrimina correctamente entre señal usada y señal ignorada.

Esto es evidencia de aprendizaje real y interpretable, complementaria (no sustituta) a la métrica
de desempeño (ReT Excel). Es también la ventaja concreta de Real-KAN sobre PPO+MLP para esta
pregunta específica: PPO+MLP no ofrece un análogo igual de directo/inspeccionable (requeriría
SHAP/Integrated Gradients sobre una red densa, más indirecto que leer un spline aprendido).

## Próximo paso (no ejecutado aquí, más caro)

`model.auto_symbolic()` podría intentar extraer una fórmula cerrada por curva — no se ejecutó por
costo (52 entradas × 32 neuronas ocultas × 32 salidas = miles de curvas candidatas). Si se quiere
una fórmula simbólica para el paper, restringir primero a las ~10 variables de mayor atribución
antes de correr `auto_symbolic`, en vez de la red completa.
