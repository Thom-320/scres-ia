# Enmienda — lectura de GR y falsadores estacionales de Paper 2

**Fecha de congelación:** 2026-08-08, antes de cualquier nueva corrida de sensibilidad.

Esta enmienda no reinterpreta retrospectivamente el resultado `ENGINE_PARTIAL`. Separa el uso que
Garrido da a `GR` del instrumento observable que los investigadores desean añadir.

## Alcance de la fuente

En Garrido, Pongutá & García-Reyes (2024), §3.2, `GR` es una variable de entrada/generador de
trayectorias de gross requirements para las variables de decisión. La fuente no define una demanda
realizada independiente contra la cual puntuar `GR`. Por tanto, la caracterización literal sólo
adjudica estructura de trayectoria: CV semanal y ACF estacional.

Los valores ya observados —CV semanal 0,177 y ACF a lag 12 de 0,839— quedan como evidencia de
`garrido_generator`, no como evidencia de forecast skill.

## Extensión nuestra

`forecast_mode="holt_winters_observable"` añade un término estacional a un instrumento definido por
este proyecto. Es una extensión de robustez, nunca una reparación de la ecuación de Garrido. Se
evalúa contra la demanda realizada al horizonte t+1 y conserva `forecast_shuffled` como placebo que
mantiene la distribución marginal y destruye la fase.

## Falsadores revisados

- `g4`: se juzga con 2.000 sorteos instrumentales de alpha/gamma, soporte, momentos y distancia KS.
  Los doce episodios científicos se reportan, pero no deciden si `U[0,1]` es un sampler válido.
- `g5`: se juzga en el horizonte correcto mediante MASE frente a naive y seasonal-naive, RMSE, sesgo
  y cross-correlation en lags −2…+2. Se incluyen una sinusoide de periodo conocido y un impulso
  aislado antes de leer el panel de producción.

## Regla de publicación

La sensibilidad permanece `DEVELOPMENT_SENSITIVITY`; no puede regraduar RQ2 ni abrir semillas de
confirmación. Si el panel estacional falla, Paper 2 se somete con el claim limitado al proceso de
demanda heredado. El límite de ejecución es 72 horas y no se permite retuning.
