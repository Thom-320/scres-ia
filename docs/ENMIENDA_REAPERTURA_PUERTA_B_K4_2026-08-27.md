# ENMIENDA 1 al preregistro `gate_b_reopening_power_v1` — K de 8 a 4

| Campo | Valor |
|---|---|
| Fecha | 2026-08-27 |
| Enmienda a | `docs/PREREGISTRO_REAPERTURA_PUERTA_B_POTENCIA_2026-08-27.md` |
| SHA-256 del preregistro enmendado | `af195737451b9fed203084458561347ebf0764974bbfa4229bbd701a7dc223b3` |
| Estado del bloque | **`9700001–9700112` sigue VIRGEN** — ninguna semilla se abrió |
| Qué cambia | `--folds 8` → `--folds 4` |
| Qué NO cambia | unidad de análisis, n, endpoint, comparador, SESOI, regla de decisión, falsadores |

## Por qué

El preregistro dice, textualmente, en su sección 3:

> **K = 8 folds** de 14 semillas. **K no es la unidad de análisis: sólo gobierna el
> ajuste.** 8 ajustes sobre 112 semillas de datos.

Esta enmienda toca únicamente eso. La unidad de análisis sigue siendo la semilla,
siguen siendo 112, y `per_seed` sigue produciendo una puntuación retenida por
semilla porque `grouped_folds` asigna cada semilla a exactamente un fold de test
para cualquier K.

## El coste medido que la motiva

Un intento previo con K=8 corrió 3 h 56 min en el Mac antes de que el PI pidiera
liberar la máquina. Su registro deja el coste medido:

```
R1r listo (596s) · R2r (1096s) · R1r+R2r (1819s)   <- simulación completa: 30 min
fold 0 listo (11493s)                              <- UN fold: 3,2 h
```

La simulación de las 17.136 corridas DES cuesta **30 minutos**. Cada fold cuesta
**3,2 h**. Con K=8 el total es ~25 h; el cuello no es la física sino los ajustes.

El mecanismo es cúbico y está identificado: el proceso gaussiano y sus variantes
con retardos resuelven un Cholesky sobre la matriz de núcleo del conjunto de
entrenamiento. Con K=8 ese conjunto es 7/8 × 17.136 = **14.994 filas**; con K=4 es
3/4 × 17.136 = **12.852 filas**. El coste por ajuste cae a (12.852/14.994)³ =
**0,63×**, y los ajustes pasan de 8 a 4:

| | K=8 | K=4 |
|---|---|---|
| filas de entrenamiento por fold | 14.994 | 12.852 |
| coste relativo por fold | 1,00 | 0,63 |
| número de folds | 8 | 4 |
| **coste total relativo** | **1,00** | **0,31** |
| estimación a velocidad Mac | ~25 h | ~8 h |

## El precio, declarado

Cada modelo se ajusta con **3/4 de los datos en vez de 7/8**. Eso desplaza a la
baja el R² retenido de **todos** los brazos por igual, porque todos comparten
folds, características y presupuesto. El estimando es un **contraste pareado**
entre brazos sobre las mismas semillas, así que un desplazamiento común se
cancela en la diferencia; lo que no se cancela es que un ajuste con menos datos
puede penalizar más a los brazos con más parámetros.

**Esa asimetría corre en contra de la hipótesis que queremos poder detectar.** Si
existe prima neural, K=4 la hace más difícil de ver, no más fácil. La enmienda es
por tanto conservadora respecto del resultado que el PI espera, y así debe
leerse si el veredicto sale `EQUIVALENT` o `UNDETERMINED`.

## Lo que esta enmienda NO autoriza

No cambia el endpoint, ni el SESOI de 0,05, ni la clase comparadora, ni la
reselección dentro del bootstrap, ni la regla de decisión de tres vías, ni
ninguno de los siete falsadores. No abre semillas fuera de `9700001–9700112`. No
reabre `gate_a2_track_b` ni `phase3_decision_surrogate`, que siguen declarados
como derrotas reales y no problemas de potencia.

**F1 se vuelve más exigente, no menos.** El falsador que comprueba que el MDE80
medido queda bajo el SESOI se evalúa sobre la sd por semilla que salga de esta
corrida. Con menos datos por ajuste esa sd puede subir. Si F1 falla, el veredicto
es `UNDETERMINED_UNDERPOWERED` y se publica con el *n* y el K que sí harían
falta — exactamente como está escrito en el preregistro.

## Dónde se ejecuta

Kaggle, kernel de tipo script, CPU, sin GPU. Motivo: el intento en el VPS murió
por OOM —`Killed process 2499640 (python) anon-rss:10650344kB`, 10,6 GB sobre
11 GB— y Kaggle ofrece ~30 GB de RAM, que elimina ese modo de fallo. Camber
quedó descartado hoy porque `stash cp` devuelve `code=13` desde ambos hosts y sin
subida no hay job.
