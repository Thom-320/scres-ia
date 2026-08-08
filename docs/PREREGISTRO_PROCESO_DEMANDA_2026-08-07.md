# Preregistro — caracterización del proceso de demanda

**Fecha:** 2026-08-07, **escrito antes de correr**
**Instrumento:** `scripts/measure_demand_process_v1.py`, sobre las primitivas selladas de
`supply_chain/arm_runner.py` (`scored_orders`, `run_falsifiers`, `seal_and_write`).
**Clase:** diagnóstico de desarrollo. Sin aprendiz, sin adjudicación.

---

## Por qué se corre, y por qué no basta leer `config.py`

Preguntado si la demanda es estática o variable, leí el contrato y **calculé el CV semanal a mano**.
Ésa es exactamente la maniobra que fabricó defectos falsos el 2026-07-30
(`measure-through-the-pipeline`). La cifra derivada a mano **no es un resultado** y no entra en
ningún artefacto. Esto mide la serie **realizada**.

## La distinción que decide algo

Hay dos preguntas y sólo la segunda importa para el claim central del proyecto:

| pregunta | qué contesta |
|---|---|
| ¿**varía**? | si `sd` semanal > 0 |
| ¿**es predecible**? | si varía **con memoria** |

Un proceso **iid varía sin ofrecer estado que condicionar**: el sorteo de mañana es independiente de
todo lo observado hoy. Si la demanda nativa de la tesis es iid, entonces **desde el lado de la
demanda no hay nada con lo que una política state-dependent pueda batir a una constante** — y eso
sería una cuarta razón estructural del `H_regime ≈ 0`, junto a las tres ya registradas en
`CLAUDE.md`. Es comprobable: la autocorrelación a lag 1.

## Dos entornos, porque no comparten proceso

| entorno | `risk_level` | `demand_scale` |
|---|---|---|
| `make_thesis_aligned_training_env` | `current` | no se aplica |
| `make_track_b_env` | `adaptive_benchmark_v2` | **sí** — `supply_chain.py:5449` multiplica por el `demand_scale` del régimen (0,95 · 1,02 · 1,08 · 1,12 · 1,00) |

Track B es además **el único entorno donde ha aparecido señal neural** (+1,44…+2,18). Si resulta ser
también el único con demanda acoplada a un estado persistente, eso es una hipótesis mecanicista, no
una coincidencia — y hay que dejarla escrita antes de ver el número.

## Diseño

12 episodios por entorno, semillas **8600001–8600012**. Acción neutra (cero) en todos los pasos: se
mide el **proceso de entrada**, no una política. La población es `scored_orders(sim)`, la misma que
usa todo lo demás, para que la demanda no se calcule sobre un conjunto distinto al de las métricas.

Agregación semanal desde `warmup_time`. **Las semanas parciales del final se descartan, no se
rellenan con ceros**: una semana parcial rellenada se lee como un colapso de demanda y fabricaría
varianza.

## Falsadores — cada uno puede fallar **y** puede pasar (regla R6)

| # | falsador | por qué puede fallar | por qué puede pasar |
|---|---|---|---|
| **f1** | Los pedidos regulares caen dentro de U(2400, 2600) del contrato. | Un multiplicador o una escala de régimen aplicada a un entorno nominalmente nativo empujaría sorteos fuera de los límites. | Son límites exactos; no hay nada que cruce cero. |
| **f2** | La serie semanal nativa **varía** (`sd` > 0). | Si `sd` = 0 la demanda es literalmente estática — que es la hipótesis bajo prueba. | Cualquier sorteo no degenerado la pasa. |
| **f3** | La autocorrelación a lag 1 de la serie nativa cae dentro de la banda iid \|r\| < 2/√n. | Un proceso con memoria (tendencia, estacionalidad, acoplamiento a régimen) queda fuera de la banda — y eso significaría que **sí** hay estado de demanda que condicionar. | Un proceso iid cae dentro. |
| **f4** | El CV semanal de `track_b` **supera** al nativo. | Si `adaptive_benchmark_v2` no mueve la demanda realizada, los dos CV coinciden y la lectura «Track B tiene estado de demanda» se cae. | Si el acoplamiento al régimen es real, lo supera. |

**f3 se juzga por la magnitud contra su propia banda, nunca por el signo** — es la regla R6, escrita
esta misma tarde tras el fallo de `F1` en L-0.

## Qué NO hace este diagnóstico

- No adjudica ningún carril ni autoriza ningún aprendiz.
- No mide headroom. Que la demanda sea iid **no demuestra** `H_regime = 0`; sólo elimina una de las
  fuentes posibles de estado. Los riesgos siguen siendo persistentes y siguen siendo candidatos.
- No abre semillas de custodia.

## Salida

`results/demand_process/result.json`, sellado con `seal_and_write`.
