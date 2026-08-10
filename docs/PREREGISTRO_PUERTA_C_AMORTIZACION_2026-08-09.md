# Preregistro — Puerta C, amortización del planificador

**Fecha:** 2026-08-09 · **Autoriza:** PI (carril de amortización, enmienda del 2026-08-09)
**Contrato marco:** `docs/CONTRATO_PROGRAMA_N_PRIMA_NEURAL_2026-08-09.md`
**Escrito ANTES de correr la etapa C0.**

## El estimando, que no es calidad

```
Delta_amortizacion = C_online(experto) - C_online(red)
```

sujeto a **no-inferioridad de calidad preregistrada**. La Puerta C no reclama que la red decida
mejor; reclama que decide **igual de bien, mucho más barato**. Es el único carril que el contrato
permite sin residual de calidad, y es el que la competencia (Ding, MAPPO) no toca.

## La regla de parada, del plan y anterior a todo

> *si el MPC no mejora a la heurística, cerrar antes de entrenar — la amortización de algo que no
> vale nada no vale nada*

Y su simétrica, que el plan no escribió y que hace falta igual:

> *si el planificador no es caro, no hay nada que amortizar*

El experto tiene que cumplir **las dos**: ser **mejor** que la heurística barata y ser **caro**.
Una sola no basta, y la conjunción es lo que la etapa C0 audita **antes** de entrenar nada.

## Etapa C0 — auditoría de expertos candidatos

Sólo existen dos candidatos a experto en el repositorio. C0 los mide contra la conjunción.

**Candidato 1 — `strong_mpc` de K3** (`results/k3/strong_mpc_terminal.json`,
`PROMOTE_K3_TO_CONFIRMATION`). Bate al mejor clásico en `ret_order` por
**+0,01242 [+0,00546, +0,01928]** sobre 300 tapas de test, con `lost` no-inferior y recurso exacto.
Cumple «mejor». **Falta comprobar «caro».**

**Candidato 2 — `DirectDESMPC` de E\*** (`results/estar_hcompute_preflight_v1`,
`H_COMPUTE_PASS_NEURAL_AMORTIZATION_ELIGIBLE`). 192 llamadas al DES y p95 de 0,1553 s en el nivel
más alto. Cumple «caro». El artefacto es `engineering_only` con `learner_trained: false` y **no
mide calidad contra ninguna heurística**, así que **falta comprobar «mejor»**.

## Falsadores de C0

* **c1_the_expert_actually_plans** — se instrumenta la política del experto y se cuentan sus
  *rollouts*, llamadas al simulador e iteraciones de solver por decisión. Falla si el conteo es
  cero, es decir si el «planificador» es una regla en forma cerrada.
  *Por qué puede fallar:* un nombre puesto por el papel esperado y no por una propiedad medida es
  el defecto que ya cometimos con «techo»; este falsador existe porque lo cometimos.
* **c2_the_expert_is_more_expensive_than_its_amortizer** — latencia por decisión medida en el mismo
  banco y con las mismas repeticiones para el experto y para una red del presupuesto declarado.
  Falla si el experto no es más caro. *Puede fallar:* una regla lineal es más barata que cualquier
  paso hacia adelante de una red, y entonces `Delta_amortizacion` es **negativo por construcción**.
* **c3_the_expert_beats_the_cheap_heuristic** — para cada candidato, ventaja de calidad medida
  contra el mejor comparador barato, con LCB95 > 0. Se lee del artefacto sellado cuando existe y se
  declara AUSENTE cuando no. *Puede fallar:* y falla explícitamente si nadie la midió nunca.
* **c4_a_control_must_separate_the_candidates** — falla si los dos candidatos reciben el mismo
  veredicto en las tres condiciones anteriores, lo que significaría que C0 no discrimina y que la
  auditoría no mide nada.
* **custody** — `NOT_APPLICABLE` en C0: no se abre semilla y no se entrena. Se leen artefactos
  sellados y se cronometra código.

## Reglas de decisión, escritas antes

| resultado de C0 | veredicto | qué pasa |
|---|---|---|
| algún candidato es **caro Y mejor** | `EXPERT_QUALIFIES_PROCEED_TO_C1` | se construye C1: imitación con presupuestos emparejados |
| todos **mejores pero baratos** | `NOTHING_TO_AMORTIZE` | se cierra sin entrenar |
| todos **caros pero sin calidad medida** | `NO_EXPERT_WITH_MEASURED_QUALITY` | se cierra sin entrenar; medir esa calidad es un experimento nuevo, no la Puerta C |
| ningún candidato pasa nada | `NO_QUALIFYING_EXPERT` | se cierra sin entrenar |

**Ningún veredicto de C0 autoriza entrenar salvo el primero.** Está escrito así a propósito: la
tentación aquí es entrenar el imitador porque el código ya está listo, y descubrir después que el
experto imitado era una regla de cuatro sumas.

## Etapa C1 — sólo si C0 lo autoriza

Congelar el experto; generar su conjunto de decisiones sobre tapas de desarrollo; entrenar
spline-GAM, MLP, KAN y recurrente con presupuestos emparejados; evaluar en tapas retenidas.
No-inferioridad de calidad con margen **SESOI 0,01** y `Delta_amortizacion` reportado como razón de
latencia y de llamadas al simulador. La escalera del contrato se aplica igual: la red debe igualar
al **mejor no neuronal**, no a la constante.

## Alcance, declarado por adelantado

K3 está clasificado **EXPLORATORIO/CONTESTADO** — su vida útil de 2 semanas contradice la ración no
perecedera de 3 años de la tesis. Si C1 llegara a correr sobre ese sustrato, su afirmación sería de
**coste** y jamás de resiliencia, y ningún número suyo se presenta como reproducción de
Garrido-Ríos (2017).
