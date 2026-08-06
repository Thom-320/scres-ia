# Preregistro v5 — el brazo que faltaba: control predictivo sobre la búsqueda, y Thompson

**Escrito y commiteado ANTES de correr.** Runner: `scripts/run_search_comparator_ladder_v5.py`.
Predecesor: `results/search_ladder_v4/result.json` (`NEURON_LEADS_BUT_NOT_ABOVE_EVERY_MEMORY_ARM`).
Caché sellada `results/surface_cache/wrap288_v1`, bloque quemado `5.300.001–012`, réplica
declarada. **Ninguna semilla nueva.**

## 1. Por qué existe esta corrida

La frase que queremos poder escribir es *«la neurona de la Fig. 5 le gana a todo lo estático y a
MPC»*. **La segunda mitad no está medida**: no hay ningún artefacto con un brazo MPC en esta
escalera. Escribirla hoy sería afirmar algo que no medimos.

## 2. Qué es «MPC» en ESTE entorno, dicho con precisión

Este entorno **no es control intra-episodio**: es **optimización por simulación sobre
configuraciones**, con presupuesto 24 sobre una rejilla de 288. Un controlador predictivo aquí no
regula un proceso — **planifica la siguiente evaluación mirando hacia delante y replanifica tras
cada observación**. Eso es exactamente horizonte deslizante sobre la búsqueda, y su forma canónica
en la literatura de optimización bayesiana es el **Knowledge Gradient**:

```
KG(x) = E[ max_i mu_{n+1}(i)  |  se evalúa en x ]  −  max_i mu_n(i)
```

Se elige `x` por su efecto sobre **el óptimo esperado después de observar**, no por su mejora
inmediata — que es la diferencia exacta entre EI (miope, 1 paso) y control predictivo (lookahead
con replanificación). Se implementa por Monte Carlo sobre la actualización lineal de la media
posterior.

**Y se añade Thompson**, porque si la pregunta es *«¿cuál es el mejor método para este entorno?»*,
el bandido que suele batir a UCB1 tiene que estar. `ucb1_transfer` es hoy el líder de la escalera;
dejar fuera su competidor natural sería elegir el comparador cómodo.

| brazo nuevo | qué es | memoria |
|---|---|---|
| `lookahead_kg` | Knowledge Gradient sobre GP, replanificando cada evaluación | no |
| `lookahead_kg_transfer` | el mismo, con las observaciones de contextos previos normalizadas por prefijo | **sí** |
| `thompson` | muestreo posterior sobre el GP | no |
| `thompson_transfer` | el mismo, con memoria entre contextos | **sí** |

**Los cuatro se corren sobre la MISMA cinta, el MISMO presupuesto y la MISMA caché que los once
brazos de v4.** Nada se re-ejecuta ni se re-baremea: se añade.

## 3. Reglas de lectura, fijadas antes de mirar

Métrica primaria `auc_regret_norm`, **menor es mejor**. Contraste por bootstrap sobre semillas
contra `neuron_memory`.

* algún brazo nuevo **mejor que `neuron_memory` con IC95 que excluye el cero** →
  **`A_CLASSICAL_SEARCH_METHOD_BEATS_THE_NEURON`**. La contribución cambia de forma: el hallazgo pasa a ser
  *«la retención es el ingrediente, y el mejor vehículo no es la neurona»*, y **se dice así**.
* algún brazo nuevo mejor en estimador puntual pero con IC que cruza cero →
  **`INDISTINGUISHABLE_FROM_THE_NEURON`**, igual que `ucb1_transfer` hoy.
* todos los nuevos peores con IC que excluye el cero →
  **`THE_NEURON_HOLDS_AGAINST_LOOKAHEAD_SEARCH`**.

**El compromiso que hace esto honesto:** el ranking completo de los quince brazos entra al
manuscrito y al cuaderno de David, gane quien gane. Si MPC nos gana, **se reporta que nos gana**.

## 4. Falsadores

| falsador | por qué puede fallar |
|---|---|
| `f1_budgets_are_matched` | los quince brazos gastan exactamente 24 evaluaciones por contexto |
| `f2_no_arm_can_read_an_unrun_cell` | `Surface` debe **lanzar** ante una lectura no visitada. Se comprueba **provocándola**, no afirmándola |
| `f3_the_new_arms_are_not_the_old_ones` | las secuencias de visitas de `lookahead_kg` y `gp_ei` deben diferir. **Falla si KG degenera en EI**, y entonces no añadimos un método sino un alias |
| `f4_the_lookahead_arm_can_win` | sobre una superficie sintética con un óptimo que exige lookahead, `lookahead_kg` debe batir a `random`. **Si no gana ahí, no sabe buscar y su derrota aquí no significa nada** |
| `f5_memory_arms_actually_carry_state` | los brazos con transferencia deben terminar con estado distinto del inicial |
| `f6_v4_arms_reproduce` | los once brazos de v4 deben dar el mismo `auc` medio a 1e-9. **Ancla externa**: falla si tocamos algo al añadir |
| `f7_no_fresh_seeds` | custodia central, réplica declarada |

**Alcance:** desarrollo sobre tapes quemados. No abre semillas y no adjudica.
