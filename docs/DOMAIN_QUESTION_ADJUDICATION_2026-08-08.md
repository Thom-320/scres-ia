# Adjudicación de las barreras de dominio — 2026-08-08

Qué dice **la fuente** sobre cada pregunta, qué queda sin establecer, y qué efecto tiene eso sobre
las lanes. La distinción que hace útil esta tabla es entre **respondida por la tesis**, **no
establecida por la tesis** y **bloqueante**. «No documentado» no es «no», y tampoco es una invitación
a suponer que sí.

Preguntas redactadas en `docs/PREGUNTAS_GARRIDO_2026-08-08.md`.

## La tabla de autoridad

| ID | Respuesta hoy | Fuente | Estado | Efecto |
|---|---|---|---|---|
| **Q11 / R09** | Hay salida permanente a `Ut` por cancelación y por desbordamiento de cola; **no** hay caducidad física de misión probada | tesis pp. 75, 97 §6.5.4 | `PI_REQUIRED` | R09 **no** reabre |
| **Q6 / Q7** | No hay recurso compartido mutuamente excluyente documentado; las recuperaciones son procesos independientes | tesis, familia de riesgos | `PI_REQUIRED` | lane cerrada |
| **Q13** | Los 21 tipos existen; la **no sustituibilidad**, la capacidad compartida y la mezcla persistente no están documentadas | tesis §§ producto y supuestos | `PI_REQUIRED` | sólo frontera, no representativo |
| **Q14** | Cadencia y lotes sí; la **economía** (tarifa fija vs pago por uso) no | tesis Op10/Op12 | `PI_REQUIRED` | sólo calibración |
| **M2 (a)** | El constructo de aceptación de la tesis es **ReT canónico** | tesis, definición de ReT | `ANSWERED` | CVaR **no** es gate de la tesis |
| **M2 (b)** | La doctrina real sobre colas es **desconocida** | ninguna | `PI_REQUIRED` | el guardrail de cola **se conserva** |
| **M1** | El orden de eventos con marca temporal idéntica **no está especificado** | tesis, matriz de salida | `BLOCKING` | replay exacto bloqueado |

## Lo que cada estado significa, y por qué importa la diferencia

**`ANSWERED`** — la fuente lo dice. M2(a) es el único caso: la tesis define ReT combinando autotomía,
recuperación, disrupción y fill rate, y **no** aparece ningún gate de CVaR, de peor producto ni de
no-inferioridad de cola. De ahí se sigue una cosa y sólo una: **ese guardrail es nuestro, no suyo.**

De ahí **no** se sigue que se pueda quitar. La ausencia de un criterio de cola en una tesis no
demuestra que la doctrina militar acepte sacrificar la peor unidad o la peor campaña, y M2(b) sigue
sin responder. Por tanto los dos claims se separan:

* **claim del constructo de tesis** — ReT canónico primario, métricas de cola secundarias;
* **claim de desplegabilidad** — los guardarraíles de cola y de peor producto **se mantienen** hasta
  que la doctrina diga otra cosa.

Ninguna respuesta reescribe el `STOP_PROGRAM_O_AFTER_CORRECTIVE_VALIDATION` ya emitido. Gobierna lo
que venga después.

**`PI_REQUIRED`** — la tesis no lo establece y ningún experimento nuestro puede resolverlo. Son
hechos del dominio. La consecuencia práctica es la misma en las cuatro: **la lane queda cerrada
hoy**, no en suspenso ni «pendiente de más cómputo».

Q11 merece un matiz porque es el reabridor más fuerte y es fácil auto-engañarse con él. La fuente
**sí** respalda abandono permanente: p. 75 nombra el tiempo de cancelación que recategoriza
backorders a `Ut` sin sacarlos del cálculo, y §6.5.4 lo implementa con tope 60 y desalojo. Lo que la
fuente **no** respalda es la **conjunción** que R09 necesita:

> plazos duros de misión **más ajustados que las recuperaciones** **y** autoridad rica de admisión.

Una sola de las dos no basta, y la variante disparada por capacidad —la que sí existe— ya está
medida y cerrada.

**`BLOCKING`** — M1 es de otra clase. No abre ni cierra headroom; determina si el replay exacto
significa lo que dice. Cuando dos eventos caen en la misma marca temporal, la tesis no revela si el
snapshot de la petición ocurre antes o después. **No hay inferencia elegante que lo arregle**: el
planificador tiene que elegir un orden, y elegirlo nosotros sin declararlo sería fabricar una
convención y presentarla como reproducción.

## El efecto del bloque virgen sobre todo lo anterior

Aunque Garrido validara mañana Q13, **eso no revive el techo clarividente retractado**. Autorizaría
diseñar un contrato físico nuevo desde cero, con su propia preregistración y su propio bloque. Un
hecho de dominio no resucita una réplica fallida — y confundir esas dos cosas sería exactamente el
mecanismo que la retractación de hoy existe para impedir.

## El árbol de decisión, fijado antes de recibir respuesta

```
Q11 plazo duro + triaje = SÍ
    -> contrato de admisión consciente de misión (POMDP)
si no, Q6/Q7 recurso escaso compartido = SÍ
    -> contrato de asignación de reparaciones bajo daño incierto (POMDP)
si no, Q13 no sustituibilidad + capacidad compartida + mezcla incierta persistente = SÍ
    -> contrato multiproducto de horizonte rodante
si no
    -> NO hay aprendiz nuevo
    -> terminar P2
    -> desarrollar «Before Learning» como paper de método
```

**La rama por defecto es la última**, y está escrita antes de conocer las respuestas precisamente
para que no se pueda derivar hacia «entrenemos algo igualmente» cuando lleguen.

## Estado del portafolio que se sigue de esta tabla

```
CONTRATO ACTUAL:
  NO NEURAL HEADROOM ESTABLISHED
  NO KAN PREMIUM ESTABLISHED
  NO NEW LEARNER AUTHORIZED
PAPER 2:
  GO
```

Con una precisión sobre el KAN, porque hay dos hipótesis distintas y sólo una está cerrada:

* `results/kan_mlp_r2_benchmark_v2/result.json` → `EQUIVALENT_BY_TOST_CHOOSE_MLP_BY_PARSIMONY`;
* `results/surrogate_architecture_bakeoff/result.json` → `KAN_SEARCHES_WORSE_THAN_A_MATCHED_MLP`.

De ahí: `KAN_PREMIUM_PRIOR = LOW` bajo los contratos evaluados. Pero **una prima neural recurrente
en un POMDP nuevo sigue siendo posible** — no es la misma hipótesis, y cerrarla por asociación sería
el mismo error de tipo que confundir los dos bucles de la Q1.
