# Preregistro — inventario: cuándo sostener el buffer, a horas-inventario iguales

**Escrito ANTES de correr.** Runner: `scripts/run_exact_inventory_headroom_v3.py`. **Mismo
esqueleto** que `run_exact_timing_headroom_v2.py`, con un solo actuador cambiado. Custodia: réplica
declarada, sin semillas nuevas.

## 0. Por qué existe, y por qué la familia anterior no podía cerrarlo

`results/exact_timing_headroom_v2/result.json` devolvió `NO_MATERIAL_HEADROOM_WITHIN_EXACT_CLASS`
—UCB95 de 0,001535 y 0,000603 contra una barra de 0,01— y su propio contrato dejó dicho por qué eso
**no** adjudicaba el preposicionamiento: **con el buffer fijado en cero, nunca probó la palanca con
la que R21 está alineado.**

R21 es *Natural disasters*: golpea las operaciones **3, 5, 6, 7 y 9 simultáneamente** con
recuperación `exp(120 h)` (`config.py:469-475`). Tumba producción aguas arriba, y el actuador que
lo cubre es **stock ya colocado abajo**. Los turnos se fijan aquí en **S1** —el nivel que ata— para
que el buffer sea la única palanca libre.

## 1. Medido antes de escribir el runner

A S1, semilla 8600001, y con presupuesto idéntico de **13 semanas de 26**:

| calendario | `L*` con R21 `current` | con R21 `increased` |
|---|---:|---:|
| nunca sostener | 0,360272 | 0,407669 |
| siempre (26 semanas) | 0,239551 | 0,289903 |
| **bloque 0–12** | **0,239551** | **0,289903** |
| **bloque 13–25** | **0,326100** | **0,382382** |
| semanas alternas | 0,239551 | 0,289903 |

**Dos calendarios con el mismo presupuesto separan 0,0866** — ocho veces la barra. La superficie de
timing aquí es real de una forma que en turnos no lo era. Y R21 escalado muerde: +0,047 en todos
los calendarios.

**Esto no es el resultado, y decirlo importa.** Que exista spread entre calendarios **no** implica
que elegirlos conociendo la tape compre nada: si el bloque 0–12 fuera el mejor en **todas** las
tapes, el hueco clarividente sería exactamente 0 pese al spread. Ésa es justamente la distinción
que ha decidido las cuatro familias anteriores, y es lo que este gate mide.

## 2. Diseño — idéntico al de V2, con el actuador cambiado

* **Dos niveles de buffer**, `0,0` y `1,0`, y **exactamente `K = 13` semanas** en `1,0`. Las
  horas-inventario son idénticas **por construcción**, no por tolerancia — el papel que jugaba
  excluir S3 en la familia de turnos.
* **Turnos fijados en S1** durante todo el episodio, así que la intensidad no puede variar.
* **Clase exacta:** los **26** inicios del bloque contiguo de 13 semanas, **enumerados todos**.
  Es la única que puede sostener ausencia, y sólo vía `UCB95 < δ`.
* **Clase enriquecida:** la exacta más 150 subconjuntos aleatorios de 13 semanas, el rankeado por
  presión y el calendario realizado por la regla. **Sólo puede decir `HEADROOM_FOUND` o
  `HEADROOM_NOT_FOUND_BY_SEARCH`.**
* **Celdas:** `R21_current` y `R21_increased`. Estimando de riesgo
  `Δ_R21 = Δ(increased) − Δ(current)`.
* **Endpoint adimensional** `L* = Σ qᵢ[eᵢ−(OPTᵢ+LTᵢ)]₊ / Σ qᵢ[T−(OPTᵢ+LTᵢ)]₊`, en `[0,1]`, con
  denominador invariante a la política. `δ = 0,01` = un punto porcentual de exposición máxima.
* **Sin rama `STOP`:** `BLOCKED_INSTRUMENT` · `HEADROOM_ESTABLISHED` (`LCB95 ≥ δ`) ·
  `NO_MATERIAL_HEADROOM_WITHIN_EXACT_CLASS` (`UCB95 < δ` **y** clase enumerada) · `INCONCLUSIVE`.
* **Los falsadores deciden el `claim_status`**, no sólo el código de salida; `f8` es el control
  autorreferencial.

Los nueve falsadores son los de V2 con `f1` reexpresado sobre semanas de buffer. Siguen pudiendo
fallar todos, y `f7` compara contra el **error estándar pareado de las diferencias**.

## 3. Lo que este gate no podrá decir

No adjudica calendarios fuera de la clase exacta. No adjudica turnos —están fijados—. No combina
los dos recursos. Y no autoriza ningún aprendiz: un techo abierto autorizaría **diseñar** una
confirmación y nada más.

**Cero retuning tras ver el resultado.** `K`, los niveles de buffer, el riesgo, el endpoint, la
barra y los falsadores quedan fijados aquí.
