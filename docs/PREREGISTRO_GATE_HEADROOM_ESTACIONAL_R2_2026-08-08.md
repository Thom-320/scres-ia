# Preregistro — gate de headroom bajo la física de Garrido: demanda estacional × R2 sorteado

**Escrito y commiteado ANTES de correr.** Runner: `scripts/run_seasonal_r2_headroom_gate_v1.py`.
Custodia: **réplica declarada del bloque `8600001–8600012`** (bloque de desarrollo del proceso de
demanda, `docs/PREREGISTRO_PROCESO_DEMANDA_2026-08-07.md`). **Ninguna semilla nueva** — no existen:
`ENMIENDA_4` deja el inventario en cero bloques vírgenes. Esto es **desarrollo por construcción** y
no puede adjudicar nada.

Sucede y **no reemplaza** a `docs/RESPUESTA_GARRIDO_R2_ALEATORIZADO_2026-08-08.md`, que dejó el
hueco nombrado con precisión:

```
R2_PROFILE_VARIATION           ANSWERED_BY_EXISTING_SCREEN   (4.860 evaluaciones, development)
R2_WITHIN_EPISODE_REALISATION  NOT_MEASURED
```

Este documento mide **lo segundo**, y lo cruza con la demanda que Garrido pidió reproducir.

## 0. Qué mide este gate, y sobre todo qué NO mide

**Mide el techo, no la conversión.** El estimando es `H_oracle`: lo que compra un oráculo que
**conoce el régimen** frente a la mejor constante robusta. Es una cota superior de todo lo que
cualquier aprendiz podría capturar.

**No mide `H_obs`**, no entrena nada, y no autoriza entrenar nada. Ésa es exactamente su virtud:
es barato, y si el techo no llega, ningún KAN, MLP o PPO puede llegar tampoco. Es la regla de
`CLAUDE.md` — *«Create headroom first… Measure `H_regime` **before** spending on a learner»*.

**Y no es una perilla que giramos hasta que gane alguien.** Las dos piezas de física nueva vienen
de la fuente: `GR_{t+v}` es la Ec. (1) de Garrido, Pongutá & García-Reyes (2024) §3.2, y R2 es la
familia de riesgos de su propia tesis. **Cero retuning después de ver el gate** — §7 lo fija como
regla vinculante, porque elegir el entorno para producir un ganador deseado es el fallo que esta
casa ya se señaló a sí misma.

## 1. Por qué esto puede tener headroom cuando nada lo ha tenido

Nuestra rejilla entera ha corrido sobre demanda **estacionaria** — `U(2400,2600)` cada 24 h, §6.3.4
de la tesis. **Un entorno estacionario no puede tener un óptimo que se mueva por demanda**: es una
imposibilidad estructural, no un resultado negativo. Que `H_regime` diera 0 bajo esa física no dice
nada sobre esta.

Y por la regla permanente del PI —*«un negativo bajo la física vieja no es un negativo bajo la
física nueva»*— ambas hipótesis se re-testean.

**El mecanismo candidato, declarado antes de medir:** con fase estacional, el buffer y los turnos
óptimos podrían variar con la fase; con R2 sorteado por episodio, podría existir valor en **inferir**
el régimen sobre la marcha, que es un mecanismo distinto del de perfil fijo — bajo perfil fijo el
régimen es conocible y una constante puede ser óptima.

**El prior honesto en contra:** el único headroom material que este proyecto ha medido —
`H_PI = 0,1515`, Program O — vino de **contención sobre un recurso escaso no fungible**, con el nulo
fungible en exactamente 0. No vino de la forma de la superficie. Y la superficie ya es materialmente
no lineal (`functional_form_diagnostics`: RESET rechaza linealidad en 6/6 contextos, ΔAIC 1.300 a
14.416 contra el lineal) **sin que exista prima neural** — el MLP busca peor que el lineal, curvatura
0,076 contra ruido 0,317. **Curvatura no es headroom.** Si la demanda estacional sólo añade
curvatura y no mueve el argmax, este gate debe cerrar en negativo, y ésa es la lectura por defecto.

## 2. El diseño — 2 × 2, y una celda es el ancla

| eje | niveles |
|---|---|
| **demanda** | `D0` heredada, `U(2400,2600)`/24 h (§6.3.4) · `D1` generador `GR_{t+v}` de Garrido 2024, Holt con `α, γ ~ U[0,1)` sobre semilla estacional de 36 periodos, **reescalado para conservar la demanda media** |
| **riesgo** | `R_fixed` perfil congelado (statu quo) · `R_draw` R1 fijo, y frecuencia e impacto de **R2 sorteados por episodio** desde soporte congelado |

`D0 × R_fixed` **es el statu quo** y sirve de ancla de reproducción (`f3`). Las otras tres celdas son
física nueva. El cruce permite además descomponer: cuánto lo lleva la demanda, cuánto el riesgo,
cuánto la interacción — gratis, una vez corrido el 2 × 2.

**El reescalado de media es obligatorio y no es cosmético.** Sin él, «más headroom» y «más carga»
son indistinguibles, y el resultado no significaría nada. Es `f2`.

## 3. Endpoints — tres, y el gate NO corre sobre el histórico

Hay una razón medida para no puntuar esto con `ret_excel`: su óptimo de reparto es **0,1**, que
entrega **50,7 %** de fill y abandona **318.621** raciones, contra **0,5**, que entrega 79,5 % y no
abandona ninguna. **Nunca se entrena ni se decide sobre `ret_excel`.**

| papel | endpoint | por qué |
|---|---|---|
| **primario, decide el gate** | `flow_fill_rate` | servicio directo; inmune tanto al defecto de abandono como a la duplicación de κ |
| **secundario, se reporta** | `R_cobb_douglas` | pasa el test de abandono (óptimo 0,5, de acuerdo con servicio en las seis regiones) |
| **sólo se reporta** | `ret_excel_risk_conditional` | métrica histórica, comparabilidad con el atlas; **no decide** |

Y una advertencia que este mismo día quedó medida: bajo `c = 1`, κ̇ es un duplicado de ζ+ε
(`corr = 0,999993`), así que el Cobb-Douglas se reporta **con su diagnóstico de independencia de κ̇
por celda**. Si los tres endpoints discrepan en signo, **la discrepancia es el hallazgo** y se
reporta como tal, no se elige el que convenga.

## 4. La barra, fijada aquí

**`LCB95(H_oracle) ≥ 0,01`** sobre el endpoint primario. Es la barra que el screen de riesgos ya
preregistró, y usarla es lo que hace ambos resultados comparables — aquel dio máximo `6,93e−05`
`[0 , 2,08e−04]`, **144× por debajo**.

Un valor entre 0,01 y 0,05 **no abre nada por sí solo**: autoriza *diseñar* una confirmación, nunca
entrenar ni reclamar.

## 5. Falsadores — cada uno con por qué puede fallar, y todos pueden pasar

| falsador | qué exige | por qué puede fallar |
|---|---|---|
| `f1_D1_really_changes_the_process` | CV semanal y ACF a lag 12 de `D1` deben separarse materialmente de `D0` | el reescalado de media podría aplanar la estacionalidad; si `D1 ≈ D0`, no hay física nueva y el eje entero es decorativo |
| `f2_mean_demand_is_preserved` | `\|mean(D1)/mean(D0) − 1\| < 0,01` | si falla, cualquier headroom queda confundido con carga y **el resultado no es interpretable en ninguna dirección** |
| `f3_anchor_reproduces` | `D0 × R_fixed` debe reproducir el `H_regime` de la superficie sellada | si no reproduce, las tres celdas nuevas no son comparables con nada y el artefacto se retira |
| `f4_the_oracle_is_an_oracle` | el oráculo con régimen conocido debe dominar débilmente a la mejor constante robusta, **por construcción** | si el oráculo pierde contra una constante, está mal implementado; control de integridad puro |
| `f5_the_uninformed_placebo_does_not_match_the_oracle` | un placebo que **varía en el mismo calendario pero no lee nada** debe quedar por debajo del oráculo | **el falsador decisivo, y ya falló una vez**: en `op12` el placebo desinformado batió a la regla condicionada al estado — el valor estaba en que el periodo variara, no en qué lo hacía variar. Si vuelve a pasar, no hay valor de estado por mucho que `H_oracle` suba |
| `f6_R_draw_really_randomises` | la frecuencia y el impacto realizados de R2 deben variar entre episodios cubriendo el soporte declarado | un sorteo inerte devolvería `R_fixed` con otro nombre |
| `f7_common_random_numbers` | todas las políticas ven **la misma realización** contrafactual | sin esto la comparación mide ruido de sorteo, no política |
| `f8_no_fresh_seeds` | réplica declarada de `8600001–8600012` | custodia central; no hay bloques vírgenes que abrir |

Ninguno es un test de signo sobre una cantidad que cruza cero, y las ramas de decisión de §6
**particionan** el espacio en vez de enumerar casos cómodos — las dos reglas que dejó el cierre de
Program L.

## 6. Reglas de lectura — las tres direcciones, fijadas de antemano

* **`LCB95 < 0,01` en las cuatro celdas** → `STOP_NO_HEADROOM_UNDER_GARRIDO_PHYSICS`. **No se
  entrena nada.** Y es un resultado, no un fracaso: cierra las dos peticiones de Garrido —su demanda
  de 2024 y su R2 aleatorizado— con una medición, bajo su propia física, y con el techo medido en
  vez de supuesto.
* **`LCB95 ≥ 0,01` y `f5` pasa** → `CEILING_OPEN_UNDER_GARRIDO_PHYSICS`. Se reporta qué celda lo
  lleva y la descomposición demanda / riesgo / interacción. **Autoriza diseñar una confirmación.
  No autoriza entrenar, ni reclamar prima neural, ni comparar KAN contra MLP.**
* **`LCB95 ≥ 0,01` y `f5` falla** → `PERIOD_VARYING_NOT_STATE_VARYING`. El techo sube pero el
  placebo lo iguala: el valor está en variar, no en leer el estado. Se reporta con ese nombre y
  **tampoco** se entrena. Es el resultado de `op12` repetido bajo física nueva, y merece decirse.

## 7. La regla que hace esto honesto o no lo hace

**Cero retuning tras ver el gate.** El generador `D1`, el soporte de R2, el reescalado de media, la
barra, los tres endpoints y los ocho falsadores quedan fijados por este documento. Si el gate cierra
en negativo, **no se ajusta la física y se vuelve a correr**. Subir el ruido o bajar la capacidad
hasta que una arquitectura gane no produce un hallazgo sobre cadenas de suministro: produce un
hallazgo sobre el entorno que construimos, y un revisor de C&IE lo ve en una tarde.

Cualquier variante posterior de la física es **una familia nueva, declarada y con su multiplicidad
pagada**, exactamente como se hizo con las 188 derivaciones métricas.

## 8. Alcance

`DEVELOPMENT_ON_DECLARED_REPLAY`. No abre semillas, no adjudica, no regradúa RQ2 de Paper 2, no
autoriza aprendices. El techo que mida es un techo de desarrollo y se cita con ese grado.
