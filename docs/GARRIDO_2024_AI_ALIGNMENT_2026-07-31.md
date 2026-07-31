# Qué pide Garrido (2024) y qué de eso ya tenemos

**Fuente:** Garrido, Pongutá & Adarme (2024), *Enhancing the Operationalization of SCRES-Based
Simulation Models with AI Algorithms: A Preliminary Exploratory Analysis*, ICCL 2024, LNCS
15168, pp. 80–94. Verificado por contenido: sha256 `3e3bc8f8…` (`scripts/external_sources.py`).

---

## 1. El argumento del paper, en cinco pasos

1. **La resiliencia (SCRES) se mide sobre todo con DES**, y él justifica por qué: datos
   sintéticos baratos, capta transitorios y eventos raros, modela flujos de material **e
   información**, y es lo más usado en la literatura.
2. **El hueco metodológico: el DES no aprende.** Cada réplica arranca de cero — en su Fig. 2,
   `L = {0 + ℓ_i}` corrida tras corrida. Lo llama, con nombre propio, el **«efecto Alzheimer»
   de los modelos de simulación**: la red simulada olvida lo aprendido de los eventos pasados.
   Ese olvido, dice, hace **incompletas** las métricas de SCRES publicadas.
3. **Dónde va el algoritmo.** En su Fig. 2 marca dos nodos deliberadamente **desconectados**:
   ③ las variables de decisión `ρ` y ⑧ la métrica de SCRES. Un algoritmo de IA colocado
   **entre esos dos nodos** convierte la red de **lazo abierto en lazo cerrado**.
4. **Qué familia de IA.** Con la clasificación de Powell (4 niveles: reglas → ML básico →
   reconocimiento de patrones → LLM), concluye que el **nivel 3, redes neuronales**, es el que
   mejor encaja: el aprendizaje entre corridas es intrínsecamente reconocimiento de patrones.
5. **La neurona concreta (Fig. 5).** Las señales de entrada `d_i` son **sus cuatro drivers**,
   ponderados por las **variables de decisión `ρ`**, sumados, y pasados por una activación `f`
   cuyo ejemplo textual es: *«¿es la medida de SCRES en la configuración `x` mayor que en la
   `x−1`?»*. La salida es SCRES.

**Los cuatro drivers son, literalmente, los que medimos** (su Fig. 4):

| driver | símbolo | qué es | en nuestro repo |
|---|---|---|---|
| periodo de autotomía | `Re(APj)` | pedidos que se «amputan» al no llegar a tiempo | `autotomy_share`, `APj` por pedido |
| periodo de recuperación | `Re(RPj)` | cuánto tarda en recuperarse del riesgo | `rpj_mean`, `rpj_p95` |
| periodo de disrupción | `Re(DPj)` | duración de la disrupción | **no lo emitimos como momento** |
| tasa de servicio | `Re(FRj)` | fill rate | `flow_fill_rate` (fuera del set de fidelidad) |

y `ReT = {Re(APj), Re(RPj), Re(DPj), Re(FRj)}`, normalizado 0–100.

Cierra con **tres candidatos nombrados**: *backpropagation*, **redes Kolmogorov–Arnold (KAN)**
y **simulation-optimization como forma de aprendizaje por refuerzo**.

## 2. Qué cambia esto para nosotros

**Primero, una corrección de encuadre.** Su Fig. 5 es un **surrogado supervisado**
`(ρ, drivers) → ReT`, no una política de RL. Nuestro trabajo con KAN estaba archivado
internamente como «demo de surrogado, no una política todavía» — resulta que **esa demo es
exactamente lo que su figura propone**, y KAN aparece por nombre en sus conclusiones. La pieza
que creíamos accesoria es la que él pide.

**Segundo, el veredicto negativo tiene destinatario.** Nuestro programa acumula evidencia de que
cerrar el lazo con control adaptativo **no paga** dentro del sobre nativo de la tesis. Eso no
contradice su paper: su paper es *exploratorio* y **postula** que integrar IA aumentará
«dramáticamente» validez y credibilidad. Nosotros tenemos lo que a él le falta: **la medición**.
Nuestro «cuándo NO entrenar» es la respuesta empírica a su pregunta teórica.

**Tercero, y es lo incómodo: de sus cuatro drivers reproducimos dos.**

| driver | estado medido |
|---|---|
| `Re(RPj)` | reproducido en dirección, pero largo (`d_k` 7,8 R1r / 3,5 R2r) |
| `Re(APj)` | **estructuralmente inalcanzable**: nuestro CTj mínimo es 54 h contra `LT = 48` |
| `Re(DPj)` | **no lo calculamos** |
| `Re(FRj)` | lo tenemos, pero fuera del set de fidelidad |

Entrenar la neurona de su Fig. 5 sobre drivers de los que dos no reproducen y uno no existe
daría un modelo que aprende **nuestro** artefacto, no su cadena.

## 3. Cómo ayuda lo que arreglamos esta semana

| arreglo | por qué importa para su agenda |
|---|---|
| tabla de seis momentos **con artefacto** y convención de ventana igualada en ambos lados | los momentos son sus drivers; sin fuente, cualquier afirmación de fidelidad sobre `Re(APj)`/`Re(RPj)` era prosa |
| equivalencia de la métrica v2 (**0 diferencias en 3.289 filas × 2 estratos**) | el endpoint que reportamos es el mismo que congelamos: resultados de antes y después del 17-jul son comparables |
| clasificación de los 29 campos del simulador | la física nueva de cumplimiento (olas de flete, distribución del retraso, su RNG) queda **dentro** de la prueba de exactitud, no fuera |
| custodia de semillas reparada + ledger de quema versionado | si el surrogado hay que validarlo fuera de muestra, quedan semillas **vírgenes** de verdad |
| promoción no ratificada revertida | la convención de simultaneidad **sigue necesitando su confirmación**, y ahora eso está escrito donde toca |
| fuentes externas por contenido, no por ruta | su tesis, su paper y los tres libros se verifican en cualquier máquina: **requisito del bundle de replicación** |

## 4. Qué correr para avanzar en lo suyo

**Paso 1 — completar los cuatro drivers (barato, sin entrenar nada).**
Emitir `Re(APj)`, `Re(RPj)`, `Re(DPj)` y `Re(FRj)` por configuración, normalizados 0–100 como
él, extendiendo `results/garrido_reproduction/reproduction.json` (ya tiene las 90
configuraciones con `buffer_hours` y `shifts`, que **son sus `ρ`**). Sin esto, la Fig. 5 no
tiene entradas.

**Paso 2 — cerrar la brecha de autotomía, que es la que bloquea un driver entero.**
Nuestro piso de 54 h es **nuestra** calibración, ajustada contra un solo observable. El brazo
`F` (olas de flete) ya existe y da `min CTj = 48,0`, que **haría alcanzable** la rama de
autotomía. Es el experimento con mejor relación valor/coste del repositorio ahora mismo.

**Paso 3 — construir su Fig. 5, tal cual.**
Un surrogado `(ρ, drivers) → ReT` sobre las 90 configuraciones, y la pregunta de activación que
él escribe: *«¿ReT en la configuración `x` es mayor que en la `x−1`?»* — una **clasificación
binaria**, no una política. Comparar backpropagation contra KAN, que es la comparación que él
nombra. Coste: minutos de CPU, cero PPO.

**Paso 4 — lo que solo él puede contestar.** Dos preguntas exactas, ya preparadas:
la convención de simultaneidad `Bt`/`Ut` (bloquea sacar v2 de «provisional»), y si su `RPj`
satura por diseño o por la lógica Simulink que no está documentada.

**Lo que NO hay que hacer todavía:** entrenar una política de RL contra estos drivers. Él lo
menciona tercero y como *simulation-optimization*; nosotros ya tenemos medido que por esa vía
no aparece margen desplegable en el sobre de su tesis.
