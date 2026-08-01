# Estado contra lo que Garrido pidió — las dos fuentes, sin adornos

Dos fuentes, y no piden lo mismo:

* **el paper ICCL 2024** — dos preguntas de investigación;
* **el borrador `v.0_neuralNet-scres`** — un artículo con hipótesis `H1–H4` y **dos secciones de
  resultados vacías**.

## 1. El borrador v.0, sección por sección

| sección | estado |
|---|---|
| Abstract / Keywords | ❌ marcador de posición literal |
| §1 Introducción, §2 Antecedentes | ✅ escritas |
| §3.2 Capa DES | ✅ escrita — 13 operaciones, tablas de la tesis, granularidad horaria |
| §3.3.1–3.3.5 Gymnasium, observación, acción, recompensa | ✅ escritas |
| §4.1 Resultados del DES | ✅ escrita — Cf0 **−4,43 %** contra la ECS de la tesis, y la degradación bajo riesgo |
| **§4.2 Resultados del híbrido** | ❌ **VACÍA** |
| **§4.3 Hallazgos** | ❌ **VACÍA** |
| §5 Limitaciones / contribución teórica | 🟡 esqueleto |

**Todo el hueco del artículo son §4.2 y §4.3.** Y el destino declarado en el encabezado es
*IEEE TAI / Journal of Simulation*, no C&IE — hay que reconciliarlo antes de dar formato.

## 2. Sus cuatro hipótesis, contra lo que tenemos

| hipótesis | qué exige | estado |
|---|---|---|
| **H1 · Learning Effect** | el híbrido **recupera más rápido** que el estático | ❌ **nunca medido**. Existe `rpj_*`, pero es atribución **por pedido**, no tiempo de recuperación del sistema; `system_ttr_*` existe y está **censurado por la derecha** |
| **H2 · Adaptation** | mejora **a lo largo de disrupciones sucesivas** (curva de aprendizaje) | 🟡 **es exactamente lo que la Fase 4 mide** — seis contextos recorridos en orden |
| **H3 · Volatility Reduction** | **menos varianza** entre intensidades de disrupción | ❌ **nunca medido** |
| **H4 · Path Dependency** | `R_t` depende del aprendizaje acumulado `L_{t−1}` | 🟡 **el contraste memoria-vs-reinicio de la Fase 4 ES esto**, y es el único sitio del proyecto donde `L_{t−1}` se aísla |

**Dos de las cuatro las cubre la Fase 4. Las otras dos no las ha medido nadie** — y son baratas:
son **lecturas distintas de las mismas corridas**, no experimentos nuevos.

## 3. Sus dos preguntas de 2024

**Q1 — ¿qué familia de algoritmos imita el aprendizaje de la cadena?**
Parcialmente respondida, y **en contra de su intuición**: su Fig. 5 tal como está dibujada es una
identidad; reformulada como `ρ → SCRES`, un **modelo lineal** explica **0,970** de su propio ReT
en su rejilla y **0,982** en un espacio continuo. Liberar las variables **no** creó la no
linealidad que una red necesitaría. La Fase 4 la reformula donde sí puede haber señal: no como
*predictor* sino como **buscador entre configuraciones**.

**Q2 — ¿cómo se integra en la estructura interna del DES?**
🟡 **la Fase 4 es la respuesta concreta**: un aprendiz colocado entre sus nodos ③ y ⑧ que
**conserva `ρ` entre corridas**, y el número es cuántas corridas ahorra frente al diseño
*un-factor-a-la-vez* de su propia tesis.

## 4. Lo que hemos estado haciendo de más — y hay que decirlo

**Sus `H1–H4` son sobre aprendizaje ENTRE corridas.** Su efecto Alzheimer es que el modelo no
retiene lo aprendido *entre escenarios de simulación*. Su Fig. 5 lo dice literalmente: la
activación es *«¿es la medida de SCRES en la configuración x mayor que en la x−1?»* — un aprendiz
**sobre configuraciones**.

**Nosotros llevamos meses atacando un problema más duro: control adaptativo DENTRO del episodio.**
Es una pregunta legítima y más ambiciosa, pero **no es la que él formuló**, y es donde llevamos
todos los negativos. `H_regime ≈ 1e-4` en superficie, buffers, nodos nuevos, mezcla de regímenes,
contención aguas abajo, observables, y hoy la contención con dientes.

Esto no invalida el trabajo — es lo que produjo el mapa de por qué la cadena es plana, y produjo
el defecto de métrica. Pero **el orden estaba invertido**.

## 5. Dónde ser laxos y dónde hay resultado seguro

**Ser laxos — el programa de headroom/RL.** El borrador **no lo necesita**. Ninguna de sus cuatro
hipótesis exige un controlador dentro del episodio. Sigue abierto (Fase 1A′ corriendo, 1B
preregistrada), pero **deja de ser la ruta crítica del artículo**.

**Resultado seguro y barato — cerrar `H1`–`H4` sobre la Fase 4.** Las cuatro salen del mismo
diseño:

| hipótesis | cómo, sin experimento nuevo |
|---|---|
| H2 | trayectoria del regret **por contexto**, ya almacenada en `per_context` |
| H4 | memoria vs reinicio, ya es el estimando principal |
| **H1** | reevaluar la configuración **final elegida** por cada estrategia con el panel temporal encendido → `system_ttr_mean`, declarando su censura |
| **H3** | **varianza** del ReT de esa configuración final entre intensidades de disrupción |

Es una extensión del runner de la Fase 4, no un programa nuevo.

**Resultado con valor de sorpresa — el defecto de métrica.** `ret_excel` premia **abandonar una
unidad**: el reparto que la maximiza entrega **50 %** de las raciones, el que la minimiza entrega
**80 %** (`docs/RESULTADO_RET_PREMIA_EL_ABANDONO_2026-07-31.md`). Su propio paper de 2024 dice que
estas métricas son *«inadequate or incomplete»* y pide **credibilidad y validez**. Esto lo
**cuantifica**, con la política que lo explota a la vista. Es material de §4.3 y probablemente lo
más citable del artículo.

## 6. El siguiente paso lógico

1. **Aterrizar la Fase 4** y escribir §4.2 con `H2` y `H4`.
2. **Extenderla para `H1` y `H3`** — mismas corridas, dos lecturas más. Cierra las cuatro
   hipótesis del borrador.
3. **§4.3 con el defecto de métrica**, que es la respuesta directa a su preocupación declarada.
4. **Abstract y reconciliación de revista**, que hoy son marcadores de posición.
5. El headroom (1A′, 1B, 1C) sigue **en paralelo**, y entra en §5 como frontera medida — no como
   promesa.
