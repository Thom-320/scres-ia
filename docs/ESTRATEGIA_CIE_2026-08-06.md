# Estrategia C&IE — leído Garrido 2024 página por página, y qué corre esta noche

## 1. Lo que dice el paper, con las páginas

Releí las 15 páginas. Tres cosas cambian o afinan lo que veníamos diciendo.

### La Fig. 2 es más precisa de lo que yo afirmaba (p. 86)

Los nodos que Garrido resalta no son «recolección de datos» y «verificación». Son:

* **③ = `Decision variables, ρ (experiment design)`**
* **⑧ = `Metric of SCRES`**

Y el lazo del diagrama va `run#1,conf#1 → run#2,conf#2 → … run#n,conf#m`, con
`L = {0 + ℓ₁}`, `L = {0 + ℓ₂}`… **La `L` se acumula entre CORRIDAS y CONFIGURACIONES.**

> El puente de IA que pide Garrido va **de la métrica de resiliencia de vuelta al diseño de
> experimentos**. Eso no es control intra-episodio: es **optimización de simulación sobre el
> espacio de configuraciones**. Nuestro bucle externo *es* ese puente, y ahora se puede citar por
> número de nodo.

### La Fig. 5 valida nuestra corrección de fuga (p. 90)

Las dendritas están etiquetadas **`drivers for SCRES dᵢ` / `simulation decision variables`**, y la
activación de ejemplo es *«¿es la medida de SCRES en la configuración x mayor que en la (x−1)?»*.

**La entrada de la neurona son las variables de decisión.** Nuestra versión original alimentaba los
*drivers* del episodio ya corrido —la fuga que retiramos el 31 de julio— y la versión corregida usa
las coordenadas de decisión. **La corrección nos dejó más fieles a su figura, no menos.** Eso entra
al manuscrito como argumento, no como disculpa.

### Los tres candidatos que nombra (p. 91, Conclusiones)

> *«back propagation neural networks, **Kolmogorov-Arnold neural networks**, and
> **simulation-optimization approach as a form of reinforcement learning**»*

Nuestro paquete mide **los tres**. Y el tercero —simulation-optimization como forma de RL— es
exactamente el bucle externo. **No estamos respondiendo una pregunta adyacente: estamos
respondiendo la suya, con su vocabulario.**

## 2. Por qué C&IE es el destino correcto, con evidencia

Garrido 2024 cita **siete** artículos de *Computers & Industrial Engineering*:

| ref | trabajo |
|---|---|
| [8] | Bruckler et al. 2024, **C&IE 192**, 110176 — revisión de métricas de SCRES |
| [9] | Carvalho et al. 2012, C&IE 62(1) |
| [26] | Habibi et al. 2023, C&IE 183 |
| [28] | Ivanov 2019, C&IE 127 |
| [34] | Moosavi & Hosseini 2021, C&IE 160 |
| [38] | Pires Ribeiro & Barbosa-Póvoa 2018, C&IE 115 |
| [41] | Rahman et al. 2022, C&IE 170 |

**La conversación en la que él inscribe su gap ya vive en C&IE.** La [8] es de 2024 y es una
revisión de *métricas* — nuestro hallazgo de que `ret_excel` premia el abandono habla directamente
a ese trabajo.

## 3. Qué tenemos, y qué falta

| # | resultado | estado | ¿confirmatorio? |
|---|---|---|---|
| R0 | bucle interno saturado, 4 contratos | sellado | desarrollo |
| R1 | Alzheimer sobrevive al normalizador honesto | sellado | desarrollo |
| R1b | la fuga medida con superficies gemelas | sellado | desarrollo |
| R2 | no separable · `H_regime` +0,0038 | sellado | desarrollo |
| R3 | la neurona bate a los 7 clásicos sin memoria | sellado | desarrollo |
| R4 | **el ingrediente es la retención**, 4 familias | sellado | desarrollo |
| R5 | KAN≡MLP≡neurona; gana `Δ_efficiency` con 5 params | sellado | desarrollo |
| R6 | +buffers ⇒ `H_regime` ×7,4 | sellado | desarrollo |
| R7 | **sólo el bandido factorizado bate su réplica marginal** | sellado | **en confirmación** |

**Todo es desarrollo.** Ése es el hueco: un manuscrito donde cada número dice «observamos» y
ninguno dice «predijimos y lo probamos».

## 4. Qué corre esta noche, y qué desbloquea

### Local — la confirmación prospectiva de R7

Bloque virgen `8.200.001–060`, 6 workers, ~360 rebanadas. **Desbloquea la única afirmación
confirmatoria del paper**: convierte R7 de réplica de desarrollo en resultado prospectivo con
estimando congelado y regla de éxito escrita de antemano. Es el resultado más novedoso que tenemos
—*lo que transfiere al cambiar el espacio de diseño es una factorización, no un aproximador*— y es
el que un revisor no ha visto antes.

### VPS — el bake-off de arquitecturas (KAN vs MLP vs DMLPA)

3 arquitecturas × 5 semillas × 60k pasos sobre Track B, **presupuesto igualado a 200k parámetros**
(KAN 204.816 · MLP 199.215 · DMLPA 187.404, todas dentro del 6,3 %). VPS a 31 pasos/s → **8,1 h**.
Manifiesto de módulos **verificado idéntico al local antes de lanzar** — la comprobación que faltó
cuando la rebanada VPS de H3′ acabó en `HOLD_SOURCE_AUDIT`.

**Desbloquea §4.5 y el pedido explícito de KAN de Garrido**, y lo hace **sin depender de los
tiempos de David**: si él sólo alcanza a correr su brazo, la comparación igual está completa. Y al
correr las tres en **una sola máquina**, `Δ_efficiency` es válido — cosa que no sería si
repartiéramos brazos entre Kaggle y otra parte.

### Kaggle — preparado, no lanzado

Los dos kernels (`MLP`, `DMLPA`) están escritos con GPU **apagada a propósito** y esperan
credenciales (`~/.kaggle/kaggle.json` no existe en esta máquina). **Con el VPS corriendo los tres
brazos, Kaggle deja de ser necesario para nosotros** y queda como la vía de David.

## 5. La prioridad, en una frase

**La confirmación de R7 es lo único que cambia la categoría del paper**; el bake-off es lo único
que cierra la pregunta de KAN que Garrido nombró. Los dos están corriendo, en pools distintos, sin
competir por CPU, y ninguno abre una semilla que no esté autorizada.

Lo que **no** corre esta noche, a propósito: nada que necesite firma nueva, nada sobre el bloque
`8.100.001–060` en cuarentena, y ningún entrenamiento neuronal fuera del bake-off de desarrollo.
