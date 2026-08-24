# CLAUDE — Revisión estadística adversarial sobre el bundle común

**Fecha:** 2026-08-24 · **Rol:** revisor estadístico adversarial de SCRES-IA
**Alcance de lectura (único):** `context_texts/` (19 TXT) y `context_reports/` (7 MD).
**Restricciones respetadas:** no se leyó ninguna otra ruta; no se modificó el repositorio; no se reabre O, O-R ni Q.
**Naturaleza:** 10 propuestas prospectivas. Ninguna re-adjudica un contrato histórico.

---

## 0. Cómo leer este documento

Cada idea separa explícitamente:

- **HECHO (TXT)** — afirmación verificable citando archivo y sección/ecuación del bundle leído.
- **HECHO (informe)** — afirmación que sólo consta en los MD de `context_reports/`; es reporte de terceros, no verificable contra los PDFs.
- **PROPUESTA** — diseño mío, no respaldado todavía por evidencia.

En §11 listo lo que **no** pude verificar y dos defectos de catalogación del propio bundle.

---

## 1. Desesgar Gate 0: el estimando `G_PI` tal como está escrito es optimista por construcción

- **Mecanismo.** **HECHO (informe):** `SINTESIS_FIXPACK_Y_PRIORIDAD.md:38` define el gate sin entrenamiento como `G_PI = mean_t[max calendar] − max_c[mean ReT]`, con la frontera completa 4^8 evaluada sobre 128 tapes/celda. **PROPUESTA (crítica):** el primer término es un `E_t[max_k X_kt]`, es decir, el máximo se toma *dentro* de cada tape y luego se promedia. Por desigualdad de Jensen `E[max_k X_k] ≥ max_k E[X_k]`, y con k = 4^8 = 65 536 calendarios el sesgo hacia arriba no es despreciable: es exactamente el sesgo de reusar los datos de búsqueda como datos de evaluación. El segundo término (`max_c mean`) no tiene ese sesgo, así que la diferencia **está sesgada en la dirección que hace pasar el gate**. **HECHO (TXT):** `a10-hong2021-fem-review-rs.txt:2713` — Eckman & Henderson (2018b) muestran que reusar las observaciones de la fase de búsqueda rompe las garantías del procedimiento de selección; `a10:...§7` lo lista como problema abierto. **PROPUESTA (fix):** partición de tapes en dos mitades disjuntas: seleccionar `argmax_k` sobre la mitad A, evaluar el `k` seleccionado sobre la mitad B, y reportar además el delta de sesgo `= G_PI(ingenuo) − G_PI(split)`.
- **Motivación:** `a10-hong2021-fem-review-rs.txt` §7 (Problema sobre reuso de datos de búsqueda) + `SINTESIS_FIXPACK_Y_PRIORIDAD.md:38`.
- **Estimando:** `G_PI^split = E_B[ ReT(k*(A)) ] − max_c E[ReT_c]`, con `k*(A)` congelado antes de mirar B. Endpoint primario: UCB95(`G_PI^split`) por celda.
- **Gate de falsación:** si UCB95(`G_PI^split`) < 0.01 en alguna celda → abandono físico, exactamente como el diseño original preveía, pero ahora sin la puerta trasera del sesgo. Si `G_PI(ingenuo) − G_PI(split) ≥ 0.01`, queda demostrado que el gate original habría autorizado 42 CPU-h por artefacto de selección.
- **Costo CPU:** **nulo adicional**. Es re-partición del mismo barrido de ~6 h. El delta de sesgo es un subproducto gratuito.
- **Riesgo de tocar contratos históricos:** **nulo formal, alto en narrativa prospectiva.** No toca O/O-R/Q. Pero endurece un gate aún no ejecutado y puede cerrar la lane antes de gastar.

---

## 2. Descomposición Kitagawa/Oaxaca de `ReT`: separar efecto intra-régimen de efecto de composición

- **Mecanismo.** **HECHO (TXT):** `WRAP_Theses_Garrido_Rios_2017.txt:3121-3134` (Eq 5.5) define `ReT` como una **función condicional a trozos**, no como una suma ponderada: para cada orden *j* devuelve una de cuatro ramas según el régimen en que caiga la orden — autotomía (`AP_j`), recuperación (`RP_j`), no-recuperación (`DP_j − RP_j`) o no-disrupción (`N-DP_j`, vía `FR_t`). **HECHO (TXT):** las cuatro ramas tienen escalas y sentidos distintos (Eq 5.1 creciente en `AP_j/LT`; Eq 5.2 recíproca en `RP_j`; Eq 5.3 con peso `Re^min`; Eq 5.4 `1 − (B_t+U_t)/D_t`). **PROPUESTA:** por tanto `mean_j ReT_j` es una **mezcla** cuyos pesos de mezcla `w_r` (fracción de órdenes en cada régimen) son ellos mismos función de la política. Un controlador puede mover el agregado sin mejorar ninguna rama, sólo redistribuyendo órdenes entre regímenes. Descomponer:
  `Δ mean(ReT) = Σ_r w̄_r · Δμ_r  (efecto intra-régimen)  +  Σ_r μ̄_r · Δw_r  (efecto composición)`
  con `w̄_r, μ̄_r` promedios de los dos brazos (descomposición simétrica, sin brazo de referencia arbitrario).
- **Motivación:** `WRAP_Theses_Garrido_Rios_2017.txt` §5.6.2–5.6.3, Eq 5.1–5.5. Corroboración indirecta: **HECHO (informe)** `SINTESIS_FIXPACK_Y_PRIORIDAD.md:17` reporta que el learner "compra fill agregado desbalanceando el producto débil" — el patrón exacto que esta descomposición mide.
- **Estimando:** par (`efecto intra-régimen`, `efecto composición`) con IC95 pareado por semilla (CRN), reportados por separado y **nunca sumados en un único número** como endpoint.
- **Gate de falsación:** si `|efecto composición| ≥ 0.5 · |Δ mean(ReT)|` en cualquier celda, se declara que el endpoint agregado **no está identificado** como mejora de resiliencia y el claim debe reformularse sobre la componente intra-régimen. Falsación de mi propia hipótesis: si el efecto de composición es <10 % en las tres celdas, esta crítica queda cerrada y `mean(ReT)` se rehabilita.
- **Costo CPU:** **nulo.** Re-análisis de trazas ya existentes, siempre que las trazas registren el régimen por orden. Si no lo registran, coste = una re-evaluación de políticas congeladas (bajo).
- **Riesgo de tocar contratos históricos:** **nulo formal, medio-alto interpretativo.** Es re-análisis descriptivo; no cambia adjudicaciones. Pero puede mostrar que parte de lo ya reportado (en cualquier contrato) es recomposición y no mejora. No debe usarse para re-adjudicar Q.

---

## 3. Auditoría de zona muerta: la rama `Re(DP_j − RP_j)` vale cero por construcción

- **Mecanismo.** **HECHO (TXT):** `WRAP_Theses_Garrido_Rios_2017.txt:3043-3049` dice literalmente que este sub-indicador "is included within the measurement of resilience in order to maintain the consistency of the model, **though the value of resilience is zero in all cases** due to the weighting parameter assigned (`Re^min`)". **PROPUESTA:** eso implica que durante todo el período de no-recuperación —el período de máxima vulnerabilidad, y precisamente donde vive el `worst_product_fill`— la métrica es **idénticamente insensible al control**. Dos políticas que difieran enormemente en cómo gestionan la vulnerabilidad profunda obtienen el mismo `ReT` en esa rama. Es un caso de medición ciega, no de ruido.
- **Motivación:** `WRAP_Theses_Garrido_Rios_2017.txt` §5.6.2, Eq 5.3 y texto circundante.
- **Estimando:** `f_dead` = fracción de pasos-orden que caen en la rama `DP_j − RP_j` bajo cada brazo; y, como endpoint secundario prerregistrado, una métrica de cola definida *dentro* de esa rama (p. ej. cuantil del backlog del producto débil condicionado a estar en la zona muerta).
- **Gate de falsación:** si `f_dead < 0.02` en las tres celdas, la ceguera es irrelevante y esta idea se descarta. Si `f_dead ≥ 0.10` y los dos brazos difieren en la métrica intra-zona con LCB95 > 0, queda demostrado que `ReT` no puede ser el único endpoint de seguridad.
- **Costo CPU:** **nulo–bajo** (instrumentación de conteo por régimen; sin re-entrenamiento).
- **Riesgo de tocar contratos históricos:** **nulo formal, medio.** Añade un endpoint secundario nuevo; no reemplaza ni reinterpreta el primario de ningún contrato sellado. El riesgo real es de gobernanza: la tentación de promover el secundario a primario post-hoc. Prohibirlo por escrito en el preregistro.

---

## 4. Prerregistrar la escala de agregación de `Re(RP_j)`: es un recíproco, no una media bien comportada

- **Mecanismo.** **HECHO (TXT):** `WRAP_Theses_Garrido_Rios_2017.txt:3039-3041`, Eq 5.2: `Re(RP_j) = Re · (1/RP_j)`, justificada por Blackhurst et al. (2011) ("resiliency and recovery time should be inversely related"). **PROPUESTA:** un recíproco de una variable positiva con masa cerca de cero produce una distribución de cola pesada por la derecha; la media muestral está dominada por unas pocas recuperaciones muy rápidas y su varianza puede ser enorme o no existir. Consecuencias concretas: (i) los IC normales/bootstrap sobre la media convergen mal; (ii) la constante de Rinott y los procedimientos IZ suponen normalidad aproximada —**HECHO (TXT)** `a10-hong2021-fem-review-rs.txt` §2 describe las formulaciones fixed-precision sobre supuestos de normalidad—; (iii) cualquier estimación de `σ_seed` calculada sobre esta escala está inflada por un puñado de episodios.
- **Motivación:** `WRAP_Theses_Garrido_Rios_2017.txt` Eq 5.2 + `a10-hong2021-fem-review-rs.txt` §2–§3.
- **Estimando:** comparar tres escalas de agregación **fijadas antes de abrir semillas**: media aritmética de `1/RP`, media armónica (equivalente a `1/mean(RP)`), y mediana. Reportar el par (`skewness`, `kurtosis`) empírico y la fracción de la varianza total aportada por el 5 % superior de episodios.
- **Gate de falsación:** si el 5 % superior de episodios aporta ≥50 % de la varianza de `Re(RP)`, la media aritmética queda vetada como agregador para todo contrato **nuevo**. Si aporta <15 %, esta objeción se cierra.
- **Costo CPU:** **nulo.** Estadística descriptiva sobre trazas existentes.
- **Riesgo de tocar contratos históricos:** **bajo-medio.** No cambia ningún resultado adjudicado; sí implica que los IC históricos calculados sobre esta escala son optimistas o pesimistas en magnitud desconocida. Debe declararse como limitación, no como invalidación.

---

## 5. Potencial jerárquico HPRS (safety ⊃ target ⊃ comfort) en vez de subobjetivos en secuencia

- **Mecanismo.** **HECHO (TXT):** `a14-hprs2024-frontiers.txt` (abstract y §1) define la tarea como un **conjunto parcialmente ordenado** de requisitos safety / target / comfort, y construye el reward de modo que "the target reward is a **function of** the safety reward and the comfort reward is a function of the safety and target rewards"; la formulación es potential-based y los autores prueban que preserva la optimalidad de la política. **HECHO (TXT):** el mismo abstract reporta que HPRS "benefits from comfort requirements when aligned with the target and safety and **ignores them when in conflict**". **PROPUESTA:** ese es exactamente el modo de fallo del contrato Q según los informes: se compró target (fill agregado) a costa de safety (cola del producto débil). Un potencial aditivo o secuencial —como el de subobjetivos ordenados— **no** impide esa sustitución: sólo cambia el orden en que se acumulan los términos. HPRS sí la impide estructuralmente, porque el término target está multiplicado/condicionado por el nivel de satisfacción del safety. Mapeo propuesto: safety = `worst_product_fill` sobre su margen; target = fill agregado / `ReT`; comfort = coste.
- **Motivación:** `a14-hprs2024-frontiers.txt` (abstract, §1, RQ2–RQ3). Contraste deliberado con `a1-okudo2021-ieee-access-subgoal.txt` (subgoals como potencial, pero **sin** orden lexicográfico entre clases de requisito) y con `a3-mueller2025-arxiv-pbrs-effectiveness.txt` (que trata escala/offset del potencial, no su estructura jerárquica).
- **Estimando:** `Δ_N` (efficacy) y cola `worst_product_fill` (safety) **con margen de no-inferioridad prospectivo**, comparando 3 brazos: (a) sin shaping, (b) potencial aditivo plano, (c) potencial HPRS jerárquico. Potencial terminal fijado en cero para preservar invarianza bajo truncación.
- **Gate de falsación:** promoción sólo si HPRS mejora la cola con LCB95 > 0 **y** no degrada `Δ_N` por debajo del margen. Falsación fuerte: si el brazo (c) mejora la cola exactamente lo mismo que (b), la jerarquía no es el mecanismo y el claim se retira. Control obligatorio: verificar empíricamente invarianza (ranking de políticas congeladas idéntico con y sin shaping bajo la misma truncación).
- **Costo CPU:** **medio.** 3 brazos × 3 celdas × semillas, con checkpoints; del orden de 80–150 CPU-h si se reusa el presupuesto de entrenamiento del smoke B0/B1. No requiere entorno nuevo.
- **Riesgo de tocar contratos históricos:** **medio.** Cambia el reward ⇒ contrato nuevo por la línea roja (`CONTEXTO_COMUN:65`). Mitigación: semillas vírgenes, SHA nuevo, mismo evaluador, y prohibición explícita de comparar contra los números de Q.

---

## 6. Certificado de infactibilidad para el problema con restricción de cola (EGO restringido)

- **Mecanismo.** **HECHO (TXT):** el archivo `a11-luo2024-scis-survey-mbrl.txt` **no** contiene una encuesta de MBRL: su contenido es Xu, Jiang, Svetozarevic & Jones, *CONFIG: Constrained Efficient Global Optimization of Expensive Black-box Functions* (arXiv 2211.00162v4). Ver §11 sobre este desajuste. **HECHO (TXT):** CONFIG resuelve `max f(x) s.a. g_i(x) ≥ 0` con `f` y `g` cajas negras caras, da cotas sublineales de regret acumulado **y** de violación acumulada, y —punto clave— "naturally provides a scheme to **declare infeasibility** when the original black-box optimization problem is infeasible", capacidad que los autores señalan que ningún método previo (incl. CEI y primal-dual) posee. **PROPUESTA:** el estado del programa Q descrito en los informes es literalmente un problema con restricción de caja negra: maximizar `Δ_N` sujeto a `worst_product_fill ≥ −δ`. Hoy la conclusión "efficacy pasó, safety falló" es un veredicto narrativo. Con EGO restringido se convierte en una afirmación con estatus formal: *no existe* configuración en el espacio de hiperparámetros/arquitectura prerregistrado que satisfaga la restricción de cola, con confianza declarada. Eso es un resultado negativo **publicable**, no un fracaso.
- **Motivación:** `a11-luo2024-scis-survey-mbrl.txt` (= Xu et al., CONFIG), abstract y §1 ("scheme to declare infeasibility"). Complemento sobre coste de tuning: `a7-gijsbrechts2022-msom-can-deep-rl.txt` §Results ("the initial tuning was computationally- and time-demanding").
- **Estimando:** región factible estimada `{x : LCB(g(x)) ≥ 0}` sobre el espacio prerregistrado (γ, escala de potencial, tamaño LSTM, timesteps), más el certificado de infactibilidad con su nivel de confianza.
- **Gate de falsación:** si CONFIG declara infactibilidad, se cierra la lane de hiperparámetros por escrito y se pasa a "física, no artefacto". Si encuentra un `x` factible, ese `x` **no es el resultado**: debe re-evaluarse con presupuesto fresco e independiente (winner's-curse cleanup, `a10:2713`) antes de reportarlo.
- **Costo CPU:** **medio.** El propio punto de CONFIG es la eficiencia muestral; ~20–40 evaluaciones de política (cada una = un entrenamiento corto + evaluación CRN). Del orden de 60–120 CPU-h, sustancialmente menos que un grid.
- **Riesgo de tocar contratos históricos:** **bajo formal, alto retórico.** No re-adjudica nada. Pero un certificado de infactibilidad es un compromiso fuerte: hay que aceptarlo por escrito antes de correrlo, igual que la advertencia ya registrada en `SINTESIS:41` sobre déficits verdaderos que ningún N puede voltear.

---

## 7. Auditar el episodio de un solo paso de Ding 2026 antes de comprar su blueprint

- **Mecanismo.** **HECHO (TXT):** `1-s2.0-S0925527326000861-main.txt:1247-1248` afirma que "determining the reconfiguration behavior of each element can be considered a complete episode, **with each episode consisting of a single-step interaction**", y Eq (54) define la recompensa acumulada por episodio como la suma de recompensas de los tres agentes "during the action taken in that **single step**". **PROPUESTA:** si el episodio es de un paso, el problema resuelto por MAPPO en Ding es esencialmente un **bandit contextual multi-agente**, no un problema de asignación de crédito temporal. La superioridad reportada de MAPPO sobre MADDPG/QMIX no es entonces evidencia de que CTDE resuelva el aspecto secuencial de la reconfiguración. Adoptar el blueprint sin corregir esto importaría la estructura de un solo paso a la lane topológica de SCRES-IA, donde `filling/repairing/recruiting` sí tienen dinámica multi-período (recursos no fungibles, tiempos de reparación).
- **HECHO (TXT) adicional, mismo archivo:** Eq (55) define la resiliencia como `R̂ = ND·DC·CC·HC·MC / AD` — un **producto de cinco índices topológicos dividido por la distancia media**, sin acotación declarada ni análisis dimensional; y los resultados comparativos de §4.2.1 se reportan como números puntuales (p. ej. MAPPO/MADDPG ≈ 50 vs QMIX 46; ≈160 vs 155 en gran escala) **sin intervalos de confianza ni número de semillas**. Contrasta con `b4-guzman2026-cie-circular.txt` (abstract), que sí declara "matched seeds, fixed horizons, and 95% confidence intervals" como su contribución metodológica principal.
- **Motivación:** `1-s2.0-S0925527326000861-main.txt` §4.1.3 (Eq 54–56) y §4.2.1; contraste con `b4-guzman2026-cie-circular.txt` abstract.
- **Estimando:** en el entorno topológico mínimo, `Δ_multi-step` = rendimiento del learner con episodio de H pasos menos el del **mismo** learner restringido a decisión miope de un paso, con CRN pareado. Métrica de resiliencia acotada en [0,1] y con unidades verificadas, prerregistrada; `R̂` de Ding se reporta sólo como métrica secundaria de comparabilidad con la literatura.
- **Gate de falsación:** si `LCB95(Δ_multi-step) ≤ 0`, la lane topológica **no** está justificada como problema secuencial y MARL no es el lever; se cierra antes de escalar a 500k–1M pasos. Este gate es más barato y más informativo que el screen de headroom sólo.
- **Costo CPU:** **bajo-medio.** El brazo miope es barato por construcción; ~15–30 CPU-h para el par en un entorno de 8–13 nodos.
- **Riesgo de tocar contratos históricos:** **nulo.** Es una lane nueva, prospectiva, con entorno nuevo. Sólo restringe cuánto se puede citar a Ding como respaldo.

---

## 8. El índice `R` de Garrido 2024 (IJPR) no es comparable entre conjuntos de comparadores

- **Mecanismo.** **HECHO (TXT):** `garrido2024_factory_resilience.txt:701-704` da la Eq (6): `R = 1/(1+exp(−(0.024·Lnζ − 0.026·Lnε + 0.04·Lnφ − 0.06·Lnτ − 0.1771·Lnκ̇)))`. **HECHO (TXT):** `:725-731` explica que esos exponentes se obtuvieron identificando "the **highest values** of ζ, ε, φ, τ y κ̇(Sij) after 10,000 simulation runs" y forzando cada argumento a 1/5 (ej.: `ζ^max ≈ 3612`, `a·Ln3612 = 0.20 ⇒ a = 0.024`). **HECHO (TXT):** `:766-769` muestra que en la forma desarrollada el término de coste entra como `−n·Ln( 7·κ̂(S12) / Σ_ij κ̂(S_ij) )`, es decir, **normalizado por el coste medio del conjunto de las 7 substrategias**. **PROPUESTA:** de ahí se siguen dos defectos de identificación: (i) los exponentes son función de **máximos muestrales**, que crecen con el número de corridas y no tienen error propagado; (ii) el valor de `R` de una estrategia depende de **qué otras estrategias estén en el conjunto**: añadir o quitar un comparador cambia el denominador y por tanto `R` de todas. **HECHO (TXT):** el ranking Eq (9) (`:855-857`, `R(S12) ≻ R(S11) ≻ R(S32) ≻ R(S22) ≻ R(S31) ≻ R(S13) ≻ R(S21)`) se deriva de boxplots con los siete parámetros de coste iguales a 1 y una sensibilidad con variaciones aleatorias en [1,2] sobre `ci, cp, cb` — sin IC y sin garantía de tipo PCS/PGS.
- **Motivación:** `garrido2024_factory_resilience.txt` §3.4–§5, Eq (1), (6), (7), (9).
- **Estimando:** (a) *leave-one-strategy-out*: recalcular `R` y el ranking Eq (9) eliminando cada substrategia una a una; (b) sensibilidad de los exponentes al número de corridas (`ζ^max` con 1k / 5k / 10k runs); (c) re-derivación del ranking como problema R&S con zona de indiferencia gerencial y PGS, no como orden de medianas.
- **Gate de falsación:** si el orden de Eq (9) cambia al eliminar cualquier substrategia, o si los exponentes se mueven >10 % entre 5k y 10k corridas, entonces `R` **no puede** usarse como endpoint de contrato en SCRES-IA y los comparadores APP dinámicos deben evaluarse sobre un endpoint libre de conjunto (p. ej. coste y fill por separado, sin escalarizar). Si el orden es invariante en las 7 eliminaciones, la objeción queda cerrada y `R` se rehabilita como métrica secundaria.
- **Costo CPU:** **muy bajo** (~1–3 CPU-h): es recomputar una fórmula cerrada sobre salidas ya simuladas, más una re-simulación reducida si no se conservan las salidas.
- **Riesgo de tocar contratos históricos:** **nulo formal.** No afecta O/O-R/Q. Afecta prospectivamente a la lane de comparadores APP: si `R` es dependiente del conjunto, cualquier claim futuro de "el learner supera a las APP en resiliencia" medido con `R` sería no identificable.

---

## 9. Screening de factores con control de potencia (CSB) antes de la ablación completa de supresores

- **Mecanismo.** **HECHO (informe):** el fix-pack enumera siete supresores candidatos (`SINTESIS:23-33`: obs pobre, reward esparsa, γ/horizonte, pasos, LSTM, comparador, potencia N). **HECHO (TXT):** `a12-fan2025-jorsc-large-scale-so.txt:631-641` describe el *variable/factor screening* como fase preliminar que "identify the effective variables ... and statistically eliminate ineffective ones", y señala que el procedimiento CSB (controlled sequential bisection) extiende el SB original "to control **both type I error and power** for screening". **PROPUESTA:** hoy el plan es ejecutar B0 vs B1 con **tres cambios simultáneos** (PBRS-Q21 + γ=1 + LSTM 128; `SINTESIS:39`). Ese diseño es confundido: si B1 gana, no se sabe cuál factor lo produjo; si pierde, no se sabe cuál lo hundió. Un screening secuencial bisectivo con potencia controlada resuelve la atribución **antes** de gastar en el factorial completo y es más barato que 2^3.
- **Motivación:** `a12-fan2025-jorsc-large-scale-so.txt` §sobre dimension reduction / factor screening; complemento de asignación de presupuesto en `a10-hong2021-fem-review-rs.txt` §5.
- **Estimando:** para cada factor `k`, el efecto principal `θ_k` con la garantía dual del CSB: `P(declarar activo | θ_k = 0) ≤ α` y `P(declarar inactivo | θ_k ≥ Δ) ≤ β`, con `Δ` = SESOI gerencial (el mismo δ = 0.01 ya en uso, fijado antes de mirar datos).
- **Gate de falsación:** los factores declarados inactivos **quedan prohibidos** para el resto del programa y no pueden reintroducirse post-hoc por inspección de resultados. Si los tres factores de B1 se declaran inactivos, la conclusión "artefacto" queda descartada por vía barata y el veredicto es "física".
- **Costo CPU:** **negativo respecto al plan actual** en el escenario esperado (el screening bisectivo evita el factorial `2^3` y concentra réplicas donde hay señal); estimo 15–35 CPU-h, frente a las 42 h del smoke pareado B0/B1 más cualquier desambiguación posterior. **Advertencia honesta:** si todos los factores resultan activos, el screening no ahorra nada y añade ~10 CPU-h.
- **Riesgo de tocar contratos históricos:** **bajo.** Cambia sólo el diseño experimental y su trazabilidad. No modifica obs/reward/γ/arquitectura por sí mismo, así que no dispara la línea roja hasta que se ejecuten los brazos.

---

## 10. Descuento dependiente de la transición como palanca de aversión al riesgo (primero tabular)

- **Mecanismo.** **HECHO (TXT):** `a5-sharma2021-symmetry-discount.txt` (abstract y §1) introduce un factor de descuento función de `(s, a, s')` en Q-learning y SARSA, prueba convergencia por aproximación estocástica **para espacios de estados y acciones finitos**, y demuestra que el descuento asimétrico "provides better control over the RL agents to learn **risk-averse** or risk-taking policy, as demonstrated in a Cliff Walking experiment". **PROPUESTA:** esto ofrece una palanca de cola **ortogonal** al reward shaping: en vez de añadir términos al reward (que abre la puerta al reward hacking y exige verificar invarianza), se descuenta **menos** las transiciones que aumentan el backlog del producto débil, haciendo que sus consecuencias futuras pesen más en el valor. Es un cambio en `γ(s,a,s')`, no en `R`.
- **Limitación declarada por adelantado (no la esconde este informe):** **HECHO (TXT):** las garantías de Sharma et al. son para tabular finito con Q-learning/SARSA. `a5` **no** establece nada sobre PPO con función de aproximación ni sobre POMDPs. Trasplantarlo directamente a PPO recurrente sería injustificado.
- **Motivación:** `a5-sharma2021-symmetry-discount.txt` §1, §3–§5. Ninguno de los tres informes de ideas previos (`IDEAS_CODEX.md`, `IDEAS_OPENCODE.md`, `IDEAS_CLAUDE_FULL.md`) cita este archivo. Complemento sobre horizontes largos y γ→1: `a4-wang2023-arxiv-freezing-slow.txt` (abstract: γ cercano a 1 debilita la contracción de Bellman; congelar estados lentos acota el regret con mucho menos cómputo; "simply omitting slow states is often a poor heuristic").
- **Estimando:** en un micro-MFSC tabular (dos productos, inventario/backlog/belief discretizados), `CVaR_10` del fill del producto débil bajo `γ` constante vs `γ(s,a,s')` asimétrico, con el óptimo exacto por iteración de valor como referencia. Endpoint secundario: coste en `Δ_N` medio de la aversión inducida.
- **Gate de falsación:** promoción a la escala DES sólo si, en el micro tabular, el descuento asimétrico mejora `CVaR_10` con LCB95 > 0 **y** el sacrificio de media es ≤ δ. Falsación: si la mejora de cola exige sacrificar más de δ en media, el mecanismo es sólo un reponderador media↔cola y no aporta sobre lo que ya hace una restricción explícita.
- **Costo CPU:** **bajo en la fase tabular** (~10–25 CPU-h, dominado por el barrido de discretización); la fase DES no se autoriza sin pasar el gate y costaría 100+ CPU-h.
- **Riesgo de tocar contratos históricos:** **nulo formal, medio prospectivo.** Cambiar γ a una función de la transición es un contrato nuevo bajo la línea roja (`CONTEXTO_COMUN:65`), con semillas vírgenes y SHA nuevo. Riesgo técnico adicional: sin la garantía teórica en el caso deep, un resultado positivo en DES sería empírico y no certificado — debe reportarse así.

---

## 11. Lo que NO pude verificar, y dos defectos del propio bundle

### 11.1 Desajustes de catalogación detectados (verificables)

1. **`context_texts/a11-luo2024-scis-survey-mbrl.txt` no corresponde a su nombre.** El contenido es Xu, Jiang, Svetozarevic & Jones, *CONFIG: Constrained Efficient Global Optimization of Expensive Black-box Functions*, arXiv 2211.00162v4 (6 Feb 2025), ICML-style, sobre optimización bayesiana con restricciones. **No** es una encuesta de model-based RL de Luo 2024 en SCIS. Consecuencia: toda afirmación del programa que se apoye en "a11 = survey MBRL" carece de respaldo en este bundle. La idea #6 de este informe se apoya en el **contenido real** del archivo, no en su nombre.
2. **`LECTURA_OBLIGATORIA_HARNESSES.md:39` cita `a2-mueller2025-arxiv-pbrs-effectiveness.pdf`.** En `context_texts/`, `a2` es Forbes et al. 2024, *Potential-Based Reward Shaping For Intrinsic Motivation* (AAMAS 2024), y Müller & Kudenko 2025 es `a3`. `IDEAS_CODEX.md` e `IDEAS_OPENCODE.md` usan la asignación correcta (a3 = Müller). El error está en el índice de lectura obligatoria.

### 11.2 Afirmaciones del programa que ningún TXT permite verificar

Las siguientes constan **sólo** en `context_reports/` y no son comprobables contra los 19 textos; las trato como HECHO (informe), nunca como evidencia bibliográfica:

- Cifras de Program Q: `worst_product_fill` −0.0104 / −0.0157 / −0.0045 y sus *t*; `max_backlog_age` +123; `service_loss_auc` +908k (`SINTESIS:11-18`).
- `σ_seed ≈ 0.032`; `H_PI = 0.1515`; potencial Q21 con CV-R² 0.44–0.50; ganadores `min_cost_flow__2` y `max_pressure__0` (`SINTESIS:27,31,32`; `IDEAS_OPENCODE:56`).
- Estado de la suite (2260/2256 passed, 38 failed) y el desajuste de hash CSSU `9cb65c7a` vs `f3fe61b1` (`CONTEXTO_COMUN:24-25`).
- Contenido de `PROMISING_LANES_REGISTRY.md`, `SUITE_CERTIFICACION.md`, `SECOND_OPINION_*.md`, y todas las referencias a líneas de código (`adjudicate_program_q.py:21-30`, `env_experimental_shifts.py:*`, `benchmark_control_reward.py:*`): fuera del alcance de lectura autorizado.
- La fórmula exacta de `G_PI` (`SINTESIS:38`): mi crítica en la idea #1 es válida **para la fórmula tal como está escrita en ese informe**; si la implementación real difiere, la crítica debe reevaluarse antes de actuar.

### 11.3 Papers citados por el programa sin TXT en este bundle

`SINTESIS:47` y `LECTURA_OBLIGATORIA:32-35,45` atribuyen contenido específico a Kim 2023 IISE, Fan 2023 JIPR, Akashi 2023, Burtea & Tsay 2024, Mousa 2024, Kotecha 2025 y Cheng 2023 EJOR. **Ninguno tiene TXT en `context_texts/`.** Nada de lo afirmado sobre ellos —incluida la frase "ahora science-backed" para la lane topológica— puede confirmarse desde este bundle. La lane topológica queda respaldada aquí únicamente por Ding 2026 (con la reserva grave de la idea #7), Guzmán 2026 y Kong 2026.

### 11.4 Reserva sobre Kong 2026 y Guzmán 2026

- **HECHO (TXT):** `b10-kong2026-eai-transformer.txt` (abstract) reporta un "resilience score of 0.892", WMAPE 14.38 %, AUC de riesgo 0.941 y mejoras de 19.3 % / 5.9 % frente a Transformer-MAPPO, **sin intervalos de confianza, sin semillas declaradas y con un score escalarizado**. Es la misma patología que la Eq (55) de Ding y que el índice `R` de Garrido (idea #8): resiliencia comprimida en un escalar cuyos pesos no son identificables.
- **HECHO (TXT):** `b4-guzman2026-cie-circular.txt` (abstract) sí declara semillas emparejadas, horizontes fijos e IC95 como su contribución metodológica, y un protocolo Value-of-Data. Es el único de los tres que ofrece un protocolo citable; su claim de transferibilidad "across four archetypal operating regimes **without retuning**" es, sin embargo, un claim de superficie de decisión y no se acompaña en el abstract de una garantía condicional del tipo `min_x PCS(x)` (**HECHO (TXT):** `a10-hong2021-fem-review-rs.txt:2498-2510` define `PCS(x)`, `E[PCS(X)]` y `min_x PCS(x)` como los tres objetivos posibles en R&S con covariables).

---

## 12. Resumen del ranking

| # | Idea | Costo CPU | Riesgo contratos | Por qué en esa posición |
|---|---|---|---|---|
| 1 | Desesgar Gate 0 (split-tape) | nulo | nulo formal | Evita autorizar 42 CPU-h por sesgo de selección; gratis |
| 2 | Descomposición Kitagawa de `ReT` | nulo | nulo formal / medio-alto interpretativo | Explica el fallo de Q con datos ya existentes |
| 3 | Auditoría de zona muerta `Re(DP−RP)` | nulo–bajo | nulo formal / medio gobernanza | Ceguera de medición documentada en la propia tesis |
| 4 | Escala de agregación de `Re(RP)` | nulo | bajo-medio | Recíproco de cola pesada; afecta todos los IC |
| 5 | Potencial jerárquico HPRS | medio (80–150 h) | medio (contrato nuevo) | Ataca estructuralmente la sustitución media↔cola |
| 6 | Certificado de infactibilidad (EGO restringido) | medio (60–120 h) | bajo formal / alto retórico | Convierte "safety falló" en resultado negativo certificado |
| 7 | Auditoría del episodio de un paso de Ding | bajo-medio (15–30 h) | nulo | Gate barato antes de financiar la lane topológica |
| 8 | Dependencia del conjunto en el índice `R` | muy bajo (1–3 h) | nulo formal | Barato, pero afecta una lane secundaria |
| 9 | Screening CSB con potencia controlada | 15–35 h (ahorra si hay factores inertes) | bajo | Desconfunde B0/B1 antes de gastar |
| 10 | Descuento dependiente de la transición | 10–25 h tabular | nulo formal / medio prospectivo | Mecanismo prometedor pero sin garantía en el caso deep |

**Criterio del ranking:** (tamaño del claim × falsabilidad barata) / (riesgo de gobernanza). Las posiciones 1–4 son auditorías de coste ~nulo que pueden ejecutarse antes de cualquier entrenamiento; 5–7 son experimentos nuevos con gate previo; 8–9 son higiene de diseño; 10 es la más especulativa y la única cuya garantía teórica no cubre el régimen en que se usaría.

**Línea roja respetada:** ninguna de las 10 ideas autoriza reabrir, reinterpretar ni re-adjudicar O, O-R o Q. Las que cambian observación, reward, γ, horizonte, arquitectura, acción o comparador (5, 10, y los brazos de 9) exigen preregistro con hash, semillas y tapes vírgenes, y gates congelados antes de entrenar.
