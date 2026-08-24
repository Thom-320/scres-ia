# REVISION_OPENCODE_COBERTURA

**Fecha:** 2026-08-24 · **Autor:** OpenCode (ox-alpha) · **Alcance:** exclusivamente `/home/ubuntu/scres-sources`.
**Insumos leídos:** `registry/BIBLIOGRAFIA_REGISTRO.json`, `registry/BIBLIOGRAFIA_VERIFICADA.json`, `registry/BIBLIOGRAFIA_SHORTLIST.md`, `reports/MANIFIESTO_PDFS.md`, `pdfs_frontier/context_reports/*.md` (CONTEXTO_COMUN, REPORT_FRONTERA, IDEAS_OPENCODE, INFORME_RECONCILIACION_BATCH_2), `BRIEF_REVISION_LITERATURA.md`.
**Cumplimiento:** no se accedió a `scres-ia-expanded-v2` ni a ningún directorio fuera de `scres-sources`; nada de este informe depende de ese repositorio.

---

## Resumen ejecutivo

- El registro verificado tiene **84 DOIs únicos** (todos con año confirmado vía Crossref/DataCite) más 1 entrada sin DOI (Audibert 2010, BAI). Distribución temporal: **50 % es 2021+**, pero la cola ≤2010 pesa 20 % y está concentrada en el track F (simopt/bandits), donde es legítima.
- La poda recomendada elimina **26 entradas firmes (+3 opcionales)** del registro: todo el bloque E_KAN (7), dos duplicados arXiv↔publicado, tres revisiones gerenciales de resiliencia, y cinco técnicos fuera de dominio (portafolios, PDE, RTS games, redes de agua, rutas).
- El **núcleo irreemplazable pre-2016 son 17 obras** (sección 3): POMDP/MPC (Kaelbling 98, Mayne 00), inventario (Zipkin 08, Treharne 02), riesgo (Rockafellar-Uryasev 00), resiliencia (Tomlin 06, Ivanov 13/14, Snyder 15/16), y el arsenal estadístico del loop externo (Jones 98, Auer 02, Kleywegt 02, Boesel 03, Frazier 08/14).
- De la **shortlist de 70**, marcaría **~28 como ruido o mal encajados**: sobre-representación de digital twin aplicado y de DRO de diseño de redes, más 4-5 entradas de venues débiles que restan credibilidad.
- **Huecos duros:** faltan las fuentes primarias de PBRS (Ng 1999; Wiewiora 2003), los algoritmos MARL base (MAPPO/QMIX), el CMDP formal (Altman 1999 / Achiam 2017), los clásicos multi-echelon/newsvendor-minimax (Clark-Scarf 1960, Gallego-Moon 1993) y una referencia de forecasting→control. Detalle en sección 5.

---

## 1. DISTRIBUCIÓN (usando `year_verified` de BIBLIOGRAFIA_VERIFICADA.json)

### 1.1 Subcampo × periodo (n = 84)

| Subcampo | 2021+ | 2016-2020 | 2011-2015 | ≤2010 | Total |
|---|---:|---:|---:|---:|---:|
| A_control_estructurado_con_creencias | 9 | 6 | 1 | 6 | 22 |
| B_hibridos_DES_RL_y_benchmarks_DRL_SCM | 15 | 3 | 1 | 2 | 21 |
| C_metricas_resiliencia_y_disrupcion | 4 | 5 | 2 | 1 | 12 |
| D_riesgo_de_cola_CVaR_distribucional | 3 | 3 | 0 | 2 | 8 |
| E_KAN | 6 | 1 | 0 | 0 | 7 |
| F_loop_externo_simopt_banditos | 0 | 1 | 2 | 6 | 9 |
| Suplementarios frontera sin topic (A11-A13, B11, B12) | 5 | 0 | 0 | 0 | 5 |
| **TOTAL** | **42** | **19** | **6** | **17** | **84** |

Porcentajes: 2021+ = 50,0 % · 2016-2020 = 22,6 % · 2011-2015 = 7,1 % · ≤2010 = 20,2 %. Con Audibert 2010 (fila sin DOI del registro, no presente en el fichero verificado) el pool sería 85 y ≤2010 subiría a 18.

### 1.2 Lecturas de la tabla

1. **La exigencia del brief (≥50 referencias priorizando 2021+) ya se cumple en el registro** (42) y de sobra sumando la shortlist (70 nuevas, todas 2021+): pool combinado ≈ 155 filas, con **112 de 2021+ (≈72 %)**.
2. **F está congelado en el tiempo**: 0 entradas 2021+ propias. Es un artefacto de clasificación, no de cobertura: Hong 2021 (`10.1007/s42524-021-0152-6`) y Cheng 2023 (`10.1016/j.ejor.2022.11.038`), que son R&S purísimo, están archivadas bajo *A_control_estructurado_con_creencias*, y los suplementarios A12/A13 (large-scale SO 2025, POMDP-R&S 2023) quedaron sin topic. La taxonomía del registro subestima la frescura del track F.
3. **E_KAN es un cuerpo extraño**: 7 entradas (8,3 % del registro) sin ninguna conexión con el resto ni con el claim del paper (ver poda en §2).
4. **C es el subcampo más viejo proporcionalmente** en sus primeras filas (Tomlin 06, Brandon-Jones 14, Ivanov 13/14, Snyder 15/16): correcto para definiciones, pero sus 4 entradas 2021+ incluyen dos revisiones gerenciales (Negri, Belhadi) poco alineadas con un paper de control.
5. **Correcciones de año declarado→verificado** (17 entradas cambian de periodo; las más relevantes: Amaran pasa de 2015 a **2016**, Snyder de 2015 a **2016**, Ivanov-Dolgui ripple a 2014/**2018**, Oroojlooyjadid MSOM a **2022**, Boute roadmap a **2022**, Katsaliaki a **2022**, Belhadi a **2024**, KAN-survey ACM a **2026**). Todas las tablas de arriba usan el año verificado.

### 1.3 Shortlist (70 candidatos, todos 2021+)

| Subcampo | n |
|---|---:|
| inventory_benchmarks_drl | 14 |
| distributionally_robust | 13 |
| digital_twin_manufacturing | 11 |
| marl_logistics | 7 |
| gnn_supply_chain | 6 |
| supply_chain_resilience | 6 |
| offline_rl_ope | 4 |
| safe_constrained_rl_ops | 3 |
| simulation_optimization_rs | 3 |
| risk_averse_rl | 2 |
| des_calibration_validation | 1 |

Por año: 2021: 6 · 2022: 8 · 2023: 12 · 2024: 20 · 2025: 13 · 2026: 11. Desequilibrio claro: **DRO y digital twin aportan 24 de 70 (34 %)**, con muchos trabajos de diseño de redes/casos que no son control de inventario ni resiliencia operacional (ver §4).

---

## 2. PODA — entradas del registro que NO deberían ir en un paper Q1 de frontera

Motivos: **[O] obsoleta** (su función ya no se cumple o el artefacto cambió), **[T] tangencial** al problema (SCRES + RL sobre DES con comparador belief-MPC), **[S] superada por trabajo posterior** (hay reemplazo directo, mejor o publicado). Los tres motivos pueden combinarse; indico el dominante.

### 2.1 Bloque E_KAN completo — [T], 7 DOIs

Para un paper de resiliencia de cadena de suministro con RL, ninguna es citable salvo que el paper adopte KAN como aproximador (que no está en el plan):

- `10.48550/arxiv.2404.19756` (KAN original) — única defendible si se menciona la arquitectura en related work; aun así prescindible.
- `10.2139/ssrn.4835325` (Wav-KAN) — además es **preprint SSRN** (`posted-content`, sin peer review según el propio registro): doble motivo.
- `10.1145/3743128` (survey KAN, ACM Comput. Surv.)
- `10.1016/j.cma.2024.117699` (DeepOKAN, mecánica de sólidos)
- `10.1038/s42256-025-01087-7` (KA-GNN, predicción molecular)
- `10.1016/j.neunet.2019.12.013` (acotaciones Kolmogorov-Arnold)
- `10.1016/j.neunet.2021.01.020` (teorema de representación revisitado)

Coste de mantenerlas: diluye el ratio 2021+-útil y sugiere colección por curiosidad, no por diseño. **Recomendación: sacar las 7 del paper; conservarlas en el registro como archivo.**

### 2.2 Fuera de dominio técnico — [T]

- `10.1137/19m125039x` (Garreis 2021, SIAM J. Optim.) — interior-point para optimización **PDE-condicionada** con medidas de riesgo coherentes. Cero conexión con SCM; la capa risk-averse del proyecto ya tiene Rockafellar-Uryasev 2000 + Dabney 2018 + Wang TRE 2021.
- `10.1080/02331930600816353` (Beliakov 2006) — CVaR no-suave para **optimización de portafolios**. La formulación computacional de CVaR canónica es Rockafellar-Uryasev (`10.21314/jor.2000.038`). [T]+[S].
- `10.1613/jair.5398` (Ontañón 2017) — bandits combinatorios para **videojuegos RTS**. Si el loop externo se formaliza como bandit, las referencias pertinentes son Auer 2002, Frazier 2008/2014 y Hong 2021. [T]
- `10.1609/aaai.v27i1.8637` (Ding 2013, bandit presupuestado) — genérica, sin anclaje a simulación ni a SCM; candidata a poda salvo que el presupuesto de calendarios se formule explícitamente como bandit presupuestado. [T] (borde)
- `10.1023/a:1021814225969` (Verweij 2003, SAA en **rutas** estocásticas) — el método SAA ya queda definido por Kleywegt-Shapiro (`10.1137/s1052623499363220`). [T]+[S] (redundante)
- `10.1049/iet-cta.2015.0657` (Ye Wang 2016, SMPC-GP para **redes de agua potable**) — dominio ajeno; para SMPC basta Farina 2016 (`10.1016/j.jprocont.2016.03.005`) + Mayne 2000. [T]

### 2.3 Resiliencia gerencial/conceptual — [T]

- `10.1111/jscm.12050` (Brandon-Jones 2014) — perspectiva recurso-contingencia, cualitativa; no alimenta métricas ni modelos.
- `10.1108/scm-06-2016-0197` (Ali 2017) — concept-mapping SLR; cubierta por la major review de Katsaliaki (`10.1007/s10479-020-03912-1`). [T]+[S]
- `10.1002/bse.2776` (Negri 2021) — SLR sostenibilidad+resiliencia; ángulo de sostenibilidad fuera del claim; solo si el paper vendiera circularidad (Guzmán 2026 ya cubre eso con control). [T]
- `10.1016/j.jclepro.2016.03.059` (Papadopoulos 2016/17, big data y resiliencia ante desastres) — empírico/big-data, tangencial y anticuado como referencia de métricas. [T]+[O]
- `10.1109/tpwrs.2017.2664141` (Panteli 2017, métricas de resiliencia en **redes eléctricas**) — las métricas que realmente usa el proyecto (TTR, fill rate, recovery) llegan por Kim 2024, Guzmán 2026 y Fan 2023. Solo citar si se necesita precedencia inter-dominio de métricas; por defecto, fuera. [T]
- `10.1007/s10479-021-03956-x` (Belhadi 2021/24, adopción empírica de IA y resiliencia) — estudio empírico-gerencial (encuesta/modelo SEM); no aporta a un paper de control. [T]

### 2.4 Duplicados y versiones — [S]

- `10.48550/arxiv.1708.05924` (Beer Game DQN, arXiv 2017) — **duplicado** de la versión publicada `10.1287/msom.2020.0939` (MSOM, verificada 2022). Citar solo la publicada.
- `10.48550/arxiv.1707.06887` (Bellemare, arXiv 2017) — el contenido está en ICML 2017 y en el libro MIT (`10.7551/mitpress/14207.001.0001`, ya en registro). Como cita Q1, el DOI arXiv es la peor opción de las tres.
- `10.1007/978-3-319-28872-7_35` (Kurniawati 2016, capítulo Springer) — solapa con la línea DESPOT ya cubierta por la referencia canónica `10.1613/jair.5328` (Ye 2017, JAIR). Mantener **uno** de los dos; prefiero Ye 2017. [S] (dentro de su propia línea)

### 2.5 Superadas como estado del arte — [S] (+[O])

- `10.1145/268437.268460` (Carson & Maria 1997, "Simulation optimization") — panorama de 1997; la función de "review de simopt" ya la hacen Amaran 2016 (`10.1007/s10479-015-2019-x`) y Fan-Hong-Jiang-Luo 2025 (`10.1007/s40305-025-00599-8`). Además figura como MANUAL en `MANIFIESTO_PDFS.md` (nunca descargado). [O]+[S]
- `10.1109/wsc.2011.6148097` (Pasupathy 2011, SimOpt library) — describe la librería tal como era en 2011; citar esa versión como estado del arte es anacrónico. Actualizar a la publicación vigente de la librería antes de citar SimOpt. [O]
- `10.24251/hicss.2018.157` (Fuji 2018, HICSS, MARL evolutivo) — workshop/conferencia con análisis ligero; el rol "MARL temprano en SCM" lo cubren con rigor Kim 2024 (`10.1080/24725854.2023.2217248`), Liu 2025 (`10.1177/10591478241305863`) y Mousa 2024 (`10.1016/j.compchemeng.2024.108783`). [S]
- `10.1016/j.dss.2008.03.007` (Chaharsooghi 2008, RL beer game) — valor histórico (primer RL multi-echelon en beer game); como estado del arte está superado por Oroojlooyjadid MSOM 2022 y Gijsbrechts 2022. Mantener solo como cita histórica de una línea; para el núcleo es prescindible. [S]
- `10.1016/j.ejor.2013.05.044` (Wu 2013, newsvendor averso al riesgo con capacidad aleatoria) — el rol "newsvendor con riesgo" queda cubierto por Yang 2018 (`10.1111/poms.12881`) y por la capa CVaR/DRO. Poda opcional (la menos urgente de esta lista). [S]/[T]

### 2.6 Redundancia interna DT+producción — [T]/[S]

- Entre `10.3390/app11072977` (Park 2021, micro smart factory DT+RL) y `10.1080/00207543.2021.1884309` (Park 2021/22, re-entrant job shop DT+RL) basta **uno** como evidencia DT+RL de planta; recomiendo conservar el de IJPR (venue más fuerte) y podar el de Applied Sciences (MDPI, peso menor).
- `10.1016/j.jmsy.2023.12.008` (Zhang 2024, AGV dispatching con DT+DRL) — despacho/routing de vehículos en piso de planta, no inventario ni resiliencia multi-echelon. [T]

**Balance de poda:** **26 entradas firmes** (31 % del registro) + **3 opcionales** (Ding 2013, Wu 2013, Chaharsooghi 2008). Tras la poda firme, el registro útil queda en **58 DOIs** con densidad 2021+ de **≈53 %** (31/58, frente al 50 % actual), sin perder ninguna definición canónica (ver §3). Si además caen las 3 opcionales, el registro útil sería 55 DOIs (≈56 % de 2021+).

---

## 3. NÚCLEO IRREEMPLAZABLE (pre-2016 que DEBE quedarse)

Estos son los clásicos que definen canónicamente algo que el paper usa; quitar cualquiera obligaría a citar de segunda mano (inaceptable en Q1):

| # | DOI | Obra (año verificado) | Por qué es irreemplazable aquí |
|---|---|---|---|
| 1 | `10.1016/s0004-3702(98)00023-x` | Kaelbling, Littman & Cassandra 1998 | **Definición canónica del POMDP** y de la representación por creencias. Todo el framing "observación parcial + belief" del proyecto (y el comparador belief-MPC) desciende de este paper. |
| 2 | `10.1613/jair.2567` | Ross, Gordon & Bagnell 2008 | Define la familia de **planificación online en espacio de creencias** a la que pertenece el comparador classical; establece la distinción planning-vs-learning que estructura el paper. |
| 3 | `10.1016/s0005-1098(99)00214-9` | Mayne et al. 2000 | **Definición canónica de MPC con restricciones** (estabilidad y optimalidad). El belief-MPC del proyecto hereda exactamente esa estructura; cualquier revisor de control lo exigirá. |
| 4 | `10.1287/mnsc.48.5.607.7807` | Treharne & Sox 2002 | Caso fundacional de **control adaptativo de inventario con demanda no estacionaria e información parcial** — el régimen exacto del proyecto (demanda variable + observación parcial). |
| 5 | `10.1287/opre.1070.0482` | Zipkin 2008 | **Definición estructural del lost-sales model** (condiciones de optimalidad base-stock). Gijsbrechts 2022 benchmarka contra estas soluciones: sin Zipkin, el baseline lost-sales queda huérfano. |
| 6 | `10.21314/jor.2000.038` | Rockafellar & Uryasev 2000 | **Definición operacional de CVaR** y su optimización. Las métricas de cola del proyecto (CVaR10, worst_product_fill con aversión) derivan de esta formulación. |
| 7 | `10.1287/mnsc.1060.0515` | Tomlin 2006 | **Canon del trade-off mitigation vs contingency** ante disrupciones: es la pregunta económica que el learner intenta responder mejor que open-loop. |
| 8 | `10.1080/00207543.2013.858836` | Ivanov et al. 2013 (verif. 2014) | Canonización del **ripple effect** y del trade-off efficiency–flexibility–resilience: el vocabulario SCRES del paper sale de aquí. |
| 9 | `10.1080/0740817x.2015.1067735` | Snyder et al. (declarado 2015, verificado 2016) | Taxonomía canónica de **modelos OR/MS de disrupción**; es la referencia-puente entre Tomlin/Ivanov y la literatura 2021+. Técnicamente cae en 2016-2020 por año verificado; la mantengo como núcleo funcional. |
| 10 | `10.1002/9780470182963` | Powell 2007 (libro ADP) | **Vocabulario canónico de programación dinámica aproximada** (aproximación de funciones de valor, políticas); posiciona al learner frente a ADP clásica sin reinventar términos. |
| 11 | `10.1023/a:1008306431147` | Jones, Schonlau & Welch 1998 (EGO) | Origen del **loop externo bayesiano** (kriging + expected improvement): ancestro directo del knowledge gradient y de todo el track F. |
| 12 | `10.1023/a:1013689704352` | Auer, Cesa-Bianchi & Fischer 2002 (UCB) | **Garantías finite-time de bandits**: base de cualquier asignación adaptativa de réplicas que el proyecto proponga. |
| 13 | `10.1137/s1052623499363220` | Kleywegt, Shapiro & Homem-de-Mello 2002 (SAA) | **Definición canónica del método Sample Average Approximation**; necesaria si se citan baselines estocásticos o el propio concepto de optimización sobre simulación. |
| 14 | `10.1137/070693424` | Frazier, Powell & Dayanik 2008 (KG) | **Política knowledge-gradient** para recolección secuencial de información: raíz de OCBA/KG en el diseño de evaluación. |
| 15 | `10.1287/opre.51.5.814.16751` | Boesel, Nelson & Kim 2003 | Protocolo canónico de **R&S como "clean-up" tras optimización por simulación**: literalmente el diseño SO→R&S del proyecto. |
| 16 | `10.1287/opre.2014.1282` | Frazier 2014 (KN) | Procedimiento **fully sequential con zona de indiferencia** y PCS garantizada: la receta directa para declarar equivalencia/no-prima (Δ_N) sin depender de LCB95 con N fijo. |
| 17 | *(sin DOI)* | Audibert et al. 2010 (Best Arm Identification) | Canon de **best-arm identification fixed-confidence**, complemento natural de KN. Está en el registro como fila sin DOI (PDF local vía HAL/arXiv; el registro solo registra a Audibert como autor). Acción requerida: añadir identificador verificable (HAL/arXiv) antes de citarlo — regla del brief. |

Notas de borde:

- `10.1613/jair.5328` (DESPOT, Ye 2017) y `10.1016/j.jprocont.2016.03.005` (Farina 2016, SMPC chance-constraints) son **cuasi-núcleo 2016-2017**: no entran en la lista pre-2016 pero sí deben quedarse (línea online-POMDP y restricciones probabilísticas del comparador, respectivamente).
- `10.1016/j.ejor.2017.06.049` (Meissner 2017/18, ADP transshipments) es defendible como antecedente ADP multi-localización, pero no canónico; sobrevive a la poda solo como related work.

---

## 4. REVISIÓN DE LA SHORTLIST (70 candidatos)

Solo tuve metadatos (título/venue/citas/OA de `BIBLIOGRAFIA_SHORTLIST.md`); no hay textos locales de estos 70, así que el juicio es de encaje bibliográfico, no de lectura. Marco **ruido/mal encajado** con motivo; el resto los doy por útiles (algunos con reserva).

### 4.1 Ruido claro (quitar)

| Fila | DOI | Motivo |
|---|---|---|
| 62 | `10.1007/s10589-021-00288-1` | Etiquetada `simulation_optimization_rs` pero es **selección de variables para regresión logística**. Mal clasificada; no es ni simopt ni R&S. |
| 49 | `10.56726/irjmets81353` | Venue tipo IRJMETS, de estándar editorial mínimo; blockchain+MARL promocional. Su presencia resta credibilidad a la bibliografía. |
| 31 | `10.52783/jisem.v10i33s.5571` | Patrón de DOI/venue (JISEM) de baja confiabilidad editorial; GNN predictiva sin relación con control. |
| 30 | `10.17559/tv-20240606001759` | Detección de fraude con GNN+knowledge graph: tangencial al 100 % y venue débil. |
| 59 | `10.20473/jisebi.7.2.138-148` | Safe-RL de inventario genérico en venue universitario menor; el tema ya lo cubren Burtea & Tsay 2024 y las filas 60-61. |
| 28 | `10.1016/j.eswa.2025.128705` | Extracción de relaciones entidad (automoción) con GNN heterogénea: NLP/documentos, nada que ver con control de red. |
| 27 | `10.1016/j.asoc.2024.112475` | GNN federada para privacidad de datos: gobernanza de datos, no control. |
| 26 | `10.1016/j.asoc.2022.109849` | Clasificación de industrias con GNN: analítica, no operación. |
| 12 | `10.1108/mbe-06-2021-0084` | Case study RFID+DT en venue gerencial (Measuring Business Excellence): sin contenido técnico aprovechable. |
| 10 | `10.1016/j.sciaf.2023.e01821` | "Implementation approach" de DT genérico: relato de implantación, no método. |
| 9 | `10.1016/j.compind.2023.103884` | Sistema de recomendación trustworthy sobre DT: otra tarea de ML. |
| 11 | `10.1016/j.ifacol.2022.09.413` | Proceedings IFAC sobre scheduling DT+RL: redundante con las entradas journal del mismo clúster; preferir revista. |
| 45 | `10.1057/s41272-021-00281-7` | Seat inventory de aerolíneas (revenue management): dominio distinto; solo sirve como analogía lejana. |
| 63 | `10.1016/j.eswa.2023.119624` | Selección de proveedores MCDM+simulación: decisión gerencial multicriterio, no el loop SO/R&S del paper. |
| 64 | `10.1007/s10479-021-04424-2` | Ídem (multi-objectivo supplier selection). |
| 67 | `10.1108/bpmj-02-2024-0073` | Encuesta empírica de resiliencia en pymes (BPMJ): gerencial, sin modelo utilizable. |
| 69 | `10.5267/j.dsl.2024.7.005` | Metaheurístico de diseño de red en venue menor: fuera del alcance RL/control. |

### 4.2 Mal encajados parciales (descartar salvo reorientación del paper)

| Fila | DOI | Motivo |
|---|---|---|
| 2, 3 | `10.1080/00207543.2024.2338878`, `10.1016/j.rcim.2025.103042` | Clúster DT de piso de planta (repuestos digitales; scheduling con transporte): manufacturing operations, no resiliencia de SCM con aprendizaje. El clúster DT ya tiene mejores embajadores (filas 4, 5, 8). |
| 13, 18, 19, 20, 21, 25 | `10.1016/j.eswa.2023.119916`, `10.1111/itor.13267`, `10.1016/j.cie.2022.108845`, `10.1016/j.cie.2022.108051`, `10.1016/j.compchemeng.2021.107307`, `10.1016/j.jclepro.2024.141563` | DRO de **diseño/configuración de red** (VMI config, cold chain farma, selección+enrutamiento perecederos, healthcare canadiense, maritime inventory routing, capacity-sharing): el paper necesita DRO solo como *framing* de ambigüedad para inventario/control, no 6 diseños de red. |
| 23 | `10.1080/00207543.2026.2702564` | DRO "bone China supply chains", 0 citas, encaje anecdótico. |
| 17 | `10.1080/24725854.2024.2323165` | Contrato de consignación con DRO: teoría de contratos, no control operativo. |
| 14 | `10.1007/s10479-024-05916-7` | Humanitaria con disrupciones correlacionadas: dominio y objetivo distintos (diseño fiable pre-desastre). Marginal. |
| 60 | `10.1007/s10458-024-09669-2` | Constrained DRL para **carbon trading**: el ángulo carbón no es el claim; el mecanismo (constrained MARL) ya está en Burtea/Akashi. |
| 58 | `10.1016/j.asoc.2025.113079` | Distributional RL para trenes: dominio ferroviario; conservar solo si se quiere un segundo ejemplo aplicado de CVaR-RL. |
| 51 | `10.1016/j.cej.2024.154464` | Diseño de cadena de hidrógeno con MARL jerárquico (CEJ): dominio energético; marginal. |
| 66 | `10.1016/j.sca.2024.100091` | Framework conceptual de DT (Supply Chain Analytics): conceptual y en venue menor; el papel "DT para recuperación" ya lo juega mejor la fila 8 (CIE) e Ivanov 2021 del registro. |

### 4.3 Caution (mantener pero citando con conciencia de venue)

- **4** `10.3390/systems12020038` y **6** `10.3390/logistics5040084`: MDPI de segundo nivel, aunque con citas altas (78 y 133). Útiles como señal de interés DT+RL, no como autoridad técnica.
- **42** `10.1007/s11518-022-5544-6` (JSSE): repaso DRL-inventario correcto pero redundante con Boute 2022 (roadmap) y Gijsbrechts 2022.

### 4.4 Lo valioso de la shortlist (para que la poda no confunda)

Fuertes y directamente en el claim: **32** (`10.1016/j.ejor.2021.10.045`, reward shaping en inventario perecedero — puente directo A-lane), **38** (`10.1287/mnsc.2022.02533`), **39** (`10.1287/mnsc.2023.4947`), **34** (`10.1016/j.ejor.2023.10.007`), **57** (`10.1016/j.compchemeng.2024.108912`, robust RL SCM), **50** (`10.1016/j.ijpe.2026.109995`, reconfiguración de resiliencia con MARL — lane B), **48** (`10.1016/j.ijpe.2026.110067`), **46** (`10.1080/00207543.2021.2020927`), **52** (`10.1016/j.apm.2023.10.039`), **35-37, 41, 43, 44** (bloque DRL-inventario), **47** (`10.1080/00207543.2025.2598025`), **53** y **55** (`10.1287/opre.2022.2271`, `10.1287/opre.2021.0781` — OPE/offline con MDP parcialmente observable: refuerzan la agenda de evaluación), **56** y **54** (OPE, secundarios), **24** (`10.1287/opre.2024.1481`, dual sourcing DRO en OR — el único DRO imprescindible), **16** (`10.1016/j.omega.2025.103443`), **15** (`10.1016/j.ejor.2025.07.065`), **22** (`10.1016/j.cor.2020.105081`), **68** (`10.1016/j.omega.2026.103609`), **1** (`10.1007/s10479-021-04382-9`, calibración DES: conecta con la validación física del simulador), **8** (`10.1016/j.cie.2024.110670`, DT de simulación en la revista objetivo CIE), **5** (`10.1007/s12351-024-00831-y`, detección de disrupción), **65** (`10.1108/scm-09-2020-0434`) y **70** (`10.1111/jscm.12304`) como citas de motivación gerencial.

Con esta limpieza, la shortlist pasaría de 70 a ~40 referencias realmente incorporables, todas 2021+, lo que deja el total del paper (~64 registro útil + ~40 shortlist) holgado por encima de las ≥50 pedidas.

---

## 5. HUECOS RESTANTES (registro + shortlist sumados)

Ordenados por severidad para un revisor de CIE:

1. **PBRS sin fuentes primarias.** Okudo 2021, Müller 2025 y HPRS 2025 aplican potential-based shaping, y los informes internos citan "Ng 1999" constantemente, pero **Ng, Harada & Russell (ICML 1999)** — el teorema de invarianza de políticas — **no está en el registro ni en la shortlist**. Tampoco **Wiewiora et al. 2003** (equivalencia shaping ↔ inicialización de Q). Son las dos fuentes que legitiman el supresor #2 del fix-pack. (Sin DOI en los ficheros: verificar y añadir antes de citar.)
2. **Algoritmos MARL base ausentes.** Kim 2024, Liu 2025 y Mousa 2024 comparan **MAPPO y QMIX**, y el plan de lane B los usa, pero ni **Yu et al. 2022 (MAPPO, NeurIPS)** ni **Rashid et al. 2018 (QMIX, ICML)** están en el pool. Un revisor los pedirá en la primera ronda.
3. **CMDP/safe RL sin fundamento formal.** Burtea & Tsay 2024 usan Lagrangian PPO y el fix-pack propone restricciones hard/soft; faltan **Altman 1999 (Constrained MDPs)** y/o **Achiam et al. 2017 (CPO)** como fuentes del formalismo.
4. **Clásicos de estructura de inventario incompletos.** Hay Zipkin 2008 (lost-sales) y Treharne 2002 (adaptativo), pero faltan **Clark & Scarf 1960** (estructura echelon multi-echelon) y la raíz minimax del newsvendor (**Scarf 1958 / Gallego & Moon 1993**), imprescindibles ahora que la shortlist empuja 13 entradas DRO: sin ellas la sección DRO queda sin genealogía.
5. **Bullwhip/beer game sin fuente clásica.** El pool usa beer game (Oroojlooyjadid) y bullwhip como métrica (Liu 2025), pero **Sterman 1989** (dinámica de la cadena de suministro) no está.
6. **SimOpt desactualizado.** La única referencia SimOpt es Pasupathy 2011 (ya marcada obsoleta en §2.5). Falta la publicación vigente de la librería (autores Eckman/Henderson/Pasupathy) si se va a mencionar SimOpt como entorno de benchmark.
7. **Forecasting→control.** Una de las ideas gated del proyecto (dosis-respuesta de calidad de forecast compartido learner/MPC) no tiene ninguna entrada bibliográfica: no hay ni un trabajo de demanda-neural + RL en el pool (los DRL-inventario de la shortlist asumen demanda simulada, no pipeline de forecast). Hueco real para el related work.
8. **Multi-objective RL.** Guzmán 2026 entrena reward multi-objetivo (coste+OTIF+emisiones) y el paper hablará de balance coste/resiliencia; no hay ninguna referencia metodológica MO-RL/Pareto en el pool.
9. **Línea POMDP online con hueco histórico menor.** Hay Ross 2008 → Kurniawati 2016/DESPOT 2017, pero falta el eslabón **POMCP (Silver & Veness 2010)** que casi toda la literatura online-POMDP cita entre ambos. Opcional pero barato de cerrar.
10. **Explicabilidad del learner.** Cero entradas XRL/interpretabilidad; para CIE convendría 1-2 referencias al discutir por qué el learner iguala al belief-MPC. Opcional.
11. **Encuesta DT-SCM canónica.** El bloque DT (registro + shortlist) es aplicado; falta una survey DT-cadena-de-suministro consolidada que sirva de ancla única del concepto. (Elegir y verificar; no la incluyo por no inventar DOI.)

---

## Verificación y límites de este informe

- Todos los años citados provienen de `year_verified` en `registry/BIBLIOGRAFIA_VERIFICADA.json` (84/84 verificados, 0 sin verificar); los conteos de la shortlist salen de `registry/BIBLIOGRAFIA_SHORTLIST.md` (70 filas, todas 2021+).
- Los juicios de §2 y §4 sobre "tangencial/obsoleto/superado" son evaluación editorial mía apoyada en títulos, venues, años y citas registradas en los ficheros; donde solo había metadatos (shortlist completa, y las ~38 entradas MANUAL del manifiesto sin texto local) no leí el contenido completo, y así debe tratarse.
- Los huecos de §5 nombran obras clásicas por autor-año **sin DOI a propósito**: no constan en los ficheros de registro y deben verificarse vía Crossref/arXiv antes de añadirlas (regla de honestidad del brief: no inventar citas).
- No se tocó ningún fichero fuera de `reports/` en este directorio.

*Generado por OpenCode (ox-alpha), 2026-08-24.*
