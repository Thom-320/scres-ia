# Resultados consolidados para C&IE — el paquete, sin prosa de manuscrito

**Estado:** el paquete de desarrollo sobre `5.300.001–012` está cerrado. La única tentativa de
confirmación prospectiva fue **abortada y puesta en cuarentena** después de escribir 17 rebanadas
sin sello; no existe resultado confirmatorio. El bloque `8.100.001–8.100.060` no es reutilizable
como bloque virgen y no hay autorización registrada para abrir otro. Todo R0–R7 debajo es
desarrollo o réplica declarada.

---

## R0 · El bucle interno está saturado (evidencia previa, sellada)

| artefacto | veredicto |
|---|---|
| `results/headroom/g3_obs_conversion_v2/result.json` | `STRUCTURED_CONTROL_SUFFICES_G3_OBS` |
| `results/headroom/g2_autotomy_threshold/result.json` | `THRESHOLD_RULE_SUFFICES` |
| `results/headroom/buffer_prediction_premium/result.json` | MLP R² 0,5548 **peor que lineal** 0,6826 |
| `results/headroom/cd_surface_prediction_premium/result.json` | ambas redes bajo el clásico; gana el spline |
| `papers/paper2/results_table.json` | `Δ_N` negativo en las tres celdas |

---

## R1 · El efecto Alzheimer sobrevive a un normalizador honesto

`results/garrido_normaliser_audit_v3/result.json` · contrato
`docs/PREREGISTRO_AUDITORIA_NORMALIZADOR_2026-08-05.md`

El aprendiz normalizaba su objetivo con el min/max de **las 288 configuraciones**, incluidas las no
corridas. Reparado con normalizador **de prefijo**:

| contraste (AUC de regret) | oráculo | **prefijo** | LCB95 |
|---|---:|---:|---:|
| memoria − reset | +0,0902 | **+0,0607** | +0,0456 |
| memoria − OFAT | +0,0491 | **+0,0482** | +0,0333 |
| memoria − azar | +0,0873 | **+0,0865** | +0,0673 |

**`f1` ancla externa: PASA.** El arnés reproduce las cuatro medias del artefacto v2 sellado
—12,417 / 19,542 / 14,889 / 6,986— así que la física es idéntica y el número del prefijo no está
confundido con deriva.

**La fuga no era lo que le ganaba a OFAT** (+0,0491 → +0,0482). Lo que inflaba era
`memoria − reset`, **por dañar al control**: bajo el prefijo el brazo reset mejora de 14,89 a 12,92
corridas.

---

## R1b · La fuga, medida en vez de argumentada

`results/twin_surface_v2/result.json` · `f6` del mismo contrato · figura `fig_a_normaliser_leak`

El test afín pasaba con **ambos** normalizadores, y por eso era insuficiente: es ciego a una fuga
invariante a escala —un brazo que leyera el **rango** o el **argmax** lo pasaría—. El test de
**superficies gemelas** no lo es: conserva intactas todas las celdas que la ruta de referencia
visitó, altera **dos que ningún brazo tocó**, y repite el mismo flujo de RNG.

| normalizador | `ofat` | `random` | `neuron_reset` | `neuron_memory` |
|---|---:|---:|---:|---:|
| oráculo | 6/6 | 6/6 | **0/6** | **0/6** |
| prefijo | 6/6 | 6/6 | **6/6** | **6/6** |

*(contextos cuya ruta de búsqueda queda inalterada)*

**Bajo el oráculo los dos brazos neuronales cambian su ruta en los seis contextos** al alterar
celdas que nunca corrieron. Bajo el prefijo, en ninguno. `ofat` y `random` no se mueven en ningún
caso — nunca llaman al normalizador. Es un falsador con **un PASA y un FALLA obligatorios**, y
ninguno de los dos puede satisfacerse por acuerdo del código consigo mismo.

---

## R2 · La superficie es difícil de buscar, y la respuesta es la misma en todas partes

`results/surface_gates_v2/result.json` · contrato `docs/ENMIENDA_GATES_SUPERFICIE_2026-08-05.md`

**`g2` separabilidad** — ΔCV-R² con validación cruzada **dejando una semilla fuera**:

| contexto | ΔCV-R² | LCB95 |
|---|---:|---:|
| R1r\|esc | +0,1586 | **+0,1487** |
| R1r | +0,1572 | **+0,1459** |
| R1r+R2r | +0,1484 | **+0,1363** |
| R1r+R2r\|esc | +0,1097 | **+0,1016** |
| R2r | +0,1021 | **+0,0628** |
| R2r\|esc | +0,0719 | +0,0410 |

**La superficie NO es separable** en cinco de seis contextos contra un umbral de 0,05. Por tanto
**OFAT no es óptimo por construcción** y existe un problema de búsqueda real. Es lo que da objeto a
todo lo que sigue.

**`g1` valor de conocer el régimen** — `H_regime` **+0,0038 [LCB95 +0,0000]** contra 0,05
(5.000 bootstrap):
**13× por debajo**. El argmax se mueve en 4 de 6 contextos y **moverse vale el 0,4 % del rango
alcanzable**.

> `NON_SEPARABLE_BUT_CONTEXT_INVARIANT`. La memoria **no adapta al régimen; evita re-derivar una
> constante**. Cualquier lectura del resultado como adaptación contextual es falsa, y la refuta
> nuestro propio gate.

---

## R3 · Contra buscadores sin memoria, la neurona gana a todos

`results/search_ladder_v3/result.json` · contrato
`docs/ENMIENDA_ESCALERA_COMPARADORES_2026-08-05.md` · `B = 24`, mismo CRN, lecturas **enforzadas**
(`Surface.value_of_visited` lanza `LookupError`, no se afirma)

| brazo | AUC de regret |
|---|---:|
| oráculo (techo) | 0,00000 |
| **neuron_memory** | **0,05203** |
| ucb1 | 0,09655 |
| ofat | 0,10024 |
| gp_ei (**optimización bayesiana**) | 0,10661 |
| lhs_local | 0,10949 |
| neuron_reset | 0,11274 |
| random | 0,13979 |
| annealing | 0,17420 |

Los siete contrastes con LCB95 > 0. **Pero `neuron_reset` (0,11274) se sienta junto a `ofat`
(0,10024) y `gp_ei` (0,10661)**: sin memoria la neurona es un buscador del montón. Ese titular
compara **un buscador con memoria contra buscadores sin memoria**, que es casi una tautología.

---

## R4 · Con memoria para todos, el ingrediente es la retención

`results/search_ladder_v4/result.json` · contrato
`docs/ENMIENDA_ESCALERA_TRANSFERENCIA_2026-08-05.md`

Los tres clásicos más fuertes reciben **exactamente la información de la neurona** —observaciones de
contextos previos, normalizadas por prefijo dentro de cada contexto, **sin etiqueta de contexto**:

| brazo | AUC |
|---|---:|
| **ucb1_transfer** | **0,04502** |
| neuron_memory | 0,05203 |
| ofat_transfer | 0,06274 |
| gp_ei_transfer | 0,08390 |
| … sin memoria | 0,0966 – 0,1742 |

**`ucb1_transfer` empata o supera a la neurona**: −0,00701 **[LCB95 −0,02434]**, el intervalo cruza
cero con el punto estimado en contra. Y **el OFAT de Garrido con memoria (0,06274) bate a la
optimización bayesiana sin memoria (0,10661)**. Bajo el orden contractual la ventaja de la neurona
sobre el OFAT **con** memoria apenas cruza el cero: **+0,01071 [LCB95 −0,00003]**.

**El valor de la memoria, por familia** (gemelo sin memoria menos su versión con memoria):

| familia | ganancia | LCB95 |
|---|---:|---:|
| ucb1 | +0,0515 | +0,0362 |
| neurona | +0,0607 | +0,0461 |
| ofat | +0,0375 | +0,0293 |
| gp_ei | +0,0227 | +0,0128 |

> **El ingrediente medido es la RETENCIÓN, no el aproximador.** Cuatro familias distintas ganan
> materialmente al cruzar estado entre corridas.

---

## R5 · KAN vs MLP vs la neurona: equivalentes en calidad, y `Delta_efficiency` decide

`results/search_surrogates/result.json` · contrato
`docs/ENMIENDA_SURROGATES_Y_EFICIENCIA_2026-08-05.md`

| brazo | AUC | parámetros | ms/decisión (mediana) |
|---|---:|---:|---:|
| surrogate_mlp | 0,04745 | 369 | 0,19 |
| surrogate_kan | 0,04800 | 380 | 0,60 |
| **neuron_memory** | 0,04901 | **5** | **0,02** |

MLP −0,00156 **[−0,00594, +0,00260]** · KAN −0,00101 **[−0,00613, +0,00426]** — **ambos intervalos
cruzan cero**: equivalencia estadística.

> `APPROXIMATOR_IS_NOT_THE_INGREDIENT_RETENTION_IS`. Con la regla de lectura fijada de antemano
> —empate en AUC lo gana el más barato— **la neurona de la Fig. 5 de Garrido gana
> `Delta_efficiency`: 74× menos parámetros y 30× menos coste por decisión**. Es el estimando que el
> contrato E\* declaró y nunca se había medido.

**Advertencia metodológica, y va al manuscrito**: el humo de 2 semillas daba KAN ganando con
UCB95 < 0. Se declaró ininterpretable **antes** de correrlo y se evaporó a n = 12.

---

## Lo que este paquete responde, y lo que no

**Q1 de Garrido — qué familia de IA imita el aprendizaje de la cadena.** Respuesta medida: **la
retención de estado entre corridas, no una arquitectura**. Cuatro familias —bandido, OFAT, GP,
neurona— ganan al retener, y las tres arquitecturas son equivalentes en calidad. Entre las
equivalentes, **la suya, la más simple, es la más barata**.

**Q2 — cómo integrarla en el DES.** La interfaz corrida y sellada: episodio DES → resultado
observable → estado retenido → siguiente configuración, con las lecturas de configuraciones no
ejecutadas **imposibles por construcción**, no prohibidas por convención.

**Lo que NO se puede afirmar.** No hay prima neural en ningún bucle. No hay adaptación al régimen
(`H_regime` +0,0038). Esto es desarrollo sobre tapes quemados: **no adjudica, no abre semillas y no
autoriza entrenar**. Las cifras retiradas 7,24 / 12,42 / 13,54 / +6,31 siguen prohibidas.

---

## R6 · La rejilla extendida: añadir variables **sí** sube `H_regime`, y aun así no llega

`results/surface_gates_extended_v2/result.json` · 4.608 configuraciones · contrato
`docs/ENMIENDA_REJILLA_EXTENDIDA_4608_2026-08-05.md`

| gate | 288 | **4.608** |
|---|---|---|
| `g2` no separabilidad | 5/6 contextos, ΔCV-R² 0,072–0,159 | **6/6**, 0,112–0,190 (LCB95 hasta +0,179) |
| `g1` `H_regime` | +0,0038 [LCB95 +0,0000] | **+0,0283 [LCB95 +0,0147]** |

**Exponer los dos buffers aguas arriba multiplica `H_regime` por 7,4×** y saca su límite inferior
del cero. Es la instrucción de Garrido del 28 de julio —añadir nodos y variables de decisión, no
alargar el episodio— **medida, y en la dirección que él predijo**.

Y aun así el gate **falla**: sigue **1,8× por debajo** del umbral de 0,05. La superficie se volvió
más difícil de buscar, no más dependiente del régimen.

---

## R7 · Transferencia de rejilla 288 → 4.608: **la representación importa, y no es la red**

`results/grid_transfer_ordered_v1/result.json` · contrato
`docs/ENMIENDA_TRANSFERENCIA_REJILLA_ORDEN_CONTRACTUAL_2026-08-05.md` · `GRID_TRANSFER_ESTABLISHED__UCB1`

Cada familia entrena su carrera de seis contextos sobre 288 y **el mismo estado retenido** busca
sobre 4.608. Control: la misma familia **arrancando en frío**. Placebo decisivo: **réplica
marginal**, que reproduce su distribución de visitas ignorando el estado.

| familia | vs arranque en frío | vs **réplica marginal** |
|---|---:|---:|
| neurona | +0,0623 **[+0,0344]** | +0,0133 [−0,0030] |
| **UCB1** | +0,0639 **[+0,0431]** | **+0,0305 [+0,0154]** |
| OFAT | +0,0152 **[+0,0094]** | +0,0030 [−0,0193] |
| GP-EI | +0,0199 **[+0,0071]** | −0,0044 [−0,0260] |

**Las cuatro transfieren mejor que arrancar de cero en este bloque de desarrollo.** Eso **refuta nuestra propia hipótesis** de
que un prior GP no cruza un cambio de espacio de diseño — por eso la enmienda lo puso como brazo a
medir y no como premisa.

**Pero sólo el bandido bate a su réplica marginal.** Para la neurona, OFAT y el GP, lo que cruzó la
frontera es **una distribución de visitas** —una tabla de consulta sobre configuraciones— y **no la
forma de la superficie**.

> **La representación que transfiere es la que está factorizada como el espacio de diseño**: un
> estadístico por nivel de factor. Ni la red, ni el proceso gaussiano. El estado de UCB1 son
> exactamente los objetos que sobreviven a añadir factores nuevos; `ρ` vive sobre coordenadas cuyo
> significado se desplaza, y los puntos del GP están todos en `(…,0,0)`.

**Advertencia metodológica, la segunda del paquete**: el humo con **una** semilla daba el orden
**exactamente invertido** —neurona y OFAT peor que en frío, el GP el único que ayudaba—. Se declaró
ininterpretable antes de correrlo, y a n = 12 se dio la vuelta.

---

## Integridad reparada por el camino, y que va al apartado de reproducibilidad

**Los tres defectos de la plataforma Q2** (`scripts/run_garrido_q2_des288_v1.py`). El grave: OFAT
re-corría **su última propuesta** en vez de la incumbente al agotar el diseño, porque el guardián
`"idx" not in locals()` era falso desde el paso 1 —`del idx` se ejecuta una vez por *contexto*—.
**Validado reintroduciendo el defecto**: el brazo con bug repite `op12_rop = 48` mientras su
incumbente es `12`, y `tests/test_q2_ofat_exhaustion.py` falla sobre él. Además, `f9` pasa por el
registro central en vez de una tupla mantenida a mano, y `all_passed` deja de contar los
`not_applicable`.

**El registro de custodia: de 10 a 32 bloques, 327 semillas recuperadas**
(`results/custody/registry_reconciliation.json`). La captura que lo justifica: **`7.100.001`**,
reservada por el preregistro del DES-288 como bloque **virgen de confirmación**, está consumida por
tres artefactos de humo —**cada uno de los cuales selló `virgin_seed_block: true`**—. La custodia
ahora devuelve `COLLISION`. Las semillas de optimizador (`9301–9310`) quedan **excluidas a
propósito**: indexan una inicialización de torch, no una cinta CRN, y registrarlas fabricaría
colisiones inexistentes.

## Figuras

`scripts/build_cie_outer_loop_figures.py` → `docs/manuscript_current/submission/elsevier/figures/`.
**Ningún número está cableado**: cada valor se carga de su `result.json`, así que una figura no
puede alejarse de la evidencia que dice mostrar y re-correr es todo el camino de actualización.

| figura | qué muestra |
|---|---|
| `fig_a_normaliser_leak` | R1b: la fuga y su reparación, medidas |
| `fig_b_surface_gates` | R2: no separabilidad por contexto, y `H_regime` contra su umbral |
| `fig_c_comparator_ladder` | R3+R4: la escalera con y sin memoria |
| `fig_d_memory_price` | R4: lo que la retención vale a cada familia |
| `fig_e_delta_efficiency` | R5: calidad contra coste, **con banda de equivalencia explícita** |

La banda de `fig_e` no es decorativa: los tres puntos difieren en la página pero **todos los
intervalos cruzan cero**, y sin ella la figura afirmaría un orden que sus propios datos niegan.

---

## Estado de confirmación

Los siete resultados de desarrollo están medidos, sellados y con falsadores. El replay R7 ordenado
seleccionó UCB1 en el bloque quemado; el resultado exploratorio con orden alfabético queda
supersedido. La tentativa confirmatoria no produjo resultado. El contrato
`docs/PREREGISTRO_CONFIRMACION_TRANSFERENCIA_REJILLA_2026-08-05.md` fijó el estimando, la potencia,
el bloque y la regla de lectura; el preflight
`results/custody/garrido_grid_transfer_confirmation_preflight.json` pasó, pero el recibo
`results/custody/garrido_grid_transfer_confirmation_abort.json` adjudicó el intento como
`CONFIRMATION_BLOCK_QUARANTINED_NO_SCIENTIFIC_RESULT`.

No hay un bloque virgen confirmatorio disponible ni autorización registrada para abrir otro. El
único resultado defendible es el de desarrollo R7; RL, PPO, MLP y KAN siguen fuera del carril.

**El eje de la red está cerrado por medición, no por cansancio.** Dentro del episodio no hay prima
neural en cuatro contratos. Entre corridas, tres aproximadores son estadísticamente equivalentes y
el más barato es el de cinco parámetros. Y al cruzar espacios de diseño, **lo que transfiere no es
un aproximador sino una factorización**.
