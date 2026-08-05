# Resultados consolidados para C&IE — el paquete, sin prosa de manuscrito

**Estado:** desarrollo sobre el bloque quemado `5.300.001–012`, réplica declarada. **Ninguna
semilla nueva, ninguna adjudicación, ningún aprendiz autorizado.** Todo lo de abajo sale de la
caché sellada `results/surface_cache/wrap288_v1` (72 rebanadas, 20.736 episodios) y de artefactos
con contrato, hash y falsadores.

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

`results/garrido_normaliser_audit/result.json` · contrato
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

`results/twin_surface/result.json` · `f6` del mismo contrato · figura `fig_a_normaliser_leak`

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

`results/surface_gates/result.json` · contrato `docs/ENMIENDA_GATES_SUPERFICIE_2026-08-05.md`

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

**`g1` valor de conocer el régimen** — `H_regime` **+0,0038 [LCB95 +0,0000]** contra 0,05:
**13× por debajo**. El argmax se mueve en 4 de 6 contextos y **moverse vale el 0,4 % del rango
alcanzable**.

> `NON_SEPARABLE_BUT_CONTEXT_INVARIANT`. La memoria **no adapta al régimen; evita re-derivar una
> constante**. Cualquier lectura del resultado como adaptación contextual es falsa, y la refuta
> nuestro propio gate.

---

## R3 · Contra buscadores sin memoria, la neurona gana a todos

`results/search_ladder/result.json` · contrato
`docs/ENMIENDA_ESCALERA_COMPARADORES_2026-08-05.md` · `B = 24`, mismo CRN, lecturas **enforzadas**
(`Surface.value_of_visited` lanza `LookupError`, no se afirma)

| brazo | AUC de regret |
|---|---:|
| oráculo (techo) | 0,00000 |
| **neuron_memory** | **0,04975** |
| ucb1 | 0,09850 |
| ofat | 0,10024 |
| neuron_reset | 0,10067 |
| gp_ei (**optimización bayesiana**) | 0,10862 |
| lhs_local | 0,11836 |
| random | 0,14516 |
| annealing | 0,16484 |

Los siete contrastes con LCB95 > 0. **Pero `neuron_reset` (0,10067) se sienta junto a `ofat`
(0,10024) y `gp_ei` (0,10862)**: sin memoria la neurona es un buscador del montón. Ese titular
compara **un buscador con memoria contra buscadores sin memoria**, que es casi una tautología.

---

## R4 · Con memoria para todos, el ingrediente es la retención

`results/search_ladder_v2/result.json` · contrato
`docs/ENMIENDA_ESCALERA_TRANSFERENCIA_2026-08-05.md`

Los tres clásicos más fuertes reciben **exactamente la información de la neurona** —observaciones de
contextos previos, normalizadas por prefijo dentro de cada contexto, **sin etiqueta de contexto**:

| brazo | AUC |
|---|---:|
| **ucb1_transfer** | **0,04253** |
| neuron_memory | 0,04975 |
| ofat_transfer | 0,06609 |
| gp_ei_transfer | 0,08366 |
| … sin memoria | 0,0985 – 0,1648 |

**`ucb1_transfer` empata o supera a la neurona**: −0,00721 **[LCB95 −0,01772]**, el intervalo cruza
cero con el punto estimado en contra. Y **el OFAT de Garrido con memoria (0,06609) bate a la
optimización bayesiana sin memoria (0,10862)**.

**El valor de la memoria, por familia** (gemelo sin memoria menos su versión con memoria):

| familia | ganancia | LCB95 |
|---|---:|---:|
| ucb1 | +0,0560 | +0,0437 |
| neurona | +0,0509 | +0,0392 |
| ofat | +0,0342 | +0,0264 |
| gp_ei | +0,0250 | +0,0164 |

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

## Lo que falta para cerrar el paper

**La transferencia de rejilla 288 → 4.608**, el único eje de transferencia vivo tras `g1`.
Contrato escrito (`docs/ENMIENDA_TRANSFERENCIA_REJILLA_2026-08-05.md`) y runner implementado
(`scripts/run_grid_transfer_v1.py`); la superficie extendida está construyéndose.

**No requiere firma**: la extensión añade **configuraciones, no cintas**, así que corre sobre el
mismo bloque quemado como réplica declarada. No hay nada que abrir.
