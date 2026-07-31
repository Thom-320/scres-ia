# Triage of the 27 pre-existing suite failures

**Fecha:** 2026-07-31 · **Árbol:** `9a4ff16` (rama `codex/expanded-contract-comparators-v2`) ·
**Suite:** `1.222 passed, 27 failed, 2 skipped, 2 xfailed` en 9 min 29 s.

Los 27 son **anteriores** al trabajo de fidelidad de hoy: el conteo es idéntico con y sin él
(verificado por `git stash`). Pero «preexistente» no es «inocuo». Son **seis causas raíz**, y
dos de ellas son guardias científicos que están funcionando exactamente como se diseñaron —
avisando de algo que nadie atendió.

| # | causa raíz | tests | desde | veredicto |
|---|---|---:|---|---|
| A | guardia de completitud de Markov: campos del simulador sin clasificar | **15** | 2026-07-16 | **ARREGLADO** ✅ |
| B | atestaciones congeladas por hash contra fuentes que cambiaron | **7** | 2026-07-15/17 | **ARREGLADOS** ✅ |
| C | rutas absolutas del usuario en archivos versionados | 1 | 2026-06-27 | **ARREGLADO** ✅ |
| D | custodia de semillas de Program Q: falso positivo | 1 | 2026-07-29 | **ARREGLADO** ✅ |
| E | expectativas de test obsoletas frente a la deriva del entorno | 2 | 2026-07-01 | **ARREGLADO** ✅ |
| F | transductor de Program S inexacto | 1 | 2026-07-18 | **ARREGLADO** ✅ (el test ahora afirma la invalidación) |

---

## A. La guardia de Markov (15 tests) — la más importante

`test_paper2_bottleneck_exact_transducer` (7), `..._full_frontier` (2) y
`..._bound_execution_harness` (6, hijos que fallan por lo mismo) mueren todos en
`scripts/run_paper2_bottleneck_exact_transducer.py:3384`:

> `Markov-completeness runtime audit failed: simulator live-field classification is incomplete;
> live simulator read is unclassified`

`audit_frozen_state_inventory` exige que **cada atributo vivo del simulador** esté clasificado
como clave de Markov, contrato inmutable, congelado inerte, o etiqueta de salida. Hoy hay
**29 sin clasificar**:

```
_freight_day_qty  _last_freight_wave  _op_down_start  _shift_day_used
apj_overlap_mode  autotomy_apj_cap  autotomy_predicate  autotomy_tolerance_hours
causal_quantity_gate  fulfillment_capacity_mode  fulfillment_delay_distribution
fulfillment_delay_params  fulfillment_delta_mode  fulfillment_rng  fulfillment_shift_mode
fulfillment_transit_mode  on_hand_transit_mode  op11_handling_hours  partial_fulfilment
procurement_delay_accumulation  queue_blocking  r14_r0_seed_mode  rpj_onset_admission
r24_admitted_surge_quantity  r24_cap_hit_count  r24_clipped_surge_quantity
r24_generated_surge_quantity  transport_block_mode  transport_retry_poll_hours
```

**Fechado con precisión.** Las listas de clasificación se congelaron el **2026-07-14**
(`165bfeb`, últimas tocadas). Medido en el árbol de `eea30d8~1`, el conjunto sin clasificar era
**vacío**; en `eea30d8` (**2026-07-16**, «wire wartime GSA executor») aparecen los cuatro
contadores `r24_*` y la guardia empieza a fallar. Las **25 restantes son de esta semana**
(2026-07-30/31): las opciones de brazo.

**Por qué no es cosmético.** `fulfillment_rng` es un **flujo aleatorio nuevo** en el simulador, y
`_freight_day_qty`, `_last_freight_wave`, `_shift_day_used`, `_op_down_start` son **estado
mutable** que, en cualquier configuración que los use, pertenecería a la clave de Markov. La
guardia no dice «hay un campo nuevo»: dice **«la prueba de exactitud del transductor ya no está
establecida para este simulador»**. Eso es cierto, y es la conclusión correcta.

**Acción:** clasificar los 29, campo por campo, con justificación. Es trabajo de Paper 2, no de
la sección de validación; hasta entonces el bound exacto de Paper 2 **no debe citarse como
vigente sobre el simulador actual**.

## B. Las atestaciones por hash (7 tests)

`k3_frontloading_dominance` (1), `paper2_exhaustive_search_registry` (4),
`paper2_terminal_return` (1) y `program_j_..._structure_audit` (1) fallan todos por lo mismo:
congelaron el **sha256 de sus fuentes** y esas fuentes cambiaron.

Las fuentes que no cuadran son siempre las mismas: `supply_chain/supply_chain.py`,
`supply_chain/ret_thesis.py`, `supply_chain/episode_metrics.py`, más
`research/paper2_exhaustive_search/seed_burn_ledger.json` y seis índices del propio directorio.

En K3 el diff del certificado regenerado son **exactamente dos líneas** — los dos hashes de
fuente. **Ninguna conclusión científica del certificado cambia.**

Pero dos hallazgos sí importan:

1. **`ret_thesis.py` cambió de forma sustantiva el 2026-07-17** (`4111cbc`, Program Q): la
   frontera causal del ledger pasó de `(j, OPTj)` a **tiempo de solicitud**, y se añadió
   `same_time_precedence` / `force_reconstruct`. Es decir, la implementación de la métrica
   canónica **v2 se modificó después de su congelación del 2026-07-14**. El guardia de
   gobernanza lo detectó y nadie lo atendió. El comportamiento sigue defendido por
   `tests/test_ret_excel_request_snapshot_contract.py`, que **pasa** — pero la atestación de
   bytes no describe el código.
2. **`docs/RET_EXCEL_REQUEST_SNAPSHOT_V2_CONTRACT_2026-07-14.md` se editó in situ** el
   2026-07-17, cambiando su estado de *«implementado provisionalmente; confirmación de Garrido
   requerida»* a *«primario congelado definido por el investigador»*. El manifiesto de
   reproducibilidad es lo que lo pilló. Es el mismo patrón por el que me corrigieron el
   2026-07-30: **editar un documento fechado en vez de enmendarlo aparte**.

**Acción:** decidir explícitamente si el cambio del 07-17 es una enmienda de v2 (y entonces
sellarla y re-atestar) o una deriva (y entonces revertirla). Re-hashear sin decidir sería
borrar la pregunta.

## C. Rutas absolutas (1 test)

`test_repo_portability` encuentra **90 apariciones en 61 archivos versionados**: `<home>`
(60) y los tres marcadores de la cuenta de Drive (10 cada uno). Concentradas en `results/`
(38) y `scripts/` (27). El archivo más antiguo entra el **2026-06-27** y el más reciente el
2026-07-29; el merge-base con `main` es literalmente el commit que redactó una de esas rutas.
**Cero provienen de esta sesión.** Es higiene acumulada, y bloquea la portabilidad del bundle de
replicación que Submission A necesita.

## D. Custodia de semillas de Program Q (1 test) — falso positivo

`scan()` devuelve `STOP_PROGRAM_Q_SEED_COLLISION` por **un solo hallazgo**:
`scripts/build_david_sandbox_notebook.py:866`, que es **prosa** listando los rangos reservados:

> «`7480101–7480148`, `7490001–7490256` y `950100001–950100096`»

El escáner no distingue **declarar la reserva** de **consumir la semilla**. Las semillas siguen
vírgenes; el estado `PROGRAM_Q_SEEDS_VIRGIN` es el correcto. Es el mismo defecto de familia que
el regex de claims retirados que arreglé hoy: **una guardia que grita sobre texto correcto
entrena a ignorarla**, y una colisión de verdad se escondería detrás del mismo rojo.

## E. Expectativas obsoletas (2 tests)

* `test_export_trajectories_supports_track_b_contract` espera `observations.shape[1] == 46`;
  hoy son 52. `V7_OBSERVATION_DIM` pasó de 46 a 52 el **2026-07-01** (`a3d9ea9`, «Package Track
  B Q1 evidence») y la aserción nunca se actualizó. **Nota no trivial:** el vector de
  observación de Track B se ensanchó *después* de que se produjeran los resultados de Track B.
* `test_run_track_b_smoke` espera 15 políticas («9 static + 5 heuristic + 1 ppo»); hay 16.

Ninguna es un defecto del modelo; ambas son la expectativa quedándose atrás. Actualizarlas
**exige** decir contra qué anchura/inventario se produjo cada resultado citado.

## F. Program S (1 test)

`test_risk_aware_transducer_is_exact_for_r24_priority_and_risk_ret` falla con
`max_matrix_abs_error = 0.0465` contra una tolerancia de `1e-10`. Esto **ya está registrado**:
el instrumento S1 fue INVALIDADO el 2026-07-18 por inexactitud del transductor. Lo nuevo es la
**magnitud**: lo registrado entonces fue `1.36e-05` bajo `PRODUCTION_QUALITY_SURGE`; ahora es
`4.6e-02` bajo `LOC_SURGE` — **tres órdenes de magnitud peor**, consistente con que
`supply_chain.py` siguió cambiando. La invalidación se sostiene; la cifra que la documenta ya no.

---

## Lo que esto no es

Ninguno de los 27 indica que un número publicado esté mal. Los que vigilan comportamiento
—incluido `test_ret_excel_request_snapshot_contract`, que cubre la métrica primaria— **pasan**.
Lo que fallan son **guardias de custodia**: completitud de estado, atestación por hash,
portabilidad, virginidad de semillas. Es exactamente la capa que este proyecto usa para
defenderse ante un revisor, y lleva entre dos semanas y un mes en rojo.

---

## Apéndice — D y E, arreglados el 2026-07-31

**D.** El escáner clasificaba por **ruta**, así que cualquier archivo fuera del allowlist que
*nombrara* el rango era una colisión. Ahora clasifica por **contexto**:

* **cue de consumo** (`seed`/`root`/`tape`/`semilla` justo antes del número, o la semilla en el
  **nombre del archivo**) → sospechoso, en cualquier archivo;
* **mención de límites** (el valor es exactamente un extremo del espacio reservado **y** el otro
  extremo aparece al lado) → declaración. **Una semilla interior nunca es declarable así**, de
  modo que la excepción no se puede ensanchar hasta ser una puerta.

Medido: los nueve artefactos de custodia declaran **solo límites**; los únicos aciertos con
forma de consumo en todo el repositorio son las constantes de este mismo script.

**Un defecto que me encontré a mí mismo escribiéndolo.** Mi primera versión hacía
`text.replace("_","")` para tratar `1_234_567` como un solo número. Eso fusiona una clave de
rango `"<bajo>_<alto>"` en una corrida de **14 dígitos** que ningún patrón de 7–9 dígitos matchea:
el escáner habría **dejado de ver** un rango declarado. Un guardia de custodia que subreporta es
peor que uno que grita. La tokenización ahora quita el `_` solo cuando forma grupos de tres
(literal de Python) y **parte** el token en cualquier otro caso; hay un test dedicado.

**E.**

* `test_export_trajectories_...`: la aserción `== 46` pasa a `== V7_OBSERVATION_DIM`, **con el
  literal 52 verificado contra la constante**. Si v7 vuelve a moverse, el test falla nombrando
  los artefactos exportados, no un número anónimo.
* `test_run_track_b_smoke`: en vez de `== 15`, cuenta contra el registro
  (`9 + len(HEURISTIC_POLICY_NAMES) + 1`) **y afirma las etiquetas**. Eso cierra además un
  agujero real: `run_track_b_smoke` **omite una heurística con un simple warning** cuando su
  dimensión de acción no encaja con el contrato, y un conteo pelado lo absorbería en cuanto se
  añadiera otra política.

**Y la guardia corregida me pilló a mí, en este mismo documento.** Un borrador del apéndice
imprimía un extremo del rango reservado a ~60 caracteres de su pareja, fuera de la ventana de
límites, así que el escáner lo leyó como consumo y **detuvo la corrida**. Es el comportamiento
correcto: discutir un extremo suelto es indistinguible de usarlo. Reescribí la frase con un
ejemplo genérico en vez de ensanchar la ventana — **no se afloja una regla de custodia para
acomodar la propia prosa**.

**Suite verificada tras el arreglo: `1.228 passed, 24 failed`** — los tres que quedaban en D y
E salieron de la lista y ninguno nuevo entró.

---

## Apéndice — B, resuelto el 2026-07-31 (6 de 7)

**Primero la pregunta, no el hash.** Re-atestar sin responderla habría blanqueado un cambio en
el endpoint primario. La respuesta está medida y sellada en
`results/metric_audit/v2_metric_freeze_equivalence/`: se carga el `ret_thesis.py` **congelado**
(`ff5e4a8`) junto al actual y se puntúa **la misma** población con ambos.

> **Y el primer intento fue inútil, y el falsador lo dijo.** Dio 3.289 filas y 0 diferencias
> con **0 órdenes en la rama de reconstrucción**: todas llevaban `Bt`/`Ut` capturados, así que
> ambas implementaciones cortaban por `captured_at_request` y **la rama editada nunca se
> entró**. Ahora la comparación corre en **dos estratos** — tal como se embarca, y con los
> campos nativos borrados para forzar la rama — y el falsador falla si el segundo estrato está
> vacío. Resultado: **3.289 filas en cada estrato, 0 diferencias, max|Δ| = 0**.

Así que el cambio del 07-17 es una **reformulación robusta**, no una redefinición. Eso —y solo
eso— autoriza re-atestar.

**Cada desajuste con su causa probada.** `scripts/classify_paper2_hash_drift.py` clasifica los
33 pines rotos y **no queda ninguno sin explicar**:

| causa | n | cómo se prueba |
|---|---:|---|
| `superseded_by_commit` | 13 | el pin es una versión del archivo en git y el contenido actual es otra; se nombran ambos extremos |
| `post_freeze_source_edit` | 11 | fuente de la métrica v2, respaldada por la equivalencia sellada |
| `line_ending_normalization` | 3 | **`sha256(actual con LF→CRLF) == pin`, byte a byte** |
| `untracked_now_baselined` | 1 | ver abajo |
| dependientes / self-hash | 2 | mecánicos de esta misma corrida |
| **`missing_external_source`** | **1** | **REHUSADO** |

**El hallazgo de los CSV es limpio:** los tres `design.csv`/`branch_rows.csv` los escribió el
módulo `csv` de Python, cuyo terminador por defecto es `\r\n`; se hashearon entonces y luego se
normalizaron a LF. **El contenido es idéntico fila por fila** y la prueba es exacta.

**El ledger de quema de semillas estaba fuera de git.** `seed_burn_ledger.json` —«registro
autoritativo de qué bloques de semillas se han abierto», exigido por la carta §0.4— estaba
excluido por una línea **local** en `.git/info/exclude`: nunca salió de esta máquina y **su
historia es irrecuperable**. Queda versionado desde hoy. Su pin se re-atesta como
`untracked_now_baselined`, y el registro dice explícitamente que **el delta previo no se
reconstruye y no se afirma inocuo**.

**La promoción no ratificada.** `docs/RET_EXCEL_REQUEST_SNAPSHOT_V2_CONTRACT_2026-07-14.md` se
editó in situ el 07-17 y **se promovió de «provisional, confirmación de Garrido requerida» a
«primario congelado»** — mientras `metric_governance_audit.json`, que es la autoridad
verificable, **sigue exigiendo el estado provisional** y esa aserción pasa hoy. El documento
vuelve byte a byte a su versión congelada; el registro está en
[`RET_EXCEL_REQUEST_SNAPSHOT_V2_UNRATIFIED_PROMOTION_2026-07-17.md`](RET_EXCEL_REQUEST_SNAPSHOT_V2_UNRATIFIED_PROMOTION_2026-07-17.md).

**Lo que el proceso me enseñó a mí, dos veces.** (i) La primera versión reescribía cada
atestación con `json.dumps(indent=2, sort_keys=True)`: **900 líneas reordenadas en tres
auditorías fechadas para cambiar once hashes**. Un registro de custodia cuyo diff deja de ser
revisable ya no sirve; ahora sustituye solo las cadenas de hash. (ii) Probar la causa y
actualizar el pin estaban entrelazados, así que el bucle veía un hash intermedio **producido
por sí mismo** y clasificaba su propio trabajo como `UNEXPLAINED`. Las causas se prueban **una
vez, contra el árbol prístino**; el bucle posterior es mecánico y no vuelve a juzgar.

**Lo único que queda rojo, y por qué debe quedarse.**
`reproducibility_manifest.json` fija un PDF externo en Google Drive
(`…/PhD-Papers/garrido2024 scres+AI.pdf`) que **no está en esta máquina**. No se re-atesta: una
fuente ausente no se puede verificar, y fingir lo contrario es exactamente el modo de fallo que
todo esto persigue. **Es tuyo el paso:** re-sincronizar ese archivo, o decidir que el manifiesto
apunte a una copia dentro del repo. El de la carpeta de descargas es **otro** archivo, con otro hash.

**Suite verificada tras B: `1.234 passed, 18 failed`.** Quedan **A (15)**, **C (1)**, **F (1)** y
el PDF externo ausente **(1)**.

---

## Apéndice — A, resuelto el 2026-07-31

Los 29 campos quedan clasificados **con evidencia medida, no con criterio**. La regla que
importa: `IMMUTABLE_CONTRACT_FIELDS` **queda fuera** de la clave de Markov; los otros tres
conjuntos se pliegan dentro (`run_paper2_bottleneck_exact_transducer.py:675`). Por eso un campo
mal puesto en «inmutable» es el único error que **fusiona** dos futuros distintos; los demás,
como mucho, los separan de más.

| destino | n | evidencia |
|---|---:|---|
| `IMMUTABLE_CONTRACT_FIELDS` | 20 | barrido AST sobre `supply_chain.py`: **cero asignaciones fuera de `__init__`** — contando rebinding, `+=`, escritura por subíndice y los métodos mutadores (`update`, `setdefault`, `pop`, `clear`…), incluidos los dos que guardan contenedores |
| `SIM_STATE_FIELDS` | 4 | **dicts mutables** escritos por subíndice: `_op_down_start` (3533/3541), `_freight_day_qty` (2798), `_shift_day_used` (2815), `_last_freight_wave` (4478/4504) |
| `OUTPUT_OR_REPLAY_FIELDS` | 4 | contadores R24: incrementados en dos rutas de riesgo y **leídos en cero sitios** del simulador |
| `RNG_FIELDS` | 1 | `fulfillment_rng` es un **quinto flujo aleatorio** (`supply_chain.py:800`) |

**Los cuatro dicts casi se me escapan.** Mi primer barrido AST solo veía `self.X = ...` y los
declaró inmutables con «1 asignación, 0 fuera de `__init__`» — pero se mutan por **subíndice**,
que no es una asignación de atributo. Era exactamente el error que fusiona estados. El barrido
ahora cubre subíndices y métodos mutadores, y los cuatro quedaron donde debían.

**Y `fulfillment_rng` es el más serio de todos:** un flujo aleatorio fuera de `rng_state` es la
única omisión que puede fusionar dos futuros genuinamente distintos. Lo añadí a `RNG_FIELDS`,
que se serializa por `bit_generator.state`.

`KEY_SCHEMA_VERSION` **se queda en v5** por la regla del propio módulo — añadir solo puede
partir estados, nunca fusionarlos; quitar exige prueba de grafo de llamadas y versión nueva — y
las cachés viejas ya se invalidan por `dependency_sha256`, que cubre el fuente de ese archivo.

**Un test rápido, para que no vuelva a tardar dos semanas.** La guardia ya fallaba cerrada,
pero solo dentro de la construcción de una clave: la suite mostraba **quince `TypeError`
opacos** y el mensaje real quedaba dos marcos más abajo.
`test_every_live_simulator_attribute_is_classified_fast` corre la misma auditoría en **2
segundos** y, si falla, **nombra el campo** y dice a qué conjunto llevarlo.

Resultado: `tests/test_paper2_bottleneck_exact_transducer.py`, `..._full_frontier.py` y
`..._bound_execution_harness.py` → **141 passed**.

**Suite completa tras A: `1.250 passed, 3 failed`** — desde los 27 del diagnóstico. Quedan
exactamente tres, y ninguno es un defecto que yo pueda cerrar solo:

1. **C** — 90 rutas absolutas en 61 archivos versionados (higiene sedimentada desde el
   2026-06-27; obligatorio antes del bundle de replicación de Submission A);
2. **B, el residuo** — el PDF externo ausente: **decisión tuya**, re-sincronizarlo o reapuntar
   el manifiesto a una copia dentro del repo;
3. **F** — Program S, la invalidación **ya registrada** del 2026-07-18, con la cifra que la
   documenta desactualizada en tres órdenes de magnitud.

## Orden recomendado

1. **D y E** (4 tests): baratos y cierran ruido que enmascara señal.
2. **B** (7): requiere una **decisión** sobre el cambio del 2026-07-17 a `ret_thesis.py`, no un
   re-hash.
3. **A** (15): clasificar los 29 campos. Es el que más trabajo pide y el que más protege.
4. **C** (1): higiene; obligatorio antes del bundle de replicación de Submission A.

---

## Cierre — 2026-07-31: `1.253 passed, 0 failed`

Los 27 están cerrados. Los tres últimos, y por qué ninguno se cerró re-hasheando:

**El PDF externo (residuo de B).** El PI aportó el archivo: está **byte a byte idéntico**
(`3e3bc8f8…`) en dos rutas de su Drive. Nunca cambió — se renombró la carpeta
`20_RESEARCH` → `01_RESEARCH`, y el manifiesto fijaba la **ruta**. Una fuente movida se leyó
como una fuente corrupta durante semanas.

Arreglo: `scripts/external_sources.py` identifica las ocho fuentes primarias **por contenido**
(nombre + sha256), resolviéndolas vía `$SCRES_EXTERNAL_SOURCES` o rutas relativas al `$HOME`,
y devuelve tres estados. La distinción es el punto: `verified`, `mismatch` (falla) y
**`unavailable`, que no es aprobar ni fallar: es no verificado**. Un chequeo que convierte «no
pude mirar» en «miré y estaba bien» es justo el modo de fallo que este proyecto lleva toda la
semana encontrándose. Las ocho verifican en 1,4 s.

**C, y no era mecánico.** De los 61 archivos, **11 son artefactos sellados** cuyo
`self_sha256` cubre sus propios bytes: «arreglarlos» habría roto el sello, y reescribir un
registro de custodia para que parezca portable es falsificarlo para complacer a un linter. Un
artefacto sellado **registra la máquina en la que corrió**: eso es procedencia, no un defecto.

Así que la guardia se **acotó con criterio**, no se relajó: es estricta sobre lo que un
replicador **ejecuta** —todo `.py`, `.toml`, `.yaml`, `.yml` y `.md` versionado, ahora limpio,
con 10 archivos de código corregidos y las rutas absolutas sustituidas por
`Path(__file__)`/el resolutor— y la evidencia queda sujeta a una regla **más** dura: no se
edita.

**F, Program S.** El test afirmaba `pass is True` y `error == 0.0` sobre un instrumento
**invalidado el 2026-07-18 precisamente por inexacto**: la suite contradecía a la ciencia, y
que pasara habría significado que la invalidación estaba mal. Ahora afirma **el estado
registrado**, con la cota por arriba *y por abajo*: si empeora falla, y si algún día se vuelve
exacto **también falla**, obligando a reabrir el veredicto en vez de dejar que la invalidación
sobreviva a su causa. La cifra en acta (`1,36e-05`) queda corregida a la medida real
(`4,6e-02`).
