# Deuda cerrada el 2026-08-08 — supersesión, división de Q1, parches de P2

Tres tareas acotadas que quedaban de las auditorías. Qué se hizo, qué encontró cada una, y qué
**no** se hizo.

## 1. Registro de supersesión legible por máquina

`research/supersession_registry.json`, generado por `scripts/build_supersession_registry_v1.py`.

**El problema real no era que no se etiquetara.** La regla del repo —un resultado retirado se
conserva y se etiqueta— se venía cumpliendo. Lo que pasaba es que la etiqueta se escribía donde
cayera: un barrido de `results/*/result*.json` encuentra la misma relación bajo `supersedes`,
`predecessor`, `predecessors`, `superseded_null`, `supersedes_for_multiplicity`,
`transform_family_supersedes_the_ceiling` y un `retraction` en texto libre. **Eso no lo lee nadie**,
ni un revisor que no conozca el vocabulario ni un agente dentro de cinco semanas.

17 aristas, 0 problemas. Cada relación lleva **su regla de lectura**, porque «superado» a secas no
dice si el número viejo se puede seguir citando:

| relación | qué permite |
|---|---|
| `SUPERSEDED_BY_FAILED_REPLICATION` | conservado como registro de lo que se creyó; su número **no** es una estimación viva |
| `SUPERSEDED_BY_CORRECTIVE_RERUN` | citable sólo como el número defectuoso, junto al defecto |
| `SUPERSEDED_IN_PART` | sigue citable, pero **sólo junto a su sucesor** y nunca por el componente reemplazado |
| `VOIDED_OF_OBJECT` | internamente sano, pero lo que medía fue retirado: ni positivo ni negativo |
| `PREDECESSOR_IN_A_VERSION_CHAIN` | el sucesor es la versión citable; el predecesor no es evidencia independiente |
| `LINEAGE_NOT_SUPERSESSION` | construido **sobre** el artefacto, pero responde otra pregunta: nada se retira |

**Encontró un hueco real y encontró un defecto mío.** El hueco:
`H_REGIME_MUST_BE_LABELLED_BY_METRIC` citaba números —0,27764, la familia de 661 transformaciones,
los 29 que pasan— que viven en `monotone_transform_family_v4`, **artefacto que ninguna fila del
claim lock citaba**. Una supersesión parcial cuyo sucesor no aparece en el registro se lee como
ninguna supersesión. Corregido con una fila nueva, `H_REGIME_IS_NOT_CURVATURE_INVARIANT`, y su
`must_be_cited_with`.

El defecto mío: leí `predecessor` como supersesión. En `monotone_transform_family_v4` significa «v3
fue reemplazado»; en `citable_risk_attitudes` significa «construido sobre», y ahí no se retira nada.
El nombre del campo no los distingue, así que los distingue el linaje del directorio. **Sub-declara
a propósito**: `monotone_transform_ceiling → monotone_transform_family_v2` es una cadena real que
esta regla archiva como linaje, porque un registro que **inventa** una supersesión es peor que uno
que se pierde un renombrado.

**Los controles están en la suite**, no en la buena intención: `tests/test_supersession_registry.py`
reintroduce cada defecto y exige que el problema aparezca —una arista curada sin su documento, un
artefacto plenamente superado citado como evidencia viva, la compañera ausente en una supersesión
parcial. Este repo ya publicó una fuga de datos real bajo un `passed: True` fijo; la lección no fue
«ten cuidado», fue «que el control forme parte de la suite».

## 2. La Q1 de Garrido son dos preguntas, y sólo una tiene respuesta

En `papers/paper2/claim_lock.json`: `q1_scopes`, `q1_scope_index` y un `q1_scope` por fila, con un
validador que **falla si una fila no lo declara**.

* `Q1_SEARCH_TRANSFER` — el bucle **externo**: estado retenido **entre** corridas de la búsqueda.
  Es el cierre de la Fig. 2 y el efecto Alzheimer. **16 filas.** Aquí vive todo lo positivo.
* `Q1_OPERATIONAL_ADAPTATION` — el bucle **interno**: una decisión condicionada al estado **dentro**
  del episodio. **5 filas, todas negativas o no replicadas.**
* `INSTRUMENT_OR_SCOPE` — ni una ni otra: reproducción, validación, etiquetado de métrica. **7 filas.**

**Por qué hoy y no antes.** Mientras ambas cosas se archivaban como «Q1», un lector podía deslizarse
de una a otra sin notarlo. El 2026-08-08 eso dejó de ser cosmético: lo que replicó fue lo primero y
lo que murió fue lo segundo. Un registro que no las distingue reporta eso como un empate — o deja
que la mitad superviviente cubra a la mitad muerta.

`H_regime` va del lado de **adaptación operativa**, y no es un detalle: pregunta si condicionar la
**elección** al régimen paga. Eso es adaptación al estado, no arrastre de estado entre corridas, y
pertenece al lado que sigue volviendo vacío.

Fila nueva: `OPERATIONAL_ADAPTATION_CEILING_DID_NOT_REPLICATE`. Es **la única del archivo cuyo
falsador debe fallar** — f5 pedía que el techo replicara, no replicó, y ese fallo *es* el hallazgo.
Sus frases prohibidas incluyen las seis maneras de leer la tabla de rasgos que la regla congelada
prohíbe, entre ellas «la búsqueda de señal tenía poca potencia», que **invierte** el resultado: 24
tapes de entrenamiento es el diseño grande, y lo que muestra es que no había techo que capturar.

## 3. Parches del manuscrito de P2

Aplicados:

* **Renombrado del comparador.** «state-blind replay» → «online cumulative replay» en introducción,
  métodos y resultados. La etiqueta vieja era falsa dos veces, y lo importante es lo segundo: ese
  replay **no es fijo**, el histograma se acumula a lo largo de la corrida. Ahora se nombra por lo
  que hace, no por lo que le falta. La mención histórica en §*The comparator, repaired* se conserva
  entrecomillada y marcada como retirada.
* **El empate post-hoc deja de reportarse como empate.** «mutually indistinguishable under Holm»
  → un patrón de **ranking**, con la lectura i.i.d. explícitamente en cuarentena porque los replays
  se acumulan sobre casos ordenados y las sesenta semillas no son réplicas intercambiables.
* **AUC frente a arrepentimiento final, en el abstract.** El endpoint se nombra y se dice por qué
  importa: bajo el arrepentimiento simple de la recomendación efectivamente desplegada en
  presupuesto 24, los seis contrastes mantienen el signo pero la resolución conjunta **colapsa a uno
  de seis**, y el orden de familias no se preserva. AUC premia llegar antes; el final sólo pregunta
  dónde acabaste.
* **Cita a `comparator_repair_v2`** en vez del predecesor (contrastes idénticos; v2 añade la
  preregistración y el digest del prior congelado por semilla).
* **«a transportable visit prior»** → «este prior congelado definido por el investigador», que es lo
  único medido.
* **La demanda «casi determinista»** queda atribuida a la literatura circundante y rechazada, en vez
  de aparecer como caracterización propia.
* **§4.2 nueva, «Two loops, one question, one answer»**, y el resultado de hoy entra en §3.7 como
  **ausencia medida**, con su número y con la declaración de que la tabla de rasgos no se leyó.

**No hecho, y es decisión tuya:** reducir el manuscrito a 8–10 claims. Eso cambia lo que el paper
**afirma**, no cómo lo dice, y hacerlo de tapadillo mientras arreglaba vocabulario habría sido
colar una decisión de alcance dentro de una tarea de higiene. El claim lock, además, va en la
dirección contraria por una razón buena: pasó de 26 a 28 filas porque dos citas no tenían fuente.
Son dos contadores distintos —filas del registro de citas frente a afirmaciones centrales del
manuscrito— y conviene no confundirlos al decidir.

El audit sigue en `BLOCKED_MANUSCRIPT_LOCK_MISMATCH` por dos avisos que son **menciones
deliberadas** dentro de sus propios descargos, más el aviso de trazabilidad de que los `claim_id` no
aparecen literalmente en la prosa. Ninguno es un fallo científico; el propio audit lo dice en sus
notas.
