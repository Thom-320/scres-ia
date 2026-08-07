# Enmienda — un registro de evidencia por experimento, no por fichero

**Escrita ANTES de construirlo.** Constructor: `scripts/build_evidence_registry_v1.py`.
Salida: `research/evidence_registry.jsonl`. Sólo lectura sobre artefactos; cero semillas.

## 1. El problema que resuelve

Un barrido de todo el repositorio contó **439 intervalos positivos** y eso pareció abundancia de
evidencia. La mayoría son **dos familias de cintas re-analizadas bajo diez nombres de directorio**.

> Contar `result.json` es contabilidad, no replicación. Contar `claim_status` es peor: es una
> ranura de esquema que un programador de junio pudo omitir y uno de julio pudo sobreafirmar.

## 2. La clave de deduplicación, que es todo el diseño

```
(contract_sha256, execution_fingerprint, seed_block_signature, endpoint, estimand_family)
```

* `execution_fingerprint` = hash del `module_manifest` (script de entrada + módulos declarados),
  no del `result.json`.
* `seed_block_signature` = el **rango**, no la lista: dos corridas sobre el mismo rango son la
  misma unidad de custodia.

**Las filas con clave INCOMPLETA no se fusionan nunca.** Un campo ausente no es una coincidencia, y
tratar `null == null` como igualdad fusionaría corridas no relacionadas — que es exactamente el
fallo que este registro existe para terminar.

## 3. El grado se DERIVA, no se copia

`claim_status` es la etiqueta de un autor. El grado sale de hechos comprobables:

| grado | condición |
|---|---|
| `CONFIRMATORY` | rol de confirmación **y** bloque que el registro de custodia reconoce como abierto virgen para ella |
| `CONFIRMATION_ROLE_WITHOUT_VIRGIN_BLOCK` | dice confirmación, pero su bloque no lo respalda |
| `NEGATIVE_OR_HALTED` | su propio estado terminal es un stop o una refutación |
| `DIAGNOSTIC` · `REPLAY` · `DEVELOPMENT` | según alcance declarado |
| `UNCONTRACTED` | **sin hash de contrato**: nada fija qué podía reclamar |

> **Un artefacto que dice `CONFIRMED` sin bloque virgen no llega a confirmatorio.**

## 4. Falsadores, con por qué cada uno puede fallar

| falsador | por qué puede fallar |
|---|---|
| `f1_the_key_collapses_a_known_duplicate_family` | la familia del meta-aprendiz son muchos directorios sobre dos juegos de cintas. **Si la clave los deja todos distintos, está contando ficheros otra vez** |
| `f2_the_key_does_not_over_merge_different_blocks` | las rebanadas local y VPS de H3 son bloques **distintos** (`6000001-90` y `6000091-120`). Fusionarlas borraría la independencia de la que depende la fusión a n=120 |
| `f3_incomplete_keys_are_never_merged` | falla si alguna fila con clave incompleta acabó marcada como duplicada |
| `f4_the_grade_is_derived_not_copied` | **si ningún artefacto autorado como `CONFIRM` recibe otro grado, el grado sólo repite `claim_status`** y no certifica nada |
| `f5_off_head_artifacts_are_included` | un glob sobre el árbol de trabajo no ve otras ramas, **y dejarlas fuera es como el censo concluyó que el proyecto tenía una sola confirmación** |

## 5. Alcance

Índice. **No adjudica nada, no cambia ningún número y no toca ningún artefacto.** Lo que habilita
es que la tabla canónica, el registro de lanes y el inventario de semillas se **generen** desde
aquí en vez de escribirse a mano — que es como acumularon cifras rancias.
