# Tabla canónica — **generada**, no escrita a mano

Generada `2026-08-07T22:57:18+00:00` desde `research/evidence_registry.jsonl` y `research/seed_custody_registry.json` por `scripts/generate_canonical_table_v1.py`.

> **No editar a mano.** Seis auditorías seguidas encontraron cifras rancias en tablas mantenidas a mano. Lo contable se genera; **la interpretación no**, y vive en los documentos enlazados abajo.

## Procedencia

| rama científica | `8ddf6f7` |
|---|---|
| `main` | `89acc81` |
| adelante / detrás | **823** / 8 |

## Evidencia, por grado derivado

216 artefactos leídos · 5 colapsados como re-reporte · **211 experimentos distintos**.

| grado | n |
|---|---:|
| `DEVELOPMENT` | 87 |
| `UNCONTRACTED` | 57 |
| `REPLAY` | 39 |
| `NEGATIVE_OR_HALTED` | 20 |
| `DIAGNOSTIC` | 5 |
| `CONFIRMATORY` | 3 |

El grado **se deriva** de hechos comprobables (rol de confirmación, bloque de custodia, presencia de contrato), nunca del `claim_status` que escribió un autor.

## Confirmatorios

| artefacto | autorado como | bloque |
|---|---|---|
| `results/garrido_h2_h3_confirmation_v1/result.json` | `CONFIRM_H2_H3_ALL_SIX_PANELS` | `96111336-97836128` |
| `results/grid_transfer_confirmation_v2/result.json` | `GRID_TRANSFER_CONFIRMED__UCB1` | `8200001-8200060` |
| `results/gsa_confirmation/result.json` | `GSA_CONFIRMED_ON_VIRGIN_BLOCK` | `7700001-7700120` |

## Custodia

Estado del registro: `BASELINE_INVENTORY_INCOMPLETE_NO_NEW_SEEDS_AUTHORIZED` · `new_seed_opening: False`

**Bloques vírgenes restantes: 0** — no queda ninguno; ninguna confirmación más es posible sin autorizar semillas nuevas.

Quemados por confirmación completa: 3 (g3a_v2_development, garrido_grid_transfer_v2_confirmation, garrido_h2_h3_confirmation_roots).

## Lo que el corpus todavía no permite

**191 de 216 claves de experimento están incompletas (88 %)**, así que el corpus no se puede deduplicar del todo. Campo ausente:

| campo | artefactos sin él |
|---|---:|
| `execution` | 140 |
| `endpoint` | 139 |
| `seed_block` | 110 |
| `contract_sha256` | 57 |
| `estimand` | 28 |

`seal_and_write` ya deriva `seed_block` y `endpoint` cuando el payload los contiene, así que esta cifra debe **bajar** con cada corrida nueva. Si sube, algo dejó de sellar.

## La interpretación, que no se genera

* [qué se puede afirmar y qué está prohibido](TABLA_CANONICA_DE_CLAIMS_2026-08-07.md) — `docs/TABLA_CANONICA_DE_CLAIMS_2026-08-07.md`
* [enmienda 1: H2, la cota de ofat, la cifra prohibida de H4](TABLA_CANONICA_DE_CLAIMS_2026-08-07_ENMIENDA_1.md) — `docs/TABLA_CANONICA_DE_CLAIMS_2026-08-07_ENMIENDA_1.md`
* [qué fichero manda por familia](INDICE_DE_ARTEFACTOS_AUTORITATIVOS.md) — `docs/INDICE_DE_ARTEFACTOS_AUTORITATIVOS.md`
* [huecos abiertos](REGISTRO_DE_HUECOS_2026-08-07.md) — `docs/REGISTRO_DE_HUECOS_2026-08-07.md`
