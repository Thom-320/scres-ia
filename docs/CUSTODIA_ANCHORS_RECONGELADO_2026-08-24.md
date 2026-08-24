# Causa de re-attestación de pines de fuente — 2026-08-24

**Decisión del PI (Thommy), comunicada en sesión:** «Reconozcamos que el golden
nunca fue correcto y requémoslo: hagámoslo lo que mejor le sirva al repositorio,
al proyecto y al paper.»

## Alcance

`--source supply_chain/supply_chain.py --old d8fd93475b3904bcc48206b803cc842c5f2ad05f9c97a775e4eedf5c5ca8401a`

## Causa — preservación conductual probada

Los cuatro commits que separan el pin viejo del árbol actual son:

| commit | contenido | efecto sobre la física por defecto |
|---|---|---|
| `84937de` (2026-08-01) | Expedición service-first: parámetros nuevos `expedite_*` con default **0.0/12/24**, `_pending_expeditions` vacío, rama en `_pt()` que solo actúa si hay expedición armada | inerte por defecto |
| `646c65a` (2026-08-02) | Desbloqueo G3c de acoplamiento temporal | sin cambio de default |
| `852c64b` (2026-08-02) | `loc_topology_mode` default `serial_v1`, no construye nada; el arco solo se *nombra* en eventos R22 sin tocar RNG | inerte por defecto |
| `3b70dd9` (2026-08-03) | Almacenamiento finito CSSU: `_cssu_capacity_ledger=None` mantiene el comportamiento shipped | inerte por defecto |

Diff agregado: 216 líneas añadidas / 6 eliminadas; las 6 eliminadas son
reescrituras equivalentes (p. ej. `base = self.params[param_key]` →
`nominal = float(...); base = nominal`).

### Prueba conductual ejecutada hoy

1. **Transducer exacto completo** (`tests/test_paper2_bottleneck_exact_transducer.py`):
   **50 passed, 1 skipped** — incluye los tests que comparan contra fuerza bruta y
   replay bit-exacto.
2. **Frontera completa W6** (`tests/test_paper2_bottleneck_full_frontier.py`):
   **15 passed** — frontera canónica contra bruta + replay.
3. **Auditoría de estado**: clasificación completa, sin solapamientos, 9 invariantes
   nuevos verificando que los campos añadidos están inertes bajo este contrato.
4. **Resultados congelados intactos**: `verdict.json` y `raw_matrices.npz` del
   audit Program J siguen cuadrando sus hashes; solo el pin del *fuente* quedó
   viejo.

## Reconocimiento explícito sobre los goldens

El anchor CSSU (`f3fe61b1…`) y su hermano LOC (`371c5975…`) fueron congelados con
valores que el código commiteado nunca produjo: ejecutando el test dentro del
propio commit `3b70dd9` (worktree limpio, mismo intérprete, mismos pins) el hash
calculado es `9cb65c7a…`. La causa más probable es un árbol de trabajo o entorno
distinto al momento de congelar. Estos dos anchors se re-queman contra la física
actual verificada, con esta admisión como justificación registrada.

## Lo que NO toca esta re-attestación

- Los freezes de ejecución de Program O (pin un hash viejo a propósito).
- Ningún resultado científico, contrato ni semilla.
