# ACTA DE FIRMA Y LANZAMIENTO — 2026-08-25

| Preregistro | SHA-256 congelado | Estado |
|---|---|---|
| `GATE0_SPLIT_TAPE_PREREGISTRO_V1.md` | `b2a6058ccf2062f36c3dbbceadf3a5f34ba503df2a2203c4a3e6135415428384` | **FIRMADO — EN EJECUCIÓN** |
| `DEPLOYABLE_COMPARATOR_PREREGISTRO_V1.md` | `ec859f8c3333a16a0d1926ff27be1b3e136f7062f4b296f11a24cdf4bad1f25b` | **FIRMADO — AUTORIZADO (cola tras Gate-0)** |

**Autorización del PI (2026-08-25, en sesión):** «Ejecuta: Gate-0 split-tape
(~6 h) y Comparador desplegable (~20–40 h)».

**Side quest prima neural:** el PI ordenó elegir camino y firmar su contrato.
Convergencia de planes: 2/3 recomiendan **régimen latente no estacionario**
(Codex E2 + subagente A), OpenCode recomienda safety-first HPRS. El gate de
semana 1 del camino ganador es el mismo G0-split ya firmado más la sonda de
aliasing; el contrato específico se redactará como
`program_latent_regime_v1.json` SOLO si los gates pasan, según el árbol de
decisión de `SIDEQUEST_PRIMA_NEURAL_PLAN.md` §4.

## Condiciones registradas antes de abrir semillas

- Bloque asignado: `7550001–7550512`. Scan de colisiones repository-wide:
  **0 colisiones** (2026-08-25, script sobre `git ls-files`, patrón 6–9 dígitos).
- Prohibido tocar: 7480101–7480148 (sellado), 7490001–7490256 (consumido por Q),
  S1–S4/Paper3 (reservados), sandbox/blind.
- Compromiso de publicación gane o pierda; prohibido cambiar estimando/SESOI/
  falsadores tras ver tapes B.

## Ejecución

- Gate-0: lanzamiento inmediato tras esta firma (~6 CPU-h).
- Comparador desplegable: autorizado, se ejecuta tras cerrar Gate-0
  (~20–40 CPU-h).
