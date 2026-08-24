# ESTADO — papers distribuidos + ideas de los 3 harnesses

**Fecha:** 2026-08-24

## Distribución de papers (verificada)

| Ubicación | Contenido |
|---|---|
| VPS `/home/ubuntu/scres-sources/` | bundle maestro: attached (4) + pdfs (25) + pdfs_frontier (15) + texts (19 .txt extraídos para lectura barata) |
| VPS repo `scres-ia-expanded-v2/research/` | symlinks `literature_pdfs_core`, `literature_pdfs_frontier` + LECTURA_OBLIGATORIA + IDEAS_* |
| Mac repo `~/Projects/research/scres-ia/research/` | literature_pdfs (26), literature_pdfs_frontier (15), literature_texts (19), LECTURA_OBLIGATORIA |
| Mac Downloads `<Mac-Downloads>/scres-papers/` | 29 PDFs sueltos + core-63/ + frontier-2021-2026/ + texts/ (19) + reports-v2 (9 informes) |

## Ideas de frontera producidas (3 harnesses, 30 ideas totales)

- **IDEAS_CODEX.md**: micro-MFSC con cota óptima exacta (¿Δ_N≈0 significa ambos casi óptimos o ambos lejos?); selección contextual de controlador; comparadores APP dinámicos S11-S32; Recurrent SAC/TD3; PBRS subobjetivos ordenados contra el guardrail de cola; dosis-respuesta de calidad de forecast; shocks hard/soft igualados en daño; incertidumbre epistemológica 2 niveles; robustez minimax; escalera R1/R2/R3.
- **IDEAS_CLAUDE_FULL.md** (el eje estadístico): winner's curse (reusar datos de selección infla mejoras — riesgo alto para claims existentes); **PBRS deja de ser invariante si el horizonte trunca** (falsable en una tarde); PGS con δ gerencial en vez de "gana la media"; diferencias pareadas CRN (CPU cero); CVaR/cuantil recuperación en vez de scores escalares; worst-case sobre estado oculto como definición falsable de resiliencia.
- **IDEAS_OPENCODE.md** (activable HOY con flags): PBRS bias-shift (~3 CPU-h), grid γ=0.95/0.97/0.99 × norm-reward (~6 CPU-h), subgoal PBRS step_level, recurrent bien configurado n_steps=2048, parche v10 1 línea.

## Pendiente que requiere a Thommy

1. Papers MANUAL vía CRAI: Kim IISE 2023, Kotecha 2025, Mousa 2024, Burtea & Tsay 2024, Liu POM 2024, Cheng EJOR 2023, Zhou WSC 2023, Fan JIPR 2023 (OA pero bot-wall), Akashi CNSM 2023, Ampratwum IEEE 2024. DOIs en REPORT_FRONTERA_2021-2026.md §MANUAL.
2. Confirmar priorización: smoke fix-pack (Gate 0 ~6h sin entrenar) vs lane topológica vs ideas nuevas.
