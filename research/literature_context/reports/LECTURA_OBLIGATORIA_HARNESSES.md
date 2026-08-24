# LECTURA OBLIGATORIA — papers en contexto para todos los harnesses

**Fecha:** 2026-08-24 · **Ordena:** Thommy (PI) · **Alcance:** Hermes, Codex, Claude Code, OpenCode y cualquier subagente

## Regla permanente

Antes de proponer diseño, experimentos o claims para SCRES-IA, los harnesses deben tener en contexto (o leer directamente) los PDFs listados abajo. Están sincronizados en **tres ubicaciones** (VPS repo, Mac repo, Mac Downloads). Si propones algo que contradice o duplica uno de estos papers, cita el paper y justifica.

## Ubicaciones (idéntico contenido)

| Dónde | Ruta |
|---|---|
| VPS — checkout científico | `/home/ubuntu/scres-ia-expanded-v2/research/literature_pdfs_core/` (25) y `literature_pdfs_frontier/` (15) |
| VPS — bundle maestro | `/home/ubuntu/scres-sources/pdfs/`, `pdfs_frontier/`, `attached/` (tesis + 3 Garrido/Ding) |
| Mac — repositorio | `~/Projects/research/scres-ia/research/literature_pdfs/` (26) y `literature_pdfs_frontier/` (15) |
| Mac — Downloads | `<Mac-Downloads>/scres-papers/` (+ `core-63/` 26, `frontier-2021-2026/` 15) |

Nota: `core-63` tiene 26 PDFs porque incluye Gijsbrechts2022 tanto en A como en B por doble clasificación temática.

## Núcleo obligatorio (léelo TODOS antes de opinar)

1. `attached/WRAP_Theses_Garrido_Rios_2017.pdf` — la tesis: DES MFSC, ReT/TAE (Eq 5.1–5.5), riesgos R1/R2/R3, supuestos §6.5 y agenda §8.6.
2. `attached/garrido2024_scres+AI.pdf` — el paper-fuente del proyecto: Alzheimer effect, cierre del loop entre nodos ③↔⑧, Fig. 5 neurona d_i·ρ.
3. `attached/garrido2024_factory_resilience.pdf` — APP puras vs híbridas (S11–S32), demanda CON variabilidad (cv, smoothing α/γ), ranking Eq 9, R Cobb-Douglas.
4. `attached/1-s2.0-S0925527326000861-main.pdf` — Ding 2026 IJPE: filling/repairing/recruiting sobre SCDN, MAPPO CTDE, métrica topológica.

## Frontera lane topológica (B, 2023–2026)

5. `pdfs_frontier/b4-guzman2026-cie-circular.pdf` — DT+MARL cooperativo, protocolo con CI95.
6. `pdfs_frontier/b10-kong2026-eai-transformer.pdf` — transformer vs LSTM en MARL logístico.
7. `pdfs_frontier/b12-ijpe2023-owmr-deep-rl.pdf` — deep RL supply chain risk (IJPE).
8. Kim 2023 IISE (transshipments Dec-POMDP) — **MANUAL**, DOI 10.1080/24725854.2023.2217248.
9. Fan 2023 JIPR (GCN repair recursos escasos) — DOI verificado, PDF MANUAL por bot-wall; descargar legalmente vía CRAI/red no filtrada.
10. Kotecha 2025 / Mousa 2024 / Burtea & Tsay 2024 (CompChemEng) — MANUAL vía CRAI.

## Fix-pack de aprendizaje (A)

11. `pdfs_frontier/a1-okudo2021-ieee-access-subgoal.pdf` — PBRS por subgoals.
12. `pdfs_frontier/a3-mueller2025-arxiv-pbrs-effectiveness.pdf` — cuándo PBRS funciona (Q-init).
13. `pdfs_frontier/a2-forbes2024-arxiv-pbrs-intrinsic.pdf` — PBRS para motivación intrínseca; suplementario, no confundir con Müller.
14. `pdfs_frontier/a4-wang2023-arxiv-freezing-slow.pdf` — γ alto + horizontes largos (freezing slow states).
15. `pdfs_frontier/a6-ni2021-arxiv-recurrent-pomdp.pdf` — recurrent model-free RL bien hecho (ICML).
16. `pdfs_frontier/a7-gijsbrechts2022-msom-can-deep-rl.pdf` — cuándo DRL NO supera heurísticas fuertes.
17. `pdfs_frontier/a8-boute2022-ejor-roadmap.pdf` — roadmap RL inventario.
18. `pdfs_frontier/a10-hong2021-fem-review-rs.pdf` — ranking & selection con garantías.
19. Cheng 2023 EJOR (sequential finite-sample) — MANUAL, 10.1016/j.ejor.2022.11.038.

**Advertencia de catálogo:** el fichero `a11-luo2024-scis-survey-mbrl.txt` contiene CONFIG (Xu et al., optimización bayesiana restringida), no una encuesta MBRL; usar su contenido real y no su nombre hasta corregir el manifiesto.

## Contexto del estado experimental (leer junto a los papers)

- `/home/ubuntu/scres-sources/reports/SINTESIS_FIXPACK_Y_PRIORIDAD.md`
- `/home/ubuntu/scres-sources/reports/REPORT_FRONTERA_2021-2026.md`
- `/home/ubuntu/scres-sources/reports/SECOND_OPINION_CODEX.md`, `SECOND_OPINION_CLAUDE.md`
- Veredicto Q vigente: `STOP_Q_NO_REPLICATED_LEARNED_ADAPTATION` pero con H_OL y equivalencia pasadas 3/3; lo único fallado es la guardrail de cola (`worst_product_fill`).

## Qué se espera de cada harness

- **Ideas nuevas**: cualquier mecanismo no cubierto por los papers de arriba es candidato a lane nueva — propónla con el paper que la motiva o declárala como hipótesis propia.
- **Sin re-inventar**: si tu idea ya está en un paper listado, cítalo y propón cómo ADAPTARLO al MFSC, no repetirlo.
- **Respetar la línea roja**: ningún rediseño toca contratos adjudicados (O/O-R/Q sellados); todo cambio nuevo = preregistro nuevo con semillas vírgenes.
