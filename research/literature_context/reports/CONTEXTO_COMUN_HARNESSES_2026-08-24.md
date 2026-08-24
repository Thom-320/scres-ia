# Contexto común SCRES-IA para Hermes y todos los harnesses

**Fecha:** 2026-08-24 · **Naturaleza:** índice operativo, no un nuevo resultado científico.

## Regla de lectura

Antes de proponer código, experimentos o claims, leer los textos de `../context_texts/` y los informes de `../context_reports/`. Los PDFs son la fuente primaria; los informes son mapas de navegación y no sustituyen al paper. Citar el nombre del PDF/DOI y, cuando importe, la página o sección.

## Estado de acceso verificado

- **15 PDFs frontera OA** existen y pasan la comprobación `%PDF-` y tamaño mínimo en `/home/ubuntu/scres-sources/pdfs_frontier/`.
- **4 PDFs nucleares** existen en `/home/ubuntu/scres-sources/attached/`: tesis Garrido-Ríos 2017, dos trabajos Garrido 2024 y Ding 2026 IJPE.
- **19 textos extraídos** existen en `context_texts/`: los 4 nucleares + los 15 frontera; sirven para lectura rápida, pero no sustituyen la paginación del PDF.
- **10 artículos siguen MANUAL**: DOI verificado, PDF no descargado por paywall/bot-wall. No declarar que fueron leídos íntegramente hasta que Thommy los descargue vía CRAI o aparezca una versión legal verificable.
- La ruta de acceso de los 10 manuales y sus DOI están en `REPORT_FRONTERA_2021-2026.md` y `MANIFIESTO_PDFS.md`.

## Qué es evidencia y qué es propuesta

### Hechos verificados en el repositorio/resultados

1. El contrato Q adjudicado permanece cerrado; no se reabre con otras semillas, endpoint o margen.
2. En Q: la adaptación contra open-loop pasa en 3/3 celdas; la equivalencia del learner frente al mejor belief-MPC pasa dentro de ±0.01 en 3/3. La prima neural ≥0.01 no está establecida.
3. `worst_product_fill` del learner frente a classical tiene LCB95 negativo en las tres celdas; frente a open-loop es positivo. Esto no licencia afirmar seguridad de cola frente al classical.
4. La suite `scres-ia` pequeña pasó 8/8. La suite `scres-ia-expanded-v2` ejecutada de forma explícita sobre `tests` dio 2260 passed, 38 failed, 7 skipped y 2 xfailed; ver `SUITE_CERTIFICACION.md`. No es suite verde.
5. El repositorio contiene un anchor/hash CSSU que no reproduce el payload actual (`9cb65c7a...` frente al golden `f3fe61b1...`); es un fallo de custodia/código pendiente, no evidencia sobre la prima neural.

### Inferencias de la literatura

- La demanda variable/no estacionaria está respaldada por Garrido 2024; la demanda i.i.d. pertenece al entorno más simple de la tesis 2017.
- El empate con belief-MPC puede ser consistente con poco headroom, pero también con supresores de información, señal de recompensa, horizonte, presupuesto de entrenamiento y comparador con conocimiento exacto. La literatura no demuestra cuál mecanismo domina en SCRES sin un experimento nuevo.
- La lane topológica es una propuesta: `filling/repairing/recruiting`, recursos no fungibles, costes explícitos, riesgos hard/soft, observación parcial y MAPPO/CTDE sobre una red pequeña. El gate correcto es medir headroom antes de entrenar.

## Lotes de papers leídos / pendientes

### Núcleo

- Garrido-Ríos 2017: `WRAP_Theses_Garrido_Rios_2017.txt`.
- Garrido 2024 SCRES+AI: `garrido2024_scres+AI.txt`.
- Garrido 2024 factory resilience: `garrido2024_factory_resilience.txt`.
- Ding 2026 IJPE: `1-s2.0-S0925527326000861-main.txt`.

### Fix-pack / evaluación

Okudo 2021, Hong 2021 R&S, Sharma 2021 discount, Ni 2021 recurrent POMDP, Gijsbrechts 2022, Boute 2022, Wang 2023, HPRS 2024, CONFIG (fichero histórico `a11-luo2024-scis-survey-mbrl`) y Müller 2025: todos en `context_texts/` y con PDF correspondiente en el lote frontera. El fichero a11 está mal catalogado; su contenido real es CONFIG, no una encuesta MBRL.

### Reconfiguración / MARL

Guzmán 2026, Kong 2026 y OWMR/IJPE 2023 están descargados en `context_texts/`/`pdfs_frontier/`. Kim 2023/2024, Fan 2023, Liu 2024/2025, Kotecha 2025, Mousa 2024, Burtea & Tsay 2024, Akashi 2023, Ampratwum 2024, Cheng 2023 y Zhou 2023 tienen registro bibliográfico y vía CRAI; no todos tienen PDF local.

## Ideas que deben someterse a gate, no asumirse como solución

1. Micro-MFSC exacto por iteración de valor para medir la brecha al óptimo.
2. R&S contextual con PCS/PGS y zona de indiferencia fijada antes de ver resultados.
3. Comparadores APP dinámicos del trabajo Garrido 2024.
4. Recurrent SAC/TD3 con contexto y critic separado, si la pregunta sigue siendo POMDP.
5. PBRS con subobjetivos de recuperación y potencial terminal cero.
6. Dosis-respuesta de calidad del forecast compartido entre learner y MPC.
7. Shocks hard/soft/complex con daño integrado igualado.
8. Incertidumbre epistemológica de inputs separada de incertidumbre de simulación.
9. Robustez minimax con ambiguity set simétrica para learner y MPC.
10. R&S con winner's-curse cleanup, CRN pareado, PGS y métricas de cola CVaR10/TTR.

La revisión estadística común está en `CLAUDE_COMMON_REVIEW_2026-08-24.md`. Su ranking añade auditorías de coste casi nulo: split de tapes para Gate 0, descomposición de ReT por régimen/composición, rama muerta `Re(DP-RP)`, dependencia de la escala recíproca `Re(RP)`, y dependencia del conjunto en el índice R de Garrido. El diseño corregido de Gate 0 está en `AUDITORIA_GATE0_SPLIT_TAPES.md`; no se ha ejecutado todavía.

## Línea roja de gobernanza

Todo lo que cambie observación, reward, gamma, horizonte, arquitectura, acción o comparador es un contrato nuevo. Debe usar semillas/tapes vírgenes, SHA nuevo, estimando y gates congelados antes de entrenar. Ninguna idea de estos informes autoriza re-adjudicar O, O-R o Q.
