# Informe de reconciliación — segunda ronda SCRES-IA

**Fecha:** 2026-08-24 · **Fuente:** cuatro subagentes + comprobaciones locales posteriores.

## Veredicto corto

La búsqueda frontera produjo 24 registros académicos verificados (Crossref/arXiv), 15 PDFs OA locales y 9 artículos manuales con DOI verificado pero PDF todavía no descargado. La suite ampliada se ejecutó completa: no está verde. Program Q sigue cerrado; el fix-pack solo puede ser Q2 nuevo.

## Hechos verificados

- Frontier PDFs en el VPS: 15; todos pasan `%PDF-` y tamaño mínimo. Los mismos 15 están en el repo del Mac y en `<Mac-Downloads>/scres-papers/frontier-2021-2026/`.
- Núcleo: 4 PDFs en `attached/`; 25 PDFs de la colección core en `pdfs/`.
- Textos extraídos: 19 TXT en el bundle común, incluidos tesis, Garrido 2024, Ding 2026 y los 15 frontera.
- El contexto común contiene seis informes de navegación y está replicado en VPS repo, Mac repo y Mac Downloads; el checksum de `CONTEXTO_COMUN_HARNESSES_2026-08-24.md` coincide en los tres sitios.
- Suite `scres-ia-expanded-v2`: 2260 passed, 38 failed, 7 skipped, 2 xfailed; 1338.55 s. Suite `scres-ia`: 8 passed, 0 failed.
- Fallo de código prioritario: `test_cssu_capacity_bridge::test_the_null_is_anchored_outside_the_code_path_it_guards`, payload calculado `9cb65c7a...` frente al golden `f3fe61b1...`.
- Fallos de entorno identificados: referencias a `.venv/bin/python` cuando el entorno real es `.venv-scres-cert`, y harnesses que llaman `pip freeze` aunque el venv usa uv sin módulo pip.

## Lectura estadística de Q

La adjudicación vigente da:

- adaptación frente a open-loop: pasa 3/3;
- equivalencia learner–best-classical/belief-MPC dentro de ±0.01: pasa 3/3;
- prima neural ≥0.01: no establecida;
- worst-product-fill frente a classical: no pasa no-inferioridad en las tres celdas;
- worst-product-fill frente a open-loop: mejora en las tres celdas;
- integridad física y replays: pasan según el adjudicator verificado.

Esto no autoriza llamar a Q una imposibilidad general de aprender. Tampoco autoriza llamarlo una prima neural positiva.

## Propuesta de fix-pack Q2

La tabla de supresores propone investigar, bajo un contrato nuevo y semillas/tapes vírgenes:

1. observación enriquecida v7/v10 sin información futura;
2. reward densa únicamente como PBRS con potencial terminal cero;
3. gamma/GAE para el horizonte largo;
4. más pasos y checkpoints antes de escalar;
5. arquitectura recurrente mayor solo si los gates anteriores pasan;
6. comparador deployable y degradación simétrica del conocimiento del MPC como análisis secundario;
7. potencia y evaluación con tapes vírgenes.

No agrupar cambios para pescar el mejor resultado sin preregistro: si se agrupan, el estimando es el paquete completo, no la atribución individual.

## Papers manuales que aún dependen de Thommy

Kim (10.1080/24725854.2023.2217248), Fan JIPR (10.1186/s43065-023-00072-x), Liu POM (10.1177/10591478241305863), Kotecha (10.1016/j.compchemeng.2025.109111), Mousa (10.1016/j.compchemeng.2024.108783), Burtea & Tsay (10.1016/j.compchemeng.2023.108518), Akashi (10.23919/cnsm59352.2023.10327883), Ampratwum (10.1109/compsac61105.2024.00111), Cheng EJOR (10.1016/j.ejor.2022.11.038) y Zhou & Peng WSC (10.1109/wsc60868.2023.10407663).

Acceso: CRAI EZProxy o la base del editor; no usar Sci-Hub. Tras descargar: comprobar magic `%PDF-`, tamaño y DOI/título antes de copiar al bundle común.

## Contexto de los harnesses

- Codex produjo `IDEAS_CODEX.md` tras leer el bundle de textos y contratos.
- OpenCode produjo `IDEAS_OPENCODE.md` con activaciones concretas de bajo costo.
- La primera revisión de Claude fue parcial: declaró acceso solo a `pdfs_frontier/`, por lo que sus conclusiones no deben presentarse como lectura de los 4 PDFs nucleares ni como deduplicación completa contra el registro.
- Se lanzó una revisión de Claude contra `pdfs_frontier/context_texts/` y `context_reports/`, el bundle común accesible; su resultado solo será incorporado después de comprobar que el archivo existe y que cita los textos realmente disponibles.

## Próximo gate recomendado

Antes de entrenar Q2: corregir/aislar los anchors CSSU y los harnesses de entorno, sin cambiar la física ni los contratos; después ejecutar un Gate 0 prospectivo que mida headroom con controladores congelados. No gastar 800k pasos hasta que el gate demuestre headroom y la suite de la lane nueva tenga sus anchors verificados.
