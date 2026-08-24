# IDEAS_CLAUDE — 10 ideas rankeadas (rol: estadístico / revisor adversarial)

> **Nota de alcance.** La sesión sólo tuvo permiso de lectura/escritura sobre `/home/ubuntu/scres-sources/pdfs_frontier/`.
> No pude leer `texts/` (tesis Garrido-Rios 2017, garrido2024_scres+AI, garrido2024_factory_resilience, Ding 2026),
> `reports/SINTESIS_FIXPACK_Y_PRIORIDAD.md` ni `PROMISING_LANES_REGISTRY.md`, ni escribir en `reports/`.
> Cito por tanto los **PDFs verificados** que sí leí. La deduplicación contra el registro de lanes NO pudo verificarse:
> lo mitigo apuntando sólo a inferencia, diseño muestral, métricas de cola y protocolos de validación (no a arquitecturas
> ni a lanes de método). Mover este archivo a `reports/IDEAS_CLAUDE.md` cuando haya permiso.

---

**1. Winner's curse: prohibir evaluar la política final con las corridas usadas para elegirla.**
Toda política seleccionada por early-stopping / tuning / "mejor semilla" hereda sesgo optimista; exigir *clean-up* con presupuesto de simulación fresco e independiente y reportar el delta sesgo = (valor de selección − valor limpio).
- Paper: `pdfs_frontier/a10-hong2021-fem-review-rs.pdf` §7 Problema 6 (Eckman & Henderson 2018b: reusar datos de búsqueda rompe PCS/PGS); `a7-gijsbrechts2022-msom-can-deep-rl.pdf` §1 (tuning caro ⇒ tentación de reusar).
- Claim grande: convierte "DRL supera al heurístico en X%" en una magnitud *corregida*; si el sesgo es del orden de la mejora reportada, buena parte de la literatura SCRES-DRL queda sin efecto demostrado. Resultado negativo publicable.
- CPU: medio (re-simular sólo finalistas con semillas disjuntas; ~1 barrido extra sobre k≈10–30 políticas).
- Riesgo histórico: **alto** — reduce (no aumenta) todas las mejoras porcentuales ya prometidas en contratos.

**2. Auditar invariancia de PBRS bajo truncación de horizonte.**
Si el harness trunca episodios, el potencial del estado terminal ≠ 0 y PBRS deja de ser invariante de política: el shaping *sí* cambia el óptimo. Test barato: ranking de políticas con/sin shaping a H ∈ {corto, medio, largo}.
- Paper: `pdfs_frontier/a3-mueller2025-arxiv-pbrs-effectiveness.pdf` §3.2 y §5.1–5.2 (Φ terminal debe ser 0; escala de Φ acotada por r_∞, r_g, Q_init, γ); `a1-okudo2021-ieee-access-subgoal.pdf` (subgoals como Φ).
- Claim grande: identifica un modo de fallo *estructural* —no de hiperparámetros— del shaping en cadenas de suministro con horizonte rodante; falsable con un experimento de una tarde.
- CPU: bajo (reusa corridas existentes + 2 horizontes adicionales).
- Riesgo histórico: **alto** para cualquier contrato que combine reward shaping con horizonte finito/rodante.

**3. Re-especificar la selección de política como R&S con garantía PGS y zona de indiferencia δ gerencial.**
En vez de "cuál gana la media", reportar P(seleccionar una política *buena*, i.e. dentro de δ del óptimo) ≥ 1−α, con δ fijado por el costo gerencial mínimo relevante, no por la varianza observada.
- Paper: `pdfs_frontier/a10-hong2021-fem-review-rs.pdf` §2.1.2–2.1.3 (IZ; PCS-IZ **no** implica PGS, Eckman-Henderson) y §7 Problema 5.
- Claim grande: primera garantía estadística auditable en SCRES-RL; además expone que PCS-IZ y PGS no son intercambiables, error frecuente en benchmarking de OM.
- CPU: bajo-medio (Rinott/KN sobre pocas decenas de alternativas).
- Riesgo histórico: medio — si δ_gerencial > ruido, mejoras previamente "significativas" pasan a ser indiferentes.

**4. Diferencias pareadas con CRN en vez de intervalos marginales por brazo.**
Reportar IC sobre la *diferencia* con semillas comunes, y verificar que CRN no invalide la constante h de Rinott (que asume independencia). El protocolo "matched seeds + IC 95%" de Guzmán se queda a medio camino si los IC son marginales.
- Paper: `pdfs_frontier/b4-guzman2026-cie-circular.pdf` abstract/§1 (matched seeds, horizontes fijos, IC 95%); `a10-hong2021-fem-review-rs.pdf` §3.1 (CRN induce correlación positiva y cambia el diseño).
- Claim grande: potencia estadística multiplicada a CPU constante; demuestra que los IC marginales solapados han estado ocultando efectos reales (y viceversa).
- CPU: **nulo** (mismo presupuesto, mejor estimador). Es re-análisis de corridas ya hechas.
- Riesgo histórico: bajo-medio — puede invertir el signo de comparaciones marginales ya reportadas.

**5. Métricas de cola: CVaR_α del costo y cuantil del tiempo de recuperación, no medias ni scores compuestos.**
Un "resilience score = 0.892" es una suma ponderada no identificable: el ranking depende de pesos arbitrarios. La resiliencia vive en la cola, no en la media.
- Paper: `pdfs_frontier/b10-kong2026-eai-transformer.pdf` abstract (score escalar 0.892, recovery 7.1 d, AUC 0.941, sin IC ni análisis de dominancia); `a10-hong2021-fem-review-rs.pdf` §6.2 (R&S multiobjetivo: conjunto de Pareto en lugar de escalarización).
- Claim grande: sustituye escalarización por frontera de Pareto con control de error tipo I/II sobre inclusión/exclusión; y desmonta el AUC como métrica en disrupciones raras (usar PR-AUC y calibración).
- CPU: **alto** (cuantiles extremos exigen N grande; mitigar con importance sampling / splitting de eventos raros).
- Riesgo histórico: **alto** — los contratos definidos sobre medias dejan de ser comparables; obliga a versionar la métrica.

**6. Tratar el escenario de disrupción como estado oculto estático y reportar el objetivo worst-case sobre su soporte.**
Robust RL y generalización sólo difieren en si se toma E[·] o min[·] sobre el estado oculto s^h; SCRES los confunde y luego llama "resiliencia" a un promedio.
- Paper: `pdfs_frontier/a6-ni2021-arxiv-recurrent-pomdp.pdf` §2 (average-case vs worst-case sobre s^h, formulación explícita) y §3 (Robust RL vs Generalization).
- Claim grande: da una definición operativa y falsable de "resiliencia" (worst-case sobre el soporte de disrupciones) que separa dos claims hoy fusionados; permite mostrar que políticas "resilientes" publicadas sólo son buenas en promedio.
- CPU: medio (mismo número de rollouts, distinta agregación + muestreo estratificado por escenario).
- Riesgo histórico: medio — cambia el diseño de muestreo de escenarios del harness (estratificado, no i.i.d.).

**7. R&S con covariables: la política ganadora como función del arquetipo, con PCS condicional / promedio / peor-caso.**
Guzmán afirma transferibilidad a 4 arquetipos sin retuning; eso es un claim de *superficie de decisión*, no de una política única, y debe evaluarse como tal.
- Paper: `pdfs_frontier/a10-hong2021-fem-review-rs.pdf` §6.4 (PCS(x) condicional, E[PCS(X)], min_x PCS(x); superficies de media que se cruzan ⇒ IZ es crítica); `b4-guzman2026-cie-circular.pdf` (4 arquetipos sectoriales).
- Claim grande: convierte "transferible" en garantía estadística sobre el dominio de covariables, y expone que en los cruces de superficies ninguna política domina.
- CPU: medio-alto (presupuesto repartido sobre el dominio de covariables).
- Riesgo histórico: medio-alto — puede demostrar que no existe la política única prometida.

**8. Asignación secuencial del presupuesto (KT / OCBA) al barrido de hiperparámetros y semillas.**
Dejar de correr N semillas iguales por configuración: la eliminación por torneo (knockout) alcanza la cota inferior de E[N] y es paralelizable sin sincronización ni comunicación.
- Paper: `pdfs_frontier/a10-hong2021-fem-review-rs.pdf` §5 (KT/KT⁺ rate-optimal, O(k) vs O(k log k), comparaciones locales); `a12-fan2025-jorsc-large-scale-so.pdf` §1 (divide & conquer, paralelización); `a7-gijsbrechts2022-msom-can-deep-rl.pdf` (tuning como cuello de botella).
- Claim grande: habilita barridos ~10× mayores a CPU constante ⇒ vuelve factibles las ideas 5, 6 y 7. Es el multiplicador de todo lo demás.
- CPU: **reduce** CPU (ése es el punto); coste de implementación bajo.
- Riesgo histórico: bajo — sólo cambia el log y la trazabilidad de experimentos.

**9. Incertidumbre de input: la distribución de demanda está estimada, y el PCS nominal no se cumple.**
Separar ruido de simulación de error de estimación del input; elegir política por peor caso sobre un conjunto de ambigüedad de parámetros de demanda/lead time (robust selection-of-the-best).
- Paper: `pdfs_frontier/a10-hong2021-fem-review-rs.pdf` §6.3 (Song et al. 2015: los procedimientos IZ fallan bajo input uncertainty; Fan-Hong-Zhang RSB y distributionally robust SB).
- Claim grande: buena parte de la ventaja reportada de DRL sobre heurísticos puede ser sobreajuste a *una* calibración de demanda; cuantificarlo es un resultado de frontera en SCRES.
- CPU: alto (bootstrap de inputs × escenarios × políticas; mitigar con las ideas 8 y 4).
- Riesgo histórico: **alto** — invalida toda comparación hecha con una única distribución calibrada.

**10. Value-of-Data como diseño factorial fraccionado, incluyendo el costo de omitir el estado lento.**
Convertir la ablación Silo vs Full en un factorial 2^(k−p) sobre canales de observación (física, pesos y semillas fijos), con corrección por multiplicidad, y añadir como factor la omisión/congelamiento del estado lento.
- Paper: `pdfs_frontier/b4-guzman2026-cie-circular.pdf` (protocolo VoD, Silo vs Full con física/pesos/semillas fijos); `pdfs_frontier/a4-wang2023-arxiv-freezing-slow.pdf` abstract y §1.1 (omitir estados lentos es mala heurística; congelarlos acota el regret con mucho menos cómputo).
- Claim grande: entrega una curva "valor marginal de la observabilidad" con efectos e interacciones estimados, que justifica o refuta la inversión en gemelo digital/sensórica — nadie lo reporta con diseño experimental formal.
- CPU: medio (fraccionado, no 2^k completo; se apoya en la idea 8).
- Riesgo histórico: medio — puede mostrar que features incluidas en contratos previos no tienen efecto detectable.

---
**Criterio del ranking**: (tamaño del claim × falsabilidad barata). 1–2 son resultados negativos casi gratuitos;
3–4 arreglan la inferencia; 5–7 redefinen qué se mide; 8 es el multiplicador de CPU; 9–10 son los más caros.
