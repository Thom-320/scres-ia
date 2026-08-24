# Síntesis de la revisión de literatura y estrategia Q1

**Fecha:** 2026-08-24 · **Fuentes:** REVISION_CODEX_LITERATURA.md,
REVISION_CLAUDE_ESTADO_ARTE.md, REVISION_OPENCODE_COBERTURA.md (los tres en
`/home/ubuntu/scres-sources/reports/`), más la bibliografía verificada en
`registry/`.

## 1. La bibliografía: misión cumplida

| Fuente | Entradas | 2021+ |
|---|---|---|
| Registro curado verificado (Crossref/DataCite) | 84 | 42 |
| Shortlist Crossref nueva | 70 | 70 |
| **Total disponible** | **154** | **112** |

- Todas las entradas tienen DOI resuelto contra Crossref o DataCite; el año es el
  **verificado**, no el declarado.
- Tras la poda editorial que propone OpenCode (~29 ruido del registro + ~28 de la
  shortlist), quedan unas **104 referencias incorporables**: muy por encima de
  las 50 exigidas, con densidad de frontera >70 %.
- Artefactos: `registry/BIBLIOGRAFIA_PAPER.{json,md}` y `bibliografia_paper.bib`
  (119 entradas ya consolidadas con clásicos canónicos etiquetados).

Los tres revisores coinciden además en los huecos que hay que cerrar a mano:
Ng 1999 y Wiewiora 2003 (PBRS primario), Yu et al. 2022 MAPPO / Rashid 2018
QMIX, Altman 1999 / Achiam 2017 (CMDP), Clark & Scarf 1960 y Gallego-Moon 1993,
Sterman 1989, y la referencia vigente de SimOpt. Son ~10 citas clásicas que un
revisor de CIE exigirá.

## 2. Diagnóstico: por qué no había prima neural

Convergencia total de Codex y Claude, con evidencia citada:

1. **El valor está en el feedback, no en la red.** El learner supera al mejor de
   los 65.536 calendarios open-loop (3/3 celdas), pero es *estadísticamente
   equivalente* al mejor controlador estructurado dentro de ±0.01 preregistrado
   (3/3). La información contingente relevante ya la captura una política
   estructurada.
2. **Parte de las rutas históricas midieron el endpoint equivocado**: en Paso 3 se
   aplicó `flow_fill_rate` donde el preregistro pedía `worst_product_fill`.
3. **La cola falla por diseño del objetivo**: el learner compra fill agregado
   desbalanceando el producto débil (`worst_product_fill` LCB95 negativo 3/3) —
   sustitución media↔mínimo que ningún índice escalarizado del campo detecta.
4. El Gate 0 propuesto tenía winner's curse (seleccionar y evaluar sobre los
   mismos tapes); corregido en `AUDITORIA_GATE0_SPLIT_TAPES.md`, todavía no
   ejecutado.

## 3. Dónde está nuestro claim (lo que nadie más hace)

Verificado contra los textos de Ding 2026, Guzmán 2026 y Kong 2026: **ninguno
tiene techo de información perfecta, ni comparador enumerado exhaustivamente, ni
equivalencia preregistrada, ni endpoint de equidad por producto**.

La novedad real es el **aparato de medición con capacidad de falsación**:
techo enumerado (H_PI=0.1515, placebo fungible exactamente 0) + comparador =
máximo de la frontera completa reseleccionado en cada resample + equivalencia
con δ y potencia declarados + guardrail de equidad que puede fallar (y falla).

**Claim primario formulable:** «el valor de cerrar el loop Alzheimer está en la
realimentación, no en la aproximación neural; y el objetivo agregado compra la
media a costa del mínimo». Cuatro componentes: H_OL positivo grande,
equivalencia formal, techo medido que descarta falta de potencia, fallo de cola
mecanístico.

## 4. Journal

**Computers & Industrial Engineering** (recomendación de Claude, ~45-55 %
con el paquete completo). Argumento decisivo: Guzmán 2026 publicó en CIE un
paper cuya contribución primaria declarada es el *protocolo de evaluación* —
prueba documental de que el editor compra este encuadre, y ancla de
posicionamiento. Fallback: IJPR → IJPE → AOR.

## 5. Qué falta para enviar (bloqueantes)

1. Resolver la identificación del endpoint y publicar como sensibilidad la
   inversión de signo endpoint×bloque (si la encuentra un revisor, el paper muere).
2. Descomposición de ReT en efecto intra-régimen vs composición (Kitagawa/Oaxaca).
3. Brazo secundario preregistrado con comparador desplegable (modelo estimado /
   misma observación parcial).
4. IC pareados CRN sobre diferencias — costo casi nulo.
5. Clean-up de winner's curse con presupuesto fresco.
6. **Arreglar los anchors de integridad** (en curso, ver abajo).
7. Preregistro con hash de cualquier brazo nuevo.
8. Los 10 papers MANUAL vía CRAI (depende de ti).
9. Respuesta documental de Garrido a RT1–RT5 (ruta crítica que no controlamos).

Estimación agregada Claude: ~6-10 semanas analista + ~40-100 CPU-h, sin
entrenamiento nuevo obligatorio para el claim primario.

## 6. Estado de la suite (trabajo propio de hoy)

| Clase | Fallos | Estado |
|---|---|---|
| Entorno `.venv`/pip ausente | 20 | **corregida** — symlink `.venv`→venv certificación + pip instalado |
| Atributos sin clasificar (CSSU/expedite/LOC) | 10 | **corregida** — clasificación + 9 invariantes nuevos; `classification_complete=True` |
| Anchors desactualizados (CSSU/LOC/Program J) | 7 | diagnosticada; **decisión del PI** |
| Artefacto ausente (`candidate_family_ledger.json`) | 1 | diagnosticada |

**Verificado hoy:** transducer 50 passed/1 skipped; harness smoke 6 passed;
checksums 4 passed. Fallo residual conocido: `terminal_readiness` por el
artefacto ausente.

### El hallazgo importante sobre el anchor CSSU

Ejecuté el test **en el propio commit que congeló el golden** (worktree
3b70dd9): **falla igual** — `9cb65c7a` ≠ `f3fe61b1`. Y los ficheros de física no
han cambiado desde ese commit. Conclusión: **el hash se congeló mal desde el
inicio** (el commit congeló un hash que su propia física nunca produjo), o la
física cambió en el mismo commit que lo congeló. No es drift ambiental: mismo
venv, mismos pins, mismo código.

Esto confirma lo que el test exige: re-congelar requiere un commit del PI que lo
declare. Yo no tomo esa decisión.

### Hallazgo adicional: overlay war-GSA

`test_overlay_freeze_is_complete_and_unopened` falla porque la reconstrucción del
manifiesto difiere del artefacto congelado en **12 valores de coma flotante de
1-2 ulp** (peor diferencia relativa 2.1e-16). El código del builder es
idéntico desde el commit que escribió el manifiesto, scipy/numpy están en los
pins exactos, y el artefacto nunca fue modificado. Es drift numérico ambiental
de última cifra en `qmc.scale`/transformaciones físicas. Remedio honesto: tolerancia ulp en la verificación de reconstrucción o re-congelado documentado;
no lo aplicé porque es un contrato de muestreo congelado.
