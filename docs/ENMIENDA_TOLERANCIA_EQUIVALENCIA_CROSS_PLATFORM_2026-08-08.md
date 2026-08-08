# Enmienda — regla de lectura de la diferencia numérica en `frozen_path_equivalence_v2`

**Congelada:** 2026-08-08, con la superficie extendida en **104 de 360 rebanadas** y **sin haber
leído ningún `mismatches` ni `max_abs_delta` de la ext**. La base ya está cerrada (360/360) y su
resultado es público dentro del proyecto; esta enmienda no lo toca y no puede cambiarlo.

## Por qué existe

El comparador de `replay_slice()` declara diferencia con **`if d > 0.0`** — igualdad bit a bit sobre
el valor de celda, los cuatro drivers y todas las claves del panel. Esa exigencia es correcta y
**ya se cumplió**: la superficie base reprodujo 103.680 celdas con `max|Δ| = 0,0` habiendo sido
producida en macOS arm64 / Python 3.11.15 y verificada en Linux x86_64 / glibc 2.43 / Python 3.14.4.
Eso es un **resultado medido**, no un supuesto, y es una de las piezas de reproducibilidad del
manuscrito.

El riesgo que esta enmienda cubre es el asimétrico: la superficie extendida corre repartida entre las
mismas dos arquitecturas sobre 1.658.880 celdas, dieciséis veces más superficie. Si **una** celda
difiere en el último bit de mantisa, hoy el certificado emite
`CURRENT_HEAD_NOT_EQUIVALENT_USE_FROZEN_RELEASE` con la ciencia intacta, y la discusión sobre qué
hacer ocurriría **después** de ver el número. Esa es exactamente la discusión que un preregistro
existe para impedir.

Cuatro auditorías externas independientes señalaron la ausencia de esta regla. Es una omisión real y
se repara antes del dato, no después.

## La regla, en dos capas

### Capa 1 — igualdad exacta, sin tolerancia y sin excepción

No admite ninguna holgura, en ninguna plataforma:

* identidades de configuración, índices de celda, contexto y semilla;
* conteos, calendarios y cualquier cantidad entera del ledger;
* secuencias de visita y el orden de evaluación;
* claves de configuración, `contract_sha256`, `module_manifest`, hashes de sellado;
* el veredicto de la cadena posterior y el diccionario `transfers`.

Una diferencia en la Capa 1 es **fallo del Certificado B sin adjudicación posible**.

### Capa 2 — equivalencia numérica declarada, sólo para floats

Aplica únicamente al valor de celda, los cuatro drivers y las claves de panel — los tres sitios donde
`replay_slice()` calcula `d`. Se declara **antes** de ver el resultado:

    atol = 1e-12
    rtol = 1e-12
    equivalente  ⟺  |x_new − x_old| ≤ atol + rtol·|x_old|

Justificación de la magnitud, no elegida por conveniencia: las cantidades comparadas son raciones y
horas de orden 1e0–1e5, de modo que `rtol = 1e-12` deja pasar del orden de una diezmilésima parte del
épsilon acumulable en una reducción de doble precisión, y **excluye por tres órdenes de magnitud**
cualquier diferencia que pudiera mover un driver o un ranking. Una divergencia real de física o de
caché es de orden 1e-3 o mayor; el control de mutación `m1_physics` la fabrica y debe seguir
detectándola.

## Cómo se adjudica — sin tocar código y sin invalidar rebanadas

El script **ya registra** `mismatches`, `max_abs_delta` y `worst_cell` por rebanada. No se modifica
nada: las 104 rebanadas de ext ya en disco y las 360 de base siguen siendo válidas, y el veredicto
primario del artefacto sigue siendo el exacto. La adjudicación se hace **sobre el artefacto emitido**:

| lo que reporte la ext | veredicto B | qué se escribe |
|---|---|---|
| `mismatches == 0` | `CURRENT_HEAD_BEHAVIOURALLY_EQUIVALENT` | equivalencia **exacta**, cross-platform, en 1.762.560 celdas |
| `mismatches > 0` y **todas** dentro de Capa 2 y Capa 1 limpia | `CURRENT_HEAD_NUMERICALLY_EQUIVALENT_NOT_BIT_EXACT` | se reporta `max_abs_delta`, el `worst_cell` y el recuento; **el release actual sigue siendo distribuible** |
| cualquier diferencia fuera de Capa 2, o cualquier diferencia de Capa 1 | `CURRENT_HEAD_NOT_EQUIVALENT_USE_FROZEN_RELEASE` | se distribuye el árbol histórico |

El veredicto intermedio es **nuevo** y por eso se declara aquí: no existía en el script y no puede
inventarse después de ver el número.

## Lo que esta enmienda no hace

* **No relaja el Certificado A.** La identidad histórica se decide aparte y sigue siendo la puerta
  científica: si A falla, A2 no entra al manuscrito, con o sin tolerancia.
* **No es una excusa retroactiva.** Si la base hubiera fallado, esta regla no la rescataría: la base
  ya cerró exacta y ese hecho se reporta como exacto, sin mencionar tolerancia alguna.
* **No cambia ninguna cifra científica.** El alcance del artefacto sigue siendo
  `PROVENANCE_ONLY_NO_SCIENTIFIC_CLAIM_NO_NEW_SEEDS`.
* **No se aplica a `search_ladder_v5` ni a `grid_transfer_confirmation_v2`.** Los contrastes, AUCs y
  cotas de esos artefactos se comparan exactos; una tolerancia sobre un estadístico de decisión sería
  otra cosa y no está autorizada.

## Falsador de esta enmienda

`f_tolerance_cannot_absorb_a_real_defect` — el control de mutación `m1_physics` perturba la física y
**debe** producir una diferencia **fuera** de la Capa 2. Si una mutación de física cayera dentro de
`atol + rtol·|x|`, la tolerancia estaría absorbiendo defectos reales y **esta enmienda queda
retirada**, volviendo el criterio a `d > 0.0` puro. El falsador puede fallar: basta con que la
perturbación elegida sea demasiado pequeña, y en ese caso la conclusión es que la regla es
inservible, no que la mutación es aceptable.

## Precedencia

Esta enmienda es posterior al preregistro de procedencia y lo enmienda **sólo** en el criterio de
lectura de floats. Todo lo demás del contrato —cobertura completa, cadena posterior, cuatro controles
de mutación, verificador independiente sólo-stdlib, dos veredictos separados— queda sin cambios.
