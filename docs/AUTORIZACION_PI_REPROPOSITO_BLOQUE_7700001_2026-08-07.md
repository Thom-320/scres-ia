# Autorización del PI — repropósito y apertura del último bloque virgen

**Fecha:** 2026-08-07. **Autoriza:** el PI (Thomas Chisica). **Ejecuta:** este agente.
Queda escrito como **decisión del PI**, no como iniciativa del ejecutor.

## 1. Qué se autoriza, exactamente

| | |
|---|---|
| bloque | `g3a_v2_development`, semillas **7.700.001–7.700.120** |
| estado previo | `RESERVED_NOT_OPENED` — **el único bloque virgen que queda en el proyecto** |
| propósito original | *"G3a asymmetric-claimant development block"*, contrato `contracts/g3a_asymmetric_claimants_v2.json` |
| **propósito nuevo** | **confirmación prospectiva de la lane GSA** bajo el objetivo declarado por el PI |
| apertura | **una sola vez**, sin segunda oportunidad |

## 2. Las dos condiciones que el PI levanta explícitamente

El registro central (`research/seed_custody_registry.json`) impone dos frenos, y **ninguno se
salta en silencio**:

1. **`submission_a_receipt_required_before_g3a_open: true`.** El PI lo **levanta** para esta
   apertura. Se levanta *para este uso concreto*, no como cambio permanente de la regla.
2. **`new_seed_opening: false` / `scientific_execution_authorized: false`.** El PI **autoriza esta
   apertura puntual**. La cabecera del registro no cambia de estado general: se añade una excepción
   nominal y fechada.

Además, esto es un **repropósito**: el bloque estaba reservado para G3a y se gasta en el GSA. G3a
queda **sin bloque virgen reservado**, y ése es el coste real de esta decisión. Consta aquí.

## 3. Por qué el GSA y no otra cosa — el razonamiento que el PI aceptó

| candidato | qué compraría | por qué no |
|---|---|---|
| confirmar **H1** | subiría una hipótesis del borrador de desarrollo a confirmación | H1 no es la espina del paper; ya hay una confirmación sobre el claim central |
| confirmar **C1** (prima neural) | nada | **el arnés no resuelve el efecto**: ±2,4 de ruido a semilla fija contra una prima de +1,44…+2,18, y sin intervalo en su artefacto. Ninguna cantidad de semillas lo arregla |
| **confirmar el GSA** | convierte **el mecanismo** de «califica en desarrollo» a confirmación prospectiva | es lo que se autoriza |

El GSA es la única lane con **η 0,78–0,91** y un margen sobre placebo desinformado cinco veces
mayor que su propio headroom. Confirmarla cuesta **~20 segundos de cómputo**.

## 4. Lo que esta autorización NO concede

* **No autoriza entrenar nada.** Si la confirmación pasa, lo que abre es el derecho a
  **preregistrar** una lane con oracle-first, exactamente como fija el certificado de agotamiento.
* **No cambia el estado general del registro** a `scientific_execution_authorized: true`.
* **No concede una segunda apertura.** Si la corrida se cae, el bloque queda quemado igual y se
  registra como tal. No hay rescate.
* **No levanta ningún guardarraíl de resiliencia.** El coste distributivo se reporta entero.

## 5. Cadena de custodia

1. Esta autorización se commitea **antes** que el preregistro.
2. El preregistro (`docs/PREREGISTRO_CONFIRMACION_GSA_2026-08-07.md`) se commitea **antes** de
   correr, con la regla de lectura fijada.
3. El registro se marca `OPEN` con recibo de apertura **antes** de la corrida.
4. La corrida se sella una vez y el bloque pasa a `BURNED_CONFIRMATION_COMPLETE`.
