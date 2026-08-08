# Enmienda 4 — retracción: no queda ningún bloque, y yo dije que sí

**Predecesores:** claim freeze `550a253` · Enmienda 1 `d7a205b` · Enmienda 2 (generada) ·
Enmienda 3 `78b13d0`.
**Motivo:** la Enmienda 1 §E3 afirma que existe un bloque de semillas sin abrir. Es falso, y ya lo
era cuando lo escribí.

**Filas superseded:** Enmienda 1 §E3, los dos párrafos de custodia. **Intacto:** todo lo demás de
E3 —el `NO-GO` de C1 y su bloqueador principal— y las reglas R1–R9.

---

## E1 · La retracción, exacta

La Enmienda 1 §E3 (`d7a205b`, línea 151) dice:

> ```
> contracts/g3a_asymmetric_claimants_v2.json  (sha 20952a3bff3c7b5b)
>   development_block: {start: 7700001, end: 7700120, status: RESERVED_NOT_OPENED}
> ```
> Dos razones independientes: [1] está **reservado para G3a**, no es una bolsa genérica de 120
> semillas; [2] está declarado `development_block` — **no es un bloque confirmatorio**.

**Ambas razones son falsas.** El registro central
(`research/seed_custody_registry.json`, `65f47eee8655e533`) da para ese bloque:

```
id:            g3a_v2_development          (7.700.001 – 7.700.120)
authorisation: docs/AUTORIZACION_PI_REPROPOSITO_BLOQUE_7700001_2026-08-07.md   (efeb4cb6680defc2)
opened_at:     2026-08-07T18:05:00Z        closed_at: 2026-08-07T18:20:00Z
opened_by:     docs/PREREGISTRO_CONFIRMACION_GSA_2026-08-07.md
outcome:       GSA_CONFIRMED_ON_VIRGIN_BLOCK_AS_A_ONE_BIT_CALENDAR_CHOICE
purpose:       "REPURPOSED 2026-08-07 by PI authorisation: prospective confirmation of the GSA
                lane under the declared resilience-only objective. G3a is left with no reserved
                block."
```

El bloque **fue repropositado por autorización escrita del PI, abierto, y cerrado**. No estaba
reservado para G3a y no se usó como bloque de desarrollo: se usó como **confirmación prospectiva**.

## E2 · Cuándo se rompió, y por qué no es un descubrimiento a posteriori

| hora (2026-08-07) | evento |
|---|---|
| **13:51** | `82f2c0a` — *«Authorisation, preregistration and runner for the last virgin block — committed before it is opened»* |
| 17:56 | `8ddf6f7` — registro de raíces de confirmación |
| **18:00:27** | `1157eec` — **mi registro de decisión**, que afirma el bloque reservado y sin abrir |
| 18:05 | bloque abierto |
| 18:20 | bloque cerrado |

La autorización llevaba **cuatro horas commiteada** cuando escribí lo contrario. La disciplina fue
correcta del otro lado —el documento dice literalmente «committed before it is opened»— y el fallo
es mío: **leí el contrato y el `status` global del registro, y no leí el registro del bloque.**

`contracts/g3a_asymmetric_claimants_v2.json` sigue diciendo `RESERVED_NOT_OPENED` porque un contrato
congelado no se edita en sitio; es el registro el que lleva el estado. Consulté la fuente que no
podía saberlo.

### R10 · El estado de custodia lo da el registro, nunca el contrato

> Un contrato declara **qué se pensaba hacer** con un bloque; el registro declara **qué se hizo**.
> Un contrato congelado conserva su estado inicial por diseño y no se actualiza cuando el bloque se
> abre. Toda afirmación de custodia se lee de `research/seed_custody_registry.json`, **fila del
> bloque**, no del contrato y no del `status` agregado — que sigue en
> `BASELINE_INVENTORY_INCOMPLETE_NO_NEW_SEEDS_AUTHORIZED` y es compatible con que el último bloque
> ya se haya quemado.

---

## E3 · El hecho de custodia que sustituye a la afirmación retirada

Enumerado sobre las 33 filas del registro: **no queda ninguna disponible.** Todas son `BURNED`,
`USED_DEVELOPMENT_NOT_VIRGIN`, `ATTEMPTED_NO_SEALED_ARTIFACT`, `PENDING_*` o cerradas.

Corroboración independiente: `results/dmlpa_kan_latent/result.json` lleva en su propio `scope`
**`DEVELOPMENT_OPEN_SEEDS_NO_CONFIRMATION_POSSIBLE_NO_VIRGIN_BLOCKS_REMAIN`**.

> **Hecho de custodia congelado: cero bloques disponibles. Ninguna confirmación nueva es posible sin
> una autorización escrita del PI que declare un rango nuevo, y el inventario base sigue incompleto.**

Esto **fortalece** el argumento que la afirmación retirada pretendía sostener. El `NO-GO` de C1 no
descansaba en el estado de ese bloque —su bloqueador es que `worst_product_fill` deja el estimando
indefinido, y sigue vigente— pero ahora tiene además un hecho operativo simple: **no hay dónde
correr una confirmación.**

### Consecuencia inmediata para el panel de robustez estacional

Se etiqueta **sensibilidad de desarrollo por defecto y por necesidad**, no por elección
conservadora. Convertirlo en confirmación exige exactamente el procedimiento de
`AUTORIZACION_PI_REPROPOSITO_BLOQUE_7700001_2026-08-07.md`: autorización escrita, preregistro y
runner commiteados **antes** de abrir, y una sola apertura.

---

## E4 · Lo que esto no cambia

- El `NO-GO` de C1 y sus precondiciones C1-A…C1-I: **intactos**.
- R1–R9: **intactas**. R4 en particular — nada de esto rescata un resultado pasado.
- El censo de confirmaciones de la Enmienda 3: **intacto**. C3 es precisamente la corrida que quemó
  este bloque, y ya estaba encuadrada como degradada a una conclusión de un bit.
- La Decisión 1 del PI (resiliencia media primaria, guardarraíles de reporte obligatorio): intacta.

## Custodia

Datado, no se edita en sitio. Sucesor: `…_ENMIENDA_5.md`. Reglas acumuladas: **R1–R10**.
