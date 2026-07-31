# Errata — `fidelity_reference_v4`, campo `excluded_source`

**Fecha:** 2026-07-31 · **Artefacto:** `results/metric_audit/fidelity_reference_v4/result.json`
(sello `32e23a79b43f76a7…`) · **Estado del artefacto:** intacto, **no editado**.

## Qué dice el artefacto

> `Rsult_1.xlsx -- its twelve configurations differ from the raw workbooks by -1,949 to +735
> rows and store Re as a pasted constant, not the live formula`

El mismo texto está en el docstring de `scripts/build_fidelity_reference_v4.py:18-20` y en §1
de `docs/MANUSCRIPT_MODEL_VALIDATION_SECTION_2026-07-31.md`.

## Las dos afirmaciones son incorrectas

**«store `Re` as a pasted constant» — falso.** Medido con `openpyxl` sobre el libro:

| hojas | fórmulas |
|---|---:|
| `Cf1`–`Cf12` (12 hojas de configuración) | **43.776 – 45.234 cada una** |
| `APj`, `RPj`, `DPj`, `Re` (4 hojas agregadas) | 26 cada una |

Las hojas de configuración están **vivas**, con decenas de miles de fórmulas de
transformación. Solo las cuatro hojas agregadas están materializadas. La afirmación es cierta
como mucho de la hoja `Re`, y se escribió como si valiera del libro entero.

**«−1.949 a +735 filas» — no traza.** El número depende de una convención de conteo que nunca
se declaró, y ninguna de las tres candidatas lo devuelve:

| convención | rango |
|---|---|
| `max_row` (medido hoy) | −1.940 / +745 |
| lo que declara el artefacto | −1.949 / +735 |
| una revisión externa, con los rangos escritos en las propias fórmulas | −1.953 / +441 |

No se sustituye un rango sin medir por otro sin medir: **el conteo se retira**.

## Motivo de exclusión corregido

`Rsult_1.xlsx` queda excluido de la referencia porque **corresponde a otra muestra** —doce
configuraciones, no las veinte canónicas— y **no conserva el ledger operacional por pedido**
en la forma que los seis momentos requieren. No por estar materializado, que es falso, ni por
un conteo de filas que no traza.

**Sí tiene valor** para estudiar el procedimiento posterior de normalización y discretización,
y así debe describirse si vuelve a mencionarse.

## Qué se corrige y qué no

* **Corregido:** el docstring de `scripts/build_fidelity_reference_v4.py` y §1 de la sección
  del manuscrito.
* **No corregido, deliberadamente:** el campo `excluded_source` **del artefacto sellado**.
  Editarlo en sitio es exactamente la falta de 2026-07-31 (`002db49`): reescribir un artefacto
  después de que se citó. Y regenerar v4 cambiaría su `self_sha256` y con él toda cita
  existente. **Esta errata es el registro; el artefacto se lee junto a ella.**
* **Consecuencia deliberada:** el builder corregido, si se vuelve a correr, produce un
  `self_sha256` distinto del `32e23a79b43f76a7…` que citan la sección y la comparación. El
  falsador `f1_reference_is_v4_and_complete` de
  `scripts/run_fidelity_comparison_v4.py` lo detecta y **detiene la corrida** en vez de
  compararse contra una referencia que nadie citó. Es la conducta buscada.
* La corrección **no toca ningún momento**: `Rsult_1` nunca entró en la referencia. Los seis
  momentos de v4 son idénticos antes y después de esta errata.
