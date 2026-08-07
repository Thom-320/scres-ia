# Enmienda — sellado retroactivo de los artefactos del bake-off de arquitecturas

## 1. El defecto

`scripts/run_architecture_bakeoff_v1.py` **no importa `seal_and_write` y no sella nada**. Sus
salidas —`results/architecture_bakeoff/result.json` y `results/architecture_bakeoff_200k/result.json`—
no tienen `self_sha256`, ni `contract_path`, ni `contract_sha256`, ni `calibration_provenance`.

**Los he citado repetidamente como «artefactos sellados», y no lo son.** Peor: uno de ellos es la
fuente de la cifra que sostiene el único positivo neuronal del proyecto —
`run_track_b_nonneural_v1.py` lee `results/architecture_bakeoff_200k/result.json` para construir
`network_means_from_sealed_artifacts`, y el nombre del campo afirma algo falso.

## 2. Lo que este sellado SÍ hace, y lo que NO puede hacer

**Sí:** fija el contenido a partir de hoy. Escribe un **registro nuevo y aparte**
(`sealed_record.json`) que contiene una copia íntegra del artefacto, el digest SHA-256 de sus
bytes tal como están en disco, el manifiesto de módulos del runner, y su propio sello. A partir de
aquí, cualquier cambio del fichero original es detectable.

**No:** no certifica procedencia. Nadie puede sellar hacia atrás lo que no se selló en su momento.
**Este registro dice «esto es lo que el fichero contenía el 2026-08-07», no «esto es lo que produjo
la corrida del 2026-08-07 a las 07:15 UTC».** La diferencia importa y el manuscrito tiene que
citarla así: los resultados del bake-off son **de desarrollo, de procedencia no certificada**, y la
prima neuronal de `track_b_v1` necesita su confirmación en bloque virgen tanto por esto como por lo
que ya estaba dicho.

## 3. Por qué un fichero aparte y no una edición

**Nunca se edita un artefacto fechado en el sitio.** Añadirle `self_sha256` cambiaría sus bytes y
destruiría la única cosa que este registro puede aportar: el digest de lo que había. El original
queda intacto.

## 4. Por qué el runner no se arregla en el mismo commit

La sonda de reproducibilidad (`docs/PREREGISTRO_SONDA_REPRODUCIBILIDAD_BAKEOFF_2026-08-07.md`)
está ejecutando **dos réplicas en serie** de ese mismo runner. La segunda todavía no ha arrancado y
releería el fichero del disco: editarlo ahora haría que `A` y `B` corrieran **código distinto**, que
es exactamente la diferencia que la sonda existe para descartar.

**Compromiso:** en cuanto la sonda cierre, `run_architecture_bakeoff_v1.py` recibe `--contract`
obligatorio (sin defecto, como `run_meta_learner_over_configs_v1.py`) y `seal_and_write`.

## 5. Falsadores

| falsador | por qué puede fallar |
|---|---|
| `f1_the_embedded_copy_is_byte_faithful` | re-serializa la copia incrustada y exige el mismo digest que los bytes del fichero en disco. Falla si el registro guarda algo distinto de lo que dice guardar |
| `f2_the_original_is_not_modified` | vuelve a leer el original después de escribir el registro y exige el mismo digest. Falla si este script tocó el artefacto fechado |
| `f3_the_runner_really_does_not_seal` | busca `seal_and_write` en el runner. **Falla si sí sella**, y entonces esta enmienda no tiene razón de existir |
| `f4_the_downstream_consumer_rows_are_present` | exige que estén las claves que `run_track_b_nonneural_v1.py` lee (`by_arch` con `mean` por arquitectura). Falla si se sella un artefacto que no es el que alimenta la cifra |

**Alcance:** custodia de instrumento. No cambia ningún número, no adjudica nada.
