# Veredicto sobre los anchors de contenido (clase C)

**Fecha:** 2026-08-24 · **Estado:** diagnóstico completo; re-congelado pendiente de
decisión del PI.

## Resumen

| Anchor | Golden congelado | Calculado hoy | En el commit que lo congeló |
|---|---|---|---|
| CSSU bridge (`3b70dd9`) | `f3fe61b1…` | `9cb65c7a…` | **`9cb65c7a…` — el golden nunca coincidió** |
| LOC graph (`852c64b`) | `371c5975…` | `eb65748e…` | no verificado directamente; misma clase |
| Program J audit (`ed27dde`) | pin `d8fd9347…` (supply_chain.py en `5cb8fb8`) | `2f348e59…` | el pin quedó viejo por 4 commits posteriores legítimos |

## Lo probado, paso a paso

1. Los ficheros de física (`supply_chain.py`, `episode_metrics.py`,
   `scientific_payload.py`, `config.py`) están **bit-idénticos** desde `3b70dd9`
   hasta HEAD: `git diff 3b70dd9..HEAD -- supply_chain/` está vacío.
2. El venv actual cumple los pins exactos del repo (`simpy 4.1.2`,
   `numpy 2.4.6`, `scipy 1.17.1`, `pandas 3.0.3`, `SALib 1.5.2`).
3. Extraje el árbol exacto de `3b70dd9` con `git archive` y computé el payload en
   ese árbol con el mismo intérprete: da **`9cb65c7a…`, no el golden
   `f3fe61b1…`**.

## Conclusión

**El golden del anchor CSSU se congeló con un valor que su propio commit jamás
produjo.** No hay drift ambiental ni regresión posterior: el hash fue tomado de
un árbol de trabajo o entorno distinto al que quedó commiteado. El propio mensaje
del test anticipa este caso y exige «re-freeze in a commit that says why».

Para LOC bridge y Program J audit la evidencia es consistente con la misma clase:
el audit de J ata el hash completo de `supply_chain.py`, que cambió
legítimamente en 4 commits después de la última re-attestación (`ed27dde`),
mientras sus otros 8 pins siguen correctos y los resultados congelados
(`verdict.json`, `raw_matrices.npz`) también.

## Camino correcto (herramientas ya existen)

El repo tiene `scripts/reattest_source_pins.py`, creado el 2026-07-31
precisamente para esto, que:

- propaga el nuevo hash por el DAG completo (dos clases de arista:
  `content_sha256` y sha256 whole-file), iterando a punto fijo;
- **exige** `--cause`: un artefacto que pruebe que el cambio es
  behaviour-preserving;
- no reformatea los JSON (compara conteo de líneas);
- deja intactos los freezes de ejecución de Program O, que deben quedar viejos.

Lo que falta es el `--cause` para esta ronda, y eso es una decisión del PI:

1. Para **Program J / K3**: la causa existe y está documentada — los cambios
   posteriores son aditivos (216+/6−, parámetros default-off, sin tocar la física
   por defecto) y los resultados congelados siguen cuadrando. Falta formalizarla
   como artefacto de preservación conductual (como hizo Thommy el 2026-07-31 con
   61 métricas × 10 configuraciones a cero diferencias).
2. Para **CSSU/LOC**: el golden nunca fue correcto; re-congelar implica admitirlo
   explícitamente y volver a quemar el anchor contra la física actual verificada.

No ejecuté ninguna re-attestación: mover un pin sin causa probada es falsificar
procedencia, según la política del propio script.
