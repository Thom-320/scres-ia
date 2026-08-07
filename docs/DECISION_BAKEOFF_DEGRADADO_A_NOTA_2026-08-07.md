# Decisión — el bake-off de arquitecturas baja a nota al pie; re-sellarlo no arreglaría nada

## 1. Las dos salidas que había

`docs/ENMIENDA_SELLADO_RETROACTIVO_BAKEOFF_2026-08-07.md` dejó el defecto documentado:
`run_architecture_bakeoff_v1.py` **nunca llamó a `seal_and_write`**, así que
`results/architecture_bakeoff/` y `results/architecture_bakeoff_200k/` no tienen `self_sha256`, ni
contrato, ni procedencia — y uno de ellos alimenta la única cifra neural del proyecto a través de
un campo que se llamaba `network_means_from_sealed_artifacts`.

Las salidas eran dos: **(a)** re-correrlo bajo el arnés de sellado, o **(b)** degradarlo a nota al
pie con el defecto declarado.

## 2. Por qué (a) ya no es una salida

`results/determinism_diagnostic/result.json` cerró en **`DEEPER_THAN_BOTH_ENVIRONMENT_LIMIT`**: las
tres configuraciones —8 envs con hilos libres, 8 envs con hilos fijados a 1, y **un solo env**—
difieren entre réplicas idénticas (deltas 0,472 · 1,620 · 1,539).

> **Re-correr el bake-off no reproduciría sus números: produciría una extracción distinta.**

Un sello nuevo certificaría la procedencia de **otro** artefacto, no la del que ya se citó durante
una semana. Sellar hacia atrás sigue siendo imposible, y ahora además sabemos que hacia adelante
tampoco converge. **(a) no repara nada; sustituye una cifra sin procedencia por otra igual de
irreproducible.**

## 3. La decisión

**El bake-off como política de control (A9) baja a nota al pie**, con el defecto declarado en la
misma frase que la cifra:

* `KAN − MLP = −0,475 [−1,548, +0,598]` — **sin separación**;
* KAN ≈ 4,1× más lenta por decisión;
* artefacto **sin sellar al ejecutarse**, procedencia no certificable, y producido por un arnés que
  hoy está medido como **no determinista** con dispersión de ±2,4 a semilla fija.

No entra en ninguna tabla del cuerpo del manuscrito. No sostiene ninguna comparación entre
arquitecturas.

## 4. Lo que arrastra consigo

`results/track_b_nonneural/result.json` lee de ese artefacto para construir la prima neural
(+1,44 a +2,18 sobre la mejor constante). Esa cifra queda **doblemente descalificada**:

1. procede de un artefacto sin procedencia certificable;
2. **cae dentro de la banda de ruido de ±2,4 del propio arnés**, y su artefacto **no le calcula
   ningún intervalo** — `network_minus_constant` es una resta pelada.

No es «sin confirmar». **El instrumento no la resuelve**, y ninguna cantidad de semillas vírgenes
lo cambia. Ése, y no la custodia ni el hueco A1, es el motivo terminal del `NO-GO` de C1.

## 5. Lo que SÍ sobrevive del trabajo de arquitecturas

`results/surrogate_architecture_bakeoff/` (A5) mide otra cosa —**la calidad de búsqueda**, no la de
control— y su contraste **sí** separa: `KAN − MLP emparejado = +0,01037 [+0,00302, +0,01893]`, con
el intervalo entero del lado desfavorable a la KAN. Ese es el que va al cuerpo, y sostiene la
contribución metodológica:

> **Ajustar mejor una superficie no implica buscar mejor sobre ella.**

## 6. Lo que queda arreglado para el futuro

`run_architecture_bakeoff_v1.py` ya exige `--contract` sin defecto y llama a `seal_and_write`. Toda
corrida futura nace sellada — lo que no arregla las dos ya emitidas, y por eso esta decisión existe.
