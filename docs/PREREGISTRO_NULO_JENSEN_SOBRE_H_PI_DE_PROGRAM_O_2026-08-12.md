# Preregistro — el nulo de Jensen sobre el `H_PI` de Program O

**Fecha:** 2026-08-12 · **Escrito ANTES de correr el nulo.**
**Semillas:** ninguna. Re-análisis de matrices en disco de una corrida ya sellada.
**No es un rescate de Program O.** O está cerrado (`second_rescue_forbidden: true`) y es inmutable.
Esto no puede promoverlo; **sólo puede retirarlo**.

---

## 1. Por qué, y por qué ahora

`H_PI = 0,1515` es **la única cifra de headroom material que le queda a este proyecto**, ~1.000× la
escala `1e-4` del sobre nativo. Todo lo que hoy sigue vivo en la familia O/Q descansa sobre ella.

Y su forma es exactamente la que hoy sabemos que el ruido puro infla. De
`scripts/screen_program_o_full_des_hpi.py:270-277`:

```python
static_index = select_static(panel["ret_visible"])          # mejor calendario ÚNICO, por medias
safe_deltas  = ret_visible[tape, safe_argmax_por_tapa] - ret_visible[tape, static_index]
safe_h_pi    = safe_deltas.mean()
```

**Media-de-máximos menos máximo-de-medias**, con el máximo tomado sobre **65.536** calendarios por
tapa. Es la misma forma que ayer dio `+0,000065` observado contra un nulo cuya media era
`+0,003978`, sesenta veces mayor (`results/program_n/phase3_decision_headroom/result.json`). Y es
la misma forma que en 2026-08-08 retiró un techo clarividente que había pasado su nulo de
interacción con p=0,0132 sobre doce tapas.

**El nulo fungible de O no controla esto.** `exact_fungible_null_h_pi = 0,0` **exacto** sólo puede
salir si bajo fungibilidad los 65.536 calendarios son idénticos — es decir, si la varianza entre
acciones se anula. Sin varianza no hay sesgo de selección que acumular. **Es un nulo de física, no
de estimador.** El estimador nunca se ha testeado.

## 2. Precondición, ya satisfecha

Antes de tener derecho a testear nada, hay que reproducir la cifra sellada desde las matrices
crudas. Hecho, **bit a bit**:

```
raw_h_pi   reproducido 0.15275340823389597   sellado 0.15275340823389597
safe_h_pi  reproducido 0.15151378920653932   sellado 0.15151378920653932
```

sobre `outputs/program_o_runs/program-o-full-des-validation-v2-20260715/artifacts/validation/
raw_calendar_matrix/rho75_share90__centered_minority_v1/` (24 tapas × 65.536 calendarios).

## 3. El nulo

Dentro de cada tapa, **permutar el eje de calendarios**, aplicando **la misma permutación a todas
las métricas** de esa tapa. Eso conserva intacto el vector completo de métricas de cada calendario
—y por tanto la máscara de seguridad sigue siendo coherente— y destruye únicamente **la identidad
del calendario entre tapas**.

Bajo ese nulo:

* el **máximo por tapa no cambia** (permutar un vector no cambia su máximo);
* el **mejor calendario único** deja de ser informativo, porque las medias por columna se
  homogeneizan.

Lo que quede es exactamente **cuánto del hueco es seleccionar el máximo de 65.536 opciones
ruidosas**. Se recomputa `safe_h_pi` y `raw_h_pi` completos en cada réplica, con
`select_static` y `safe_mask` reales, no con un atajo.

**N = 1.000 réplicas**, semilla `20260812`, sobre el perfil primario
`rho75_share90__centered_minority_v1`. Se reportan también los otros perfiles presentes en disco.

## 4. Falsadores

* **n1_el_observado_supera_su_nulo** — `safe_h_pi` observado > p95 del nulo. **Puede fallar**, y si
  falla retira la única cifra de headroom material del proyecto. Es el objeto del ejercicio.
* **n2_la_reproduccion_es_exacta** — el estadístico recomputado sobre la matriz sin permutar debe
  coincidir con el sellado a `1e-12`. *Puede fallar* si leo las matrices mal, y entonces nada de lo
  demás vale.
* **n3_el_nulo_no_es_degenerado** — la desviación típica del nulo debe ser > 0. *Puede fallar* si la
  permutación no mueve nada, y entonces el test no testea.
* **n4_el_nulo_fungible_es_degenerado_como_se_afirma** — bajo el perfil fungible, la varianza entre
  calendarios debe ser ~0, lo que probaría que el nulo fungible **no puede** controlar el sesgo de
  Jensen. *Puede fallar*: si hay varianza y aun así H_PI da 0 exacto, mi diagnóstico es erróneo y el
  nulo fungible sí controlaba algo.

## 5. Reglas de decisión

| resultado | veredicto | consecuencia |
|---|---|---|
| observado > p95 del nulo | `H_PI_SURVIVES_ITS_JENSEN_NULL` | el premio es real; R1 (restricción en el objetivo) queda justificada |
| observado ≤ p95 | `H_PI_IS_SELECTION_BIAS` | se retira `H_PI = 0,1515` y **la familia O/Q se cierra por estimador** |

**Ningún resultado abre semillas ni promueve a Program O.** Un `SURVIVES` sólo autoriza pedirle al
PI el siguiente experimento; un `IS_SELECTION_BIAS` obliga a una retractación y a una arista de
supersesión.

## 6. Lo que este preregistro NO permite

Cambiar N, la semilla, el perfil primario o la definición del estadístico después de ver el
resultado. Ni presentar un `SURVIVES` como validación de Program O: O sigue cerrado y su conversión
observable segura sigue sin establecerse.
