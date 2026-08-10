# Preregistro — Puerta A2: el comparador que `f3` no supo comprobar

**Fecha:** 2026-08-09. **Congelado antes de escribir el runner.**
Contrato marco: `docs/CONTRATO_PROGRAMA_N_PRIMA_NEURAL_2026-08-09.md`.
Predecesor: `results/program_n/gate_a_track_b/result.json` = `NO_VALID_NONNEURAL_COMPARATOR`.

## 1. Qué falló, y era mi falsador

En la Puerta A la red ganó —**+0,4699 [+0,2372, +0,7024]** sobre la regla, 36 de 48 tapes— y el
veredicto fue `NO_VALID_NONNEURAL_COMPARATOR` porque `f3` exigía que la regla batiera a la
constante y no lo hizo: **+0,0246 [−0,0302, +0,0754]**.

`f3` existía para impedir un comparador de paja. Lo que ocurrió es lo contrario: **la constante
está saturada**. Ajusta en 98,21 contra 98,30 de la regla, así que la versión adaptativa no tiene
nada que mejorar. Escribí un falsador que **confunde «el comparador es débil» con «el comparador es
tan bueno que su versión adaptativa no aporta»**.

## 2. El arreglo, que son dos cambios y no uno

**(a) El mejor no-neuronal es un máximo, no un brazo por decreto.**

```
best_nonneural = argmax_{sobre el bloque de AJUSTE} { constante, umbral, realimentacion lineal, EWMA }
```

**(b) Su validez se comprueba contra un suelo absoluto, no contra otro brazo del mismo lado.**
El mejor no-neuronal debe batir a `random_action` y a `untrained_net` por margen. Eso **puede
fallar** —si toda la familia no neuronal colapsa a ruido— y no puede fallar por la razón espuria
que hundió la Puerta A.

## 3. Dos brazos estructurados nuevos, y uno de ellos usa memoria

La Puerta A dejó un hallazgo que este diseño tiene que atacar: **en `track_b_v1` una constante bien
buscada casi agota la clase no neuronal que sabíamos escribir**. Así que se amplía la clase:

| brazo | qué es | ¿usa historia? |
|---|---|---|
| `constant_best` | mejor constante de 8 dimensiones | no |
| `threshold_rule` | constante desplazada por señales del último frame | no |
| **`linear_feedback`** | `a = clip(W·obs_last + b)`, ley de realimentación lineal ajustada por CEM | no |
| **`ewma_rule`** | `a = clip(base + gain·tanh(EWMA_λ(señales)))`, con `λ` ajustado | **sí** |

**`ewma_rule` es el brazo que la Puerta A necesitaba y no tenía.** Allí la red batió a su placebo
congelado pero **no** al barajado, así que «tener historia ayuda, el orden no está demostrado». Un
filtro exponencial es la forma más simple de usar historia **con orden**, y es el comparador
honesto para cualquier afirmación de memoria.

## 4. El estimando

```
Delta_calidad = mean_tape( mlp - best_nonneural )
```

emparejado por tape sobre las **48 de evaluación**, bootstrap pareado de 20.000. SESOI `+0,01` del
contrato marco, que **no se mueve** después de ver el intervalo.

## 5. Semillas

**Bloque nuevo `9300001–9300120`**, verificado libre: cero colisiones con el registro y cero con
semillas de artefactos sellados. El bloque de la Puerta A (`9200001–9200120`) está **quemado** y no
se reutiliza.

* entrenamiento `9300001–9300005` · ajuste `9300011–9300034` · evaluación `9300051–9300098`.

## 6. Falsadores

| id | exige | por qué puede fallar |
|---|---|---|
| `f1_blocks_are_disjoint` | los tres bloques no se tocan | ajustar donde se mide infla todo |
| `f2_training_moved_the_policy` | la red entrenada bate a la no entrenada | 200k pasos pueden no bastar |
| `f3_nonneural_family_beats_the_floor` | el mejor no-neuronal bate a random y al no entrenado por margen | **el arreglo**: puede fallar si la familia colapsa, no si un brazo empata con otro |
| `f4_quality_premium_over_the_best_nonneural` | `LCB95 >= +0,01` | puede fallar, y con la clase ampliada es más fácil que falle |
| `f5_beats_the_memory_comparator` | bate a `ewma_rule` específicamente | **puede fallar**: es el brazo que usa historia con orden |
| `f6_beats_both_history_placebos` | bate a barajado y congelado | en la Puerta A el barajado ya cruzó cero |
| `f7_budget_is_matched` | parámetros dentro del 10 % | un presupuesto desigual mide capacidad |
| `f8_a_control_must_differ` | random pierde contra el mejor no-neuronal | un arnés que no distingue nada acuerda con todo |

## 7. Reglas de lectura

1. `f1` o `f3` fallan → `BLOCKED_INSTRUMENT` o `NONNEURAL_FAMILY_COLLAPSED`; nada más se lee.
2. `f4` + `f5` + `f6` → `TRACK_B_MEMORY_PREMIUM_CONFIRMED`.
3. `f4` + `f5`, sin `f6` → `PREMIUM_OVER_STRUCTURED_MEMORY_BUT_PLACEBOS_UNRESOLVED`.
4. `f4` sin `f5` → `PREMIUM_IS_CAPACITY_NOT_MEMORY`, que es lo que la Puerta A ya sugería.
5. Sin `f4` → `NO_QUALITY_PREMIUM_AGAINST_THE_WIDENED_CLASS`. **Es un resultado**, y sería el más
   informativo de todos: diría que la prima de la Puerta A sólo existía frente a una clase
   demasiado estrecha.

**No hay rama que diga «casi».**
