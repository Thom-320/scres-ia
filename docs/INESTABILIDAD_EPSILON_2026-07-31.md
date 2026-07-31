# La inestabilidad de `epsilon`: el barrido es demasiado ancho para pasar nunca

**Status:** `INSTRUMENT_DIAGNOSIS`. Nada cambiado.

## 1. Los dos términos de `dominates` se mueven en direcciones opuestas

`supply_chain/fidelity_moments.py:184-186`:

```python
no_worse = all(a[k] <= b[k] + epsilon for k in live)
strictly = any(a[k] <  b[k] - epsilon for k in live)
```

Al crecer `epsilon`, **`no_worse` se vuelve más fácil** (más pares dominan) y **`strictly`
más difícil** (menos pares dominan). El conjunto no dominado **no es monótono** en `epsilon`,
así que puede crecer o encoger — y de hecho hace las dos cosas en corridas distintas.

Visible en el mismo par: `LD` **empieza** a dominar a `A` al pasar `epsilon` a 1,0 y **deja**
de dominarlo a 2,0.

## 2. Los volteos los mandan brechas genuinamente marginales

| corrida / familia | par que voltea | momento crítico | brecha `d_k` |
|---|---|---|---:|
| enlace×atribución R1r | `L` domina a `A` en `eps` 2,0 | `rpj_p95` | **+1,09** |
| enlace×atribución R2r | `A` deja de dominar a `C` en 0,5 | `ret_mean` | **+0,06** |
| `δ` R1r | `D` deja de dominar a `A` en 2,0 | `rpj_mean` | **+0,11** |
| `δ` R1r | `D` deja de dominar a `LD` en 1,0 | `ret_mean` | **+0,05** |
| `δ` R1r | `L` deja de dominar a `LD` en 1,0 | `autotomy_share` | **+0,00** |

Brechas de **0,00 a 1,09**. El barrido va de **0,25 a 2,0 — un rango de 8×**. Con brechas de
ese tamaño, **el rango garantiza volteos**.

El caso de brecha **+0,00** es el más revelador: el momento es **idéntico** entre brazos, y
el volteo lo produce solo `strictly`, que exige una diferencia mayor que `epsilon` y no puede
cumplirse cuando la diferencia es cero.

## 3. La conclusión sobre el instrumento

**El barrido de `epsilon`, tal como está, no puede pasar cuando los brazos difieren en menos
de 2 `d_k`** — y nuestros brazos difieren en 0,05–1,09. Por eso **bloqueó las tres corridas
que discriminaban** y solo pasó en la que tenía tres brazos idénticos (`829151b`), donde no
había nada que voltear.

Un chequeo que solo pasa cuando no hay diferencias no está distinguiendo órdenes frágiles de
órdenes sólidas: **está bloqueando todo resultado discriminante**.

**Esto no invalida la regla del contrato maestro** —que un conjunto que se mueve con `epsilon`
se reporte inestable— pero sí dice que **el rango barrido es una elección nuestra y está mal
elegida**. `0,25–2,0` alrededor de un `EPSILON = 0,5` declarado es −50%/+300%.

## 4. Qué propondría, y no lo aplico

1. **Barrer una banda justificada** por la escala del problema: `EPSILON` declarado ±50%
   (0,25–0,75), no 8×. Un `epsilon` de 2,0 significa «ignoro diferencias de 2 errores
   estándar combinados», que es más de lo que separa cualquier par que nos interese.
2. **Reportar la tabla de volteos de §2 en lugar de un booleano.** Localiza *qué* comparación
   es frágil y *por cuánto*, que es información; el booleano solo dice «no mires».
3. **Excluir de `strictly` los momentos con brecha exactamente 0**, que no pueden aportar
   estricticidad y hoy solo generan volteos artificiales.

**No lo aplico** porque cambiar el rango del barrido después de que haya bloqueado tres
resultados es exactamente la clase de movimiento que estos contratos existen para impedir.
Requiere enmienda firmada, con el rango declarado **antes** de volver a mirar los veredictos.

## 5. Lo que esto sí resuelve

La inestabilidad **no era un defecto de las corridas ni de los brazos**. Era el instrumento
midiendo con una regla ocho veces más ancha que las diferencias que tiene que juzgar. Los
resultados de esas tres corridas siguen siendo válidos en todo lo demás — los conjuntos no
dominados en `EPSILON = 0,5` son los que se reportaron; lo único no sostenible es afirmar que
son robustos al rango barrido.
