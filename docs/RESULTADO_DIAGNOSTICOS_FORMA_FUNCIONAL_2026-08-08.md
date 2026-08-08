# AIC y Ramsey RESET — corrimos el test que Garrido pidió, y refuta su hipótesis

**Fecha:** 2026-08-08 · **Petición:** Garrido, reunión 2026-08-07
**Artefacto:** `results/functional_form_diagnostics/result.json` (sello `4c240eb65906a757`,
fichero `504589fc230b86a0`) · **Instrumento:** `scripts/run_functional_form_diagnostics_v1.py`
**Clase:** diagnóstico de desarrollo sobre superficie sellada. Sin semillas, sin adjudicación.

---

## Su hipótesis, textual

> *«Para justificar por qué CAN no supera al MLP, se debe verificar si las variables tienen
> comportamiento lineal… Hipótesis de Garrido: las variables reflejan relaciones lineales, de ahí la
> poca diferencia entre modelos.»*

Es una hipótesis buena, comprobable, y **es falsa**. La comprobamos con las dos pruebas que él
nombró, en el lenguaje que pidió.

## Datos

Superficie base sellada de la confirmación de transferencia: **288 configuraciones × 6 contextos ×
60 semillas = 17.280 filas por contexto**. Respuesta: `ret_excel_risk_conditional`. Predictores: los
cuatro factores base. **Los contextos se ajustan por separado** — agrupar seis regímenes de riesgo en
una sola regresión fabricaría curvatura a partir de saltos de régimen y luego se la atribuiría a los
factores.

## Resultado — la linealidad se rechaza en los seis contextos

| contexto | R² fuera de muestra, lineal | R² quad+interacc. | **ganancia** | ΔAIC (lineal − mejor) | RESET F (pot. 2) | p |
|---|---:|---:|---:|---:|---:|---:|
| R1r | 0,5913 | 0,8212 | **+0,2299** | 14.416 | 2.234,2 | →0 |
| R1r+R2r | 0,5934 | 0,8014 | **+0,2080** | 12.545 | 2.463,2 | →0 |
| R1r\|esc | 0,5598 | 0,7932 | **+0,2334** | 13.179 | 1.928,2 | →0 |
| R1r+R2r\|esc | 0,5715 | 0,7638 | **+0,1923** | 10.495 | 1.008,8 | 3,3e−215 |
| R2r\|esc | 0,2778 | 0,3670 | +0,0892 | 2.361 | 383,7 | 1,6e−84 |
| R2r | 0,1697 | 0,2276 | +0,0578 | 1.300 | 427,6 | 7,6e−94 |

**AIC selecciona `quadratic_interactions` en los seis.** **El R² fuera de muestra, con folds cortados
por semilla, selecciona `quadratic_interactions` en los seis.** Los dos selectores coinciden, así que
no es sobreajuste dentro de muestra.

Y la magnitud no es marginal: en la familia R1r la no linealidad compra **entre +0,19 y +0,23 de R²
fuera de muestra** — cerca de un quinto de la varianza. En la familia R2r es menor (+0,06 a +0,09)
pero sigue siendo real.

> **La superficie no es lineal. Lo es de forma decisiva y materialmente.**

## Por qué esto refuerza el paper en vez de debilitarlo

Garrido propuso la linealidad como **explicación** del empate KAN/MLP. Su propio test la elimina.

Lo que queda en pie, y ahora mucho más fuerte:

- la superficie **sí** tiene estructura no lineal grande y verificable fuera de muestra;
- y **aun así** una KAN con parámetros emparejados **busca peor** que un MLP (+0,01037
  [+0,00302, +0,01893], donde menos es mejor);
- y una neurona de **cinco parámetros** gana a las siete arquitecturas del bake-off.

Antes podíamos decir «la red no paga». Ahora podemos decir algo que un revisor no puede devolver:
**la red no paga sobre una superficie cuya no linealidad está demostrada con el test que el propio
experto de dominio eligió.** «Vuestra superficie era demasiado fácil» deja de ser una objeción
disponible.

## Lo que estos diagnósticos NO establecen — va en el artefacto, no sólo aquí

- **AIC es un criterio relativo** entre modelos sobre la misma respuesta. Un AIC menor para el
  lineal no habría establecido que la superficie es lineal; sólo que los parámetros extra no se
  pagaron **en esta muestra**.
- **Un RESET no significativo es no-rechazo, no evidencia de linealidad.** Aquí rechaza, así que la
  asimetría nos favorece — pero la regla se declara igual.
- **Ninguno de los dos habla de la pregunta que hace el manuscrito**: si un aproximador más
  expresivo **busca** mejor. Ajuste predictivo y eficiencia de búsqueda secuencial son cantidades
  distintas, y este proyecto ya midió un caso donde apuntan en direcciones opuestas.
- Los contextos se ajustan por separado a propósito; cualquier afirmación agrupada confundiría
  curvatura de factores con salto de régimen.

## Falsadores

Los cuatro pasan, y cada uno puede fallar (regla R6):

| falsador | por qué puede fallar |
|---|---|
| `h1_the_factors_move_the_response` | este proyecto ya midió 4,56 M de unidades de materia prima moviendo **exactamente cero** ReT; un diseño inerte haría inútil todo lo demás |
| `h2_aic_compared_on_identical_rows` | AIC no tiene guarda interna contra comparar modelos ajustados a filas distintas |
| `h3_reset_is_computable` | una matriz de diseño casi singular vuelve degenerada la regresión auxiliar, y una excepción tragada parecería un no-rechazo limpio |
| `h4_both_selectors_were_compared` | falla si el script reportó un selector y no el otro |

## Dónde va en el manuscrito

**Suplemento**, bajo *«functional-form diagnostics requested by the domain expert»*, con la frase de
que no son árbitros al lado. La evidencia primaria sigue siendo desempeño fuera de muestra y regret
de búsqueda.

Pero el **hallazgo** —la superficie es no lineal y la KAN sigue sin pagar— pertenece a la discusión
de RQ3, porque cierra la explicación alternativa más natural que tiene un revisor.
