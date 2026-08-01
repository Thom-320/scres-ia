# Corrección — el buscador leía configuraciones que no había corrido

**Afecta a:** `docs/RESULTADO_META_APRENDIZ_2026-07-31.md` y al artefacto
`results/garrido_meta_learner/result.json` (sello `230a0074a10f12ee…`).
**Origen:** revisión externa del plan «resolver Garrido 2024 con la tesis WRAP», cuyo punto 3
señala que aprender `drivers → ReT` es una tarea inválida porque la descomposición ya contiene la
respuesta. Fui a comprobarlo al código y el problema era peor de lo que el plan suponía.

## El defecto

`scripts/run_meta_learner_over_configs_v1.py`, al elegir la siguiente configuración:

```python
preds = [neuron.predict(features(CONFIGS[i], table[i][1])) for i in unseen]
```

`table[i][1]` es el **vector de drivers** de la configuración `i` **en este contexto** — una
propiedad de un episodio **ya simulado**. Para las configuraciones `unseen`, es decir las que la
estrategia todavía no ha corrido, **eso es leer la respuesta**.

## Y el falsador que debía cazarlo no podía fallar

```python
"f5_no_context_leakage": {"passed": True, ...}
```

`passed` estaba **hardcodeado**, y su «mecanismo» declaraba que *«la neurona sólo lleva `ρ`, nunca
filas»* — cierto sobre la memoria **entre** contextos, y **mudo** sobre el ranking **dentro** de
un contexto. Un falsador que no puede fallar no es un falsador, y esa es una regla que yo mismo
tengo escrita.

## Qué se retira y qué se mantiene

| afirmación | estado |
|---|---|
| **memoria vs reinicio** = +6,31 corridas [+5,18, +7,49] | **el estimando sobrevive** — los dos brazos leían exactamente lo mismo y sólo difieren en si `ρ` persiste — pero **el número se retira**: se midió bajo una búsqueda con fuga |
| **memoria vs OFAT** = +5,18 | **RETIRADO**. OFAT no recibe esa información; la comparación no era justa |
| **memoria vs aleatorio** = +12,31 | **RETIRADO**, misma razón |
| curva `H2` +0,00 → +10,00 | **forma retirada como medida**; el cero del primer contexto sigue siendo una comprobación estructural válida |
| `H4` como estimando (`ρ` persiste o no) | **se mantiene**, es el contraste que la fuga no distingue |

**Nada de esto toca los resultados de headroom.** La fuga vive sólo en este runner.

## El arreglo

1. **El modelo va sobre `ρ`** — las coordenadas de decisión, que es lo que un planificador tiene
   en el momento de decidir. Los drivers quedan como diagnóstico reportado.
2. **`f5` es ahora una prueba que puede fallar**: se replica la búsqueda entera sobre una
   **superficie sombra** con los vectores de drivers **permutados entre configuraciones**
   (valores intactos), y se exige que la **secuencia visitada** no cambie **ni un índice**.
3. **Se validó que el falsador caza el defecto**: se reintrodujo la fuga en una copia de control y
   se comprobó que `f5` falla con ella. Un falsador que nunca se ha visto fallar no está probado.

## Cómo lo interpretamos — y por qué el arreglo es MÁS fiel, no una desviación

**Su Fig. 5 es un concepto, no una especificación.** Su paper es explícitamente exploratorio
(*«the scope of this study is purely exploratory»*): dibuja una neurona cuyas dendritas son los
cuatro drivers `d_i`, ponderadas por `ρ`, con una activación del tipo *«¿es la medida de SCRES en
la configuración x mayor que en la x−1?»*. Lo que pide es **la idea**: poner un aprendiz entre
sus nodos ③ y ⑧ para que la cadena recuerde entre corridas. Operacionalizarla es **nuestro**
trabajo, y una lectura literal no sería fidelidad — sería pedantería.

**La interpretación que adoptamos, y la razón:**

| en su figura | en nuestro caso | por qué |
|---|---|---|
| dendritas `d_i` = los cuatro drivers | **la señal de actualización** tras cada corrida | un driver es una propiedad de un episodio ya simulado: es lo que el DES **reporta**, el nodo ⑧ |
| pesos `ρ` | **lo que el aprendiz retiene** entre configuraciones y contextos | es literalmente `L_{t−1}`, la variable de estado endógena que el borrador introduce |
| activación «¿ReT(x) > ReT(x−1)?» | **el gradiente** que compara la configuración actual con lo aprendido | su forma comparativa, hecha continua para que pueda entrenar |
| — | **entrada del modelo = las variables de decisión** | son el nodo ③, y son lo único que un planificador tiene **antes** de correr |

**La decisión de diseño que esto fuerza, dicha claramente:** su figura no distingue entre *lo que
el aprendiz observa* y *lo que el aprendiz usa para elegir*, porque a nivel conceptual no hace
falta. Al implementarla sí: **los drivers entran por la actualización, las variables de decisión
por la predicción.** Mezclarlas fue mi error — y es el error que un lector cuidadoso predeciría de
una implementación apresurada de esta figura, lo cual la convierte en una **nota metodológica
útil para el paper**, no en una objeción a su propuesta.

Eso es material de §4.3, redactado como *«así se operacionaliza su Fig. 5, y ésta es la
distinción que hay que hacer al bajarla a código»*.

## Reproducción

Corrida corregida lanzada en el **VPS** (`ovh-agent-lab`, 6 vCPU, ocioso) →
`results/garrido_meta_learner_v2/result.json`. Smoke local de 2 réplicas ya con los seis
falsadores en PASA, y el efecto sobrevive al arreglo (memoria 6,25 vs reinicio 10,92). Los
números definitivos salen de la corrida del VPS, no del smoke.
