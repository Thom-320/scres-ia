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

## El hallazgo que esto deja, y no es un parche

> **La Fig. 5 de Garrido, tomada literalmente, no puede elegir la siguiente configuración.**

Sus dendritas son los cuatro drivers `d_i`, y un driver es una propiedad de un episodio **ya
corrido**. Su neurona, tal como está dibujada, **evalúa** —su activación es *«¿es ReT en la
configuración x mayor que en la x−1?»*, que se pregunta **después** de correr `x`— pero **no
planifica**. Para cerrar el lazo entre sus nodos ③ y ⑧ hace falta un modelo sobre `ρ`, que es
precisamente el nodo ③.

Eso es una observación sustantiva sobre su propuesta, y es material de §4.3 del borrador.

## Reproducción

Corrida corregida lanzada en el **VPS** (`ovh-agent-lab`, 6 vCPU, ocioso) →
`results/garrido_meta_learner_v2/result.json`. Smoke local de 2 réplicas ya con los seis
falsadores en PASA, y el efecto sobrevive al arreglo (memoria 6,25 vs reinicio 10,92). Los
números definitivos salen de la corrida del VPS, no del smoke.
