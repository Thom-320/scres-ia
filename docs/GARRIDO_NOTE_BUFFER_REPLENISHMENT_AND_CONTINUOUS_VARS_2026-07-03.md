# Nota para Garrido: reabastecimiento de buffers y variables continuas en el DES — 2026-07-03

## 1. Qué preguntó Garrido (reunión 2026-07-02)

En la reunión del 2 de julio ("Supply chain resilience — CVaR metric, decision variables, and
neural network integration with David"), Garrido preguntó explícitamente: **¿de dónde se
reabastecen los buffers durante la simulación?** Su preocupación, en sus propias palabras
resumidas: *"el agente no puede inventar inventario de la nada; si decide reabastecer un buffer
aguas abajo, debe incrementar shifts o producción aguas arriba."*

Esta es una preocupación legítima y, como se detalla abajo, **tenía razón**: el mecanismo que
usábamos en ese momento (la familia de experimentos "per-op buffer") sí tenía ese problema.

## 2. Qué encontramos

El mecanismo antiguo (`_top_up_inventory_buffer` en `supply_chain/supply_chain.py:693`) hacía
literalmente esto:

```python
shortfall = max(0.0, float(target) - float(container.level))
if shortfall > 0.0:
    return container.put(shortfall)
```

`container.put(shortfall)` añade unidades directamente al contenedor de inventario **sin
descontarlas de ningún origen aguas arriba ni verificar capacidad de producción/abastecimiento**.
Es decir: el agente podía pedir "sube el buffer de Op9 a X" y el simulador simplemente lo llenaba,
sin pasar por producción, sin consumir materia prima, sin respetar la capacidad de las operaciones
aguas arriba. Exactamente el problema que Garrido señaló.

## 3. Cómo lo arreglamos

Reemplazamos ese mecanismo por un contrato de acción que **modula los parámetros del sistema de
reabastecimiento que el simulador YA usa de forma físicamente consistente**: la política
periódica de orden (Q, ROP) que gobierna cuándo y cuánto se reordena en Op3 y Op9.

El agente ahora controla, de forma continua, 4 multiplicadores sobre parámetros existentes:

- `op3_q`, `op9_q`: multiplicador sobre la cantidad de orden (Q) de cada operación.
- `op3_rop`, `op9_rop`: multiplicador sobre el punto de reorden (ROP, reorder point).

No se inventó ningún mecanismo nuevo de creación de inventario: subir el ROP hace que el sistema
reordene antes (más colchón de seguridad vía órdenes reales, más frecuentes); subir Q aumenta el
tamaño del lote pedido. Ambos pasan por la cadena real de producción/abastecimiento del DES, que sí
respeta capacidad — no por un atajo que crea materia de la nada.

## 4. Qué pasó con la versión corregida

Con este mecanismo (conservación respetada), sí encontramos un **headroom estático real** que el
mecanismo antiguo nunca mostró: la mejor política estática (grilla fina, oráculo) supera a la mejor
constante simple por +0.0042 en Excel ReT (`outputs/experiments/track_a_v2_conservation_5d_gate_2026-07-03/`).
Es decir: el problema que Garrido plantea (reabastecimiento realista) sí abre una oportunidad de
mejora sobre políticas puramente constantes.

Pero: **tres intentos independientes y rigurosos de RL (PPO+behavior-cloning, critic-pretraining,
retune de GAE-lambda) fallaron en convertir ese headroom en una política aprendida ganadora** (0/5
seeds positivos en los tres casos). El headroom es real; el aprendizaje por refuerzo, con las
herramientas estándar que probamos, no lo aprovecha. Es un resultado negativo honesto y más preciso
que el original — no "no hay oportunidad", sino "hay oportunidad, pero RL estándar no la aprende".

## 5. Nuestra propuesta

**No recomendamos reabrir "buffers por nodo aguas abajo del CDC" como una variable nueva y
separada**, por dos razones:

1. Ya implementamos, con el mecanismo de conservación correcto, el equivalente funcional de esa
   idea (Q/ROP en los nodos existentes) y el aprendizaje falló tres veces de forma consistente.
   Añadir más nodos con el mismo tipo de variable (inventario/buffer) es muy probable que se tope
   con el mismo obstáculo — no hay evidencia de que el problema sea "pocos nodos", sino que el RL
   no logra explotar headroom de tipo inventario/buffer en general.
2. **Track B (autoridad de despacho en Op10/Op12, aguas abajo del CDC) ya es el resultado ganador
   confirmado** (+0.000438 Excel ReT, CI95 completamente positivo, 10 seeds). Captura la misma
   intuición de fondo que pedía Garrido — dar más autoridad al agente aguas abajo del CDC — pero a
   través de la variable que sí conecta con el cuello de botella real (capacidad de despacho), no
   inventario. Es la misma dirección estratégica, con evidencia de que funciona.

**Recomendación:** presentarle a Garrido el hallazgo del punto 4 como respuesta directa y honesta a
su pregunta (su preocupación era válida, la corregimos, medimos el headroom real, y documentamos
que RL no lo convierte), y señalar que el camino que sí funciona para "más autoridad aguas abajo"
es Track B. Esto cierra su pregunta con rigor sin gastar más cómputo en una variante de baja
probabilidad de éxito.

## 6. Sobre la duda de variables continuas vs. DES (Discrete Event Simulation)

Pregunta planteada: *si usamos variables de decisión continuas, ¿no estaríamos yendo en contra de
la naturaleza de una simulación de eventos discretos (DES)?*

**No hay conflicto, y la razón es que son dos cosas distintas:**

- **"DES" describe cómo avanza el RELOJ de la simulación**: el estado cambia en instantes de
  tiempo discretos y asíncronos, disparados por eventos (llegada de material, fin de producción,
  salida de un pedido), no por pasos de tiempo fijos ni por ecuaciones diferenciales continuas. Esa
  es la propiedad "discreta" del DES: el *mecanismo de avance de tiempo*, no el tipo de dato de las
  variables que alimentan las decisiones.
- **Las variables de decisión (acciones) son parámetros externos** que se evalúan en puntos de
  revisión periódica (en nuestro caso, semanal) y que alimentan la lógica interna del motor de
  eventos. Si esa variable es un número real (una fracción de buffer, un multiplicador de Q/ROP,
  un multiplicador de despacho) o un entero categórico (shift 1/2/3) es completamente ortogonal a
  cómo el simulador procesa sus eventos internos. El motor de eventos sigue siendo tan "discreto"
  como siempre.
- **Precedente en el propio trabajo de Garrido**: su tesis (2017) ya usa variables continuas
  (fracciones de buffer como f0.05, f0.10, f0.25...) dentro de un DES — prueba de que su propio
  diseño ya combina ambas cosas sin problema.
- **Garrido mismo lo recomendó**: en la reunión del 2 de julio sugirió explícitamente preferir
  variables continuas sobre discretas ("más gradiente, más fácil encontrar óptimos"). Implementar
  variables continuas está alineado con su propia sugerencia, no en contra de ella.

**Conclusión: la duda es comprensible pero no bloquea nada — DES y variables continuas de decisión
son ortogonales, y de hecho el propio Garrido prefiere continuas.**
