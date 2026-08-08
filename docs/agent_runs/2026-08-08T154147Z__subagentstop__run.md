# Agent run — SubagentStop

> **`UNADJUDICATED_DO_NOT_CITE`** — transcripción cruda de un agente, guardada por el hook
> `SubagentStop`. Puede contener afirmaciones **retiradas**; el canon vive en los artefactos
> sellados y en las enmiendas, nunca aquí.

- **event**: `SubagentStop`
- **session_id**: `fc8fefc1-10a6-4eea-bcbf-d92791120134`
- **agent_id**: `a6193c6b8a8e3824c`
- **cwd**: `<HOME>/Projects/research/scres-ia`
- **transcript_path**: `<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134.jsonl`
- **saved_at**: `2026-08-08T15:41:47.401321+00:00`
- **commit**: `71de444a`
- **answer read from**: `<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134.jsonl`

## Final response

`Re(APj)` ya está cerrado. Se corrió el **31 de julio**, contrato `PREREGISTRO_CIERRE_AUTOTOMIA_2026-07-31.md`, 12 raíces vírgenes, ambas familias, **los cinco falsadores pasan**. Y el resultado es mejor material que un cierre.

| brazo | `autotomy_share` R1r | `d_k` fidelidad | `ret_mean` R1r `d_k` |
|---|---:|---:|---:|
| A — constante 54 (statu quo) | 0,000000 | 12,40 | 2,18 |
| **F — olas de flete solas** | 0,655341 | **62,61** | **50,70** |
| FD — olas + δ | 0,000000 | 12,40 | 2,89 |
| **FDB — olas + δ + banda 0,05 h** | **0,003122** | **1,26** | 3,13 |
| Garrido (v4) | 0,004334 | — | — |

El brazo que yo te nombré —olas de flete solo— **sobredispara 151×** y arrastra la métrica a `d_k` 50,70. El que sí funciona (olas + δ + predicado de banda) baja la distancia de fidelidad de **12,40 a 1,26** y **no se adopta**, porque `ret_mean` empeora **+0,95 SE combinados** contra un `EPSILON = 0,5` declarado antes de correr.

**El hallazgo es ése: reproducir su autotomía degrada su propia métrica.** El precio de encenderla está medido — **1 parámetro** (una banda leída de *sus* filas, nunca ajustada a las nuestras) y 0,95 SE. Eso convierte una ausencia en una frontera, y una frontera es lo que se lleva a un paper. La cuenta honesta es **3 de 4 drivers vivos y uno apagado con su precio en la mano**.

---

## Tu pregunta de fondo: ¿headroom por demanda, riesgos y no-linealidad?

**La parte de no-linealidad ya la corrimos, y la respuesta es no.**

`results/functional_form_diagnostics/result.json` — pedido por el propio Garrido — en los seis contextos: RESET **rechaza linealidad**, AIC selecciona `quadratic_interactions`, y el R² held-out por semilla selecciona lo mismo. Los ΔAIC contra el lineal van de **1.300 a 14.416**. La superficie ya es materialmente no lineal.

Y con esa superficie el MLP **busca peor que el lineal** (`neural-premium-needs-curvature-above-noise`: curvatura 0,076 contra ruido 0,317). Así que la explicación *«KAN no gana porque la superficie es lineal»* está refutada con datos nuestros.

**La razón es una distinción que hay que separar, y es la clave de tu pregunta:**

> **Curvatura ≠ headroom.** La curvatura es una propiedad *estática* de la superficie de respuesta. El headroom es *condicional*: que **el óptimo se mueva con el estado**. Puedes hacer una superficie arbitrariamente retorcida y, si el argmax no se mueve, una constante bien elegida gana y ninguna red compra nada.

Eso es exactamente lo que midió Program O: `H_PI = 0,1515` bajo un recurso compartido **no fungible** — y **exactamente 0** al volverlo fungible. El mecanismo no era la forma de la superficie: era la contención.

## Dónde sí veo puerta, y es justo lo que Garrido pidió

**La demanda estacional de 2024 es física nueva, y es la primera en mucho tiempo que puede mover el óptimo.** Nuestra rejilla entera ha corrido sobre demanda estacionaria — `U(2400,2600)` cada 24 h, su §6.3.4. Un entorno estacionario **no puede** tener un óptimo que se mueva por demanda; es una imposibilidad estructural, no un resultado. Con `GR_{t+v}` de Holt y fase estacional, el buffer y los turnos óptimos plausiblemente **varían con la fase**. Eso es un óptimo móvil por construcción, que es literalmente lo que mide `H_regime`.

**Y el R2 sorteado dentro del episodio sigue abierto.** El screen de riesgos cerró la lectura de *perfiles fijos* (`6,93e-05` contra la barra 0,01), pero `docs/RESPUESTA_GARRIDO_R2_ALEATORIZADO_2026-08-08.md` deja explícito que **no cerró la realización estocástica intra-episodio**. Por nuestra propia regla — *«un negativo bajo la física vieja no es un negativo bajo la física nueva»* — ambas se re-testean.

## La trampa, que tú mismo insinuaste

*«Si creamos algo muy no lineal, ¿lograríamos que el RL tenga ventaja?»* — **Sí, siempre.** Se puede diseñar un entorno donde gane cualquier arquitectura. Eso no es un hallazgo sobre cadenas de suministro; es un hallazgo sobre el entorno que construimos, y un revisor de C&IE lo ve en una tarde. La revisión externa marca el mismo riesgo en la reunión: subir ruido y bajar capacidad esperando un 5–10 % de ventaja KAN es **elegir el entorno para producir el ganador deseado**.

El discriminante es limpio: la no-linealidad debe venir **de su fuente** —su generador de 2024, sus riesgos R2— y no de una perilla que giramos hasta que alguien gane. Lo primero es ciencia; lo segundo es un artefacto.

## Lo que propongo hacer ahora

**El gate de headroom bajo la física nueva, preregistrado, sin entrenar nada.** Es tu propia regla de `CLAUDE.md`: medir `H_regime` **antes** de gastar en un aprendiz.

- Demanda `GR_{t+v}` de Garrido 2024, reescalada para conservar la demanda media (para no confundir «más headroom» con «más carga»)
- R2 con frecuencia e impacto sorteados por episodio desde soporte congelado; misma realización contrafactual para todas las políticas
- Oráculo-con-régimen-conocido contra la mejor constante robusta, más el placebo desinformado
- **Si `H_regime < 0,01`: STOP, y no se entrena KAN ni MLP ni PPO**

Cuesta poco, es exactamente lo que Garrido pidió en sus dos frentes, y decide sin sesgo si hay algo que capturar. ¿Lo preregistro?

## Raw payload

```json
{
 "agent_id": "a6193c6b8a8e3824c",
 "agent_transcript_path": "<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134/subagents/agent-a6193c6b8a8e3824c.jsonl",
 "agent_type": "",
 "background_tasks": [
  {
   "command": "SP=<TMP>/-Users-thom-Projects-research-scres-ia/2c6b3a01-62a0-499d-957d-694424855dff/scratchpad\nuntil ssh -o ConnectTimeout=15 ovh-agent-lab 'pgrep -f vps_capture.py > /dev/null' 2>/dev/null; do sleep 30; done\nuntil ! ssh -o ConnectTimeout=15 ovh-agent-lab 'pgrep -f vps_capture.py > /dev/null' 2>/dev/null; do sleep 120; done\nscp -q ovh-agent-lab:/tmp/vps_capture.json \"$SP/vps_capture.json\"\necho \"=== CAPTURA DEL VPS LISTA ===\"\n.venv/bin/python -c \"\nimport json;d=json.load(open('$SP/vps_capture.json'))\nprint('host:',d['host'])\nfor s,v in d['slices'].items():\n    print(f\\\"  {s}: {v['n_differing']} celdas difieren de {v['cells']}\\\")\n\"",
   "description": "Watcher: VPS divergence capture",
   "id": "bj4tha3xz",
   "status": "running",
   "type": "shell"
  }
 ],
 "cwd": "<HOME>/Projects/research/scres-ia",
 "effort": {
  "level": "medium"
 },
 "hook_event_name": "SubagentStop",
 "last_assistant_message": "preregistra el gate de headroom con demanda estacional y R2 sorteado",
 "permission_mode": "bypassPermissions",
 "prompt_id": "55ed0ce1-6377-48e8-960d-a664710211a3",
 "session_crons": [],
 "session_id": "fc8fefc1-10a6-4eea-bcbf-d92791120134",
 "stop_hook_active": false,
 "transcript_path": "<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134.jsonl"
}
```
