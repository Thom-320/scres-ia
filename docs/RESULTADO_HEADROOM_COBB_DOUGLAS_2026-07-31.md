# Resultado — **no hay headroom ni con una métrica sana**, y ahora se sabe por qué

**Artefacto:** `results/headroom/cobb_douglas_v1/result.json` (sello `8ef2833be75bb736…`,
`NO_HEADROOM_EVEN_UNDER_A_SOUND_METRIC`) · **los siete falsadores PASAN** · preregistro
`docs/PREREGISTRO_HEADROOM_COBB_DOUGLAS_2026-07-31.md`, commiteado antes de correr.

## 1. La medición

Los mismos seis regímenes de la Fase 1A, el mismo barrido de nueve repartos, **las tres métricas
en una sola corrida y una sola cadencia**:

| métrica | `H_regime` | LCB95 | nivel |
|---|---:|---:|---:|
| **`R_cobb_douglas`** | **0,000000** | 0,000000 | 0,5495 |
| `ret_excel_risk_conditional` | 0,000000 | 0,000000 | 0,0027 |
| `flow_fill_rate` | 0,000000 | 0,000000 | 0,6880 |

**Exactamente cero en las tres.** Y la razón está a la vista, sin necesidad de interpretar:

| métrica | reparto óptimo, por régimen |
|---|---|
| `R_cobb_douglas` | **0,5 · 0,5 · 0,5 · 0,5 · 0,5 · 0,5** |
| `flow_fill_rate` | **0,5 · 0,5 · 0,5 · 0,5 · 0,5 · 0,5** |
| `ret_excel_risk_conditional` | **0,1 · 0,1 · 0,1 · 0,1 · 0,1 · 0,1** |

**El óptimo es invariante entre regímenes bajo las tres.** Conocer el régimen vale exactamente
nada porque **no hay nada que decidir**: el reparto equilibrado gana siempre, y la escalada de R23
—×3 en frecuencia, ×2 en impacto— no lo mueve.

## 2. Por qué esto es la conclusión fuerte, y no otra derrota

La objeción evidente contra veinte experimentos de `H_regime ≈ 1e-4` era: **«mediste con una
métrica rota»**. Hoy quedó demostrado que lo estaba — ReT prefiere el reparto que entrega el 50 %
de las raciones sobre el que entrega el 80 %.

Esta corrida cierra esa objeción:

> **Bajo Cobb-Douglas, que demostradamente NO premia el abandono —su óptimo coincide con el del
> servicio en los seis regímenes—, el headroom es exactamente cero.**

El «cuándo NO cerrar el lazo» deja de apoyarse en una métrica cuestionable. **Se sostiene sobre
dos instrumentos independientes que discrepan en todo menos en esto.**

## 3. Y `f1` deja una tercera evidencia contra ReT

`f1` exigía que Cobb-Douglas y ReT **discreparan** en el óptimo — si coincidían, la corrida no
tenía premisa. **Discrepan en los seis regímenes**: `0,5` contra `0,1`, sin una sola excepción.
Dos operacionalizaciones de «resiliencia» que eligen políticas opuestas en el 100 % de las celdas.

**Un cuarto dato, no buscado:** el `argmax` de ReT **cambia con la cadencia de paso**. En la
Fase 1A (`sim.run()`) era `0,9` para `R1r+R2r`; aquí (paso diario) es `0,1`. El de Cobb-Douglas es
`0,5` en las dos. La dependencia de cadencia de ReT ya estaba registrada; ahora se ve que llega a
**invertir la política que recomienda**.

## 4. Lo que esto cierra y lo que deja abierto

**Cierra:** la contención —aguas abajo y en el cuello, simétrica y asimétrica, fungible y no
fungible— **no produce una decisión dependiente del estado en esta cadena**, y el resultado no es
un artefacto del instrumento.

**Deja abierto, honestamente:**

* **La resolución de Cobb-Douglas.** `f5` pasa, pero su rango entre repartos es del orden del
  1 %. Ordena bien; no es un microscopio.
* **`H_regime` es el valor de conocer el RÉGIMEN para elegir una CONSTANTE.** No acota una
  política que conmuta dentro del episodio — esa corrección ya está en `4d7a173`. Lo que está
  cerrado es la contención como fuente de headroom, no toda forma concebible de control.
* **Fases 1B (presupuesto de expedición) y 1C (autotomía)** siguen preregistradas y sin correr.
  Atacan las otras dos causas del mapa, no ésta.

## 5. Para el paper

Esto y el meta-aprendiz son **los dos resultados centrales, y no compiten**: uno dice **dónde SÍ
paga la IA** en el flujo de trabajo de Garrido (aprender **entre corridas**: 6,31 corridas
ahorradas), y el otro dice **dónde NO** (cerrar el lazo **dentro** del episodio: cero, con dos
instrumentos). Juntos son una respuesta a sus dos preguntas, no un vacío.
