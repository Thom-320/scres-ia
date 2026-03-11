**Intro deck for Garrido meeting**

This outline is the non-technical opening. It should come before the static baseline table and before any PPO detail.

## Slide 1. From DES to adaptive control

**Title**

Del simulador DES al control adaptativo

**Bullets**

- El DES ya permite simular el sistema completo.
- El siguiente paso natural es usarlo como base para decisiones de control.
- El objetivo ya no es solo correr escenarios, sino aprender políticas.

**What to say**

Después de construir el DES, el siguiente paso natural no era solo seguir corriendo simulaciones, sino usar ese simulador como base para tomar decisiones de control. Es decir, pasar de evaluar escenarios a aprender políticas.

## Slide 2. Why DES alone is not enough

**Title**

El límite del DES puro

**Bullets**

- El DES puro permite comparar políticas fijas o reglas manuales.
- No aprende a ajustar decisiones según el estado operativo.
- Para control adaptativo se necesita una formulación secuencial.

**What to say**

Con DES puedo comparar políticas fijas, por ejemplo mantener siempre el mismo turno o usar reglas manuales. Pero si quiero que el sistema ajuste decisiones dinámicamente según el estado operativo, necesito formularlo como un problema secuencial de control.

## Slide 3. How the DES becomes an RL environment

**Title**

Del simulador al entorno de aprendizaje

**Bullets**

- Cada 168 horas el agente observa un resumen del estado.
- Luego toma una acción de control.
- El simulador avanza y devuelve una recompensa de desempeño.

**What to say**

Lo que hice fue envolver el simulador como un entorno tipo Gymnasium. Cada cierto intervalo de tiempo, el agente observa un resumen del estado del sistema, toma una acción de control, el simulador evoluciona, y luego recibe una recompensa según el desempeño operativo.

Versión aún más simple: en vez de correr la simulación pasivamente, ahora el simulador se vuelve interactivo: observa, decide, avanza y evalúa.

## Slide 4. Why PPO first

**Title**

Por qué usé PPO

**Bullets**

- Es un algoritmo estándar y robusto para control secuencial.
- Sirve como baseline serio antes de probar modelos más complejos.
- Permite validar primero el carril experimental.

**What to say**

Para la primera versión usé PPO, que es un algoritmo estándar de reinforcement learning para problemas de control. No escogí PPO porque fuera exótico, sino porque es una base robusta, conocida y adecuada para este tipo de interfaz.

## Slide 5. MLP and Stable-Baselines3

**Title**

Herramientas usadas: MLP + Stable-Baselines3

**Bullets**

- La política usa una MLP porque la observación es un vector compacto de variables operativas.
- No se necesitaban imágenes, transformers ni una arquitectura compleja en esta etapa.
- Stable-Baselines3 aporta una implementación estándar y confiable de PPO.

**What to say**

El agente usa una red neuronal simple, un multilayer perceptron, porque la observación del entorno es un vector compacto de variables operativas. Para este punto del proyecto, una MLP era la opción estándar y suficiente.

La implementación la hice con Stable-Baselines3, que es una librería estándar de Python para reinforcement learning. La ventaja es que ya trae implementaciones estables de algoritmos como PPO, así que no tuve que programar el algoritmo desde cero y pude concentrarme en el entorno, la reward y la evaluación.

Frase corta:

> SB3 me da el motor del algoritmo; yo me concentro en diseñar bien el problema.

## Slide 6. Current status

**Title**

Estado actual del carril RL

**Bullets**

- Primero validé que la acción de turnos sí cambia materialmente el sistema.
- Después separé la métrica de evaluación de la reward de entrenamiento.
- Ahora el agente ya compite con los mejores baselines fijos, aunque la evidencia sigue siendo preliminar.

**What to say**

Primero validé que la acción de turnos sí cambia materialmente el sistema. Después corregí la formulación de reward para separar la métrica de evaluación de la señal de entrenamiento. Con eso, el agente ya muestra políticas adaptativas competitivas con los mejores baselines fijos, aunque por ahora lo estamos tratando como evidencia preliminar y no como resultado definitivo.

## Transition to evidence slides

After these six slides, switch to:

1. `01_static_shift_baselines.png`
2. `03_best_regime_summary.png`
3. `04_policy_comparison.png`
4. `05_action_mix.png`
5. `02_ppo_training_reward_curve.png` only if David insists on “loss”
