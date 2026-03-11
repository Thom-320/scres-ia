**Presentación corta: qué hice después del DES**

Esta versión está pensada para Garrido y David como introducción breve, sin entrar todavía en toda la parte técnica de benchmarking.

## Idea central

Después de validar el DES, el siguiente paso no fue inventar otra simulación, sino **ponerle encima una capa de decisión adaptativa**. La pregunta no era “si el DES funciona”, porque eso ya estaba bastante encaminado. La pregunta era:

> Si el sistema enfrenta disrupciones y cambios operativos, ¿podemos usar aprendizaje para ajustar decisiones de control mejor que una política fija?

Ese es el motivo de la interfaz RL.

## Qué hice, en lenguaje simple

1. **Tomé el DES existente como base**
   - No reemplacé el simulador.
   - No convertí todo el modelo en otra cosa.
   - El DES sigue siendo el corazón del proyecto.

2. **Lo envolví como un entorno de decisión secuencial**
   - Cada semana simulada, el agente observa un resumen del estado operativo.
   - Luego toma una acción.
   - El simulador avanza.
   - Se mide el efecto en servicio, backorders, costo y resiliencia reportada.

3. **Probé un baseline de RL simple primero**
   - En vez de saltar a una arquitectura exótica, empecé con PPO.
   - La política usa una `MlpPolicy`, o sea, un **Multilayer Perceptron** estándar.
   - La idea fue validar primero el carril experimental antes de hablar de arquitecturas más complejas.

## Qué es Stable-Baselines3

`Stable-Baselines3` es una librería estándar de Python para reinforcement learning.

En términos simples:

- ya trae implementaciones confiables de algoritmos como `PPO`, `SAC` y otros
- evita que yo tenga que escribir desde cero el algoritmo de entrenamiento
- me permite concentrarme en:
  - el entorno
  - la observación
  - la acción
  - la reward
  - la evaluación

Entonces, cuando digo que usé `Stable-Baselines3`, lo que significa no es “usé una caja negra mágica”, sino:

> usé una implementación estándar y reproducible del algoritmo, para que el foco metodológico estuviera en el entorno MFSC y en la formulación del problema.

## Qué es PPO, en versión no técnica

`PPO` (`Proximal Policy Optimization`) es un algoritmo de RL para aprender políticas de decisión.

En lenguaje simple:

- el agente prueba acciones
- ve la recompensa que obtiene
- y va ajustando su política poco a poco para no cambiar de forma demasiado brusca

Lo importante aquí no es la matemática interna del algoritmo, sino que:

- es un baseline serio y ampliamente usado
- funciona bien para empezar en problemas de control secuencial
- permite probar rápido si el entorno y la reward están bien planteados

## Qué es un Multilayer Perceptron aquí

El `MLP` es la red neuronal más simple del carril actual.

Aquí hace esto:

- recibe el vector de observación
- lo procesa en capas densas
- produce una acción

No modela explícitamente grafos ni relaciones estructurales complejas.  
Precisamente por eso lo usé primero:

> si ni siquiera un baseline estándar puede aprender algo útil, entonces el problema probablemente está en la formulación del entorno o la reward, no en la arquitectura.

Eso fue exactamente lo que ayudó a descubrir la auditoría de reward.

## Qué se aprendió con este enfoque

Este baseline simple ya permitió encontrar varias cosas importantes:

1. la acción de turnos sí importa
2. `ReT_thesis` no era buena reward de entrenamiento
3. hubo que separar:
   - `ReT_thesis` para evaluación
   - `control_v1` para entrenamiento
4. con esa corrección, PPO ya muestra políticas mixtas competitivas bajo ciertos regímenes

O sea:

> el MLP no fue el punto final del proyecto, pero sí fue una herramienta útil para validar el carril experimental y encontrar el cuello de botella real.

## Cómo hablar de David y DKANA sin conflicto

La forma correcta de decirlo es:

- **lo mío** fue construir y validar el baseline adaptativo sobre el DES con PPO/MLP
- **lo de David** es una arquitectura más sofisticada y estructurada, pensada como siguiente comparación o extensión

No son enemigos.
Cumplen funciones distintas:

- PPO/MLP = baseline funcional y benchmark inicial
- DKANA = arquitectura propuesta más compleja para la siguiente fase

## Frase corta para decir en la reunión

> Después del DES, mi objetivo fue abrir un carril de control adaptativo reproducible usando un baseline RL estándar. Por eso usé PPO de Stable-Baselines3 con una MLP simple: no para decir que esa es la arquitectura final, sino para validar el entorno, la reward y el protocolo experimental antes de movernos a modelos más complejos.

## Versión ultra corta de 20 segundos

> Después del DES, implementé una interfaz RL para convertir la simulación en un problema de decisión secuencial. Usé PPO de Stable-Baselines3 con una MLP como baseline inicial porque necesitaba una referencia simple, reproducible y seria antes de probar arquitecturas más complejas. Eso permitió validar el carril experimental y detectar que el problema principal estaba en la reward, no en el simulador.
