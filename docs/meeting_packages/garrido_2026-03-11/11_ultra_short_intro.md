**Ultra-short intro (90 seconds)**

Después de construir el DES, el siguiente paso fue usarlo no solo para simular escenarios, sino como base para tomar decisiones adaptativas. Con el DES puro uno puede comparar políticas fijas, pero si uno quiere que el sistema ajuste decisiones según el estado operativo, toca formularlo como un problema secuencial de control.

Entonces lo que hice fue envolver el simulador como un entorno tipo Gymnasium: el agente observa un resumen del estado, toma una acción, el simulador avanza y luego recibe una reward según el desempeño.

Para la primera implementación usé PPO, porque es un algoritmo estándar y robusto para control. Y para no implementar PPO desde cero, usé Stable-Baselines3, que ya trae una implementación confiable del algoritmo.

La política usa una red neuronal simple, un multilayer perceptron, porque la observación del entorno es un vector compacto de variables operativas. No necesitaba una arquitectura más compleja en esta etapa.

Lo importante no es tanto la librería, sino que ya tengo el DES convertido en un entorno de decisión, con una reward operativa corregida y con resultados preliminares donde el agente ya compite con los mejores baselines fijos.
