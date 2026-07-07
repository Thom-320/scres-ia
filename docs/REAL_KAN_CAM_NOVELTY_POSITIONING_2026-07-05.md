# Real-KAN/CAM como novedad arquitectónica defendible — 2026-07-05

## Decisión de encuadre

Para el paper Q1, la ruta más sólida es separar dos mensajes:

1. **Spine empírico y reviewer-safe:** PPO+MLP sobre Track B 8D, idealmente sin forecast si la corrida final fixed-RNG confirma que queda suficientemente cerca. Este es el resultado más limpio para demostrar aprendizaje adaptativo sobre la métrica ReT Excel de Garrido.
2. **Sidecar de novedad arquitectónica:** Real-KAN/CAM como una arquitectura interpretable que puede conectarse al mismo DES, al mismo contrato de acción y a la misma métrica ReT Excel. Aquí está la respuesta a la preocupación de Garrido: no dependemos de un MLP genérico como única red neuronal; también tenemos una arquitectura moderna, spline-based e inspeccionable.

No conviene vender Real-KAN como reemplazo automático del spine mientras su ganancia en ReT sea marginal y su costo operacional sea alto. Sí conviene venderlo como **evidencia de novedad metodológica e interpretabilidad**.

## Qué tiene de novedoso Real-KAN/CAM aquí

La literatura KAN reciente propone reemplazar las activaciones fijas de un MLP por funciones univariadas aprendibles, usualmente splines, en las conexiones. La promesa no es solo performance: es que esas funciones pueden visualizarse, podarse o incluso aproximarse simbólicamente.

En nuestro repositorio, eso ya está implementado de forma real, no decorativa:

- `scripts/real_kan_extractor.py` usa la clase oficial `kan.KAN` de pyKAN dentro de un `BaseFeaturesExtractor` de Stable-Baselines3.
- La política sigue siendo PPO, pero el extractor de features ya no es un MLP denso puro: es una red Kolmogorov-Arnold con splines aprendibles.
- Durante entrenamiento se apagan `save_act` y `symbolic_enabled` por velocidad; después se reactivan sobre checkpoints congelados para inspeccionar qué aprendió.

## Qué evidencia local tenemos

La auditoría `docs/REAL_KAN_INTERPRETABILITY_VERDICT_2026-07-04.md` muestra que Real-KAN aprende patrones no triviales:

- El top-10 de variables concentra 47.7% de la atribución total sobre 52 variables, más de 2.5x lo esperado por azar.
- Las variables más importantes son operacionalmente sensatas: `rations_theatre_norm`, `backorder_rate`, `cum_downhours_fraction`, estados de régimen, `r14_defect_prob`, `backlog_age_norm`.
- La spline de `rations_theatre_norm` muestra una forma de umbral-y-rampa: plana en baja presión, subida marcada cuando la presión cruza cierto nivel.
- Variables sin señal relevante aparecen con atribución cero y curvas planas/vacías.

Lectura: Real-KAN no está “ganando” solo por ruido. Está aprendiendo una representación interpretable de presión operacional y recuperación.

## Resultado cuantitativo actual

Bajo fixed-RNG 5-seed:

- PPO+MLP fixed-RNG: ReT Excel `0.00592064`, costo `0.700`.
- Real-KAN fixed-RNG: ReT Excel `0.00594620`, costo `1.000`.
- Delta Real-KAN vs PPO+MLP: `+0.00002556`, positivo en 5/5 seeds.

Bajo el paquete Real-KAN 10-seed previo:

- Real-KAN gana marginalmente a PPO+MLP en ReT Excel, con CI positivo.
- Pero compra esa ganancia con uso de recursos sustancialmente más alto.

Por eso el encuadre honesto es:

> Real-KAN/CAM confirma que una arquitectura interpretable tipo Kolmogorov-Arnold puede integrarse al DES y producir una política competitiva en ReT Excel. Su valor principal para este paper no es desplazar al MLP eficiente, sino aportar novedad, interpretabilidad y una lectura mecanística de qué variables usa la política.

## Cómo presentarlo a Garrido

Mensaje corto sugerido:

> “Para evitar una crítica de falta de novedad, no nos quedamos solo con PPO+MLP. También implementamos una variante Real-KAN/CAM usando una librería existente y conectada directamente al mismo entorno. El resultado es interesante: mejora ligeramente la resiliencia pura, pero usa más recursos. Por eso no la pondría como política principal eficiente todavía; la pondría como evidencia de que una arquitectura interpretable y novedosa funciona en nuestro caso. Además, KAN nos permite mirar splines y atribuciones, es decir, ver qué señales aprendió la red.”

## Qué no decir

Evitar estas frases:

- “KAN prueba que el paper es novedoso.”
- “Real-KAN reemplaza a PPO+MLP.”
- “CAM/KAN es preventivo.”
- “La arquitectura explica causalmente todas las decisiones.”

Mejor decir:

- “Real-KAN/CAM es un sidecar arquitectónico confirmado.”
- “Aporta interpretabilidad y novedad metodológica.”
- “Es competitivo en ReT Excel, pero su trade-off operacional es más caro.”
- “La prevención causal no quedó demostrada bajo el reward actual.”

## Referencias externas útiles

- Liu et al. (2024/2025), **KAN: Kolmogorov-Arnold Networks**, ICLR 2025. Define KAN como alternativa a MLP con funciones aprendibles en las conexiones y énfasis en interpretabilidad.
  URL: https://arxiv.org/abs/2404.19756
- Ji et al. (2024), **A Comprehensive Survey on Kolmogorov Arnold Networks**. Resume aplicaciones, interpretabilidad, visualización, pruning y simbolificación.
  URL: https://arxiv.org/html/2407.11075v7
- Huang et al. (2025), **Interpretable deep reinforcement learning with Kolmogorov-Arnold networks for autonomous driving**. Muestra una línea cercana: KAN dentro de DRL para mejorar interpretabilidad.
  URL: https://www.sciencedirect.com/science/article/pii/S0968090X25003900
- Tsang et al. (2025), **Can Kolmogorov-Arnold Network (KAN) Replace MLP in Reinforcement Learning for Stochastic Inventory Control?**. Es especialmente relevante porque conecta KAN, RL e inventario.
  URL: https://dl.acm.org/doi/10.1145/3760622.3760623

## Próximo paso recomendado para Real-KAN/CAM

No mezclarlo con no-forecast todavía. Primero cerrar el spine no-forecast 15-seed. Después:

1. Extender Real-KAN fixed-RNG de 5 a 10 seeds si Garrido quiere una tabla arquitectónica más fuerte.
2. Añadir una figura pequeña de interpretabilidad: top variables + una spline umbral-rampa.
3. Presentarlo como “arquitectura interpretable alternativa” y no como “nuevo spine”.

