**Question Bank para la reunión**

**Si preguntan: “¿Por qué 5 dimensiones de acción?”**

Porque elegimos una interfaz compacta y pragmática sobre palancas operativamente plausibles: inventario upstream y downstream en `Op3` y `Op9`, más capacidad de ensamblaje mediante shifts. No estamos afirmando que sea un control total de las 13 operaciones, sino un benchmark tractable y operativamente interpretable.

**Si preguntan: “¿La propiedad de Markov se cumple?”**

La formulación correcta es que usamos una aproximación Markoviana práctica para control. El DES interno tiene un estado más rico que la observación expuesta al agente, así que reconocemos una caveat de observabilidad parcial. Lo relevante aquí es que la observación sea suficientemente informativa para benchmarking secuencial útil, no una prueba formal del sistema completo como MDP exacto.

**Si preguntan: “Entonces `ReT_thesis` ya no sirve?”**

Sí sirve, pero como métrica de evaluación. Lo que vimos es que no era la mejor reward de entrenamiento para control. Por eso separamos evaluación y aprendizaje.

**Si preguntan: “¿PPO ya ganó?”**

No lo presentaría así todavía. La formulación correcta es que los resultados son preliminares pero prometedores bajo una región estrecha del espacio de pesos, y que la validación fuerte está en curso con más seeds, más timesteps y stochastic processing times.

**Si David pide cross-validation loss**

En PPO no hay una cross-validation loss estándar como en ANN/GNN supervisadas. Lo correcto aquí es mostrar training reward, evaluación out-of-sample contra baselines fijos y métricas operativas como fill rate, backorder rate y mezcla de shifts.

**Si preguntan: “¿Qué sigue?”**

Cerrar las corridas largas en `increased` y `severe`, exportar los bundles auditables, y luego decidir si el régimen ganador es suficientemente robusto o si el resultado debe reportarse como local pero no global.
