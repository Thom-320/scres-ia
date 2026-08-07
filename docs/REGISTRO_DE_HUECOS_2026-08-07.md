# Registro de huecos abiertos — 7 de agosto de 2026

Todo lo que hoy quedó **sin cerrar**, con qué lo cerraría y qué se rompe si no se cierra. Escrito
para que ninguno sobreviva por olvido.

## A · Huecos que invalidan una afirmación si no se cierran

### A1 · `worst_product_fill` nunca se aplicó en el paso 3
El preregistro lo nombra guardarraíl bloqueante; el runner sólo persiste `flow_fill_rate`, un
agregado que **no ve un producto abandonado**. Consecuencia: `NO_STRUCTURED_CONTROLLER_CONVERTS` es
diagnóstico de desarrollo y **no define el residual neural**, y no se puede adjudicar DDMRP.
**Cierre:** que el runner persista fill por producto y re-correr. ~5 h.

### A2 · Procedencia rota de Paper 2
`supply_chain/supply_chain.py` lleva 12+ commits desde el manifiesto del 14-jul, con cambios de
física y métrica (`64b75ce` manda 145 pedidos a cero; `ea246ac` «at double the ret_mean error»).
Re-congelar sería afirmar algo falso. **Cierre: decisión del PI** — re-correr los artefactos bajo la
física actual, o retirar por escrito los claims que anclan.

### A3 · El signo del lane MPC no es estable
Cambia con endpoint, incumbente y bloque de tapes (auditoría verificada). **Cierre:** fijar
endpoint primario, incumbente y SESOI **antes** de la siguiente campaña, y correr las cuatro
combinaciones para acotar la sensibilidad.

### A4 · `sumBt` sigue sin interpretación válida
Ninguna convención reconstruye la columna en más del 1,09 % de 47.780 filas. La reproducción exacta
de la fórmula Excel está probada; **la reproducción conductual del DES no**.
**Cierre:** preguntarle a Garrido qué es `sumBt`. Es `blocked_domain_fact`, y lo decidimos nosotros
si él no responde.

## B · Huecos de instrumento

### B1 · MPC no es construible en `track_b_v1`
`copy.deepcopy(env)` falla con `cannot pickle 'generator' object`: simpy guarda generadores. Sin
ramificación no hay MPC de disparo. Las dos salidas —replay cuadrático sobre 104 decisiones, o MPC
sobre modelo aprendido— son caras o circulares.
**Cierre recomendado:** reportarlo como hallazgo metodológico. Explica por qué el MPC «empataba
fácil» sólo donde el horizonte era corto.

### B2 · El barrido de variantes corrió con hiperparámetros distintos
`n_steps` 2048 en vez de 512 y `ent_coef` 0,0 en vez de 0,01. Los contrastes **dentro** del barrido
sobreviven; los **entre** corridas no. Ya retirado el «el presupuesto compra 5,4 puntos».
**Cierre:** la reconfirmación a 200k, ya corriendo con hiperparámetros igualados.

### B3 · 20 tests siguen en rojo
De 21, uno arreglado. La suite pasó de 0 el 31-jul a 21 hoy. Una suite en rojo es una suite que
nadie lee — y así sobrevivió el fallo de custodia.
**Cierre:** una pasada de triaje. `docs/TRIAJE_SUITE_21_FALLOS_2026-08-06.md` tiene el desglose.

### B4 · `ret_excel` sigue siendo el default del runner del paso 3
Lo pasamos explícito en los cuatro shards, pero **el default sigue premiando el abandono**. Es una
trampa esperando al siguiente que lo invoque.
**Cierre:** hacerlo obligatorio, como se hizo con `--contract`.

## C · Resultados que existen pero no están confirmados

### C1 · `NEURAL_PREMIUM_LIKELY_IN_TRACK_B`
Las redes baten a la mejor constante por **+1,44 a +2,18**. Es el primer positivo neural del
proyecto. **Necesita confirmación en bloque virgen**, y el registro central está en
`NO_NEW_SEEDS_AUTHORIZED`. **Cierre: firma del PI** sobre un bloque nuevo.

### C2 · `nhead4` y `1layer`
Separan a 100k bajo hiperparámetros equivocados. La reconfirmación a 200k corre **sobre las mismas
semillas de desarrollo**, así que contesta «¿aguanta en convergencia?» y **no** «¿replica en datos
nuevos?».

### C3 · La interpretabilidad de la KAN es de una sola partición
Sin validación cruzada, sin estabilidad de formas entre folds. Las curvas son **cortes de
respuesta**, no las funciones de arista internas.

### C4 · El holdout v0 se abrió contra su preregistro
Replicó el STOP, así que no cambió nada — pero el bloque está gastado y registrado como
`BURNED_OPENED_AGAINST_PREREGISTRATION`.

## D · Lo que no se ha tocado

### D1 · `program_l_route_recourse_env` — **el más prometedor**
`Discrete(3)` HOLD/ROUTE_1/ROUTE_2 sobre el DES completo, 21 observaciones. **Cero resultados en
`results/`.** Su propio docstring dice que existe para correr el gate de headroom pre-aprendiz.
Es la única decisión del repositorio que **sólo existe si algo falla**, así que una constante no
puede ser competitiva por construcción — lo contrario de lo que acabamos de medir.
**Desbloqueado hoy:** los tapes se construyen con `materialize_tape(seed, family, n, label,
contract_id="program_e_policy_realizability_v1", tape_prefix=...)`.

### D2 · Capacidad finita en `wdc`/`al`/`sb`
Los helpers existen en `node_capacity.py`; los **niveles** están cableados, la **capacidad** no.

### D3 · `event_triggered_env`, `v2_preventive_env`, `dra2_policy_env`
Sin tocar en esta campaña. El primero es el único que prueba si el óptimo varía **en el instante de
un choque** — distinto de variar entre periodos fijos, que es lo que medimos y cerramos.

### D4 · El manuscrito de C&IE
No escrito. Todos los números existen.

## E · Fuera de nuestra máquina

### E1 · La campaña del VPS no tiene suelo
Seis entrenamientos de `train_program_b_service_safe_learner.py` —`PPO_MLP` y `RecurrentPPO_MLP`,
semillas 8201–8203, 10 h— comparan **dos redes entre sí**. Es exactamente el error que acabamos de
corregir aquí: sin comparador constante, un empate o una victoria no dicen si hace falta una red.
**Cierre:** correr el mismo suelo constante→umbral sobre ese entorno antes de leer sus resultados.

### E2 · La discrepancia de 4,2 puntos con David
Probablemente B2, pero **no confirmado**. Su cuaderno genera `david_export.json` con `n_envs`,
`torch`, `device_chosen` y el hash de sus objetos editados. **Cierre:** pedirle ese fichero.
