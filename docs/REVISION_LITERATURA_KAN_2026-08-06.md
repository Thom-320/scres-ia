# Revisión de literatura — el preprint de KAN y el hueco de novedad

**Compromiso de Garrido, reunión del 28 de julio, con fecha límite del 3 de agosto.** Se entrega el
6 de agosto, con **tres días de retraso**, y así queda dicho.

## 1. La pregunta que hizo, y la respuesta

> *«Arxiv es preprint; buscar si está publicado en un journal reconocido.»*

El trabajo es **«KAN or MLP: A Fairer Comparison»**, Runpeng Yu, Weihao Yu y Xinchao Wang,
[arXiv:2407.16674](https://arxiv.org/abs/2407.16674).

| campo | contenido |
|---|---|
| enviado | 23 de julio de 2024 |
| última revisión | 17 de agosto de 2024 |
| **Journal reference** | **ausente** |
| **DOI (no-arXiv)** | **ausente** |
| **Comments** | **«Technical Report»** |

**Sigue siendo preprint, y lleva dos años sin publicarse.** Sus propios autores lo etiquetan como
informe técnico, no como artículo enviado a revisión.

## 2. Por qué eso NO nos ayuda tanto como parece

La tentación es usar el estatus de preprint para desestimarlo. **Sería un error, y quedaríamos
expuestos**, porque su hallazgo lo **replicamos nosotros de forma independiente y en nuestra propia
cadena**:

| medición nuestra | resultado |
|---|---|
| bake-off a 200.000 parámetros igualados, `track_b_v1`, semillas 9491–9495 | **KAN − MLP = −0,475 [−1,548 · +0,598]** — nada separa |
| coste por decisión, mismo host | KAN **2,82 ms** vs MLP **0,69 ms** = **4,1×** |
| sustitutos dentro de la búsqueda, presupuesto igualado | KAN 380 par. y MLP 369 par. **empatan con una neurona de 5 parámetros**; KAN cuesta **34×** por decisión |

**No podemos atacar el preprint por su venue cuando nuestros propios números lo confirman.** La
posición defendible es la contraria y es más fuerte: *«coincidimos con ellos, en un dominio que
ellos no tocaron, y por eso nuestro aporte no es que la KAN gane.»*

## 3. El detalle de su ablación que sí deberíamos usar

Su hallazgo fino: **la ventaja de la KAN en representación simbólica viene de la activación
B-spline**, y **al ponerle B-splines a un MLP, el MLP iguala o supera a la KAN**. Es decir, lo que
funciona no es la arquitectura sino la base de funciones.

Eso encaja exactamente con lo que medimos en tres frentes distintos: **el ingrediente es la
retención, no el aproximador.** Es el mismo tipo de resultado, y podemos citarlos como apoyo en vez
de como adversario.

## 4. El hueco de novedad — sigue abierto, y es la buena noticia

Búsqueda cruzada de **KAN × resiliencia de cadena de suministro × simulación de eventos discretos**:
hay trabajo reciente en resiliencia con DES ([Ivanov, *ITOR*
2026](https://onlinelibrary.wiley.com/doi/10.1111/itor.13612); [simulation-driven optimization,
*C&IE* 2026](https://www.sciencedirect.com/science/article/pii/S0360835226002640); [límites
metodológicos, *JSSSE* 2025](https://link.springer.com/article/10.1007/s11518-025-5642-3)) y hay
KANs aplicadas a física y a análisis de supervivencia — **pero ninguna publicación combina los
tres**.

**El argumento de novedad de Garrido se sostiene.** Y conviene notar que la revista que él puso como
objetivo principal, *Computers & Industrial Engineering*, **ya está publicando optimización guiada
por simulación para cadenas resilientes en 2026**: el hueco es real y la revista es receptiva al
tema.

## 5. Lo que esto obliga a decir en la reunión

El argumento de venta de la KAN tenía dos patas: **ahorro de parámetros** e **interpretabilidad**.

* **La primera está medida y no se sostiene** — a parámetros igualados no gana, y cuesta 4,1×.
* **La segunda no la hemos tocado.** No hemos producido ni un artefacto de interpretabilidad, y es
  precisamente donde el preprint **no** contradice a Garrido, porque su comparación es de exactitud
  y no de auditabilidad.

**Si la KAN va a entrar al paper, tiene que entrar por la interpretabilidad, y eso hay que
construirlo.** Vender ahorro de parámetros es vender lo único que ya medimos que es falso.

---

**Pendiente continuo:** monitorear literatura nueva de KAN + resiliencia. Esta revisión es del
6 de agosto de 2026 y hay que repetirla antes de enviar.
