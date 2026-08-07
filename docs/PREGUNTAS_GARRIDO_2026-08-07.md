# Cinco preguntas para Alexander Garrido — hechos de dominio, no de método

Cada una es un **hecho externo** que sólo él puede dar y que nosotros no podemos medir. Ninguna
pide su opinión sobre nuestros resultados, y **ninguna rescata nada del pasado**: todas definen
endpoints de campañas futuras. Están ordenadas por lo que desbloquean.

---

## 1. `sumBt` — la única que amenaza la validez de todo lo demás

**La pregunta:** en el libro de Excel de la tesis, ¿qué es exactamente la columna `sumBt`? ¿Es un
acumulado de pedidos en backorder, de cantidad, o un contador que se reinicia por periodo?

**Por qué importa:** reproducimos su **fórmula** de ReT exactamente. Lo que no reproducimos es la
**columna**: ninguna convención que hemos ensayado la reconstruye en más del **1,09 % de 47.780
filas**. Eso significa que la reproducción algebraica está probada y la **reproducción conductual
del DES no**.

**Qué desbloquea:** hoy no podemos afirmar que nuestro DES reproduce el suyo *pedido a pedido*.
Podemos afirmar que reproduce su fórmula y sus seis hipótesis de moderación en dirección. Con esta
respuesta, o cerramos la brecha o la declaramos por escrito como límite.

**Si no responde:** fijamos una convención propia, la declaramos como decisión nuestra, y retiramos
del manuscrito todo lenguaje de «reproducción conductual».

---

## 2. Doctrina de servicio: ¿piso vinculante por peor reclamante, o resiliencia media?

**La pregunta:** en la operación real, ¿existe un requisito **vinculante** sobre el nivel de
servicio del **peor** teatro / CSSU / producto — un piso que no se puede incumplir aunque el
promedio mejore? ¿O la medida de aceptación operativa es la **resiliencia media** de la red?

**Por qué importa:** es la bifurcación que decide cómo se lee nuestro resultado más fuerte. El
Programa O tiene **confirmada la ventaja media** de la política de creencia (LCB95 simultáneo
+0,043 a +0,066 en tres celdas, 27/27 placebos batidos, recursos físicamente iguales, 1.451
replays sin fallos). Lo que **no** quedó establecido es la no-inferioridad **conjunta en la cola**
(CVaR10): los dos estimados son positivos (+0,035 y +0,020) y lo que falla es la certificación
simultánea sobre 69 estimandos.

**Qué desbloquea:** si la doctrina es media, el endpoint primario de las campañas futuras es la
media y la cola se reporta. Si hay piso por peor reclamante, el endpoint tiene que ser conjunto
desde el preregistro.

**Nota que le debemos:** esta respuesta **no promueve** el resultado del Programa O. El
guardarraíl se congeló antes de ver los datos y no se retira retroactivamente.

---

## 3. Op11 — capacidad y tiempo físico de manejo

**La pregunta:** ¿cuál es la capacidad de manejo de Op11 y cuánto tiempo físico consume una
unidad? ¿Puede Op11 atender a más de un CSSU simultáneamente, o los serializa?

**Por qué importa:** nuestro gate de liveness pasa entero (`GATE_A_PASS`, 6/6 falsadores, latencia
de activación 24 h, masa conservada 242.500/424.936) y se detiene en `gate_b` con
`HOLD_OP11_PHYSICS_UNSPECIFIED`, porque tenemos `op11_handling_hours = 0,0` — es decir, hoy Op11 no
cuesta nada y por eso no puede haber competencia entre CSSU.

**Qué desbloquea:** la competencia real entre CSSU, que es el único mecanismo por el que una
decisión de asignación puede tener valor en esta cadena.

---

## 4. Deadlines de misión con autoridad de abandono

**La pregunta:** ¿existen **plazos duros** tras los cuales un pedido deja de tener valor
operativo — no «llega tarde», sino **deja de servir**? Y si existen, ¿tiene el planificador
**autoridad doctrinal** para abandonar o reordenar pedidos en función de ellos?

**Por qué importa:** es el reabridor más fuerte que nos queda. Nuestro certificado de agotamiento
identifica que sin plazos permanentes la familia colapsa a un caso ya explorado. Con plazos **más
cortos** que los tiempos de recuperación observados (24–120 h en R21/R23/R22) **y** autoridad de
triaje, aparece una decisión que hoy no existe.

**Qué desbloquea:** la única vía thesis-native a un caso positivo. Sin ella, esa familia queda
cerrada por hecho de dominio y no por medición.

---

## 5. Rutas: ¿dos reales, flota finita, aviso predespacho?

**La pregunta:** ¿existen **dos rutas físicamente distintas** hacia el mismo destino? ¿La flota es
**finita** —un vehículo que debe volver antes de salir de nuevo—? ¿Hay alguna **señal previa al
despacho** sobre el estado de la ruta, aunque sea imperfecta?

**Por qué importa:** es la licencia de dominio de la única decisión de nuestro repositorio donde
**una constante no puede ser competitiva por construcción** — si la ruta óptima depende de una
contingencia observable, ninguna política fija puede acertar siempre.

**Lo que ya medimos, y su límite:** el screen estilizado da `H_PI = 0,110`, `H_obs = 0,0749`,
`η = 0,678`, con dos controles limpios: la celda nula da **cero exacto**, y cuando las dos rutas
caen juntas el headroom vuelve a **cero exacto**. El gradiente de calidad de señal es monótono y
**empieza en negativo** (−0,0087 con señal mala). Pero hay un freno que su respuesta **no levanta**:
la política adaptativa usa **+0,90 viajes** más que el comparador, así que hoy no compiten con los
mismos recursos.

**Qué desbloquea:** si dice que sí, ahí va el siguiente experimento —con los recursos igualados
primero—. Si dice que no, la lane se cierra por hecho de dominio y lo escribimos así.

---

## Lo que NO le preguntamos, y por qué

No le pedimos que valide nuestros resultados, ni que elija entre métodos, ni que opine sobre si una
red hace falta. Esas son preguntas nuestras y las contestamos midiendo.

**Regla vinculante que nos aplicamos:** una respuesta que reabre una familia **no autoriza
entrenar**. Autoriza **preregistrar** el contrato correspondiente, con el oráculo primero.
