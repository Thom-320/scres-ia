# Runbook — cuatro incidentes de la campaña de verificación, y la regla que deja cada uno

Ninguno de estos es ciencia. Todos costaron tiempo o estuvieron a punto de costar una conclusión,
y los cuatro tienen la misma forma: **un mecanismo silencioso deshaciendo o escondiendo una
decisión**. Se registran aquí, fuera del plan científico, porque es donde sirven.

---

## 1. Un bucle de rsync deshaciendo una cuarentena, sin rastro en ningún log

**Qué pasó.** `scripts/handoff_ext_surface_to_vps.sh` termina con un lazo que recoge las rebanadas
del VPS cada cinco minutos mientras haya procesos vivos. Seguía corriendo horas después de cumplir
su función. Cuando aparté tres rebanadas divergentes de `shards/`, el lazo **las restauró** en el
siguiente ciclo. Lo detecté por los mtimes —eran los del VPS, no los de mi recomputación local— y
no por ningún mensaje de error.

**Por qué es peor que un fallo ruidoso.** Un proceso que falla deja traza. Éste hacía exactamente lo
que se le pidió, y lo que se le pidió había dejado de ser correcto. La cuarentena parecía aplicada:
el fichero estaba en el directorio correcto y también en el que no debía.

**Reglas.**
- Un lazo de sincronización lleva **condición de parada propia**, no «mientras haya procesos».
- Antes de apartar, mover o borrar un artefacto, **matar todo lo que escriba en ese directorio** y
  verificar con `pgrep` que murió.
- Después de una cuarentena, **releer el directorio** y comprobar que lo apartado no volvió.
- Un `mtime` que no cuadra con la acción que acabas de hacer es una alarma, no una curiosidad.

---

## 2. Fusionar resultados antes de leerlos

**Qué pasó.** Al volver los kernels de Kaggle copié sus rebanadas a `shards/` y **después** miré sus
conteos de diferencias. Catorce traían divergencias de último bit y una traía `max|Δ| = 7662`.
Estuvieron en el árbol autoritativo unos minutos.

**Regla.** Se inspecciona y luego se fusiona. Nunca al revés. Un resultado externo entra en
cuarentena por defecto y sale de ella por inspección, no por optimismo.

---

## 3. Un entorno remoto sin versiones fijadas

**Qué pasó.** El kernel de Kaggle corría `pip install -q simpy numpy pandas scipy scikit-learn` sin
pins. Sus resultados no pueden ser la verificación autoritativa de nada, sea cual sea su veredicto,
porque el entorno que los produjo no está especificado.

**Regla.** Cualquier pool que participe en una verificación declara su entorno **antes** de correr:
lockfile o versiones explícitas, y `platform`, `machine`, `python` y `commit` grabados en el bundle
que devuelve. El kernel sí grababa lo segundo; sin lo primero no basta.

---

## 4. Una rama de fallo que descartaba el diagnóstico

**Qué pasó.** `rerun_chain` capturaba ambos flujos y devolvía sólo `stderr`. El runner aguas abajo
anuncia un falsador fallido imprimiendo a **stdout** y saliendo con 1, así que el fallo quedó
registrado como `{"ran": false, "returncode": 1, "stderr": ""}` — un informe cuyo contenido entero
es que algo falló. Recuperar el mensaje costó re-correr una cadena de ochenta minutos.

Cuando por fin se leyó, decía que la cadena había reproducido **todas** las claves científicas y que
lo único fallado era custodia de semillas, por un flag que esperaba un nombre de bloque y recibió una
ruta.

**Regla.** Una rama de fallo devuelve **todo** lo que tenga: stdout, stderr, código de salida y si el
fichero de salida llegó a escribirse. «Algo salió mal» es lo menos útil que puede decir un informe de
error, y es lo que dice por defecto cuando nadie decide lo contrario.

---

## Lo que estos cuatro tienen en común

Tres de los cuatro fueron **descubiertos por inspección, no por una alarma**: mtimes que no cuadraban,
conteos leídos tarde, un stderr vacío. La campaña tenía falsadores para la ciencia y ninguno para la
operación. La regla general que dejan es que **el instrumento también necesita controles**, y que el
más barato de todos es mirar la hora de modificación de un fichero que acabas de mover.
