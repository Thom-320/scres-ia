# Superado — calculado con la carrera en orden alfabético

`load_cache` construía la lista de contextos con `sorted(rglob)`, que da
`R1r, R1r+R2r, R1r+R2r|esc, R1r|esc, R2r, R2r|esc` —alfabético por nombre de directorio— en vez
del orden que declara el contrato, `R1r, R2r, R1r+R2r, R1r|esc, R2r|esc, R1r+R2r|esc`.

Para un brazo que arrastra estado **el orden de contextos ES la carrera**, así que este artefacto
mide una carrera que ningún contrato declaró.

**El veredicto no cambió al recomputarlo**, pero los números sí. Los citables son:

* R3 → `results/search_ladder_ordered/result.json`
* R4 → `results/search_ladder_v2_ordered/result.json`
* R7 → `results/grid_transfer_ordered_v1/result.json`

Este artefacto se conserva como procedencia y para que la corrección sea auditable. **No se cita.**
