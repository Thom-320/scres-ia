#!/bin/bash
cd $HOME/runs/scres-bss/penalizada
export SCRES_REPO=$HOME/runs/scres-bss/repo OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for r in ${REPS:-0 1 2 3}; do
  echo "rep $r start $(date -u +%FT%TZ)"
  ../venv/bin/python -I scripts/bss_inner.py inner 8 $r $r 4 0:7480001-7480048 1:7480001-7480048 2:7480001-7480048
  echo "rep $r end $(date -u +%FT%TZ) rc=$?"
done
echo ALLDONE
