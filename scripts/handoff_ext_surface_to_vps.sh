#!/bin/bash
# Hand the extended-surface replay a second machine the moment the first one frees up.
#
# WHY A DISJOINT REPARTITION AND NOT JUST "ALSO RUN IT ON THE VPS". The shard scheme skips a slice
# whose result file already exists, but it builds that list ONCE at start-up and the two machines
# do not share a filesystem. Two pools on overlapping partitions would therefore recompute the same
# slices and the duplicated work would eat the whole saving. So local is restarted on a disjoint
# share at the same moment the VPS joins; slices already on disk are skipped, so the only loss is
# the up-to-8 slices in flight.
#
# WHY THE VPS GETS 25% AND NOT ITS 15% "FAIR SHARE". Measured, the VPS is ~4x slower per core, which
# makes 15% its share on paper. But the local machine is at capacity -- eight shard processes plus
# the downstream-chain replay plus the desktop leave it delivering well under eight clean cores, and
# free memory is down to ~111MB -- while the VPS sits idle with nothing else on it. Capacity that is
# actually free beats capacity that is nominally faster.
set -u
REPO="$HOME/Projects/research/scres-ia"
SP="/private/tmp/claude-501/-Users-thom-Projects-research-scres-ia/c4c70a0d-d0da-4c53-824e-5d7a181768e6/scratchpad"
VPS=ovh-agent-lab
cd "$REPO" || exit 1

echo "[$(date +%H:%M)] esperando a que el VPS cierre la superficie base…"
until [ "$(ssh -o ConnectTimeout=10 $VPS 'ls scres-ia/results/frozen_path_equivalence_v2/shards 2>/dev/null | wc -l' 2>/dev/null)" -ge 360 ]; do
  sleep 120
done
echo "[$(date +%H:%M)] base cerrada. Recogiendo sus shards antes de tocar nada."
rsync -az $VPS:scres-ia/results/frozen_path_equivalence_v2/shards/ results/frozen_path_equivalence_v2/shards/
echo "  base recogida: $(ls results/frozen_path_equivalence_v2/shards | grep -c '^base__') rebanadas"

echo "[$(date +%H:%M)] enviando el caché extendido (649 MB)…"
rsync -az results/surface_cache/garrido_transfer_confirmation_v2_ext/ \
  $VPS:scres-ia/results/surface_cache/garrido_transfer_confirmation_v2_ext/ || exit 1

# Repartition. 40 partitions: local keeps 0-29, the VPS takes 30-39.
echo "[$(date +%H:%M)] reparticionando: local 0-29 de 40, VPS 30-39 de 40"
pkill -f "verify_frozen_path_equivalence_v2.py --phase surface --surface ext"
sleep 3
LOCAL=("0,8,16,24" "1,9,17,25" "2,10,18,26" "3,11,19,27" "4,12,20,28" "5,13,21,29" "6,14,22" "7,15,23")
for i in "${!LOCAL[@]}"; do
  nohup .venv/bin/python scripts/verify_frozen_path_equivalence_v2.py \
    --phase surface --surface ext --of 40 --shards "${LOCAL[$i]}" \
    > "$SP/ext_local_$i.log" 2>&1 &
done
echo "  local relanzado: ${#LOCAL[@]} procesos"

rsync -az scripts/verify_frozen_path_equivalence_v2.py $VPS:scres-ia/scripts/
ssh $VPS 'cd scres-ia && for s in 30,35 31,36 32,37 33,38 34,39; do
  nohup .venv/bin/python scripts/verify_frozen_path_equivalence_v2.py \
    --phase surface --surface ext --of 40 --shards $s > /tmp/ext_vps_${s%%,*}.log 2>&1 &
done; echo "  VPS lanzado: 5 procesos"'

# Keep pulling the VPS shards back so a dropped link never costs finished work.
#
# STOP FILE, BECAUSE "WHILE THERE ARE PROCESSES" IS THE WRONG CONDITION. On 2026-08-08 this loop was
# still alive hours after its job was done and RESTORED three slices that had just been quarantined
# for cross-platform divergence -- silently, because rsync succeeding is not an error. It was caught
# by their mtimes, not by any log. A sync loop that can undo a decision needs a way to be told the
# decision was made.
STOP="$REPO/results/frozen_path_equivalence_v2/.stop_handoff"
rm -f "$STOP"
while [ ! -e "$STOP" ] && { pgrep -f "phase surface --surface ext" > /dev/null || \
      ssh -o ConnectTimeout=10 $VPS 'pgrep -f "phase surface --surface ext"' > /dev/null 2>&1; }; do
  sleep 300
  rsync -az $VPS:scres-ia/results/frozen_path_equivalence_v2/shards/ \
    results/frozen_path_equivalence_v2/shards/ 2>/dev/null
  echo "  [$(date +%H:%M)] ext: $(ls results/frozen_path_equivalence_v2/shards | grep -c '^ext__')/360"
done
[ -e "$STOP" ] || rsync -az $VPS:scres-ia/results/frozen_path_equivalence_v2/shards/ \
  results/frozen_path_equivalence_v2/shards/ 2>/dev/null
echo "[$(date +%H:%M)] SUPERFICIE COMPLETA · ext $(ls results/frozen_path_equivalence_v2/shards | grep -c '^ext__')/360 · base $(ls results/frozen_path_equivalence_v2/shards | grep -c '^base__')/360"
