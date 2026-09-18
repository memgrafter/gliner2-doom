#!/bin/bash
# Direct-policy training pipeline (ticket pro-zekx). Resumable: each stage is skipped
# when its artifact exists. Usage: scripts/pipeline.sh [--force-from S1|S2|...]
# Log: scripts/pipeline.log (appends). ~2.3 h on the M4 CPU from scratch (ONNX encoder; Metal is the default now).
set -u
cd "$(dirname "$0")/.."
export ORT_DYLIB_PATH="$PWD/vendor/onnxruntime/lib/libonnxruntime.dylib"
B=./target/release/gliner2-doom
W=$HOME/virtualenvs/gliner2-doom-vizdoom/lib/python3.12/site-packages/vizdoom
TRAIN_MAPS="freedoom2:map01 freedoom2:map02 freedoom2:map03 freedoom2:map04 freedoom2:map05 freedoom2:map06 freedoom2:map07 freedoom2:map08 freedoom1:e1m1 freedoom1:e1m2 freedoom1:e1m3 freedoom1:e1m4"
HOLDOUT="freedoom2:map09 freedoom2:map10 freedoom2:map12 freedoom1:e2m1 freedoom1:e2m2"
FORCE="${2:-}"; [ "${1:-}" = "--force-from" ] || FORCE=""
q() { grep -vE "^\[bridge\]|^engine|^  encoded|^  ep "; }
stamp() { date "+%H:%M $*"; }
need() { [ -n "$FORCE" ] && [[ "$1" > "$FORCE" || "$1" == "$FORCE" ]] && return 0; [ ! -e "$2" ]; }
exec >> scripts/pipeline.log 2>&1
stamp "pipeline start (force-from='${FORCE}')"
if need S1 data2/train.jsonl; then
  stamp "S1 collect oracle: defend + 12 training maps"; rm -rf data2
  $B collect --scenario defend_the_center --episodes 40 --epsilon 0.2 --out-dir data2 2>&1 | q
  for m in $TRAIN_MAPS; do w=${m%%:*}; mp=${m##*:}; $B collect --episodes 8 --epsilon 0.2 --out-dir data2 --wad "$W/$w.wad" --map "$mp" 2>&1 | q | sed "s/^/[$m] /"; done
fi
if need S2 heads/v4.safetensors; then
  if [ -e heads/v4.best.safetensors ]; then
    stamp "S2 train v4: resuming from the best-epoch checkpoint of an interrupted run (10 more epochs)"
    $B train --data-dir data2 --out heads/v4.safetensors --epochs 10 --init-from heads/v4.best.safetensors --notes "v4: oracle eps0.2 on defend + 12 training maps; resumed from interrupted checkpoint" 2>&1 | q
  else
    stamp "S2 train v4 (25 epochs, from scratch)"
    $B train --data-dir data2 --out heads/v4.safetensors --epochs 25 --notes "v4: oracle eps0.2 on defend + 12 training maps; corpses filtered, pickups sought" 2>&1 | q
  fi
  $B eval --head heads/v4.safetensors --data data2/test.jsonl 2>&1 | q | head -5
fi
if need S3 data2/.dagger_done; then
  stamp "S3 DAgger: v4 drives training maps (4 eps each) + defend (10), oracle labels"
  $B collect --scenario defend_the_center --episodes 10 --epsilon 0.05 --out-dir data2 --head heads/v4.safetensors 2>&1 | q
  for m in $TRAIN_MAPS; do w=${m%%:*}; mp=${m##*:}; $B collect --episodes 4 --epsilon 0.05 --out-dir data2 --wad "$W/$w.wad" --map "$mp" --head heads/v4.safetensors 2>&1 | q | sed "s/^/[$m] /"; done
  touch data2/.dagger_done
fi
if need S4 heads/v5.safetensors; then
  stamp "S4 train v5 (warm start v4, 15 epochs)"
  $B train --data-dir data2 --out heads/v5.safetensors --epochs 15 --init-from heads/v4.safetensors --notes "v5: v4 data + DAgger round (v4 drives), warm start" 2>&1 | q
  $B eval --head heads/v5.safetensors --data data2/test.jsonl 2>&1 | q | head -5
fi
if need S5 data2_rwr/train.jsonl; then
  stamp "S5 RWR rollouts: v5 drives with eps 0.1, engine reward only"; rm -rf data2_rwr
  $B collect --scenario defend_the_center --episodes 12 --epsilon 0.1 --out-dir data2_rwr --head heads/v5.safetensors 2>&1 | q
  for m in $TRAIN_MAPS; do w=${m%%:*}; mp=${m##*:}; $B collect --episodes 4 --epsilon 0.1 --out-dir data2_rwr --wad "$W/$w.wad" --map "$mp" --head heads/v5.safetensors 2>&1 | q | sed "s/^/[$m] /"; done
  for c in states.bin states.metal.bin; do [ -e data2/$c ] && { ln data2/$c data2_rwr/$c 2>/dev/null || cp data2/$c data2_rwr/$c; }; done
fi
if need S6 heads/v6.safetensors; then
  stamp "S6 train v6 (RWR, warm start v5, 10 epochs, no oracle labels)"
  $B train --data-dir data2_rwr --out heads/v6.safetensors --epochs 10 --rwr --rwr-beta 5 --init-from heads/v5.safetensors --lr 3e-4 --notes "v6: reward-weighted regression on v5's own rollouts (kills 10, items 3, health 0.05, ammo -0.05), warm start v5" 2>&1 | q
fi
stamp "S7 held-out maps, sync play 2 eps each: oracle | v3 | v5 | v6"
for m in $HOLDOUT; do w=${m%%:*}; mp=${m##*:}; echo "=== holdout $m"; echo "-- oracle"; $B collect --episodes 2 --epsilon 0.0 --out-dir /tmp/gliner2-doom-holdout-oracle --wad "$W/$w.wad" --map "$mp" 2>&1 | grep -E "^  ep"; for h in v3 v5 v6; do [ -e heads/$h.safetensors ] || continue; echo "-- $h"; $B play --policy direct --head heads/$h.safetensors --episodes 2 --wad "$W/$w.wad" --map "$mp" --timeout-tics 2100 2>&1 | grep -E "^episode"; done; done
stamp "S8 defend_the_center regression, real-time 5 eps: v5 | v6"
for h in v5 v6; do [ -e heads/$h.safetensors ] || continue; echo "-- $h"; $B live --policy direct --head heads/$h.safetensors --episodes 5 --no-tui 2>&1 | grep -E "^episodes|tics in"; done
rm -f _vizdoom.ini
stamp "ALL DONE"
