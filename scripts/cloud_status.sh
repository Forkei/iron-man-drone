#!/usr/bin/env bash
# Quick status of the M3 sweep: GPU, per-run latest epoch + alive, log tails.
export M3_CKPT_DIR="${M3_CKPT_DIR:-$PWD/ckpts}"
echo "=== GPU ==="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
echo "=== runs (checkpoint epoch + process state) ==="
for d in "$M3_CKPT_DIR"/*/; do
  [ -d "$d" ] || continue
  name=$(basename "$d")
  last=$(ls "$d" 2>/dev/null | grep -E '^epoch_|^final' | sort | tail -1)
  alive=$(pgrep -f -- "--run_name $name" >/dev/null && echo RUNNING || echo STOPPED)
  echo "  $name: ${last:-<none>}  [$alive]"
done
echo "=== latest train_log rows ==="
for f in experiments/*/train_log.csv; do
  echo "-- $(dirname "$f" | xargs basename)"; tail -1 "$f" 2>/dev/null
done
