#!/usr/bin/env bash
# M3 v2 reward-lever sweep — runs the hypothesis + reserve levers IN PARALLEL on one
# GPU (each ~6.5 GB, so ~3 fit on a 24 GB card → max utilization). Validation length
# by default (50M steps ≈ ~1h each on a dedicated 4090; ~2-3h wall when sharing).
#
# Run after cloud_setup.sh, from repo root:
#   bash scripts/cloud_sweep.sh            # 50M validation sweep
#   STEPS=500000000 RUNS=wtrack6 bash scripts/cloud_sweep.sh   # full single run
#
# Compare vs the v1 baseline already measured (W_TRACK=2 W_CRASH=10 = m3_run1):
#   figure-eight 0.287m, eval fair CF 92%, tracking MED ~0.17m.
set -euo pipefail
source .venv/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false        # share one GPU across the runs
export M3_CKPT_DIR="$PWD/ckpts"                   # cloud-local checkpoints (not /home/forke)
STEPS=${STEPS:-50000000}
RUNS=${RUNS:-all}
mkdir -p logs ckpts

launch () {  # name  ENVVARS...
  local name="$1"; shift
  echo "launch $name : $*"
  env "$@" nohup python -u scripts/train_m3.py --run_name "$name" --total_steps "$STEPS" \
      > "logs/$name.log" 2>&1 &
  echo "  pid $! -> logs/$name.log"
}

# A: primary v2 hypothesis — up-weight tracking
[[ "$RUNS" == "all" || "$RUNS" == *wtrack6* ]] && launch m3_v2_wtrack6  W_TRACK=6 W_CRASH=10
# B: reserve lever — reduce crash-penalty conservatism only
[[ "$RUNS" == "all" || "$RUNS" == *crash4*   ]] && launch m3_v2_crash4   W_TRACK=2 W_CRASH=4
# C: combined — more tracking + less crash
[[ "$RUNS" == "all" || "$RUNS" == *both*     ]] && launch m3_v2_both     W_TRACK=6 W_CRASH=4

echo
echo "launched. monitor:  tail -f logs/*.log    or    bash scripts/cloud_status.sh"
echo "when a run hits ~epoch 400+, early fig8 read:"
echo "  python -u scripts/diag_fig8.py --ckpt ckpts/m3_v2_wtrack6/\$(ls ckpts/m3_v2_wtrack6 | sort | tail -1)"
