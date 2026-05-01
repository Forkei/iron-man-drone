#!/usr/bin/env bash
# M1 training run — SimpleFlight reproduction.
# Calls SimpleFlight's train.py with our frozen hyperparameters.
#
# GATE: Read notes/M1_hypothesis.md before running this.
#
# Usage: bash scripts/train_m1.sh [--resume CHECKPOINT_PATH]
#
# Outputs to: experiments/m1_baseline/RUN_TIMESTAMP/
#
# NOTE: Hydra override keys (algo.actor_lr etc.) must match what's in
# simpleflight/SimpleFlight/cfg/. Verify by running:
#   python train.py --cfg job
# and cross-checking with our config.yaml before the first real run.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SF_DIR="${REPO_ROOT}/simpleflight/SimpleFlight"
CONFIG="${REPO_ROOT}/experiments/m1_baseline/config.yaml"

# Gate: hypothesis doc must exist and be non-empty
HYPOTHESIS="${REPO_ROOT}/notes/M1_hypothesis.md"
if [ ! -s "${HYPOTHESIS}" ]; then
    echo "ERROR: notes/M1_hypothesis.md is missing or empty."
    echo "Write the hypothesis doc before running training."
    echo "This is a gating requirement — no training without it."
    exit 1
fi
echo "[GATE PASSED] Hypothesis doc found."

# Timestamp-stamped run directory
RUN_TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${REPO_ROOT}/experiments/m1_baseline/${RUN_TS}"
mkdir -p "${RUN_DIR}/checkpoints"
mkdir -p "${RUN_DIR}/logs"

# Copy frozen config into run directory
cp "${CONFIG}" "${RUN_DIR}/config_frozen.yaml"

echo "=== M1 Training Run ==="
echo "Run dir: ${RUN_DIR}"
echo ""

# Read hyperparameters from config
# (These match SimpleFlight Table VI + our M1 spec)
NUM_ENVS=128
TOTAL_FRAMES=$((15000 * 128 * 32))  # epochs × envs × horizon
ACTOR_LR=3e-4
CRITIC_LR=1e-4
ENTROPY_COEFF=1e-3

echo "Key hyperparameters (frozen):"
echo "  num_envs:       ${NUM_ENVS}"
echo "  actor_lr:       ${ACTOR_LR}"
echo "  critic_lr:      ${CRITIC_LR}"
echo "  entropy_coeff:  ${ENTROPY_COEFF}"
echo "  total_frames:   ${TOTAL_FRAMES}"
echo ""
echo "SANITY CHECK: Before epoch 1, verify entropy < 10% of reward."
echo "If not, STOP and reduce entropy_coeff before continuing."
echo ""

# Pre-flight VRAM check
conda run -n sim python -c "
import torch
if torch.cuda.is_available():
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'VRAM available: {vram_gb:.1f} GB')
    if vram_gb < 6:
        print('WARNING: < 6 GB VRAM detected. Reduce num_envs to 64.')
else:
    print('ERROR: CUDA not available. Cannot train.')
    exit(1)
"

# Train using SimpleFlight's train.py with Hydra overrides.
# TODO: verify these override keys against simpleflight/SimpleFlight/cfg/
# by running: python train.py --cfg job
conda run -n sim python "${SF_DIR}/scripts/train.py" \
    task=Track \
    algo=ppo \
    headless=true \
    num_envs=${NUM_ENVS} \
    algo.actor_lr=${ACTOR_LR} \
    algo.critic_lr=${CRITIC_LR} \
    algo.entropy_coeff=${ENTROPY_COEFF} \
    algo.gamma=0.99 \
    algo.gae_lambda=0.95 \
    algo.clip_eps=0.2 \
    algo.critic_updates=16 \
    algo.horizon=32 \
    total_frames=${TOTAL_FRAMES} \
    checkpoint_dir="${RUN_DIR}/checkpoints" \
    logger.log_dir="${RUN_DIR}/logs" \
    wandb.mode=offline \
    2>&1 | tee "${RUN_DIR}/train.log"

echo ""
echo "=== Training complete ==="
echo "Checkpoint: ${RUN_DIR}/checkpoints/"
echo ""
echo "Next: run eval"
echo "  bash scripts/eval_m1.sh --checkpoint ${RUN_DIR}/checkpoints/final.pt"
