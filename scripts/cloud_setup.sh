#!/usr/bin/env bash
# Cloud GPU bootstrap for iron-man-drone (M3 v2 reward-lever sweep).
# Target: a fresh Ubuntu 22.04/24.04 box with an NVIDIA GPU + CUDA 12 driver
# (RunPod / Vast.ai / Lambda). Run from the repo root after cloning:
#
#   git clone https://github.com/Forkei/iron-man-drone.git
#   cd iron-man-drone && git checkout m2-dev
#   bash scripts/cloud_setup.sh
#
# Versions known-good from the local 4070 env (RTX 4070, driver 596, CUDA 12):
#   warp-lang 1.13.0, mujoco-warp 3.8.0.3, jax (cuda12). Adjust if pip resolves
#   incompatible wheels — verify step at the end will catch it.
set -euo pipefail

echo "[1/5] GPU visible?"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
  || { echo "ERROR: no GPU / driver. Pick a CUDA-12 GPU image."; exit 1; }

echo "[2/5] system python + venv"
sudo apt-get update -qq && sudo apt-get install -y python3-venv python3-pip git libgl1 2>/dev/null || true
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel setuptools

echo "[3/5] JAX (CUDA 12)"
pip install -U "jax[cuda12]"

echo "[4/5] sim + RL deps"
pip install "warp-lang==1.13.0"
pip install "mujoco>=3.1.6"
# mujoco_warp: try pinned PyPI, fall back to DeepMind git if the pin is unavailable.
pip install "mujoco-warp==3.8.0.3" \
  || pip install "mujoco-warp" \
  || pip install "git+https://github.com/google-deepmind/mujoco_warp.git"
pip install flax optax distrax orbax-checkpoint numpy matplotlib opencv-python-headless
pip install -e .

echo "[5/5] verify"
python -c "import jax; print('jax', jax.__version__, '->', jax.devices())"
python -c "import warp,mujoco_warp; print('warp', warp.__version__, 'mujoco_warp OK')"
python -c "import mujoco,flax,optax,distrax,orbax.checkpoint; print('rl libs OK')"
python -c "import jax,jax.numpy as jnp; print('GPU matmul', float(jnp.ones((1024,1024))@jnp.ones((1024,1024)) is not None))"
echo
echo "SETUP DONE. Next: bash scripts/cloud_sweep.sh   (or set STEPS=... first)"
