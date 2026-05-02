#!/usr/bin/env bash
# M1 environment setup — WSL2 Ubuntu 22.04 / 24.04
# Stack: Python 3.10 + JAX (CUDA 12) + MuJoCo MJX
#
# Much simpler than Isaac Sim — no Omniverse Launcher needed.
# Prerequisites:
#   1. WSL2 with Ubuntu 22.04 or 24.04
#   2. NVIDIA driver >= 525 on Windows (provides CUDA passthrough to WSL2)
#      Do NOT apt install nvidia-driver in WSL2 — the Windows driver handles it.
#
# Usage: bash scripts/setup_env.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== Iron Man Drone — M1 Environment Setup (MJX stack) ==="
echo "Repo root: ${REPO_ROOT}"
echo ""

# ── Step 1: Verify NVIDIA GPU is visible in WSL2 ──────────────────────────────
echo "[1/6] Checking NVIDIA GPU visibility..."
if ! nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found."
    echo ""
    echo "WSL2 CUDA setup:"
    echo "  1. Update Windows NVIDIA driver to >= 525 (https://www.nvidia.com/drivers)"
    echo "  2. nvidia-smi should then work inside WSL2 automatically."
    echo "  3. Do NOT run 'apt install nvidia-driver' inside WSL2."
    echo "  See: https://docs.nvidia.com/cuda/wsl-user-guide/"
    exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo ""

# ── Step 2: System dependencies ───────────────────────────────────────────────
echo "[2/6] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y wget curl git build-essential \
    libgl1-mesa-glx libglib2.0-0 \
    python3-pip python3-venv
echo ""

# ── Step 3: Conda ─────────────────────────────────────────────────────────────
echo "[3/6] Checking Conda..."
if ! command -v conda &>/dev/null; then
    echo "Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "${HOME}/miniconda3"
    rm /tmp/miniconda.sh
    eval "$("${HOME}/miniconda3/bin/conda" shell.bash hook)"
    conda init bash
fi
conda --version
echo ""

# ── Step 4: Create conda environment ─────────────────────────────────────────
echo "[4/6] Creating conda environment 'drone' (Python 3.10)..."
if ! conda env list | grep -q "^drone "; then
    conda create -n drone python=3.10 -y
fi

# ── Step 5: Install Python packages ──────────────────────────────────────────
echo "[5/6] Installing Python packages..."
conda run -n drone bash -c "
    set -e
    cd ${REPO_ROOT}

    # JAX with CUDA 12 (must come before other JAX-dependent packages)
    pip install --upgrade 'jax[cuda12]' --quiet

    # Verify JAX can see the GPU
    python -c \"
import jax
devices = jax.devices()
print(f'JAX devices: {devices}')
assert any('cuda' in str(d).lower() or 'gpu' in str(d).lower() for d in devices), \
    'No GPU detected by JAX. Check CUDA driver setup.'
print('JAX GPU: OK')
\"

    # MuJoCo with MJX (bundled since 3.0)
    pip install 'mujoco>=3.1.6' --quiet

    # Flax + Optax + Distrax
    pip install 'flax>=0.8.0' 'optax>=0.2.2' 'distrax>=0.1.5' --quiet

    # Orbax for checkpointing
    pip install 'orbax-checkpoint>=0.4.0' --quiet

    # Config + logging
    pip install 'hydra-core>=1.3.2' 'wandb>=0.16.0' 'tensorboard>=2.15.0' --quiet
    pip install 'matplotlib>=3.8.0' 'tqdm>=4.66.0' 'pyyaml>=6.0' --quiet

    # Install the iron_man_drone package
    pip install -e . --quiet

    echo 'All packages installed.'
"
echo ""

# ── Step 6: Sanity checks ─────────────────────────────────────────────────────
echo "[6/6] Running sanity checks..."
conda run -n drone python -c "
import sys
print(f'Python: {sys.version.split()[0]}')

import jax
import jax.numpy as jnp
print(f'JAX: {jax.__version__}')
print(f'Devices: {jax.devices()}')

import mujoco
from mujoco import mjx
print(f'MuJoCo: {mujoco.__version__}')

import flax, optax, distrax
print(f'Flax: {flax.__version__}  Optax: {optax.__version__}')

# Test MJX vmap
import numpy as np
xml = '''
<mujoco>
  <option timestep=\"0.01\"/>
  <worldbody>
    <body>
      <freejoint/>
      <inertial mass=\"1\" pos=\"0 0 0\" diaginertia=\"1 1 1\"/>
      <geom type=\"sphere\" size=\"0.1\"/>
    </body>
  </worldbody>
</mujoco>
'''
model = mujoco.MjModel.from_xml_string(xml)
mx = mjx.put_model(model)
dx = mjx.make_data(mx)

# Vmap over a batch of 4 envs
batch_step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))
batch_dx = jax.tree_util.tree_map(lambda x: jnp.stack([x]*4), dx)
out = batch_step(mx, batch_dx)
print(f'MJX vmap test: {out.qpos.shape} [expected (4, 7)]')
assert out.qpos.shape == (4, 7), f'Unexpected shape: {out.qpos.shape}'
print('MJX GPU-parallel: OK')
print()
print('All checks passed. Run: conda activate drone && python scripts/train_m1.py')
"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Activate environment: conda activate drone"
echo "Read gate doc:        cat notes/M1_hypothesis.md"
echo "Start training:       python scripts/train_m1.py"
