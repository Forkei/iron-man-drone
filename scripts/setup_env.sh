#!/usr/bin/env bash
# M1 environment setup — WSL2 Ubuntu 22.04 / 24.04
# Uses SimpleFlight's exact stack: Isaac Sim 2022.2.0 + Python 3.7
#
# Prerequisites:
#   1. WSL2 with Ubuntu 22.04 or 24.04 installed
#   2. NVIDIA driver >= 525 installed on Windows (WSL2 CUDA passthrough)
#   3. NVIDIA Omniverse Launcher downloaded and available in WSL
#
# Run this script once. It is idempotent.
# Usage: bash scripts/setup_env.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIMPLEFLIGHT_DIR="${REPO_ROOT}/simpleflight"
ISAAC_SIM_VERSION="2022.2.0"
ISAAC_SIM_PATH="${HOME}/.local/share/ov/pkg/isaac_sim-${ISAAC_SIM_VERSION}"

echo "=== Iron Man Drone — M1 Environment Setup ==="
echo "Repo root: ${REPO_ROOT}"
echo ""

# ── Step 1: Verify WSL2 CUDA is visible ──────────────────────────────────────
echo "[1/8] Checking NVIDIA GPU visibility in WSL2..."
if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found."
    echo "  On Windows: make sure your NVIDIA driver is >= 525."
    echo "  In WSL2: nvidia-smi should work via driver passthrough."
    echo "  See: https://docs.nvidia.com/cuda/wsl-user-guide/"
    exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo ""

# ── Step 2: Install system dependencies ───────────────────────────────────────
echo "[2/8] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y \
    wget curl git git-lfs build-essential \
    libgl1-mesa-glx libglib2.0-0 \
    libx11-6 libxext6 libxrender1 \
    python3-pip

# ── Step 3: Install Miniconda if absent ───────────────────────────────────────
echo "[3/8] Checking Conda..."
if ! command -v conda &>/dev/null; then
    echo "Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "${HOME}/miniconda3"
    rm /tmp/miniconda.sh
    echo 'export PATH="${HOME}/miniconda3/bin:${PATH}"' >> "${HOME}/.bashrc"
    export PATH="${HOME}/miniconda3/bin:${PATH}"
fi
conda --version
echo ""

# ── Step 4: Isaac Sim 2022.2.0 ────────────────────────────────────────────────
echo "[4/8] Checking Isaac Sim 2022.2.0..."
if [ ! -d "${ISAAC_SIM_PATH}" ]; then
    echo ""
    echo "ACTION REQUIRED: Isaac Sim 2022.2.0 must be installed manually."
    echo ""
    echo "  1. Install NVIDIA Omniverse Launcher from:"
    echo "     https://www.nvidia.com/en-us/omniverse/download/"
    echo "     (Download the Linux version, run in WSL2)"
    echo ""
    echo "  2. From the Launcher: Exchange → Apps → Isaac Sim"
    echo "     Select version: 2022.2.0"
    echo "     Install to default path: ~/.local/share/ov/pkg/"
    echo ""
    echo "  3. Re-run this script after installation."
    echo ""
    echo "  NOTE: Isaac Sim 2022.2.0 is ~10GB download."
    echo "  Alternative: Install via Nucleus if you have an NGC account."
    echo ""
    echo "After Isaac Sim is installed, expected path:"
    echo "  ${ISAAC_SIM_PATH}"
    echo ""
    echo "Script paused. Re-run after Isaac Sim install."
    exit 0
fi

echo "Isaac Sim found at: ${ISAAC_SIM_PATH}"
echo 'export ISAACSIM_PATH="${HOME}/.local/share/ov/pkg/isaac_sim-2022.2.0"' >> "${HOME}/.bashrc" 2>/dev/null || true
export ISAACSIM_PATH="${ISAAC_SIM_PATH}"
echo ""

# ── Step 5: Clone SimpleFlight ────────────────────────────────────────────────
echo "[5/8] Cloning SimpleFlight (thu-uav/SimpleFlight)..."
if [ ! -d "${SIMPLEFLIGHT_DIR}/SimpleFlight/.git" ]; then
    git clone https://github.com/thu-uav/SimpleFlight.git "${SIMPLEFLIGHT_DIR}/SimpleFlight"
    git -C "${SIMPLEFLIGHT_DIR}/SimpleFlight" submodule update --init --recursive
else
    echo "Already cloned, skipping."
fi
echo ""

# ── Step 6: Create conda environment (Python 3.7 — required by Isaac Sim 2022) ──
echo "[6/8] Creating conda environment 'sim' (Python 3.7)..."
if ! conda env list | grep -q "^sim "; then
    conda create -n sim python=3.7 -y
fi

# Copy Isaac Sim conda hooks (sets up PATH, LD_LIBRARY_PATH on activate)
CONDA_PREFIX=$(conda run -n sim python -c "import sys; print(sys.prefix)")
cp -r "${SIMPLEFLIGHT_DIR}/SimpleFlight/conda_setup/etc" "${CONDA_PREFIX}/"
echo "Conda activation hooks installed."
echo ""

# ── Step 7: Install SimpleFlight + pinned deps ────────────────────────────────
echo "[7/8] Installing Python packages in 'sim' environment..."
conda run -n sim bash -c "
    cd ${SIMPLEFLIGHT_DIR}/SimpleFlight

    # Install SimpleFlight package
    pip install -e . --quiet

    # Pinned submodule versions (exact paper reproduction)
    cd third_party/tensordict
    git checkout 5e6205c
    pip install -e . --no-build-isolation --quiet

    cd ../torchrl
    git checkout e39e701
    pip install -e . --no-build-isolation --quiet

    cd ../..

    # Additional monitoring tools
    pip install tensorboard wandb --quiet
"
echo ""

# ── Step 8: Sanity check ──────────────────────────────────────────────────────
echo "[8/8] Running sanity checks..."
conda run -n sim python -c "
import sys
print(f'Python: {sys.version}')
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. conda activate sim"
echo "  2. Run baseline eval:  bash scripts/run_eval_baseline.sh"
echo "  3. Run training:       bash scripts/run_train_m1.sh"
echo ""
echo "See notes/M1_hypothesis.md before starting any training run."
