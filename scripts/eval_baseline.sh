#!/usr/bin/env bash
# Evaluate SimpleFlight's published checkpoints on all benchmark trajectories.
# Run this BEFORE training to validate toolchain.
# Success = numbers within ~2× of paper Table III.
#
# Usage: bash scripts/eval_baseline.sh
# Expected output: MED (x-y) per trajectory, printed to console + saved to
#   experiments/baseline_eval/results.txt

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SF_DIR="${REPO_ROOT}/simpleflight/SimpleFlight"
RESULTS_DIR="${REPO_ROOT}/experiments/baseline_eval"
mkdir -p "${RESULTS_DIR}"

if [ ! -d "${SF_DIR}" ]; then
    echo "ERROR: SimpleFlight not found at ${SF_DIR}"
    echo "Run scripts/setup_env.sh first."
    exit 1
fi

echo "=== SimpleFlight Baseline Checkpoint Evaluation ==="
echo "Results will be saved to: ${RESULTS_DIR}/results.txt"
echo ""

# Target numbers from paper Table III (MED in meters, x-y plane)
# We accept up to 2× these values as toolchain-valid.
echo "Paper Table III reference (target: within 2×):"
echo "  figure_eight_normal: 0.028 m  (our target: < 0.056)"
echo "  figure_eight_slow:   ~0.020 m (our target: < 0.040)"
echo "  figure_eight_fast:   ~0.050 m (our target: < 0.100)"
echo "  pentagram_slow:      ~0.030 m (our target: < 0.060)"
echo "  pentagram_fast:      ~0.060 m (our target: < 0.120)"
echo "  random_polynomial:   ~0.030 m (our target: < 0.060)"
echo "  random_zigzag:       ~0.050 m (our target: < 0.100)"
echo ""

TRAJECTORIES=(
    "figure_eight_slow"
    "figure_eight_normal"
    "figure_eight_fast"
    "pentagram_slow"
    "pentagram_fast"
    "random_polynomial"
    "random_zigzag"
)

{
    echo "SimpleFlight Baseline Eval — $(date)"
    echo "Checkpoint: ${SF_DIR}/models/"
    echo ""
} > "${RESULTS_DIR}/results.txt"

for TRAJ in "${TRAJECTORIES[@]}"; do
    echo "Evaluating: ${TRAJ}..."
    conda run -n sim python "${SF_DIR}/scripts/eval.py" \
        task=Track \
        task.traj_type="${TRAJ}" \
        checkpoint="${SF_DIR}/models/deploy.pt" \
        headless=true \
        num_envs=32 \
        2>&1 | tee -a "${RESULTS_DIR}/${TRAJ}.log" | grep -E "MED|Error|error" || true
    echo ""
done

echo ""
echo "Evaluation complete. Full logs in: ${RESULTS_DIR}/"
echo ""
echo "PASS/FAIL check:"
echo "  If any trajectory MED > 2× paper value → toolchain is broken."
echo "  Fix before starting training."
