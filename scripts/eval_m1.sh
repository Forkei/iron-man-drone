#!/usr/bin/env bash
# Evaluate our trained M1 policy on all benchmark trajectories.
# Compare to SimpleFlight paper Table III.
#
# Usage: bash scripts/eval_m1.sh --checkpoint PATH_TO_CHECKPOINT

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SF_DIR="${REPO_ROOT}/simpleflight/SimpleFlight"

CHECKPOINT=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "${CHECKPOINT}" ]; then
    echo "Usage: bash scripts/eval_m1.sh --checkpoint PATH"
    exit 1
fi

if [ ! -f "${CHECKPOINT}" ]; then
    echo "ERROR: checkpoint not found: ${CHECKPOINT}"
    exit 1
fi

RESULTS_DIR="${REPO_ROOT}/experiments/m1_baseline/eval_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RESULTS_DIR}"

echo "=== M1 Policy Evaluation ==="
echo "Checkpoint: ${CHECKPOINT}"
echo "Results: ${RESULTS_DIR}"
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

# Paper Table III reference values (MED in meters)
declare -A PAPER_MED=(
    ["figure_eight_slow"]="0.020"
    ["figure_eight_normal"]="0.028"
    ["figure_eight_fast"]="0.050"
    ["pentagram_slow"]="0.030"
    ["pentagram_fast"]="0.060"
    ["random_polynomial"]="0.030"
    ["random_zigzag"]="0.050"
)

RESULTS_FILE="${RESULTS_DIR}/med_results.csv"
echo "trajectory,our_med,paper_med,ratio,pass" > "${RESULTS_FILE}"

for TRAJ in "${TRAJECTORIES[@]}"; do
    echo "Evaluating: ${TRAJ}..."

    OUR_MED=$(conda run -n sim python "${SF_DIR}/scripts/eval.py" \
        task=Track \
        task.traj_type="${TRAJ}" \
        checkpoint="${CHECKPOINT}" \
        headless=true \
        num_envs=32 \
        2>&1 | grep "MED" | awk '{print $NF}' | head -1)

    if [ -z "${OUR_MED}" ]; then
        OUR_MED="FAILED"
        RATIO="N/A"
        PASS="FAIL"
    else
        PAPER="${PAPER_MED[$TRAJ]}"
        RATIO=$(python3 -c "print(f'{float('${OUR_MED}') / float('${PAPER}'):.2f}')" 2>/dev/null || echo "N/A")
        PASS=$(python3 -c "print('PASS' if float('${OUR_MED}') < 2 * float('${PAPER}') else 'FAIL')" 2>/dev/null || echo "N/A")
    fi

    echo "  MED: ${OUR_MED} m  (paper: ${PAPER_MED[$TRAJ]} m, ratio: ${RATIO})  [${PASS}]"
    echo "${TRAJ},${OUR_MED},${PAPER_MED[$TRAJ]},${RATIO},${PASS}" >> "${RESULTS_FILE}"
done

echo ""
echo "Full results: ${RESULTS_FILE}"
echo ""
echo "M1 SUCCESS CRITERIA (all must pass):"
echo "  - All trajectories complete without crash"
echo "  - figure_eight_normal MED < 0.056 m (2× paper's 0.028 m)"
echo "  - All trajectories MED < 2× paper values"
echo ""
echo "If PASS: write experiments/m1_baseline/M1_results.md and tag m1-baseline."
echo "If FAIL: see notes/M1_hypothesis.md failure mode section."
