#!/bin/bash

# D0 Baseline Reproduction Evaluation Script
#
# Usage:
#   bash eval_d0.sh <experiment_id> [device] [num_rollouts] [num_envs_per_motion]
#
# Example:
#   bash eval_d0.sh 1103_twist2 cuda:0 10 100
#
# Example 2 (with dash in exptid):
#   bash eval_d0.sh "0126-6-982679" cuda:0 10 100

set -e  # Exit on error

EXPTID=${1:-"test_experiment"}
DEVICE=${2:-"cuda:0"}
NUM_ROLLOUTS=${3:-10}
NUM_ENVS_PER_MOTION=${4:-100}

TASK_NAME="g1_stu_future_cjm"
PROJ_NAME="g1_stu_future_cjm"

echo "========================================================================"
echo "D0 BASELINE REPRODUCTION - OFFLINE EVALUATION"
echo "========================================================================"
echo "Task:                  ${TASK_NAME}"
echo "Project:               ${PROJ_NAME}"
echo "Experiment:            ${EXPTID}"
echo "Device:                ${DEVICE}"
echo "Num Rollouts:          ${NUM_ROLLOUTS}"
echo "Num Envs per Motion:   ${NUM_ENVS_PER_MOTION}"
echo "Total Environments:   $((NUM_ENVS_PER_MOTION * 2))"
echo "========================================================================"
echo ""

# Change to legged_gym root directory
cd legged_gym

# Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

cd legged_gym/scripts

# Run offline evaluation
python offline_eval.py \
    --task "${TASK_NAME}" \
    --proj_name "${PROJ_NAME}" \
    --exptid "${EXPTID}" \
    --device "${DEVICE}" \
    --num_rollouts ${NUM_ROLLOUTS} \
    --num_envs_per_motion ${NUM_ENVS_PER_MOTION} \
    --seed 42

echo ""
echo "========================================================================"
echo "D0 Evaluation Complete!"
echo "========================================================================"
echo "Results can be found in:"
echo "  logs/${PROJ_NAME}/${EXPTID}/D0_evaluation/"
echo "    - evaluation_results.json  (Raw results)"
echo "    - evaluation_plots.png    (Visualization)"
echo "========================================================================"
