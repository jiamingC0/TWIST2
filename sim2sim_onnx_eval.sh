#!/bin/bash

# Automated ONNX Model Evaluation with Motion Tracking
# Evaluates multiple .onnx models sequentially
#bash sim2sim_onnx_eval.sh /home/galbot/MyTWIST2/TWIST2/assets/example_motions/251215-083327.pkl /home/galbot/WorkSpace/TWIST2/legged_gym/logs/g1_stu_future/0126-6-982679

SCRIPT_DIR=$(dirname $(realpath $0))
cd ${SCRIPT_DIR}

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <motion_file> <onnx_dir> [options]"
    echo ""
    echo "Arguments:"
    echo "  motion_file    Path to .pkl motion file (required)"
    echo "  onnx_dir       Path to directory containing .onnx models (required)"
    echo ""
    echo "Options:"
    echo "  --num_runs N    Number of runs per model (default: 5)"
    echo "  --redis_ip IP   Redis IP (default: localhost)"
    echo "  --output_dir DIR Output directory for results"
    echo "  --reverse        Evaluate models in reverse order (descending)"
    echo ""
    echo "Example:"
    echo "  $0 assets/example_motions/0807_yanjie_walk_001.pkl assets/ckpts/onnx/"
    echo "  $0 assets/example_motions/0807_yanjie_walk_001.pkl assets/ckpts/onnx/ --num_runs 10"
    echo "  $0 assets/example_motions/0807_yanjie_walk_001.pkl assets/ckpts/onnx/ --reverse"
    echo ""
    exit 1
fi

MOTION_FILE=$(realpath $1)
ONNX_DIR=$(realpath $2)
NUM_RUNS=5
REDIS_IP="localhost"
OUTPUT_DIR=""
REVERSE=""

shift 2
while [ $# -gt 0 ]; do
    case $1 in
        --num_runs)
            NUM_RUNS=$2
            shift
            ;;
        --redis_ip)
            REDIS_IP=$2
            shift
            ;;
        --output_dir)
            OUTPUT_DIR=$2
            shift
            ;;
        --reverse)
            REVERSE="--reverse"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
    shift
done

echo "========================================================================"
echo "AUTOMATED ONNX MODEL EVALUATION"
echo "========================================================================"
echo "Motion file (.pkl):      ${MOTION_FILE}"
echo "ONNX directory:           ${ONNX_DIR}"
echo "Runs per model:           ${NUM_RUNS}"
echo "Redis IP:                ${REDIS_IP}"
echo "Output directory:          ${OUTPUT_DIR:-./onnx_evaluation_results}"
echo "Reverse order:            ${REVERSE:+Yes}"
echo "========================================================================"
echo ""

python sim2sim_onnx_eval_cjm.py \
    --motion_file "${MOTION_FILE}" \
    --onnx_dir "${ONNX_DIR}" \
    --redis_ip "${REDIS_IP}" \
    --num_runs ${NUM_RUNS} \
    ${OUTPUT_DIR:+--output_dir "${OUTPUT_DIR}"} \
    ${REVERSE}

echo ""
echo "========================================================================"
echo "Evaluation Complete!"
echo "========================================================================"
echo "Results saved in: ${OUTPUT_DIR:-./onnx_evaluation_results}/"
echo "  - evaluation_results.json  (Detailed results)"
echo "========================================================================"
