SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
# Ensure module imports work no matter where this script is invoked from.
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH+:$PYTHONPATH}"
cd "$SCRIPT_DIR"
# ckpt_path=${SCRIPT_DIR}/assets/ckpts/twist2_1017_20k.onnx
ckpt_path=${SCRIPT_DIR}/legged_gym/logs/g1_stu_future_cjm/0130-6-f6de44/model_20000.onnx



#0130-6-f6de44/model_20000.onnx
#0130-7-e1819d/model_22500.onnx
#0130-3-9657fe/model_12500.onnx
#0130-5-7ba8fc/model_2000.onnx

# cd deploy_real

python "$SCRIPT_DIR/deploy_real/server_low_level_g1_sim.py" \
    --xml "$SCRIPT_DIR/assets/g1/g1_sim2sim_29dof.xml" \
    --policy ${ckpt_path} \
    --device cuda \
    --measure_fps 1 \
    --policy_frequency 100 \
    --limit_fps 1 \
    # --record_proprio \
