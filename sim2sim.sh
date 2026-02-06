SCRIPT_DIR=$(dirname $(realpath $0))
# ckpt_path=${SCRIPT_DIR}/assets/ckpts/twist2_1017_20k.onnx
ckpt_path=${SCRIPT_DIR}/legged_gym/logs/g1_stu_future_cjm/0130-6-f6de44/model_30000.onnx


#0130-6-f6de44/model_30000.onnx


cd deploy_real

python server_low_level_g1_sim_cjm.py \
    --xml ../assets/g1/g1_sim2sim_29dof.xml \
    --policy ${ckpt_path} \
    --device cuda \
    --measure_fps 1 \
    --policy_frequency 100 \
    --limit_fps 1 \
    # --record_proprio \
