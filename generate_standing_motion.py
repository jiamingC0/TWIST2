import pickle
import numpy as np

# ===== 参数 =====
SRC_PKL = "/home/galbot/Galbot/track/storage/data/mocap/20251215_wufengchao_bvh_pkl/251215-083327.pkl"
OUT_PKL = "/home/galbot/MyTWIST2/TWIST2/assets/example_motions/standing_60s.pkl"
DURATION = 60.0  # seconds

# ===== 读取原始参考 =====
with open(SRC_PKL, "rb") as f:
    src = pickle.load(f)

fps = float(src["fps"])
T = int(DURATION * fps)

print(f"Using fps = {fps}, T = {T}")

# ===== 取第一帧作为站立姿态 =====
root_pos_0 = src["root_pos"][0]          # (3,)
root_rot_0 = src["root_rot"][0]          # (4,)
dof_pos_0  = src["dof_pos"][0]           # (29,)
local_body_pos_0 = src["local_body_pos"][0]  # (38, 3)

# ===== 构造站立轨迹 =====
stand_data = {
    "fps": fps,
    "root_pos": np.repeat(root_pos_0[None, :], T, axis=0),
    "root_rot": np.repeat(root_rot_0[None, :], T, axis=0),
    "dof_pos":  np.repeat(dof_pos_0[None, :], T, axis=0),
    "local_body_pos": np.repeat(local_body_pos_0[None, :, :], T, axis=0),
    "link_body_list": src["link_body_list"],  # 原样继承
}

# ===== 保存 =====
with open(OUT_PKL, "wb") as f:
    pickle.dump(stand_data, f)

print(f"Saved stand reference: {OUT_PKL}")
