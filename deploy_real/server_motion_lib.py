#!/usr/bin/env python
import argparse
import time
import redis
import json
import numpy as np
import isaacgym
import torch
from rich import print
import os
import mujoco
from mujoco.viewer import launch_passive
from pose.utils.motion_lib_pkl import MotionLib
from data_utils.rot_utils import euler_from_quaternion_torch, quat_rotate_inverse_torch

from data_utils.params import DEFAULT_MIMIC_OBS


def read_stl_binary(filepath):
    """读取二进制 STL 文件，返回所有顶点"""
    import struct
    with open(filepath, 'rb') as f:
        header = f.read(80)
        num_triangles = struct.unpack('<I', f.read(4))[0]

        vertices = []
        for _ in range(num_triangles):
            # 跳过法向量 (12 bytes)
            f.read(12)

            # 读取 3 个顶点
            for _ in range(3):
                v = struct.unpack('<3f', f.read(12))
                vertices.append(v)

            # 跳过属性 (2 bytes)
            f.read(2)

        return np.array(vertices, dtype=np.float32)


class FootAnalyzer:
    """脚掌最低点分析器"""
    def __init__(self, mesh_dir, left_stl_name="left_ankle_roll_link.STL", right_stl_name="right_ankle_roll_link.STL"):
        self.left_vertices = None
        self.right_vertices = None

        left_stl = os.path.join(mesh_dir, left_stl_name)
        right_stl = os.path.join(mesh_dir, right_stl_name)

        if os.path.exists(left_stl):
            self.left_vertices = read_stl_binary(left_stl)
            print(f"[FootAnalyzer] Loaded {left_stl_name}: {len(self.left_vertices)} vertices")
        else:
            print(f"[FootAnalyzer] Warning: {left_stl} not found")

        if os.path.exists(right_stl):
            self.right_vertices = read_stl_binary(right_stl)
            print(f"[FootAnalyzer] Loaded {right_stl_name}: {len(self.right_vertices)} vertices")
        else:
            print(f"[FootAnalyzer] Warning: {right_stl} not found")

    def get_foot_min_z(self, vertices, body_pos, body_mat):
        """计算 STL 顶点的最低 z 值"""
        min_z = float('inf')
        for vert in vertices:
            # 转换到世界坐标系
            v_world = body_pos + body_mat @ vert
            min_z = min(min_z, v_world[2])
        return min_z

    def analyze(self, sim_model, sim_data, t_step):
        """分析当前帧的脚掌最低点"""
        if self.left_vertices is None or self.right_vertices is None:
            return None

        try:
            # 获取脚踝 body 的全局位置和旋转
            left_ankle_body_id = sim_model.body("left_ankle_roll_link").id
            right_ankle_body_id = sim_model.body("right_ankle_roll_link").id

            left_ankle_pos = sim_data.xpos[left_ankle_body_id]
            right_ankle_pos = sim_data.xpos[right_ankle_body_id]

            # 获取 body 的旋转矩阵
            left_ankle_mat = sim_data.xmat[left_ankle_body_id].reshape(3, 3)
            right_ankle_mat = sim_data.xmat[right_ankle_body_id].reshape(3, 3)

            # 计算脚掌最低点
            left_foot_min_z = self.get_foot_min_z(self.left_vertices, left_ankle_pos, left_ankle_mat)
            right_foot_min_z = self.get_foot_min_z(self.right_vertices, right_ankle_pos, right_ankle_mat)

            return {
                'left_foot_min_z': left_foot_min_z,
                'right_foot_min_z': right_foot_min_z,
                'min_z': min(left_foot_min_z, right_foot_min_z)
            }
        except Exception as e:
            print(f"[FootAnalyzer] Error analyzing step {t_step}: {e}")
            return None


def quat_apply_torch(q, v):
    """
    Apply quaternion rotation to vector(s).
    q: quaternion(s) [N, 4] or [4] (xyzw format)
    v: vector(s) to rotate [N, 3] or [3]
    """
    if q.dim() == 1:
        q = q.unsqueeze(0)
    if v.dim() == 1:
        v = v.unsqueeze(0)

    q_w = q[:, 3:4]
    q_vec = q[:, :3]

    # Rodrigues formula: v' = v + 2 * q_w * (q_vec x v) + 2 * (q_vec x (q_vec x v))
    t = 2.0 * torch.cross(q_vec, v, dim=-1)
    result = v + q_w * t + torch.cross(q_vec, t, dim=-1)

    return result.squeeze() if result.shape[0] == 1 else result


def quat_apply(q, v):
    """
    Apply quaternion rotation to vector(s) - numpy version.
    q: quaternion [4] (xyzw format)
    v: vector [3] or array of vectors [N, 3]
    """
    if v.ndim == 1:
        v = v.reshape(1, 3)
    q_w = q[3]
    q_vec = q[:3]

    # Rodrigues formula: v' = v + 2 * q_w * (q_vec x v) + 2 * (q_vec x (q_vec x v))
    t = 2.0 * np.cross(q_vec, v, axis=-1)
    result = v + q_w * t + np.cross(q_vec, t, axis=-1)

    return result.squeeze() if result.shape[0] == 1 else result


def build_mimic_obs(
    motion_lib: MotionLib,
    t_step: int,
    control_dt: float,
    tar_motion_steps,
    robot_type: str = "g1",
    mask_indicator: bool = False
):
    """
    Build the mimic_obs at time-step t_step, referencing the code in MimicRunner.
    """
    device = torch.device("cuda")
    # Build times
    motion_times = torch.tensor([t_step * control_dt], device=device).unsqueeze(-1)
    obs_motion_times = tar_motion_steps * control_dt + motion_times
    obs_motion_times = obs_motion_times.flatten()
    
    # Suppose we only have a single motion in the .pkl
    motion_ids = torch.zeros(len(tar_motion_steps), dtype=torch.int, device=device)
    
    # Retrieve motion frames
    root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, local_key_body_pos, root_pos_delta_local, root_rot_delta_local = motion_lib.calc_motion_frame(motion_ids, obs_motion_times)

    # Convert to euler (roll, pitch, yaw)
    roll, pitch, yaw = euler_from_quaternion_torch(root_rot, scalar_first=False)
    roll = roll.reshape(1, -1, 1)
    pitch = pitch.reshape(1, -1, 1)
    yaw = yaw.reshape(1, -1, 1)

    # Transform velocities to root frame
    root_vel_local = quat_rotate_inverse_torch(root_rot, root_vel, scalar_first=False).reshape(1, -1, 3)
    root_ang_vel_local = quat_rotate_inverse_torch(root_rot, root_ang_vel, scalar_first=False).reshape(1, -1, 3)
    root_vel = root_vel.reshape(1, -1, 3)
    root_ang_vel = root_ang_vel.reshape(1, -1, 3)

    root_pos = root_pos.reshape(1, -1, 3)
    dof_pos = dof_pos.reshape(1, -1, dof_pos.shape[-1])
    
    # mimic_obs_buf = torch.cat((
    #             root_pos,
    #             roll, pitch, yaw,
    #             # root_vel,
    #             # root_ang_vel,
    #             root_vel_local,
    #             root_ang_vel_local,
    #             dof_pos 
    #         ), dim=-1)[:, 0:1]  # shape (1, 1, ?)
    # print("root_vel_local: ", root_vel_local)
    # Modified for better observability: root_vel_xy + root_pos_z + roll_pitch + yaw_ang_vel + dof_pos
    if mask_indicator:
        mimic_obs_buf = torch.cat((
                    # root position: xy velocity + z position
                    root_vel_local[..., :2], # 2 dims (xy velocity instead of xy position)
                    root_pos[..., 2:3], # 1 dim (z position)
                    # root rotation: roll/pitch + yaw angular velocity
                    roll, pitch, # 2 dims (roll/pitch orientation)
                    root_ang_vel_local[..., 2:3], # 1 dim (yaw angular velocity)
                    dof_pos,
                ), dim=-1)[:, :]  # shape (1, 1, 6 + num_dof)
        # append mask indicator 1
        mask_indicator = torch.ones(1, mimic_obs_buf.shape[1], 1).to(device)
        mimic_obs_buf = torch.cat((mimic_obs_buf, mask_indicator), dim=-1)
    else:
        mimic_obs_buf = torch.cat((
                    # root position: xy velocity + z position
                    root_vel_local[..., :2], # 2 dims (xy velocity instead of xy position)
                    root_pos[..., 2:3], # 1 dim (z position)
                    # root rotation: roll/pitch + yaw angular velocity
                    roll, pitch, # 2 dims (roll/pitch orientation)
                    root_ang_vel_local[..., 2:3], # 1 dim (yaw angular velocity)
                    dof_pos,
                ), dim=-1)[:, :]  # shape (1, 1, 6 + num_dof)

    # print("root height: ", root_pos[..., 2:3].detach().cpu().numpy().squeeze())
    mimic_obs_buf = mimic_obs_buf.reshape(1, -1)

    return mimic_obs_buf.detach().cpu().numpy().squeeze(), root_pos.detach().cpu().numpy().squeeze(), \
        root_rot.detach().cpu().numpy().squeeze(), dof_pos.detach().cpu().numpy().squeeze(), \
            root_vel.detach().cpu().numpy().squeeze(), root_ang_vel.detach().cpu().numpy().squeeze(), \
            local_key_body_pos[0].detach().cpu().numpy()


def main(args, xml_file, robot_base):
    # Remote control state
    motion_started = False if args.use_remote_control else True

    if args.use_remote_control:
        print("[Motion Server] Remote control enabled. Waiting for start signal from robot controller...")

    # 初始化脚掌分析器（仅对 R1 机器人）
    foot_analyzer = None
    if args.vis and args.robot == "unitree_r1":
        HERE = os.path.dirname(os.path.abspath(__file__))
        mesh_dir = os.path.join(HERE, "../assets/unitree_r1/meshes")
        foot_analyzer = FootAnalyzer(mesh_dir)

    if args.vis:
        sim_model = mujoco.MjModel.from_xml_path(xml_file)
        sim_data = mujoco.MjData(sim_model)
        viewer = launch_passive(model=sim_model, data=sim_data, show_left_ui=False, show_right_ui=False)
            
    # 1. Connect to Redis
    redis_ip = args.redis_ip
    # redis_client = redis.Redis(host="localhost", port=6379, db=0)
    # redis_client = redis.Redis(host="127.0.0.1", port=6379, db=0)
    # redis_client = redis.Redis(host="192.168.110.24", port=6379, db=0)
    redis_client = redis.Redis(host=redis_ip, port=6379, db=0)
    redis_client.ping()


    # 2. Load motion library
    device = "cuda" if torch.cuda.is_available() else "cpu"
    motion_lib = MotionLib(args.motion_file, device=device)

    # 3. Prepare the steps array
    tar_motion_steps = [int(x.strip()) for x in args.steps.split(",")]
    tar_motion_steps_tensor = torch.tensor(tar_motion_steps, device=device, dtype=torch.int)

    # 4. Parse body names to visualize
    # 全部身体部位名称
    all_body_names = motion_lib._body_link_list

    # 解析用户指定的身体部位
    selected_body_indices = []
    if args.vis_body_names:
        # 用户指定了具体的身体部位名称
        requested_names = [name.strip() for name in args.vis_body_names.split(",")]
        for name in requested_names:
            if name in all_body_names:
                selected_body_indices.append(all_body_names.index(name))
            else:
                print(f"[Warning] Body '{name}' not found in motion library")
    elif args.vis_body_indices:
        # 用户指定了身体部位索引
        try:
            selected_body_indices = [int(idx.strip()) for idx in args.vis_body_indices.split(",")]
            # 验证索引有效性
            valid_indices = []
            for idx in selected_body_indices:
                if 0 <= idx < len(all_body_names):
                    valid_indices.append(idx)
                else:
                    print(f"[Warning] Index {idx} out of range (0-{len(all_body_names)-1})")
            selected_body_indices = valid_indices
        except ValueError:
            print("[Error] Invalid indices format, please use comma-separated integers")
            selected_body_indices = []
    elif args.vis_body_pos:
        # 可视化所有身体部位
        selected_body_indices = list(range(len(all_body_names)))
    else:
        # 默认可视化关键部位
        default_key_bodies = [
            'left_rubber_hand', 'right_rubber_hand',  # 手
            'left_ankle_roll_link', 'right_ankle_roll_link',  # 脚踝
            'left_knee_link', 'right_knee_link',  # 膝盖
            'left_elbow_link', 'right_elbow_link',  # 手肘
            'head_mocap'  # 头部
        ]
        for name in default_key_bodies:
            if name in all_body_names:
                selected_body_indices.append(all_body_names.index(name))

    selected_body_names = [all_body_names[i] for i in selected_body_indices]
    print(f"[Motion Server] Visualizing {len(selected_body_names)} body parts: {selected_body_names}")

    # Remove matplotlib visualization code - use MuJoCo markers instead

    # 4. Loop over time steps and publish mimic obs
    control_dt = 0.02
    
    # 4.5 Extract start frame for end frame if option is enabled
    start_frame_mimic_obs = None
    if args.send_start_frame_as_end_frame:
        start_frame_mimic_obs, _, _, _, _, _, _ = build_mimic_obs(
            motion_lib=motion_lib,
            t_step=0,
            control_dt=control_dt,
            tar_motion_steps=tar_motion_steps_tensor,
            robot_type=args.robot
        )
    # compute num_steps based on motion length
    motion_id = torch.tensor([0], device=device, dtype=torch.long)
    motion_length = motion_lib.get_motion_length(motion_id)
    num_steps = int(motion_length / control_dt)
    
    print(f"[Motion Server] Streaming for {num_steps} steps at dt={control_dt:.3f} seconds...")
    print(f"[Motion Server] Body link list: {all_body_names}")

    last_mimic_obs = DEFAULT_MIMIC_OBS[args.robot]

    # 检查是否有脚部索引
    foot_indices = [i for i, name in enumerate(all_body_names) if 'ankle' in name or 'foot' in name or 'toe' in name]
    if foot_indices:
        print(f"[Motion Server] Found foot body parts at indices: {foot_indices}")
    else:
        print(f"[Motion Server] Warning: No foot body parts found in link_body_list")
    
    # Helper function to check remote control signals
    def check_remote_control_signals():
        if not args.use_remote_control:
            return True, False  # motion_active, should_exit
        
        try:
            # Check for start signal (B button from robot controller)
            start_signal = redis_client.get("motion_start_signal")
            start_pressed = start_signal == b"1" if start_signal else False
            
            # Check for exit signal (Select button from robot controller)
            exit_signal = redis_client.get("motion_exit_signal") 
            exit_pressed = exit_signal == b"1" if exit_signal else False
            
            return start_pressed, exit_pressed
        except Exception as e:
            return False, False
    
    if args.use_remote_control:
        # reset start and exit signal to 0
        redis_client.set("motion_start_signal", "0")
        redis_client.set("motion_exit_signal", "0")
    
    try:
        # for t_step in range(num_steps):
        t_step = 0
        while True:
            t0 = time.time()
            
            # Handle remote control logic
            if args.use_remote_control:
                # Check remote control signals
                start_pressed, exit_pressed = check_remote_control_signals()

                if exit_pressed:
                    print("[Motion Server] Exit signal received, stopping...")
                    break
                    
                if not motion_started and start_pressed:
                    print("[Motion Server] Start signal received, beginning motion...")
                    motion_started = True
                elif not motion_started:
                    # Keep sending default pose while waiting for start signal
                    idle_mimic_obs = start_frame_mimic_obs if args.send_start_frame_as_end_frame and start_frame_mimic_obs is not None else DEFAULT_MIMIC_OBS[args.robot]
                    redis_client.set(f"action_body_{args.robot}", json.dumps(idle_mimic_obs.tolist()))
                    redis_client.set(f"action_hand_left_{args.robot}", json.dumps(np.zeros(7).tolist()))
                    redis_client.set(f"action_hand_right_{args.robot}", json.dumps(np.zeros(7).tolist()))

                    # Sleep and continue to next iteration
                    elapsed = time.time() - t0
                    if elapsed < control_dt:
                        time.sleep(control_dt - elapsed)
                    continue

            # Build a mimic obs from the motion library
            mimic_obs, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, local_key_body_pos = build_mimic_obs(
                motion_lib=motion_lib,
                t_step=t_step,
                control_dt=control_dt,
                tar_motion_steps=tar_motion_steps_tensor,
                robot_type=args.robot
            )

            # Convert to JSON (list) to put into Redis
            mimic_obs_list = mimic_obs.tolist() if mimic_obs.ndim == 1 else mimic_obs.flatten().tolist()
            redis_client.set(f"action_body_{args.robot}", json.dumps(mimic_obs_list))
            redis_client.set(f"action_hand_left_{args.robot}", json.dumps(np.zeros(7).tolist()))
            redis_client.set(f"action_hand_right_{args.robot}", json.dumps(np.zeros(7).tolist()))
            redis_client.set(f"action_neck_{args.robot}", json.dumps(np.zeros(2).tolist()))
            last_mimic_obs = mimic_obs
            
            # Print or log it
            # print(f"Step {t_step:4d} => mimic_obs shape = {mimic_obs.shape} published...", end="\r")
            if t_step <= 3:
                print(f"{t_step} root_rot: \n {root_rot}")
            if args.vis:
                sim_data.qpos[:3] = root_pos
                # filp rot
                # root_rot = root_rot[[1,2,3,0]]
                root_rot = root_rot[[3,0,1,2]]
                if t_step <= 3:
                    print(f"{t_step} root_rot_wxyz: \n {root_rot}")
                sim_data.qpos[3:7] = root_rot
                sim_data.qpos[7:] = dof_pos
                if t_step < 5:
                    print(f"{t_step} sim_data.qpos: \n {sim_data.qpos[:7]}")
                mujoco.mj_forward(sim_model, sim_data)

                # 分析脚掌最低点（仅 R1 机器人）
                if foot_analyzer is not None and t_step % 30 == 0:
                    result = foot_analyzer.analyze(sim_model, sim_data, t_step)
                    if result is not None:
                        print(f"Step {t_step}: left_foot_min_z={result['left_foot_min_z']:.4f}m, "
                              f"right_foot_min_z={result['right_foot_min_z']:.4f}m, "
                              f"min_z={result['min_z']:.4f}m")

                mujoco.mj_step(sim_model, sim_data)  # 物理步进以计算接触力

                # 可视化选中的身体部位 - 在 MuJoCo 中绘制小球
                if args.vis_body_pos and selected_body_indices:
                    # 提取选中身体部位的位置（局部坐标）
                    selected_body_pos = local_key_body_pos[selected_body_indices]

                    # 将局部坐标转换为全局坐标
                    # root_rot 格式是 [x, y, z, w]，需要转换为 [x, y, z, w]
                    selected_pos_tensor = torch.from_numpy(selected_body_pos).float()
                    root_rot_xyzw = torch.from_numpy(np.array([root_rot[1], root_rot[2], root_rot[3], root_rot[0]])).float()

                    # 应用旋转: pos_global = root_rot * pos_local + root_pos
                    selected_pos_global = quat_apply_torch(root_rot_xyzw.unsqueeze(0), selected_pos_tensor).squeeze(0).numpy() + root_pos

                    # 使用 viewer.user_scn 绘制小球
                    viewer.user_scn.ngeom = len(selected_body_indices)
                    # 单位旋转矩阵 (identity matrix) - (9,1)形状
                    identity_mat = np.eye(3, dtype=np.float64).reshape(9, 1)

                    for i, (pos, name) in enumerate(zip(selected_pos_global, selected_body_names)):
                        # 设置小球颜色：左侧红色，右侧蓝色，其他黄色 - (4,1)形状
                        if 'left' in name:
                            color = np.array([1, 0, 0, 1], dtype=np.float32).reshape(4, 1)  # 红色 (rgba)
                        elif 'right' in name:
                            color = np.array([0, 0, 1, 1], dtype=np.float32).reshape(4, 1)  # 蓝色 (rgba)
                        else:
                            color = np.array([1, 1, 0, 1], dtype=np.float32).reshape(4, 1)  # 黄色 (rgba)

                        # 创建小球几何体
                        # mjv_initGeom(geom, type, size, pos, mat, rgba)
                        # size: (3,1), pos: (3,1), mat: (9,1), rgba: (4,1)
                        mujoco.mjv_initGeom(
                            viewer.user_scn.geoms[i],
                            mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([0.03, 0.03, 0.03], dtype=np.float64).reshape(3, 1),  # size (半径)
                            np.array(pos, dtype=np.float64).reshape(3, 1),  # position
                            identity_mat,  # rotation matrix (9,1)
                            color  # rgba color (4,1)
                        )

                    robot_base_pos = sim_data.xpos[sim_model.body(robot_base).id]
                    viewer.cam.lookat = robot_base_pos

                    # 每隔30步打印一次位置到终端
                    # if t_step % 30 == 0:
                    #     print(f"\n--- Selected Body Positions at step {t_step} ---")
                    #     for name, pos_local, pos_global in zip(selected_body_names, selected_body_pos, selected_pos_global):
                    #         print(f"  {name:30s}: local=[{pos_local[0]:7.3f}, {pos_local[1]:7.3f}, {pos_local[2]:7.3f}]")
                    #         print(f"{'':33s} global=[{pos_global[0]:7.3f}, {pos_global[1]:7.3f}, {pos_global[2]:7.3f}]")
                else:
                    robot_base_pos = sim_data.xpos[sim_model.body(robot_base).id]
                    viewer.cam.lookat = robot_base_pos

                # 检查机器人是否悬空（通过脚踝高度判断）
                try:
                    left_ankle_pos = sim_data.xpos[sim_model.body("left_ankle_roll_link").id]
                    right_ankle_pos = sim_data.xpos[sim_model.body("right_ankle_roll_link").id]
                    ground_height = 0.0  # 地面在 z=0
                    ankle_height_threshold = 0.05  # 脚踝离地阈值 (5cm)
                    
                    left_foot_height = left_ankle_pos[2]  # z坐标
                    right_foot_height = right_ankle_pos[2]
                    is_airborne = (left_foot_height > ankle_height_threshold and right_foot_height > ankle_height_threshold)

                    if t_step % 30 == 0:
                        print(f"Step {t_step}: left_ankle_z={left_foot_height:.3f}m, right_ankle_z={right_foot_height:.3f}m, airborne={is_airborne}")
                except Exception as e:
                    # 如果 body 名称不匹配，跳过检测
                    pass

                viewer.sync()
            
            t_step += 1
            if t_step >= num_steps:
                break
            # Sleep to maintain real-time pace
            elapsed = time.time() - t0
            if elapsed < control_dt:
                time.sleep(control_dt - elapsed)
    
      
    except Exception as e:
        print(f"[Motion Server] Error: {e}")
        print("[Motion Server] Keyboard interrupt. Interpolating to default mimic_obs...")
        # do linear interpolation to the last mimic_obs
        time_back_to_default = 2.0
        target_mimic_obs = start_frame_mimic_obs if args.send_start_frame_as_end_frame and start_frame_mimic_obs is not None else DEFAULT_MIMIC_OBS[args.robot]
        for i in range(int(time_back_to_default / control_dt)):
            interp_mimic_obs = last_mimic_obs + (target_mimic_obs - last_mimic_obs) * (i / (time_back_to_default / control_dt))
            redis_client.set(f"action_body_{args.robot}", json.dumps(interp_mimic_obs.tolist()))
            time.sleep(control_dt)
        redis_client.set(f"action_body_{args.robot}", json.dumps(target_mimic_obs.tolist()))
        last_mimic_obs = target_mimic_obs
        if args.vis:
            viewer.close()
        time.sleep(0.5)
        exit()
    finally:
        print("[Motion Server] Exiting...Interpolating to default mimic_obs...")
        # do linear interpolation to the last mimic_obs
        time_back_to_default = 2.0
        target_mimic_obs = start_frame_mimic_obs if args.send_start_frame_as_end_frame and start_frame_mimic_obs is not None else DEFAULT_MIMIC_OBS[args.robot]
        for i in range(int(time_back_to_default / control_dt)):
            interp_mimic_obs = last_mimic_obs + (target_mimic_obs - last_mimic_obs) * (i / (time_back_to_default / control_dt))
            redis_client.set(f"action_body_{args.robot}", json.dumps(interp_mimic_obs.tolist()))
            time.sleep(control_dt)
        redis_client.set(f"action_body_{args.robot}", json.dumps(target_mimic_obs.tolist()))
        last_mimic_obs = target_mimic_obs
        if args.vis:
            viewer.close()
        time.sleep(0.5)
        exit()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion_file", help="Path to your *.pkl motion file for MotionLib", 
                        default="../motion_data/OMOMO_g1_GMR/sub1_clothesstand_067.pkl"
                        )
    parser.add_argument("--robot", type=str, default="unitree_g1_with_hands", choices=["unitree_g1", "unitree_g1_with_hands","unitree_r1"])
    parser.add_argument("--steps", type=str,
                        # default="1,3,5,10,15,20,30,40,50",
                        default="1",
                        help="Comma-separated steps for future frames (tar_motion_steps)")
    parser.add_argument("--vis", action="store_true", help="Visualize the motion")
    parser.add_argument("--vis_body_pos", action="store_true", help="Visualize selected body positions in 3D plot")
    parser.add_argument("--vis_body_names", type=str, default="",
                        help="Comma-separated body names to visualize (e.g., 'left_hand,right_knee,head')")
    parser.add_argument("--vis_body_indices", type=str, default="",
                        help="Comma-separated body indices to visualize (e.g., '0,5,10')")
    parser.add_argument("--use_remote_control", action="store_true", help="Use remote control signals from robot controller")
    parser.add_argument("--send_start_frame_as_end_frame", action="store_true", help="Use motion's first frame as end frame instead of default pose")
    parser.add_argument("--redis_ip", type=str, default="localhost", help="Redis IP")
    args = parser.parse_args()

    args.vis = True
    

    print("Robot type: ", args.robot)
    print("Motion file: ", args.motion_file)
    print("Steps: ", args.steps)
    
    HERE = os.path.dirname(os.path.abspath(__file__))
    
    if args.robot == "unitree_g1" or args.robot == "unitree_g1_with_hands":
        xml_file = f"{HERE}/../assets/g1/g1_mocap_29dof.xml"
        robot_base = "pelvis"
    elif args.robot == "unitree_r1":
        xml_file = f"{HERE}/../assets/unitree_r1/scene_r1.xml"
        robot_base = "pelvis_link"
    else:
        raise ValueError(f"robot type {args.robot} not supported")
    
    
    main(args, xml_file, robot_base)
