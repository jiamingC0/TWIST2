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
import matplotlib.pyplot as plt
from pose.utils.motion_lib_pkl import MotionLib
from data_utils.rot_utils import euler_from_quaternion_torch, quat_rotate_inverse_torch

from data_utils.params import DEFAULT_MIMIC_OBS


def build_mimic_obs(
    motion_lib: MotionLib,
    t_step: int,
    control_dt: float,
    tar_motion_steps,
    device=None,
    robot_type: str = "g1",
    mask_indicator: bool = False
):
    """
    Build the mimic_obs at time-step t_step, referencing the code in MimicRunner.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            root_vel.detach().cpu().numpy().squeeze(), root_ang_vel.detach().cpu().numpy().squeeze()


class MotionServer:
    def __init__(self,
                 motion_file,
                 robot="unitree_g1_with_hands",
                 redis_ip="localhost",
                 steps="1",
                 use_remote_control=False,
                 send_start_frame_as_end_frame=False,
                 show_viewer=False):
        self.motion_file = motion_file
        self.robot = robot
        self.redis_ip = redis_ip
        self.steps = steps
        self.use_remote_control = use_remote_control
        self.send_start_frame_as_end_frame = send_start_frame_as_end_frame
        self.show_viewer = show_viewer

        # Remote control state
        self.motion_started = False if use_remote_control else True
        self.should_stop = False

        if self.robot == "unitree_g1" or self.robot == "unitree_g1_with_hands":
            self.xml_file = f"{os.path.dirname(os.path.abspath(__file__))}/../assets/g1/g1_mocap_29dof.xml"
            self.robot_base = "pelvis"
        else:
            raise ValueError(f"robot type {self.robot} not supported")

        # Connect to Redis
        self.redis_client = redis.Redis(host=self.redis_ip, port=6379, db=0)
        self.redis_client.ping()

        # Load motion library
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.motion_lib = MotionLib(self.motion_file, device=self.device)

        # Prepare the steps array
        self.tar_motion_steps = [int(x.strip()) for x in self.steps.split(",")]
        self.tar_motion_steps_tensor = torch.tensor(self.tar_motion_steps, device=self.device, dtype=torch.int)

        # Control parameters
        self.control_dt = 0.02
        motion_id = torch.tensor([0], device=self.device, dtype=torch.long)
        motion_length = self.motion_lib.get_motion_length(motion_id)
        motion_length_scalar = motion_length.item() if hasattr(motion_length, 'item') else float(motion_length)
        self.num_steps = int(motion_length_scalar / self.control_dt)

        print(f"[MotionServer] Motion file: {self.motion_file}")
        print(f"[MotionServer] Robot: {self.robot}")
        print(f"[MotionServer] Num steps: {self.num_steps} ({motion_length_scalar:.2f}s)")

        # Setup viewer if needed
        self.viewer = None
        self.sim_model = mujoco.MjModel.from_xml_path(self.xml_file)
        
        if self.show_viewer:
            self.sim_data = mujoco.MjData(self.sim_model)
            self.viewer = launch_passive(model=self.sim_model, data=self.sim_data, show_left_ui=False, show_right_ui=False)

        # Extract start frame for end frame if option is enabled
        self.start_frame_mimic_obs = None
        if self.send_start_frame_as_end_frame:
            self.start_frame_mimic_obs, _, _, _, _, _ = build_mimic_obs(
                motion_lib=self.motion_lib,
                t_step=0,
                control_dt=self.control_dt,
                tar_motion_steps=self.tar_motion_steps_tensor,
                device=self.device,
                robot_type=self.robot
            )

        self.last_mimic_obs = DEFAULT_MIMIC_OBS[self.robot]

        if self.use_remote_control:
            # Reset start and exit signal to 0
            self.redis_client.set("motion_start_signal", "0")
            self.redis_client.set("motion_exit_signal", "0")

    def stop(self):
        """Signal the server to stop."""
        self.should_stop = True

    def check_remote_control_signals(self):
        if not self.use_remote_control:
            return True, False  # motion_active, should_exit

        try:
            # Check for start signal (B button from robot controller)
            start_signal = self.redis_client.get("motion_start_signal")
            start_pressed = start_signal == b"1" if start_signal else False

            # Check for exit signal (Select button from robot controller)
            exit_signal = self.redis_client.get("motion_exit_signal")
            exit_pressed = exit_signal == b"1" if exit_signal else False

            return start_pressed, exit_pressed
        except Exception as e:
            return False, False

    def run(self):
        """Run motion server loop. Returns True if completed normally, False if stopped early."""
        print(f"[MotionServer] Streaming for {self.num_steps} steps at dt={self.control_dt:.3f} seconds...")
        time.sleep(1.0)
        try:
            t_step = 0
            while t_step < self.num_steps and not self.should_stop:
                t0 = time.time()
                # Handle remote control logic
                if self.use_remote_control:
                    # Check remote control signals
                    start_pressed, exit_pressed = self.check_remote_control_signals()

                    if exit_pressed:
                        print("[MotionServer] Exit signal received, stopping...")
                        return False

                    if not self.motion_started and start_pressed:
                        print("[MotionServer] Start signal received, beginning motion...")
                        self.motion_started = True
                    elif not self.motion_started:
                        # Keep sending default pose while waiting for start signal
                        idle_mimic_obs = self.start_frame_mimic_obs if self.send_start_frame_as_end_frame and self.start_frame_mimic_obs is not None else DEFAULT_MIMIC_OBS[self.robot]
                        self.redis_client.set(f"action_body_{self.robot}", json.dumps(idle_mimic_obs.tolist()))
                        self.redis_client.set(f"action_hand_left_{self.robot}", json.dumps(np.zeros(7).tolist()))
                        self.redis_client.set(f"action_hand_right_{self.robot}", json.dumps(np.zeros(7).tolist()))

                        # Sleep and continue to next iteration
                        elapsed = time.time() - t0
                        if elapsed < self.control_dt:
                            time.sleep(self.control_dt - elapsed)
                        continue

                # Build a mimic obs from the motion library
                mimic_obs, root_pos, root_rot, dof_pos, root_vel, root_ang_vel = build_mimic_obs(
                    motion_lib=self.motion_lib,
                    t_step=t_step,
                    control_dt=self.control_dt,
                    tar_motion_steps=self.tar_motion_steps_tensor,
                    device=self.device,
                    robot_type=self.robot
                )
                # Convert to JSON (list) to put into Redis
                mimic_obs_list = mimic_obs.tolist() if mimic_obs.ndim == 1 else mimic_obs.flatten().tolist()
                self.redis_client.set(f"action_body_{self.robot}", json.dumps(mimic_obs_list))
                self.redis_client.set(f"action_hand_left_{self.robot}", json.dumps(np.zeros(7).tolist()))
                self.redis_client.set(f"action_hand_right_{self.robot}", json.dumps(np.zeros(7).tolist()))
                self.redis_client.set(f"action_neck_{self.robot}", json.dumps(np.zeros(2).tolist()))
                self.redis_client.set("t_state", int(time.time() * 1000))  # current timestamp in ms
                self.last_mimic_obs = mimic_obs

                # Print or log it
                print(f"Step {t_step:4d}/{self.num_steps} => mimic_obs shape = {mimic_obs.shape}", end="\r")
                
                if self.show_viewer:
                    self.sim_data.qpos[:3] = root_pos
                    root_rot = root_rot[[3,0,1,2]]
                    self.sim_data.qpos[3:7] = root_rot
                    self.sim_data.qpos[7:] = dof_pos
                    mujoco.mj_forward(self.sim_model, self.sim_data)
                    robot_base_pos = self.sim_data.xpos[self.sim_model.body(self.robot_base).id]
                    self.viewer.cam.lookat = robot_base_pos
                    self.viewer.cam.distance = 2.0
                    self.viewer.sync()

                t_step += 1

                # Sleep to maintain real-time pace
                elapsed = time.time() - t0
                if elapsed < self.control_dt:
                    time.sleep(self.control_dt - elapsed)

            print()
            if self.should_stop:
                print("[MotionServer] Stopped by external signal")
                return False
            else:
                print("[MotionServer] Motion completed normally")
                return True

        except Exception as e:
            print(f"[MotionServer] Error: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up and interpolate to default pose."""
        print("[MotionServer] Cleaning up... Interpolating to default mimic_obs...")
        time_back_to_default = 2.0
        target_mimic_obs = self.start_frame_mimic_obs if self.send_start_frame_as_end_frame and self.start_frame_mimic_obs is not None else DEFAULT_MIMIC_OBS[self.robot]
        for i in range(int(time_back_to_default / self.control_dt)):
            interp_mimic_obs = self.last_mimic_obs + (target_mimic_obs - self.last_mimic_obs) * (i / (time_back_to_default / self.control_dt))
            self.redis_client.set(f"action_body_{self.robot}", json.dumps(interp_mimic_obs.tolist()))
            time.sleep(self.control_dt)
        self.redis_client.set(f"action_body_{self.robot}", json.dumps(target_mimic_obs.tolist()))

        if self.viewer:
            self.viewer.close()


def main(args, xml_file, robot_base):
    # Remote control state
    motion_started = False if args.use_remote_control else True

    if args.use_remote_control:
        print("[Motion Server] Remote control enabled. Waiting for start signal from robot controller...")

    if args.vis:
        sim_model = mujoco.MjModel.from_xml_path(xml_file)
        sim_data = mujoco.MjData(sim_model)
        viewer = launch_passive(model=sim_model, data=sim_data, show_left_ui=False, show_right_ui=False)
            

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion_file", help="Path to your *.pkl motion file for MotionLib",
                        default="../motion_data/OMOMO_g1_GMR/sub1_clothesstand_067.pkl"
                        )
    parser.add_argument("--robot", type=str, default="unitree_g1_with_hands", choices=["unitree_g1", "unitree_g1_with_hands"])
    parser.add_argument("--steps", type=str,
                        # default="1,3,5,10,15,20,30,40,50",
                        default="1",
                        help="Comma-separated steps for future frames (tar_motion_steps)")
    parser.add_argument("--vis", action="store_true", help="Visualize the motion")
    parser.add_argument("--use_remote_control", action="store_true", help="Use remote control signals from robot controller")
    parser.add_argument("--send_start_frame_as_end_frame", action="store_true", help="Use motion's first frame as end frame instead of default pose")
    parser.add_argument("--redis_ip", type=str, default="localhost", help="Redis IP")
    args = parser.parse_args()

    print("Robot type: ", args.robot)
    print("Motion file: ", args.motion_file)
    print("Steps: ", args.steps)

    HERE = os.path.dirname(os.path.abspath(__file__))

    if args.robot == "unitree_g1" or args.robot == "unitree_g1_with_hands":
        xml_file = f"{HERE}/../assets/g1/g1_mocap_29dof.xml"
        robot_base = "pelvis"
    else:
        raise ValueError(f"robot type {args.robot} not supported")

    # Use MotionServer class
    server = MotionServer(
        motion_file=args.motion_file,
        robot=args.robot,
        redis_ip=args.redis_ip,
        steps=args.steps,
        use_remote_control=args.use_remote_control,
        send_start_frame_as_end_frame=args.send_start_frame_as_end_frame,
        show_viewer=args.vis
    )
    server.run()
