import argparse
import json
import time
import numpy as np
import redis
import mujoco
import torch
from rich import print
from collections import deque
import mujoco.viewer as mjv
from tqdm import tqdm
import os
from data_utils.rot_utils import quatToEuler, quat_rotate_inverse_np
from deploy_real.redis_protocol import RedisKeys, motion_phase_is_stand, is_truthy, normalize_motion_phase
from deploy_real.redis_io import RedisIO
from deploy_real.metrics_recorder import MetricsRecorder
from deploy_real.obs_builder import ObservationBuilder, ObsConfig
from deploy_real.policy_runner import PolicyRunner, PolicyConfig

try:
    import onnxruntime as ort
except ImportError:
    ort = None


class OnnxPolicyWrapper:
    """Minimal wrapper so ONNXRuntime policies mimic TorchScript call signature."""

    def __init__(self, session, input_name, output_index=0):
        self.session = session
        self.input_name = input_name
        self.output_index = output_index

    def __call__(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        if isinstance(obs_tensor, torch.Tensor):
            obs_np = obs_tensor.detach().cpu().numpy()
        else:
            obs_np = np.asarray(obs_tensor, dtype=np.float32)
        outputs = self.session.run(None, {self.input_name: obs_np})
        result = outputs[self.output_index]
        if not isinstance(result, np.ndarray):
            result = np.asarray(result, dtype=np.float32)
        return torch.from_numpy(result.astype(np.float32))


def load_onnx_policy(policy_path: str, device: str) -> OnnxPolicyWrapper:
    if ort is None:
        raise ImportError("onnxruntime is required for ONNX policy inference but is not installed.")
    providers = []
    available = ort.get_available_providers()
    if device.startswith('cuda'):
        if 'CUDAExecutionProvider' in available:
            providers.append('CUDAExecutionProvider')
        else:
            print("CUDAExecutionProvider not available in onnxruntime; falling back to CPUExecutionProvider.")
    providers.append('CPUExecutionProvider')
    session = ort.InferenceSession(policy_path, providers=providers)
    input_name = session.get_inputs()[0].name
    print(f"ONNX policy loaded from {policy_path} using providers: {session.get_providers()}")
    return OnnxPolicyWrapper(session, input_name)


class RealTimePolicyController:
    def __init__(self,
                 xml_file,
                 policy_path,
                 device='cuda',
                 record_video=False,
                 record_proprio=False,
                 measure_fps=False,
                 limit_fps=True,
                 policy_frequency=50,
                 show_viewer=True,
                 ):
        self.measure_fps = measure_fps
        self.limit_fps = limit_fps
        self.show_viewer = show_viewer
        self.should_stop = False

        self.redis_client = None
        try:
            self.redis_io = RedisIO.connect(host='localhost', port=6379, db=0)
            self.redis_client = self.redis_io.client
            self.redis_pipeline = self.redis_io.pipeline
        except Exception as e:
            print(f"Error connecting to Redis: {e}")
        self.redis_keys = RedisKeys()

        self.device = device
        self.policy = load_onnx_policy(policy_path, device)

        # Create MuJoCo sim
        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.model.opt.timestep = 0.001
        self.data = mujoco.MjData(self.model)

        self.viewer = None
        if self.show_viewer:
            self.viewer = mjv.launch_passive(self.model, self.data, show_left_ui=False, show_right_ui=False)
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = 0
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 0
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = 0
            self.viewer.cam.distance = 2.0

        self.num_actions = 29
        self.sim_duration = 100000.0
        self.sim_dt = 0.001
        # real frequency = 1 / (decimation * sim_dt)
        # ==> decimation = 1 / (real frequency * sim_dt)
        self.sim_decimation = 1 / (policy_frequency * self.sim_dt)
        print(f"sim_decimation: {self.sim_decimation}")

        self.last_action = np.zeros(self.num_actions, dtype=np.float32)

        # G1 specific configuration
        self.default_dof_pos = np.array([
                -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,  # left leg (6)
                -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,  # right leg (6)
                0.0, 0.0, 0.0, # torso (3)
                0.0, 0.4, 0.0, 1.2, 0.0, 0.0, 0.0, # left arm (7)
                0.0, -0.4, 0.0, 1.2, 0.0, 0.0, 0.0, # right arm (7)
            ])

        self.mujoco_default_dof_pos = np.concatenate([
            np.array([0, 0, 0.793]),
            np.array([1, 0, 0, 0]),
             np.array([-0.2, 0.0, 0.0, 0.4, -0.2, 0.0,  # left leg (6)
                -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,  # right leg (6)
                0.0, 0.0, 0.0, # torso (3)
                0.0, 0.2, 0.0, 1.2, 0.0, 0.0, 0.0, # left arm (7)
                0.0, -0.2, 0.0, 1.2, 0.0, 0.0, 0.0, # right arm (7)
                ])
        ])

        self.stiffness = np.array([
                100, 100, 100, 150, 40, 40,
                100, 100, 100, 150, 40, 40,
                150, 150, 150,
                40, 40, 40, 40, 4.0, 4.0, 4.0,
                40, 40, 40, 40, 4.0, 4.0, 4.0,
            ])
        self.damping = np.array([
                2, 2, 2, 4, 2, 2,
                2, 2, 2, 4, 2, 2,
                4, 4, 4,
                5, 5, 5, 5, 0.2, 0.2, 0.2,
                5, 5, 5, 5, 0.2, 0.2, 0.2,
            ])

        
        self.torque_limits = np.array([
                100, 100, 100, 150, 40, 40,
                100, 100, 100, 150, 40, 40,
                150, 150, 150,
                40, 40, 40, 40, 4.0, 4.0, 4.0,
                40, 40, 40, 40, 4.0, 4.0, 4.0,
            ])

        self.action_scale = np.array([
                0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                0.5, 0.5, 0.5,
                0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
            ])

        self.policy_runner = PolicyRunner(
            self.policy,
            self.device,
            PolicyConfig(
                action_scale=self.action_scale,
                default_dof_pos=self.default_dof_pos,
                action_clip=10.0,
            ),
        )

        self.ankle_idx = [4, 5, 10, 11]
        n_task_id = 1
        self.n_mimic_obs = 35 + n_task_id  # 6 + 29 (modified: root_vel_xy + root_pos_z + roll_pitch + yaw_ang_vel + dof_pos)
        self.n_proprio = 3 + 2 + 3*29    # from config analysis
        self.n_obs_single = 35 + n_task_id + 3 + 2 + 3*29  # n_mimic_obs + n_proprio = 35 + 92 = 127
        self.history_len = 10
        
        self.total_obs_size = self.n_obs_single * (self.history_len + 1) + self.n_mimic_obs   # 127*11 + 35 = 1402

        print(f"TWIST2 Controller Configuration:")
        print(f"  n_mimic_obs: {self.n_mimic_obs}")
        print(f"  n_proprio: {self.n_proprio}")
        print(f"  n_obs_single: {self.n_obs_single}")
        print(f"  history_len: {self.history_len}")
        print(f"  total_obs_size: {self.total_obs_size}")

        # Initialize history buffer
        self.proprio_history_buf = deque(maxlen=self.history_len)
        for _ in range(self.history_len):
            self.proprio_history_buf.append(np.zeros(self.n_obs_single, dtype=np.float32))

        self.obs_builder = ObservationBuilder(
            ObsConfig(
                n_mimic_obs=self.n_mimic_obs,
                n_proprio=self.n_proprio,
                n_obs_single=self.n_obs_single,
                history_len=self.history_len,
                total_obs_size=self.total_obs_size,
            ),
            default_dof_pos=self.default_dof_pos,
            ankle_idx=self.ankle_idx,
        )

        # Recording
        self.record_video = record_video
        self.record_proprio = record_proprio
        self.proprio_recordings = [] if record_proprio else None
        

    def reset_sim(self):
        """Reset simulation to initial state"""
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def reset(self, init_pos):
        """Reset robot to initial position"""
        self.data.qpos[:] = init_pos
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def extract_data(self):
        """Extract robot state data"""
        n_dof = self.num_actions
        dof_pos = self.data.qpos[7:7+n_dof]
        dof_vel = self.data.qvel[6:6+n_dof]
        quat = self.data.qpos[3:7]
        ang_vel = self.data.qvel[3:6]
        root_pos = self.data.qpos[0:3]  # root position [x, y, z]
        root_vel = self.data.qvel[0:3]  # root linear velocity
        sim_torque = self.data.ctrl
        return dof_pos, dof_vel, quat, ang_vel, sim_torque, root_pos, root_vel

    def stop(self):
        """Signal the controller to stop."""
        self.should_stop = True

    def check_fall_from_redis(self):
        """Check for fall signal from Redis by reading robot state."""
        try:
            # Read robot state from Redis
            state_body_json = self.redis_io.get_state_body("unitree_g1_with_hands")
            if not state_body_json:
                return False
            state_body = state_body_json
            # state_body format: [ang_vel (3), roll_pitch (2), dof_pos (29)] = 34 dims
            # ang_vel indices: 0-2, roll_pitch indices: 3-4
            roll = state_body[3]
            pitch = state_body[4]

            # Check orientation fall: roll or pitch exceeds 60 degrees (1.05 radians)
            orientation_fall = abs(roll) > 1.05 or abs(pitch) > 1.05

            # Check height fall: root position z < 0.4m
            height_fall = False
            root_pos_json = self.redis_io.get_root_pos("unitree_g1_with_hands")
            if root_pos_json:
                root_pos = root_pos_json
                root_height = root_pos[2]  # z-axis
                height_fall = root_height < 0.4  # 40cm threshold

            if not height_fall and orientation_fall:
                root_height = root_pos_json[2] if root_pos_json else 0
                print(f"[PolicyController] Fall detected: roll={roll:.2f}, pitch={pitch:.2f}, root_height={root_height:.2f}")

            return height_fall or orientation_fall
        except Exception as e:
            return False

    def check_motion_server_active(self):
        """Check if motion server is still active by checking Redis updates."""
        try:
            # Check timestamp
            t_state = self.redis_io.get_t_state()
            if t_state:
                t_state_ms = int(t_state)
                current_ms = int(time.time() * 1000)
                # If no update for 5 seconds, assume motion server stopped (increased from 2s to 5s)
                return (current_ms - t_state_ms) < 5000
            return False
        except:
            return False

    def run(self, timeout=None, collect_metrics=False):
        """
        Main simulation loop.
        Returns: 'completed', 'timeout', 'motion_server_stopped', 'fell'
        """
        print("Starting TWIST2 simulation...")

        # Video recording setup
        if self.record_video:
            import imageio
            mp4_writer = imageio.get_writer('twist2_simulation.mp4', fps=30)
        else:
            mp4_writer = None

        self.reset_sim()
        self.reset(self.mujoco_default_dof_pos)

        steps = int(self.sim_duration / self.sim_dt)
        if timeout:
            steps = int(timeout / self.sim_dt)

        pbar = tqdm(range(steps), desc="Simulating TWIST2...", 
           position=1, 
           leave=False, 
           ncols=100,
           bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        # Send initial proprio to redis
        initial_obs = np.zeros(self.n_obs_single, dtype=np.float32)
        self.redis_io.set_state_body("unitree_g1_with_hands", initial_obs)
        self.redis_io.set_state_hands_neck("unitree_g1_with_hands")
        self.redis_io.flush()

        measure_fps = self.measure_fps
        fps_measurements = []
        fps_iteration_count = 0
        fps_measurement_target = 1000
        last_policy_time = None

        # Add policy execution FPS tracking for frequent printing
        policy_execution_times = []
        policy_step_count = 0
        policy_fps_print_interval = 100

        start_time = time.time()
        fall_detection_delay = 2.0  # Delay before checking for falls
        motion_grace_seconds = 10.0
        saw_t_state = False
        start_time_ms = int(start_time * 1000)
        metrics_recorder = MetricsRecorder()

        def _format_result(status):
            if not collect_metrics:
                return status
            elapsed_time = time.time() - start_time
            policy_dt = float(self.sim_decimation * self.sim_dt)
            steps_total = metrics_recorder.total.steps
            if steps_total <= 0:
                return {
                    "status": status,
                    "steps": 0,
                    "elapsed_time": elapsed_time,
                    "policy_dt": policy_dt,
                    "sim_time": 0.0,
                    "metrics_total": {},
                    "metrics_stand": {},
                    "metrics_motion": {},
                }
            sim_time = steps_total * policy_dt

            metrics_out = metrics_recorder.as_dict()
            metrics_out_total = metrics_out["total"]
            metrics_out_stand = metrics_out["stand"]
            metrics_out_motion = metrics_out["motion"]
            result = {
                "status": status,
                "steps": steps_total,
                "elapsed_time": elapsed_time,
                "policy_dt": policy_dt,
                "sim_time": sim_time,
                "metrics_total": metrics_out_total,
                "metrics_stand": metrics_out_stand,
                "metrics_motion": metrics_out_motion,
            }
            if status == "fell":
                result["fell_time"] = elapsed_time
            return result

        try:
            for i in pbar:
                if self.should_stop:
                    print("[PolicyController] Stopped by external signal")
                    return _format_result('stopped')

                # Check timeout
                if timeout and (time.time() - start_time) > timeout:
                    print("[PolicyController] Timeout reached")
                    return _format_result('timeout')

                # Check if motion server is still active
                if not saw_t_state:
                    try:
                        t_state = self.redis_io.get_t_state()
                        if t_state:
                            t_state_ms = int(t_state)
                            if t_state_ms >= start_time_ms:
                                saw_t_state = True
                    except Exception:
                        pass

                # Only enforce motion server active check after initial grace or after t_state seen.
                if (saw_t_state or (time.time() - start_time) > motion_grace_seconds):
                    if not self.check_motion_server_active():
                        try:
                            t_state = self.redis_client.get(self.redis_keys.T_STATE)
                            if t_state:
                                t_state_ms = int(t_state)
                                current_ms = int(time.time() * 1000)
                                print(f"[PolicyController] Motion server stopped: last_t_state_age_ms={current_ms - t_state_ms}, saw_t_state={saw_t_state}, t_state_ms={t_state_ms}, start_time_ms={start_time_ms}")
                            else:
                                print(f"[PolicyController] Motion server stopped: no t_state, saw_t_state={saw_t_state}")
                        except Exception as e:
                            print(f"[PolicyController] Motion server stopped: error reading t_state ({e})")
                        return _format_result('motion_server_stopped')

                t_start = time.time()
                dof_pos, dof_vel, quat, ang_vel, sim_torque, root_pos, root_vel = self.extract_data()

                if i % self.sim_decimation == 0:
                    # Build proprioceptive observation
                    rpy = quatToEuler(quat)
                    obs_proprio = self.obs_builder.build_proprio(
                        ang_vel=ang_vel,
                        rpy=rpy,
                        dof_pos=dof_pos,
                        dof_vel=dof_vel,
                        last_action=self.last_action,
                    )

                    state_body = self.obs_builder.build_state_body(ang_vel, rpy, dof_pos)

                    # Send proprio to redis
                    self.redis_io.set_state_body("unitree_g1_with_hands", state_body)
                    self.redis_io.set_state_hands_neck("unitree_g1_with_hands")
                    self.redis_io.set_root_pos("unitree_g1_with_hands", root_pos)
                    self.redis_io.set_t_state()
                    self.redis_io.flush()

                    # Get mimic obs from Redis
                    action_mimic, action_left_hand, action_right_hand, action_neck = self.redis_io.get_actions("unitree_g1_with_hands")
                    if action_mimic is None:
                        continue
                    # 输出action_mimic最后一个元素
                    # print(f"Action mimic last element: {action_mimic[-1]}")
                    # Construct observation for TWIST2 controller
                    obs_full, obs_hist, obs_buf = self.obs_builder.build_full_obs(
                        action_mimic=np.asarray(action_mimic),
                        obs_proprio=obs_proprio,
                        history_buf=self.proprio_history_buf,
                    )
                    self.proprio_history_buf.append(obs_full)

                    # Run policy
                    raw_action = self.policy_runner.infer_action(obs_buf)

                    if collect_metrics:
                        # Read motion phase for segmented metrics
                        phase_raw = None
                        try:
                            phase_raw = self.redis_io.get_motion_phase()
                        except Exception:
                            phase_raw = None
                        phase_raw = normalize_motion_phase(phase_raw)
                        if motion_phase_is_stand(phase_raw):
                            phase_bucket = "stand"
                        elif phase_raw == "motion":
                            phase_bucket = "motion"
                        else:
                            phase_bucket = None

                        root_vel_local = quat_rotate_inverse_np(quat, root_vel, scalar_first=True)
                        metrics_recorder.update(
                            action_mimic=np.asarray(action_mimic) if action_mimic is not None else None,
                            dof_pos=dof_pos,
                            rpy=rpy,
                            ang_vel=ang_vel,
                            root_pos=root_pos,
                            root_vel_local=root_vel_local,
                            raw_action=raw_action,
                            sim_torque=sim_torque,
                            dof_vel=dof_vel,
                            phase_bucket=phase_bucket,
                        )

                        # If motion done flag is set, finish gracefully.
                        try:
                            motion_done = self.redis_io.get_motion_done()
                            if is_truthy(motion_done):
                                print("[PolicyController] Motion done signal received")
                                return _format_result('motion_done')
                        except Exception:
                            pass
                        try:
                            policy_stop = self.redis_io.get_policy_stop()
                            if is_truthy(policy_stop):
                                print("[PolicyController] Policy stop signal received")
                                return _format_result('policy_stop')
                        except Exception:
                            pass

                    # Measure and track policy execution FPS
                    current_time = time.time()
                    if last_policy_time is not None:
                        policy_interval = current_time - last_policy_time
                        current_policy_fps = 1.0 / policy_interval

                        # For frequent printing (every 100 steps)
                        policy_execution_times.append(policy_interval)
                        policy_step_count += 1

                        # Print policy execution FPS every 100 steps
                        if policy_step_count % policy_fps_print_interval == 0:
                            recent_intervals = policy_execution_times[-policy_fps_print_interval:]
                            avg_interval = np.mean(recent_intervals)
                            avg_execution_fps = 1.0 / avg_interval
                            # print(f"Policy Execution FPS (last {policy_fps_print_interval} steps): {avg_execution_fps:.2f} Hz (avg interval: {avg_interval*1000:.2f}ms)")

                        # For detailed measurement (every 1000 steps)
                        if measure_fps:
                            fps_measurements.append(current_policy_fps)
                            fps_iteration_count += 1

                            if fps_iteration_count == fps_measurement_target:
                                avg_fps = np.mean(fps_measurements)
                                max_fps = np.max(fps_measurements)
                                min_fps = np.min(fps_measurements)
                                std_fps = np.std(fps_measurements)
                                print(f"\n=== Policy Execution FPS Results (steps {fps_iteration_count-fps_measurement_target+1}-{fps_iteration_count}) ===")
                                print(f"Average Policy FPS: {avg_fps:.2f}")
                                print(f"Max Policy FPS: {max_fps:.2f}")
                                print(f"Min Policy FPS: {min_fps:.2f}")
                                print(f"Std Policy FPS: {std_fps:.2f}")
                                print(f"Expected FPS (from decimation): {1.0/(self.sim_decimation * self.sim_dt):.2f}")
                                print(f"=================================================================================\n")
                                # Reset for next 1000 measurements
                                fps_measurements = []
                                fps_iteration_count = 0
                    last_policy_time = current_time

                    self.last_action = raw_action
                    scaled_actions = self.policy_runner.scale_action(raw_action)
                    pd_target = self.policy_runner.to_pd_target(scaled_actions)

                    # self.redis_client.set("action_low_level_unitree_g1", json.dumps(raw_action.tolist()))

                    # Update camera to follow pelvis
                    if self.viewer:
                        pelvis_pos = self.data.xpos[self.model.body("pelvis").id]
                        self.viewer.cam.lookat = pelvis_pos
                        self.viewer.sync()

                    if mp4_writer is not None:
                        img = self.viewer.read_pixels()
                        mp4_writer.append_data(img)

                    # Record proprio if enabled
                    if self.record_proprio:
                        proprio_data = {
                            'timestamp': time.time(),
                            'dof_pos': dof_pos.tolist(),
                            'dof_vel': dof_vel.tolist(),
                            'rpy': rpy.tolist(),
                            'ang_vel': ang_vel.tolist(),
                            'target_dof_pos': action_mimic.tolist()[-29:],
                        }
                        self.proprio_recordings.append(proprio_data)

                    # Check for fall (after delay to avoid initialization false positives)
                    if (time.time() - start_time) > fall_detection_delay:
                        if self.check_fall_from_redis():
                            print("[PolicyController] Robot fell detected")
                            return _format_result('fell')


                # PD control
                torque = (pd_target - dof_pos) * self.stiffness - dof_vel * self.damping
                torque = np.clip(torque, -self.torque_limits, self.torque_limits)

                self.data.ctrl[:] = torque
                mujoco.mj_step(self.model, self.data)

                # Sleep to maintain real-time pace
                if self.limit_fps:
                    elapsed = time.time() - t_start
                    if elapsed < self.sim_dt:
                        time.sleep(self.sim_dt - elapsed)

            print("[PolicyController] Simulation completed normally")
            return _format_result('completed')

        except Exception as e:
            print(f"Error in run: {e}")
            import traceback
            traceback.print_exc()
            return _format_result('error')
        finally:
            if mp4_writer is not None:
                mp4_writer.close()
                print("Video saved as twist2_simulation.mp4")

            # Save proprio recordings if enabled
            if self.record_proprio and self.proprio_recordings:
                import pickle
                with open('twist2_proprio_recordings.pkl', 'wb') as f:
                    pickle.dump(self.proprio_recordings, f)
                print("Proprioceptive recordings saved as twist2_proprio_recordings.pkl")

            if self.viewer:
                self.viewer.close()
            print("Simulation finished.")


def main():
    parser = argparse.ArgumentParser(description='Run TWIST2 policy in simulation')
    parser.add_argument('--xml', type=str, default='../assets/g1/g1_sim2sim.xml',
                        help='Path to MuJoCo XML file')
    parser.add_argument('--policy', type=str, required=True,
                        help='Path to TWIST2 ONNX policy file')
    parser.add_argument('--device', type=str, 
                        default='cuda',
                        help='Device to run policy on (cuda/cpu)')
    parser.add_argument('--record_video', action='store_true',
                        help='Record video of simulation')
    parser.add_argument('--record_proprio', action='store_true',
                        help='Record proprioceptive data')
    parser.add_argument("--measure_fps", help="Measure FPS", default=0, type=int)
    parser.add_argument("--limit_fps", help="Limit FPS with sleep", default=1, type=int)
    parser.add_argument("--policy_frequency", help="Policy frequency", default=100, type=int)
    args = parser.parse_args()
    
    # Verify policy file exists
    if not os.path.exists(args.policy):
        print(f"Error: Policy file {args.policy} does not exist")
        return
    
    # Verify XML file exists
    if not os.path.exists(args.xml):
        print(f"Error: XML file {args.xml} does not exist")
        return
    
    print(f"Starting TWIST2 simulation controller...")
    print(f"  XML file: {args.xml}")
    print(f"  Policy file: {args.policy}")
    print(f"  Device: {args.device}")
    print(f"  Record video: {args.record_video}")
    print(f"  Record proprio: {args.record_proprio}")
    print(f"  Measure FPS: {args.measure_fps}")
    print(f"  Limit FPS: {args.limit_fps}")
    controller = RealTimePolicyController(
        xml_file=args.xml,
        policy_path=args.policy,
        device=args.device,
        record_video=args.record_video,
        record_proprio=args.record_proprio,
        measure_fps=args.measure_fps,
        limit_fps=args.limit_fps,
        policy_frequency=args.policy_frequency,
        show_viewer=True,
    )
    controller.run()


if __name__ == "__main__":
    main()
