#!/usr/bin/env python3
"""
Automated ONNX Model Evaluation with Motion Tracking
- Uses direct class calls instead of subprocess
- Loads .pkl motion file
- Evaluates multiple .onnx models sequentially
- Records metrics similar to offline_eval.py
"""

import argparse
import os
import sys
import json
import time
import multiprocessing as mp
from pathlib import Path
import numpy as np
from termcolor import cprint
from collections import defaultdict
import pickle
from tqdm import tqdm
from deploy_real.redis_io import RedisIO


# Set multiprocessing start method early (must be before creating any processes)
mp.set_start_method('spawn', force=True)

# Add parent directories to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir / "deploy_real"))
sys.path.append(str(current_dir / "legged_gym" / "legged_gym"))

from deploy_real.server_motion_lib_cjm import MotionServer
from deploy_real.server_low_level_g1_sim_cjm import RealTimePolicyController
from pose.utils.motion_lib_pkl import MotionLib
import torch

def find_onnx_models(onnx_dir, reverse=False):
    """Find all .onnx models and sort by iteration number."""
    import glob
    onnx_files = glob.glob(os.path.join(onnx_dir, "*.onnx"))
    if not onnx_files:
        cprint(f"No .onnx files found in {onnx_dir}", "red")
        return []

    # Extract iteration numbers and sort
    def extract_iter(filename):
        basename = os.path.basename(filename)
        # Try pattern: model_12345.onnx
        parts = basename.replace('.onnx', '').split('_')
        if len(parts) >= 2 and parts[-1].isdigit():
            return int(parts[-1])
        return 0

    onnx_files.sort(key=extract_iter, reverse=reverse)
    return onnx_files


def get_motion_duration(motion_file):
    """Get duration of motion from pkl file."""
    try:
        with open(motion_file, 'rb') as f:
            motion_data = pickle.load(f)
            # Assuming motion data has frames and each frame is 20ms (50Hz)
            num_frames = len(motion_data)
            print(f"num_frames: {num_frames:.2f} motion_file {motion_file}")
            duration = num_frames * 0.02  # 20ms per frame
            return duration
    except Exception as e:
        cprint(f"Warning: Could not read motion duration from {motion_file}: {e}", "yellow")
        return 30.0  # Default 30 seconds


def _motion_server_wrapper(queue, args):
    """Wrapper for motion server subprocess."""
    motion_file, redis_ip, play_standing_after_motion, motion_viewer = args
    motion_server = MotionServer(
        motion_file=motion_file,
        robot="unitree_g1_with_hands",
        redis_ip=redis_ip,
        steps="1",
        use_remote_control=False,
        send_start_frame_as_end_frame=False,
        show_viewer=motion_viewer,
        play_standing_after_motion=play_standing_after_motion,
    )
    result = motion_server.run()
    queue.put(('motion', result))


def _policy_controller_wrapper(queue, args):
    """Wrapper for policy controller subprocess."""
    onnx_file, timeout, policy_viewer = args
    policy_controller = RealTimePolicyController(
        xml_file="assets/g1/g1_sim2sim_29dof.xml",
        policy_path=onnx_file,
        device="cpu",
        record_video=False,
        record_proprio=False,
        measure_fps=False,
        limit_fps=True,
        policy_frequency=100,
        show_viewer=policy_viewer
    )
    # Wait until motion server starts streaming (t_state appears).
    try:
        redis_io = RedisIO.connect(host='localhost', port=6379, db=0)
        start_wait = time.time()
        logged = False
        while True:
            t_state = redis_io.get_t_state()
            if t_state:
                if not logged:
                    cprint(f"[PolicyWait] t_state detected: {t_state}", "yellow")
                    logged = True
                break
            if time.time() - start_wait > 10.0:
                cprint("Warning: t_state not found within 10s, starting policy anyway.", "yellow")
                break
            time.sleep(0.05)
    except Exception as e:
        cprint(f"Warning: failed to wait for t_state: {e}", "yellow")
    result = policy_controller.run(timeout=timeout, collect_metrics=True)
    queue.put(('policy', result))


def run_single_experiment(motion_file, motion_length, onnx_file, redis_ip="localhost", exp_idx=0, num_runs=5, motion_viewer=False, policy_viewer=False):
    """
    Run a single experiment with specified ONNX model using multiprocessing.
    Returns metrics dict if experiment completed successfully, None otherwise.
    """
    cprint(f"\n{'='*70}", "cyan")
    cprint(f"Experiment {exp_idx+1}: {os.path.basename(onnx_file)}", "cyan")
    cprint(f"{'='*70}", "cyan")

    # Get motion duration from pkl file
    
    play_standing_after_motion = True
    #设置延长的时间 10s (pre 5s + post 5s) 或者 0s
    extend_time = 10.0 if play_standing_after_motion else 0.0
    motion_duration = float(motion_length) + extend_time
    cprint(f"Motion duration: {motion_duration:.2f} seconds", "cyan")

    results = []

    try:
        # Run experiment num_runs times using multiprocessing
        for run in range(num_runs):
            cprint(f"\n  Run {run+1}/{num_runs}...", "yellow")

            # Use a large timeout to avoid wall-clock affecting evaluation quality.
            timeout = motion_duration * 5.0 + 30.0
            timed_out = False

            # Clear stale heartbeat/flags so policy doesn't see old state.
            redis_io = None
            try:
                redis_io = RedisIO.connect(host=redis_ip, port=6379, db=0)
                redis_io.clear_flags()
            except Exception as e:
                cprint(f"Warning: failed to clear redis flags before run: {e}", "yellow")

            # Start motion server subprocess
            motion_queue = mp.Queue()
            motion_process = mp.Process(
                target=_motion_server_wrapper,
                args=(motion_queue, (motion_file, redis_ip, play_standing_after_motion, motion_viewer))
            )
            motion_process.start()
            
            time.sleep(2.0)
            
            # Start policy controller subprocess
            policy_queue = mp.Queue()
            policy_process = mp.Process(
                target=_policy_controller_wrapper,
                args=(policy_queue, (onnx_file, timeout, policy_viewer))
            )
            policy_process.start()

            # Monitor processes
            start_time = time.time()
            completed = False
            fell = False
            policy_stopped_early = False
            policy_stop_logged = False
            motion_done_flag = False

            motion_completed = False
            motion_completed_time = None
            while time.time() - start_time < timeout:
                # Check if motion process finished
                if not motion_process.is_alive():
                    completed = True
                    cprint(f"    Motion server completed", "green")
                    try:
                        if redis_io is not None:
                            redis_io.set_policy_stop(True)
                    except Exception:
                        pass
                    motion_completed = True
                    motion_completed_time = time.time()
                    break

                # Check if policy process finished
                if not policy_process.is_alive():
                    policy_stopped_early = True
                    if not policy_stop_logged:
                        cprint(f"    Policy controller stopped (waiting for motion)...", "yellow")
                        policy_stop_logged = True
                    try:
                        if redis_io is not None:
                            motion_done = redis_io.get_motion_done()
                            if motion_done and motion_done in (b"1", b"true", b"True"):
                                motion_done_flag = True
                    except Exception:
                        pass
                    # If policy stops and motion is not done, end motion to match spec.
                    if not motion_done_flag:
                        break

                time.sleep(0.1)
            else:
                timed_out = True

            # If motion completed, give policy a short grace to report metrics.
            if motion_completed and policy_process.is_alive():
                grace_start = time.time()
                grace_seconds = 5.0
                while time.time() - grace_start < grace_seconds:
                    if not policy_process.is_alive():
                        break
                    time.sleep(0.05)

            # Terminate processes if still running
            motion_timeout = False
            policy_timeout = False
            policy_killed_after_motion = False
            if motion_process.is_alive():
                if timed_out:
                    cprint(f"    Terminating motion server (timeout)...", "yellow")
                    motion_timeout = True
                    motion_process.terminate()
                    motion_process.join(timeout=2)
                    if motion_process.is_alive():
                        motion_process.kill()
                        motion_process.join(timeout=2)
                elif policy_stopped_early and not motion_done_flag:
                    cprint(f"    Terminating motion server (policy stopped)...", "yellow")
                    motion_timeout = True
                    motion_process.terminate()
                    motion_process.join(timeout=2)
                    if motion_process.is_alive():
                        motion_process.kill()
                        motion_process.join(timeout=2)
            if policy_process.is_alive():
                if timed_out:
                    cprint(f"    Terminating policy controller (timeout)...", "yellow")
                    policy_timeout = True
                    policy_process.terminate()
                    policy_process.join(timeout=2)
                    if policy_process.is_alive():
                        policy_process.kill()
                        policy_process.join(timeout=2)
                elif completed:
                    # Wait up to 5s for policy to exit after motion completion.
                    grace_start = time.time()
                    grace_seconds = 5.0
                    while time.time() - grace_start < grace_seconds:
                        if not policy_process.is_alive():
                            break
                        time.sleep(0.05)
                    if policy_process.is_alive():
                        cprint(f"    Terminating policy controller (after 5s grace)...", "yellow")
                        policy_killed_after_motion = True
                        policy_process.terminate()
                        policy_process.join(timeout=2)
                        if policy_process.is_alive():
                            policy_process.kill()
                            policy_process.join(timeout=2)

            # Get results from queues (if any)
            motion_result = None
            policy_result = None
            try:
                motion_result = motion_queue.get(timeout=0.1)
            except:
                pass
            try:
                policy_result = policy_queue.get(timeout=0.1)
            except:
                pass
            cprint(f"    Motion exitcode: {motion_process.exitcode}", "cyan")
            cprint(f"    Policy exitcode: {policy_process.exitcode}", "cyan")

            # Parse policy and motion results
            policy_status = None
            policy_metrics = None
            policy_steps = 0
            policy_elapsed_time = None
            policy_sim_time = None
            policy_fell_time = None
            if policy_result:
                policy_payload = policy_result[1]
                if isinstance(policy_payload, dict):
                    policy_status = policy_payload.get("status")
                    policy_metrics = {
                        "total": policy_payload.get("metrics_total"),
                        "stand": policy_payload.get("metrics_stand"),
                        "motion": policy_payload.get("metrics_motion"),
                    }
                    policy_steps = int(policy_payload.get("steps", 0))
                    policy_elapsed_time = policy_payload.get("elapsed_time")
                    policy_sim_time = policy_payload.get("sim_time")
                    policy_fell_time = policy_payload.get("fell_time")
                else:
                    policy_status = policy_payload
            else:
                if policy_timeout:
                    policy_status = "killed_timeout"
                elif policy_killed_after_motion:
                    policy_status = "killed_after_motion"
                else:
                    exitcode = policy_process.exitcode
                    if exitcode is None:
                        policy_status = "no_result"
                    elif exitcode == 0:
                        policy_status = "exited_no_result"
                    else:
                        policy_status = f"exitcode_{exitcode}"

            motion_status = None
            if motion_result:
                motion_payload = motion_result[1]
                if isinstance(motion_payload, str):
                    motion_status = motion_payload
                elif isinstance(motion_payload, bool):
                    motion_status = "completed" if motion_payload else "stopped"
                else:
                    motion_status = "unknown"
            else:
                if motion_timeout:
                    motion_status = "killed_timeout"
                else:
                    exitcode = motion_process.exitcode
                    if exitcode is None:
                        motion_status = "no_result"
                    elif exitcode == 0:
                        motion_status = "exited_no_result"
                    else:
                        motion_status = f"exitcode_{exitcode}"

            # Check for fall based on policy result
            if policy_status == 'fell':
                fell = True
                cprint(f"    Robot fell detected", "red")

            run_duration = time.time() - start_time
            completion_ratio = run_duration / motion_duration if motion_duration > 0 else 0.0
            if motion_status == "completed":
                completion_ratio = 1.0
            elif completion_ratio > 1.0:
                completion_ratio = 1.0

            # Success is primarily defined by motion completion and no fall.
            completed = (
                motion_status == "completed"
                and not fell
                and completion_ratio >= 0.98
            )

            failure_reason = None
            incomplete_details = []
            if not completed:
                if policy_status in ("fell", "timeout", "motion_server_stopped", "stopped", "error"):
                    failure_reason = policy_status
                elif motion_status and motion_status != "completed":
                    failure_reason = f"motion_{motion_status}"
                else:
                    failure_reason = "incomplete"

                if policy_status != "completed":
                    incomplete_details.append(f"policy_status={policy_status}")
                if motion_status != "completed":
                    incomplete_details.append(f"motion_status={motion_status}")
                if completion_ratio < 0.98:
                    incomplete_details.append(f"completion_ratio={completion_ratio:.3f}<0.98")

            result = {
                'run': run + 1,
                'completed': completed,
                'fell': fell,
                'duration': run_duration,
                'completion_ratio': completion_ratio,
                'motion_file': motion_file,
                'motion_duration': motion_duration,
                'policy_status': policy_status,
                'motion_status': motion_status,
                'failure_reason': failure_reason,
                'incomplete_details': incomplete_details,
                'policy_steps': policy_steps,
                'policy_elapsed_time': policy_elapsed_time,
                'policy_sim_time': policy_sim_time,
                'policy_fell_time': policy_fell_time,
                'policy_metrics': policy_metrics
            }
            results.append(result)

            if completed:
                cprint(f"    Run {run+1}: Completed ✓", "green")
            else:
                reason = failure_reason or "failed"
                details = ""
                if incomplete_details:
                    details = f" | {', '.join(incomplete_details)}"
                cprint(f"    Run {run+1}: Failed ({reason}){details}", "red")

            # Per-run detailed summary
            if policy_elapsed_time is not None:
                cprint(f"    Elapsed: {policy_elapsed_time:.2f}s | SimTime: {policy_sim_time:.2f}s | Steps: {policy_steps}", "white")
            if fell and policy_fell_time is not None:
                cprint(f"    Fell at: {policy_fell_time:.2f}s", "red")
            pm_all = policy_metrics or {}
            if pm_all:
                def _print_metrics_block(title, pm):
                    if not pm:
                        return
                    cprint(f"    Metrics ({title}):", "white")
                    if pm.get("tracking_dof_abs_mean") is not None:
                        cprint(f"      tracking_dof_abs_mean: {pm['tracking_dof_abs_mean']:.4f}", "white")
                    if pm.get("tracking_dof_abs_std") is not None:
                        cprint(f"      tracking_dof_abs_std:  {pm['tracking_dof_abs_std']:.4f}", "white")
                    if pm.get("root_pos_z_abs_mean") is not None:
                        cprint(f"      root_pos_z_abs_mean:   {pm['root_pos_z_abs_mean']:.4f}", "white")
                    if pm.get("root_vel_xy_l2_mean") is not None:
                        cprint(f"      root_vel_xy_l2_mean:   {pm['root_vel_xy_l2_mean']:.4f}", "white")
                    if pm.get("roll_pitch_abs_mean") is not None:
                        cprint(f"      roll_pitch_abs_mean:   {pm['roll_pitch_abs_mean']:.4f}", "white")
                    if pm.get("yaw_ang_vel_abs_mean") is not None:
                        cprint(f"      yaw_ang_vel_abs_mean:  {pm['yaw_ang_vel_abs_mean']:.4f}", "white")
                    if pm.get("action_delta_abs_mean") is not None:
                        cprint(f"      action_delta_abs_mean: {pm['action_delta_abs_mean']:.4f}", "white")
                    if pm.get("torque_abs_mean") is not None:
                        cprint(f"      torque_abs_mean:       {pm['torque_abs_mean']:.4f}", "white")
                    if pm.get("power_abs_mean") is not None:
                        cprint(f"      power_abs_mean:        {pm['power_abs_mean']:.4f}", "white")
                    if pm.get("root_height_std") is not None:
                        cprint(f"      root_height_std:       {pm['root_height_std']:.4f}", "white")

                _print_metrics_block("total", pm_all.get("total"))
                _print_metrics_block("stand", pm_all.get("stand"))
                _print_metrics_block("motion", pm_all.get("motion"))

            time.sleep(1.0)  # Brief pause between runs

        # Compute aggregate metrics for this model
        completed_runs = [r for r in results if r['completed']]
        fell_runs = [r for r in results if r['fell']]
        success_rate = len([r for r in results if r['completed'] and not r['fell']]) / num_runs
        fall_rate = len(fell_runs) / num_runs
        completion_rate = len(completed_runs) / num_runs
        avg_completion_ratio = float(np.mean([r['completion_ratio'] for r in results])) if results else 0.0

        # Aggregate policy metrics (weighted by steps)
        def _weighted_mean_metric(metric_key, phase_key="total"):
            total = 0.0
            weight = 0
            for r in results:
                m = (r.get("policy_metrics") or {}).get(phase_key) or {}
                steps = m.get("steps", 0)
                if metric_key in m and steps > 0:
                    total += float(m[metric_key]) * steps
                    weight += steps
            return (total / weight) if weight > 0 else None

        def _mean_metric(metric_key, phase_key="total"):
            vals = []
            for r in results:
                m = (r.get("policy_metrics") or {}).get(phase_key) or {}
                if metric_key in m:
                    vals.append(float(m[metric_key]))
            return float(np.mean(vals)) if vals else None

        model_metrics = {
            'model_name': os.path.basename(onnx_file),
            'num_runs': num_runs,
            'success_rate': success_rate,
            'fall_rate': fall_rate,
            'completion_rate': completion_rate,
            'avg_completion_ratio': avg_completion_ratio,
            'num_successful': len([r for r in results if r['completed'] and not r['fell']]),
            'num_fell': len(fell_runs),
            'num_completed': len(completed_runs),
            'avg_duration': float(np.mean([r['duration'] for r in results])),
            'failure_reasons': dict((k, len([r for r in results if r['failure_reason'] == k]))
                                    for k in sorted(set(r['failure_reason'] for r in results if r['failure_reason']))),
            'policy_metrics_weighted': {
                'total': {
                    'tracking_dof_abs_mean': _weighted_mean_metric('tracking_dof_abs_mean', 'total'),
                    'root_pos_z_abs_mean': _weighted_mean_metric('root_pos_z_abs_mean', 'total'),
                    'root_vel_xy_l2_mean': _weighted_mean_metric('root_vel_xy_l2_mean', 'total'),
                    'roll_pitch_abs_mean': _weighted_mean_metric('roll_pitch_abs_mean', 'total'),
                    'yaw_ang_vel_abs_mean': _weighted_mean_metric('yaw_ang_vel_abs_mean', 'total'),
                    'action_delta_abs_mean': _weighted_mean_metric('action_delta_abs_mean', 'total'),
                    'torque_abs_mean': _weighted_mean_metric('torque_abs_mean', 'total'),
                    'power_abs_mean': _weighted_mean_metric('power_abs_mean', 'total'),
                    'root_height_std': _weighted_mean_metric('root_height_std', 'total'),
                    'tracking_dof_abs_std': _weighted_mean_metric('tracking_dof_abs_std', 'total'),
                    'mimic_obs_mismatch_count_mean': _mean_metric('mimic_obs_mismatch_count', 'total'),
                },
                'stand': {
                    'tracking_dof_abs_mean': _weighted_mean_metric('tracking_dof_abs_mean', 'stand'),
                    'root_pos_z_abs_mean': _weighted_mean_metric('root_pos_z_abs_mean', 'stand'),
                    'root_vel_xy_l2_mean': _weighted_mean_metric('root_vel_xy_l2_mean', 'stand'),
                    'roll_pitch_abs_mean': _weighted_mean_metric('roll_pitch_abs_mean', 'stand'),
                    'yaw_ang_vel_abs_mean': _weighted_mean_metric('yaw_ang_vel_abs_mean', 'stand'),
                    'action_delta_abs_mean': _weighted_mean_metric('action_delta_abs_mean', 'stand'),
                    'torque_abs_mean': _weighted_mean_metric('torque_abs_mean', 'stand'),
                    'power_abs_mean': _weighted_mean_metric('power_abs_mean', 'stand'),
                    'root_height_std': _weighted_mean_metric('root_height_std', 'stand'),
                    'tracking_dof_abs_std': _weighted_mean_metric('tracking_dof_abs_std', 'stand'),
                    'mimic_obs_mismatch_count_mean': _mean_metric('mimic_obs_mismatch_count', 'stand'),
                },
                'motion': {
                    'tracking_dof_abs_mean': _weighted_mean_metric('tracking_dof_abs_mean', 'motion'),
                    'root_pos_z_abs_mean': _weighted_mean_metric('root_pos_z_abs_mean', 'motion'),
                    'root_vel_xy_l2_mean': _weighted_mean_metric('root_vel_xy_l2_mean', 'motion'),
                    'roll_pitch_abs_mean': _weighted_mean_metric('roll_pitch_abs_mean', 'motion'),
                    'yaw_ang_vel_abs_mean': _weighted_mean_metric('yaw_ang_vel_abs_mean', 'motion'),
                    'action_delta_abs_mean': _weighted_mean_metric('action_delta_abs_mean', 'motion'),
                    'torque_abs_mean': _weighted_mean_metric('torque_abs_mean', 'motion'),
                    'power_abs_mean': _weighted_mean_metric('power_abs_mean', 'motion'),
                    'root_height_std': _weighted_mean_metric('root_height_std', 'motion'),
                    'tracking_dof_abs_std': _weighted_mean_metric('tracking_dof_abs_std', 'motion'),
                    'mimic_obs_mismatch_count_mean': _mean_metric('mimic_obs_mismatch_count', 'motion'),
                },
            },
            'runs': results
        }

        return model_metrics

    except Exception as e:
        cprint(f"Error during experiment: {e}", "red")
        import traceback
        traceback.print_exc()
        return None


def evaluate_all_models(motion_file, onnx_dir, redis_ip="localhost", num_runs=5, output_dir=None, reverse=False, motion_viewer=False, policy_viewer=False):
    """Evaluate all ONNX models in the specified directory."""
    onnx_files = find_onnx_models(onnx_dir, reverse=reverse)

    if not onnx_files:
        cprint("No ONNX models found. Exiting.", "red")
        return

    cprint("="*70, "cyan")
    cprint("AUTOMATED ONNX MODEL EVALUATION", "cyan")
    cprint("="*70, "cyan")
    cprint(f"Motion file:   {motion_file}", "green")
    cprint(f"ONNX directory: {onnx_dir}", "green")
    cprint(f"Found {len(onnx_files)} models", "green")
    cprint(f"Runs per model: {num_runs}", "green")
    cprint("="*70, "cyan")

    all_results = []
    
    motion_lib = MotionLib(motion_file, device="cpu")
    motion_id = torch.tensor([0], device="cpu", dtype=torch.long)
    motion_length = motion_lib.get_motion_length(motion_id)

    for exp_idx, onnx_file in enumerate(onnx_files):
        result = run_single_experiment(
            motion_file, motion_length, onnx_file, redis_ip, exp_idx, num_runs, motion_viewer, policy_viewer
        )

        if result:
            all_results.append(result)
            print_result_summary(result)

        time.sleep(2.0)  # Brief pause between models

    # Generate summary report
    if all_results:
        generate_summary(all_results, output_dir, onnx_dir)


def print_result_summary(result):
    """Print summary for a single model result."""
    cprint(f"\n[Summary: {result['model_name']}]", "yellow")
    cprint(f"  Success Rate:     {result['success_rate']*100:.1f}%", "green")
    cprint(f"  Fall Rate:        {result['fall_rate']*100:.1f}%", "red" if result['fall_rate'] > 0 else "green")
    cprint(f"  Completion Rate:  {result['completion_rate']*100:.1f}%", "cyan")
    cprint(f"  Avg Completion:   {result['avg_completion_ratio']*100:.1f}%", "cyan")
    cprint(f"  Avg Duration:      {result['avg_duration']:.2f}s", "white")
    cprint(f"  Runs: {result['num_successful']}/{result['num_runs']} successful, {result['num_fell']} fell", "white")
    if result.get("failure_reasons"):
        cprint(f"  Fail Reasons:      {result['failure_reasons']}", "white")
    pm = result.get("policy_metrics_weighted") or {}
    pm_total = pm.get("total") or {}
    if pm_total.get("tracking_dof_abs_mean") is not None:
        cprint(f"  Track DOF Err:     {pm_total['tracking_dof_abs_mean']:.4f}", "white")
    if pm_total.get("root_vel_xy_l2_mean") is not None:
        cprint(f"  Root Vel Err:      {pm_total['root_vel_xy_l2_mean']:.4f}", "white")
    if pm_total.get("roll_pitch_abs_mean") is not None:
        cprint(f"  Roll/Pitch Err:    {pm_total['roll_pitch_abs_mean']:.4f}", "white")


def generate_summary(all_results, output_dir, onnx_dir=None):
    """Generate comprehensive summary report."""
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "onnx_evaluation_results")

    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    onnx_dir_name = os.path.basename(os.path.normpath(onnx_dir)) if onnx_dir else "onnx"
    run_dir = os.path.join(output_dir, f"{timestamp}-{onnx_dir_name}")
    os.makedirs(run_dir, exist_ok=True)

    # Save detailed JSON results
    results_path = os.path.join(run_dir, "evaluation_results.json")
    results_path_ts = os.path.join(run_dir, f"evaluation_results_{timestamp}.json")
    with open(results_path, 'w') as f:
        json.dump({
            'motion_file': all_results[0]['runs'][0].get('motion_file', 'unknown') if all_results and all_results[0].get('runs') else 'unknown',
            'total_models': len(all_results),
            'results': all_results
        }, f, indent=2)
    with open(results_path_ts, 'w') as f:
        json.dump({
            'motion_file': all_results[0]['runs'][0].get('motion_file', 'unknown') if all_results and all_results[0].get('runs') else 'unknown',
            'total_models': len(all_results),
            'results': all_results
        }, f, indent=2)

    cprint(f"\nDetailed results saved to: {results_path}", "green")
    cprint(f"Timestamped results saved to: {results_path_ts}", "green")

    # Print overall summary
    cprint(f"\n{'='*70}", "cyan")
    cprint("OVERALL SUMMARY", "cyan")
    cprint(f"{'='*70}", "cyan")

    for result in all_results:
        cprint(f"\n{result['model_name']}", "yellow")
        cprint(f"  Success Rate:     {result['success_rate']*100:.1f}%",
                 "green" if result['success_rate'] > 0.8 else "yellow")
        cprint(f"  Fall Rate:        {result['fall_rate']*100:.1f}%",
                 "red" if result['fall_rate'] > 0.2 else "green")
        cprint(f"  Completion Rate:  {result['completion_rate']*100:.1f}%", "cyan")
        cprint(f"  Avg Completion:   {result['avg_completion_ratio']*100:.1f}%", "cyan")

    # Best model
    def _best_key(result):
        tracking = result.get('policy_metrics_weighted', {}).get('total', {}).get('tracking_dof_abs_mean')
        tracking_score = -tracking if tracking is not None else float('-inf')
        return (result['success_rate'], tracking_score)

    best_success = max(all_results, key=_best_key)
    cprint(f"\n{'='*70}", "cyan")
    cprint("BEST MODEL", "cyan")
    cprint(f"{'='*70}", "cyan")
    cprint(f"Model: {best_success['model_name']}", "green")
    cprint(f"Success Rate: {best_success['success_rate']*100:.1f}%", "green")
    cprint(f"Fall Rate:    {best_success['fall_rate']*100:.1f}%", "green")

    cprint(f"\n{'='*70}\n", "cyan")

    # Write summary text file
    summary_path = os.path.join(run_dir, "result.txt")
    with open(summary_path, "w") as f:
        f.write("OVERALL SUMMARY\n")
        f.write("=" * 70 + "\n")
        for result in all_results:
            pm = (result.get("policy_metrics_weighted") or {}).get("total") or {}
            f.write(f"\n{result['model_name']}\n")
            f.write(f"  Success Rate:     {result['success_rate']*100:.1f}%\n")
            f.write(f"  Fall Rate:        {result['fall_rate']*100:.1f}%\n")
            f.write(f"  Completion Rate:  {result['completion_rate']*100:.1f}%\n")
            f.write(f"  Avg Completion:   {result['avg_completion_ratio']*100:.1f}%\n")
            # 输出参与最优评判的关键指标
            f.write(f"  Track DOF Err:    {pm.get('tracking_dof_abs_mean')}\n")
            f.write(f"  Root Vel Err:     {pm.get('root_vel_xy_l2_mean')}\n")
            f.write(f"  Roll/Pitch Err:   {pm.get('roll_pitch_abs_mean')}\n")
            f.write(f"  Yaw Ang Err:      {pm.get('yaw_ang_vel_abs_mean')}\n")
            f.write(f"  Torque Mean:      {pm.get('torque_abs_mean')}\n")
            f.write(f"  Power Mean:       {pm.get('power_abs_mean')}\n")
            f.write(f"  Root Height Std:  {pm.get('root_height_std')}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("BEST MODEL (by success_rate, then lowest tracking_dof_abs_mean)\n")
        f.write("=" * 70 + "\n")
        f.write(f"Model: {best_success['model_name']}\n")
        f.write(f"Success Rate: {best_success['success_rate']*100:.1f}%\n")
        f.write(f"Fall Rate:    {best_success['fall_rate']*100:.1f}%\n")
        pm_best = (best_success.get("policy_metrics_weighted") or {}).get("total") or {}
        f.write(f"Track DOF Err: {pm_best.get('tracking_dof_abs_mean')}\n")
        f.write(f"Root Vel Err:  {pm_best.get('root_vel_xy_l2_mean')}\n")
        f.write(f"Roll/Pitch Err:{pm_best.get('roll_pitch_abs_mean')}\n")
        f.write(f"Yaw Ang Err:   {pm_best.get('yaw_ang_vel_abs_mean')}\n")
        f.write(f"Torque Mean:   {pm_best.get('torque_abs_mean')}\n")
        f.write(f"Power Mean:    {pm_best.get('power_abs_mean')}\n")
        f.write(f"Root Height Std:{pm_best.get('root_height_std')}\n")
    cprint(f"Summary saved to: {summary_path}", "green")


def main():
    parser = argparse.ArgumentParser(description='Automated ONNX Model Evaluation with Motion Tracking')
    parser.add_argument('--motion_file', type=str, required=True,
                        help='Path to .pkl motion file')
    parser.add_argument('--onnx_dir', type=str, required=True,
                        help='Path to directory containing .onnx models')
    parser.add_argument('--redis_ip', type=str, default='localhost',
                        help='Redis IP address')
    parser.add_argument('--num_runs', type=int, default=5,
                        help='Number of runs per model (default: 5)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for results')
    parser.add_argument('--reverse', action='store_true',
                        help='Evaluate models in reverse order (descending)')
    parser.add_argument('--motion_viewer', action='store_true',
                        help='Enable MotionServer viewer')
    parser.add_argument('--policy_viewer', action='store_true',
                        help='Enable PolicyController viewer')

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.motion_file):
        cprint(f"Error: Motion file not found: {args.motion_file}", "red")
        return

    if not os.path.isdir(args.onnx_dir):
        cprint(f"Error: ONNX directory not found: {args.onnx_dir}", "red")
        return

    # Run evaluation
    evaluate_all_models(
        motion_file=args.motion_file,
        onnx_dir=args.onnx_dir,
        redis_ip=args.redis_ip,
        num_runs=args.num_runs,
        output_dir=args.output_dir,
        reverse=args.reverse,
        motion_viewer=args.motion_viewer,
        policy_viewer=args.policy_viewer
    )


if __name__ == "__main__":
    main()
