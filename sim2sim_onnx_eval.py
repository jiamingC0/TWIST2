#!/usr/bin/env python3
"""
Automated ONNX Model Evaluation with Motion Tracking
- Loads .pkl motion file
- Evaluates multiple .onnx models sequentially
- Records metrics similar to offline_eval.py
"""

import argparse
import os
import sys
import json
import time
import signal
import subprocess
import glob
from pathlib import Path
import numpy as np
from termcolor import cprint
from collections import defaultdict
import redis
import pickle

# Add parent directories to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir / "deploy_real"))
sys.path.append(str(current_dir / "legged_gym" / "legged_gym"))


def find_onnx_models(onnx_dir, reverse=False):
    """Find all .onnx models and sort by iteration number."""
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
            duration = num_frames * 0.02  # 20ms per frame
            return duration
    except Exception as e:
        cprint(f"Warning: Could not read motion duration from {motion_file}: {e}", "yellow")
        return 30.0  # Default 30 seconds


def check_fall_from_redis(redis_client):
    """Check for fall signal from Redis by reading robot state."""
    try:
        # Read robot state from Redis
        state_body_json = redis_client.get("state_body_unitree_g1_with_hands")
        if not state_body_json:
            return False
        state_body = json.loads(state_body_json)
        # state_body format: [ang_vel (3), roll_pitch (2), dof_pos (29)] = 34 dims
        # ang_vel indices: 0-2, roll_pitch indices: 3-4
        roll = state_body[3]
        pitch = state_body[4]

        # Check orientation fall: roll or pitch exceeds 60 degrees (1.05 radians)
        orientation_fall = abs(roll) > 1.05 or abs(pitch) > 1.05

        # Check height fall: root position z < 0.4m
        height_fall = False
        root_pos_json = redis_client.get("root_pos_unitree_g1_with_hands")
        if root_pos_json:
            root_pos = json.loads(root_pos_json)
            root_height = root_pos[2]  # z-axis
            height_fall = root_height < 0.4  # 40cm threshold
        if not height_fall and orientation_fall:
            cprint(f"    Fall detected: {roll:.2f}, {pitch:.2f} root_height: {root_height:.2f}", "red")
            
        return height_fall
    except Exception as e:
        pass
    return False


def run_single_experiment(motion_file, onnx_file, redis_ip="localhost", exp_idx=0, num_runs=5):
    """
    Run a single experiment with specified ONNX model.
    Returns metrics dict if experiment completed successfully, None otherwise.
    """
    cprint(f"\n{'='*70}", "cyan")
    cprint(f"Experiment {exp_idx+1}: {os.path.basename(onnx_file)}", "cyan")
    cprint(f"{'='*70}", "cyan")

    # Get motion duration from pkl file
    motion_duration = get_motion_duration(motion_file)
    cprint(f"Motion duration: {motion_duration:.2f} seconds", "cyan")
    timeout = motion_duration + 5.0  # Add 5s buffer

    # Start motion server and policy controller
    deploy_real_dir = Path(__file__).parent / "deploy_real"

    motion_cmd = [
        "python", str(deploy_real_dir / "server_motion_lib.py"),
        "--motion_file", motion_file,
        "--robot", "unitree_g1_with_hands",
        "--redis_ip", redis_ip
    ]

    policy_cmd = [
        "python", str(deploy_real_dir / "server_low_level_g1_sim.py"),
        "--xml", "assets/g1/g1_sim2sim_29dof.xml",
        "--policy", onnx_file,
        "--device", "cuda",
        "--measure_fps", "0",
        "--limit_fps", "1",
        "--policy_frequency", "100"
    ]

    motion_process = None
    policy_process = None
    redis_client = None
    results = []

    try:
        # Connect to Redis for fall detection
        redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

        # Run experiment num_runs times
        for run in range(num_runs):
            cprint(f"\n  Run {run+1}/{num_runs}...", "yellow")

            # Start motion server
            motion_process = subprocess.Popen(
                motion_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # time.sleep(2.0)  # Wait for motion server to initialize

            # Start policy controller
            policy_process = subprocess.Popen(
                policy_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Monitor for completion (motion end or fall)
            start_time = time.time()
            completed = False
            fell = False

            while time.time() - start_time < timeout:
                if policy_process.poll() is not None:
                    # Policy process finished
                    if policy_process.returncode != 0:
                        cprint(f"    Policy process crashed (exit code {policy_process.returncode})", "red")
                        fell = True
                    completed = True
                    break

                if motion_process.poll() is not None:
                    # Motion server finished
                    cprint(f"    Motion server completed", "green")
                    completed = True
                    break

                # Check for fall detection by reading robot state from Redis
                # Skip first 2 seconds to avoid false positives during initialization
                if time.time() - start_time > 2.0:
                    try:
                        if check_fall_from_redis(redis_client):
                            cprint(f"    Robot fell detected (roll/pitch > 60°)", "red")
                            fell = True
                            completed = True
                            break
                    except Exception as e:
                        # Redis connection may fail, continue
                        pass

                time.sleep(0.1)

            # Terminate processes
            if motion_process.poll() is None:
                motion_process.terminate()
                try:
                    motion_process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    motion_process.kill()

            if policy_process.poll() is None:
                policy_process.terminate()
                try:
                    policy_process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    policy_process.kill()

            # Record result
            result = {
                'run': run + 1,
                'completed': completed,
                'fell': fell,
                'duration': time.time() - start_time
            }
            results.append(result)

            if fell:
                cprint(f"    Run {run+1}: Fell ❌", "red")
            elif completed:
                cprint(f"    Run {run+1}: Completed ✓", "green")
            else:
                cprint(f"    Run {run+1}: Timeout", "yellow")

            time.sleep(1.0)  # Brief pause between runs

        # Compute aggregate metrics for this model
        completed_runs = [r for r in results if r['completed']]
        fell_runs = [r for r in results if r['fell']]
        success_rate = len([r for r in results if r['completed'] and not r['fell']]) / num_runs
        fall_rate = len(fell_runs) / num_runs
        completion_rate = len(completed_runs) / num_runs

        model_metrics = {
            'model_name': os.path.basename(onnx_file),
            'num_runs': num_runs,
            'success_rate': success_rate,
            'fall_rate': fall_rate,
            'completion_rate': completion_rate,
            'num_successful': len([r for r in results if r['completed'] and not r['fell']]),
            'num_fell': len(fell_runs),
            'num_completed': len(completed_runs),
            'avg_duration': np.mean([r['duration'] for r in results]),
            'runs': results
        }

        return model_metrics

    except Exception as e:
        cprint(f"Error during experiment: {e}", "red")
        import traceback
        traceback.print_exc()
        return None

    finally:
        # Cleanup processes
        if motion_process and motion_process.poll() is None:
            motion_process.terminate()
            try:
                motion_process.wait(timeout=2)
            except:
                motion_process.kill()

        if policy_process and policy_process.poll() is None:
            policy_process.terminate()
            try:
                policy_process.wait(timeout=2)
            except:
                policy_process.kill()


def evaluate_all_models(motion_file, onnx_dir, redis_ip="localhost", num_runs=5, output_dir=None, reverse=False):
    """Evaluate all ONNX models in the specified directory."""
    onnx_files = find_onnx_models(onnx_dir, reverse)

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

    for exp_idx, onnx_file in enumerate(onnx_files):
        result = run_single_experiment(
            motion_file, onnx_file, redis_ip, exp_idx, num_runs
        )

        if result:
            all_results.append(result)
            print_result_summary(result)

        time.sleep(2.0)  # Brief pause between models

    # Generate summary report
    if all_results:
        generate_summary(all_results, output_dir)


def print_result_summary(result):
    """Print summary for a single model result."""
    cprint(f"\n[Summary: {result['model_name']}]", "yellow")
    cprint(f"  Success Rate:     {result['success_rate']*100:.1f}%", "green")
    cprint(f"  Fall Rate:        {result['fall_rate']*100:.1f}%", "red" if result['fall_rate'] > 0 else "green")
    cprint(f"  Completion Rate:  {result['completion_rate']*100:.1f}%", "cyan")
    cprint(f"  Avg Duration:      {result['avg_duration']:.2f}s", "white")
    cprint(f"  Runs: {result['num_successful']}/{result['num_runs']} successful, {result['num_fell']} fell", "white")


def generate_summary(all_results, output_dir):
    """Generate comprehensive summary report."""
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "onnx_evaluation_results")

    os.makedirs(output_dir, exist_ok=True)

    # Save detailed JSON results
    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump({
            'motion_file': 'motion',
            'total_models': len(all_results),
            'results': all_results
        }, f, indent=2)

    cprint(f"\nDetailed results saved to: {results_path}", "green")

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

    # Best model
    best_success = max(all_results, key=lambda x: x['success_rate'])
    cprint(f"\n{'='*70}", "cyan")
    cprint("BEST MODEL", "cyan")
    cprint(f"{'='*70}", "cyan")
    cprint(f"Model: {best_success['model_name']}", "green")
    cprint(f"Success Rate: {best_success['success_rate']*100:.1f}%", "green")
    cprint(f"Fall Rate:    {best_success['fall_rate']*100:.1f}%", "green")

    cprint(f"\n{'='*70}\n", "cyan")


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
        reverse=args.reverse
    )


if __name__ == "__main__":
    main()
