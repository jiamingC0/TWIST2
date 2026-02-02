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
    motion_file, redis_ip = args
    motion_server = MotionServer(
        motion_file=motion_file,
        robot="unitree_g1_with_hands",
        redis_ip=redis_ip,
        steps="1",
        use_remote_control=False,
        send_start_frame_as_end_frame=False,
        show_viewer=True
    )
    result = motion_server.run()
    queue.put(('motion', result))


def _policy_controller_wrapper(queue, args):
    """Wrapper for policy controller subprocess."""
    onnx_file, timeout = args
    policy_controller = RealTimePolicyController(
        xml_file="assets/g1/g1_sim2sim_29dof.xml",
        policy_path=onnx_file,
        device="cpu",
        record_video=False,
        record_proprio=False,
        measure_fps=False,
        limit_fps=True,
        policy_frequency=100,
        show_viewer=True
    )
    result = policy_controller.run(timeout=timeout)
    queue.put(('policy', result))


def run_single_experiment(motion_file, motion_length, onnx_file, redis_ip="localhost", exp_idx=0, num_runs=5):
    """
    Run a single experiment with specified ONNX model using multiprocessing.
    Returns metrics dict if experiment completed successfully, None otherwise.
    """
    cprint(f"\n{'='*70}", "cyan")
    cprint(f"Experiment {exp_idx+1}: {os.path.basename(onnx_file)}", "cyan")
    cprint(f"{'='*70}", "cyan")

    # Get motion duration from pkl file
    
        
    motion_duration = float(motion_length)
    cprint(f"Motion duration: {motion_duration:.2f} seconds", "cyan")

    results = []

    try:
        # Run experiment num_runs times using multiprocessing
        for run in range(num_runs):
            cprint(f"\n  Run {run+1}/{num_runs}...", "yellow")

            timeout = motion_duration + 5.0

            # Start motion server subprocess
            motion_queue = mp.Queue()
            motion_process = mp.Process(
                target=_motion_server_wrapper,
                args=(motion_queue, (motion_file, redis_ip))
            )
            motion_process.start()
            
            time.sleep(2.0)
            
            # Start policy controller subprocess
            policy_queue = mp.Queue()
            policy_process = mp.Process(
                target=_policy_controller_wrapper,
                args=(policy_queue, (onnx_file, timeout))
            )
            policy_process.start()
            
            

            # Monitor processes
            start_time = time.time()
            completed = False
            fell = False

            while time.time() - start_time < timeout:
                # Check if motion process finished
                if not motion_process.is_alive():
                    completed = True
                    cprint(f"    Motion server completed", "green")
                    break

                # Check if policy process finished
                if not policy_process.is_alive():
                    completed = True
                    cprint(f"    Policy controller stopped", "yellow")
                    break

                time.sleep(0.1)

            # Terminate processes if still running
            if motion_process.is_alive():
                cprint(f"    Terminating motion server (timeout)...", "yellow")
                motion_process.terminate()
                motion_process.join(timeout=2)
                if motion_process.is_alive():
                    motion_process.kill()
                    motion_process.join(timeout=2)
            if policy_process.is_alive():
                cprint(f"    Terminating policy controller (timeout)...", "yellow")
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

            # Check for fall based on policy result
            if policy_result and policy_result[1] == 'fell':
                fell = True
                cprint(f"    Robot fell detected", "red")

            result = {
                'run': run + 1,
                'completed': completed,
                'fell': fell,
                'duration': time.time() - start_time
            }
            results.append(result)

            if completed:
                cprint(f"    Run {run+1}: Completed ✓", "green")
            else:
                cprint(f"    Run {run+1}: Timeout", "red")

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


def evaluate_all_models(motion_file, onnx_dir, redis_ip="localhost", num_runs=5, output_dir=None, reverse=False):
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
            motion_file, motion_length, onnx_file, redis_ip, exp_idx, num_runs
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
