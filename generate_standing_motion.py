#!/usr/bin/env python3
"""
Generate a long-duration standing motion PKL file for stable standing.
Based on the structure of 251215-083327.pkl
"""

import pickle
import numpy as np
import sys
from pathlib import Path

def generate_standing_motion(num_frames=3000, fps=50.0, source_file=None):
    """
    Generate a standing motion with default standing pose.

    Args:
        num_frames: Number of frames (default: 3000 = 60 seconds at 50Hz)
        fps: Frame rate (default: 50.0 Hz)
        source_file: Path to source pkl file to extract link_body_list

    Returns:
        Dictionary containing the motion data
    """
    # Default standing pose for G1 robot
    # 29 DOFs:
    # Left leg (0-5): hip_yaw, hip_roll, hip_pitch, knee_pitch, ankle_pitch, ankle_roll
    # Right leg (6-11): hip_yaw, hip_roll, hip_pitch, knee_pitch, ankle_pitch, ankle_roll
    # Waist (12-14): yaw, roll, pitch
    # Left arm (15-21): shoulder_pitch, shoulder_roll, shoulder_yaw, elbow_pitch, elbow_roll, wrist_yaw, wrist_roll
    # Right arm (22-28): shoulder_pitch, shoulder_roll, shoulder_yaw, elbow_pitch, elbow_roll, wrist_yaw, wrist_roll
    
    default_dof_pos = np.array([
        # Left leg - neutral standing
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # [0-5]
        # Right leg - neutral standing
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # [6-11]
        # Waist - upright
        0.0, 0.0, 0.0,  # [12-14]
        # Left arm - hanging down (slightly forward)
        0.3, 0.2, 0.0, -1.0, 0.0, 0.0, 0.0,  # [15-21]
        # Right arm - hanging down (slightly forward)
        0.3, 0.2, 0.0, -1.0, 0.0, 0.0, 0.0,  # [22-28]
    ], dtype=np.float64)
    
    # Create motion data
    motion_data = {
        'fps': fps,
        'root_pos': np.zeros((num_frames, 3), dtype=np.float64),  # Position at origin
        'root_rot': np.array([[1.0, 0.0, 0.0, 0.0]] * num_frames, dtype=np.float64).reshape(num_frames, 4),  # Identity quaternion
        'dof_pos': np.tile(default_dof_pos, (num_frames, 1)).astype(np.float64),  # Constant standing pose
        'local_body_pos': np.zeros((num_frames, 38, 3), dtype=np.float32),  # All bodies at origin relative to root
        'link_body_list': [],  # Will be filled from source file if provided
    }
    
    # Set root height to typical standing height
    motion_data['root_pos'][:, 2] = 0.85  # ~85cm standing height
    
    # Load link_body_list from source file if provided
    if source_file and Path(source_file).exists():
        with open(source_file, 'rb') as f:
            source_data = pickle.load(f)
            if 'link_body_list' in source_data:
                motion_data['link_body_list'] = source_data['link_body_list']
                print(f"Loaded link_body_list from source file: {len(motion_data['link_body_list'])} bodies")
            else:
                print(f"Warning: link_body_list not found in source file, using empty list")
    else:
        print(f"Warning: source file not found or not provided, using empty link_body_list")
    
    return motion_data


def main():
    # Parse command line arguments
    if len(sys.argv) > 1:
        num_frames = int(sys.argv[1])
    else:
        num_frames = 3000  # Default: 60 seconds at 50Hz
    
    # Source file for link_body_list
    source_file = '/home/galbot/Galbot/track/storage/data/mocap/20251215_wufengchao_bvh_pkl/251215-083327.pkl'
    
    # Output file
    output_file = '/home/galbot/MyTWIST2/TWIST2/assets/example_motions/standing_60s.pkl'
    
    # Generate standing motion
    print(f"Generating standing motion with {num_frames} frames ({num_frames/50:.1f} seconds at 50Hz)...")
    motion_data = generate_standing_motion(num_frames=num_frames, fps=50.0, source_file=source_file)
    
    # Save to PKL file
    with open(output_file, 'wb') as f:
        pickle.dump(motion_data, f)
    
    print(f"Standing motion saved to: {output_file}")
    print(f"  Frames: {num_frames}")
    print(f"  Duration: {num_frames/50:.1f} seconds")
    print(f"  FPS: {motion_data['fps']}")
    print(f"  Root shape: {motion_data['root_pos'].shape}")
    print(f"  DOF shape: {motion_data['dof_pos'].shape}")
    print(f"  Local body shape: {motion_data['local_body_pos'].shape}")


if __name__ == '__main__':
    main()
