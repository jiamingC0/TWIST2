#!/bin/bash

# Script to convert student policy with future motion support to ONNX
# Usage: bash to_onnx.sh $YOUR_POLICY_OR_FOLDER_PATH

input_path=$1

cd legged_gym/legged_gym/scripts

# Check if input is a file or folder
if [ -f "$input_path" ]; then
    # Single file
    echo "Converting single file: $input_path"
    python save_onnx.py --ckpt_path "$input_path"
elif [ -d "$input_path" ]; then
    # Folder - convert all .pt files
    echo "Converting all .pt files in folder: $input_path"
    for ckpt_path in "$input_path"/*.pt; do
        if [ -f "$ckpt_path" ]; then
            echo "Converting: $ckpt_path"
            python save_onnx.py --ckpt_path "$ckpt_path"
        fi
    done
else
    echo "Error: $input_path is not a valid file or folder"
    exit 1
fi
