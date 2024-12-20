#!/bin/bash

# Ensure the script exits on any error
set -e

# Add local bin to PATH for torchrun
export PATH=$PATH:/home/user/.local/bin

# Get available GPU count
export GPU_COUNT=$(nvidia-smi -L | wc -l)

# Verify GPUs are available
if [ "$GPU_COUNT" -eq 0 ]; then
    echo "Error: No GPUs found. This example requires GPU support."
    exit 1
fi

echo "Found $GPU_COUNT GPUs"

# Launch distributed training
# --standalone: single-node multi-GPU training
# --node_rank=0: this is the first and only node
# --nproc_per_node: spawn one process per GPU
torchrun \
    --standalone \
    --node_rank=0 \
    --nproc_per_node=$GPU_COUNT \
    example.py
