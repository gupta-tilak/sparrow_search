#!/bin/bash

# Set environment variables
export CUDA_VISIBLE_DEVICES=0  # Specify GPU device
export DATA_DIR=/home/user/WS-Extrapolation/sparrow_search/data
export LOG_DIR=/home/user/WS-Extrapolation/sparrow_search/logs
export MODEL_DIR=/home/user/WS-Extrapolation/sparrow_search/models
export CHECKPOINT_DIR=/home/user/WS-Extrapolation/sparrow_search/checkpoints
export SAVE_DIR=/home/user/WS-Extrapolation/sparrow_search/results

# Activate virtual environment (if using one)
source venv/bin/activate

# Install required packages
pip install -r requirements.txt

# Run the script with nohup
nohup python -u sparrow_search_remote_server.py > wind_speed.log 2>&1 &

# Save the process ID
echo $! > process.pid