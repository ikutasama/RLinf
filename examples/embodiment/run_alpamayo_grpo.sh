#!/bin/bash
# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Alpamayo-R1 GRPO Training Script for RLinf Framework
# Usage: bash examples/embodiment/run_alpamayo_grpo.sh [CONFIG_NAME] [GPU_IDS]
# Example: bash examples/embodiment/run_alpamayo_grpo.sh alpamayo_grpo 0-3

set -e

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/train_embodied_agent.py"

# Set environment variables
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"

# Default configuration
if [ -z "$1" ]; then
    CONFIG_NAME="alpamayo_grpo"
else
    CONFIG_NAME=$1
fi

# Default GPU configuration
if [ -z "$2" ]; then
    GPU_IDS="0"
else
    GPU_IDS=$2
fi

# Set visible GPUs
export CUDA_VISIBLE_DEVICES=${GPU_IDS}

# Set Python path
export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

# Set HuggingFace mirror for users in mainland China (optional)
# export HF_ENDPOINT=https://hf-mirror.com

echo "============================================================"
echo "Alpamayo-R1 GRPO Training with RLinf Framework"
echo "============================================================"
echo "Configuration: ${CONFIG_NAME}"
echo "GPUs: ${GPU_IDS}"
echo "Repository: ${REPO_PATH}"
echo "============================================================"

# Create log directory
LOG_DIR="${REPO_PATH}/logs/alpamayo_grpo/$(date +'%Y%m%d-%H%M%S')-${CONFIG_NAME}"
MEGA_LOG_FILE="${LOG_DIR}/run_alpamayo_grpo.log"
mkdir -p "${LOG_DIR}"

# Build command with gpu_ids parameter
CMD="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ --config-name ${CONFIG_NAME} cluster.component_placement=actor,env,rollout:${GPU_IDS} runner.logger.log_path=${LOG_DIR}"

echo "Command: ${CMD}"
echo "Log file: ${MEGA_LOG_FILE}"
echo "============================================================"
echo ""

# Execute and log
echo ${CMD} > ${MEGA_LOG_FILE}
${CMD} 2>&1 | tee -a ${MEGA_LOG_FILE}

echo ""
echo "============================================================"
echo "Training completed!"
echo "Logs saved to: ${LOG_DIR}"
echo "============================================================"
