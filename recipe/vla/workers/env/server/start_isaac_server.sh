#!/bin/bash
# Isaac Lab Multi-Server-Group Startup Script
# Starts multiple independent Isaac Server groups for pipeline parallel training
# Each server group corresponds to one pipeline stage for physical isolation
# All server groups share the same GPUs (time-interleaved)
#
# Note: NUM_SERVER_GROUPS must match env.rollout.pipeline_stage_num in training config

set -e

WORKSPACE=/workspace/verl_vla/

# ============================================
# Configuration
# ============================================

# Number of tasks: libero_10 benchmark has 10 different tasks
NUM_TASKS=${NUM_TASKS:-10}

# Envs per task: how many env instances to run per task
GROUP_SIZE=${GROUP_SIZE:-16}

# Number of GPUs: all server groups share the same GPUs
NUM_GPUS=${NUM_GPUS:-8}

# Number of server groups - MUST match pipeline_stage_num in training config
# Each pipeline stage uses its own server group for physical isolation
NUM_SERVER_GROUPS=${NUM_SERVER_GROUPS:-2}

# Base port for server group 0 (subsequent groups use +50 spacing)
BASE_PORT=${BASE_PORT:-5556}
PORT_SPACING=${PORT_SPACING:-50}

# Base master port for torch.distributed (subsequent groups use +1)
BASE_MASTER_PORT=${BASE_MASTER_PORT:-29500}

# Action mode
export LIBERO_OSC_TYPE=pose_rel

# ============================================
# Calculate and Display Configuration
# ============================================

TOTAL_ENVS=$((NUM_TASKS * GROUP_SIZE))
TASKS_PER_GPU=$((NUM_TASKS / NUM_GPUS))
ENVS_PER_GPU=$((TASKS_PER_GPU * GROUP_SIZE))

echo "============================================"
echo "Isaac Lab Multi-Server-Group Mode"
echo "============================================"
echo "  GPU count:           ${NUM_GPUS} (shared by all groups)"
echo "  Total tasks:         ${NUM_TASKS}"
echo "  Tasks per GPU:       ${TASKS_PER_GPU}"
echo "  Envs per task:       ${GROUP_SIZE}"
echo "  Envs per GPU:        ${ENVS_PER_GPU}"
echo "  Total envs/group:    ${TOTAL_ENVS}"
echo "============================================"
echo "  Server groups:       ${NUM_SERVER_GROUPS}"
echo "  (Must match pipeline_stage_num in training config)"
echo "============================================"

for i in $(seq 0 $((NUM_SERVER_GROUPS - 1))); do
    SERVER_PORT=$((BASE_PORT + i * PORT_SPACING))
    MASTER_PORT=$((BASE_MASTER_PORT + i))
    echo "  Server Group $i (Stage $i):"
    echo "    Port range:        ${SERVER_PORT} - $((SERVER_PORT + NUM_GPUS - 1))"
    echo "    Master Port:       ${MASTER_PORT}"
done
echo "============================================"

# Python executable
PYTHON="/workspace/isaaclab/_isaac_sim/python.sh"

# Create log directory
mkdir -p ${WORKSPACE}/logs

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ============================================
# Start all server groups
# ============================================

SERVER_PIDS=()

for i in $(seq 0 $((NUM_SERVER_GROUPS - 1))); do
    SERVER_PORT=$((BASE_PORT + i * PORT_SPACING))
    MASTER_PORT=$((BASE_MASTER_PORT + i))
    
    echo "[$(date)] Starting Server Group $i (Stage $i)..."
    
    ${PYTHON} -m torch.distributed.run \
        --nproc_per_node=${NUM_GPUS} \
        --master_port=${MASTER_PORT} \
        ${WORKSPACE}/verl/recipe/vla/workers/env/server/isaac_server.py \
            --num_tasks ${NUM_TASKS} \
            --group_size ${GROUP_SIZE} \
            --port ${SERVER_PORT} \
            --use_tcp \
            --distributed \
        2>&1 | tee ${WORKSPACE}/logs/isaac_server${i}_${TIMESTAMP}.log &
    
    SERVER_PIDS+=($!)
    echo "[$(date)] Server Group $i started with PID: ${SERVER_PIDS[$i]}"
    
    # Wait between server group starts to avoid initialization conflicts
    if [ $i -lt $((NUM_SERVER_GROUPS - 1)) ]; then
        sleep 10
    fi
done

echo "============================================"
echo "All ${NUM_SERVER_GROUPS} server groups started!"
for i in $(seq 0 $((NUM_SERVER_GROUPS - 1))); do
    echo "  Server Group $i PID: ${SERVER_PIDS[$i]}"
done
echo ""
echo "To stop all servers:"
echo "  kill ${SERVER_PIDS[*]}"
echo "============================================"

# Wait for all processes
wait ${SERVER_PIDS[*]}
