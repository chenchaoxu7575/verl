#!/bin/bash
set -x

echo "Ray Actor Mode: Isaac Sim runs as Ray actors (unified resource management)"

WORKSPACE=/workspace/verl_vla/

libero_train_path=$WORKSPACE/data/libero_rl/train.parquet
libero_test_path=$WORKSPACE/data/libero_rl/test.parquet

train_files=$libero_train_path
test_files=$libero_test_path

OUTPUT_DIR=${MLP_MODEL_OUTPUT:-"$WORKSPACE/models/vla_libero_grpo_ray"}
VIDEO_OUTPUT=${MLP_MODEL_OUTPUT:-"$WORKSPACE"}/video/$(date +%Y%m%d_%H%M%S)
TIMELINE_FILE=${TIMELINE_FILE:-"$WORKSPACE/logs/ray_timeline_$(date +%Y%m%d_%H%M%S).json"}
SFT_MODEL_PATH=${SFT_MODEL_PATH:-"$WORKSPACE/data/Openvla-oft-SFT-libero10-trajall"}

# for rollout and train
NUM_NODES=1
# for simulator
SIM_NODES=1
NUM_ENV_GPUS=8
NUM_ROLLOUT_GPUS=8
STAGE_NUM=2
BATCH_SIZE=16
# rollout.n should equal to num_envs for isaac env
# GROUP_SIZE / ROLLOUT_N = max trajectories per task (need >= 2 for uneven data distribution)
ROLLOUT_N=8
SERVER_GROUP_SIZE=16

# 512 is required for libero benchmark, but you can reduce it in debugging to run faster
MAX_EPISODE_STEPS=128

# Number of tasks in the benchmark
# Must satisfy: NUM_TASKS × GROUP_SIZE = NUM_ROLLOUT_GPUS × STAGE_NUM × ROLLOUT_N
NUM_TASKS=10

# isaac or libero
# NOT SUPPORTED: libero means original libero benchmark with mujoco sim
# isaac means libero benchmark using isaac sim
SIM_TYPE=${SIM_TYPE:-"isaac"}
PROJECT_NAME="vla-ray-isaac"
EXPERIMENT_NAME="${SIM_TYPE}_ray_rl"

# ============================================
# Isaac Ray Actor Mode Configuration
# ============================================
# Ray actor mode: Isaac Sim runs as Ray actors
# - Ray manages all GPU resources uniformly
# - No manual server startup needed
# - Actors are created and scheduled by Ray
# ============================================

# Enable Ray actor mode (recommended)
USE_RAY_ACTORS=True

# Disable legacy ZMQ server mode
USE_SERVER_MODE=False

# Number of Isaac actors per stage (one per GPU)
NUM_ISAAC_ACTORS=$NUM_ENV_GPUS

# Isaac environment ID
ISAAC_ENV_ID="Isaac-Libero-Franka-OscPose-Camera-All-Tasks-v0"

# Camera settings
CAMERA_HEIGHT=256
CAMERA_WIDTH=256

# Group size (envs per task) - configure directly
# This is the number of parallel environments per task
GROUP_SIZE=16

# TOTAL_ENVS = environments participating in training (from rollout config)
# ISAAC_ENVS = total environments in Isaac Sim (may be more, extras stay idle)
TOTAL_ENVS=$((NUM_ROLLOUT_GPUS * STAGE_NUM * ROLLOUT_N))
ISAAC_ENVS=$((NUM_TASKS * GROUP_SIZE))

echo "============================================"
echo "Environment allocation:"
echo "  Training envs (total_envs): ${TOTAL_ENVS} = ${NUM_ROLLOUT_GPUS} gpus × ${STAGE_NUM} stages × ${ROLLOUT_N} rollout_n"
echo "  Isaac Sim envs: ${ISAAC_ENVS} = ${NUM_TASKS} tasks × ${GROUP_SIZE} group_size"
if [ $ISAAC_ENVS -lt $TOTAL_ENVS ]; then
    echo "ERROR: Not enough Isaac envs! Need at least ${TOTAL_ENVS}"
    exit 1
fi
echo "  Idle envs: $((ISAAC_ENVS - TOTAL_ENVS))"
echo "============================================"

echo "============================================"
echo "Isaac Ray Actor Mode Configuration:"
echo "  Mode: Ray Actor (unified resource management)"
echo "  Isaac actors per stage: ${NUM_ISAAC_ACTORS}"
echo "  Pipeline stages: ${STAGE_NUM}"
echo "  Total actors: $((NUM_ISAAC_ACTORS * STAGE_NUM))"
echo "  Tasks: ${NUM_TASKS}"
echo "  Total envs: ${TOTAL_ENVS} (${NUM_ROLLOUT_GPUS} gpus × ${STAGE_NUM} stages × ${ROLLOUT_N} envs)"
echo "  Group size (envs/task): ${GROUP_SIZE}"
echo "============================================"
echo ""
echo "No manual server startup needed - Ray manages everything!"
echo "============================================"

ISSC_PYTHON="/workspace/isaaclab/_isaac_sim/python.sh"
PYTHON=python
if [ -f "$ISSC_PYTHON" ]; then
    PYTHON=$ISSC_PYTHON
fi

# avoiding warnings
mkdir -p /root/LIBERO/libero/libero/../datasets

SAVE_VIDEO=True

export PYTHONRECURSIONLIMIT=10000
# uncomment this to see full error messages
# export HYDRA_FULL_ERROR=1

$PYTHON -m recipe.vla.main_ppo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=${BATCH_SIZE} \
    data.val_batch_size=${BATCH_SIZE} \
    +data.num_tasks=${NUM_TASKS} \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    env.train.num_envs=$ROLLOUT_N \
    data.max_prompt_length=256 \
    data.max_response_length=128 \
    env.rollout.pipeline_stage_num=$STAGE_NUM \
    env.train.simulator_type=$SIM_TYPE \
    env.actor.model.num_action_chunks=8 \
    env.actor.model.action_dim=7 \
    env.train.only_eval=False \
    env.train.max_episode_steps=$MAX_EPISODE_STEPS \
    env.train.video_cfg.save_video=$SAVE_VIDEO \
    env.train.video_cfg.video_base_dir=${VIDEO_OUTPUT} \
    env.train.seed=42 \
    env.disagg_sim.enable=True \
    env.disagg_sim.nnodes=$SIM_NODES \
    env.train.use_server_mode=$USE_SERVER_MODE \
    env.train.use_ray_actors=$USE_RAY_ACTORS \
    env.train.num_isaac_actors=$NUM_ISAAC_ACTORS \
    env.train.num_tasks=$NUM_TASKS \
    env.train.group_size=$GROUP_SIZE \
    env.train.env_id=$ISAAC_ENV_ID \
    env.train.camera_height=$CAMERA_HEIGHT \
    env.train.camera_width=$CAMERA_WIDTH \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.model.path=$SFT_MODEL_PATH \
    actor_rollout_ref.rollout.mode=async_envloop \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.grad_clip=1 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.num_images_in_input=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.trust_remote_code=False \
    actor_rollout_ref.actor.entropy_coeff=0. \
    actor_rollout_ref.rollout.temperature=1.6 \
    actor_rollout_ref.rollout.prompt_length=512 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.00 \
    trainer.logger=['console'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$OUTPUT_DIR \
    trainer.n_gpus_per_node=$NUM_ROLLOUT_GPUS \
    env.train.server_group_size=$SERVER_GROUP_SIZE \
    +trainer.n_env_gpus_per_node=$NUM_ENV_GPUS \
    +trainer.n_rollout_gpus_per_node=$NUM_ROLLOUT_GPUS \
    trainer.nnodes=$NUM_NODES \
    env.train.total_envs=$TOTAL_ENVS \
    trainer.save_freq=30 \
    trainer.test_freq=-1 \
    trainer.total_epochs=20 \
    trainer.val_only=False \
    trainer.total_training_steps=3 \
    algorithm.adv_estimator=reinforce_plus_plus \
    trainer.val_before_train=False \
    trainer.resume_mode=disable \
    +ray_kwargs.timeline_json_file=$TIMELINE_FILE \
    $@ 2>&1 | tee $WORKSPACE/logs/vla_isaac_ray_$(date +%Y%m%d_%H%M%S).log

