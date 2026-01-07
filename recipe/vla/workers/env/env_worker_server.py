# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
EnvWorker Server Mode (Ray-based)

A lightweight EnvWorker that uses IsaacServerManager to manage
Isaac Lab simulations via Ray servers.

Key features:
    1. Uses Ray server handles for communication
    2. Isaac servers managed by IsaacServerManager
    3. Data transfer via Ray's object store (efficient)
    4. Servers are co-located in same Ray cluster as training

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    Ray Cluster                                  │
    │                                                                 │
    │  ┌─────────────────────┐      ┌───────────────────────────────┐│
    │  │  EnvWorkerServer    │ ───▶ │ IsaacServerManager            ││
    │  │   (Ray Worker)      │      │   ├─ Stage 0: [Server0, ...] ││
    │  │                     │ ◀─── │   └─ Stage 1: [Server0, ...] ││
    │  └─────────────────────┘      └───────────────────────────────┘│
    └─────────────────────────────────────────────────────────────────┘

Usage:
    # In main_ppo.py:
    from recipe.vla.workers.env import EnvWorkerServer

    # Or configure in YAML:
    env:
        train:
            isaac_server_mode: true
"""

import logging
import os
import uuid
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig
from torch.distributed.device_mesh import init_device_mesh

from recipe.vla.envs.action_utils import prepare_actions, put_info_on_image, save_rollout_video, tile_images
from recipe.vla.isaac_server import IsaacServerManager
from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.device import get_device_name

logger = logging.getLogger(__name__)


def extract_images_and_states(obs):
    """Extract images and states from Isaac Lab MultiTaskObservationsCfg format."""
    result = {}

    # Extract image from rgb_camera
    if "rgb_camera" in obs:
        rgb_camera = obs["rgb_camera"]
        if isinstance(rgb_camera, dict):
            if "agentview_cam" in rgb_camera:
                result["full_image"] = rgb_camera["agentview_cam"]
                result["camera_name"] = "agentview"
        elif isinstance(rgb_camera, np.ndarray):
            if rgb_camera.ndim == 4 and rgb_camera.shape[-1] >= 3:
                result["full_image"] = rgb_camera[:, :, :, :3]
                result["camera_name"] = "agentview"

    # Extract state from policy
    if "policy" in obs:
        policy = obs["policy"]
        if isinstance(policy, dict):
            state_list = []
            for key in ["eef_pose", "gripper_pos"]:
                if key in policy:
                    val = policy[key]
                    if hasattr(val, "shape"):
                        if len(val.shape) > 2:
                            val = val.reshape(val.shape[0], -1)
                        state_list.append(val)
            if state_list:
                result["state"] = np.concatenate(state_list, axis=-1)
        elif isinstance(policy, np.ndarray):
            result["state"] = policy

    return result


def create_env_batch_dataproto(obs, rews, terminations, truncations, infos, meta=None, task_descriptions=None):
    """Create DataProto from environment outputs."""
    ret_dict = {"obs": obs, "rews": rews, "terminations": terminations, "truncations": truncations, "infos": infos}
    if meta is not None:
        ret_dict.update(meta=meta)

    images_and_states = extract_images_and_states(ret_dict["obs"])

    tensor_batch = {
        "full_image": torch.from_numpy(images_and_states["full_image"]).cpu().contiguous(),
        "state": torch.from_numpy(images_and_states["state"]).cpu().contiguous(),
        "rews": torch.from_numpy(ret_dict["rews"]).cpu().contiguous(),
        "terminations": torch.from_numpy(ret_dict["terminations"]).cpu().contiguous(),
        "truncations": torch.from_numpy(ret_dict["truncations"]).cpu().contiguous(),
    }

    if task_descriptions is None:
        task_descriptions = obs.get("task_descriptions", [])
    non_tensor_batch = {"task_descriptions": task_descriptions}
    output = DataProto.from_dict(tensors=tensor_batch, non_tensors=non_tensor_batch)

    return output


class EnvWorkerServer(Worker):
    """
    EnvWorker that uses Ray-based IsaacServerManager.

    Uses Ray-based IsaacServerManager for Isaac Lab simulations.

    Key features:
    1. Uses IsaacServerManager for managing Ray servers
    2. Direct Ray server calls for communication
    3. Can receive manager from trainer (servers created externally)
    """

    def __init__(self, config: DictConfig, server_manager: Optional[IsaacServerManager] = None):
        """
        Initialize EnvWorkerServer.

        Args:
            config: Configuration dict
            server_manager: Optional pre-created IsaacServerManager.
                          If None, will be created in init_worker().
        """
        Worker.__init__(self)
        self.cfg = config

        # Isaac configuration
        self.env_id = config.train.get("env_id", "Isaac-Libero-Franka-OscPose-Camera-All-Tasks-v0")
        self.num_isaac_servers = config.train.get("num_isaac_servers", 8)
        self.camera_height = config.train.get("camera_height", 256)
        self.camera_width = config.train.get("camera_width", 256)

        # Pre-created manager (from trainer) or None
        self._external_manager = server_manager
        self.manager: IsaacServerManager = None

        # Stage configuration
        self.stage_num = self.cfg.rollout.pipeline_stage_num
        self.num_envs = self.cfg.train.num_envs

        # group_size = envs per task (from config)
        num_tasks = config.train.get("num_tasks", 10)
        self.group_size = config.train.get("group_size", 16)

        # Initialize distributed
        # EnvWorkerServer is a lightweight coordinator - may not have GPU
        # Use gloo-only backend if no GPU available
        import torch.distributed

        if not torch.distributed.is_initialized():
            # Use rank and world_size from environment (already set by Worker.__init__)
            rank = self.rank
            world_size = self.world_size

            # Check if we have GPU
            has_gpu = torch.cuda.is_available() and torch.cuda.device_count() > 0
            if has_gpu:
                # Use both gloo (CPU) and nccl (GPU)
                backend = "cpu:gloo,cuda:nccl"
            else:
                # No GPU - use gloo only
                backend = "gloo"
                logger.info(f"[rank={rank}] No GPU available, using gloo backend only")

            torch.distributed.init_process_group(
                backend=backend,
                rank=rank,
                world_size=world_size,
            )
        
        # Note: Worker base class already set self._rank and self._world_size from env vars
        # We don't override them here - they should match torch.distributed values

        # total_envs from config
        # In Server mode with single EnvWorkerServer, this is the global total
        self.total_envs = self.cfg.train.get("total_envs", 128)
        
        logger.info(
            f"[rank={self.rank}] EnvWorkerServer: "
            f"total_envs={self.total_envs}, world_size={self.world_size}"
        )

        device_name = "cpu" if not torch.cuda.is_available() else get_device_name()
        env_device_mesh = init_device_mesh(device_name, mesh_shape=(self.world_size, 1), mesh_dim_names=["dp", "tp"])
        self._register_dispatch_collect_info("env", dp_rank=env_device_mesh["dp"].get_local_rank(), is_collect=True)

        # Hash-key based env mapping
        self._trajectory_registry: dict[str, dict] = {}
        self._task_allocation_offset: dict[int, int] = {}
        self._active_trajectory_keys: list[str] = []

        # Video saving
        self.video_cfg = self.cfg.train.get("video_cfg", {})
        self.save_video = self.video_cfg.get("save_video", False)
        self.video_base_dir = self.video_cfg.get("video_base_dir", "/tmp/videos")
        self.render_images: dict[int, dict[int, list[np.ndarray]]] = {}
        self.video_cnt: dict[int, dict[int, int]] = {}
        self.camera_view = "agentview"

        logger.info(f"[rank={self.rank}] EnvWorkerServer initialized")
        logger.info(f"[rank={self.rank}] Using Ray actors")
        logger.info(f"[rank={self.rank}] Total envs: {self.total_envs}, stages: {self.stage_num}")

    def set_server_manager(self, manager: IsaacServerManager):
        """
        Set the server manager (created externally by trainer).

        This allows the trainer to create and manage the actors,
        and pass the manager to this worker.
        """
        self.manager = manager
        logger.info(f"[rank={self.rank}] Actor manager set from trainer")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_worker(self):
        """Initialize the Isaac server manager.

        If actor_manager was provided in __init__ or set via set_server_manager(),
        use that. Otherwise, create a new manager.
        """
        if self._external_manager is not None:
            self.manager = self._external_manager
            logger.info(f"[rank={self.rank}] Using external server manager")
        elif self.manager is not None:
            logger.info(f"[rank={self.rank}] Using pre-set server manager")
        else:
            # Create new manager
            num_tasks = self.cfg.train.get("num_tasks", 10)

            # Use "sim" accelerator_type to schedule actors to sim nodes
            # Sim nodes are started with python.sh which sets up Isaac environment
            # So we don't need to pass runtime_env - actors inherit sim node's environment
            accelerator_type = "sim"

            logger.info(f"[rank={self.rank}] Creating IsaacServerManager...")
            logger.info(f"[rank={self.rank}] Using accelerator_type='{accelerator_type}' to schedule to sim nodes")
            logger.info(f"[rank={self.rank}] num_tasks={num_tasks}, group_size={self.group_size}")

            self.manager = IsaacServerManager(
                num_stages=self.stage_num,
                num_servers_per_stage=self.num_isaac_servers,
                num_tasks=num_tasks,
                group_size=self.group_size,  # Envs per task (fixed)
                env_id=self.env_id,
                render_last_only=True,
                camera_height=self.camera_height,
                camera_width=self.camera_width,
                accelerator_type=accelerator_type,
                # Don't pass runtime_env - let servers inherit sim node's environment
                # Sim nodes are started with python.sh which sets PYTHONPATH, LD_LIBRARY_PATH, etc.
            )

            if not self.manager.initialize():
                raise RuntimeError("Failed to initialize IsaacServerManager")

        logger.info(
            f"[rank={self.rank}] Connected to Isaac servers: "
            f"{self.stage_num} stages × {self.num_isaac_servers} servers, "
            f"{self.manager.num_tasks} tasks, {self.manager._total_envs} total_envs"
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_simulator(self):
        """No-op for Ray mode - actors are already initialized."""
        logger.info(f"[rank={self.rank}] init_simulator called (no-op in Ray mode)")
        return

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="env"), blocking=False)
    def env_interact_step(self, data: DataProto) -> DataProto:
        """
        Interact with the environment through Ray actors.

        Uses Ray actor calls for environment interaction.
        """
        chunk_actions = data.non_tensor_batch["actions"]
        stage_id: int = data.meta_info["stage_id"]

        trajectory_keys = data.non_tensor_batch.get("trajectory_keys", None)

        if trajectory_keys is not None:
            self._validate_trajectory_keys(trajectory_keys)
        else:
            logger.warning(f"[rank={self.rank}] trajectory_keys not provided, using order from reset")
            # Each stage processes total_envs / stage_num environments
            envs_per_stage = self.total_envs // self.stage_num
            stage_start = stage_id * envs_per_stage
            stage_end = (stage_id + 1) * envs_per_stage
            trajectory_keys = self._active_trajectory_keys[stage_start:stage_end]
            self._validate_trajectory_keys(trajectory_keys)

        # Prepare actions
        chunk_actions = prepare_actions(
            simulator_type=self.cfg.train.simulator_type,
            raw_chunk_actions=chunk_actions,
            num_action_chunks=self.cfg.actor.model.num_action_chunks,
            action_dim=self.cfg.actor.model.action_dim,
        )

        if isinstance(chunk_actions, torch.Tensor):
            chunk_actions = chunk_actions.cpu().numpy()

        step_idx = data.meta_info.get("step_idx", 0)
        max_steps = data.meta_info.get("max_steps", 1)

        # Route to stage-specific actors
        response = self._actor_group_step(chunk_actions, trajectory_keys, stage_id, step_idx, max_steps)

        if response is None or response.get("status") != "ok":
            raise RuntimeError(f"[rank={self.rank}] Step request failed: {response}")

        # Build response DataProto
        env_info_list = {}

        terminations = response["terminations"]
        truncations = response["truncations"]
        chunk_dones = np.logical_or(terminations, truncations)

        if chunk_dones.any():
            infos = response.get("infos", {})
            if "final_info" in infos:
                final_info = infos["final_info"]
                for key in final_info.get("episode", {}):
                    env_info_list[key] = final_info["episode"][key][chunk_dones[:, -1]]

        # Generate task_descriptions from trajectory registry
        task_descriptions = []
        for key in trajectory_keys:
            task_id = self._trajectory_registry[key]["task_id"]
            task_descriptions.append(f"Task {task_id}")

        env_batch = create_env_batch_dataproto(
            obs=response["obs"],
            rews=response["rewards"],
            terminations=terminations,
            truncations=truncations,
            infos=response.get("infos", {}),
            meta=env_info_list,
            task_descriptions=task_descriptions,
        )

        # Video saving
        if self.save_video:
            self._collect_video_frames(response, trajectory_keys, stage_id)

        return env_batch

    def _actor_group_step(
        self,
        chunk_actions: np.ndarray,
        trajectory_keys: list,
        stage_id: int,
        step_idx: int = 0,
        max_steps: int = 1,
        render_last_only: bool = True,
    ) -> dict:
        """
        Execute step on the stage-specific actor group.

        Uses IsaacServerManager.step_batched() for parallel execution.
        """
        # Group by actor_rank
        actor_groups = defaultdict(lambda: {"indices": [], "actions": [], "original_positions": []})

        for i, key in enumerate(trajectory_keys):
            info = self._trajectory_registry[key]
            actor_rank = info["actor_rank"]
            actor_groups[actor_rank]["indices"].append(info["env_index"])
            actor_groups[actor_rank]["actions"].append(chunk_actions[i])
            actor_groups[actor_rank]["original_positions"].append(i)

        # Log step info
        actor_summary = {rank: len(group["indices"]) for rank, group in actor_groups.items()}
        print(
            f"[EnvWorker Ray] Step {step_idx + 1}/{max_steps} Stage {stage_id}: "
            f"{len(trajectory_keys)} envs -> {len(actor_groups)} actors {actor_summary}",
            flush=True,
        )

        # Build batched requests
        actor_requests = {}
        position_map = {}
        for rank, group in actor_groups.items():
            actions = np.array(group["actions"])
            indices = group["indices"]
            actor_requests[rank] = (actions, indices)
            position_map[rank] = group["original_positions"]

        # Send requests via manager
        responses = self.manager.step_batched(actor_requests, stage_id=stage_id, render_last_only=render_last_only)

        # Validate responses
        all_responses = {}
        for rank, response in responses.items():
            if response is None or response.get("status") != "ok":
                raise RuntimeError(
                    f"[rank={self.rank}] Step request to stage {stage_id} actor {rank} failed: {response}"
                )
            all_responses[rank] = {
                "response": response,
                "original_positions": position_map[rank],
            }

        # Aggregate responses
        return self._aggregate_responses(all_responses, len(trajectory_keys))

    def _aggregate_responses(self, all_responses: dict, num_envs: int) -> dict:
        """Aggregate responses from multiple actors into a single response."""
        first_response = next(iter(all_responses.values()))["response"]
        rewards_shape = first_response["rewards"].shape[1:] if len(first_response["rewards"].shape) > 1 else ()
        term_shape = first_response["terminations"].shape[1:] if len(first_response["terminations"].shape) > 1 else ()

        aggregated_rewards = np.zeros((num_envs,) + rewards_shape, dtype=first_response["rewards"].dtype)
        aggregated_terminations = np.zeros((num_envs,) + term_shape, dtype=first_response["terminations"].dtype)
        aggregated_truncations = np.zeros((num_envs,) + term_shape, dtype=first_response["truncations"].dtype)

        def init_aggregated_obs(obs_dict, num_envs):
            result = {}
            for key, value in obs_dict.items():
                if isinstance(value, dict):
                    result[key] = init_aggregated_obs(value, num_envs)
                elif isinstance(value, np.ndarray):
                    result[key] = np.zeros((num_envs,) + value.shape[1:], dtype=value.dtype)
                else:
                    result[key] = [None] * num_envs
            return result

        aggregated_obs = init_aggregated_obs(first_response["obs"], num_envs)

        def fill_aggregated_obs(agg_obs, response_obs, positions):
            for key, value in response_obs.items():
                if isinstance(value, dict):
                    fill_aggregated_obs(agg_obs[key], value, positions)
                elif isinstance(value, np.ndarray):
                    for i, pos in enumerate(positions):
                        agg_obs[key][pos] = value[i]
                else:
                    for i, pos in enumerate(positions):
                        agg_obs[key][pos] = value[i] if hasattr(value, "__getitem__") else value

        for actor_rank, data in all_responses.items():
            response = data["response"]
            positions = data["original_positions"]

            for i, pos in enumerate(positions):
                aggregated_rewards[pos] = response["rewards"][i]
                aggregated_terminations[pos] = response["terminations"][i]
                aggregated_truncations[pos] = response["truncations"][i]

            fill_aggregated_obs(aggregated_obs, response["obs"], positions)

        return {
            "status": "ok",
            "obs": aggregated_obs,
            "rewards": aggregated_rewards,
            "terminations": aggregated_terminations,
            "truncations": aggregated_truncations,
            "infos": {},
        }

    def _validate_trajectory_keys(self, trajectory_keys: list[str]) -> None:
        """Validate that all trajectory_keys exist in the registry."""
        invalid_keys = [key for key in trajectory_keys if key not in self._trajectory_registry]

        if invalid_keys:
            suffix = "..." if len(invalid_keys) > 5 else ""
            raise RuntimeError(
                f"[rank={self.rank}] Invalid trajectory_keys: {invalid_keys[:5]}{suffix}. "
                f"Valid keys: {list(self._trajectory_registry.keys())[:5]}..."
            )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_all_state_ids(self):
        """Get all available state IDs."""
        return list(range(self.manager.num_tasks))

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="env"), blocking=False)
    def reset_envs_to_state_ids(self, data: DataProto):
        """
        Reset environments to specified state IDs (task IDs).

        Uses IsaacServerManager for batched reset across all stages.
        """
        state_ids_list = list(data.non_tensor_batch["state_ids"])
        task_ids_list = list(data.non_tensor_batch["task_ids"])

        logger.debug(
            f"[rank={self.rank}] reset_envs_to_state_ids: {len(state_ids_list)} envs, "
            f"unique_tasks={len(set(task_ids_list))}"
        )

        expected_count = self.total_envs
        assert len(state_ids_list) == expected_count, (
            f"[rank={self.rank}] Expected {expected_count} state_ids, got {len(state_ids_list)}"
        )

        # Clear previous mappings
        self._trajectory_registry.clear()
        self._task_allocation_offset.clear()
        self._active_trajectory_keys.clear()

        trajectory_keys = []
        num_trajectories = self.total_envs // self.num_envs

        # Group by actor_rank for batched reset
        actor_env_groups = defaultdict(lambda: {"env_indices": [], "traj_indices": []})
        traj_to_actor = {}

        for traj_idx in range(num_trajectories):
            traj_start = traj_idx * self.num_envs
            traj_end = (traj_idx + 1) * self.num_envs
            traj_task_ids = task_ids_list[traj_start:traj_end]

            unique_task_ids = set(traj_task_ids)
            if len(unique_task_ids) != 1:
                raise RuntimeError(f"[rank={self.rank}] Trajectory {traj_idx} has mixed task_ids: {unique_task_ids}")

            task_id = traj_task_ids[0]

            # Get actor's env_indices for this task
            task_env_indices = self.manager.get_env_indices_for_task(task_id)
            group_size = len(task_env_indices)

            # Get which actor handles this task
            actor_rank = self.manager.get_server_for_task(task_id)

            # Sequential allocation within each task's env pool
            offset = self._task_allocation_offset.get(task_id, 0)

            if offset + self.num_envs > group_size:
                raise RuntimeError(
                    f"[rank={self.rank}] Task {task_id} has only {group_size} envs, but need {offset + self.num_envs}"
                )

            env_indices = task_env_indices[offset : offset + self.num_envs]
            self._task_allocation_offset[task_id] = offset + self.num_envs

            # Generate unique hash_key for each env
            traj_env_keys = []
            for i, env_idx in enumerate(env_indices):
                env_key = str(uuid.uuid4())[:8]
                self._trajectory_registry[env_key] = {
                    "env_index": env_idx,
                    "task_id": task_id,
                    "traj_idx": traj_idx,
                    "actor_rank": actor_rank,  # Changed from server_rank
                }
                traj_env_keys.append(env_key)

            trajectory_keys.extend(traj_env_keys)

            # Group env_indices by actor for batched reset
            actor_env_groups[actor_rank]["env_indices"].extend(env_indices)
            actor_env_groups[actor_rank]["traj_indices"].append(traj_idx)
            traj_to_actor[traj_idx] = actor_rank

            logger.info(
                f"[rank={self.rank}] Traj {traj_idx}: task_id={task_id}, actor_rank={actor_rank}, "
                f"env_indices [{env_indices[0]}-{env_indices[-1]}]"
            )

        # Batched reset - reset all stages
        print(
            f"[EnvWorker Ray] Batched Reset: {num_trajectories} trajectories -> "
            f"{len(actor_env_groups)} actor(s) x {self.stage_num} stages",
            flush=True,
        )

        actor_requests = {rank: group["env_indices"] for rank, group in actor_env_groups.items()}

        # Reset all stages using manager
        actor_responses_per_stage = self.manager.reset_all_stages_batched(actor_requests, stabilize=True)

        # Validate responses
        for stage_id, stage_responses in actor_responses_per_stage.items():
            for rank, response in stage_responses.items():
                if response.get("status") != "ok":
                    raise RuntimeError(f"[rank={self.rank}] Reset to stage {stage_id} actor {rank} failed: {response}")

        # Reconstruct result_list in trajectory order
        actor_positions_per_stage = {
            stage_id: {rank: 0 for rank in actor_env_groups.keys()} for stage_id in range(self.stage_num)
        }
        result_list = []

        for traj_idx in range(num_trajectories):
            stage_id = traj_idx % self.stage_num
            actor_rank = traj_to_actor[traj_idx]
            obs = actor_responses_per_stage[stage_id][actor_rank]["obs"]
            pos = actor_positions_per_stage[stage_id][actor_rank]

            # Extract this trajectory's observations
            traj_obs = {}
            for key, value in obs.items():
                if isinstance(value, np.ndarray):
                    traj_obs[key] = value[pos : pos + self.num_envs]
                elif isinstance(value, dict):
                    traj_obs[key] = {}
                    for k2, v2 in value.items():
                        if isinstance(v2, np.ndarray):
                            traj_obs[key][k2] = v2[pos : pos + self.num_envs]
                        else:
                            traj_obs[key][k2] = v2
                else:
                    traj_obs[key] = value

            result_list.append(traj_obs)
            actor_positions_per_stage[stage_id][actor_rank] += self.num_envs

        # Store active keys
        self._active_trajectory_keys = trajectory_keys

        logger.info(f"[rank={self.rank}] Reset complete: {len(trajectory_keys)} env keys registered")

        # Build output DataProto
        output_tensor_dict = {}
        output_non_tensor_dict = {}

        images_and_states_list = [extract_images_and_states(obs) for obs in result_list]
        if images_and_states_list and images_and_states_list[0]:
            for k in images_and_states_list[0].keys():
                if isinstance(images_and_states_list[0][k], np.ndarray):
                    output_tensor_dict[k] = torch.from_numpy(
                        np.concatenate([obs[k] for obs in images_and_states_list], axis=0)
                    )

        # Task descriptions
        task_descriptions = []
        for traj_idx, obs in enumerate(result_list):
            if "task_descriptions" in obs:
                task_descriptions.extend(obs["task_descriptions"])
            else:
                env_key = trajectory_keys[traj_idx * self.num_envs]
                task_id = self._trajectory_registry[env_key]["task_id"]
                task_descriptions.extend([f"Task {task_id}"] * self.num_envs)
        output_non_tensor_dict["task_descriptions"] = task_descriptions

        output_non_tensor_dict["trajectory_keys"] = trajectory_keys

        output = DataProto.from_dict(tensors=output_tensor_dict, non_tensors=output_non_tensor_dict)
        return output

    def _collect_video_frames(self, response, trajectory_keys, stage_id):
        """Collect images for video saving - organized by stage and task_id."""
        images_and_states = extract_images_and_states(response["obs"])
        if "full_image" not in images_and_states:
            return

        if "camera_name" in images_and_states:
            self.camera_view = images_and_states["camera_name"]

        full_images = images_and_states["full_image"]
        rewards = response.get("rewards", np.zeros(len(trajectory_keys)))
        terminations = response.get("terminations", np.zeros(len(trajectory_keys)))

        if stage_id not in self.render_images:
            self.render_images[stage_id] = {}
            self.video_cnt[stage_id] = {}

        # Group images by task_id
        task_id_to_images = {}
        for i, key in enumerate(trajectory_keys):
            task_id = self._trajectory_registry[key]["task_id"]
            img = full_images[i]

            # Add info overlay
            if len(rewards.shape) > 1:
                reward_val = float(rewards[i, -1])
            else:
                reward_val = float(rewards[i])

            if len(terminations.shape) > 1:
                term_val = bool(terminations[i, -1])
            else:
                term_val = bool(terminations[i])

            plot_info = {"reward": reward_val, "done": term_val, "task": task_id}
            img = put_info_on_image(img, plot_info)

            if task_id not in task_id_to_images:
                task_id_to_images[task_id] = []
            task_id_to_images[task_id].append(img)

        # Tile and store images for each task_id
        for task_id, task_images in task_id_to_images.items():
            if task_id not in self.render_images[stage_id]:
                self.render_images[stage_id][task_id] = []
                self.video_cnt[stage_id][task_id] = 0

            nrows = int(np.ceil(np.sqrt(len(task_images))))
            tiled = tile_images(task_images, nrows=nrows)
            self.render_images[stage_id][task_id].append(tiled)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def finish_rollout(self, mode="train"):
        """Finish rollout - save videos organized by stage and task_id."""
        if self.save_video and self.render_images:
            total_videos = 0
            for stage_id, stage_data in self.render_images.items():
                for task_id, frames in stage_data.items():
                    if not frames:
                        continue

                    output_dir = os.path.join(self.video_base_dir, mode, f"stage_{stage_id}", f"task_{task_id}")
                    os.makedirs(output_dir, exist_ok=True)

                    cnt = self.video_cnt.get(stage_id, {}).get(task_id, 0)
                    video_name = f"rollout_{cnt:04d}_{self.camera_view}"

                    save_rollout_video(frames, output_dir, video_name)

                    self.video_cnt[stage_id][task_id] = cnt + 1
                    total_videos += 1

            self.render_images = {}
            logger.info(f"[rank={self.rank}] Saved {total_videos} videos to {self.video_base_dir}/{mode}/")
        return

    def __del__(self):
        """Clean up - don't close manager if it was provided externally."""
        # Only close if we created the manager ourselves
        if hasattr(self, 'manager') and self.manager and \
            hasattr(self, '_external_manager') and self._external_manager is None:
            self.manager.close()
