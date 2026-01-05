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
Isaac Lab Simulation Actor (Ray-based)

A Ray Actor that runs Isaac Lab multi-task simulation on a single GPU.
This replaces the ZMQ-based server with Ray's native actor model for:
- Unified resource management by Ray
- Simplified deployment (no manual server startup)
- Efficient data transfer via Ray's object store

Architecture:
    - One IsaacSimActor per GPU
    - Each Actor handles a subset of tasks (like distributed ZMQ server)
    - Multiple Actors managed by IsaacSimActorManager (one manager per stage)

Usage:
    # Create actor (Ray will schedule to available GPU)
    actor = IsaacSimActor.options(num_gpus=1).remote(
        env_id="Isaac-Libero-Franka-OscPose-Camera-All-Tasks-v0",
        num_tasks=5,
        group_size=8,  # Envs per task
        actor_rank=0,
        task_offset=0,
    )

    # Initialize (must call before step/reset)
    ray.get(actor.init_env.remote())

    # Use step/reset
    result = ray.get(actor.step.remote(actions, env_indices))
"""

import logging
import os
from typing import Any, Optional

import numpy as np
import ray

logger = logging.getLogger("IsaacSimActor")


def setup_per_process_caches(actor_rank: int, stage_id: int = 0):
    """Setup per-process cache directories to avoid locking conflicts.

    Same as in server.py - prevents OptiX/shader cache conflicts.
    Must be called BEFORE importing any Isaac/Omniverse modules.

    Important: Each (stage_id, actor_rank) pair needs unique cache directories
    to avoid conflicts between different stage actors.
    """
    # Use stage_id/actor_rank to create unique cache paths
    cache_suffix = f"stage_{stage_id}/rank_{actor_rank}"

    # === OptiX Cache ===
    optix_base = "/tmp/optix_cache"  # Don't inherit from env - always use fresh
    optix_rank_cache = os.path.join(optix_base, cache_suffix)
    os.makedirs(optix_rank_cache, exist_ok=True)
    os.environ["OPTIX_CACHE_PATH"] = optix_rank_cache
    os.environ["OPTIX7_CACHE_PATH"] = optix_rank_cache

    # === NVIDIA Driver Shader Cache ===
    shader_base = "/tmp/nv_shader_cache"  # Don't inherit from env
    shader_rank_cache = os.path.join(shader_base, cache_suffix)
    os.makedirs(shader_rank_cache, exist_ok=True)
    os.environ["__GL_SHADER_DISK_CACHE"] = "1"
    os.environ["__GL_SHADER_DISK_CACHE_PATH"] = shader_rank_cache
    os.environ["__GL_SHADER_DISK_CACHE_SKIP_CLEANUP"] = "1"

    # === Omniverse Kit Cache (includes shader cache) ===
    ov_base = "/tmp/ov_cache"  # Don't inherit from env
    ov_rank_cache = os.path.join(ov_base, cache_suffix)
    os.makedirs(ov_rank_cache, exist_ok=True)
    os.environ["OMNI_KIT_CACHE_DIR"] = ov_rank_cache
    os.environ["OMNI_USER_CACHE_DIR"] = ov_rank_cache  # Also set user cache
    os.environ["CARB_DATA_PATH"] = os.path.join(ov_rank_cache, "carb")  # Carbonite data

    # Use print to ensure visibility in Ray logs
    print(f"[Stage {stage_id} Rank {actor_rank}] Cache directories configured:", flush=True)
    print(f"  OptiX:  {optix_rank_cache}", flush=True)
    print(f"  Shader: {shader_rank_cache}", flush=True)
    print(f"  OV Kit: {ov_rank_cache}", flush=True)


@ray.remote(num_gpus=1)
class IsaacSimActor:
    """
    Ray Actor for Isaac Lab multi-task simulation.

    This is a drop-in replacement for the ZMQ-based IsaacMultiTaskServer,
    but managed by Ray instead of running as an independent process.

    Key differences from ZMQ server:
    1. No ZMQ socket - uses Ray's actor methods directly
    2. GPU allocated by Ray scheduler (num_gpus=1)
    3. Data transferred via Ray's object store (automatic serialization)
    4. Lifecycle managed by Ray (no manual start/stop)

    Thread Safety:
        Ray actors are single-threaded by default, so all method calls
        are serialized. This matches the ZMQ REP socket behavior.
    """

    def __init__(
        self,
        env_id: str = "Isaac-Libero-Franka-OscPose-Camera-All-Tasks-v0",
        num_tasks: int = 5,
        group_size: int = 8,  # Fixed envs per task (same as ZMQ mode)
        actor_rank: int = 0,
        task_offset: int = 0,
        render_last_only: bool = True,
        camera_height: int = 256,
        camera_width: int = 256,
        stage_id: int = 0,
    ):
        """
        Initialize the Isaac Sim Actor.

        Args:
            env_id: Gymnasium environment ID for multi-task Isaac env
            num_tasks: Number of tasks this actor handles
            group_size: Number of parallel envs per task (fixed for all tasks)
            actor_rank: Rank of this actor (0 to num_actors-1)
            task_offset: Global task ID offset for this actor
            render_last_only: If True, only render on the last step of action chunks
            camera_height: Camera image height
            camera_width: Camera image width
            stage_id: Pipeline stage ID (for logging and identification)
        """
        self.env_id = env_id
        self.num_tasks = num_tasks
        self.group_size = group_size
        self.actor_rank = actor_rank
        self.task_offset = task_offset
        self.render_last_only = render_last_only
        self.camera_height = camera_height
        self.camera_width = camera_width
        self.stage_id = stage_id

        # Total envs = num_tasks * group_size (same as ZMQ mode)
        self.total_envs = num_tasks * group_size

        # Task ID to env indices mapping (local to this actor)
        # task_id here is LOCAL (0 to num_tasks-1)
        self.task_to_env_indices = {
            task_id: list(range(task_id * group_size, (task_id + 1) * group_size)) for task_id in range(num_tasks)
        }

        # Will be initialized in init_env()
        self.env = None
        self.simulation_app = None
        self.device = None
        self.action_dim = None
        self._initialized = False

        logger.info(
            f"[Stage {stage_id} Actor {actor_rank}] Created: "
            f"{num_tasks} tasks (offset={task_offset}), "
            f"group_size={group_size}, {self.total_envs} envs"
        )

    def init_env(self) -> dict:
        """
        Initialize the Isaac Lab environment.

        This is separated from __init__ because Isaac Lab initialization
        requires GPU context which may not be ready during actor creation.

        Returns:
            dict with status and environment info
        """
        if self._initialized:
            return {
                "status": "ok",
                "message": "Already initialized",
                "num_tasks": self.num_tasks,
                "group_size": self.group_size,
                "total_envs": self.total_envs,
            }

        import torch

        # Setup per-process caches (unique per stage + actor to avoid conflicts)
        setup_per_process_caches(self.actor_rank, self.stage_id)

        # Set environment variables for Isaac Lab config (same as ZMQ mode)
        # GROUP_SIZE = envs per task (not total envs!)
        os.environ["GROUP_SIZE"] = str(self.group_size)
        os.environ["TASK_OFFSET"] = str(self.task_offset)
        os.environ["NUM_TASKS"] = str(self.num_tasks)

        logger.info(f"[Stage {self.stage_id} Actor {self.actor_rank}] Initializing Isaac environment: {self.env_id}")

        # Detect GPU
        num_gpus = torch.cuda.device_count()
        # Ray assigns GPU via CUDA_VISIBLE_DEVICES, so device is always "cuda:0" from actor's view
        self.device = "cuda:0"

        logger.info(f"[Stage {self.stage_id} Actor {self.actor_rank}] Visible GPUs: {num_gpus}, using {self.device}")

        # Import Isaac Lab components - follow IsaacEnv pattern exactly
        import gymnasium as gym
        from isaaclab.app import AppLauncher

        # Use simple kwargs initialization (same as IsaacEnv)
        launch_args = {"headless": True, "enable_cameras": True}
        app_launcher = AppLauncher(**launch_args)
        self.simulation_app = app_launcher.app

        # Force franka registration (same as IsaacEnv)
        import isaaclab_playground.tasks.manipulation.libero.config.franka  # noqa

        # Now import Isaac Lab task utilities
        from isaaclab_tasks.utils import parse_env_cfg

        # Parse environment config (same as IsaacEnv._init_env)
        env_cfg = parse_env_cfg(self.env_id, num_envs=self.total_envs)

        # Configure environment (following IsaacEnv pattern exactly)
        env_cfg.env_name = f"stage{self.stage_id}_actor{self.actor_rank}"
        env_cfg.sim.device = self.device
        env_cfg.sim.physx.enable_ccd = True
        env_cfg.terminations.time_out = None
        env_cfg.observations.policy.concatenate_terms = False

        # Override camera dimensions if supported
        if hasattr(env_cfg, "camera_height"):
            env_cfg.camera_height = self.camera_height
            env_cfg.camera_width = self.camera_width
            if hasattr(env_cfg, "recreate_cameras"):
                env_cfg.recreate_cameras()
            logger.info(
                f"[Stage {self.stage_id} Actor {self.actor_rank}] Set camera: {self.camera_width}x{self.camera_height}"
            )

        # Set task offset if supported
        if hasattr(env_cfg, "task_offset"):
            env_cfg.task_offset = self.task_offset

        # Ensure correct num_envs
        if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "num_envs"):
            env_cfg.scene.num_envs = self.total_envs

        # Create environment (same as IsaacEnv)
        self.env = gym.make(self.env_id, cfg=env_cfg).unwrapped
        self.action_dim = self.env.action_space.shape[-1]

        # Verify envs created
        actual_num_envs = self.env.num_envs
        if actual_num_envs != self.total_envs:
            logger.warning(
                f"[Stage {self.stage_id} Actor {self.actor_rank}] "
                f"Env count mismatch: requested {self.total_envs}, got {actual_num_envs}"
            )
            self.total_envs = actual_num_envs
            # Rebuild task mapping
            self.group_size = actual_num_envs // self.num_tasks if self.num_tasks > 0 else actual_num_envs
            self.task_to_env_indices = {
                task_id: list(range(task_id * self.group_size, (task_id + 1) * self.group_size))
                for task_id in range(self.num_tasks)
            }

        # Initial reset
        self.env.reset()
        self._initialized = True

        logger.info(
            f"[Stage {self.stage_id} Actor {self.actor_rank}] "
            f"Initialized: {self.total_envs} envs, action_dim={self.action_dim}"
        )

        return {
            "status": "ok",
            "num_tasks": self.num_tasks,
            "group_size": self.group_size,
            "total_envs": self.total_envs,
            "action_dim": self.action_dim,
            "task_offset": self.task_offset,
            "actor_rank": self.actor_rank,
            "stage_id": self.stage_id,
        }

    def get_task_mapping(self) -> dict:
        """Return the task ID to env indices mapping."""
        return {
            "status": "ok",
            "task_to_env_indices": self.task_to_env_indices,
            "num_tasks": self.num_tasks,
            "group_size": self.group_size,
            "total_envs": self.total_envs,
            "task_offset": self.task_offset,
            "actor_rank": self.actor_rank,
            "stage_id": self.stage_id,
        }

    def step(self, actions: np.ndarray, env_indices: list) -> dict:
        """
        Execute step on specified environments.

        Args:
            actions: Actions array, shape (len(env_indices), action_dim) or
                     (len(env_indices), num_chunks, action_dim) for chunked actions
            env_indices: List of env indices to step

        Returns:
            dict with obs, rewards, terminations, truncations, infos
        """
        if not self._initialized:
            raise RuntimeError("Actor not initialized. Call init_env() first.")

        import torch

        actions = np.array(actions)

        logger.debug(
            f"[Stage {self.stage_id} Actor {self.actor_rank}] "
            f"Step: {len(env_indices)} envs, indices={env_indices[:5]}..."
        )

        # Check if actions have chunk dimension: [num_envs, num_chunks, action_dim]
        if len(actions.shape) == 3:
            return self._handle_chunk_step(actions, env_indices)

        # Single step: [num_envs, action_dim]
        full_actions = torch.zeros(self.total_envs, self.action_dim, device=self.device)
        full_actions[env_indices] = torch.tensor(actions, device=self.device, dtype=torch.float32)

        # Step all envs
        obs, rewards, terminations, truncations, infos = self.env.step(full_actions)

        # Extract results for requested env indices
        response = {
            "status": "ok",
            "obs": self._extract_obs(obs, env_indices),
            "rewards": rewards[env_indices].cpu().numpy(),
            "terminations": terminations[env_indices].cpu().numpy(),
            "truncations": truncations[env_indices].cpu().numpy(),
            "infos": self._extract_infos(infos, env_indices),
        }

        return response

    def _handle_chunk_step(self, chunk_actions: np.ndarray, env_indices: list) -> dict:
        """
        Handle action chunks: execute each chunk sequentially.

        Args:
            chunk_actions: [num_envs, num_chunks, action_dim]
            env_indices: list of env indices to step

        Returns:
            dict with accumulated rewards and final obs
        """
        import torch

        num_chunks = chunk_actions.shape[1]

        chunk_rewards = []
        chunk_terminations = []
        chunk_truncations = []

        # Save original render_interval to restore later
        original_render_interval = None
        if self.render_last_only and hasattr(self.env.unwrapped, "cfg"):
            original_render_interval = self.env.unwrapped.cfg.sim.render_interval

        for chunk_idx in range(num_chunks):
            is_last_chunk = chunk_idx == num_chunks - 1

            # Control rendering for efficiency
            if self.render_last_only and original_render_interval is not None:
                if is_last_chunk:
                    self.env.unwrapped.cfg.sim.render_interval = original_render_interval
                else:
                    self.env.unwrapped.cfg.sim.render_interval = 999999

            # Get actions for this chunk
            actions = chunk_actions[:, chunk_idx, :]

            # Build full action tensor
            full_actions = torch.zeros(self.total_envs, self.action_dim, device=self.device)
            full_actions[env_indices] = torch.tensor(actions, device=self.device, dtype=torch.float32)

            # Step all envs
            obs, rewards, terminations, truncations, infos = self.env.step(full_actions)

            # Collect results for this chunk
            chunk_rewards.append(rewards[env_indices].cpu().numpy())
            chunk_terminations.append(terminations[env_indices].cpu().numpy())
            chunk_truncations.append(truncations[env_indices].cpu().numpy())

        # Restore original render_interval
        if self.render_last_only and original_render_interval is not None:
            self.env.unwrapped.cfg.sim.render_interval = original_render_interval

        # Stack chunk results: [num_envs, num_chunks]
        stacked_rewards = np.stack(chunk_rewards, axis=1)
        stacked_terminations = np.stack(chunk_terminations, axis=1)
        stacked_truncations = np.stack(chunk_truncations, axis=1)

        return {
            "status": "ok",
            "obs": self._extract_obs(obs, env_indices),
            "rewards": stacked_rewards,
            "terminations": stacked_terminations,
            "truncations": stacked_truncations,
            "infos": self._extract_infos(infos, env_indices),
        }

    def reset(self, env_indices: Optional[list] = None, stabilize: bool = True) -> dict:
        """
        Reset specified environments.

        Args:
            env_indices: List of env indices to reset (None for all)
            stabilize: Whether to run stabilization steps (10 zero-action steps)

        Returns:
            dict with obs
        """
        if not self._initialized:
            raise RuntimeError("Actor not initialized. Call init_env() first.")

        import torch

        if env_indices is None:
            env_indices = list(range(self.total_envs))

        logger.debug(f"[Stage {self.stage_id} Actor {self.actor_rank}] Reset: {len(env_indices)} envs")

        # Validate env_indices
        actual_num_envs = self.env.unwrapped.num_envs
        max_idx = max(env_indices) if env_indices else 0
        if max_idx >= actual_num_envs:
            raise RuntimeError(f"env_indices out of bounds: max={max_idx}, but env only has {actual_num_envs} envs")

        # Use Isaac Lab's internal _reset_idx for partial reset
        reset_env_ids = torch.tensor(env_indices, device=self.device, dtype=torch.long)
        self.env.unwrapped._reset_idx(reset_env_ids)

        # Stabilize: run zero-action steps to let physics settle
        if stabilize:
            zero_actions = torch.zeros(self.total_envs, self.action_dim, device=self.device)
            for _ in range(10):
                obs, _, _, _, infos = self.env.step(zero_actions)
        else:
            obs = self.env.unwrapped.observation_manager.compute()

        return {
            "status": "ok",
            "obs": self._extract_obs(obs, env_indices),
        }

    def ping(self) -> dict:
        """Health check."""
        return {
            "status": "ok",
            "message": "pong",
            "initialized": self._initialized,
            "stage_id": self.stage_id,
            "actor_rank": self.actor_rank,
        }

    def close(self) -> dict:
        """Clean up resources."""
        logger.info(f"[Stage {self.stage_id} Actor {self.actor_rank}] Closing...")

        if self.env:
            self.env.close()
            self.env = None

        if self.simulation_app:
            self.simulation_app.close()
            self.simulation_app = None

        self._initialized = False

        return {"status": "ok", "message": "Actor closed"}

    def _to_cpu_numpy(self, value: Any) -> Any:
        """Recursively convert any CUDA tensors to CPU numpy arrays."""
        import torch

        if isinstance(value, torch.Tensor):
            return value.cpu().numpy()
        elif isinstance(value, dict):
            return {k: self._to_cpu_numpy(v) for k, v in value.items()}
        elif isinstance(value, list | tuple):
            return type(value)(self._to_cpu_numpy(v) for v in value)
        elif isinstance(value, np.ndarray):
            return value
        else:
            if hasattr(value, "cpu") and callable(value.cpu):
                try:
                    return value.cpu().numpy()
                except Exception:
                    pass
            return value

    def _extract_obs(self, obs: Any, env_indices: list) -> dict:
        """Extract observations for specified env indices."""
        import torch

        if isinstance(obs, dict):
            result = {}
            for key, value in obs.items():
                if isinstance(value, torch.Tensor):
                    result[key] = value[env_indices].cpu().numpy()
                elif isinstance(value, dict):
                    result[key] = self._extract_obs(value, env_indices)
                elif isinstance(value, np.ndarray):
                    result[key] = value[env_indices]
                else:
                    result[key] = self._to_cpu_numpy(value)
            return result
        elif isinstance(obs, torch.Tensor):
            return obs[env_indices].cpu().numpy()
        elif isinstance(obs, np.ndarray):
            return obs[env_indices]
        else:
            return self._to_cpu_numpy(obs)

    def _extract_infos(self, infos: dict, env_indices: list) -> dict:
        """Extract infos for specified env indices."""
        import torch

        result = {}
        if infos:
            for key, value in infos.items():
                if isinstance(value, torch.Tensor):
                    if value.dim() > 0 and value.shape[0] >= max(env_indices) + 1:
                        result[key] = value[env_indices].cpu().numpy()
                    else:
                        result[key] = value.cpu().numpy()
                elif isinstance(value, np.ndarray):
                    if value.ndim > 0 and value.shape[0] >= max(env_indices) + 1:
                        result[key] = value[env_indices]
                    else:
                        result[key] = value
                elif isinstance(value, dict):
                    result[key] = self._extract_infos(value, env_indices)
                else:
                    result[key] = self._to_cpu_numpy(value)
        return result
