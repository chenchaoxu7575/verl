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
Isaac Sim Actor Manager

Manages multiple IsaacSimActors across GPUs and pipeline stages.
This replaces IsaacMultiServerClient for the Ray-based architecture.

Architecture:
    ┌────────────────────────────────────────────────────────────────────┐
    │                 IsaacSimActorManager                               │
    │                                                                    │
    │  Stage 0 (actors[0]):          Stage 1 (actors[1]):               │
    │  ┌──────────────────────┐      ┌──────────────────────┐           │
    │  │ Actor 0 (GPU 0)      │      │ Actor 0 (GPU 0)      │           │
    │  │ Tasks: 0-4           │      │ Tasks: 0-4           │           │
    │  ├──────────────────────┤      ├──────────────────────┤           │
    │  │ Actor 1 (GPU 1)      │      │ Actor 1 (GPU 1)      │           │
    │  │ Tasks: 5-9           │      │ Tasks: 5-9           │           │
    │  ├──────────────────────┤      ├──────────────────────┤           │
    │  │ ...                  │      │ ...                  │           │
    │  └──────────────────────┘      └──────────────────────┘           │
    └────────────────────────────────────────────────────────────────────┘

Key Concepts:
    - Each stage has its own set of actors (physical isolation between stages)
    - Each actor handles a subset of tasks
    - All actors in a stage share the same task → actor mapping
    - Batched operations are parallelized via ray.get() on multiple actors

Usage:
    manager = IsaacSimActorManager(
        num_stages=2,
        num_actors_per_stage=8,
        num_tasks=40,
        group_size=8,
    )
    manager.initialize()  # Creates and initializes all actors

    # Step on specific stage
    result = manager.step_batched(server_requests, stage_id=0)

    # Reset all stages
    results = manager.reset_batched(server_requests, stage_id=0)
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import ray

from .ray_actor import IsaacSimActor

logger = logging.getLogger(__name__)


class IsaacSimActorManager:
    """
    Manager for multiple IsaacSimActors across stages and GPUs.

    This is the Ray-based replacement for IsaacMultiServerClient.

    Design:
    - Creates num_stages × num_actors_per_stage actors
    - Each stage is physically isolated (separate actor instances)
    - Provides batched step/reset operations for efficiency

    Thread Safety:
    - Manager methods are thread-safe for concurrent calls from different stages
    - Uses ThreadPoolExecutor for parallel ray.get() calls
    """

    def __init__(
        self,
        num_stages: int = 2,
        num_actors_per_stage: int = 8,
        num_tasks: int = 40,
        group_size: int = 8,
        env_id: str = "Isaac-Libero-Franka-OscPose-Camera-All-Tasks-v0",
        render_last_only: bool = True,
        camera_height: int = 256,
        camera_width: int = 256,
        placement_group=None,
        accelerator_type: Optional[str] = None,
        runtime_env: Optional[dict] = None,
    ):
        """
        Initialize the actor manager.

        Args:
            num_stages: Number of pipeline stages (each gets its own actor set)
            num_actors_per_stage: Number of actors per stage (typically = num GPUs)
            num_tasks: Total number of tasks (split across actors)
            group_size: Number of envs per task (fixed for all tasks)
            env_id: Isaac Lab environment ID
            render_last_only: Only render last step of action chunks
            camera_height: Camera image height
            camera_width: Camera image width
            placement_group: Optional Ray placement group for scheduling
            accelerator_type: Optional accelerator type label (e.g., "sim")
            runtime_env: Optional Ray runtime environment for actors
        """
        self.num_stages = num_stages
        self.num_actors_per_stage = num_actors_per_stage
        self.num_tasks = num_tasks
        self.group_size = group_size
        self.env_id = env_id
        self.render_last_only = render_last_only
        self.camera_height = camera_height
        self.camera_width = camera_width
        self.placement_group = placement_group
        self.accelerator_type = accelerator_type
        self.runtime_env = runtime_env

        # Total envs = num_tasks * group_size (same as ZMQ mode)
        self._total_envs = num_tasks * group_size

        # Calculate task distribution across actors (same as ZMQ mode)
        # Tasks are distributed evenly; some actors get +1 task if remainder > 0
        self.tasks_per_actor = num_tasks // num_actors_per_stage
        self.remainder_tasks = num_tasks % num_actors_per_stage

        # actors[stage_id][actor_rank] = IsaacSimActor handle
        self.actors: list[list[ray.ObjectRef]] = []

        # Global task mapping (same for all stages)
        # task_id → actor_rank
        self._task_to_actor: dict[int, int] = {}
        # task_id → local env indices on that actor
        self._task_to_env_indices: dict[int, list[int]] = {}
        # actor_rank → task_offset
        self._actor_task_offsets: list[int] = []

        self._initialized = False

        # Thread pool for parallel operations
        self._executor = ThreadPoolExecutor(max_workers=num_actors_per_stage)

        # Calculate GPU requirements
        # Each GPU is shared by num_stages actors (time-sharing across pipeline stages)
        # e.g., 2 stages → 2 actors per GPU → 0.5 GPU/actor
        actors_per_gpu = num_stages
        self.gpu_per_actor = 1.0 / actors_per_gpu
        total_gpus_needed = num_actors_per_stage  # One GPU per actor rank, shared by stages

        logger.info(
            f"IsaacSimActorManager created: "
            f"{num_stages} stages × {num_actors_per_stage} actors = {num_stages * num_actors_per_stage} total actors, "
            f"{num_tasks} tasks × {group_size} group_size = {self._total_envs} total_envs"
        )
        if self.remainder_tasks > 0:
            logger.info(
                f"Task distribution: actors 0-{self.remainder_tasks - 1} handle {self.tasks_per_actor + 1} tasks, "
                f"actors {self.remainder_tasks}-{num_actors_per_stage - 1} handle {self.tasks_per_actor} tasks"
            )
        logger.info(
            f"GPU allocation: {actors_per_gpu} actors/GPU, "
            f"{self.gpu_per_actor:.2f} GPU/actor, "
            f"total GPUs needed: {total_gpus_needed}"
        )

    def initialize(self) -> bool:
        """
        Create and initialize all actors.

        This creates num_stages × num_actors_per_stage actors,
        each running on its own GPU (managed by Ray).

        Returns:
            True if all actors initialized successfully
        """
        if self._initialized:
            logger.warning("Manager already initialized")
            return True

        logger.info("Creating IsaacSimActors...")

        # Build task distribution
        self._build_task_mapping()

        # Create actors for each stage
        for stage_id in range(self.num_stages):
            stage_actors = []
            for actor_rank in range(self.num_actors_per_stage):
                # Calculate tasks for this actor
                if actor_rank < self.remainder_tasks:
                    actor_num_tasks = self.tasks_per_actor + 1
                    task_offset = actor_rank * (self.tasks_per_actor + 1)
                else:
                    actor_num_tasks = self.tasks_per_actor
                    task_offset = (
                        self.remainder_tasks * (self.tasks_per_actor + 1)
                        + (actor_rank - self.remainder_tasks) * self.tasks_per_actor
                    )

                # Build actor options
                # Each actor gets 1/num_stages GPU so that all stages can time-share the same GPUs
                # Example: 2 stages × 8 actors/stage = 16 actors, each with 0.5 GPU = 8 GPUs total
                actor_options = {"num_gpus": self.gpu_per_actor}

                # Add runtime_env if provided (same as EnvWorker uses)
                if self.runtime_env is not None:
                    actor_options["runtime_env"] = self.runtime_env

                # Add placement group if provided
                if self.placement_group is not None:
                    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

                    bundle_idx = actor_rank % len(self.placement_group.bundle_specs)
                    actor_options["scheduling_strategy"] = PlacementGroupSchedulingStrategy(
                        placement_group=self.placement_group,
                        placement_group_bundle_index=bundle_idx,
                    )

                # Add accelerator type if specified
                if self.accelerator_type is not None:
                    actor_options["resources"] = {self.accelerator_type: 1e-4}

                # Actor's total envs = num_tasks * group_size
                actor_total_envs = actor_num_tasks * self.group_size

                # Create actor (same interface as ZMQ mode)
                actor = IsaacSimActor.options(**actor_options).remote(
                    env_id=self.env_id,
                    num_tasks=actor_num_tasks,
                    group_size=self.group_size,  # Fixed envs per task
                    actor_rank=actor_rank,
                    task_offset=task_offset,
                    render_last_only=self.render_last_only,
                    camera_height=self.camera_height,
                    camera_width=self.camera_width,
                    stage_id=stage_id,
                )
                stage_actors.append(actor)

                logger.info(
                    f"Created actor: stage={stage_id}, rank={actor_rank}, "
                    f"tasks={task_offset}-{task_offset + actor_num_tasks - 1}, "
                    f"envs={actor_total_envs}, gpu={self.gpu_per_actor:.2f}"
                )

            self.actors.append(stage_actors)

        # Initialize all actors in parallel
        logger.info("Initializing all actors...")
        init_futures = []
        for stage_id, stage_actors in enumerate(self.actors):
            for actor in stage_actors:
                init_futures.append(actor.init_env.remote())

        # Wait for all initializations
        try:
            results = ray.get(init_futures)
            for result in results:
                if result.get("status") != "ok":
                    logger.error(f"Actor initialization failed: {result}")
                    return False
        except Exception as e:
            logger.error(f"Actor initialization error: {e}")
            return False

        # Verify actors are initialized
        ray.get(self.actors[0][0].get_task_mapping.remote())

        self._initialized = True
        logger.info(
            f"All actors initialized: "
            f"{self.num_stages * self.num_actors_per_stage} actors, "
            f"{self._total_envs} envs per stage"
        )

        return True

    def _build_task_mapping(self):
        """Build global task → actor mapping (same logic as ZMQ mode)."""
        self._task_to_actor.clear()
        self._task_to_env_indices.clear()
        self._actor_task_offsets.clear()

        for actor_rank in range(self.num_actors_per_stage):
            # Calculate tasks for this actor (same as ZMQ mode)
            if actor_rank < self.remainder_tasks:
                actor_num_tasks = self.tasks_per_actor + 1
                task_offset = actor_rank * (self.tasks_per_actor + 1)
            else:
                actor_num_tasks = self.tasks_per_actor
                task_offset = (
                    self.remainder_tasks * (self.tasks_per_actor + 1)
                    + (actor_rank - self.remainder_tasks) * self.tasks_per_actor
                )

            self._actor_task_offsets.append(task_offset)

            # Map global task_ids to this actor
            # local_task_id is 0-based within the actor
            for local_task_id in range(actor_num_tasks):
                global_task_id = task_offset + local_task_id
                self._task_to_actor[global_task_id] = actor_rank

                # Local env indices for this task on the actor
                # Each task has group_size envs, indexed locally within the actor
                self._task_to_env_indices[global_task_id] = list(
                    range(local_task_id * self.group_size, (local_task_id + 1) * self.group_size)
                )

        logger.debug(f"Task mapping built: {self.num_tasks} tasks across {self.num_actors_per_stage} actors")

    def get_actor_for_task(self, global_task_id: int) -> int:
        """Get the actor rank that handles a given global task ID."""
        return self._task_to_actor.get(global_task_id, 0)

    def get_env_indices_for_task(self, global_task_id: int) -> list:
        """
        Get LOCAL env indices for a given global task ID.

        These are indices local to the actor that handles this task.
        """
        return self._task_to_env_indices.get(global_task_id, [])

    def step(
        self,
        actions: np.ndarray,
        env_indices: list,
        actor_rank: int,
        stage_id: int,
        render_last_only: bool = True,
    ) -> dict:
        """
        Send step command to a specific actor.

        Args:
            actions: Actions array
            env_indices: LOCAL env indices on that actor
            actor_rank: Which actor
            stage_id: Which pipeline stage
            render_last_only: Only render last step of action chunks

        Returns:
            Response dict
        """
        if not self._initialized:
            raise RuntimeError("Manager not initialized. Call initialize() first.")

        if stage_id >= len(self.actors) or actor_rank >= len(self.actors[stage_id]):
            raise ValueError(f"Invalid stage_id={stage_id} or actor_rank={actor_rank}")

        return ray.get(self.actors[stage_id][actor_rank].step.remote(actions, env_indices))

    def reset(
        self,
        env_indices: list,
        actor_rank: int,
        stage_id: int,
        stabilize: bool = True,
    ) -> dict:
        """
        Send reset command to a specific actor.

        Args:
            env_indices: LOCAL env indices on that actor
            actor_rank: Which actor
            stage_id: Which pipeline stage
            stabilize: Whether to run stabilization steps

        Returns:
            Response dict
        """
        if not self._initialized:
            raise RuntimeError("Manager not initialized. Call initialize() first.")

        if stage_id >= len(self.actors) or actor_rank >= len(self.actors[stage_id]):
            raise ValueError(f"Invalid stage_id={stage_id} or actor_rank={actor_rank}")

        return ray.get(self.actors[stage_id][actor_rank].reset.remote(env_indices, stabilize))

    def step_batched(
        self,
        actor_requests: dict[int, tuple[np.ndarray, list]],
        stage_id: int,
        render_last_only: bool = True,
    ) -> dict[int, dict]:
        """
        Send step commands to multiple actors CONCURRENTLY.

        Uses ray.get() for parallel execution across actors.

        Args:
            actor_requests: Dict mapping actor_rank -> (actions, env_indices)
                e.g., {0: (actions_0, [0,1,2]), 2: (actions_2, [8,9,10])}
            stage_id: Which pipeline stage
            render_last_only: Only render last step of action chunks

        Returns:
            Dict mapping actor_rank -> response dict
        """
        if not self._initialized:
            raise RuntimeError("Manager not initialized. Call initialize() first.")

        # Submit all step calls
        futures = {}
        for actor_rank, (actions, indices) in actor_requests.items():
            if actor_rank >= len(self.actors[stage_id]):
                logger.error(f"Invalid actor_rank={actor_rank}")
                continue
            future = self.actors[stage_id][actor_rank].step.remote(actions, indices)
            futures[actor_rank] = future

        # Wait for all results
        results = {}
        for actor_rank, future in futures.items():
            try:
                results[actor_rank] = ray.get(future)
            except Exception as e:
                logger.error(f"Step failed on actor {actor_rank}: {e}")
                results[actor_rank] = {"status": "error", "message": str(e)}

        return results

    def reset_batched(
        self,
        actor_requests: dict[int, list],
        stage_id: int,
        stabilize: bool = True,
    ) -> dict[int, dict]:
        """
        Send reset commands to multiple actors CONCURRENTLY.

        Args:
            actor_requests: Dict mapping actor_rank -> env_indices
                e.g., {0: [0,1,2,3], 2: [8,9,10,11]}
            stage_id: Which pipeline stage
            stabilize: Whether to run stabilization steps

        Returns:
            Dict mapping actor_rank -> response dict
        """
        if not self._initialized:
            raise RuntimeError("Manager not initialized. Call initialize() first.")

        # Submit all reset calls
        futures = {}
        for actor_rank, indices in actor_requests.items():
            if actor_rank >= len(self.actors[stage_id]):
                logger.error(f"Invalid actor_rank={actor_rank}")
                continue
            future = self.actors[stage_id][actor_rank].reset.remote(indices, stabilize)
            futures[actor_rank] = future

        # Wait for all results
        results = {}
        for actor_rank, future in futures.items():
            try:
                results[actor_rank] = ray.get(future)
            except Exception as e:
                logger.error(f"Reset failed on actor {actor_rank}: {e}")
                results[actor_rank] = {"status": "error", "message": str(e)}

        return results

    def reset_all_stages_batched(
        self,
        actor_requests: dict[int, list],
        stabilize: bool = True,
    ) -> dict[int, dict[int, dict]]:
        """
        Send reset commands to all stages with the same actor_requests.

        This is useful for initial reset where all stages need the same env states.

        Args:
            actor_requests: Dict mapping actor_rank -> env_indices
            stabilize: Whether to run stabilization steps

        Returns:
            Dict mapping stage_id -> (actor_rank -> response dict)
        """
        if not self._initialized:
            raise RuntimeError("Manager not initialized. Call initialize() first.")

        # Submit all reset calls for all stages
        futures_by_stage = {}
        for stage_id in range(self.num_stages):
            futures = {}
            for actor_rank, indices in actor_requests.items():
                if actor_rank >= len(self.actors[stage_id]):
                    continue
                future = self.actors[stage_id][actor_rank].reset.remote(indices, stabilize)
                futures[actor_rank] = future
            futures_by_stage[stage_id] = futures

        # Wait for all results
        results = {}
        for stage_id, futures in futures_by_stage.items():
            stage_results = {}
            for actor_rank, future in futures.items():
                try:
                    stage_results[actor_rank] = ray.get(future)
                except Exception as e:
                    logger.error(f"Reset failed on stage {stage_id} actor {actor_rank}: {e}")
                    stage_results[actor_rank] = {"status": "error", "message": str(e)}
            results[stage_id] = stage_results

        return results

    @property
    def num_tasks(self) -> int:
        """Get total number of tasks."""
        return self._num_tasks

    @num_tasks.setter
    def num_tasks(self, value: int):
        self._num_tasks = value

    @property
    def total_envs(self) -> int:
        """Get total number of envs per stage."""
        return self._total_envs

    @property
    def initialized(self) -> bool:
        """Check if manager is initialized."""
        return self._initialized

    def close(self):
        """Close all actors and clean up resources."""
        logger.info("Closing all actors...")

        # Close thread pool
        self._executor.shutdown(wait=True)

        # Close all actors
        close_futures = []
        for stage_actors in self.actors:
            for actor in stage_actors:
                close_futures.append(actor.close.remote())

        # Wait for all closes
        if close_futures:
            try:
                ray.get(close_futures)
            except Exception as e:
                logger.warning(f"Error closing actors: {e}")

        # Kill actors
        for stage_actors in self.actors:
            for actor in stage_actors:
                try:
                    ray.kill(actor)
                except Exception:
                    pass

        self.actors = []
        self._initialized = False

        logger.info("All actors closed")

    def __del__(self):
        """Clean up on deletion."""
        if self._initialized:
            self.close()
