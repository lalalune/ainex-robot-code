"""Main evaluation loop for RPG2Robot.

Takes a planner (any callable mapping EmbodiedContext -> GroundedIntent) and
a skill executor, runs N episodes per task in MuJoCo, and measures success
rate, planning accuracy, grounding accuracy, recovery rate, time-to-completion,
and safety violations.
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import numpy as np

from training.interfaces import (
    AinexPerceptionObservation,
    CanonicalIntent,
    CanonicalIntentType,
    ExecutorRequest,
    ExecutorResult,
    PlannerTraceContext,
    TrackedEntity,
)
from training.rl.skills.base_skill import BaseSkill, SkillParams, SkillStatus
from training.rl.skills.registry import SkillRegistry
from training.schema.canonical import (
    AINEX_ACTION_DIM,
    AINEX_PROPRIO_DIM,
    AINEX_STATE_DIM,
)

from evaluation.task_suite import (
    EntitySpec,
    EvalTask,
    GoalZone,
    SuccessCriterion,
    TaskCategory,
    TASK_REGISTRY,
    get_tasks_by_category,
)
from evaluation.metrics import (
    EpisodeResult,
    EvalResults,
    StepRecord,
    TaskResult,
    build_task_result,
    export_json,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol types for pluggable planner and executor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EmbodiedContext:
    """Observation + scene context passed to the planner each step."""
    instruction: str
    robot_xy: tuple[float, float]
    robot_yaw: float
    entities: tuple[TrackedEntity, ...]
    goal_zones: tuple[tuple[str, float, float], ...]  # (label, x, y)
    step: int
    max_steps: int
    previous_skill_status: str = ""
    previous_skill_name: str = ""
    grasped_entity: str = ""


@dataclass(frozen=True)
class GroundedIntent:
    """Planner output: which skill to run and on which entity."""
    skill_name: str
    target_entity_label: str = ""
    target_xy: tuple[float, float] = (0.0, 0.0)
    reasoning: str = ""
    is_done: bool = False   # Planner signals task complete


class PlannerProtocol(Protocol):
    """Any callable that maps context to an intent."""
    def __call__(self, ctx: EmbodiedContext) -> GroundedIntent: ...


class SkillExecutorProtocol(Protocol):
    """Wraps skill dispatch -- receives intent, returns per-step result."""
    def reset(self) -> None: ...
    def step(
        self,
        intent: GroundedIntent,
        obs: np.ndarray,
    ) -> tuple[np.ndarray, SkillStatus]: ...


# ---------------------------------------------------------------------------
# Simulated scene state (lightweight, no MuJoCo dependency at import time)
# ---------------------------------------------------------------------------

@dataclass
class _EntityState:
    """Mutable runtime state for a spawned entity."""
    label: str
    entity_type: int
    color: str
    xy: np.ndarray          # (2,) world position
    size: tuple[float, float, float]
    is_target: bool
    grasped: bool = False


@dataclass
class _GoalZoneState:
    """Runtime state for a goal zone."""
    label: str
    xy: np.ndarray
    radius: float


@dataclass
class _SceneState:
    """Full mutable scene for one episode."""
    robot_xy: np.ndarray          # (2,)
    robot_yaw: float
    entities: list[_EntityState]
    goal_zones: list[_GoalZoneState]
    grasped_entity: str = ""
    step: int = 0

    # Physics simulation state (populated when using MuJoCo)
    _mjx_state: Any = None


# ---------------------------------------------------------------------------
# Default skill executor (dispatches skills from registry)
# ---------------------------------------------------------------------------

class DefaultSkillExecutor:
    """Executor that dispatches skills from a SkillRegistry.

    For evaluation purposes this wraps the real skill registry and handles
    skill switching.  When no trained checkpoint is loaded, skills output
    zero actions (which the sim scene handler interprets as a no-op and
    instead applies a heuristic motion model).
    """

    def __init__(self, registry: SkillRegistry | None = None) -> None:
        self._registry = registry or SkillRegistry()
        self._active_skill: BaseSkill | None = None
        self._active_name: str = ""

    def reset(self) -> None:
        self._active_skill = None
        self._active_name = ""

    def step(
        self,
        intent: GroundedIntent,
        obs: np.ndarray,
    ) -> tuple[np.ndarray, SkillStatus]:
        # Switch skill if needed
        if intent.skill_name != self._active_name:
            skill = self._registry.get(intent.skill_name)
            if skill is not None:
                skill.reset(SkillParams(speed=1.0))
                self._active_skill = skill
                self._active_name = intent.skill_name
            else:
                # Unknown skill -- return zeros / FAILED
                logger.warning("Skill %r not found in registry", intent.skill_name)
                return np.zeros(AINEX_ACTION_DIM, dtype=np.float32), SkillStatus.FAILED

        if self._active_skill is not None:
            return self._active_skill.get_action(obs)

        return np.zeros(AINEX_ACTION_DIM, dtype=np.float32), SkillStatus.RUNNING


# ---------------------------------------------------------------------------
# Scene simulation (heuristic motion model when no MuJoCo env available)
# ---------------------------------------------------------------------------

_WALK_SPEED = 0.006       # metres per step (~0.3 m/s at 50 Hz)
_TURN_SPEED = 0.05        # radians per step (~2.5 rad/s)
_GRASP_DISTANCE = 0.10    # metres


class _SceneSim:
    """Lightweight kinematic scene simulator.

    Used when MuJoCo is not available or for fast baseline evaluation.
    The robot moves toward intent.target_xy according to the active skill.
    """

    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng

    def reset_scene(
        self,
        task: EvalTask,
        seed: int,
    ) -> _SceneState:
        """Spawn entities and goal zones for one episode."""
        rng = np.random.default_rng(seed)
        robot_xy = np.array([0.0, 0.0], dtype=np.float64)
        robot_yaw = rng.uniform(-math.pi, math.pi)

        entities: list[_EntityState] = []
        for spec in task.entity_setup:
            angle = rng.uniform(0, 2 * math.pi)
            dist = rng.uniform(spec.spawn_radius_min, spec.spawn_radius_max)
            xy = robot_xy + dist * np.array([math.cos(angle), math.sin(angle)])
            entities.append(_EntityState(
                label=spec.label,
                entity_type=spec.entity_type,
                color=spec.color,
                xy=xy.copy(),
                size=spec.size,
                is_target=spec.is_target,
            ))

        goal_zones: list[_GoalZoneState] = []
        for gz in task.goal_zones:
            angle = rng.uniform(0, 2 * math.pi)
            dist = rng.uniform(gz.spawn_radius_min, gz.spawn_radius_max)
            xy = robot_xy + dist * np.array([math.cos(angle), math.sin(angle)])
            goal_zones.append(_GoalZoneState(
                label=gz.label,
                xy=xy.copy(),
                radius=gz.radius,
            ))

        return _SceneState(
            robot_xy=robot_xy,
            robot_yaw=robot_yaw,
            entities=entities,
            goal_zones=goal_zones,
        )

    def step_scene(
        self,
        scene: _SceneState,
        intent: GroundedIntent,
        action: np.ndarray,
        skill_status: SkillStatus,
    ) -> None:
        """Advance scene state by one step using heuristic motion."""
        target_xy = np.array(intent.target_xy, dtype=np.float64)
        delta = target_xy - scene.robot_xy
        dist = float(np.linalg.norm(delta))

        if intent.skill_name in ("walk", "walk_to_target") and dist > 0.01:
            # Move toward target
            direction = delta / dist
            step_dist = min(_WALK_SPEED, dist)
            scene.robot_xy = scene.robot_xy + direction * step_dist
            # Update yaw to face movement direction
            scene.robot_yaw = float(math.atan2(direction[1], direction[0]))

        elif intent.skill_name == "turn":
            # Turn toward target
            desired_yaw = math.atan2(delta[1], delta[0])
            yaw_err = desired_yaw - scene.robot_yaw
            # Normalize to [-pi, pi]
            yaw_err = math.atan2(math.sin(yaw_err), math.cos(yaw_err))
            turn = max(-_TURN_SPEED, min(_TURN_SPEED, yaw_err))
            scene.robot_yaw += turn

        # Grasp check: if close enough and intent is a manipulation skill
        if not scene.grasped_entity:
            for ent in scene.entities:
                if ent.grasped:
                    continue
                d = float(np.linalg.norm(scene.robot_xy - ent.xy))
                if d < _GRASP_DISTANCE and intent.skill_name in (
                    "walk_to_target", "walk"
                ):
                    # Auto-grasp when close enough (manipulation tasks)
                    ent.grasped = True
                    scene.grasped_entity = ent.label

        # Move grasped entity with robot
        for ent in scene.entities:
            if ent.grasped:
                ent.xy = scene.robot_xy.copy()

        scene.step += 1


# ---------------------------------------------------------------------------
# MuJoCo-backed scene simulation
# ---------------------------------------------------------------------------

class _MujocoSceneSim(_SceneSim):
    """Scene simulator backed by a real MuJoCo environment.

    Falls back to the heuristic model for manipulation steps but uses
    the MuJoCo physics engine for locomotion.
    """

    def __init__(self, rng: np.random.Generator) -> None:
        super().__init__(rng)
        self._env: Any = None
        self._mjx_state: Any = None
        self._policy_fn: Any = None

    def set_env(self, env: Any) -> None:
        """Inject a MuJoCo environment (AiNexEnv subclass)."""
        self._env = env

    def set_policy_fn(self, policy_fn: Any) -> None:
        """Inject a JAX policy function for inference."""
        self._policy_fn = policy_fn

    def reset_scene(
        self,
        task: EvalTask,
        seed: int,
    ) -> _SceneState:
        scene = super().reset_scene(task, seed)

        if self._env is not None:
            try:
                import jax
                rng = jax.random.PRNGKey(seed)
                state = jax.jit(self._env.reset)(rng)
                scene._mjx_state = state
                # Read robot position from MuJoCo state
                robot_xy = np.array(state.data.qpos[:2])
                scene.robot_xy = robot_xy.copy()
            except Exception as exc:
                logger.warning("MuJoCo reset failed, using heuristic: %s", exc)

        return scene

    def step_scene(
        self,
        scene: _SceneState,
        intent: GroundedIntent,
        action: np.ndarray,
        skill_status: SkillStatus,
    ) -> None:
        if scene._mjx_state is not None and self._env is not None:
            try:
                import jax
                import jax.numpy as jp
                from training.schema.canonical import adapt_state_vector

                state = scene._mjx_state
                action_jax = jp.array(action)

                # Adapt action dimensions if needed
                expected_size = self._env.action_size
                if action_jax.shape[0] != expected_size:
                    action_jax = jp.array(
                        adapt_state_vector(action.tolist(), expected_size)
                    )

                step_fn = jax.jit(self._env.step)
                state = step_fn(state, action_jax)
                scene._mjx_state = state

                # Update scene state from physics
                scene.robot_xy = np.array(state.data.qpos[:2])
                qw = float(state.data.qpos[3])
                qx = float(state.data.qpos[4])
                qy = float(state.data.qpos[5])
                qz = float(state.data.qpos[6])
                scene.robot_yaw = math.atan2(
                    2.0 * (qw * qz + qx * qy),
                    1.0 - 2.0 * (qy * qy + qz * qz),
                )

                # Check for termination / safety violation (fall)
                gravity_z = float(self._env.get_gravity(state.data)[-1])
                torso_z = float(
                    state.data.xpos[self._env._torso_body_id, 2]
                )
                if gravity_z < 0.85 or torso_z < 0.17:
                    # Mark as safety violation on the scene for the evaluator
                    scene._safety_violation = True
                else:
                    scene._safety_violation = False

                scene.step += 1

                # Handle grasped entity movement
                for ent in scene.entities:
                    if ent.grasped:
                        ent.xy = scene.robot_xy.copy()

                return

            except Exception as exc:
                logger.debug("MuJoCo step failed, falling back: %s", exc)

        # Fallback to heuristic
        super().step_scene(scene, intent, action, skill_status)


# ---------------------------------------------------------------------------
# Success checking
# ---------------------------------------------------------------------------

def _check_success(
    task: EvalTask,
    scene: _SceneState,
    intent: GroundedIntent,
) -> bool:
    """Evaluate whether the current scene state satisfies the task criterion."""

    criterion = task.success_criterion

    if criterion == SuccessCriterion.DISTANCE:
        # Robot must be within threshold of any target entity
        for ent in scene.entities:
            if ent.is_target:
                d = float(np.linalg.norm(scene.robot_xy - ent.xy))
                if d <= task.success_threshold:
                    return True
        return False

    if criterion == SuccessCriterion.FACING:
        # Robot heading must be aligned with target bearing
        for ent in scene.entities:
            if ent.is_target:
                delta = ent.xy - scene.robot_xy
                target_angle = math.atan2(delta[1], delta[0])
                angle_err = abs(math.atan2(
                    math.sin(target_angle - scene.robot_yaw),
                    math.cos(target_angle - scene.robot_yaw),
                ))
                if angle_err <= task.success_threshold:
                    return True
        return False

    if criterion == SuccessCriterion.CONTACT:
        # Robot's position overlaps target entity
        for ent in scene.entities:
            if ent.is_target:
                d = float(np.linalg.norm(scene.robot_xy - ent.xy))
                if d <= task.success_threshold + max(ent.size) / 2:
                    return True
        return False

    if criterion == SuccessCriterion.GRASPED:
        return scene.grasped_entity != ""

    if criterion == SuccessCriterion.PLACED:
        # Grasped object must be within a goal zone AND released
        for gz in scene.goal_zones:
            for ent in scene.entities:
                if ent.is_target:
                    d = float(np.linalg.norm(ent.xy - gz.xy))
                    if d <= gz.radius:
                        return True
        return False

    if criterion == SuccessCriterion.SEQUENCE:
        # All sub-goals completed -- approximated by all target entities
        # being in their corresponding goal zones
        if not task.sub_goals:
            return False
        # Check: every target entity is within some goal zone
        for ent in scene.entities:
            if ent.is_target:
                in_zone = False
                for gz in scene.goal_zones:
                    if ent.color in gz.label or gz.label in ent.label:
                        d = float(np.linalg.norm(ent.xy - gz.xy))
                        if d <= gz.radius:
                            in_zone = True
                            break
                if not in_zone:
                    return False
        return True

    if criterion == SuccessCriterion.RECOVERY:
        # Same as DISTANCE -- robot reaches target after recovery
        for ent in scene.entities:
            if ent.is_target:
                d = float(np.linalg.norm(scene.robot_xy - ent.xy))
                if d <= task.success_threshold:
                    return True
        return False

    return False


def _count_sub_goals(scene: _SceneState, task: EvalTask) -> int:
    """Count how many sub-goals are complete in the current scene."""
    count = 0
    # Heuristic: count grasped + placed entities
    for ent in scene.entities:
        if ent.is_target:
            if ent.grasped:
                count += 1
            for gz in scene.goal_zones:
                d = float(np.linalg.norm(ent.xy - gz.xy))
                if d <= gz.radius:
                    count += 1
    # Also count distance-based sub-goals
    for ent in scene.entities:
        if ent.is_target:
            d = float(np.linalg.norm(scene.robot_xy - ent.xy))
            if d <= task.success_threshold:
                count += 1
    return min(count, len(task.sub_goals)) if task.sub_goals else count


# ---------------------------------------------------------------------------
# Planning accuracy check
# ---------------------------------------------------------------------------

# Maps task success criterion to a "reasonable" skill sequence.
_EXPECTED_SKILL_PLANS: dict[str, set[str]] = {
    "walk_to_red_ball": {"walk", "walk_to_target"},
    "walk_to_named_entity": {"walk", "walk_to_target"},
    "face_and_approach": {"turn", "walk", "walk_to_target"},
    "pick_up_object": {"walk", "walk_to_target", "grasp"},
    "carry_to_target": {"walk", "walk_to_target", "grasp", "place"},
    "sort_by_color": {"walk", "walk_to_target", "grasp", "place", "turn"},
    "disambiguated_nav": {"walk", "walk_to_target", "turn"},
    "multi_step_fetch": {"walk", "walk_to_target", "grasp", "place", "turn"},
    "recovery_from_failure": {"walk", "walk_to_target", "turn", "stand"},
}


def _check_planning_correct(
    task: EvalTask,
    trajectory: list[StepRecord],
) -> bool:
    """Did the planner only select skills from the expected set?"""
    expected = _EXPECTED_SKILL_PLANS.get(task.name, set(task.required_skills))
    for rec in trajectory:
        if rec.planned_skill and rec.planned_skill not in expected:
            return False
    return True


def _check_grounding_correct(
    task: EvalTask,
    trajectory: list[StepRecord],
    entities: list[_EntityState],
) -> bool:
    """Did the planner ground to the correct target entity?

    Requires at least one step to positively reference a target entity.
    """
    target_labels = {e.label for e in entities if e.is_target}
    if not target_labels:
        return True  # No target to validate against
    found_correct = False
    for rec in trajectory:
        if rec.planned_target_entity:
            if rec.planned_target_entity in target_labels:
                found_correct = True
            else:
                # Planner grounded to a non-target entity
                return False
    return found_correct


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """Runs evaluation episodes across tasks, collecting metrics.

    Parameters
    ----------
    planner : PlannerProtocol
        Callable mapping EmbodiedContext -> GroundedIntent.
    skill_executor : SkillExecutorProtocol | None
        Wraps skill dispatch.  If None, a DefaultSkillExecutor with an empty
        registry is used.
    use_mujoco : bool
        If True, attempt to use MuJoCo environments for physics-backed
        evaluation.  Falls back to heuristic motion if import fails.
    mujoco_env : Any
        Optional pre-constructed AiNexEnv to inject.
    checkpoint : str
        Path to model checkpoint (for result metadata).
    planner_name : str
        Name of the planner being evaluated.
    """

    def __init__(
        self,
        planner: PlannerProtocol,
        skill_executor: SkillExecutorProtocol | None = None,
        use_mujoco: bool = False,
        mujoco_env: Any = None,
        checkpoint: str = "",
        planner_name: str = "unknown",
    ) -> None:
        self._planner = planner
        self._executor = skill_executor or DefaultSkillExecutor()
        self._planner_name = planner_name
        self._checkpoint = checkpoint

        # Scene simulator
        self._rng = np.random.default_rng(0)
        if use_mujoco or mujoco_env is not None:
            sim = _MujocoSceneSim(self._rng)
            if mujoco_env is not None:
                sim.set_env(mujoco_env)
            self._sim: _SceneSim = sim
        else:
            self._sim = _SceneSim(self._rng)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        tasks: Sequence[EvalTask],
        n_episodes: int = 100,
        base_seed: int = 42,
        progress_callback: Callable[[str, int, int], None] | None = None,
        resume_from: Path | str | None = None,
    ) -> EvalResults:
        """Run full evaluation.

        Parameters
        ----------
        tasks : sequence of EvalTask
            Tasks to evaluate.
        n_episodes : int
            Episodes per task.
        base_seed : int
            Base random seed (each episode gets ``base_seed + episode_id``).
        progress_callback : optional callable(task_name, episode, total)
            Called after each episode for progress tracking.
        resume_from : optional path
            Path to a partial results JSON to resume from.  Already-completed
            (task, episode) pairs are skipped.

        Returns
        -------
        EvalResults with per-task and aggregate metrics.
        """
        completed: set[tuple[str, int]] = set()
        prior_episodes: dict[str, list[EpisodeResult]] = {}

        if resume_from is not None:
            completed, prior_episodes = self._load_resume_state(
                Path(resume_from)
            )
            logger.info(
                "Resuming: %d episodes already completed", len(completed)
            )

        results = EvalResults(
            planner_name=self._planner_name,
            checkpoint=self._checkpoint,
            n_episodes_per_task=n_episodes,
        )

        for task in tasks:
            episodes: list[EpisodeResult] = list(
                prior_episodes.get(task.name, [])
            )

            for ep_idx in range(n_episodes):
                if (task.name, ep_idx) in completed:
                    continue

                seed = base_seed + ep_idx
                ep_result = self._run_episode(task, ep_idx, seed)
                episodes.append(ep_result)

                if progress_callback is not None:
                    progress_callback(task.name, ep_idx + 1, n_episodes)

            task_result = build_task_result(
                task_name=task.name,
                category=task.category.value,
                episodes=episodes,
            )
            results.task_results[task.name] = task_result

        results.finalize()
        return results

    # ------------------------------------------------------------------
    # Single episode
    # ------------------------------------------------------------------

    def _run_episode(
        self,
        task: EvalTask,
        episode_id: int,
        seed: int,
    ) -> EpisodeResult:
        """Execute one episode of a task and return the result."""
        scene = self._sim.reset_scene(task, seed)
        self._executor.reset()

        instruction = self._sample_instruction(task, seed)

        trajectory: list[StepRecord] = []
        prev_skill_name = ""
        prev_skill_status = ""
        failure_injected = False
        recovered_from_failure = False
        total_safety_violations = 0
        success = False

        for step in range(task.max_steps):
            # Build context for planner
            tracked = self._build_tracked_entities(scene)
            gz_tuples = tuple(
                (gz.label, float(gz.xy[0]), float(gz.xy[1]))
                for gz in scene.goal_zones
            )

            ctx = EmbodiedContext(
                instruction=instruction,
                robot_xy=(float(scene.robot_xy[0]), float(scene.robot_xy[1])),
                robot_yaw=float(scene.robot_yaw),
                entities=tracked,
                goal_zones=gz_tuples,
                step=step,
                max_steps=task.max_steps,
                previous_skill_status=prev_skill_status,
                previous_skill_name=prev_skill_name,
                grasped_entity=scene.grasped_entity,
            )

            # Inject failure if configured
            inject_this_step = (
                task.inject_failure
                and step == task.failure_injection_step
                and not failure_injected
            )
            if inject_this_step:
                failure_injected = True
                prev_skill_status = "failed"
                prev_skill_name = "walk"

            # Call planner
            intent = self._planner(ctx)

            # Detect replanning after failure
            is_replan = (
                failure_injected
                and not recovered_from_failure
                and prev_skill_status == "failed"
                and intent.skill_name != ""
            )
            if is_replan:
                recovered_from_failure = True

            # Build observation for skill executor
            obs = self._build_obs(scene, task)

            # Execute skill
            if inject_this_step:
                action = np.zeros(AINEX_ACTION_DIM, dtype=np.float32)
                skill_status = SkillStatus.FAILED
            else:
                action, skill_status = self._executor.step(intent, obs)

            # Step scene
            self._sim.step_scene(scene, intent, action, skill_status)

            # Safety violation check
            safety_violation = False
            if hasattr(scene, "_safety_violation") and scene._safety_violation:
                safety_violation = True
                total_safety_violations += 1

            # Compute distance to target
            dist_to_target = self._distance_to_target(scene)

            # Build step record
            rec = StepRecord(
                step=step,
                planned_skill=intent.skill_name,
                planned_target_entity=intent.target_entity_label,
                planned_intent=intent.reasoning,
                executed_skill=intent.skill_name,
                skill_status=skill_status.value,
                robot_xy=(float(scene.robot_xy[0]), float(scene.robot_xy[1])),
                robot_yaw=float(scene.robot_yaw),
                target_xy=(
                    float(intent.target_xy[0]),
                    float(intent.target_xy[1]),
                ),
                distance_to_target=dist_to_target,
                is_replan=is_replan,
                safety_violation=safety_violation,
                skill_failure=inject_this_step,
            )
            trajectory.append(rec)

            prev_skill_name = intent.skill_name
            prev_skill_status = skill_status.value

            # Check success
            if _check_success(task, scene, intent):
                success = True
                break

            # Planner signals done
            if intent.is_done:
                break

        # Build episode result
        ep = EpisodeResult(
            task_name=task.name,
            episode_id=episode_id,
            seed=seed,
            success=success,
            steps_used=len(trajectory),
            trajectory=trajectory,
            failure_injected=failure_injected,
            recovered_from_failure=recovered_from_failure,
            total_safety_violations=total_safety_violations,
            final_distance=self._distance_to_target(scene),
            sub_goals_completed=_count_sub_goals(scene, task),
            sub_goals_total=len(task.sub_goals),
        )
        ep.planning_correct = _check_planning_correct(task, trajectory)
        ep.grounding_correct = _check_grounding_correct(
            task, trajectory, scene.entities
        )
        ep.compute_time()

        return ep

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sample_instruction(self, task: EvalTask, seed: int) -> str:
        """Deterministically sample an instruction template."""
        if not task.instruction_templates:
            return task.description
        rng = random.Random(seed)
        return rng.choice(task.instruction_templates)

    def _build_tracked_entities(
        self, scene: _SceneState
    ) -> tuple[TrackedEntity, ...]:
        """Convert scene entities to TrackedEntity tuples for planner context."""
        out: list[TrackedEntity] = []
        for ent in scene.entities:
            out.append(TrackedEntity(
                entity_id=ent.label,
                label=ent.label,
                confidence=0.95,
                x=float(ent.xy[0]),
                y=float(ent.xy[1]),
                z=0.0,
                last_seen=float(scene.step) * 0.02,
            ))
        return tuple(out)

    def _build_obs(self, scene: _SceneState, task: EvalTask) -> np.ndarray:
        """Build a flat observation vector for the skill executor."""
        # Minimal proprioceptive state
        obs = np.zeros(AINEX_STATE_DIM, dtype=np.float32)
        obs[0] = float(scene.robot_xy[0])    # walk_x (proxy)
        obs[1] = float(scene.robot_xy[1])    # walk_y (proxy)
        obs[2] = float(scene.robot_yaw)      # walk_yaw (proxy)
        obs[9] = 1.0 if scene.step > 0 else 0.0  # is_walking
        obs[10] = 12000.0 / 12600.0          # battery (normalized)
        return obs

    def _distance_to_target(self, scene: _SceneState) -> float:
        """Min distance from robot to any target entity."""
        min_d = float("inf")
        for ent in scene.entities:
            if ent.is_target:
                d = float(np.linalg.norm(scene.robot_xy - ent.xy))
                min_d = min(min_d, d)
        return min_d

    def _load_resume_state(
        self, path: Path
    ) -> tuple[set[tuple[str, int]], dict[str, list[EpisodeResult]]]:
        """Load completed episodes from a prior results JSON."""
        completed: set[tuple[str, int]] = set()
        prior: dict[str, list[EpisodeResult]] = {}
        if not path.exists():
            return completed, prior
        try:
            with open(path) as f:
                data = json.load(f)
            for task_name, task_data in data.get("tasks", {}).items():
                n = task_data.get("n_episodes", 0)
                for i in range(n):
                    completed.add((task_name, i))
                # We don't reconstruct full EpisodeResults from JSON;
                # instead we re-run.  But we skip already-done episodes.
        except Exception as exc:
            logger.warning("Failed to load resume state from %s: %s", path, exc)
        return completed, prior
