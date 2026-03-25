"""Evaluation task definitions for RPG2Robot.

Defines 9 evaluation tasks across three categories (navigation, manipulation,
language-grounded) with success criteria, entity setups, and instruction
templates.  Each task is a frozen dataclass consumed by the Evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TaskCategory(Enum):
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    LANGUAGE = "language"


class SuccessCriterion(Enum):
    """How to judge whether an episode succeeded."""
    DISTANCE = "distance"         # Robot within threshold of target position.
    CONTACT = "contact"           # End-effector touching target entity.
    GRASPED = "grasped"           # Object reported as grasped by gripper sensor.
    PLACED = "placed"             # Object placed within threshold of goal zone.
    FACING = "facing"             # Robot heading aligned with target bearing.
    SEQUENCE = "sequence"         # All sub-goals completed in order.
    RECOVERY = "recovery"         # System recovered from injected failure.


# ---------------------------------------------------------------------------
# Entity setup spec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EntitySpec:
    """Describes one entity to spawn in the evaluation scene."""
    label: str
    entity_type: int          # 0=UNKNOWN,1=PERSON,2=OBJECT,3=LANDMARK,4=FURNITURE,5=DOOR
    color: str = "default"    # For sort_by_color: "red","green","blue", etc.
    size: tuple[float, float, float] = (0.15, 0.15, 0.15)  # (w, h, d) metres
    spawn_radius_min: float = 0.5   # Min distance from robot at reset
    spawn_radius_max: float = 2.5   # Max distance from robot at reset
    is_target: bool = False         # Whether this is the task target


@dataclass(frozen=True)
class GoalZone:
    """A region the robot or object must reach."""
    label: str
    radius: float = 0.20     # Goal zone radius in metres
    spawn_radius_min: float = 1.0
    spawn_radius_max: float = 3.0


# ---------------------------------------------------------------------------
# Core task dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvalTask:
    """Full specification of a single evaluation task."""

    name: str
    category: TaskCategory
    description: str

    # Skills the planner should select from / chain together.
    required_skills: tuple[str, ...]

    # How to evaluate success.
    success_criterion: SuccessCriterion
    success_threshold: float = 0.15   # metres for DISTANCE, radians for FACING

    # Episode budget.
    max_steps: int = 500

    # Entities to spawn each episode.
    entity_setup: tuple[EntitySpec, ...] = ()

    # Goal zones (used by carry/sort tasks).
    goal_zones: tuple[GoalZone, ...] = ()

    # Instruction templates.  At runtime one is sampled and entity labels are
    # substituted for {target}, {destination}, etc.
    instruction_templates: tuple[str, ...] = ()

    # For multi-step tasks: ordered sub-goal sequence.
    sub_goals: tuple[str, ...] = ()

    # Whether to inject a skill failure mid-episode (for recovery tasks).
    inject_failure: bool = False
    failure_injection_step: int = 0


# ---------------------------------------------------------------------------
# ---- NAVIGATION TASKS (3) ----
# ---------------------------------------------------------------------------

WALK_TO_RED_BALL = EvalTask(
    name="walk_to_red_ball",
    category=TaskCategory.NAVIGATION,
    description="Walk to a red ball placed at a random position.",
    required_skills=("walk", "turn", "stand"),
    success_criterion=SuccessCriterion.DISTANCE,
    success_threshold=0.15,
    max_steps=500,
    entity_setup=(
        EntitySpec(
            label="red_ball",
            entity_type=2,  # OBJECT
            color="red",
            size=(0.10, 0.10, 0.10),
            spawn_radius_min=0.8,
            spawn_radius_max=2.5,
            is_target=True,
        ),
    ),
    instruction_templates=(
        "Walk to the red ball.",
        "Go to the red ball.",
        "Navigate to the red ball on the ground.",
        "Move toward the red ball.",
        "Approach the red ball.",
    ),
)

WALK_TO_NAMED_ENTITY = EvalTask(
    name="walk_to_named_entity",
    category=TaskCategory.NAVIGATION,
    description="Walk to a named entity among distractors.",
    required_skills=("walk", "turn", "stand"),
    success_criterion=SuccessCriterion.DISTANCE,
    success_threshold=0.20,
    max_steps=600,
    entity_setup=(
        EntitySpec(
            label="blue_box",
            entity_type=2,
            color="blue",
            size=(0.20, 0.20, 0.20),
            spawn_radius_min=1.0,
            spawn_radius_max=3.0,
            is_target=True,
        ),
        EntitySpec(
            label="green_cylinder",
            entity_type=2,
            color="green",
            size=(0.15, 0.30, 0.15),
            spawn_radius_min=0.8,
            spawn_radius_max=2.5,
        ),
        EntitySpec(
            label="red_cone",
            entity_type=2,
            color="red",
            size=(0.12, 0.25, 0.12),
            spawn_radius_min=0.8,
            spawn_radius_max=2.5,
        ),
    ),
    instruction_templates=(
        "Walk to the blue box.",
        "Go to the blue box, ignoring the other objects.",
        "Navigate to the blue box.",
        "Find the blue box and approach it.",
    ),
)

FACE_AND_APPROACH = EvalTask(
    name="face_and_approach",
    category=TaskCategory.NAVIGATION,
    description="Turn to face a target entity, then walk toward it.",
    required_skills=("turn", "walk", "stand"),
    success_criterion=SuccessCriterion.DISTANCE,
    success_threshold=0.15,
    max_steps=600,
    entity_setup=(
        EntitySpec(
            label="person",
            entity_type=1,  # PERSON
            size=(0.40, 1.70, 0.30),
            spawn_radius_min=1.5,
            spawn_radius_max=3.5,
            is_target=True,
        ),
    ),
    instruction_templates=(
        "Turn to face the person and walk toward them.",
        "Face the person, then approach.",
        "Look at the person and go to them.",
        "Rotate toward the person and walk over.",
    ),
)

# ---------------------------------------------------------------------------
# ---- MANIPULATION TASKS (3) ----
# ---------------------------------------------------------------------------

PICK_UP_OBJECT = EvalTask(
    name="pick_up_object",
    category=TaskCategory.MANIPULATION,
    description="Walk to an object and pick it up.",
    required_skills=("walk", "turn", "stand", "walk_to_target"),
    success_criterion=SuccessCriterion.GRASPED,
    success_threshold=0.05,
    max_steps=800,
    entity_setup=(
        EntitySpec(
            label="small_cube",
            entity_type=2,
            color="yellow",
            size=(0.06, 0.06, 0.06),
            spawn_radius_min=0.5,
            spawn_radius_max=1.5,
            is_target=True,
        ),
    ),
    instruction_templates=(
        "Pick up the yellow cube.",
        "Grab the small cube.",
        "Go to the yellow cube and pick it up.",
        "Get the cube from the floor.",
    ),
)

CARRY_TO_TARGET = EvalTask(
    name="carry_to_target",
    category=TaskCategory.MANIPULATION,
    description="Pick up an object and carry it to a goal zone.",
    required_skills=("walk", "turn", "stand", "walk_to_target"),
    success_criterion=SuccessCriterion.PLACED,
    success_threshold=0.20,
    max_steps=1000,
    entity_setup=(
        EntitySpec(
            label="red_block",
            entity_type=2,
            color="red",
            size=(0.08, 0.08, 0.08),
            spawn_radius_min=0.5,
            spawn_radius_max=1.5,
            is_target=True,
        ),
    ),
    goal_zones=(
        GoalZone(
            label="drop_zone",
            radius=0.20,
            spawn_radius_min=1.5,
            spawn_radius_max=3.0,
        ),
    ),
    instruction_templates=(
        "Pick up the red block and carry it to the drop zone.",
        "Bring the red block to the marked area.",
        "Grab the red block and place it in the drop zone.",
        "Carry the red block to the goal.",
    ),
    sub_goals=("grasp_red_block", "navigate_to_drop_zone", "release"),
)

SORT_BY_COLOR = EvalTask(
    name="sort_by_color",
    category=TaskCategory.MANIPULATION,
    description="Sort objects into goal zones matching their color.",
    required_skills=("walk", "turn", "stand", "walk_to_target"),
    success_criterion=SuccessCriterion.SEQUENCE,
    success_threshold=0.20,
    max_steps=2000,
    entity_setup=(
        EntitySpec(
            label="red_cube",
            entity_type=2,
            color="red",
            size=(0.06, 0.06, 0.06),
            spawn_radius_min=0.5,
            spawn_radius_max=1.5,
            is_target=True,
        ),
        EntitySpec(
            label="blue_cube",
            entity_type=2,
            color="blue",
            size=(0.06, 0.06, 0.06),
            spawn_radius_min=0.5,
            spawn_radius_max=1.5,
            is_target=True,
        ),
    ),
    goal_zones=(
        GoalZone(label="red_zone", radius=0.20, spawn_radius_min=1.5, spawn_radius_max=3.0),
        GoalZone(label="blue_zone", radius=0.20, spawn_radius_min=1.5, spawn_radius_max=3.0),
    ),
    instruction_templates=(
        "Sort the cubes by color: red cube to the red zone, blue cube to the blue zone.",
        "Put each colored cube in the matching zone.",
        "Carry the red cube to the red zone and the blue cube to the blue zone.",
    ),
    sub_goals=(
        "grasp_red_cube", "navigate_to_red_zone", "release",
        "grasp_blue_cube", "navigate_to_blue_zone", "release",
    ),
)

# ---------------------------------------------------------------------------
# ---- LANGUAGE-GROUNDED TASKS (3) ----
# ---------------------------------------------------------------------------

DISAMBIGUATED_NAV = EvalTask(
    name="disambiguated_nav",
    category=TaskCategory.LANGUAGE,
    description="Navigate to the correct entity when multiple similar ones exist.",
    required_skills=("walk", "turn", "stand"),
    success_criterion=SuccessCriterion.DISTANCE,
    success_threshold=0.20,
    max_steps=700,
    entity_setup=(
        EntitySpec(
            label="large_red_box",
            entity_type=2,
            color="red",
            size=(0.30, 0.30, 0.30),
            spawn_radius_min=1.0,
            spawn_radius_max=3.0,
            is_target=True,
        ),
        EntitySpec(
            label="small_red_box",
            entity_type=2,
            color="red",
            size=(0.10, 0.10, 0.10),
            spawn_radius_min=1.0,
            spawn_radius_max=3.0,
        ),
        EntitySpec(
            label="large_blue_box",
            entity_type=2,
            color="blue",
            size=(0.30, 0.30, 0.30),
            spawn_radius_min=1.0,
            spawn_radius_max=3.0,
        ),
    ),
    instruction_templates=(
        "Go to the large red box.",
        "Walk to the big red box, not the small one.",
        "Navigate to the larger of the two red boxes.",
        "Approach the large red box.",
    ),
)

MULTI_STEP_FETCH = EvalTask(
    name="multi_step_fetch",
    category=TaskCategory.LANGUAGE,
    description="Follow a multi-step instruction: go to X, pick up Y, bring to Z.",
    required_skills=("walk", "turn", "stand", "walk_to_target"),
    success_criterion=SuccessCriterion.SEQUENCE,
    success_threshold=0.20,
    max_steps=1500,
    entity_setup=(
        EntitySpec(
            label="green_ball",
            entity_type=2,
            color="green",
            size=(0.08, 0.08, 0.08),
            spawn_radius_min=0.8,
            spawn_radius_max=2.0,
            is_target=True,
        ),
        EntitySpec(
            label="table",
            entity_type=4,  # FURNITURE
            size=(0.60, 0.50, 0.40),
            spawn_radius_min=2.0,
            spawn_radius_max=3.5,
        ),
    ),
    goal_zones=(
        GoalZone(
            label="table_surface",
            radius=0.25,
            spawn_radius_min=2.0,
            spawn_radius_max=3.5,
        ),
    ),
    instruction_templates=(
        "Pick up the green ball and put it on the table.",
        "Grab the green ball, then carry it to the table.",
        "Fetch the green ball and place it on the table surface.",
    ),
    sub_goals=(
        "navigate_to_green_ball",
        "grasp_green_ball",
        "navigate_to_table",
        "release",
    ),
)

RECOVERY_FROM_FAILURE = EvalTask(
    name="recovery_from_failure",
    category=TaskCategory.LANGUAGE,
    description="Complete a task after an injected skill failure forces replanning.",
    required_skills=("walk", "turn", "stand", "walk_to_target"),
    success_criterion=SuccessCriterion.RECOVERY,
    success_threshold=0.20,
    max_steps=1000,
    entity_setup=(
        EntitySpec(
            label="target_cone",
            entity_type=2,
            color="orange",
            size=(0.12, 0.25, 0.12),
            spawn_radius_min=1.0,
            spawn_radius_max=3.0,
            is_target=True,
        ),
    ),
    instruction_templates=(
        "Walk to the orange cone.",
        "Go to the orange cone.",
        "Navigate to the cone.",
    ),
    inject_failure=True,
    failure_injection_step=150,
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TASK_REGISTRY: dict[str, EvalTask] = {
    task.name: task
    for task in [
        # Navigation
        WALK_TO_RED_BALL,
        WALK_TO_NAMED_ENTITY,
        FACE_AND_APPROACH,
        # Manipulation
        PICK_UP_OBJECT,
        CARRY_TO_TARGET,
        SORT_BY_COLOR,
        # Language
        DISAMBIGUATED_NAV,
        MULTI_STEP_FETCH,
        RECOVERY_FROM_FAILURE,
    ]
}


def get_tasks_by_category(category: TaskCategory) -> list[EvalTask]:
    """Return all tasks belonging to a category."""
    return [t for t in TASK_REGISTRY.values() if t.category == category]


def get_task_by_name(name: str) -> EvalTask:
    """Look up a single task.  Raises KeyError if not found."""
    return TASK_REGISTRY[name]


def list_task_names() -> list[str]:
    """Return sorted list of all task names."""
    return sorted(TASK_REGISTRY.keys())
