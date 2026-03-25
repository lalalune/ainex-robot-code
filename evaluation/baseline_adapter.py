"""Adapter bridging baselines/ planners to the evaluation/ interface.

The ``baselines/`` package planners consume
``training.schema.embodied_context.EmbodiedContext`` (rich, simulation-grade
context with 3-D positions, IMU, entity slots, etc.) and produce
**GroundedIntent dicts** (keyed by ``"intent"``, ``"target_entity_id"``,
``"target_position"``, ...).

The ``evaluation/`` harness uses its own lightweight
``evaluation.evaluator.EmbodiedContext`` dataclass (with 2-D ``robot_xy``,
``entities: tuple[TrackedEntity]``, etc.) and expects a
``evaluation.evaluator.GroundedIntent`` dataclass back.

This module provides ``BaselineAdapter`` which:

1. Converts the evaluator's ``EmbodiedContext`` -> training schema's
   ``EmbodiedContext`` so baseline planners can consume it.
2. Converts the baselines' GroundedIntent dict -> evaluator's
   ``GroundedIntent`` dataclass so the ``Evaluator`` can consume it.
3. Wraps any ``BasePlanner`` subclass so it satisfies the evaluator's
   ``PlannerProtocol``  (i.e. ``__call__(ctx) -> GroundedIntent``).
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

from training.interfaces import TrackedEntity
from training.schema.embodied_context import (
    ContextEntity as TrainingContextEntity,
    EmbodiedContext as TrainingEmbodiedContext,
)
from baselines.base_planner import BasePlanner

from evaluation.evaluator import (
    EmbodiedContext as EvalEmbodiedContext,
    GroundedIntent as EvalGroundedIntent,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Intent -> skill name mapping
# ---------------------------------------------------------------------------

# Maps canonical intent type strings (from baselines output) to the skill
# names that the evaluation harness understands.
_INTENT_TO_SKILL: dict[str, str] = {
    "NAVIGATE_TO_ENTITY": "walk",
    "NAVIGATE_TO_POSITION": "walk_to_target",
    "FACE_ENTITY": "turn",
    "PICKUP_ENTITY": "walk_to_target",  # eval doesn't have a grasp skill
    "EMOTE": "wave",
    "SPEAK": "stand",
    "IDLE": "stand",
    "ABORT": "stand",
}


def _intent_to_skill(intent_str: str, source_action_name: str = "") -> str:
    """Convert a canonical intent string to an evaluator skill name.

    If ``source_action_name`` is provided and looks like a valid skill name,
    prefer it (the baseline planner may have already resolved it).
    """
    # Some baselines set source_action_name to a useful skill hint
    if source_action_name and source_action_name in (
        "walk", "turn", "stand", "walk_to_target", "wave", "bow",
    ):
        return source_action_name

    return _INTENT_TO_SKILL.get(intent_str.upper(), "stand")


def _is_done_from_intent(intent_str: str) -> bool:
    """Determine if the intent signals task completion."""
    return intent_str.upper() in ("IDLE", "ABORT")


# ---------------------------------------------------------------------------
# Eval -> Training context conversion
# ---------------------------------------------------------------------------

def _tracked_entity_to_context_entity(
    te: TrackedEntity,
    robot_xy: tuple[float, float],
    robot_yaw: float,
) -> TrainingContextEntity:
    """Convert evaluator ``TrackedEntity`` to training ``ContextEntity``.

    ``TrackedEntity`` stores 2-D (x, y) positions; ``ContextEntity`` needs
    3-D plus derived fields like ``distance_to_agent`` and
    ``bearing_to_agent``.
    """
    dx = te.x - robot_xy[0]
    dy = te.y - robot_xy[1]
    distance = math.sqrt(dx * dx + dy * dy)
    # Bearing relative to agent heading (0 = ahead, positive = left)
    world_angle = math.atan2(dy, dx)
    bearing = math.atan2(
        math.sin(world_angle - robot_yaw),
        math.cos(world_angle - robot_yaw),
    )

    return TrainingContextEntity(
        entity_id=te.entity_id,
        entity_type="object",  # TrackedEntity has no type field; default
        label=te.label,
        position=(te.x, te.y, te.z),
        velocity=(0.0, 0.0, 0.0),
        size=(0.0, 0.0, 0.0),
        confidence=te.confidence,
        distance_to_agent=distance,
        bearing_to_agent=bearing,
        source="evaluation",
    )


def eval_context_to_training_context(
    ctx: EvalEmbodiedContext,
) -> TrainingEmbodiedContext:
    """Convert evaluator ``EmbodiedContext`` to training-schema ``EmbodiedContext``.

    The training context is much richer, so fields without a direct
    counterpart in the evaluator context are left at their defaults.
    """
    training_entities = tuple(
        _tracked_entity_to_context_entity(te, ctx.robot_xy, ctx.robot_yaw)
        for te in ctx.entities
    )

    return TrainingEmbodiedContext(
        source="evaluation",
        timestamp=time.time(),
        agent_position=(ctx.robot_xy[0], ctx.robot_xy[1], 0.0),
        agent_yaw=ctx.robot_yaw,
        entities=training_entities,
        language_instruction=ctx.instruction,
        task_description=ctx.instruction,
    )


# ---------------------------------------------------------------------------
# Baseline output dict -> eval GroundedIntent conversion
# ---------------------------------------------------------------------------

def baseline_result_to_grounded_intent(
    result: dict[str, Any],
) -> EvalGroundedIntent:
    """Convert a baseline planner's GroundedIntent dict to the evaluator's
    ``GroundedIntent`` dataclass.
    """
    intent_str = result.get("intent", "IDLE")
    source_action = result.get("source_action_name", "")
    skill_name = _intent_to_skill(intent_str, source_action)
    reasoning = result.get("reasoning", "")

    target_label = result.get("target_entity_label", "")
    target_position = result.get("target_position", [0.0, 0.0, 0.0])
    # Extract 2-D (x, y) from the 3-D position
    if isinstance(target_position, (list, tuple)) and len(target_position) >= 2:
        target_xy = (float(target_position[0]), float(target_position[1]))
    else:
        target_xy = (0.0, 0.0)

    is_done = _is_done_from_intent(intent_str)

    return EvalGroundedIntent(
        skill_name=skill_name,
        target_entity_label=target_label,
        target_xy=target_xy,
        reasoning=f"[{intent_str}] {reasoning}",
        is_done=is_done,
    )


# ---------------------------------------------------------------------------
# BaselineAdapter -- the main entry point
# ---------------------------------------------------------------------------

class BaselineAdapter:
    """Adapts a ``baselines/`` planner to the ``evaluation/`` interface.

    After wrapping, the adapter is callable with the evaluator's
    ``EmbodiedContext`` and returns the evaluator's ``GroundedIntent``,
    making it directly usable as a ``PlannerProtocol`` in the ``Evaluator``.

    Usage::

        from baselines import RPG2RobotPlanner
        planner = BaselineAdapter(RPG2RobotPlanner())
        intent = planner(eval_context)  # returns eval GroundedIntent
    """

    def __init__(self, baseline_planner: BasePlanner) -> None:
        self.planner = baseline_planner

    def __call__(self, eval_context: EvalEmbodiedContext) -> EvalGroundedIntent:
        # 1. Convert eval context -> training context
        training_ctx = eval_context_to_training_context(eval_context)

        # 2. Call the real baseline planner
        result_dict = self.planner.plan(training_ctx)

        # 3. Convert result dict -> eval GroundedIntent
        return baseline_result_to_grounded_intent(result_dict)

    def reset(self) -> None:
        """Proxy reset to the underlying baseline planner."""
        self.planner.reset()

    @property
    def name(self) -> str:
        return self.planner.name

    @property
    def metrics(self) -> Any:
        return self.planner.metrics

    def __repr__(self) -> str:
        return f"BaselineAdapter({self.planner!r})"
