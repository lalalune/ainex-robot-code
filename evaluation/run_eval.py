#!/usr/bin/env python3
"""CLI runner for RPG2Robot evaluation.

Usage:
    python -m evaluation.run_eval --planner rpg2robot --tasks all --episodes 500 --output results/
    python -m evaluation.run_eval --planner scripted --tasks navigation --episodes 100
    python -m evaluation.run_eval --planner rpg2robot --tasks walk_to_red_ball --episodes 50 --export-latex
    python -m evaluation.run_eval --planner flat_rl --checkpoint path/to/model --tasks all --episodes 200

    # Use the full baselines/ package planners (LLM calls, recovery, etc.)
    # instead of the simplified inline planners:
    python -m evaluation.run_eval --planner rpg2robot --use-baselines-package --tasks all
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from training.interfaces import (
    CanonicalIntent,
    CanonicalIntentType,
    TrackedEntity,
)
from training.rl.skills.base_skill import BaseSkill, SkillParams, SkillStatus
from training.rl.skills.registry import SkillRegistry

from evaluation.evaluator import (
    DefaultSkillExecutor,
    EmbodiedContext,
    Evaluator,
    GroundedIntent,
)
from evaluation.task_suite import (
    EvalTask,
    TaskCategory,
    TASK_REGISTRY,
    get_tasks_by_category,
    get_task_by_name,
    list_task_names,
)
from evaluation.metrics import (
    EvalResults,
    export_csv,
    export_json,
    export_latex,
    export_trajectories_json,
)
from evaluation.baseline_adapter import BaselineAdapter

logger = logging.getLogger("evaluation")


# ---------------------------------------------------------------------------
# Baseline planners
# ---------------------------------------------------------------------------

class ZeroShotPlanner:
    """Baseline: always outputs walk toward the first tracked entity.

    No task decomposition, no language grounding -- just walk forward.
    """

    def __call__(self, ctx: EmbodiedContext) -> GroundedIntent:
        if ctx.entities:
            ent = ctx.entities[0]
            return GroundedIntent(
                skill_name="walk",
                target_entity_label=ent.label,
                target_xy=(ent.x, ent.y),
                reasoning="zero_shot: walk to first entity",
            )
        return GroundedIntent(
            skill_name="stand",
            reasoning="zero_shot: no entities visible",
            is_done=True,
        )


class ScriptedPlanner:
    """Baseline: hand-coded heuristic planner.

    Parses the instruction for keywords and selects skills accordingly.
    Uses a simple state machine: navigate -> grasp -> navigate -> release.
    No replanning on failure — demonstrates a rigid rule-based system.
    """

    def __init__(self) -> None:
        self._phase = "navigate"  # navigate | grasp | carry | release | done

    def __call__(self, ctx: EmbodiedContext) -> GroundedIntent:
        # Reset state machine at start of each episode
        if ctx.step == 0:
            self._phase = "navigate"

        instr = ctx.instruction.lower()
        is_manip = any(kw in instr for kw in (
            "pick up", "grab", "get", "fetch", "carry", "bring",
            "put", "place", "sort",
        ))
        target_ent = self._find_target(ctx)

        if not target_ent:
            return GroundedIntent(
                skill_name="stand",
                reasoning="scripted: no target found",
                is_done=True,
            )

        target_xy = (target_ent.x, target_ent.y)
        target_label = target_ent.label

        delta_x = target_ent.x - ctx.robot_xy[0]
        delta_y = target_ent.y - ctx.robot_xy[1]
        dist = math.sqrt(delta_x**2 + delta_y**2)
        target_angle = math.atan2(delta_y, delta_x)
        yaw_err = abs(math.atan2(
            math.sin(target_angle - ctx.robot_yaw),
            math.cos(target_angle - ctx.robot_yaw),
        ))

        # For manipulation: simple state machine (no replanning)
        if is_manip:
            needs_carry = any(kw in instr for kw in (
                "carry", "bring", "put", "place", "sort",
            ))

            # Phase: navigate to object
            if self._phase == "navigate" and not ctx.grasped_entity:
                if dist < 0.08:
                    self._phase = "grasp"
                elif yaw_err > 0.5:
                    return GroundedIntent(
                        skill_name="turn", target_entity_label=target_label,
                        target_xy=target_xy, reasoning="scripted: turn to object",
                    )
                else:
                    return GroundedIntent(
                        skill_name="walk_to_target", target_entity_label=target_label,
                        target_xy=target_xy, reasoning="scripted: walk to object",
                    )

            # Phase: grasp
            if self._phase == "grasp":
                if ctx.grasped_entity:
                    self._phase = "carry" if needs_carry else "done"
                else:
                    return GroundedIntent(
                        skill_name="grasp", target_entity_label=target_label,
                        target_xy=target_xy, reasoning="scripted: grasp object",
                    )

            # Phase: carry to goal zone
            if self._phase == "carry" and ctx.goal_zones:
                gz_label, gx, gy = ctx.goal_zones[0]
                gz_xy = (gx, gy)
                dx = gx - ctx.robot_xy[0]
                dy = gy - ctx.robot_xy[1]
                gz_dist = math.sqrt(dx**2 + dy**2)
                if gz_dist < 0.15:
                    self._phase = "release"
                else:
                    gz_angle = math.atan2(dy, dx)
                    gz_yaw_err = abs(math.atan2(
                        math.sin(gz_angle - ctx.robot_yaw),
                        math.cos(gz_angle - ctx.robot_yaw),
                    ))
                    if gz_yaw_err > 0.5:
                        return GroundedIntent(
                            skill_name="turn", target_entity_label=gz_label,
                            target_xy=gz_xy, reasoning="scripted: turn to goal",
                        )
                    return GroundedIntent(
                        skill_name="walk_to_target", target_entity_label=gz_label,
                        target_xy=gz_xy, reasoning="scripted: carry to goal",
                    )

            # Phase: release
            if self._phase == "release":
                self._phase = "done"
                return GroundedIntent(
                    skill_name="release", target_entity_label=target_label,
                    target_xy=(ctx.robot_xy[0], ctx.robot_xy[1]),
                    reasoning="scripted: release object",
                )

            if self._phase == "done":
                return GroundedIntent(
                    skill_name="stand", reasoning="scripted: task complete",
                    is_done=True,
                )

        # Navigation-only tasks
        if yaw_err > 0.5:
            return GroundedIntent(
                skill_name="turn",
                target_entity_label=target_label,
                target_xy=target_xy,
                reasoning="scripted: turn to face target",
            )

        if dist < 0.15:
            return GroundedIntent(
                skill_name="stand",
                target_entity_label=target_label,
                target_xy=target_xy,
                reasoning="scripted: reached target",
                is_done=True,
            )

        return GroundedIntent(
            skill_name="walk",
            target_entity_label=target_label,
            target_xy=target_xy,
            reasoning="scripted: walk toward target",
        )

    def _find_target(self, ctx: EmbodiedContext) -> TrackedEntity | None:
        """Find the target entity by matching instruction keywords."""
        instr = ctx.instruction.lower()
        best: TrackedEntity | None = None
        best_score = -1

        for ent in ctx.entities:
            score = 0
            label = ent.label.lower().replace("_", " ")
            for word in label.split():
                if word in instr:
                    score += 1
            if score > best_score:
                best_score = score
                best = ent

        return best


class SayCanPlanner:
    """Baseline: SayCan-style planner.

    Scores each skill by affordance (is it executable?) and selects
    the highest-scoring skill-entity pair.  Includes grasp/release
    affordances for manipulation, but uses flat scoring (no hierarchical
    decomposition), so it struggles with multi-step tasks.
    """

    def __call__(self, ctx: EmbodiedContext) -> GroundedIntent:
        target_ent = self._select_target(ctx)
        if not target_ent:
            return GroundedIntent(
                skill_name="stand",
                reasoning="saycan: no entities",
                is_done=True,
            )

        target_xy = (target_ent.x, target_ent.y)
        target_label = target_ent.label
        dx = target_ent.x - ctx.robot_xy[0]
        dy = target_ent.y - ctx.robot_xy[1]
        dist = math.sqrt(dx**2 + dy**2)
        target_angle = math.atan2(dy, dx)
        yaw_err = abs(math.atan2(
            math.sin(target_angle - ctx.robot_yaw),
            math.cos(target_angle - ctx.robot_yaw),
        ))

        instr = ctx.instruction.lower()
        is_manip = any(kw in instr for kw in (
            "pick up", "grab", "get", "fetch", "carry", "bring",
            "put", "place", "sort",
        ))

        # Affordance scoring
        scores: dict[str, tuple[float, str, tuple[float, float]]] = {}
        # (score, target_label, target_xy)
        scores["turn"] = (1.0 if yaw_err > 0.3 else 0.1, target_label, target_xy)
        walk_score = max(0.0, 1.0 - yaw_err / math.pi) * min(1.0, dist)
        scores["walk"] = (walk_score, target_label, target_xy)
        scores["walk_to_target"] = (walk_score * 1.1, target_label, target_xy)
        scores["stand"] = (1.0 if dist < 0.15 and not is_manip else 0.0, target_label, target_xy)

        # Manipulation affordances
        if is_manip:
            # Grasp: high when close and not holding
            grasp_score = 2.0 if (dist < 0.10 and not ctx.grasped_entity) else 0.0
            scores["grasp"] = (grasp_score, target_label, target_xy)

            # If holding an object, retarget navigation to goal zone
            if ctx.grasped_entity and ctx.goal_zones:
                gz_label, gx, gy = ctx.goal_zones[0]
                gz_xy = (gx, gy)
                gz_dx = gx - ctx.robot_xy[0]
                gz_dy = gy - ctx.robot_xy[1]
                gz_dist = math.sqrt(gz_dx**2 + gz_dy**2)
                gz_angle = math.atan2(gz_dy, gz_dx)
                gz_yaw_err = abs(math.atan2(
                    math.sin(gz_angle - ctx.robot_yaw),
                    math.cos(gz_angle - ctx.robot_yaw),
                ))

                # Override nav scores to point to goal zone
                scores["turn"] = (1.0 if gz_yaw_err > 0.3 else 0.1, gz_label, gz_xy)
                gz_walk = max(0.0, 1.0 - gz_yaw_err / math.pi) * min(1.0, gz_dist)
                scores["walk"] = (gz_walk, gz_label, gz_xy)
                scores["walk_to_target"] = (gz_walk * 1.1, gz_label, gz_xy)
                scores["stand"] = (0.0, target_label, target_xy)

                # Release: high when near goal zone
                release_score = 2.0 if gz_dist < 0.20 else 0.0
                scores["release"] = (release_score, target_label, gz_xy)
            else:
                scores["release"] = (0.0, target_label, target_xy)

        # Select highest scoring skill
        best_skill = max(scores, key=lambda k: scores[k][0])
        best_score, best_label, best_xy = scores[best_skill]

        if best_skill == "stand":
            return GroundedIntent(
                skill_name="stand",
                target_entity_label=best_label,
                target_xy=best_xy,
                reasoning=f"saycan: arrived (dist={dist:.3f})",
                is_done=True,
            )

        return GroundedIntent(
            skill_name=best_skill,
            target_entity_label=best_label,
            target_xy=best_xy,
            reasoning=f"saycan: {best_skill} (score={best_score:.2f})",
        )

    def _select_target(self, ctx: EmbodiedContext) -> TrackedEntity | None:
        """Select target entity by instruction-keyword matching."""
        instr = ctx.instruction.lower()
        best: TrackedEntity | None = None
        best_score = -1
        for ent in ctx.entities:
            score = 0
            label = ent.label.lower().replace("_", " ")
            for word in label.split():
                if word in instr:
                    score += 1
            if score > best_score:
                best_score = score
                best = ent
        return best


class FlatRLPlanner:
    """Baseline: flat RL policy (no skill decomposition).

    Outputs walk_to_target for every step -- a monolithic policy that
    tries to reach the target directly without hierarchical planning.
    """

    def __call__(self, ctx: EmbodiedContext) -> GroundedIntent:
        target_ent = self._nearest_target(ctx)
        if not target_ent:
            return GroundedIntent(
                skill_name="walk",
                reasoning="flat_rl: no target, walk forward",
                target_xy=(
                    ctx.robot_xy[0] + math.cos(ctx.robot_yaw),
                    ctx.robot_xy[1] + math.sin(ctx.robot_yaw),
                ),
            )

        return GroundedIntent(
            skill_name="walk_to_target",
            target_entity_label=target_ent.label,
            target_xy=(target_ent.x, target_ent.y),
            reasoning="flat_rl: walk_to_target",
        )

    def _nearest_target(self, ctx: EmbodiedContext) -> TrackedEntity | None:
        if not ctx.entities:
            return None
        return min(
            ctx.entities,
            key=lambda e: math.sqrt(
                (e.x - ctx.robot_xy[0])**2 + (e.y - ctx.robot_xy[1])**2
            ),
        )


class RPG2RobotPlanner:
    """RPG2Robot planner -- hierarchical skill-based planner.

    Implements the full RPG2Robot planning loop:
    1. Parse instruction to identify task type and target entities.
    2. Decompose into skill+target sequence (turn -> walk -> grasp -> ...).
    3. Track skill completion and advance to next sub-goal.
    4. Handle failure by replanning from current state.

    When a checkpoint is provided, loads a trained planner model.
    Otherwise uses a strong heuristic that demonstrates the
    hierarchical structure.
    """

    def __init__(self, checkpoint: str | None = None) -> None:
        self._checkpoint = checkpoint
        self._model: Any = None
        # Plan is a list of (skill_name, target_label_or_"goal") tuples
        self._plan: list[tuple[str, str]] = []
        self._plan_idx: int = 0
        self._last_instruction: str = ""

        if checkpoint:
            self._load_checkpoint(checkpoint)

    def _load_checkpoint(self, path: str) -> None:
        """Load trained planner checkpoint."""
        ckpt_path = Path(path)
        if not ckpt_path.exists():
            logger.warning("Checkpoint not found: %s (using heuristic)", path)
            return
        try:
            import torch
            self._model = torch.load(path, map_location="cpu", weights_only=False)
            logger.info("Loaded planner checkpoint: %s", path)
        except Exception as exc:
            logger.warning("Failed to load checkpoint %s: %s", path, exc)

    def __call__(self, ctx: EmbodiedContext) -> GroundedIntent:
        # Reset at start of new episode or re-plan on failure/instruction change
        if (ctx.step == 0
                or ctx.instruction != self._last_instruction
                or ctx.previous_skill_status == "failed"):
            self._last_instruction = ctx.instruction
            self._plan = self._decompose(ctx)
            self._plan_idx = 0

        if self._plan_idx >= len(self._plan):
            return GroundedIntent(
                skill_name="stand",
                reasoning="rpg2robot: plan complete",
                is_done=True,
            )

        current_skill, current_target = self._plan[self._plan_idx]

        # Resolve target: could be an entity label or a goal zone label
        target_ent = self._resolve_target(ctx, current_target)
        target_xy = self._resolve_target_xy(ctx, current_target)

        if target_xy is None:
            return GroundedIntent(
                skill_name="stand",
                reasoning="rpg2robot: cannot resolve target",
                is_done=True,
            )

        # Check completion conditions for current skill
        advance = False
        if current_skill == "turn":
            dx = target_xy[0] - ctx.robot_xy[0]
            dy = target_xy[1] - ctx.robot_xy[1]
            target_angle = math.atan2(dy, dx)
            yaw_err = abs(math.atan2(
                math.sin(target_angle - ctx.robot_yaw),
                math.cos(target_angle - ctx.robot_yaw),
            ))
            advance = yaw_err < 0.3

        elif current_skill in ("walk", "walk_to_target"):
            dx = target_xy[0] - ctx.robot_xy[0]
            dy = target_xy[1] - ctx.robot_xy[1]
            dist = math.sqrt(dx**2 + dy**2)
            advance = dist < 0.08  # Tight threshold to ensure grasp range

        elif current_skill == "grasp":
            advance = ctx.grasped_entity != ""

        elif current_skill in ("release", "place"):
            advance = ctx.grasped_entity == ""

        elif current_skill == "stand":
            advance = True

        if advance:
            self._plan_idx += 1
            if self._plan_idx >= len(self._plan):
                return GroundedIntent(
                    skill_name="stand",
                    target_entity_label=current_target,
                    target_xy=target_xy,
                    reasoning="rpg2robot: plan complete",
                    is_done=True,
                )
            current_skill, current_target = self._plan[self._plan_idx]
            target_xy = self._resolve_target_xy(ctx, current_target) or target_xy

        return GroundedIntent(
            skill_name=current_skill,
            target_entity_label=current_target,
            target_xy=target_xy,
            reasoning=f"rpg2robot: step {self._plan_idx + 1}/{len(self._plan)} ({current_skill} -> {current_target})",
        )

    def _decompose(self, ctx: EmbodiedContext) -> list[tuple[str, str]]:
        """Decompose instruction into (skill, target) sequence."""
        instr = ctx.instruction.lower()
        entities = {e.label: e for e in ctx.entities}

        # Identify primary target entity
        target = self._identify_target(ctx)

        # Identify goal zone target (for carry/place tasks)
        goal_target = ""
        if ctx.goal_zones:
            goal_target = ctx.goal_zones[0][0]  # First goal zone label

        # Sort: pick up each entity and carry to matching zone
        if any(kw in instr for kw in ("sort", "each")):
            plan: list[tuple[str, str]] = []
            for ent in ctx.entities:
                # Find matching goal zone by color
                matching_zone = ""
                for gz_label, _, _ in ctx.goal_zones:
                    if ent.label.split("_")[0] in gz_label:
                        matching_zone = gz_label
                        break
                if not matching_zone and ctx.goal_zones:
                    matching_zone = ctx.goal_zones[0][0]
                plan.extend([
                    ("turn", ent.label),
                    ("walk_to_target", ent.label),
                    ("grasp", ent.label),
                    ("turn", matching_zone),
                    ("walk_to_target", matching_zone),
                    ("release", ent.label),
                ])
            return plan

        # Carry / fetch / bring: pick up then deliver to goal
        has_pick = any(kw in instr for kw in ("pick up", "grab", "get", "fetch"))
        has_carry = any(kw in instr for kw in ("carry", "bring", "put", "place", "deliver"))

        if has_carry:
            # Full pick-and-place sequence
            return [
                ("turn", target),
                ("walk_to_target", target),
                ("grasp", target),
                ("turn", goal_target or target),
                ("walk_to_target", goal_target or target),
                ("release", target),
            ]

        if has_pick:
            # Just pick up
            return [
                ("turn", target),
                ("walk_to_target", target),
                ("grasp", target),
            ]

        # Face and approach
        if any(kw in instr for kw in ("face", "look at", "rotate toward")):
            return [("turn", target), ("walk", target)]

        # Default: navigation
        return [("turn", target), ("walk", target)]

    def _identify_target(self, ctx: EmbodiedContext) -> str:
        """Identify the target entity from instruction."""
        instr = ctx.instruction.lower()
        best_label = ""
        best_score = -1

        for ent in ctx.entities:
            score = 0
            label = ent.label.lower().replace("_", " ")
            for word in label.split():
                if len(word) > 2 and word in instr:
                    score += 1
            # Bonus for color match
            for color in ("red", "blue", "green", "yellow", "orange"):
                if color in label and color in instr:
                    score += 2
            # Bonus for size match
            for size in ("large", "big", "small", "little"):
                if size in label and size in instr:
                    score += 2
            if score > best_score:
                best_score = score
                best_label = ent.label

        return best_label

    def _resolve_target(
        self, ctx: EmbodiedContext, label: str
    ) -> TrackedEntity | None:
        """Find TrackedEntity matching label."""
        for ent in ctx.entities:
            if ent.label == label:
                return ent
        return ctx.entities[0] if ctx.entities else None

    def _resolve_target_xy(
        self, ctx: EmbodiedContext, label: str
    ) -> tuple[float, float] | None:
        """Resolve a label (entity or goal zone) to xy coordinates."""
        # Check entities first
        for ent in ctx.entities:
            if ent.label == label:
                return (ent.x, ent.y)
        # Check goal zones
        for gz_label, gx, gy in ctx.goal_zones:
            if gz_label == label:
                return (gx, gy)
        # Fallback to first entity
        if ctx.entities:
            return (ctx.entities[0].x, ctx.entities[0].y)
        return None


# ---------------------------------------------------------------------------
# Ablation variants of RPG2Robot
# ---------------------------------------------------------------------------

class RPG2Robot_NoReplan(RPG2RobotPlanner):
    """Ablation: RPG2Robot without failure replanning."""

    def __call__(self, ctx: EmbodiedContext) -> GroundedIntent:
        # Only plan once at step 0, never replan on failure
        if ctx.step == 0 or ctx.instruction != self._last_instruction:
            self._last_instruction = ctx.instruction
            self._plan = self._decompose(ctx)
            self._plan_idx = 0
        # Skip the failure replan logic — proceed with original plan
        if self._plan_idx >= len(self._plan):
            return GroundedIntent(skill_name="stand", reasoning="ablation:no_replan done", is_done=True)

        current_skill, current_target = self._plan[self._plan_idx]
        target_xy = self._resolve_target_xy(ctx, current_target)
        if target_xy is None:
            return GroundedIntent(skill_name="stand", reasoning="no target", is_done=True)

        advance = False
        if current_skill == "turn":
            dx = target_xy[0] - ctx.robot_xy[0]
            dy = target_xy[1] - ctx.robot_xy[1]
            yaw_err = abs(math.atan2(math.sin(math.atan2(dy, dx) - ctx.robot_yaw), math.cos(math.atan2(dy, dx) - ctx.robot_yaw)))
            advance = yaw_err < 0.3
        elif current_skill in ("walk", "walk_to_target"):
            dist = math.sqrt((target_xy[0] - ctx.robot_xy[0])**2 + (target_xy[1] - ctx.robot_xy[1])**2)
            advance = dist < 0.08
        elif current_skill == "grasp":
            advance = ctx.grasped_entity != ""
        elif current_skill in ("release", "place"):
            advance = ctx.grasped_entity == ""
        elif current_skill == "stand":
            advance = True

        if advance:
            self._plan_idx += 1
            if self._plan_idx >= len(self._plan):
                return GroundedIntent(skill_name="stand", target_entity_label=current_target, target_xy=target_xy, reasoning="ablation:no_replan complete", is_done=True)
            current_skill, current_target = self._plan[self._plan_idx]
            target_xy = self._resolve_target_xy(ctx, current_target) or target_xy

        return GroundedIntent(skill_name=current_skill, target_entity_label=current_target, target_xy=target_xy, reasoning=f"ablation:no_replan {self._plan_idx}/{len(self._plan)}")


class RPG2Robot_NoGrasp(RPG2RobotPlanner):
    """Ablation: RPG2Robot without explicit grasp/release — uses walk proximity only."""

    def _decompose(self, ctx: EmbodiedContext) -> list[tuple[str, str]]:
        instr = ctx.instruction.lower()
        target = self._identify_target(ctx)
        goal_target = ctx.goal_zones[0][0] if ctx.goal_zones else target

        if any(kw in instr for kw in ("sort", "each")):
            plan: list[tuple[str, str]] = []
            for ent in ctx.entities:
                matching_zone = ""
                for gz_label, _, _ in ctx.goal_zones:
                    if ent.label.split("_")[0] in gz_label:
                        matching_zone = gz_label
                        break
                if not matching_zone and ctx.goal_zones:
                    matching_zone = ctx.goal_zones[0][0]
                plan.extend([("turn", ent.label), ("walk_to_target", ent.label), ("walk_to_target", matching_zone)])
            return plan

        has_carry = any(kw in instr for kw in ("carry", "bring", "put", "place", "deliver"))
        has_pick = any(kw in instr for kw in ("pick up", "grab", "get", "fetch"))
        if has_carry:
            return [("turn", target), ("walk_to_target", target), ("walk_to_target", goal_target)]
        if has_pick:
            return [("turn", target), ("walk_to_target", target)]
        if any(kw in instr for kw in ("face", "look at", "rotate toward")):
            return [("turn", target), ("walk", target)]
        return [("turn", target), ("walk", target)]


class RPG2Robot_NoDecompose(RPG2RobotPlanner):
    """Ablation: RPG2Robot with target ID but no skill decomposition (single walk)."""

    def _decompose(self, ctx: EmbodiedContext) -> list[tuple[str, str]]:
        target = self._identify_target(ctx)
        return [("walk_to_target", target)]


class RPG2Robot_NoMultitarget(RPG2RobotPlanner):
    """Ablation: RPG2Robot without multi-target support (single target for sort)."""

    def _decompose(self, ctx: EmbodiedContext) -> list[tuple[str, str]]:
        instr = ctx.instruction.lower()
        target = self._identify_target(ctx)
        goal_target = ctx.goal_zones[0][0] if ctx.goal_zones else target

        # Sort treated as single-target carry (only first entity)
        if any(kw in instr for kw in ("sort", "each")):
            return [
                ("turn", target), ("walk_to_target", target),
                ("grasp", target), ("turn", goal_target),
                ("walk_to_target", goal_target), ("release", target),
            ]

        # Everything else same as full RPG2Robot
        has_carry = any(kw in instr for kw in ("carry", "bring", "put", "place", "deliver"))
        has_pick = any(kw in instr for kw in ("pick up", "grab", "get", "fetch"))
        if has_carry:
            return [("turn", target), ("walk_to_target", target), ("grasp", target), ("turn", goal_target), ("walk_to_target", goal_target), ("release", target)]
        if has_pick:
            return [("turn", target), ("walk_to_target", target), ("grasp", target)]
        if any(kw in instr for kw in ("face", "look at", "rotate toward")):
            return [("turn", target), ("walk", target)]
        return [("turn", target), ("walk", target)]


# ---------------------------------------------------------------------------
# Planner factory
# ---------------------------------------------------------------------------

PLANNER_REGISTRY: dict[str, type | Any] = {
    "zero_shot": ZeroShotPlanner,
    "scripted": ScriptedPlanner,
    "saycan": SayCanPlanner,
    "flat_rl": FlatRLPlanner,
    "rpg2robot": RPG2RobotPlanner,
    # Ablation variants
    "rpg2robot_no_replan": RPG2Robot_NoReplan,
    "rpg2robot_no_grasp": RPG2Robot_NoGrasp,
    "rpg2robot_no_decompose": RPG2Robot_NoDecompose,
    "rpg2robot_no_multitarget": RPG2Robot_NoMultitarget,
}


def build_planner(name: str, checkpoint: str | None = None) -> Any:
    """Construct an inline planner by name (simplified, no LLM calls)."""
    if name not in PLANNER_REGISTRY:
        raise ValueError(
            f"Unknown planner {name!r}. "
            f"Available: {sorted(PLANNER_REGISTRY.keys())}"
        )
    cls = PLANNER_REGISTRY[name]
    if name == "rpg2robot" and checkpoint:
        return cls(checkpoint=checkpoint)
    return cls()


def build_baselines_planner(
    name: str, checkpoint: str | None = None
) -> BaselineAdapter:
    """Construct a planner from the baselines/ package, wrapped in an adapter.

    The baselines/ package contains the *real* implementations with LLM
    calls, affordance scoring, recovery logic, episode memory, etc.  This
    function imports them, instantiates the requested planner, and wraps it
    in a ``BaselineAdapter`` so it conforms to the evaluator's interface.
    """
    from baselines import (
        FlatRLPlanner as BL_FlatRLPlanner,
        RPG2RobotPlanner as BL_RPG2RobotPlanner,
        SayCanPlanner as BL_SayCanPlanner,
        ScriptedPlanner as BL_ScriptedPlanner,
        ZeroShotPlanner as BL_ZeroShotPlanner,
    )

    baselines_registry: dict[str, type] = {
        "zero_shot": BL_ZeroShotPlanner,
        "scripted": BL_ScriptedPlanner,
        "saycan": BL_SayCanPlanner,
        "flat_rl": BL_FlatRLPlanner,
        "rpg2robot": BL_RPG2RobotPlanner,
    }

    if name not in baselines_registry:
        raise ValueError(
            f"Unknown baseline planner {name!r}. "
            f"Available: {sorted(baselines_registry.keys())}"
        )

    cls = baselines_registry[name]

    # Construct with the appropriate kwargs per planner type
    if name == "rpg2robot" and checkpoint:
        planner = cls(finetuned_model_path=checkpoint)
    elif name == "flat_rl" and checkpoint:
        planner = cls(checkpoint_path=checkpoint)
    else:
        planner = cls()

    logger.info(
        "Using baselines/ package planner: %s (adapted via BaselineAdapter)",
        planner,
    )
    return BaselineAdapter(planner)


# ---------------------------------------------------------------------------
# Task selection
# ---------------------------------------------------------------------------

def resolve_tasks(task_spec: str) -> list[EvalTask]:
    """Resolve a task specification string to a list of EvalTask objects.

    Supports:
      - "all" : all 9 tasks
      - "navigation" / "manipulation" / "language" : category
      - specific task name (e.g. "walk_to_red_ball")
      - comma-separated list of task names
    """
    spec = task_spec.strip().lower()

    if spec == "all":
        return list(TASK_REGISTRY.values())

    # Category
    for cat in TaskCategory:
        if spec == cat.value:
            return get_tasks_by_category(cat)

    # Comma-separated list or single name
    names = [n.strip() for n in spec.split(",")]
    tasks: list[EvalTask] = []
    for name in names:
        if name in TASK_REGISTRY:
            tasks.append(TASK_REGISTRY[name])
        else:
            raise ValueError(
                f"Unknown task {name!r}. Available: {list_task_names()}"
            )
    return tasks


# ---------------------------------------------------------------------------
# Progress bar
# ---------------------------------------------------------------------------

class _ProgressBar:
    """Simple terminal progress bar."""

    def __init__(self, total_tasks: int, episodes_per_task: int) -> None:
        self._total_tasks = total_tasks
        self._eps_per_task = episodes_per_task
        self._total = total_tasks * episodes_per_task
        self._done = 0
        self._current_task = ""
        self._start_time = time.monotonic()

    def update(self, task_name: str, episode: int, total: int) -> None:
        self._current_task = task_name
        self._done += 1
        elapsed = time.monotonic() - self._start_time
        rate = self._done / max(elapsed, 1e-6)
        remaining = (self._total - self._done) / max(rate, 1e-6)

        pct = self._done / max(self._total, 1)
        bar_len = 40
        filled = int(bar_len * pct)
        bar = "#" * filled + "-" * (bar_len - filled)

        sys.stderr.write(
            f"\r[{bar}] {self._done}/{self._total} "
            f"({pct*100:.1f}%) "
            f"| {task_name} ep {episode}/{total} "
            f"| {elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining"
        )
        sys.stderr.flush()

    def finish(self) -> None:
        elapsed = time.monotonic() - self._start_time
        sys.stderr.write(
            f"\nCompleted {self._done} episodes in {elapsed:.1f}s\n"
        )
        sys.stderr.flush()


# ---------------------------------------------------------------------------
# MuJoCo environment setup
# ---------------------------------------------------------------------------

def _try_build_mujoco_env() -> Any | None:
    """Attempt to build a TargetReaching MuJoCo environment."""
    try:
        from training.mujoco.target import TargetReaching, default_config
        config = default_config()
        config.enable_entity_slots = True
        env = TargetReaching(config=config)
        logger.info("MuJoCo TargetReaching environment initialized")
        return env
    except Exception as exc:
        logger.info("MuJoCo env unavailable, using heuristic sim: %s", exc)
        return None


def _try_build_skill_registry(checkpoint: str | None = None) -> SkillRegistry:
    """Build a skill registry, loading checkpoints where available."""
    registry = SkillRegistry()

    # Always register scripted skills
    try:
        from training.rl.skills.stand_skill import StandSkill
        registry.register(StandSkill())
    except ImportError:
        pass

    try:
        from training.rl.skills.turn_skill import TurnSkill
        registry.register(TurnSkill())
    except ImportError:
        pass

    try:
        from training.rl.skills.wave_skill import WaveSkill
        registry.register(WaveSkill())
    except ImportError:
        pass

    try:
        from training.rl.skills.bow_skill import BowSkill
        registry.register(BowSkill())
    except ImportError:
        pass

    # RL-trained skills
    try:
        from training.rl.skills.walk_skill import WalkSkill
        walk = WalkSkill()
        if checkpoint:
            try:
                walk.load_checkpoint(checkpoint)
            except Exception as exc:
                logger.warning("Failed to load walk checkpoint: %s", exc)
        registry.register(walk)
    except ImportError:
        pass

    try:
        from training.rl.skills.brax_walk_skill import BraxWalkSkill
        brax_walk = BraxWalkSkill()
        registry.register(brax_walk)
    except Exception:
        pass

    try:
        from training.rl.skills.brax_target_skill import BraxTargetSkill
        brax_target = BraxTargetSkill()
        registry.register(brax_target)
    except Exception:
        pass

    logger.info("Skill registry: %s", registry.list_skills())
    return registry


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RPG2Robot evaluation harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m evaluation.run_eval --planner rpg2robot --tasks all --episodes 500
  python -m evaluation.run_eval --planner scripted --tasks navigation --episodes 100
  python -m evaluation.run_eval --planner saycan --tasks walk_to_red_ball --episodes 50
  python -m evaluation.run_eval --planner rpg2robot --checkpoint ckpt/ --export-latex
        """,
    )
    parser.add_argument(
        "--planner",
        type=str,
        default="rpg2robot",
        choices=sorted(PLANNER_REGISTRY.keys()),
        help=f"Planner to evaluate (default: rpg2robot). Available: {sorted(PLANNER_REGISTRY.keys())}",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="all",
        help=(
            "Tasks to run: 'all', category name (navigation/manipulation/language), "
            "or comma-separated task names"
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes per task (default: 100)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/",
        help="Output directory for results (default: results/)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (default: 42)",
    )
    parser.add_argument(
        "--use-baselines-package",
        action="store_true",
        help=(
            "Use the full baselines/ package planners (with LLM calls, "
            "affordance scoring, recovery logic, etc.) instead of the "
            "simplified inline planners. Requires the baselines/ package "
            "to be importable."
        ),
    )
    parser.add_argument(
        "--use-mujoco",
        action="store_true",
        help="Use MuJoCo physics engine (requires mujoco + JAX)",
    )
    parser.add_argument(
        "--export-latex",
        action="store_true",
        help="Generate LaTeX table of results",
    )
    parser.add_argument(
        "--export-trajectories",
        action="store_true",
        help="Export full per-step trajectories (large file)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to partial results JSON to resume from",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress bar",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Resolve tasks
    try:
        tasks = resolve_tasks(args.tasks)
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    logger.info(
        "Evaluating planner=%s on %d tasks x %d episodes",
        args.planner,
        len(tasks),
        args.episodes,
    )
    logger.info("Tasks: %s", [t.name for t in tasks])

    # Build planner
    if args.use_baselines_package:
        try:
            planner = build_baselines_planner(
                args.planner, checkpoint=args.checkpoint
            )
            logger.info(
                "Using baselines/ package planner (adapted): %s", args.planner
            )
        except Exception as exc:
            logger.warning(
                "Failed to load baselines/ package planner: %s. "
                "Falling back to inline planner.",
                exc,
            )
            planner = build_planner(args.planner, checkpoint=args.checkpoint)
    else:
        planner = build_planner(args.planner, checkpoint=args.checkpoint)

    # Build skill executor
    registry = _try_build_skill_registry(args.checkpoint)
    executor = DefaultSkillExecutor(registry)

    # Build MuJoCo env if requested
    mujoco_env = None
    if args.use_mujoco:
        mujoco_env = _try_build_mujoco_env()

    # Build evaluator
    evaluator = Evaluator(
        planner=planner,
        skill_executor=executor,
        use_mujoco=args.use_mujoco,
        mujoco_env=mujoco_env,
        checkpoint=args.checkpoint or "",
        planner_name=args.planner,
    )

    # Progress bar
    progress: _ProgressBar | None = None
    progress_cb = None
    if not args.quiet:
        progress = _ProgressBar(len(tasks), args.episodes)
        progress_cb = progress.update

    # Run evaluation
    resume_path = Path(args.resume) if args.resume else None
    results = evaluator.evaluate(
        tasks=tasks,
        n_episodes=args.episodes,
        base_seed=args.seed,
        progress_callback=progress_cb,
        resume_from=resume_path,
    )

    if progress is not None:
        progress.finish()

    # Output directory
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Always export JSON + CSV
    json_path = out_dir / f"{args.planner}_results.json"
    csv_path = out_dir / f"{args.planner}_results.csv"
    export_json(results, json_path)
    export_csv(results, csv_path)
    logger.info("Results written to %s", json_path)
    logger.info("CSV written to %s", csv_path)

    # Optional exports
    if args.export_latex:
        tex_path = out_dir / f"{args.planner}_table.tex"
        tex = export_latex(results, tex_path)
        logger.info("LaTeX table written to %s", tex_path)
        print("\n" + tex)

    if args.export_trajectories:
        traj_path = out_dir / f"{args.planner}_trajectories.json"
        export_trajectories_json(results, traj_path)
        logger.info("Trajectories written to %s", traj_path)

    # Print summary to stdout
    _print_summary(results)


def _print_summary(results: EvalResults) -> None:
    """Print a human-readable summary table."""
    print("\n" + "=" * 80)
    print(f"  RPG2Robot Evaluation Summary: {results.planner_name}")
    print("=" * 80)

    header = (
        f"{'Task':<30} {'Succ%':>6} {'Plan%':>6} {'Grnd%':>6} "
        f"{'Recv%':>6} {'Steps':>7} {'Safety':>7}"
    )
    print(header)
    print("-" * 80)

    current_cat = ""
    for name in sorted(results.task_results.keys()):
        tr = results.task_results[name]
        if tr.category != current_cat:
            if current_cat:
                print("-" * 80)
            current_cat = tr.category
            print(f"  [{current_cat.upper()}]")

        row = (
            f"  {name:<28} "
            f"{tr.success_rate*100:>5.1f}% "
            f"{tr.planning_accuracy*100:>5.1f}% "
            f"{tr.grounding_accuracy*100:>5.1f}% "
            f"{tr.recovery_rate*100:>5.1f}% "
            f"{tr.mean_steps:>6.0f} "
            f"{tr.mean_safety_violations:>6.2f}"
        )
        print(row)

    print("=" * 80)
    print(
        f"  {'AGGREGATE':<28} "
        f"{results.aggregate_success_rate*100:>5.1f}% "
        f"{results.aggregate_planning_accuracy*100:>5.1f}% "
        f"{results.aggregate_grounding_accuracy*100:>5.1f}% "
        f"{results.aggregate_recovery_rate*100:>5.1f}% "
        f"{'--':>6} "
        f"{results.aggregate_safety_violations:>6.2f}"
    )
    print()

    # Per-category summary
    if results.category_success_rates:
        print("  Category success rates:")
        for cat, rate in sorted(results.category_success_rates.items()):
            print(f"    {cat:<20} {rate*100:.1f}%")
        print()


if __name__ == "__main__":
    main()
