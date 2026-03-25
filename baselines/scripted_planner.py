"""Scripted decision-tree planner baseline.

A hand-coded rule-based planner that pattern-matches the language instruction
to select an intent.  Entity matching uses simple string similarity on
entity type and label.

This baseline represents the traditional robotics approach of writing
explicit if/else logic.  It has no learning, no recovery behavior, and
fails permanently on the first error it cannot handle.

Design:
- Each rule checks ``instruction.lower()`` for keyword patterns.
- Entity resolution picks the entity whose label best matches the
  object noun extracted from the instruction.
- Multi-step tasks (e.g. "bring X to Y") are decomposed into a sequence
  stored in ``self._plan_queue``; only the head is returned each call.
"""

from __future__ import annotations

import logging
import re
from collections import deque
from difflib import SequenceMatcher
from typing import Any

from baselines.base_planner import BasePlanner, grounded_intent, idle_intent
from training.interfaces import CanonicalIntentType
from training.schema.embodied_context import ContextEntity, EmbodiedContext

logger = logging.getLogger(__name__)


class ScriptedPlanner(BasePlanner):
    """Hand-coded decision tree planner.

    Parameters
    ----------
    proximity_threshold:
        Maximum distance (metres) for an entity to be considered reachable
        for grasp/place actions without navigating first.
    name:
        Planner name for logging.
    """

    def __init__(
        self,
        proximity_threshold: float = 1.0,
        name: str = "scripted",
    ) -> None:
        super().__init__(name=name)
        self.proximity_threshold = proximity_threshold
        self._plan_queue: deque[dict[str, Any]] = deque()
        self._failed: bool = False
        self._failure_reason: str = ""

    # -- public API --------------------------------------------------------

    def plan(self, context: dict[str, Any] | EmbodiedContext) -> dict[str, Any]:
        """Plan using keyword matching on the instruction."""
        return self._timed_plan(context)

    def reset(self) -> None:
        super().reset()
        self._plan_queue.clear()
        self._failed = False
        self._failure_reason = ""

    # -- internals ---------------------------------------------------------

    def _do_plan(self, context: dict[str, Any] | EmbodiedContext) -> dict[str, Any]:
        ctx = self._ensure_context(context)

        # If permanently failed, always return IDLE
        if self._failed:
            return idle_intent(
                reasoning=f"Permanently failed: {self._failure_reason}"
            )

        # If we have queued sub-actions, pop and return the next one
        if self._plan_queue:
            return self._plan_queue.popleft()

        instruction = (ctx.language_instruction or ctx.task_description or "").strip().lower()
        if not instruction:
            return idle_intent(reasoning="No instruction provided.")

        # Try each rule in priority order
        for rule_fn in (
            self._rule_bring,
            self._rule_carry,
            self._rule_pick_up,
            self._rule_walk_to,
            self._rule_go_to,
            self._rule_face,
            self._rule_look_at,
            self._rule_wave,
            self._rule_speak,
            self._rule_stop,
        ):
            result = rule_fn(instruction, ctx)
            if result is not None:
                return result

        # No rule matched -- fallback to nearest entity
        if ctx.entities:
            nearest = min(ctx.entities, key=lambda e: e.distance_to_agent)
            return grounded_intent(
                intent="NAVIGATE_TO_ENTITY",
                target_entity_id=nearest.entity_id,
                target_entity_label=nearest.label,
                target_position=nearest.position,
                reasoning=f"Fallback: no rule matched, navigating to nearest entity {nearest.label!r}.",
            )

        return idle_intent(reasoning=f"No rule matched instruction: {instruction!r}")

    # -- rules -------------------------------------------------------------

    def _rule_walk_to(
        self, instruction: str, ctx: EmbodiedContext
    ) -> dict[str, Any] | None:
        """'walk to <X>' -> NAVIGATE_TO_ENTITY."""
        match = re.search(r"\bwalk\s+to\s+(?:the\s+)?(.+?)(?:\s*[.,!?]|$)", instruction)
        if not match:
            return None
        target_noun = match.group(1).strip()
        entity = self._resolve_entity(target_noun, ctx.entities)
        if entity is None:
            return self._fail(f"Cannot find entity matching {target_noun!r}")
        return grounded_intent(
            intent="NAVIGATE_TO_ENTITY",
            target_entity_id=entity.entity_id,
            target_entity_label=entity.label,
            target_position=entity.position,
            reasoning=f"Rule: walk to {entity.label!r}",
        )

    def _rule_go_to(
        self, instruction: str, ctx: EmbodiedContext
    ) -> dict[str, Any] | None:
        """'go to <X>' / 'navigate to <X>' / 'move to <X>' -> NAVIGATE_TO_ENTITY."""
        match = re.search(
            r"\b(?:go|navigate|move|approach)\s+(?:to\s+)?(?:the\s+)?(.+?)(?:\s*[.,!?]|$)",
            instruction,
        )
        if not match:
            return None
        target_noun = match.group(1).strip()
        entity = self._resolve_entity(target_noun, ctx.entities)
        if entity is None:
            return self._fail(f"Cannot find entity matching {target_noun!r}")
        return grounded_intent(
            intent="NAVIGATE_TO_ENTITY",
            target_entity_id=entity.entity_id,
            target_entity_label=entity.label,
            target_position=entity.position,
            reasoning=f"Rule: go to {entity.label!r}",
        )

    def _rule_pick_up(
        self, instruction: str, ctx: EmbodiedContext
    ) -> dict[str, Any] | None:
        """'pick up <X>' / 'grab <X>' / 'get <X>' -> navigate + grasp sequence."""
        match = re.search(
            r"\b(?:pick\s+up|grab|get|take)\s+(?:the\s+)?(.+?)(?:\s*[.,!?]|$)",
            instruction,
        )
        if not match:
            return None
        target_noun = match.group(1).strip()
        entity = self._resolve_entity(target_noun, ctx.entities)
        if entity is None:
            return self._fail(f"Cannot find entity matching {target_noun!r}")

        # Build a 2-step plan: navigate, then grasp
        nav_action = grounded_intent(
            intent="NAVIGATE_TO_ENTITY",
            target_entity_id=entity.entity_id,
            target_entity_label=entity.label,
            target_position=entity.position,
            source_action_name="navigate",
            reasoning=f"Rule: navigate to {entity.label!r} before pickup",
        )
        grasp_action = grounded_intent(
            intent="PICKUP_ENTITY",
            target_entity_id=entity.entity_id,
            target_entity_label=entity.label,
            target_position=entity.position,
            source_action_name="grasp",
            reasoning=f"Rule: pick up {entity.label!r}",
        )

        # If already close, skip navigation
        if entity.distance_to_agent <= self.proximity_threshold:
            return grasp_action

        # Queue the grasp for next call, return navigate now
        self._plan_queue.append(grasp_action)
        return nav_action

    def _rule_bring(
        self, instruction: str, ctx: EmbodiedContext
    ) -> dict[str, Any] | None:
        """'bring <X> to <Y>' -> navigate(X) + grasp(X) + navigate(Y) + place(X)."""
        match = re.search(
            r"\bbring\s+(?:the\s+)?(.+?)\s+to\s+(?:the\s+)?(.+?)(?:\s*[.,!?]|$)",
            instruction,
        )
        if not match:
            return None
        obj_noun = match.group(1).strip()
        dest_noun = match.group(2).strip()

        obj_entity = self._resolve_entity(obj_noun, ctx.entities)
        dest_entity = self._resolve_entity(dest_noun, ctx.entities)

        if obj_entity is None:
            return self._fail(f"Cannot find object entity matching {obj_noun!r}")
        if dest_entity is None:
            return self._fail(f"Cannot find destination entity matching {dest_noun!r}")

        # 4-step plan: navigate to object, grasp, navigate to destination, place
        steps: list[dict[str, Any]] = [
            grounded_intent(
                intent="NAVIGATE_TO_ENTITY",
                target_entity_id=obj_entity.entity_id,
                target_entity_label=obj_entity.label,
                target_position=obj_entity.position,
                source_action_name="navigate",
                reasoning=f"Rule: navigate to {obj_entity.label!r} for pickup",
            ),
            grounded_intent(
                intent="PICKUP_ENTITY",
                target_entity_id=obj_entity.entity_id,
                target_entity_label=obj_entity.label,
                target_position=obj_entity.position,
                source_action_name="grasp",
                reasoning=f"Rule: grasp {obj_entity.label!r}",
            ),
            grounded_intent(
                intent="NAVIGATE_TO_ENTITY",
                target_entity_id=dest_entity.entity_id,
                target_entity_label=dest_entity.label,
                target_position=dest_entity.position,
                source_action_name="navigate",
                reasoning=f"Rule: carry {obj_entity.label!r} to {dest_entity.label!r}",
            ),
            grounded_intent(
                intent="NAVIGATE_TO_POSITION",
                target_entity_id=dest_entity.entity_id,
                target_entity_label=dest_entity.label,
                target_position=dest_entity.position,
                source_action_name="place",
                reasoning=f"Rule: place {obj_entity.label!r} at {dest_entity.label!r}",
            ),
        ]

        # Skip initial navigate if already close to the object
        if obj_entity.distance_to_agent <= self.proximity_threshold:
            steps = steps[1:]

        # Return first step, queue the rest
        for step in steps[1:]:
            self._plan_queue.append(step)
        return steps[0]

    def _rule_carry(
        self, instruction: str, ctx: EmbodiedContext
    ) -> dict[str, Any] | None:
        """'carry <X> to <Y>' -- alias for bring."""
        match = re.search(
            r"\bcarry\s+(?:the\s+)?(.+?)\s+to\s+(?:the\s+)?(.+?)(?:\s*[.,!?]|$)",
            instruction,
        )
        if not match:
            return None
        # Rewrite as bring and delegate
        rewritten = instruction.replace("carry", "bring", 1)
        return self._rule_bring(rewritten, ctx)

    def _rule_face(
        self, instruction: str, ctx: EmbodiedContext
    ) -> dict[str, Any] | None:
        """'face <X>' / 'face the <X>' -> FACE_ENTITY."""
        match = re.search(r"\bface\s+(?:the\s+)?(.+?)(?:\s*[.,!?]|$)", instruction)
        if not match:
            return None
        target_noun = match.group(1).strip()
        entity = self._resolve_entity(target_noun, ctx.entities)
        if entity is None:
            return self._fail(f"Cannot find entity matching {target_noun!r}")
        return grounded_intent(
            intent="FACE_ENTITY",
            target_entity_id=entity.entity_id,
            target_entity_label=entity.label,
            target_position=entity.position,
            reasoning=f"Rule: face {entity.label!r}",
        )

    def _rule_look_at(
        self, instruction: str, ctx: EmbodiedContext
    ) -> dict[str, Any] | None:
        """'look at <X>' -> FACE_ENTITY."""
        match = re.search(r"\blook\s+at\s+(?:the\s+)?(.+?)(?:\s*[.,!?]|$)", instruction)
        if not match:
            return None
        target_noun = match.group(1).strip()
        entity = self._resolve_entity(target_noun, ctx.entities)
        if entity is None:
            return self._fail(f"Cannot find entity matching {target_noun!r}")
        return grounded_intent(
            intent="FACE_ENTITY",
            target_entity_id=entity.entity_id,
            target_entity_label=entity.label,
            target_position=entity.position,
            reasoning=f"Rule: look at {entity.label!r}",
        )

    def _rule_wave(
        self, instruction: str, ctx: EmbodiedContext
    ) -> dict[str, Any] | None:
        """'wave' / 'wave at' / 'bow' / 'dance' -> EMOTE."""
        match = re.search(r"\b(wave|bow|dance|greet|hello)\b", instruction)
        if not match:
            return None
        gesture = match.group(1)
        return grounded_intent(
            intent="EMOTE",
            source_action_name=gesture,
            reasoning=f"Rule: emote {gesture}",
        )

    def _rule_speak(
        self, instruction: str, ctx: EmbodiedContext
    ) -> dict[str, Any] | None:
        """'say <X>' / 'tell <entity> <X>' -> SPEAK."""
        match = re.search(r'\b(?:say|tell\s+\w+)\s+"?(.+?)"?\s*$', instruction)
        if not match:
            return None
        utterance = match.group(1).strip().strip('"')
        return grounded_intent(
            intent="SPEAK",
            source_action_name="speak",
            reasoning=f"Rule: speak '{utterance}'",
            constraints=[utterance],
        )

    def _rule_stop(
        self, instruction: str, ctx: EmbodiedContext
    ) -> dict[str, Any] | None:
        """'stop' / 'idle' / 'wait' -> IDLE."""
        if re.search(r"\b(stop|idle|wait|stand\s+still|stay)\b", instruction):
            return idle_intent(reasoning="Rule: stop/idle instruction")
        return None

    # -- entity resolution -------------------------------------------------

    @staticmethod
    def _resolve_entity(
        noun: str, entities: tuple[ContextEntity, ...]
    ) -> ContextEntity | None:
        """Find the entity whose label or type best matches the given noun.

        Uses ``SequenceMatcher`` ratio for fuzzy matching.  Returns ``None``
        if no entity scores above a minimum threshold.
        """
        if not entities:
            return None

        noun_lower = noun.lower()
        best_entity: ContextEntity | None = None
        best_score: float = 0.0
        min_threshold = 0.35

        for entity in entities:
            # Score against label
            label_score = SequenceMatcher(
                None, noun_lower, entity.label.lower()
            ).ratio()
            # Score against entity type
            type_score = SequenceMatcher(
                None, noun_lower, entity.entity_type.lower()
            ).ratio()
            # Exact substring bonus
            substring_bonus = 0.0
            if noun_lower in entity.label.lower():
                substring_bonus = 0.3
            elif entity.label.lower() in noun_lower:
                substring_bonus = 0.2

            score = max(label_score, type_score) + substring_bonus

            if score > best_score:
                best_score = score
                best_entity = entity

        if best_score >= min_threshold:
            return best_entity
        return None

    # -- failure -----------------------------------------------------------

    def _fail(self, reason: str) -> dict[str, Any]:
        """Mark the planner as permanently failed (no recovery)."""
        self._failed = True
        self._failure_reason = reason
        logger.warning("ScriptedPlanner permanent failure: %s", reason)
        return idle_intent(reasoning=f"Failed: {reason}")
