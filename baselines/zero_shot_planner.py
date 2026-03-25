"""Zero-shot LLM planner baseline.

Uses a pretrained LLM (Claude or OpenAI) with structured prompting to map
an ``EmbodiedContext`` to a ``GroundedIntent``.  No RPG training data is
used -- the model relies entirely on its pretrained world knowledge.

This baseline measures whether a general-purpose LLM can perform embodied
task planning without any domain-specific fine-tuning.

Tracked metrics:
- **hallucination_rate**: fraction of plans referencing non-existent entities
- **parse_failures**: responses that could not be parsed into a valid intent
- **latency**: per-call wall-clock time (includes API round-trip)
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from typing import Any

from baselines.base_planner import BasePlanner, grounded_intent, idle_intent, PlannerMetrics
from training.interfaces import CanonicalIntentType
from training.schema.embodied_context import ContextEntity, EmbodiedContext

logger = logging.getLogger(__name__)

# All valid intent type strings
_VALID_INTENTS: set[str] = {e.value for e in CanonicalIntentType}

# Default system prompt for the zero-shot planner
_SYSTEM_PROMPT = """\
You are a task planner for a humanoid robot.  Given the robot's current
perception (entities, state, instruction), output a single JSON action.

Valid intent types: {intents}

Output format (strict JSON, no markdown):
{{
  "intent": "<INTENT_TYPE>",
  "target_entity_id": "<entity_id or empty>",
  "target_entity_label": "<label or empty>",
  "target_position": [x, y, z],
  "source_action_name": "<skill name or empty>",
  "reasoning": "<one sentence explanation>"
}}

Rules:
- ONLY reference entities that appear in the provided entity list.
- If the instruction is unclear or no entities match, output intent "IDLE".
- Pick the single best next action, not a sequence.
""".format(intents=", ".join(sorted(_VALID_INTENTS)))


class ZeroShotPlanner(BasePlanner):
    """Zero-shot LLM planner -- no RPG training data, pure prompting.

    Parameters
    ----------
    model:
        Model identifier.  Supported prefixes:
        - ``"claude-"`` -> Anthropic API
        - ``"gpt-"`` / ``"o1-"`` / ``"o3-"`` -> OpenAI API
    temperature:
        Sampling temperature for the LLM.
    max_tokens:
        Maximum response tokens.
    api_key:
        Optional API key override.  Falls back to ``ANTHROPIC_API_KEY`` or
        ``OPENAI_API_KEY`` environment variables.
    system_prompt:
        Override the default system prompt.
    """

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0.0,
        max_tokens: int = 512,
        api_key: str | None = None,
        system_prompt: str | None = None,
        name: str = "zero_shot",
    ) -> None:
        super().__init__(name=name)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or _SYSTEM_PROMPT
        self._api_key = api_key
        self._backend = self._resolve_backend(model)

    # -- public API --------------------------------------------------------

    def plan(self, context: dict[str, Any] | EmbodiedContext) -> dict[str, Any]:
        """Generate a GroundedIntent via zero-shot LLM prompting."""
        return self._timed_plan(context)

    def reset(self) -> None:
        super().reset()

    # -- internals ---------------------------------------------------------

    def _do_plan(self, context: dict[str, Any] | EmbodiedContext) -> dict[str, Any]:
        ctx = self._ensure_context(context)
        entity_ids = {e.entity_id for e in ctx.entities}
        user_prompt = self._format_user_prompt(ctx)

        try:
            raw_response = self._call_llm(user_prompt)
        except Exception:
            logger.exception("LLM API call failed for %s", self.model)
            self.metrics.parse_failures += 1
            return self._fallback_action(ctx, reason="LLM API error")

        intent_dict = self._parse_response(raw_response)
        if intent_dict is None:
            self.metrics.parse_failures += 1
            return self._fallback_action(ctx, reason="Could not parse LLM response")

        # Hallucination check: does the plan reference a non-existent entity?
        ref_id = intent_dict.get("target_entity_id", "")
        if ref_id and ref_id not in entity_ids:
            self.metrics.hallucinations += 1
            logger.warning(
                "Hallucinated entity %r (available: %s)", ref_id, entity_ids
            )
            # Try to fix by finding closest match
            fixed = self._fix_entity_reference(ref_id, ctx.entities)
            if fixed:
                intent_dict["target_entity_id"] = fixed.entity_id
                intent_dict["target_entity_label"] = fixed.label
                intent_dict["reasoning"] += f" [entity corrected from {ref_id!r}]"
            else:
                return self._fallback_action(
                    ctx, reason=f"Hallucinated entity {ref_id!r}, no close match"
                )

        return intent_dict

    # -- prompt formatting -------------------------------------------------

    @staticmethod
    def _format_user_prompt(ctx: EmbodiedContext) -> str:
        """Build the user-turn prompt from an EmbodiedContext."""
        parts: list[str] = []

        # Entity list
        if ctx.entities:
            parts.append("Nearby entities:")
            for e in ctx.entities:
                bearing_desc = e.bearing_description()
                parts.append(
                    f"  - id={e.entity_id!r} label={e.label!r} type={e.entity_type} "
                    f"dist={e.distance_to_agent:.2f}m bearing={bearing_desc} "
                    f"conf={e.confidence:.2f}"
                )
        else:
            parts.append("No entities detected nearby.")

        # Agent state
        state_desc = "walking" if ctx.is_walking else "standing still"
        parts.append(f"\nAgent state: {state_desc}")
        if ctx.battery_mv > 0:
            parts.append(f"Battery: {ctx.battery_mv}mV")

        # Instruction
        if ctx.language_instruction:
            parts.append(f"\nInstruction: {ctx.language_instruction}")
        elif ctx.task_description:
            parts.append(f"\nTask: {ctx.task_description}")
        else:
            parts.append("\nNo instruction provided.")

        parts.append("\nOutput your JSON action:")
        return "\n".join(parts)

    # -- LLM API calls -----------------------------------------------------

    def _call_llm(self, user_prompt: str) -> str:
        """Dispatch to the appropriate LLM backend."""
        if self._backend == "anthropic":
            return self._call_anthropic(user_prompt)
        elif self._backend == "openai":
            return self._call_openai(user_prompt)
        else:
            raise ValueError(f"Unsupported backend: {self._backend}")

    def _call_anthropic(self, user_prompt: str) -> str:
        """Call the Anthropic Messages API."""
        import anthropic  # lazy import

        api_key = self._api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        client = anthropic.Anthropic(api_key=api_key)

        message = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=self.system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        # Extract text from the first content block
        for block in message.content:
            if hasattr(block, "text"):
                return block.text
        return ""

    def _call_openai(self, user_prompt: str) -> str:
        """Call the OpenAI Chat Completions API."""
        import openai  # lazy import

        api_key = self._api_key or os.environ.get("OPENAI_API_KEY", "")
        client = openai.OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        choice = response.choices[0]
        return choice.message.content or ""

    # -- response parsing --------------------------------------------------

    @staticmethod
    def _parse_response(raw: str) -> dict[str, Any] | None:
        """Parse an LLM response string into a GroundedIntent dict.

        Handles:
        - Pure JSON responses
        - JSON wrapped in markdown code fences
        - Minor formatting issues (trailing commas, unquoted keys)
        """
        if not raw or not raw.strip():
            return None

        text = raw.strip()

        # Strip markdown code fences
        fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1).strip()

        # Try direct JSON parse
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # Try to extract the first JSON object from the text
            brace_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
            if brace_match:
                try:
                    parsed = json.loads(brace_match.group(0))
                except json.JSONDecodeError:
                    return None
            else:
                return None

        if not isinstance(parsed, dict):
            return None

        # Validate and normalise
        intent_str = str(parsed.get("intent", "")).upper().strip()
        if intent_str not in _VALID_INTENTS:
            # Try to fuzzy-match
            for valid in _VALID_INTENTS:
                if intent_str in valid or valid in intent_str:
                    intent_str = valid
                    break
            else:
                return None

        target_pos = parsed.get("target_position", [0.0, 0.0, 0.0])
        if isinstance(target_pos, (list, tuple)) and len(target_pos) >= 3:
            target_pos = [float(target_pos[0]), float(target_pos[1]), float(target_pos[2])]
        else:
            target_pos = [0.0, 0.0, 0.0]

        return grounded_intent(
            intent=intent_str,
            target_entity_id=str(parsed.get("target_entity_id", "")),
            target_entity_label=str(parsed.get("target_entity_label", "")),
            target_position=target_pos,
            source_action_name=str(parsed.get("source_action_name", "")),
            reasoning=str(parsed.get("reasoning", "")),
        )

    # -- fallback / recovery -----------------------------------------------

    @staticmethod
    def _fallback_action(
        ctx: EmbodiedContext, reason: str = "fallback"
    ) -> dict[str, Any]:
        """Pick a random valid action when parsing fails.

        Prefers navigating to a nearby entity if any are visible.
        """
        if ctx.entities:
            target = random.choice(ctx.entities)
            return grounded_intent(
                intent="NAVIGATE_TO_ENTITY",
                target_entity_id=target.entity_id,
                target_entity_label=target.label,
                target_position=target.position,
                reasoning=f"Fallback: {reason}. Navigating to random entity {target.label!r}.",
            )
        return idle_intent(reasoning=f"Fallback: {reason}. No entities available.")

    @staticmethod
    def _fix_entity_reference(
        bad_id: str, entities: tuple[ContextEntity, ...]
    ) -> ContextEntity | None:
        """Try to find the closest matching entity by ID or label substring."""
        bad_lower = bad_id.lower()
        best: ContextEntity | None = None
        best_score = 0

        for e in entities:
            # Check ID substring overlap
            score = 0
            if bad_lower in e.entity_id.lower():
                score += 3
            if bad_lower in e.label.lower():
                score += 2
            if e.entity_id.lower() in bad_lower:
                score += 1
            if e.label.lower() in bad_lower:
                score += 1
            if score > best_score:
                best_score = score
                best = e

        return best if best_score > 0 else None

    # -- backend resolution ------------------------------------------------

    @staticmethod
    def _resolve_backend(model: str) -> str:
        """Determine API backend from model name."""
        model_lower = model.lower()
        if model_lower.startswith("claude"):
            return "anthropic"
        if any(model_lower.startswith(p) for p in ("gpt-", "o1-", "o3-", "o4-")):
            return "openai"
        # Default to anthropic for unknown models
        logger.warning("Unknown model prefix %r, defaulting to Anthropic backend", model)
        return "anthropic"
