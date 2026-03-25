"""SayCan-style affordance-weighted planner baseline.

Implements the SayCan algorithm (Ahn et al., 2022): an LLM proposes
candidate actions scored by *language usefulness*, while a separate
*affordance model* scores each action's physical feasibility given the
current world state.  The product of the two scores selects the action.

    selected = argmax_a [ P_LLM(a | instruction, history) * P_aff(a | state) ]

Key differences from the RPG2Robot system:
- No RPG training data; relies on a pretrained LLM + hardcoded affordances.
- Affordance model is a set of hand-tuned functions (distance, reachability,
  entity type).
- No recovery from failures -- the scored list is static per step.

The planner maintains a **skill library** of (intent_type, precondition)
pairs and scores every (skill, entity) combination each planning step.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from baselines.base_planner import BasePlanner, grounded_intent, idle_intent
from training.interfaces import CanonicalIntentType
from training.schema.embodied_context import ContextEntity, EmbodiedContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Skill library: each skill has a name, intent type, and affordance scorer
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Skill:
    """A single entry in the SayCan skill library."""
    name: str
    intent_type: str  # CanonicalIntentType value
    description: str
    # Natural-language verbs that signal this skill in instructions
    trigger_verbs: tuple[str, ...]
    # Whether this skill requires a target entity
    needs_entity: bool = True


# Default skill library for the AiNex robot
DEFAULT_SKILLS: tuple[Skill, ...] = (
    Skill(
        name="navigate",
        intent_type="NAVIGATE_TO_ENTITY",
        description="Walk to a specific entity",
        trigger_verbs=("walk", "go", "move", "navigate", "approach", "head"),
        needs_entity=True,
    ),
    Skill(
        name="face",
        intent_type="FACE_ENTITY",
        description="Turn to face a specific entity",
        trigger_verbs=("face", "look", "turn", "orient"),
        needs_entity=True,
    ),
    Skill(
        name="pick_up",
        intent_type="PICKUP_ENTITY",
        description="Pick up a nearby graspable entity",
        trigger_verbs=("pick", "grab", "get", "take", "grasp", "collect"),
        needs_entity=True,
    ),
    Skill(
        name="emote_wave",
        intent_type="EMOTE",
        description="Perform a wave gesture",
        trigger_verbs=("wave", "greet", "hello"),
        needs_entity=False,
    ),
    Skill(
        name="emote_bow",
        intent_type="EMOTE",
        description="Perform a bow gesture",
        trigger_verbs=("bow",),
        needs_entity=False,
    ),
    Skill(
        name="speak",
        intent_type="SPEAK",
        description="Say something",
        trigger_verbs=("say", "tell", "speak", "announce"),
        needs_entity=False,
    ),
    Skill(
        name="idle",
        intent_type="IDLE",
        description="Do nothing, stand still",
        trigger_verbs=("stop", "idle", "wait", "stand", "stay", "pause"),
        needs_entity=False,
    ),
)


# ---------------------------------------------------------------------------
# Affordance scoring functions
# ---------------------------------------------------------------------------

def _affordance_navigate(entity: ContextEntity, ctx: EmbodiedContext) -> float:
    """Affordance: can the agent navigate to this entity?

    Higher score for entities within a reasonable range.  Very close
    entities get a slightly lower score (already there).
    """
    d = entity.distance_to_agent
    if d < 0.3:
        return 0.3  # Already at target
    if d > 10.0:
        return 0.05  # Very far, unlikely reachable
    # Sigmoid-ish curve: peaks around 1-4 metres
    return max(0.05, 1.0 - d / 12.0)


def _affordance_face(entity: ContextEntity, ctx: EmbodiedContext) -> float:
    """Affordance: can the agent face this entity?

    Feasible for entities within visual range.  Very close entities are
    hard to look at meaningfully.
    """
    d = entity.distance_to_agent
    if d < 0.15:
        return 0.2
    if d > 15.0:
        return 0.1
    return max(0.1, 1.0 - d / 20.0)


def _affordance_pickup(entity: ContextEntity, ctx: EmbodiedContext) -> float:
    """Affordance: can the agent pick up this entity?

    Must be close (< 0.8m).  Objects and small entities are graspable;
    persons, landmarks, furniture are not.
    """
    d = entity.distance_to_agent
    etype = entity.entity_type.lower()

    # Non-graspable types
    if etype in ("person", "landmark", "furniture", "door"):
        return 0.02

    # Distance check
    if d > 2.0:
        return 0.05
    if d > 0.8:
        return 0.15

    # Close and graspable
    confidence_factor = max(0.1, entity.confidence)
    return 0.85 * confidence_factor


def _affordance_emote(entity: ContextEntity | None, ctx: EmbodiedContext) -> float:
    """Affordance: can the agent perform a gesture?

    Always feasible unless the robot is currently walking (harder to emote
    while in motion).
    """
    if ctx.is_walking:
        return 0.4
    return 0.9


def _affordance_speak(entity: ContextEntity | None, ctx: EmbodiedContext) -> float:
    """Affordance: can the agent speak? Always feasible."""
    return 0.9


def _affordance_idle(entity: ContextEntity | None, ctx: EmbodiedContext) -> float:
    """Affordance: can the agent idle? Always feasible."""
    return 0.95


_AFFORDANCE_FNS: dict[str, Callable[..., float]] = {
    "NAVIGATE_TO_ENTITY": _affordance_navigate,
    "FACE_ENTITY": _affordance_face,
    "PICKUP_ENTITY": _affordance_pickup,
    "EMOTE": _affordance_emote,
    "SPEAK": _affordance_speak,
    "IDLE": _affordance_idle,
}


# ---------------------------------------------------------------------------
# Candidate action (internal)
# ---------------------------------------------------------------------------

@dataclass
class _ScoredCandidate:
    """A (skill, entity) pair with language and affordance scores."""
    skill: Skill
    entity: ContextEntity | None
    language_score: float = 0.0
    affordance_score: float = 0.0

    @property
    def combined_score(self) -> float:
        return self.language_score * self.affordance_score

    def to_grounded_intent(self) -> dict[str, Any]:
        """Convert to a GroundedIntent dict."""
        kwargs: dict[str, Any] = {
            "intent": self.skill.intent_type,
            "source_action_name": self.skill.name,
            "reasoning": (
                f"SayCan: {self.skill.name}"
                f"({self.entity.label if self.entity else 'no-target'}) "
                f"score={self.combined_score:.3f} "
                f"(lang={self.language_score:.3f} x aff={self.affordance_score:.3f})"
            ),
        }
        if self.entity is not None:
            kwargs["target_entity_id"] = self.entity.entity_id
            kwargs["target_entity_label"] = self.entity.label
            kwargs["target_position"] = self.entity.position
        return grounded_intent(**kwargs)


# ---------------------------------------------------------------------------
# LLM scoring prompt
# ---------------------------------------------------------------------------

_SCORING_SYSTEM_PROMPT = """\
You are scoring candidate robot actions for relevance to an instruction.
For each candidate, output a relevance score between 0.0 and 1.0.

Output format (strict JSON array, no markdown):
[
  {"candidate_index": 0, "score": 0.8},
  {"candidate_index": 1, "score": 0.2},
  ...
]
"""


# ---------------------------------------------------------------------------
# SayCan planner
# ---------------------------------------------------------------------------

class SayCanPlanner(BasePlanner):
    """SayCan-style planner: LLM language-usefulness x affordance scoring.

    Parameters
    ----------
    model:
        LLM model identifier for language scoring.  Supports Claude and
        OpenAI models (same dispatch as ``ZeroShotPlanner``).
    skills:
        Override the default skill library.
    temperature:
        LLM sampling temperature.
    max_candidates:
        Maximum number of candidates to evaluate (limits API cost).
    use_heuristic_scoring:
        If ``True`` (or if no API key is available), use a heuristic
        keyword-matching scorer instead of the LLM.  Useful for
        deterministic evaluation runs.
    api_key:
        Optional API key override.
    name:
        Planner name for logging.
    """

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        skills: tuple[Skill, ...] | None = None,
        temperature: float = 0.0,
        max_candidates: int = 64,
        use_heuristic_scoring: bool = False,
        api_key: str | None = None,
        name: str = "saycan",
    ) -> None:
        super().__init__(name=name)
        self.model = model
        self.skills = skills or DEFAULT_SKILLS
        self.temperature = temperature
        self.max_candidates = max_candidates
        self.use_heuristic_scoring = use_heuristic_scoring
        self._api_key = api_key
        self._has_api_key = bool(
            api_key
            or os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )

    # -- public API --------------------------------------------------------

    def plan(self, context: dict[str, Any] | EmbodiedContext) -> dict[str, Any]:
        """Score all (skill, entity) candidates and return the best."""
        return self._timed_plan(context)

    def reset(self) -> None:
        super().reset()

    # -- internals ---------------------------------------------------------

    def _do_plan(self, context: dict[str, Any] | EmbodiedContext) -> dict[str, Any]:
        ctx = self._ensure_context(context)
        instruction = (ctx.language_instruction or ctx.task_description or "").strip()

        # 1. Generate candidate (skill, entity) pairs
        candidates = self._generate_candidates(ctx)
        if not candidates:
            return idle_intent(reasoning="SayCan: no candidates generated")

        # 2. Score language usefulness
        if self.use_heuristic_scoring or not self._has_api_key:
            self._score_language_heuristic(candidates, instruction)
        else:
            self._score_language_llm(candidates, instruction)

        # 3. Score affordances
        self._score_affordances(candidates, ctx)

        # 4. Select best by product of scores
        candidates.sort(key=lambda c: c.combined_score, reverse=True)
        best = candidates[0]

        logger.debug(
            "SayCan selected: %s(%s) score=%.3f (lang=%.3f x aff=%.3f)",
            best.skill.name,
            best.entity.label if best.entity else "none",
            best.combined_score,
            best.language_score,
            best.affordance_score,
        )

        return best.to_grounded_intent()

    # -- candidate generation ----------------------------------------------

    def _generate_candidates(self, ctx: EmbodiedContext) -> list[_ScoredCandidate]:
        """Generate all (skill, entity?) candidate pairs.

        Entity-requiring skills produce one candidate per visible entity.
        Non-entity skills produce exactly one candidate each.
        """
        candidates: list[_ScoredCandidate] = []

        for skill in self.skills:
            if skill.needs_entity:
                for entity in ctx.entities:
                    candidates.append(_ScoredCandidate(skill=skill, entity=entity))
            else:
                candidates.append(_ScoredCandidate(skill=skill, entity=None))

        # Limit to avoid excessive API calls
        if len(candidates) > self.max_candidates:
            candidates = candidates[: self.max_candidates]

        return candidates

    # -- language scoring --------------------------------------------------

    def _score_language_heuristic(
        self, candidates: list[_ScoredCandidate], instruction: str
    ) -> None:
        """Score language usefulness with keyword heuristics (no API call).

        Each candidate gets a score in [0.01, 1.0] based on:
        - Verb overlap between instruction and skill trigger_verbs
        - Entity label/type overlap with instruction nouns
        """
        inst_lower = instruction.lower()
        inst_words = set(re.findall(r"[a-z]+", inst_lower))

        for cand in candidates:
            score = 0.01  # base score so nothing is exactly zero

            # Verb match
            for verb in cand.skill.trigger_verbs:
                if verb in inst_lower:
                    score += 0.45
                    break
                # Partial match (verb is a prefix of an instruction word)
                for w in inst_words:
                    if w.startswith(verb) or verb.startswith(w):
                        score += 0.2
                        break

            # Entity match (if applicable)
            if cand.entity is not None:
                label_lower = cand.entity.label.lower()
                type_lower = cand.entity.entity_type.lower()
                label_words = set(re.findall(r"[a-z]+", label_lower))

                # Direct substring
                if label_lower in inst_lower:
                    score += 0.35
                elif type_lower in inst_lower:
                    score += 0.2
                else:
                    # Word overlap
                    overlap = inst_words & label_words
                    if overlap:
                        score += 0.15 * len(overlap)

                # Colour matching
                colours = {"red", "blue", "green", "yellow", "orange", "purple",
                           "white", "black", "pink", "brown", "gray", "grey"}
                for colour in colours:
                    if colour in inst_lower and colour in label_lower:
                        score += 0.2
                        break

            cand.language_score = min(score, 1.0)

    def _score_language_llm(
        self, candidates: list[_ScoredCandidate], instruction: str
    ) -> None:
        """Score language usefulness via LLM API call.

        Sends a batch prompt asking the LLM to score all candidates at once.
        Falls back to heuristic scoring on any API error.
        """
        # Build candidate descriptions
        descriptions: list[str] = []
        for i, cand in enumerate(candidates):
            entity_desc = (
                f"entity={cand.entity.label!r} ({cand.entity.entity_type})"
                if cand.entity
                else "no target entity"
            )
            descriptions.append(
                f"  {i}: {cand.skill.name} - {cand.skill.description} [{entity_desc}]"
            )

        user_prompt = (
            f"Instruction: \"{instruction}\"\n\n"
            f"Candidate actions:\n"
            + "\n".join(descriptions)
            + "\n\nScore each candidate's relevance to the instruction (0.0-1.0)."
        )

        try:
            raw = self._call_llm(user_prompt)
            scores = self._parse_scores(raw, len(candidates))
            for i, cand in enumerate(candidates):
                cand.language_score = scores.get(i, 0.1)
        except Exception:
            logger.exception("SayCan LLM scoring failed, falling back to heuristic")
            self._score_language_heuristic(candidates, instruction)

    def _call_llm(self, user_prompt: str) -> str:
        """Call the LLM API (Anthropic or OpenAI)."""
        model_lower = self.model.lower()

        if model_lower.startswith("claude"):
            import anthropic
            api_key = self._api_key or os.environ.get("ANTHROPIC_API_KEY", "")
            client = anthropic.Anthropic(api_key=api_key)
            message = client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=self.temperature,
                system=_SCORING_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            for block in message.content:
                if hasattr(block, "text"):
                    return block.text
            return ""
        else:
            import openai
            api_key = self._api_key or os.environ.get("OPENAI_API_KEY", "")
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=self.model,
                max_tokens=1024,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": _SCORING_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content or ""

    @staticmethod
    def _parse_scores(raw: str, n_candidates: int) -> dict[int, float]:
        """Parse LLM score response into {index: score} dict."""
        text = raw.strip()

        # Strip markdown fences
        fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1).strip()

        # Try to parse as JSON array
        try:
            arr = json.loads(text)
            if isinstance(arr, list):
                scores: dict[int, float] = {}
                for item in arr:
                    if isinstance(item, dict):
                        idx = int(item.get("candidate_index", -1))
                        sc = float(item.get("score", 0.1))
                        if 0 <= idx < n_candidates:
                            scores[idx] = max(0.0, min(1.0, sc))
                return scores
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # Fallback: try to extract individual score assignments
        scores = {}
        for m in re.finditer(r"(\d+)\s*[:=]\s*([\d.]+)", text):
            idx = int(m.group(1))
            sc = float(m.group(2))
            if 0 <= idx < n_candidates:
                scores[idx] = max(0.0, min(1.0, sc))
        return scores

    # -- affordance scoring ------------------------------------------------

    @staticmethod
    def _score_affordances(
        candidates: list[_ScoredCandidate], ctx: EmbodiedContext
    ) -> None:
        """Score physical affordances for every candidate."""
        for cand in candidates:
            aff_fn = _AFFORDANCE_FNS.get(cand.skill.intent_type)
            if aff_fn is None:
                cand.affordance_score = 0.5
                continue

            if cand.entity is not None:
                cand.affordance_score = aff_fn(cand.entity, ctx)
            else:
                # Non-entity skills (emote, speak, idle)
                cand.affordance_score = aff_fn(None, ctx)
