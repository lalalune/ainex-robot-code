"""Abstract base class for all baseline planners.

Every planner receives an ``EmbodiedContext`` (as a dict or dataclass) and
returns a **GroundedIntent** -- a plain dict mirroring the fields of
``CanonicalIntent`` from ``training.interfaces``.

GroundedIntent schema::

    {
        "intent": str,          # CanonicalIntentType value, e.g. "NAVIGATE_TO_ENTITY"
        "target_entity_id": str,
        "target_entity_label": str,
        "target_position": [float, float, float],
        "source_action_name": str,
        "reasoning": str,
        "constraints": [str, ...],
    }
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from training.interfaces import CanonicalIntent, CanonicalIntentType
from training.schema.embodied_context import EmbodiedContext


# ---------------------------------------------------------------------------
# GroundedIntent helpers
# ---------------------------------------------------------------------------

def grounded_intent(
    intent: str | CanonicalIntentType,
    *,
    target_entity_id: str = "",
    target_entity_label: str = "",
    target_position: tuple[float, float, float] | list[float] = (0.0, 0.0, 0.0),
    source_action_name: str = "",
    reasoning: str = "",
    constraints: list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    """Convenience factory for a GroundedIntent dict."""
    intent_str = intent.value if isinstance(intent, CanonicalIntentType) else str(intent)
    return {
        "intent": intent_str,
        "target_entity_id": target_entity_id,
        "target_entity_label": target_entity_label,
        "target_position": list(target_position),
        "source_action_name": source_action_name,
        "reasoning": reasoning,
        "constraints": list(constraints),
    }


def grounded_intent_from_canonical(ci: CanonicalIntent) -> dict[str, Any]:
    """Convert a ``CanonicalIntent`` dataclass to a GroundedIntent dict."""
    return {
        "intent": ci.intent.value,
        "target_entity_id": ci.target_entity_id,
        "target_entity_label": ci.target_entity_label,
        "target_position": list(ci.target_position) if ci.target_position else [0.0, 0.0, 0.0],
        "source_action_name": ci.source_action_name,
        "reasoning": ci.reasoning,
        "constraints": list(ci.constraints),
    }


def idle_intent(reasoning: str = "No action determined.") -> dict[str, Any]:
    """Return an IDLE GroundedIntent."""
    return grounded_intent("IDLE", reasoning=reasoning)


# ---------------------------------------------------------------------------
# Planner metrics
# ---------------------------------------------------------------------------

@dataclass
class PlannerMetrics:
    """Lightweight metrics container collected during planning."""
    total_calls: int = 0
    total_latency_s: float = 0.0
    parse_failures: int = 0
    hallucinations: int = 0
    recovery_attempts: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def mean_latency_s(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.total_latency_s / self.total_calls

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_calls": self.total_calls,
            "total_latency_s": round(self.total_latency_s, 4),
            "mean_latency_s": round(self.mean_latency_s, 4),
            "parse_failures": self.parse_failures,
            "hallucinations": self.hallucinations,
            "recovery_attempts": self.recovery_attempts,
            **self.extra,
        }


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BasePlanner(ABC):
    """Abstract interface for baseline planners.

    Subclasses must implement ``plan()`` which maps an ``EmbodiedContext``
    (provided as a dict or dataclass) to a GroundedIntent dict.

    Lifecycle::

        planner = SomePlanner(...)
        planner.reset()                   # start of episode
        for step in episode:
            intent = planner.plan(ctx)    # EmbodiedContext -> GroundedIntent
        metrics = planner.metrics         # episode statistics
    """

    def __init__(self, name: str = "base") -> None:
        self.name = name
        self.metrics = PlannerMetrics()

    # -- public API --------------------------------------------------------

    @abstractmethod
    def plan(self, context: dict[str, Any] | EmbodiedContext) -> dict[str, Any]:
        """Map an observation context to a GroundedIntent dict.

        Parameters
        ----------
        context:
            Either an ``EmbodiedContext`` instance or a plain dict with the
            same schema (as produced by ``EmbodiedContext.to_dict()``).

        Returns
        -------
        dict[str, Any]
            A GroundedIntent dict.  See module docstring for schema.
        """
        ...

    def reset(self) -> None:
        """Reset planner state at the start of an episode."""
        self.metrics = PlannerMetrics()

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _ensure_context(context: dict[str, Any] | EmbodiedContext) -> EmbodiedContext:
        """Normalise the input to an ``EmbodiedContext`` instance."""
        if isinstance(context, EmbodiedContext):
            return context
        return EmbodiedContext.from_dict(context)

    def _timed_plan(self, context: dict[str, Any] | EmbodiedContext) -> dict[str, Any]:
        """Wrapper that records latency and call count. Subclasses can call
        ``_do_plan`` from here instead of ``plan`` directly."""
        t0 = time.monotonic()
        result = self._do_plan(context)
        dt = time.monotonic() - t0
        self.metrics.total_calls += 1
        self.metrics.total_latency_s += dt
        return result

    def _do_plan(self, context: dict[str, Any] | EmbodiedContext) -> dict[str, Any]:
        """Override in subclass when using ``_timed_plan``."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
