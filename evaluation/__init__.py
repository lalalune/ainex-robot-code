"""RPG2Robot evaluation harness.

Evaluates planner + skill-executor pipelines across 9 standardized tasks
spanning navigation, manipulation, and language-grounded control.

Usage:
    python -m evaluation.run_eval --planner rpg2robot --tasks all --episodes 500
"""

from evaluation.task_suite import (
    EvalTask,
    SuccessCriterion,
    TaskCategory,
    TASK_REGISTRY,
    get_tasks_by_category,
    get_task_by_name,
)
from evaluation.metrics import (
    StepRecord,
    EpisodeResult,
    TaskResult,
    EvalResults,
)
from evaluation.evaluator import Evaluator

__all__ = [
    "EvalTask",
    "SuccessCriterion",
    "TaskCategory",
    "TASK_REGISTRY",
    "get_tasks_by_category",
    "get_task_by_name",
    "StepRecord",
    "EpisodeResult",
    "TaskResult",
    "EvalResults",
    "Evaluator",
]
