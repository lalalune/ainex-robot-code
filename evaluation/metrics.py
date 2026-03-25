"""Metric computation and result data structures for RPG2Robot evaluation.

Provides dataclasses for per-step, per-episode, per-task, and aggregate
results, plus helpers to compute derived metrics and export to JSON, CSV,
and LaTeX table formats.
"""

from __future__ import annotations

import csv
import io
import json
import math
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence


# ---------------------------------------------------------------------------
# Per-step record
# ---------------------------------------------------------------------------

@dataclass
class StepRecord:
    """One control step within an episode."""
    step: int
    # Planner output
    planned_skill: str = ""
    planned_target_entity: str = ""
    planned_intent: str = ""
    # Executor output
    executed_skill: str = ""
    skill_status: str = ""         # running / completed / failed
    # State snapshot
    robot_xy: tuple[float, float] = (0.0, 0.0)
    robot_yaw: float = 0.0
    target_xy: tuple[float, float] = (0.0, 0.0)
    distance_to_target: float = float("inf")
    # Flags
    is_replan: bool = False        # True if planner re-planned this step
    safety_violation: bool = False  # True if joint limit or fall detected
    skill_failure: bool = False    # True if executor reported failure


# ---------------------------------------------------------------------------
# Per-episode result
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    """Result of running one episode of a task."""
    task_name: str
    episode_id: int
    seed: int

    success: bool = False
    steps_used: int = 0
    time_to_completion: float = float("inf")  # seconds (steps * ctrl_dt)

    # Detailed trajectory
    trajectory: list[StepRecord] = field(default_factory=list)

    # Planner-level accuracy (did the planner select correct skills?)
    planning_correct: bool = False

    # Grounding accuracy (did the planner ground to the correct entity?)
    grounding_correct: bool = False

    # Recovery tracking
    failure_injected: bool = False
    recovered_from_failure: bool = False

    # Safety
    total_safety_violations: int = 0

    # Sub-goal tracking (for multi-step tasks)
    sub_goals_completed: int = 0
    sub_goals_total: int = 0

    # Final distance to target (for shaped analysis)
    final_distance: float = float("inf")

    def ctrl_dt(self) -> float:
        """Control timestep -- default 50 Hz."""
        return 0.02

    def compute_time(self) -> None:
        """Compute wall-clock-equivalent time from steps."""
        if self.success:
            self.time_to_completion = self.steps_used * self.ctrl_dt()


# ---------------------------------------------------------------------------
# Per-task aggregated result
# ---------------------------------------------------------------------------

@dataclass
class TaskResult:
    """Aggregated metrics for one task across all episodes."""
    task_name: str
    category: str
    n_episodes: int = 0

    # Core metrics
    success_rate: float = 0.0
    planning_accuracy: float = 0.0
    grounding_accuracy: float = 0.0
    recovery_rate: float = 0.0

    # Time statistics (over successful episodes)
    mean_time_to_completion: float = float("inf")
    std_time_to_completion: float = 0.0
    median_time_to_completion: float = float("inf")

    # Steps statistics
    mean_steps: float = 0.0
    std_steps: float = 0.0

    # Distance statistics (over all episodes)
    mean_final_distance: float = float("inf")
    std_final_distance: float = 0.0

    # Safety
    mean_safety_violations: float = 0.0
    total_safety_violations: int = 0

    # Sub-goal completion rate (multi-step tasks)
    mean_sub_goal_completion: float = 0.0

    # Raw episode results
    episodes: list[EpisodeResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Full evaluation results
# ---------------------------------------------------------------------------

@dataclass
class EvalResults:
    """Complete evaluation results across all tasks."""
    planner_name: str
    checkpoint: str = ""
    n_episodes_per_task: int = 0
    total_episodes: int = 0

    # Per-task
    task_results: dict[str, TaskResult] = field(default_factory=dict)

    # Aggregates (computed via finalize())
    aggregate_success_rate: float = 0.0
    aggregate_planning_accuracy: float = 0.0
    aggregate_grounding_accuracy: float = 0.0
    aggregate_recovery_rate: float = 0.0
    aggregate_mean_time: float = float("inf")
    aggregate_safety_violations: float = 0.0

    # Per-category aggregates
    category_success_rates: dict[str, float] = field(default_factory=dict)

    def finalize(self) -> None:
        """Compute aggregate statistics from per-task results."""
        if not self.task_results:
            return
        tasks = list(self.task_results.values())
        self.total_episodes = sum(t.n_episodes for t in tasks)

        # Weighted-average success rate (equal weight per task)
        self.aggregate_success_rate = _safe_mean(
            [t.success_rate for t in tasks]
        )
        self.aggregate_planning_accuracy = _safe_mean(
            [t.planning_accuracy for t in tasks]
        )
        self.aggregate_grounding_accuracy = _safe_mean(
            [t.grounding_accuracy for t in tasks]
        )

        # Recovery rate: only count tasks that inject failures
        recovery_tasks = [t for t in tasks if any(
            e.failure_injected for e in t.episodes
        )]
        if recovery_tasks:
            self.aggregate_recovery_rate = _safe_mean(
                [t.recovery_rate for t in recovery_tasks]
            )

        # Mean time over successful episodes across all tasks
        all_times = []
        for t in tasks:
            for ep in t.episodes:
                if ep.success and math.isfinite(ep.time_to_completion):
                    all_times.append(ep.time_to_completion)
        self.aggregate_mean_time = _safe_mean(all_times)

        self.aggregate_safety_violations = _safe_mean(
            [t.mean_safety_violations for t in tasks]
        )

        # Per-category aggregates
        categories: dict[str, list[float]] = {}
        for t in tasks:
            categories.setdefault(t.category, []).append(t.success_rate)
        self.category_success_rates = {
            cat: _safe_mean(rates) for cat, rates in categories.items()
        }


# ---------------------------------------------------------------------------
# Metric computation helpers
# ---------------------------------------------------------------------------

def compute_success_rate(episodes: Sequence[EpisodeResult]) -> float:
    """Fraction of episodes that succeeded."""
    if not episodes:
        return 0.0
    return sum(1 for e in episodes if e.success) / len(episodes)


def compute_planning_accuracy(episodes: Sequence[EpisodeResult]) -> float:
    """Fraction of episodes where planner selected correct skill sequence."""
    if not episodes:
        return 0.0
    return sum(1 for e in episodes if e.planning_correct) / len(episodes)


def compute_grounding_accuracy(episodes: Sequence[EpisodeResult]) -> float:
    """Fraction of episodes where planner grounded to the correct entity."""
    if not episodes:
        return 0.0
    return sum(1 for e in episodes if e.grounding_correct) / len(episodes)


def compute_recovery_rate(episodes: Sequence[EpisodeResult]) -> float:
    """Of episodes with injected failure, fraction that recovered and succeeded."""
    failed_eps = [e for e in episodes if e.failure_injected]
    if not failed_eps:
        return 0.0
    return sum(1 for e in failed_eps if e.recovered_from_failure) / len(failed_eps)


def compute_time_stats(
    episodes: Sequence[EpisodeResult],
) -> tuple[float, float, float]:
    """Mean, std, median completion time for successful episodes."""
    times = [
        e.time_to_completion
        for e in episodes
        if e.success and math.isfinite(e.time_to_completion)
    ]
    if not times:
        return float("inf"), 0.0, float("inf")
    mean = statistics.mean(times)
    std = statistics.stdev(times) if len(times) > 1 else 0.0
    median = statistics.median(times)
    return mean, std, median


def compute_step_stats(
    episodes: Sequence[EpisodeResult],
) -> tuple[float, float]:
    """Mean and std of steps used across all episodes."""
    steps = [e.steps_used for e in episodes]
    if not steps:
        return 0.0, 0.0
    mean = statistics.mean(steps)
    std = statistics.stdev(steps) if len(steps) > 1 else 0.0
    return mean, std


def compute_distance_stats(
    episodes: Sequence[EpisodeResult],
) -> tuple[float, float]:
    """Mean and std of final distance to target across all episodes."""
    dists = [
        e.final_distance
        for e in episodes
        if math.isfinite(e.final_distance)
    ]
    if not dists:
        return float("inf"), 0.0
    mean = statistics.mean(dists)
    std = statistics.stdev(dists) if len(dists) > 1 else 0.0
    return mean, std


def compute_safety_violations(
    episodes: Sequence[EpisodeResult],
) -> tuple[float, int]:
    """Mean violations per episode, total violations."""
    total = sum(e.total_safety_violations for e in episodes)
    mean = total / len(episodes) if episodes else 0.0
    return mean, total


def compute_sub_goal_completion(
    episodes: Sequence[EpisodeResult],
) -> float:
    """Mean fraction of sub-goals completed per episode."""
    rates: list[float] = []
    for e in episodes:
        if e.sub_goals_total > 0:
            rates.append(e.sub_goals_completed / e.sub_goals_total)
    return _safe_mean(rates)


def build_task_result(
    task_name: str,
    category: str,
    episodes: list[EpisodeResult],
) -> TaskResult:
    """Aggregate a list of EpisodeResults into a TaskResult."""
    result = TaskResult(
        task_name=task_name,
        category=category,
        n_episodes=len(episodes),
        episodes=episodes,
    )
    result.success_rate = compute_success_rate(episodes)
    result.planning_accuracy = compute_planning_accuracy(episodes)
    result.grounding_accuracy = compute_grounding_accuracy(episodes)
    result.recovery_rate = compute_recovery_rate(episodes)

    mean_t, std_t, med_t = compute_time_stats(episodes)
    result.mean_time_to_completion = mean_t
    result.std_time_to_completion = std_t
    result.median_time_to_completion = med_t

    mean_s, std_s = compute_step_stats(episodes)
    result.mean_steps = mean_s
    result.std_steps = std_s

    mean_d, std_d = compute_distance_stats(episodes)
    result.mean_final_distance = mean_d
    result.std_final_distance = std_d

    mean_sv, total_sv = compute_safety_violations(episodes)
    result.mean_safety_violations = mean_sv
    result.total_safety_violations = total_sv

    result.mean_sub_goal_completion = compute_sub_goal_completion(episodes)

    return result


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def _sanitize_for_json(obj: Any) -> Any:
    """Recursively make an object JSON-serializable."""
    if isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return None
        return obj
    if isinstance(obj, (int, str, bool, type(None))):
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if hasattr(obj, "__dataclass_fields__"):
        return _sanitize_for_json(asdict(obj))
    return str(obj)


def export_json(results: EvalResults, path: Path | str) -> None:
    """Write full results to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Build a serializable snapshot (exclude raw trajectories for size)
    data: dict[str, Any] = {
        "planner": results.planner_name,
        "checkpoint": results.checkpoint,
        "n_episodes_per_task": results.n_episodes_per_task,
        "total_episodes": results.total_episodes,
        "aggregate": {
            "success_rate": results.aggregate_success_rate,
            "planning_accuracy": results.aggregate_planning_accuracy,
            "grounding_accuracy": results.aggregate_grounding_accuracy,
            "recovery_rate": results.aggregate_recovery_rate,
            "mean_time": results.aggregate_mean_time,
            "safety_violations": results.aggregate_safety_violations,
        },
        "category_success_rates": results.category_success_rates,
        "tasks": {},
    }
    for name, tr in results.task_results.items():
        task_dict: dict[str, Any] = {
            "category": tr.category,
            "n_episodes": tr.n_episodes,
            "success_rate": tr.success_rate,
            "planning_accuracy": tr.planning_accuracy,
            "grounding_accuracy": tr.grounding_accuracy,
            "recovery_rate": tr.recovery_rate,
            "mean_time_to_completion": tr.mean_time_to_completion,
            "std_time_to_completion": tr.std_time_to_completion,
            "median_time_to_completion": tr.median_time_to_completion,
            "mean_steps": tr.mean_steps,
            "std_steps": tr.std_steps,
            "mean_final_distance": tr.mean_final_distance,
            "std_final_distance": tr.std_final_distance,
            "mean_safety_violations": tr.mean_safety_violations,
            "total_safety_violations": tr.total_safety_violations,
            "mean_sub_goal_completion": tr.mean_sub_goal_completion,
        }
        data["tasks"][name] = task_dict

    data = _sanitize_for_json(data)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def export_csv(results: EvalResults, path: Path | str) -> None:
    """Write per-task summary to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "task", "category", "n_episodes",
        "success_rate", "planning_accuracy", "grounding_accuracy",
        "recovery_rate", "mean_time", "mean_steps",
        "mean_final_distance", "mean_safety_violations",
        "mean_sub_goal_completion",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, tr in sorted(results.task_results.items()):
            writer.writerow({
                "task": name,
                "category": tr.category,
                "n_episodes": tr.n_episodes,
                "success_rate": f"{tr.success_rate:.3f}",
                "planning_accuracy": f"{tr.planning_accuracy:.3f}",
                "grounding_accuracy": f"{tr.grounding_accuracy:.3f}",
                "recovery_rate": f"{tr.recovery_rate:.3f}",
                "mean_time": _fmt_float(tr.mean_time_to_completion),
                "mean_steps": f"{tr.mean_steps:.1f}",
                "mean_final_distance": _fmt_float(tr.mean_final_distance),
                "mean_safety_violations": f"{tr.mean_safety_violations:.2f}",
                "mean_sub_goal_completion": f"{tr.mean_sub_goal_completion:.3f}",
            })


def export_latex(results: EvalResults, path: Path | str | None = None) -> str:
    """Generate a LaTeX table of results.  Optionally write to file.

    Returns the LaTeX string.
    """
    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{RPG2Robot evaluation results (\texttt{" + _tex_escape(results.planner_name) + r"})}")
    lines.append(r"\label{tab:eval-results}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l c c c c c c}")
    lines.append(r"\toprule")
    lines.append(
        r"Task & Succ.\% & Plan.\% & Ground.\% & Recov.\% & Steps & Safety \\"
    )
    lines.append(r"\midrule")

    current_cat = ""
    for name in sorted(results.task_results.keys()):
        tr = results.task_results[name]
        # Category separator
        if tr.category != current_cat:
            if current_cat:
                lines.append(r"\midrule")
            current_cat = tr.category
            lines.append(
                r"\multicolumn{7}{l}{\textit{" + _tex_escape(current_cat) + r"}} \\"
            )

        row = (
            f"  {_tex_escape(name)} "
            f"& {tr.success_rate * 100:.1f} "
            f"& {tr.planning_accuracy * 100:.1f} "
            f"& {tr.grounding_accuracy * 100:.1f} "
            f"& {tr.recovery_rate * 100:.1f} "
            f"& {tr.mean_steps:.0f} "
            f"& {tr.mean_safety_violations:.2f} "
            r"\\"
        )
        lines.append(row)

    # Aggregate row
    lines.append(r"\midrule")
    agg = (
        r"  \textbf{Aggregate} "
        f"& \\textbf{{{results.aggregate_success_rate * 100:.1f}}} "
        f"& \\textbf{{{results.aggregate_planning_accuracy * 100:.1f}}} "
        f"& \\textbf{{{results.aggregate_grounding_accuracy * 100:.1f}}} "
        f"& \\textbf{{{results.aggregate_recovery_rate * 100:.1f}}} "
        f"& -- "
        f"& \\textbf{{{results.aggregate_safety_violations:.2f}}} "
        r"\\"
    )
    lines.append(agg)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    tex = "\n".join(lines)
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(tex)
    return tex


def export_trajectories_json(
    results: EvalResults,
    path: Path | str,
) -> None:
    """Write full per-step trajectories to a (potentially large) JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data: dict[str, Any] = {}
    for task_name, tr in results.task_results.items():
        episodes: list[dict[str, Any]] = []
        for ep in tr.episodes:
            ep_dict: dict[str, Any] = {
                "episode_id": ep.episode_id,
                "seed": ep.seed,
                "success": ep.success,
                "steps_used": ep.steps_used,
                "final_distance": ep.final_distance,
                "trajectory": [asdict(s) for s in ep.trajectory],
            }
            episodes.append(ep_dict)
        data[task_name] = episodes

    data = _sanitize_for_json(data)
    with open(path, "w") as f:
        json.dump(data, f, indent=1)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_mean(values: Sequence[float]) -> float:
    """Mean that returns 0.0 for empty sequences."""
    if not values:
        return 0.0
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return 0.0
    return statistics.mean(finite)


def _fmt_float(v: float) -> str:
    """Format a float for CSV, handling inf/nan."""
    if math.isinf(v) or math.isnan(v):
        return "N/A"
    return f"{v:.3f}"


def _tex_escape(s: str) -> str:
    """Minimal LaTeX escaping for table content."""
    return s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")
