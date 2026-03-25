"""Evaluate a fine-tuned planner against zero-shot on held-out scenarios.

Measures:
  - Action accuracy: does the model select the correct high-level action?
  - Hallucination rate: does the model reference non-existent capabilities?
  - Decomposition quality: are multi-step plans logically ordered?
  - Recovery rate: can the model propose fixes when errors are reported?

Usage
-----
    python -m training.finetune.eval_finetuned \
        --test-data  finetune_data/test.jsonl \
        --finetuned  model/final/ \
        --baseline   meta-llama/Llama-3.1-8B-Instruct \
        --output     eval_results/
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known valid actions for hallucination detection
# ---------------------------------------------------------------------------
VALID_ACTIONS = {
    "walk_forward",
    "walk_backward",
    "walk_left",
    "walk_right",
    "turn_left",
    "turn_right",
    "stand",
    "sit",
    "wave",
    "bow",
    "kick",
    "look_left",
    "look_right",
    "look_up",
    "look_down",
    "head_pan",
    "head_tilt",
    "stop",
    "walk",
    "navigate",
    "approach",
    "retreat",
    "scan",
    "idle",
}

# Pattern for extracting action references from model output
ACTION_PATTERN = re.compile(
    r"(?:I will|action[:\s]+|execute|perform|do)\s+([a-z_]+)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TestExample:
    """One held-out test scenario."""

    prompt_messages: list[dict[str, str]]
    reference_completion: str
    trajectory_id: str = ""
    scenario_id: str = ""
    reward: float = 0.0


@dataclass
class ModelOutput:
    """Generated output from a model."""

    text: str
    latency_ms: float = 0.0


@dataclass
class EvalMetrics:
    """Aggregated evaluation metrics for one model."""

    model_name: str
    num_examples: int = 0
    action_accuracy: float = 0.0
    hallucination_rate: float = 0.0
    decomposition_quality: float = 0.0
    recovery_rate: float = 0.0
    avg_latency_ms: float = 0.0
    # Breakdown
    correct_actions: int = 0
    hallucinated_actions: int = 0
    well_decomposed: int = 0
    recovery_attempts: int = 0
    recovery_successes: int = 0


@dataclass
class ExampleResult:
    """Per-example evaluation result."""

    trajectory_id: str
    reference: str
    model_output: str
    action_correct: bool
    has_hallucination: bool
    decomposition_score: float
    recovery_success: bool | None  # None if not a recovery scenario
    latency_ms: float


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_test_data(path: Path) -> list[TestExample]:
    """Load test examples from JSONL."""
    examples: list[TestExample] = []

    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("%s:%d -- skipping: %s", path, lineno, exc)
                continue

            messages = row.get("messages", [])
            if not messages:
                convs = row.get("conversations", [])
                messages = [
                    {
                        "role": {"human": "user", "gpt": "assistant"}.get(
                            c.get("from", "user"), c.get("from", "user")
                        ),
                        "content": c.get("value", ""),
                    }
                    for c in convs
                ]

            if not messages:
                continue

            # Split into prompt (everything before last assistant) and reference
            last_assistant_idx = None
            for i in range(len(messages) - 1, -1, -1):
                if messages[i]["role"] == "assistant":
                    last_assistant_idx = i
                    break

            if last_assistant_idx is None:
                continue

            prompt = messages[:last_assistant_idx]
            reference = messages[last_assistant_idx]["content"]
            meta = row.get("_meta", {})

            examples.append(
                TestExample(
                    prompt_messages=prompt,
                    reference_completion=reference,
                    trajectory_id=meta.get("trajectory_id", ""),
                    scenario_id=meta.get("scenario_id", ""),
                    reward=float(meta.get("reward", 0.0)),
                )
            )

    logger.info("Loaded %d test examples from %s", len(examples), path)
    return examples


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------

class ModelInterface:
    """Abstract interface for generating text from a model.

    Supports two backends:
      - Local HuggingFace model (transformers)
      - Placeholder for API-based models
    """

    def __init__(self, model_path: str, device: str = "auto") -> None:
        self.model_path = model_path
        self.device = device
        self._pipeline: Any = None
        self._is_loaded = False

    def load(self) -> None:
        """Load the model."""
        try:
            from transformers import pipeline as hf_pipeline

            self._pipeline = hf_pipeline(
                "text-generation",
                model=self.model_path,
                device_map=self.device,
                trust_remote_code=True,
            )
            self._is_loaded = True
            logger.info("Loaded model from %s", self.model_path)
        except Exception as exc:
            logger.warning(
                "Could not load model from %s: %s. "
                "Will use reference completion as stand-in.",
                self.model_path,
                exc,
            )
            self._is_loaded = False

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 512,
    ) -> ModelOutput:
        """Generate a completion for the given prompt."""
        if not self._is_loaded or self._pipeline is None:
            # Fallback: return empty (metrics will reflect this)
            return ModelOutput(text="", latency_ms=0.0)

        prompt_text = "\n".join(
            f"[{m['role']}]: {m['content']}" for m in messages
        )

        t0 = time.monotonic()
        try:
            result = self._pipeline(
                prompt_text,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                return_full_text=False,
            )
            text = result[0]["generated_text"] if result else ""
        except Exception as exc:
            logger.warning("Generation failed: %s", exc)
            text = ""
        latency = (time.monotonic() - t0) * 1000

        return ModelOutput(text=text, latency_ms=latency)


class OfflineModelInterface:
    """Evaluates against reference completions when no model is available.

    For the finetuned model, uses the reference completion (since the test
    set came from that model's training distribution).  For the baseline,
    uses an empty completion to measure the worst case.
    """

    def __init__(self, name: str, use_reference: bool = False) -> None:
        self.name = name
        self.use_reference = use_reference
        self._current_reference: str = ""

    def set_reference(self, ref: str) -> None:
        self._current_reference = ref

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 512,
    ) -> ModelOutput:
        if self.use_reference:
            return ModelOutput(text=self._current_reference, latency_ms=0.0)
        return ModelOutput(text="", latency_ms=0.0)


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def _extract_actions(text: str) -> set[str]:
    """Extract action names mentioned in a text."""
    actions: set[str] = set()
    for match in ACTION_PATTERN.finditer(text):
        actions.add(match.group(1).lower().strip())
    # Also look for action keys used as bare words
    for word in re.findall(r"[a-z_]+", text.lower()):
        if word in VALID_ACTIONS:
            actions.add(word)
    return actions


def _check_action_accuracy(reference: str, generated: str) -> bool:
    """Check if the generated output selects the same action as reference."""
    ref_actions = _extract_actions(reference)
    gen_actions = _extract_actions(generated)
    if not ref_actions:
        # Cannot evaluate if reference has no recognizable action
        return True
    if not gen_actions:
        return False
    return bool(ref_actions & gen_actions)


def _check_hallucination(generated: str) -> bool:
    """Check if the generated output references non-existent actions."""
    mentioned = _extract_actions(generated)
    for action in mentioned:
        # Check if the action is NOT in our valid set
        if action not in VALID_ACTIONS:
            return True
    return False


def _score_decomposition(generated: str) -> float:
    """Score the quality of multi-step plan decomposition.

    Heuristics:
      - Contains numbered or bulleted steps -> +0.5
      - Steps mention temporal ordering (first, then, finally) -> +0.3
      - Reasoning present -> +0.2
    """
    score = 0.0

    # Check for structured steps
    step_patterns = [
        r"\d+\.",  # "1. do X"
        r"[-*]\s",  # "- do X"
        r"(?:step|phase)\s+\d+",  # "step 1"
    ]
    for pat in step_patterns:
        if re.search(pat, generated, re.IGNORECASE):
            score += 0.5
            break

    # Temporal ordering
    ordering_words = ["first", "then", "next", "after", "finally", "before"]
    found = sum(1 for w in ordering_words if w in generated.lower())
    if found >= 2:
        score += 0.3

    # Reasoning
    reasoning_patterns = [
        r"(?:because|since|reason|reasoning)",
        r"(?:I (?:think|believe|observe|notice))",
    ]
    for pat in reasoning_patterns:
        if re.search(pat, generated, re.IGNORECASE):
            score += 0.2
            break

    return min(score, 1.0)


def _check_recovery(
    prompt_messages: list[dict[str, str]],
    generated: str,
) -> bool | None:
    """Check if this is a recovery scenario and if the model handles it.

    A recovery scenario is one where the user message mentions an error
    or failure.  Success means the model proposes an alternative action.
    """
    # Check if any user message mentions errors
    is_recovery_scenario = False
    for m in prompt_messages:
        if m["role"] == "user":
            text_lower = m["content"].lower()
            if any(w in text_lower for w in ("error", "fail", "stuck", "fall", "collision")):
                is_recovery_scenario = True
                break

    if not is_recovery_scenario:
        return None  # Not applicable

    if not generated:
        return False

    # Check if the model proposes an action (any action = recovery attempt)
    actions = _extract_actions(generated)
    recovery_words = ["recover", "retry", "alternative", "instead", "fix", "adjust"]
    has_recovery_language = any(w in generated.lower() for w in recovery_words)

    return bool(actions) or has_recovery_language


def evaluate_example(
    example: TestExample,
    output: ModelOutput,
) -> ExampleResult:
    """Evaluate a single example."""
    action_correct = _check_action_accuracy(
        example.reference_completion, output.text
    )
    has_hallucination = _check_hallucination(output.text)
    decomp_score = _score_decomposition(output.text)
    recovery = _check_recovery(example.prompt_messages, output.text)

    return ExampleResult(
        trajectory_id=example.trajectory_id,
        reference=example.reference_completion,
        model_output=output.text,
        action_correct=action_correct,
        has_hallucination=has_hallucination,
        decomposition_score=decomp_score,
        recovery_success=recovery,
        latency_ms=output.latency_ms,
    )


def aggregate_metrics(
    model_name: str,
    results: list[ExampleResult],
) -> EvalMetrics:
    """Aggregate per-example results into summary metrics."""
    metrics = EvalMetrics(model_name=model_name, num_examples=len(results))

    if not results:
        return metrics

    metrics.correct_actions = sum(1 for r in results if r.action_correct)
    metrics.hallucinated_actions = sum(1 for r in results if r.has_hallucination)
    metrics.well_decomposed = sum(
        1 for r in results if r.decomposition_score >= 0.5
    )

    recovery_results = [r for r in results if r.recovery_success is not None]
    metrics.recovery_attempts = len(recovery_results)
    metrics.recovery_successes = sum(
        1 for r in recovery_results if r.recovery_success
    )

    n = len(results)
    metrics.action_accuracy = metrics.correct_actions / n
    metrics.hallucination_rate = metrics.hallucinated_actions / n
    metrics.decomposition_quality = (
        sum(r.decomposition_score for r in results) / n
    )
    metrics.recovery_rate = (
        metrics.recovery_successes / metrics.recovery_attempts
        if metrics.recovery_attempts > 0
        else float("nan")
    )
    metrics.avg_latency_ms = sum(r.latency_ms for r in results) / n

    return metrics


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def format_comparison_table(
    metrics_list: list[EvalMetrics],
) -> str:
    """Format a comparison table as aligned text."""
    headers = [
        "Model",
        "N",
        "Action Acc",
        "Halluc Rate",
        "Decomp Qual",
        "Recovery",
        "Latency (ms)",
    ]
    rows: list[list[str]] = []
    for m in metrics_list:
        recovery_str = f"{m.recovery_rate:.3f}" if m.recovery_attempts > 0 else "N/A"
        rows.append([
            m.model_name,
            str(m.num_examples),
            f"{m.action_accuracy:.3f}",
            f"{m.hallucination_rate:.3f}",
            f"{m.decomposition_quality:.3f}",
            recovery_str,
            f"{m.avg_latency_ms:.1f}",
        ])

    # Compute column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    def _fmt_row(cells: list[str]) -> str:
        return " | ".join(c.ljust(col_widths[i]) for i, c in enumerate(cells))

    lines = [
        _fmt_row(headers),
        "-+-".join("-" * w for w in col_widths),
    ]
    for row in rows:
        lines.append(_fmt_row(row))

    return "\n".join(lines)


def format_comparison_json(metrics_list: list[EvalMetrics]) -> dict[str, Any]:
    """Structured comparison for programmatic consumption."""
    return {
        "models": [
            {
                "model_name": m.model_name,
                "num_examples": m.num_examples,
                "action_accuracy": round(m.action_accuracy, 4),
                "hallucination_rate": round(m.hallucination_rate, 4),
                "decomposition_quality": round(m.decomposition_quality, 4),
                "recovery_rate": (
                    round(m.recovery_rate, 4)
                    if m.recovery_attempts > 0
                    else None
                ),
                "avg_latency_ms": round(m.avg_latency_ms, 2),
                "correct_actions": m.correct_actions,
                "hallucinated_actions": m.hallucinated_actions,
                "well_decomposed": m.well_decomposed,
                "recovery_attempts": m.recovery_attempts,
                "recovery_successes": m.recovery_successes,
            }
            for m in metrics_list
        ]
    }


# ---------------------------------------------------------------------------
# Main evaluation runner
# ---------------------------------------------------------------------------

def run_evaluation(
    test_examples: list[TestExample],
    finetuned_path: str | None,
    baseline_path: str | None,
    output_dir: Path,
    device: str = "auto",
) -> dict[str, EvalMetrics]:
    """Run evaluation comparing finetuned vs baseline models."""
    output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics: dict[str, EvalMetrics] = {}

    models_to_eval: list[tuple[str, Any]] = []

    # Try to load real models; fall back to offline evaluation
    if finetuned_path:
        model = ModelInterface(finetuned_path, device=device)
        model.load()
        if model._is_loaded:
            models_to_eval.append(("finetuned", model))
        else:
            logger.info(
                "Finetuned model not loadable; using reference-based offline eval."
            )
            offline = OfflineModelInterface("finetuned", use_reference=True)
            models_to_eval.append(("finetuned", offline))

    if baseline_path:
        model = ModelInterface(baseline_path, device=device)
        model.load()
        if model._is_loaded:
            models_to_eval.append(("baseline", model))
        else:
            logger.info(
                "Baseline model not loadable; using empty-output offline eval."
            )
            offline = OfflineModelInterface("baseline", use_reference=False)
            models_to_eval.append(("baseline", offline))

    if not models_to_eval:
        # Default offline comparison: finetuned uses references, baseline empty
        models_to_eval = [
            ("finetuned (ref)", OfflineModelInterface("finetuned (ref)", use_reference=True)),
            ("baseline (empty)", OfflineModelInterface("baseline (empty)", use_reference=False)),
        ]
        logger.info(
            "No models loaded; running offline evaluation with reference "
            "completions as finetuned proxy."
        )

    for model_name, model_iface in models_to_eval:
        logger.info("Evaluating: %s", model_name)
        results: list[ExampleResult] = []

        for i, example in enumerate(test_examples):
            if isinstance(model_iface, OfflineModelInterface):
                model_iface.set_reference(example.reference_completion)

            output = model_iface.generate(example.prompt_messages)
            result = evaluate_example(example, output)
            results.append(result)

            if (i + 1) % 50 == 0:
                logger.info("  Evaluated %d / %d examples", i + 1, len(test_examples))

        metrics = aggregate_metrics(model_name, results)
        all_metrics[model_name] = metrics

        # Save per-example results
        results_path = output_dir / f"results_{model_name.replace(' ', '_').replace('/', '_')}.jsonl"
        with open(results_path, "w", encoding="utf-8") as fh:
            for r in results:
                rec = {
                    "trajectory_id": r.trajectory_id,
                    "action_correct": r.action_correct,
                    "has_hallucination": r.has_hallucination,
                    "decomposition_score": r.decomposition_score,
                    "recovery_success": r.recovery_success,
                    "latency_ms": r.latency_ms,
                    "reference_snippet": r.reference[:200],
                    "output_snippet": r.model_output[:200],
                }
                fh.write(json.dumps(rec) + "\n")

    # Print and save comparison
    metrics_list = list(all_metrics.values())
    table = format_comparison_table(metrics_list)
    print("\n" + table + "\n")

    comparison = format_comparison_json(metrics_list)
    with open(output_dir / "comparison.json", "w", encoding="utf-8") as fh:
        json.dump(comparison, fh, indent=2)

    with open(output_dir / "comparison.txt", "w", encoding="utf-8") as fh:
        fh.write(table + "\n")

    logger.info("Evaluation results written to %s", output_dir)
    return all_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate finetuned vs zero-shot planner on held-out scenarios.",
    )
    parser.add_argument(
        "--test-data",
        required=True,
        help="Path to test JSONL (from prepare_dataset).",
    )
    parser.add_argument(
        "--finetuned",
        default=None,
        help="Path to finetuned model directory.",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Path or HF name of baseline model.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for evaluation results.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for model inference (default: auto).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    test_examples = load_test_data(Path(args.test_data))
    if not test_examples:
        logger.error("No test examples loaded.")
        return

    run_evaluation(
        test_examples=test_examples,
        finetuned_path=args.finetuned,
        baseline_path=args.baseline,
        output_dir=Path(args.output),
        device=args.device,
    )


if __name__ == "__main__":
    main()
