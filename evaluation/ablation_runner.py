#!/usr/bin/env python3
"""
Ablation study runner for the RPG2Robot paper.

Runs systematic experiments varying one factor at a time:
1. RPG training data volume scaling
2. Provider ablation (remove one provider at a time)
3. Skill hierarchy depth (flat, 2-level, 3-level)
4. Cross-domain transfer (train on X, eval on Y)
5. Domain randomization ablation

Usage:
    python -m evaluation.ablation_runner --study data_scaling --output results/ablations/
    python -m evaluation.ablation_runner --study all --output results/ablations/
"""

import argparse
import json
import os
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional
from pathlib import Path


class AblationStudy(str, Enum):
    DATA_SCALING = "data_scaling"
    PROVIDER_ABLATION = "provider_ablation"
    HIERARCHY_DEPTH = "hierarchy_depth"
    CROSS_DOMAIN = "cross_domain"
    DOMAIN_RAND = "domain_rand"


@dataclass
class AblationConfig:
    """Configuration for a single ablation condition."""
    study: str
    condition_name: str
    description: str
    episodes_per_task: int = 100
    tasks: list = field(default_factory=lambda: ["all"])
    planner: str = "rpg2robot"
    planner_kwargs: dict = field(default_factory=dict)
    skill_config: dict = field(default_factory=dict)
    env_config: dict = field(default_factory=dict)


@dataclass
class AblationResult:
    """Results from a single ablation condition."""
    config: dict
    metrics: dict
    per_task: dict
    wall_time_sec: float
    timestamp: str


# ── Study 1: RPG Training Data Volume Scaling ────────────────────────────────

DATA_SCALING_CONDITIONS = [
    AblationConfig(
        study="data_scaling",
        condition_name="0_episodes",
        description="Zero-shot (no RPG data)",
        planner="zero_shot",
    ),
    AblationConfig(
        study="data_scaling",
        condition_name="500_episodes",
        description="500 RPG episodes",
        planner_kwargs={"rpg_episodes": 500},
    ),
    AblationConfig(
        study="data_scaling",
        condition_name="1000_episodes",
        description="1,000 RPG episodes",
        planner_kwargs={"rpg_episodes": 1000},
    ),
    AblationConfig(
        study="data_scaling",
        condition_name="2000_episodes",
        description="2,000 RPG episodes",
        planner_kwargs={"rpg_episodes": 2000},
    ),
    AblationConfig(
        study="data_scaling",
        condition_name="3000_episodes",
        description="3,000 RPG episodes",
        planner_kwargs={"rpg_episodes": 3000},
    ),
    AblationConfig(
        study="data_scaling",
        condition_name="5000_episodes",
        description="5,000 RPG episodes",
        planner_kwargs={"rpg_episodes": 5000},
    ),
    AblationConfig(
        study="data_scaling",
        condition_name="7000_episodes",
        description="7,000 RPG episodes",
        planner_kwargs={"rpg_episodes": 7000},
    ),
    AblationConfig(
        study="data_scaling",
        condition_name="10000_episodes",
        description="10,000 RPG episodes (full)",
        planner_kwargs={"rpg_episodes": 10000},
    ),
]


# ── Study 2: Provider Ablation ────────────────────────────────────────────────

ALL_PROVIDERS = [
    "gameState", "inventory", "skills", "equipment", "nearbyEntities",
    "availableActions", "quest", "map", "localChat", "socialMemory",
    "worldSnapshot", "decisionTrace", "personality", "guardrails", "possibilities",
]

PROVIDER_ABLATION_CONDITIONS = [
    AblationConfig(
        study="provider_ablation",
        condition_name="full",
        description="All 15 providers",
        planner_kwargs={"providers": ALL_PROVIDERS},
    ),
]

# Add one condition per removed provider
for provider in ["availableActions", "nearbyEntities", "worldSnapshot",
                 "decisionTrace", "socialMemory", "quest"]:
    remaining = [p for p in ALL_PROVIDERS if p != provider]
    PROVIDER_ABLATION_CONDITIONS.append(AblationConfig(
        study="provider_ablation",
        condition_name=f"minus_{provider}",
        description=f"Without {provider}",
        planner_kwargs={"providers": remaining},
    ))

# Minimal providers
PROVIDER_ABLATION_CONDITIONS.append(AblationConfig(
    study="provider_ablation",
    condition_name="minimal",
    description="Only gameState + nearbyEntities",
    planner_kwargs={"providers": ["gameState", "nearbyEntities"]},
))


# ── Study 3: Hierarchy Depth ─────────────────────────────────────────────────

HIERARCHY_DEPTH_CONDITIONS = [
    AblationConfig(
        study="hierarchy_depth",
        condition_name="flat",
        description="Single end-to-end policy (no hierarchy)",
        planner="flat_rl",
        skill_config={"hierarchy": "flat"},
    ),
    AblationConfig(
        study="hierarchy_depth",
        condition_name="2_level",
        description="Planner + monolithic skill policy",
        skill_config={"hierarchy": "2_level"},
    ),
    AblationConfig(
        study="hierarchy_depth",
        condition_name="3_level",
        description="Planner + meta-policy + individual skills (full)",
        skill_config={"hierarchy": "3_level"},
    ),
]


# ── Study 4: Cross-Domain Transfer ───────────────────────────────────────────

CROSS_DOMAIN_CONDITIONS = [
    AblationConfig(
        study="cross_domain",
        condition_name="hyperscape_to_hyperscape",
        description="Train on Hyperscape, eval on Hyperscape",
        env_config={"train_domain": "hyperscape", "eval_domain": "hyperscape"},
    ),
    AblationConfig(
        study="cross_domain",
        condition_name="hyperscape_to_mujoco",
        description="Train on Hyperscape, eval on MuJoCo",
        env_config={"train_domain": "hyperscape", "eval_domain": "mujoco"},
    ),
    AblationConfig(
        study="cross_domain",
        condition_name="mujoco_to_mujoco",
        description="Train on MuJoCo, eval on MuJoCo",
        env_config={"train_domain": "mujoco", "eval_domain": "mujoco"},
    ),
    AblationConfig(
        study="cross_domain",
        condition_name="hyperscape_to_real",
        description="Train on Hyperscape, eval on real AiNex",
        env_config={"train_domain": "hyperscape", "eval_domain": "real"},
    ),
    AblationConfig(
        study="cross_domain",
        condition_name="zero_shot_to_mujoco",
        description="Zero-shot (no training), eval on MuJoCo",
        planner="zero_shot",
        env_config={"train_domain": "none", "eval_domain": "mujoco"},
    ),
]


# ── Study 5: Domain Randomization ────────────────────────────────────────────

DOMAIN_RAND_CONDITIONS = [
    AblationConfig(
        study="domain_rand",
        condition_name="no_rand",
        description="No domain randomization",
        env_config={"domain_rand": False},
    ),
    AblationConfig(
        study="domain_rand",
        condition_name="friction_only",
        description="Only friction randomization",
        env_config={"domain_rand": True, "rand_params": ["friction"]},
    ),
    AblationConfig(
        study="domain_rand",
        condition_name="full_rand",
        description="Full domain randomization",
        env_config={"domain_rand": True, "rand_params": "all"},
    ),
]


STUDY_CONDITIONS = {
    AblationStudy.DATA_SCALING: DATA_SCALING_CONDITIONS,
    AblationStudy.PROVIDER_ABLATION: PROVIDER_ABLATION_CONDITIONS,
    AblationStudy.HIERARCHY_DEPTH: HIERARCHY_DEPTH_CONDITIONS,
    AblationStudy.CROSS_DOMAIN: CROSS_DOMAIN_CONDITIONS,
    AblationStudy.DOMAIN_RAND: DOMAIN_RAND_CONDITIONS,
}


def run_condition(config: AblationConfig, output_dir: str) -> AblationResult:
    """Run a single ablation condition and return results."""
    print(f"\n{'='*60}")
    print(f"Running: {config.condition_name}")
    print(f"  Study: {config.study}")
    print(f"  Description: {config.description}")
    print(f"  Planner: {config.planner}")
    print(f"  Episodes/task: {config.episodes_per_task}")
    print(f"{'='*60}")

    start_time = time.time()

    # In a full implementation, this would:
    # 1. Initialize the appropriate planner
    # 2. Initialize the MuJoCo environment with the right config
    # 3. Run the evaluation harness for each task
    # 4. Collect and aggregate metrics

    # For now, generate the evaluation command
    cmd_parts = [
        "python", "-m", "evaluation.run_eval",
        "--planner", config.planner,
        "--episodes", str(config.episodes_per_task),
        "--output", os.path.join(output_dir, config.condition_name),
    ]

    if config.tasks != ["all"]:
        cmd_parts.extend(["--tasks", ",".join(config.tasks)])

    for key, value in config.planner_kwargs.items():
        cmd_parts.extend([f"--planner-{key}", str(value)])

    for key, value in config.env_config.items():
        cmd_parts.extend([f"--env-{key}", str(value)])

    for key, value in config.skill_config.items():
        cmd_parts.extend([f"--skill-{key}", str(value)])

    print(f"  Command: {' '.join(cmd_parts)}")

    # Placeholder: in production, run the actual evaluation
    # For now, record the command and config
    elapsed = time.time() - start_time

    result = AblationResult(
        config=asdict(config),
        metrics={"status": "pending", "command": " ".join(cmd_parts)},
        per_task={},
        wall_time_sec=elapsed,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )

    # Save individual result
    result_path = os.path.join(output_dir, f"{config.condition_name}_result.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump(asdict(result), f, indent=2)

    return result


def run_study(study: AblationStudy, output_dir: str):
    """Run all conditions in an ablation study."""
    conditions = STUDY_CONDITIONS[study]
    study_dir = os.path.join(output_dir, study.value)
    os.makedirs(study_dir, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"# ABLATION STUDY: {study.value}")
    print(f"# Conditions: {len(conditions)}")
    print(f"# Output: {study_dir}")
    print(f"{'#'*60}")

    results = []
    for config in conditions:
        result = run_condition(config, study_dir)
        results.append(result)

    # Save combined results
    combined_path = os.path.join(study_dir, "all_results.json")
    with open(combined_path, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    # Generate summary table
    print(f"\n{'='*60}")
    print(f"SUMMARY: {study.value}")
    print(f"{'='*60}")
    print(f"{'Condition':<30} {'Status':<15}")
    print("-"*45)
    for r in results:
        print(f"{r.config['condition_name']:<30} {r.metrics.get('status', '?'):<15}")

    return results


def generate_latex_tables(output_dir: str):
    """Generate LaTeX tables from ablation results."""
    tables_dir = os.path.join(output_dir, "latex")
    os.makedirs(tables_dir, exist_ok=True)

    for study in AblationStudy:
        study_dir = os.path.join(output_dir, study.value)
        results_path = os.path.join(study_dir, "all_results.json")
        if not os.path.exists(results_path):
            continue

        with open(results_path) as f:
            results = json.load(f)

        # Generate a stub LaTeX table
        latex = f"% Auto-generated ablation table for {study.value}\n"
        latex += "\\begin{table}[t]\n\\centering\n"
        latex += f"\\caption{{\\textbf{{Ablation: {study.value.replace('_', ' ').title()}}}}}\n"
        latex += "\\scriptsize\n"
        latex += "\\begin{tabular}{@{}lc@{}}\n\\toprule\n"
        latex += "\\textbf{Condition} & \\textbf{Success (\\%)} \\\\\n\\midrule\n"

        for r in results:
            name = r['config']['condition_name'].replace('_', '\\_')
            status = r['metrics'].get('status', 'pending')
            latex += f"{name} & {status} \\\\\n"

        latex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"

        table_path = os.path.join(tables_dir, f"ablation_{study.value}.tex")
        with open(table_path, 'w') as f:
            f.write(latex)
        print(f"Saved LaTeX table: {table_path}")


def generate_run_script(output_dir: str):
    """Generate a shell script that runs all ablation conditions."""
    script_path = os.path.join(output_dir, "run_all_ablations.sh")
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Auto-generated script to run all ablation studies\n")
        f.write(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("set -e\n\n")

        for study in AblationStudy:
            conditions = STUDY_CONDITIONS[study]
            f.write(f"echo '=== Study: {study.value} ({len(conditions)} conditions) ==='\n")
            for config in conditions:
                cmd = (f"python -m evaluation.run_eval "
                       f"--planner {config.planner} "
                       f"--episodes {config.episodes_per_task} "
                       f"--output {output_dir}/{study.value}/{config.condition_name}")
                f.write(f"echo 'Running: {config.condition_name}'\n")
                f.write(f"{cmd}\n\n")
            f.write("\n")

        f.write("echo 'All ablation studies complete!'\n")

    os.chmod(script_path, 0o755)
    print(f"Saved run script: {script_path}")


def main():
    parser = argparse.ArgumentParser(description="Run ablation studies for RPG2Robot paper")
    parser.add_argument("--study", choices=[s.value for s in AblationStudy] + ["all"],
                        default="all", help="Which ablation study to run")
    parser.add_argument("--output", default="results/ablations",
                        help="Output directory for results")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Episodes per task per condition")
    parser.add_argument("--generate-scripts-only", action="store_true",
                        help="Only generate run scripts, don't execute")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)

    if args.generate_scripts_only:
        generate_run_script(output_dir)
        generate_latex_tables(output_dir)
        return

    if args.study == "all":
        for study in AblationStudy:
            # Update episodes per condition
            for cond in STUDY_CONDITIONS[study]:
                cond.episodes_per_task = args.episodes
            run_study(study, output_dir)
    else:
        study = AblationStudy(args.study)
        for cond in STUDY_CONDITIONS[study]:
            cond.episodes_per_task = args.episodes
        run_study(study, output_dir)

    generate_latex_tables(output_dir)
    generate_run_script(output_dir)
    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()
