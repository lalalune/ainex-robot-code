"""Baseline planners for RPG2Robot ablation comparison.

Each planner maps an EmbodiedContext to a GroundedIntent (dict form of
CanonicalIntent) for evaluation against the full RPG2Robot system.

Planners
--------
- **ZeroShotPlanner** -- LLM zero-shot structured prompting (no RPG data).
- **ScriptedPlanner** -- Hand-coded decision tree with string matching.
- **SayCanPlanner** -- SayCan-style affordance-weighted LLM scoring.
- **FlatRLPlanner** -- End-to-end PPO policy, no planner/skill hierarchy.
- **RPG2RobotPlanner** -- Full system: RPG-fine-tuned LLM + grounding + recovery.
"""

from baselines.base_planner import BasePlanner
from baselines.flat_rl_planner import FlatRLPlanner
from baselines.rpg2robot_planner import RPG2RobotPlanner
from baselines.saycan_planner import SayCanPlanner
from baselines.scripted_planner import ScriptedPlanner
from baselines.zero_shot_planner import ZeroShotPlanner

__all__ = [
    "BasePlanner",
    "FlatRLPlanner",
    "RPG2RobotPlanner",
    "SayCanPlanner",
    "ScriptedPlanner",
    "ZeroShotPlanner",
]
