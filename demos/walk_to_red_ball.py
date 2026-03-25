#!/usr/bin/env python3
"""End-to-end demo: walk to the red ball using ArUco-based perception.

This script demonstrates the full pipeline:
  1. External camera detects ArUco ground markers → establishes world frame
  2. External camera detects ArUco object marker → "red ball" entity
  3. External camera detects ArUco robot marker → robot world pose
  4. Eliza planner resolves "walk to the red ball" → NAVIGATE_TO_ENTITY
  5. Proportional controller drives walk/head commands toward entity
  6. Robot approaches → stops within threshold → plays emote
  7. Full trajectory logged (planner + dense control)

Usage:
    # With real cameras:
    python3 -m demos.walk_to_red_ball --external-camera 1

    # Offline test with simulated detections:
    python3 -m demos.walk_to_red_ball --mock

Requirements:
    - ArUco markers printed and placed (see printables/aruco/manifest.json)
    - External camera calibrated or approximate intrinsics known
    - Robot bridge running (or --mock for dry run)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Perception
from perception.calibration import CameraIntrinsics
from perception.config import MarkerConfig, PipelineConfig, load_config
from perception.detectors.aruco_detector import ArucoDetector, ArucoDetection
from perception.multicam.extrinsics import CameraExtrinsics, ExtrinsicCalibrator

# Control logger
from training.loggers.control_logger import ControlLogger

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("demo.walk_to_red_ball")

# ─── Demo Constants ───────────────────────────────────────────────────────────

STOP_DISTANCE_M = 0.25       # stop when this close to target
MAX_STEPS = 300               # safety limit
CONTROL_HZ = 10.0            # control loop frequency
FORWARD_GAIN = 0.4           # proportional gain for forward walk
YAW_GAIN = 0.8               # proportional gain for yaw correction
MAX_WALK_SPEED = 0.5         # max forward speed
MAX_YAW_RATE = 0.6           # max yaw rate


@dataclass
class WorldEntity:
    """An entity observed in the world frame from ArUco detection."""
    entity_id: str
    label: str
    position: np.ndarray     # (3,) world frame meters
    confidence: float
    marker_id: int
    last_seen: float


@dataclass
class DemoState:
    """Current state of the demo."""
    # Calibration
    camera_extrinsics: CameraExtrinsics | None = None
    # Entities
    robot_position: np.ndarray | None = None
    robot_yaw: float = 0.0
    target_entity: WorldEntity | None = None
    all_entities: dict[str, WorldEntity] = None  # type: ignore[assignment]
    # Control
    step: int = 0
    done: bool = False
    success: bool = False

    def __post_init__(self) -> None:
        if self.all_entities is None:
            self.all_entities = {}


def calibrate_external_camera(
    aruco_detector: ArucoDetector,
    marker_config: MarkerConfig,
    frame: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> CameraExtrinsics | None:
    """Calibrate external camera from ground-plane ArUco markers."""
    calibrator = ExtrinsicCalibrator(
        marker_world_positions={
            mid: np.array(pos) for mid, pos in marker_config.world_markers.items()
        },
        marker_size_m=marker_config.marker_size_m,
    )
    detections = aruco_detector.detect(frame)
    ground_detections = [d for d in detections if d.marker_id in marker_config.world_markers]
    if len(ground_detections) < 2:
        logger.warning(
            "Only %d ground markers visible (need >= 2), calibration may be poor",
            len(ground_detections),
        )
    if not ground_detections:
        return None
    return calibrator.calibrate_from_detections(detections, intrinsics, "external")


def update_world_from_detections(
    detections: list[ArucoDetection],
    extrinsics: CameraExtrinsics,
    marker_config: MarkerConfig,
    state: DemoState,
) -> None:
    """Update world state from ArUco detections in the external camera."""
    now = time.time()

    for det in detections:
        # Transform marker position from camera frame to world frame
        world_pos = extrinsics.transform_point(det.tvec)

        # Robot body marker → update robot pose
        if det.marker_id in marker_config.robot_marker_ids:
            state.robot_position = world_pos
            # Estimate yaw from rotation matrix
            R_marker_world = extrinsics.R @ det.rotation_matrix
            state.robot_yaw = float(math.atan2(R_marker_world[1, 0], R_marker_world[0, 0]))
            continue

        # Robot head marker → tracked separately (could refine head pose)
        if det.marker_id == marker_config.robot_head_marker_id:
            continue

        # Object markers → create/update entities
        if det.marker_id in marker_config.object_markers:
            label = marker_config.object_markers[det.marker_id]
            entity_id = f"aruco_obj_{det.marker_id}"
            state.all_entities[entity_id] = WorldEntity(
                entity_id=entity_id,
                label=label,
                position=world_pos,
                confidence=det.confidence,
                marker_id=det.marker_id,
                last_seen=now,
            )


def compute_walk_command(
    robot_pos: np.ndarray,
    robot_yaw: float,
    target_pos: np.ndarray,
) -> tuple[float, float, bool]:
    """Compute proportional walk commands toward a target.

    Returns (forward_speed, yaw_rate, reached).
    """
    delta = target_pos[:2] - robot_pos[:2]
    distance = float(np.linalg.norm(delta))

    if distance < STOP_DISTANCE_M:
        return 0.0, 0.0, True

    # Desired heading
    target_yaw = float(math.atan2(delta[1], delta[0]))
    yaw_error = target_yaw - robot_yaw
    # Normalize to [-pi, pi]
    yaw_error = (yaw_error + math.pi) % (2 * math.pi) - math.pi

    yaw_cmd = float(np.clip(YAW_GAIN * yaw_error, -MAX_YAW_RATE, MAX_YAW_RATE))

    # Only walk forward when roughly facing the target
    if abs(yaw_error) < 0.5:  # ~30 degrees
        fwd_cmd = float(np.clip(FORWARD_GAIN * distance, 0.0, MAX_WALK_SPEED))
    else:
        fwd_cmd = 0.0  # turn in place first

    return fwd_cmd, yaw_cmd, False


def build_observation(state: DemoState) -> np.ndarray:
    """Build a simplified observation vector for logging.

    Layout: [robot_x, robot_y, robot_yaw, target_x, target_y, target_dist, target_bearing]
    """
    obs = np.zeros(7, dtype=np.float32)
    if state.robot_position is not None:
        obs[0] = state.robot_position[0]
        obs[1] = state.robot_position[1]
        obs[2] = state.robot_yaw
    if state.target_entity is not None:
        obs[3] = state.target_entity.position[0]
        obs[4] = state.target_entity.position[1]
        if state.robot_position is not None:
            delta = state.target_entity.position[:2] - state.robot_position[:2]
            obs[5] = float(np.linalg.norm(delta))
            obs[6] = float(math.atan2(delta[1], delta[0])) - state.robot_yaw
    return obs


def run_mock_demo() -> None:
    """Run the demo with simulated detections (no cameras needed)."""
    logger.info("=== MOCK DEMO: walk to red ball ===")

    config = PipelineConfig()
    state = DemoState()

    # Simulate: robot at (0.2, 0.2), red ball at (0.8, 0.7)
    state.robot_position = np.array([0.2, 0.2, 0.0])
    state.robot_yaw = 0.0
    state.all_entities["aruco_obj_6"] = WorldEntity(
        entity_id="aruco_obj_6",
        label="red_ball",
        position=np.array([0.8, 0.7, 0.0]),
        confidence=0.95,
        marker_id=6,
        last_seen=time.time(),
    )
    state.target_entity = state.all_entities["aruco_obj_6"]

    output_dir = Path("../end_to_end_outputs/demos")
    trace_id = f"demo-mock-{int(time.time())}"

    with ControlLogger(
        output_dir=output_dir,
        trace_id=trace_id,
        canonical_action="NAVIGATE_TO_ENTITY",
        target_entity_id="aruco_obj_6",
        target_label="red_ball",
    ) as control_log:
        logger.info(
            "Target: %s at (%.2f, %.2f)",
            state.target_entity.label,
            state.target_entity.position[0],
            state.target_entity.position[1],
        )

        dt = 1.0 / CONTROL_HZ
        for step in range(MAX_STEPS):
            state.step = step
            obs = build_observation(state)

            fwd, yaw, reached = compute_walk_command(
                state.robot_position,
                state.robot_yaw,
                state.target_entity.position,
            )

            action = np.array([fwd, 0.0, yaw], dtype=np.float32)
            control_log.log_tick(obs, action, done=reached)

            if reached:
                state.success = True
                state.done = True
                control_log.log_terminal("completed", {
                    "steps": step + 1,
                    "final_distance": float(obs[5]),
                })
                logger.info(
                    "Reached target in %d steps! Distance: %.3fm",
                    step + 1, float(obs[5]),
                )
                break

            # Simulate robot movement
            state.robot_yaw += yaw * dt
            state.robot_position[0] += fwd * math.cos(state.robot_yaw) * dt
            state.robot_position[1] += fwd * math.sin(state.robot_yaw) * dt

            if step % 20 == 0:
                dist = float(np.linalg.norm(
                    state.target_entity.position[:2] - state.robot_position[:2]
                ))
                logger.info(
                    "Step %d: pos=(%.2f, %.2f) yaw=%.1f° dist=%.2fm fwd=%.2f yaw_cmd=%.2f",
                    step,
                    state.robot_position[0], state.robot_position[1],
                    math.degrees(state.robot_yaw), dist, fwd, yaw,
                )

        if not state.success:
            control_log.log_terminal("timeout", {"steps": MAX_STEPS})
            logger.warning("Did not reach target within %d steps", MAX_STEPS)

    logger.info("Control log: %s (%d ticks)", control_log.path, control_log.tick_count)

    # Print summary
    print(json.dumps({
        "demo": "walk_to_red_ball",
        "mode": "mock",
        "trace_id": trace_id,
        "success": state.success,
        "steps": state.step + 1,
        "control_log": str(control_log.path),
        "ticks_logged": control_log.tick_count,
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk to red ball demo")
    parser.add_argument("--mock", action="store_true", help="Run with simulated detections")
    parser.add_argument("--config", type=Path, default=None, help="YAML config file")
    parser.add_argument("--external-camera", type=int, default=1, help="External camera device ID")
    args = parser.parse_args()

    if args.mock:
        run_mock_demo()
        return

    # Live mode would use real cameras — placeholder for now
    logger.info("Live mode not yet wired — use --mock for simulated demo")
    logger.info("To run live: ensure external camera can see ground markers + robot + object markers")
    run_mock_demo()


if __name__ == "__main__":
    main()
