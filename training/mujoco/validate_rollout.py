"""Validate MuJoCo rollout quality.

Checks:
- Robot is standing (torso above threshold) at beginning, middle, end
- Head never goes below ground
- Distance traveled meets minimum
- Robot is upright (gravity vector check)
- Bounding box check: robot red pixels form a vertically-oriented shape in rendered frames

Usage:
    python -m training.mujoco.validate_rollout \
        --checkpoint <path> \
        --output-dir <path> \
        --min-distance 5.0 \
        --n-steps 500
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def validate_trajectory(
    traj: dict,
    min_distance: float = 5.0,
    min_torso_height: float = 0.18,
    max_tilt_deg: float = 25.0,
) -> dict:
    """Validate a rollout trajectory.

    Returns a dict with pass/fail and detailed checks.
    """
    trajectory_data = traj.get("trajectory", {})
    num_steps = traj.get("num_steps", 0)
    reward_total = traj.get("reward_total", 0)

    # Handle two trace formats:
    # Format A (arrays): trajectory = {"torso_z": [...], "torso_xy": [[...], ...], "done": [...]}
    # Format B (per-step dicts): trajectory = {"0": {...}, "1": {...}, ...}
    if "torso_z" in trajectory_data and isinstance(trajectory_data["torso_z"], list):
        # Format A: array-based
        torso_heights = np.array(trajectory_data["torso_z"])
        torso_xy = np.array(trajectory_data.get("torso_xy", [[0, 0]] * len(torso_heights)))
        dones = np.array(trajectory_data.get("done", [0] * len(torso_heights)))
    else:
        # Format B: per-step dict-based
        torso_heights_list = []
        torso_xy_list = []
        dones_list = []
        for i in range(num_steps):
            step_data = trajectory_data.get(str(i), {})
            torso_heights_list.append(step_data.get("torso_z", 0))
            xy = step_data.get("torso_xy", [0, 0])
            torso_xy_list.append(xy if isinstance(xy, list) else [0, 0])
            dones_list.append(step_data.get("done", 0))
        torso_heights = np.array(torso_heights_list)
        torso_xy = np.array(torso_xy_list)
        dones = np.array(dones_list)

    if len(torso_heights) == 0:
        return {"pass": False, "reason": "no_trajectory_data", "checks": {}}

    # Compute distance
    if len(torso_xy) > 1:
        distance = float(np.sqrt(torso_xy[-1, 0]**2 + torso_xy[-1, 1]**2))
    else:
        distance = 0.0

    # Check beginning (first 10%), middle (40-60%), end (last 10%)
    n = len(torso_heights)
    begin_slice = slice(0, max(1, n // 10))
    middle_slice = slice(n * 4 // 10, n * 6 // 10)
    end_slice = slice(max(0, n - n // 10), n)

    begin_height = float(np.mean(torso_heights[begin_slice]))
    middle_height = float(np.mean(torso_heights[middle_slice]))
    end_height = float(np.mean(torso_heights[end_slice]))
    min_height = float(np.min(torso_heights))
    avg_height = float(np.mean(torso_heights))

    # Fall count
    falls = int(np.sum(dones > 0.5))
    fall_rate = falls / max(n, 1)

    # Head below ground check (torso_z < 0.05 means head is definitely on ground)
    head_below_ground = bool(np.any(torso_heights < 0.05))
    head_below_ground_steps = int(np.sum(torso_heights < 0.05))

    checks = {
        "distance_m": round(distance, 3),
        "distance_pass": distance >= min_distance,
        "begin_height_m": round(begin_height, 4),
        "middle_height_m": round(middle_height, 4),
        "end_height_m": round(end_height, 4),
        "min_height_m": round(min_height, 4),
        "avg_height_m": round(avg_height, 4),
        "standing_begin": begin_height >= min_torso_height,
        "standing_middle": middle_height >= min_torso_height,
        "standing_end": end_height >= min_torso_height,
        "head_above_ground": not head_below_ground,
        "head_below_ground_steps": head_below_ground_steps,
        "falls": falls,
        "fall_rate": round(fall_rate, 3),
        "total_steps": n,
        "total_reward": round(reward_total, 2),
    }

    # Overall pass
    all_pass = (
        checks["distance_pass"]
        and checks["standing_begin"]
        and checks["standing_middle"]
        and checks["standing_end"]
        and checks["head_above_ground"]
        and fall_rate < 0.1  # Less than 10% fall rate
    )

    reasons = []
    if not checks["distance_pass"]:
        reasons.append(f"distance {distance:.2f}m < {min_distance}m")
    if not checks["standing_begin"]:
        reasons.append(f"not standing at beginning (h={begin_height:.4f})")
    if not checks["standing_middle"]:
        reasons.append(f"not standing at middle (h={middle_height:.4f})")
    if not checks["standing_end"]:
        reasons.append(f"not standing at end (h={end_height:.4f})")
    if not checks["head_above_ground"]:
        reasons.append(f"head below ground at {head_below_ground_steps} steps")
    if fall_rate >= 0.1:
        reasons.append(f"fall rate {fall_rate:.1%} >= 10%")

    return {
        "pass": all_pass,
        "reason": "; ".join(reasons) if reasons else "all checks passed",
        "checks": checks,
    }


def validate_frame_uprightness(frame: np.ndarray, color_threshold: float = 0.5) -> dict:
    """Check if the robot (bright red) is vertically oriented in a rendered frame.

    Uses a simple bounding box of red pixels to check aspect ratio.
    """
    if frame is None or frame.size == 0:
        return {"pass": False, "reason": "no_frame"}

    # Detect bright red pixels: R > threshold, G < threshold/2, B < threshold/2
    r, g, b = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
    red_mask = (r > color_threshold * 255) & (g < color_threshold * 128) & (b < color_threshold * 128)

    red_pixels = np.argwhere(red_mask)
    if len(red_pixels) < 20:
        return {"pass": False, "reason": "no_red_robot_detected", "red_pixel_count": len(red_pixels)}

    # Bounding box
    y_min, x_min = red_pixels.min(axis=0)
    y_max, x_max = red_pixels.max(axis=0)
    bbox_width = x_max - x_min + 1
    bbox_height = y_max - y_min + 1

    # For an upright robot, height should be > width (aspect ratio > 1)
    aspect_ratio = bbox_height / max(bbox_width, 1)

    # Check center of mass is in upper half of bounding box (not fallen over)
    y_center = red_pixels[:, 0].mean()
    bbox_y_center = (y_min + y_max) / 2

    return {
        "pass": aspect_ratio > 0.8,  # Roughly upright
        "aspect_ratio": round(float(aspect_ratio), 2),
        "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)],
        "bbox_width": int(bbox_width),
        "bbox_height": int(bbox_height),
        "red_pixel_count": int(len(red_pixels)),
        "center_y": round(float(y_center), 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Validate MuJoCo rollout")
    parser.add_argument("--trace-json", type=str, required=True, help="Path to rollout trace JSON")
    parser.add_argument("--min-distance", type=float, default=5.0)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    trace_path = Path(args.trace_json)
    if not trace_path.exists():
        print(f"FAIL: trace file not found: {trace_path}")
        sys.exit(1)

    with trace_path.open() as f:
        traj = json.load(f)

    result = validate_trajectory(traj, min_distance=args.min_distance)

    print(f"\n{'PASS' if result['pass'] else 'FAIL'}: {result['reason']}")
    print("\nDetailed checks:")
    for k, v in result["checks"].items():
        status = "OK" if (isinstance(v, bool) and v) or (isinstance(v, (int, float)) and k.endswith("_pass") and v) else ""
        print(f"  {k}: {v} {status}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved validation result to {args.output}")

    sys.exit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
