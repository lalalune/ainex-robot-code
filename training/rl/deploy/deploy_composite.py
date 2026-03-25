"""Deploy composite walking + upper body policy to the real AiNex robot.

Extends the deploy_walking.py pattern to control all 24 joints using
CompositeSkill (BraxWalkSkill + UpperBodySkill). Supports wave, and
any future upper body tasks trained via train_upper.py.

Architecture:
    deploy_composite.py (20-50Hz loop)
        → CompositeSkill.get_full_action()
        → 24-dim joint targets (radians)
        → joint_name → servo_id + radians → pulse conversion
        → WebSocket → bridge server → ROS backend → bus servos

Usage:
    # Dry run (log commands, don't send):
    python3 -m training.rl.deploy.deploy_composite --dry-run

    # Wave while walking:
    python3 -m training.rl.deploy.deploy_composite --task wave --hz 20 --duration 10

    # Test with mock backend:
    python3 -m training.rl.deploy.deploy_composite --bridge ws://localhost:9100 --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import signal
import time
from pathlib import Path

import numpy as np

from training.rl.skills.composite_skill import CompositeSkill, NUM_TOTAL_JOINTS
from training.rl.skills.rl_wave_skill import (
    TASK_OBS_DIM,
    WAVE_AMPLITUDE,
    WAVE_ELBOW_PITCH,
    WAVE_ELBOW_YAW,
    WAVE_FREQUENCY,
    WAVE_SHOULDER_PITCH,
)
from training.mujoco.ainex_constants import ALL_JOINT_NAMES, LEG_JOINT_NAMES
from bridge.isaaclab.joint_map import (
    joint_name_to_servo_id,
    radians_to_pulse,
)

# Safety limits
MAX_JOINT_DELTA = 0.1      # rad — max single-step change
FALL_PITCH_THRESHOLD = 0.8  # rad — ~46 deg
FALL_ROLL_THRESHOLD = 0.8


class DeployComposite:
    """Deploy composite walking + upper body policy to real robot."""

    def __init__(
        self,
        walking_checkpoint: str = "checkpoints/mujoco_locomotion_v13_flat_feet",
        upper_checkpoint: str = "checkpoints/mujoco_wave/final_params",
        task: str = "wave",
        hz: float = 20.0,
        duration: float = 30.0,
        ramp_seconds: float = 3.0,
        dry_run: bool = False,
        vx: float = 0.3,
        vy: float = 0.0,
        vyaw: float = 0.0,
        max_joint_delta: float = MAX_JOINT_DELTA,
    ):
        self.task = task
        self.hz = hz
        self.dt = 1.0 / hz
        self.duration = duration
        self.ramp_seconds = ramp_seconds
        self.dry_run = dry_run
        self.max_joint_delta = max_joint_delta

        # Load composite skill
        task_obs_dim = TASK_OBS_DIM if task == "wave" else 0
        print(f"Loading composite policy: walk={walking_checkpoint} upper={upper_checkpoint}")
        self.skill = CompositeSkill(
            walking_checkpoint=walking_checkpoint,
            upper_checkpoint=upper_checkpoint,
            task_obs_dim=task_obs_dim,
        )
        self.skill.set_command(vx=vx, vy=vy, vyaw=vyaw)
        print(f"Policy loaded. Task: {task}, Command: vx={vx}, vy={vy}, vyaw={vyaw}")

        # State
        self._last_targets = np.zeros(NUM_TOTAL_JOINTS, dtype=np.float32)
        self._imu_roll = 0.0
        self._imu_pitch = 0.0
        self._gyro = np.zeros(3, dtype=np.float32)
        self._joint_feedback: np.ndarray | None = None
        self._step = 0
        self._stopped = False
        self._start_time = 0.0

    def joint_targets_to_servo_commands(
        self, targets: np.ndarray,
    ) -> list[dict[str, int]]:
        """Convert 24-dim joint targets (radians) to servo commands."""
        commands = []
        for i, name in enumerate(ALL_JOINT_NAMES):
            rad = float(targets[i])
            servo_id = joint_name_to_servo_id(name)
            pulse = radians_to_pulse(rad, servo_id)
            commands.append({"id": servo_id, "position": pulse})
        return commands

    def compute_task_obs(self, elapsed: float) -> np.ndarray | None:
        """Compute task observation for the upper body policy."""
        if self.task != "wave":
            return None
        phase = elapsed * 2.0 * math.pi * WAVE_FREQUENCY
        return np.array([
            math.sin(phase),
            math.cos(phase),
            WAVE_SHOULDER_PITCH,
            WAVE_AMPLITUDE * math.sin(phase),
            WAVE_ELBOW_PITCH,
            WAVE_ELBOW_YAW,
        ], dtype=np.float32)

    def safety_clamp(
        self, targets: np.ndarray, ramp_factor: float,
    ) -> np.ndarray:
        """Apply safety limits: ramp blending + per-step delta clamp."""
        default = np.zeros(NUM_TOTAL_JOINTS, dtype=np.float32)
        blended = default + (targets - default) * ramp_factor
        delta = blended - self._last_targets
        clamped_delta = np.clip(delta, -self.max_joint_delta, self.max_joint_delta)
        result = self._last_targets + clamped_delta
        self._last_targets = result.copy()
        return result

    def check_fall(self) -> bool:
        if abs(self._imu_pitch) > FALL_PITCH_THRESHOLD:
            print(f"FALL DETECTED: pitch={self._imu_pitch:.3f} rad")
            return True
        if abs(self._imu_roll) > FALL_ROLL_THRESHOLD:
            print(f"FALL DETECTED: roll={self._imu_roll:.3f} rad")
            return True
        return False

    def policy_step(self, elapsed: float, ramp_factor: float) -> list[dict[str, int]]:
        """Run one policy step, return servo commands for all 24 joints."""
        self._step += 1
        task_obs = self.compute_task_obs(elapsed)

        targets = self.skill.get_full_action(
            gyro=self._gyro,
            imu_roll=self._imu_roll,
            imu_pitch=self._imu_pitch,
            joint_positions=self._joint_feedback,
            task_obs=task_obs,
        )

        safe_targets = self.safety_clamp(targets, ramp_factor)
        return self.joint_targets_to_servo_commands(safe_targets)

    async def run_with_bridge(self, bridge_url: str) -> None:
        """Run the deploy loop via WebSocket bridge."""
        import websockets

        print(f"Connecting to bridge at {bridge_url}...")
        async with websockets.connect(bridge_url) as ws:
            print("Connected to bridge.")

            bridge_hz = min(self.hz, 30.0)
            await self._send_command(ws, "policy.start", {
                "task": f"deploy_composite_{self.task}",
                "hz": bridge_hz,
            })
            resp = await self._recv_response(ws)
            if not resp.get("ok"):
                print(f"Failed to start policy: {resp.get('message')}")
                return

            print(f"Policy mode started. {self.hz}Hz for {self.duration}s "
                  f"(ramp: {self.ramp_seconds}s)")
            if self.dry_run:
                print("DRY RUN — commands logged but not sent")
            print()

            try:
                await self._control_loop(ws)
            except KeyboardInterrupt:
                print("\nInterrupted by user.")
            finally:
                print("Stopping policy mode...")
                await self._send_standing_pose(ws)
                await asyncio.sleep(0.5)
                await self._send_command(ws, "policy.stop", {})
                try:
                    await self._recv_response(ws)
                except Exception:
                    pass
                print("Policy stopped.")

    async def _control_loop(self, ws) -> None:
        self._start_time = time.monotonic()
        last_status_time = self._start_time

        while not self._stopped:
            t0 = time.monotonic()
            elapsed = t0 - self._start_time

            if elapsed >= self.duration:
                print(f"\nDuration limit reached ({self.duration}s).")
                break

            ramp = min(1.0, elapsed / self.ramp_seconds) if self.ramp_seconds > 0 else 1.0

            if self.check_fall():
                print("Emergency stop: fall detected!")
                break

            servo_cmds = self.policy_step(elapsed, ramp)

            if not self.dry_run:
                action_payload = {
                    "joint_positions": {
                        ALL_JOINT_NAMES[i]: float(self._last_targets[i])
                        for i in range(NUM_TOTAL_JOINTS)
                    },
                    "duration": int(self.dt * 1000),
                }
                await self._send_command(ws, "policy.tick", action_payload)
                await self._recv_response(ws)
                await self._process_events(ws)

            if t0 - last_status_time >= 2.0:
                self._print_status(elapsed, ramp)
                last_status_time = t0

            step_time = time.monotonic() - t0
            sleep_time = self.dt - step_time
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    async def _send_standing_pose(self, ws) -> None:
        action_payload = {
            "joint_positions": {
                name: 0.0 for name in ALL_JOINT_NAMES
            },
            "duration": 500,
        }
        await self._send_command(ws, "policy.tick", action_payload)
        try:
            await self._recv_response(ws)
        except Exception:
            pass

    async def _send_command(self, ws, command: str, payload: dict) -> None:
        msg = {
            "type": "command",
            "request_id": f"composite-{self._step}",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "command": command,
            "payload": payload,
        }
        await ws.send(json.dumps(msg))

    async def _recv_response(self, ws) -> dict:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            try:
                remaining = max(0.01, deadline - time.monotonic())
                raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
                data = json.loads(raw)
                if "event" in data:
                    if "data" in data:
                        self._update_telemetry(data["data"])
                    continue
                if "data" in data:
                    self._update_telemetry(data["data"])
                return data
            except asyncio.TimeoutError:
                break
        return {"ok": False, "message": "timeout"}

    async def _process_events(self, ws) -> None:
        try:
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=0.001)
                data = json.loads(raw)
                if data.get("event") == "telemetry.basic":
                    self._update_telemetry(data.get("data", {}))
        except asyncio.TimeoutError:
            pass  # expected — no more pending events
        except Exception as exc:
            print(f"  WARNING: event processing error: {exc}")

    def _update_telemetry(self, data: dict) -> None:
        if "imu_roll" in data:
            self._imu_roll = float(data["imu_roll"])
        if "imu_pitch" in data:
            self._imu_pitch = float(data["imu_pitch"])
        if "gyro" in data and isinstance(data["gyro"], list):
            self._gyro = np.array(data["gyro"][:3], dtype=np.float32)
        if "joint_positions" in data and isinstance(data["joint_positions"], dict):
            jp = data["joint_positions"]
            feedback = np.zeros(NUM_TOTAL_JOINTS, dtype=np.float32)
            for i, name in enumerate(ALL_JOINT_NAMES):
                if name in jp:
                    feedback[i] = float(jp[name])
            self._joint_feedback = feedback

    def _print_status(self, elapsed: float, ramp: float) -> None:
        dry = " [DRY]" if self.dry_run else ""
        fb = "YES" if self._joint_feedback is not None else "NO"
        # Show a few representative upper body joints
        r_sho = self._last_targets[12] if NUM_TOTAL_JOINTS > 12 else 0.0  # head_pan index
        print(
            f"  [{self.task.upper()}{dry}] t={elapsed:.1f}s ramp={ramp:.2f} "
            f"step={self._step} imu=({self._imu_roll:.3f},{self._imu_pitch:.3f}) "
            f"fb={fb} head_pan={self._last_targets[12]:.3f}"
        )

    def run_dry(self) -> None:
        """Run in dry mode — no bridge, just log commands."""
        print(f"DRY RUN: {self.task} for {self.duration}s at {self.hz}Hz")
        print(f"Ramp: {self.ramp_seconds}s")
        print()

        start = time.monotonic()
        while time.monotonic() - start < self.duration:
            elapsed = time.monotonic() - start
            ramp = min(1.0, elapsed / self.ramp_seconds) if self.ramp_seconds > 0 else 1.0

            servo_cmds = self.policy_step(elapsed, ramp)

            if self._step % int(self.hz * 2) == 1:
                self._print_status(elapsed, ramp)
                if self._step % int(self.hz * 10) == 1:
                    print("    Joint targets (radians):")
                    for i, name in enumerate(ALL_JOINT_NAMES):
                        print(f"      {name:15s}: {self._last_targets[i]:+.4f} rad "
                              f"→ pulse {servo_cmds[i]['position']}")

            time.sleep(self.dt)

        print(f"\nDry run complete. {self._step} steps.")


def main():
    parser = argparse.ArgumentParser(
        description="Deploy composite walking + upper body policy to AiNex robot"
    )
    parser.add_argument(
        "--walking-checkpoint", type=str,
        default="checkpoints/mujoco_locomotion_v13_flat_feet",
        help="Walking policy checkpoint",
    )
    parser.add_argument(
        "--upper-checkpoint", type=str,
        default="checkpoints/mujoco_wave/final_params",
        help="Upper body policy checkpoint",
    )
    parser.add_argument(
        "--task", type=str, default="wave",
        choices=["wave"],
        help="Upper body task (default: wave)",
    )
    parser.add_argument(
        "--bridge", type=str, default="ws://localhost:9100",
        help="Bridge WebSocket URL",
    )
    parser.add_argument("--hz", type=float, default=20.0)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--ramp-seconds", type=float, default=3.0)
    parser.add_argument("--vx", type=float, default=0.3)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--vyaw", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-delta", type=float, default=MAX_JOINT_DELTA)

    args = parser.parse_args()

    if args.hz > 50:
        print("WARNING: Hz > 50 exceeds training frequency. Clamping to 50.")
        args.hz = 50.0

    deployer = DeployComposite(
        walking_checkpoint=args.walking_checkpoint,
        upper_checkpoint=args.upper_checkpoint,
        task=args.task,
        hz=args.hz,
        duration=args.duration,
        ramp_seconds=args.ramp_seconds,
        dry_run=args.dry_run,
        vx=args.vx,
        vy=args.vy,
        vyaw=args.vyaw,
        max_joint_delta=args.max_delta,
    )

    def signal_handler(sig, frame):
        print("\nCtrl+C received, stopping...")
        deployer._stopped = True
    signal.signal(signal.SIGINT, signal_handler)

    if args.dry_run:
        deployer.run_dry()
    else:
        print(f"\n{'='*60}")
        print(f"DEPLOYING COMPOSITE POLICY TO REAL ROBOT")
        print(f"{'='*60}")
        print(f"  Walking:    {args.walking_checkpoint}")
        print(f"  Upper body: {args.upper_checkpoint}")
        print(f"  Task:       {args.task}")
        print(f"  Bridge:     {args.bridge}")
        print(f"  Frequency:  {args.hz} Hz")
        print(f"  Duration:   {args.duration}s")
        print(f"  Command:    vx={args.vx} vy={args.vy} vyaw={args.vyaw}")
        print(f"{'='*60}")
        print()
        asyncio.run(deployer.run_with_bridge(args.bridge))


if __name__ == "__main__":
    main()
