#!/usr/bin/env python3
"""Launch the Robot Tracking Visualizer dashboard.

Usage
-----
    python run_visualizer.py [options]

    --robot-url URL      Robot IP camera URL (ws:// or http://)
    --usb-device N       USB camera device number (default: 0)
    --config PATH        Config YAML (default: perception/configs/tracking_visualizer.yaml)
    --port N             Dashboard port (default: 5555)
    --host HOST          Dashboard bind address (default: 0.0.0.0)
    -v, --verbose        Debug logging
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure the perception package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from perception.tracking_visualizer.dashboard import TrackingDashboard


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Robot Tracking Visualizer — real-time multi-camera dashboard"
    )
    parser.add_argument(
        "--robot-url", default="",
        help="Robot IP camera URL (ws:// or http://)",
    )
    parser.add_argument(
        "--usb-device", type=int, default=0,
        help="USB camera device number",
    )
    parser.add_argument(
        "--config",
        default="perception/configs/tracking_visualizer.yaml",
        help="Pipeline config YAML path",
    )
    parser.add_argument("--port", type=int, default=5555, help="Web port")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Resolve config path (try relative to script dir if not found)
    cfg = args.config
    if not Path(cfg).exists():
        alt = Path(__file__).parent / cfg
        if alt.exists():
            cfg = str(alt)

    dashboard = TrackingDashboard(
        robot_camera_url=args.robot_url,
        usb_camera_device=args.usb_device,
        config_path=cfg,
        host=args.host,
        port=args.port,
    )

    try:
        dashboard.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
        dashboard.stop()


if __name__ == "__main__":
    main()
