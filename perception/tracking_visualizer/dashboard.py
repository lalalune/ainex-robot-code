"""Flask web dashboard for real-time tracking visualization.

Serves a browser-based dashboard with dual camera MJPEG streams
(robot IP camera + USB camera), detection overlays, a bird's-eye
scene view, and calibration / config controls.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from pathlib import Path
from typing import Generator

import numpy as np

try:
    import cv2

    _HAS_CV2 = True
except ImportError:
    cv2 = None  # type: ignore[assignment]
    _HAS_CV2 = False

try:
    from flask import Flask, Response, jsonify, request

    _HAS_FLASK = True
except ImportError:
    _HAS_FLASK = False

from perception.calibration import CameraIntrinsics
from perception.config import PipelineConfig, load_config
from perception.detectors.aruco_detector import ArucoDetector
from perception.frame_source import OpenCVSource

from perception.tracking_visualizer.calibrator import RuntimeCalibrator
from perception.tracking_visualizer.overlay import draw_all_overlays
from perception.tracking_visualizer.scene_view import SceneRenderer
from perception.tracking_visualizer.websocket_camera import IPCameraSource

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedded HTML dashboard
# ---------------------------------------------------------------------------

DASHBOARD_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Robot Tracking Visualizer</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0f0f1a;color:#e0e0e0;font-family:'Segoe UI',system-ui,sans-serif}
.header{background:#161628;padding:10px 20px;display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #2a2a4a}
.header h1{font-size:16px;color:#7c8aff;font-weight:600}
.header .sb{font-size:12px;color:#888}
.grid{display:grid;grid-template-columns:1fr 1fr;grid-template-rows:1fr 1fr;gap:6px;padding:6px;height:calc(100vh - 44px)}
.panel{background:#161628;border-radius:6px;overflow:hidden;position:relative;display:flex;flex-direction:column}
.ph{padding:6px 12px;background:#1c1c36;font-size:12px;font-weight:600;color:#aaa;display:flex;justify-content:space-between}
.badge{padding:1px 8px;border-radius:10px;font-size:10px}
.badge.on{background:#2a4a2a;color:#4a4}.badge.off{background:#4a2a2a;color:#a44}
.panel img{flex:1;width:100%;object-fit:contain;background:#0a0a14;min-height:0}
.cp{overflow-y:auto;padding:12px}
h3{color:#7c8aff;font-size:13px;margin:14px 0 6px;padding-bottom:4px;border-bottom:1px solid #2a2a4a}
h3:first-child{margin-top:0}
.btn{background:#252548;border:1px solid #3a3a5a;color:#ccc;padding:6px 14px;border-radius:4px;cursor:pointer;font-size:12px;margin:2px;transition:all .15s}
.btn:hover{background:#353568;border-color:#5a5a8a;color:#fff}
.btn.p{background:#2a3a7a;border-color:#4a5aaa}
.btn.p:hover{background:#3a4a9a}
label{display:flex;align-items:center;gap:6px;margin:4px 0;font-size:12px;cursor:pointer}
label input[type=checkbox]{accent-color:#7c8aff}
input[type=number]{background:#1a1a30;border:1px solid #3a3a5a;color:#ddd;padding:4px 6px;border-radius:3px;width:70px;font-size:12px}
.og{display:grid;grid-template-columns:auto 1fr auto;gap:4px 8px;align-items:center;font-size:12px}
.msg{padding:6px 10px;margin:6px 0;border-radius:4px;font-size:11px}
.msg.i{background:#1a2a4a;color:#8ac}.msg.ok{background:#1a3a2a;color:#8c8}.msg.er{background:#3a1a1a;color:#c88}
#sd{font-size:11px;color:#888;line-height:1.6}
</style>
</head>
<body>
<div class="header">
  <h1>Robot Tracking Visualizer</h1>
  <div class="sb" id="fp">--</div>
</div>
<div class="grid">
  <div class="panel">
    <div class="ph">Robot Camera (IP)<span class="badge off" id="br">--</span></div>
    <img src="/video/robot" alt="Robot Camera">
  </div>
  <div class="panel">
    <div class="ph">USB Camera (External)<span class="badge off" id="bu">--</span></div>
    <img src="/video/usb" alt="USB Camera">
  </div>
  <div class="panel">
    <div class="ph">Scene View (Bird's Eye)<span class="badge on" id="bs">LIVE</span></div>
    <img src="/video/scene" alt="Scene View">
  </div>
  <div class="panel cp">
    <h3>System Status</h3>
    <div id="sd">Connecting...</div>

    <h3>Detection Overlays</h3>
    <label><input type="checkbox" id="show-aruco" checked onchange="tog(this)"> ArUco Markers</label>
    <label><input type="checkbox" id="show-faces" checked onchange="tog(this)"> Face Detection</label>
    <label><input type="checkbox" id="show-skeletons" checked onchange="tog(this)"> Skeleton Pose</label>
    <label><input type="checkbox" id="show-objects" checked onchange="tog(this)"> Object Detection</label>

    <h3>Robot Marker Offset</h3>
    <p style="font-size:11px;color:#888;margin-bottom:6px">
      Offset from body ArUco to robot center (metres).
      Marker is on back-right shoulder.
    </p>
    <div class="og">
      <span>X (fwd):</span><input type="number" id="ox" step="0.01" value="0.05"><span>m</span>
      <span>Y (left):</span><input type="number" id="oy" step="0.01" value="0.04"><span>m</span>
      <span>Z (up):</span><input type="number" id="oz" step="0.01" value="0.0"><span>m</span>
    </div>
    <button class="btn p" onclick="saveOff()" style="margin-top:6px">Apply Offset</button>
    <div id="om"></div>

    <h3>Floor Calibration</h3>
    <div>
      <button class="btn p" onclick="cal('start')">Start</button>
      <button class="btn" onclick="cal('capture')">Capture Frame</button>
      <button class="btn p" onclick="cal('finish')">Finish &amp; Save</button>
      <button class="btn" onclick="cal('auto')">Auto-Calibrate</button>
    </div>
    <div id="cm" class="msg i" style="display:none"></div>

    <h3>Floor Markers</h3>
    <div id="fm" style="font-size:11px;color:#999"></div>
  </div>
</div>
<script>
async function tog(el){
  const k=el.id.replace('show-','');
  await fetch('/api/config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({[k]:el.checked})});
}
async function saveOff(){
  const x=parseFloat(document.getElementById('ox').value)||0;
  const y=parseFloat(document.getElementById('oy').value)||0;
  const z=parseFloat(document.getElementById('oz').value)||0;
  const r=await fetch('/api/robot_offset',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({x,y,z})});
  const d=await r.json();const el=document.getElementById('om');
  el.className='msg ok';el.style.display='block';el.textContent=d.message||'Applied';
  setTimeout(()=>el.style.display='none',3000);
}
async function cal(a){
  const r=await fetch('/api/calibrate/'+a,{method:'POST'});
  const d=await r.json();const el=document.getElementById('cm');
  el.className='msg '+(d.ok?'ok':'i');el.style.display='block';el.textContent=d.message;
}
let _first=true;
setInterval(async()=>{
  try{
    const r=await fetch('/api/status');const d=await r.json();
    document.getElementById('fp').textContent=
      'FPS: '+d.fps.toFixed(1)+' | Entities: '+d.entity_count+' | Markers: '+d.marker_count;
    const rb=document.getElementById('br');
    rb.textContent=d.robot_camera?'LIVE':'OFF';rb.className='badge '+(d.robot_camera?'on':'off');
    const ub=document.getElementById('bu');
    ub.textContent=d.usb_camera?'LIVE':'OFF';ub.className='badge '+(d.usb_camera?'on':'off');
    let s='';
    if(d.robot_position)s+='Robot: ('+d.robot_position.map(v=>v.toFixed(2)).join(', ')+')<br>';
    s+='Detected: '+d.marker_count+' markers, '+d.entity_count+' entities<br>';
    if(d.calibration)s+='Calibration: '+d.calibration+'<br>';
    s+='Uptime: '+d.uptime+'s';
    document.getElementById('sd').innerHTML=s;
    if(d.robot_offset&&_first){
      document.getElementById('ox').value=d.robot_offset.x.toFixed(3);
      document.getElementById('oy').value=d.robot_offset.y.toFixed(3);
      document.getElementById('oz').value=d.robot_offset.z.toFixed(3);
      _first=false;
    }
    if(d.floor_markers){
      document.getElementById('fm').innerHTML=Object.entries(d.floor_markers)
        .map(([id,pos])=>'ID '+id+': ('+pos.map(v=>v.toFixed(2)).join(', ')+')')
        .join('<br>');
    }
  }catch(e){}
},1500);
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _no_signal_frame(w: int, h: int, text: str = "NO SIGNAL") -> np.ndarray:
    """Render a dark placeholder frame."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[::2, :] = 20
    if _HAS_CV2:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        x = (w - tw) // 2
        y = (h + th) // 2
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 60, 80), 2)
    return frame


# ---------------------------------------------------------------------------
# Per-camera processor
# ---------------------------------------------------------------------------

class CameraProcessor:
    """Reads frames, runs detectors, draws overlays."""

    def __init__(
        self,
        name: str,
        intrinsics: CameraIntrinsics,
        marker_size_m: float = 0.0508,
    ) -> None:
        self.name = name
        self.intrinsics = intrinsics
        self._aruco = ArucoDetector(
            intrinsics=intrinsics, marker_size_m=marker_size_m,
        )
        self._face_det = None
        self._skel_est = None
        self._obj_det = None
        self._loaded = False

        # Latest results (guarded by lock)
        self.latest_frame: np.ndarray | None = None
        self.latest_annotated: np.ndarray | None = None
        self.latest_aruco: list = []
        self.latest_faces: list = []
        self.latest_skeletons: list = []
        self.latest_objects: list = []
        self.lock = threading.Lock()
        self.fps = 0.0
        self._fps_t0 = time.monotonic()
        self._fps_n = 0

    def load_detectors(self) -> None:
        """Load heavy ML models (call from background thread)."""
        if self._loaded:
            return
        self._loaded = True

        try:
            from perception.detectors.face_detector import FaceDetector
            fd = FaceDetector(confidence_threshold=0.5)
            if fd.is_available:
                self._face_det = fd
                logger.info("[%s] Face detector ready", self.name)
        except Exception as e:
            logger.info("[%s] Face detector unavailable: %s", self.name, e)

        try:
            from perception.detectors.skeleton_estimator import SkeletonEstimator
            se = SkeletonEstimator(confidence_threshold=0.3)
            if se.is_available:
                self._skel_est = se
                logger.info("[%s] Skeleton estimator ready", self.name)
        except Exception as e:
            logger.info("[%s] Skeleton estimator unavailable: %s", self.name, e)

        try:
            from perception.detectors.object_detector import ObjectDetector
            od = ObjectDetector(confidence_threshold=0.5)
            if od.is_available:
                self._obj_det = od
                logger.info("[%s] Object detector ready", self.name)
        except Exception as e:
            logger.info("[%s] Object detector unavailable: %s", self.name, e)

    def process(self, frame: np.ndarray, show: dict) -> np.ndarray:
        """Run detectors + overlay.  Returns annotated frame."""
        # FPS bookkeeping
        self._fps_n += 1
        now = time.monotonic()
        if now - self._fps_t0 >= 1.0:
            self.fps = self._fps_n / (now - self._fps_t0)
            self._fps_n = 0
            self._fps_t0 = now

        with self.lock:
            self.latest_frame = frame.copy()

        aruco = self._aruco.detect(frame)
        faces = self._face_det.detect(frame) if self._face_det else []
        skeletons = self._skel_est.estimate(frame) if self._skel_est else []
        objects = self._obj_det.detect(frame) if self._obj_det else []

        with self.lock:
            self.latest_aruco = aruco
            self.latest_faces = faces
            self.latest_skeletons = skeletons
            self.latest_objects = objects

        annotated = draw_all_overlays(
            frame,
            aruco=aruco,
            faces=faces,
            skeletons=skeletons,
            objects=objects,
            intrinsics=self.intrinsics,
            show_aruco=show.get("aruco", True),
            show_faces=show.get("faces", True),
            show_skeletons=show.get("skeletons", True),
            show_objects=show.get("objects", True),
        )

        if _HAS_CV2:
            cv2.putText(
                annotated, f"{self.fps:.1f} FPS", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA,
            )

        with self.lock:
            self.latest_annotated = annotated

        return annotated


# ---------------------------------------------------------------------------
# Main dashboard
# ---------------------------------------------------------------------------

class TrackingDashboard:
    """Ties cameras, detectors, scene view, calibration into a web UI."""

    def __init__(
        self,
        robot_camera_url: str = "",
        usb_camera_device: int = 0,
        config_path: str | None = None,
        host: str = "0.0.0.0",
        port: int = 5555,
    ) -> None:
        cfg_path = Path(config_path) if config_path else None
        self._cfg = load_config(cfg_path)

        self._host = host
        self._port = port
        self._t0 = time.monotonic()

        # Intrinsics
        self._usb_intr = CameraIntrinsics(
            fx=self._cfg.external_camera.fx,
            fy=self._cfg.external_camera.fy,
            cx=self._cfg.external_camera.cx,
            cy=self._cfg.external_camera.cy,
            dist_coeffs=self._cfg.external_camera.dist_coeffs,
            width=self._cfg.external_camera.width,
            height=self._cfg.external_camera.height,
        )
        self._robot_intr = CameraIntrinsics(
            fx=self._cfg.camera.fx,
            fy=self._cfg.camera.fy,
            cx=self._cfg.camera.cx,
            cy=self._cfg.camera.cy,
            dist_coeffs=self._cfg.camera.dist_coeffs,
            width=self._cfg.camera.width,
            height=self._cfg.camera.height,
        )

        # Sources
        self._robot_url = robot_camera_url
        self._usb_dev = usb_camera_device
        self._robot_src: IPCameraSource | None = None
        self._usb_src: OpenCVSource | None = None

        # Processors
        self._robot_proc = CameraProcessor(
            "robot", self._robot_intr, self._cfg.markers.marker_size_m,
        )
        self._usb_proc = CameraProcessor(
            "usb", self._usb_intr, self._cfg.markers.marker_size_m,
        )

        # Scene
        self._scene = SceneRenderer(canvas_size=800, world_range=3.0)
        self._scene.update_floor_markers(self._cfg.markers.world_markers)

        # Calibrator
        self._cal = RuntimeCalibrator(
            world_markers=self._cfg.markers.world_markers,
            marker_size_m=self._cfg.markers.marker_size_m,
        )

        # UI toggles
        self._show: dict[str, bool] = {
            "aruco": True,
            "faces": True,
            "skeletons": True,
            "objects": True,
        }

        # No-signal placeholders
        self._ns_robot = _no_signal_frame(
            self._cfg.camera.width, self._cfg.camera.height,
            "ROBOT CAMERA - NO SIGNAL",
        )
        self._ns_usb = _no_signal_frame(
            self._cfg.external_camera.width,
            self._cfg.external_camera.height,
            "USB CAMERA - NO SIGNAL",
        )

        self._scene_frame: np.ndarray = self._scene.render()
        self._scene_lock = threading.Lock()
        self._running = False

    # -- lifecycle --

    def start(self) -> None:
        self._running = True

        # Robot camera
        if self._robot_url:
            logger.info("Opening robot camera: %s", self._robot_url)
            self._robot_src = IPCameraSource(self._robot_url)

        # USB camera
        try:
            logger.info("Opening USB camera: device %d", self._usb_dev)
            self._usb_src = OpenCVSource(
                device=self._usb_dev,
                width=self._cfg.external_camera.width,
                height=self._cfg.external_camera.height,
            )
            if not self._usb_src.is_open:
                logger.warning("USB camera not available")
                self._usb_src = None
        except Exception as e:
            logger.warning("USB camera open failed: %s", e)
            self._usb_src = None

        # Background threads
        threading.Thread(target=self._load_detectors, daemon=True).start()
        threading.Thread(target=self._robot_loop, daemon=True).start()
        threading.Thread(target=self._usb_loop, daemon=True).start()
        threading.Thread(target=self._scene_loop, daemon=True).start()

        self._run_flask()

    def stop(self) -> None:
        self._running = False
        if self._robot_src:
            self._robot_src.release()
        if self._usb_src:
            self._usb_src.release()

    # -- background threads --

    def _load_detectors(self) -> None:
        logger.info("Loading ML detectors (may take a moment)...")
        self._robot_proc.load_detectors()
        self._usb_proc.load_detectors()
        logger.info("Detectors ready")

    def _robot_loop(self) -> None:
        while self._running:
            if self._robot_src is not None:
                ok, frame = self._robot_src.read()
                if ok:
                    self._robot_proc.process(frame, self._show)
                    continue
            time.sleep(0.05)

    def _usb_loop(self) -> None:
        while self._running:
            if self._usb_src is not None and self._usb_src.is_open:
                ok, frame = self._usb_src.read()
                if ok:
                    self._usb_proc.process(frame, self._show)
                    self._update_scene_from_usb()
                    continue
            time.sleep(0.05)

    def _scene_loop(self) -> None:
        while self._running:
            rendered = self._scene.render()
            with self._scene_lock:
                self._scene_frame = rendered
            time.sleep(0.05)

    # -- scene update from USB camera ArUco --

    def _update_scene_from_usb(self) -> None:
        with self._usb_proc.lock:
            aruco = list(self._usb_proc.latest_aruco)
        if not aruco:
            return

        # Auto-calibrate extrinsics if missing
        ext = self._cal.state.extrinsics
        if ext is None:
            ext = self._cal.quick_calibrate(aruco, self._usb_intr, "external")
        if ext is None:
            return

        robot_ids = set(self._cfg.markers.robot_marker_ids)
        head_id = self._cfg.markers.robot_head_marker_id
        robot_pos = None
        robot_heading = 0.0
        head_pos = None

        for det in aruco:
            if det.marker_id in robot_ids:
                world_pos = ext.transform_point(det.tvec)
                R_m, _ = cv2.Rodrigues(det.rvec)
                R_w = ext.R @ R_m
                robot_pos = self._cal.robot_offset.apply(world_pos, R_w)
                mz = R_w[:, 2]
                robot_heading = math.atan2(float(mz[1]), float(mz[0]))
            elif det.marker_id == head_id:
                head_pos = ext.transform_point(det.tvec)

        self._scene.update_robot_pose(robot_pos, robot_heading, head_pos)

        # Pass object detections as entities
        entities: list[dict] = []
        with self._usb_proc.lock:
            for obj in self._usb_proc.latest_objects:
                entities.append({
                    "label": obj.class_name,
                    "position": [0, 0, 0],
                    "type": obj.class_name,
                    "confidence": obj.confidence,
                    "velocity": [0, 0, 0],
                })
        self._scene.update_entities(entities)

    # -- MJPEG generator --

    def _mjpeg_stream(
        self,
        get_frame,
        fallback: np.ndarray,
    ) -> Generator[bytes, None, None]:
        while self._running:
            frame = get_frame()
            if frame is None:
                frame = fallback
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            )
            time.sleep(0.033)

    # -- Flask app --

    def _run_flask(self) -> None:
        if not _HAS_FLASK:
            logger.error("Flask not installed.  pip install flask")
            return

        app = Flask(__name__)
        app.logger.setLevel(logging.WARNING)
        dashboard = self  # closure reference

        @app.route("/")
        def index():
            return DASHBOARD_HTML

        @app.route("/video/robot")
        def video_robot():
            def _get():
                with dashboard._robot_proc.lock:
                    return dashboard._robot_proc.latest_annotated
            return Response(
                dashboard._mjpeg_stream(_get, dashboard._ns_robot),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

        @app.route("/video/usb")
        def video_usb():
            def _get():
                with dashboard._usb_proc.lock:
                    return dashboard._usb_proc.latest_annotated
            return Response(
                dashboard._mjpeg_stream(_get, dashboard._ns_usb),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

        @app.route("/video/scene")
        def video_scene():
            def _get():
                with dashboard._scene_lock:
                    return dashboard._scene_frame
            return Response(
                dashboard._mjpeg_stream(_get, dashboard._scene.render()),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

        @app.route("/api/status")
        def api_status():
            rp = None
            if dashboard._scene._robot_position is not None:
                rp = dashboard._scene._robot_position.tolist()
            return jsonify({
                "fps": max(dashboard._robot_proc.fps, dashboard._usb_proc.fps),
                "robot_camera": (
                    dashboard._robot_src is not None
                    and dashboard._robot_src.is_connected
                ),
                "usb_camera": (
                    dashboard._usb_src is not None
                    and dashboard._usb_src.is_open
                ),
                "entity_count": len(dashboard._scene._entities),
                "marker_count": (
                    len(dashboard._usb_proc.latest_aruco)
                    + len(dashboard._robot_proc.latest_aruco)
                ),
                "robot_position": rp,
                "calibration": dashboard._cal.state.message,
                "uptime": int(time.monotonic() - dashboard._t0),
                "robot_offset": {
                    "x": dashboard._cal.robot_offset.x,
                    "y": dashboard._cal.robot_offset.y,
                    "z": dashboard._cal.robot_offset.z,
                },
                "floor_markers": {
                    str(k): v
                    for k, v in dashboard._cfg.markers.world_markers.items()
                },
                "show": dashboard._show,
            })

        @app.route("/api/config", methods=["POST"])
        def api_config():
            data = request.get_json() or {}
            for k in ("aruco", "faces", "skeletons", "objects"):
                if k in data:
                    dashboard._show[k] = bool(data[k])
            return jsonify({"ok": True, "show": dashboard._show})

        @app.route("/api/robot_offset", methods=["POST"])
        def api_robot_offset():
            data = request.get_json() or {}
            dashboard._cal.set_robot_offset(
                x=float(data.get("x", 0)),
                y=float(data.get("y", 0)),
                z=float(data.get("z", 0)),
            )
            o = dashboard._cal.robot_offset
            return jsonify({
                "ok": True,
                "message": f"Offset: ({o.x:.3f}, {o.y:.3f}, {o.z:.3f})",
            })

        @app.route("/api/calibrate/<action>", methods=["POST"])
        def api_calibrate(action: str):
            if action == "start":
                msg = dashboard._cal.start_floor_calibration()
                return jsonify({"ok": True, "message": msg})
            if action == "capture":
                with dashboard._usb_proc.lock:
                    f = dashboard._usb_proc.latest_frame
                if f is None:
                    return jsonify({"ok": False, "message": "No frame available"})
                msg = dashboard._cal.capture_frame(f)
                return jsonify({"ok": True, "message": msg})
            if action == "finish":
                ad = ArucoDetector(
                    intrinsics=dashboard._usb_intr,
                    marker_size_m=dashboard._cfg.markers.marker_size_m,
                )
                ext, msg = dashboard._cal.finish_floor_calibration(
                    dashboard._usb_intr, ad, "external",
                )
                return jsonify({"ok": ext is not None, "message": msg})
            if action == "auto":
                with dashboard._usb_proc.lock:
                    a = list(dashboard._usb_proc.latest_aruco)
                if not a:
                    return jsonify({"ok": False, "message": "No markers visible"})
                ext = dashboard._cal.quick_calibrate(
                    a, dashboard._usb_intr, "external",
                )
                if ext is None:
                    return jsonify({
                        "ok": False,
                        "message": "Need floor markers visible",
                    })
                return jsonify({
                    "ok": True,
                    "message": f"Auto-calibrated! Error: {ext.reprojection_error:.3f}px",
                })
            return jsonify({"ok": False, "message": f"Unknown: {action}"})

        # Silence werkzeug per-request logs
        logging.getLogger("werkzeug").setLevel(logging.WARNING)

        print(f"\n  Tracking Visualizer Dashboard: http://localhost:{self._port}\n")
        app.run(
            host=self._host,
            port=self._port,
            threaded=True,
            use_reloader=False,
        )
