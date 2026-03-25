#!/usr/bin/env python3
"""HTTP server for ainex-remote control page with camera proxy, PTY WebSocket, and transcription."""
import hashlib
import base64
import http.server
import json
import os
import shutil
import socket
import ssl
import struct
import subprocess
import tempfile
import threading
import urllib.parse

import ptyprocess

PORT = 8888
HTTPS_PORT = 8443
VIDEO_HOST = "127.0.0.1"
VIDEO_PORT = 8080
ROSBRIDGE_HOST = "127.0.0.1"
ROSBRIDGE_PORT = 9090
DIRECTORY = os.path.dirname(os.path.abspath(__file__))
CERT_FILE = os.path.join(DIRECTORY, "cert.pem")
KEY_FILE = os.path.join(DIRECTORY, "key.pem")

# Claude CLI — find via NVM or PATH
CLAUDE_CMD = shutil.which("claude")
if not CLAUDE_CMD:
    import glob as _glob
    _nvm = _glob.glob(os.path.expanduser("~/.nvm/versions/node/*/bin/claude"))
    if _nvm:
        CLAUDE_CMD = sorted(_nvm)[-1]

# whisper.cpp auto-detection
WHISPER_CMD = None
WHISPER_MODEL = None
for _cmd in [shutil.which("whisper-cli"),
             os.path.expanduser("~/whisper.cpp/build/bin/whisper-cli"),
             os.path.expanduser("~/whisper.cpp/main")]:
    if _cmd and os.path.isfile(_cmd):
        WHISPER_CMD = _cmd
        break
for _model in [os.path.expanduser("~/whisper.cpp/models/ggml-tiny.en.bin"),
               os.path.expanduser("~/whisper.cpp/models/ggml-base.en.bin")]:
    if os.path.isfile(_model):
        WHISPER_MODEL = _model
        break

WS_MAGIC_GUID = "258EAFA5-E914-47DA-95CA-5AB4C6BC7430"


# =============================================================================
# WebSocket frame helpers (RFC 6455)
# =============================================================================
def ws_accept_key(key):
    """Compute Sec-WebSocket-Accept from client key."""
    digest = hashlib.sha1((key + WS_MAGIC_GUID).encode()).digest()
    return base64.b64encode(digest).decode()


def ws_read_frame(sock):
    """Read one WebSocket frame from a (possibly TLS-wrapped) socket.
    Returns (opcode, payload_bytes) or (None, None) on close/error."""
    def recv_exact(n):
        buf = b""
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("socket closed")
            buf += chunk
        return buf

    try:
        hdr = recv_exact(2)
    except Exception:
        return None, None

    opcode = hdr[0] & 0x0F
    masked = bool(hdr[1] & 0x80)
    length = hdr[1] & 0x7F

    if length == 126:
        length = struct.unpack("!H", recv_exact(2))[0]
    elif length == 127:
        length = struct.unpack("!Q", recv_exact(8))[0]

    if masked:
        mask = recv_exact(4)

    if length > 10 * 1024 * 1024:  # 10MB max
        return None, None

    payload = recv_exact(length) if length > 0 else b""

    if masked:
        payload = bytes(payload[i] ^ mask[i % 4] for i in range(len(payload)))

    return opcode, payload


def ws_encode_text(text):
    """Encode a text frame (server→client, unmasked)."""
    data = text.encode("utf-8") if isinstance(text, str) else text
    out = bytearray()
    out.append(0x81)  # FIN + text opcode
    length = len(data)
    if length < 126:
        out.append(length)
    elif length < 65536:
        out.append(126)
        out.extend(struct.pack("!H", length))
    else:
        out.append(127)
        out.extend(struct.pack("!Q", length))
    out.extend(data)
    return bytes(out)


def ws_encode_close(code=1000):
    """Encode a close frame."""
    payload = struct.pack("!H", code)
    return bytes([0x88, len(payload)]) + payload


def ws_encode_pong(data=b""):
    """Encode a pong frame."""
    out = bytearray([0x8A, len(data)])
    out.extend(data)
    return bytes(out)


# =============================================================================
# PtySession — singleton managing a persistent Claude Code PTY
# =============================================================================
class PtySession:
    def __init__(self):
        self._lock = threading.Lock()
        self._proc = None
        self._clients = set()
        self._clients_lock = threading.Lock()
        self._replay_buf = bytearray()
        self._replay_max = 64 * 1024
        self._read_thread = None

    @property
    def alive(self):
        return self._proc is not None and self._proc.isalive()

    def spawn(self):
        """Spawn (or respawn) the Claude CLI in a PTY."""
        with self._lock:
            if self.alive:
                return
            if self._proc:
                try:
                    self._proc.terminate(force=True)
                except Exception:
                    pass

            cmd = CLAUDE_CMD or "claude"
            # Build env with NVM node on PATH
            env = os.environ.copy()
            env["TERM"] = "xterm-256color"
            env["COLUMNS"] = "120"
            env["LINES"] = "40"
            nvm_bin = os.path.dirname(cmd)
            if nvm_bin not in env.get("PATH", ""):
                env["PATH"] = nvm_bin + ":" + env.get("PATH", "")

            self._proc = ptyprocess.PtyProcess.spawn(
                [cmd], env=env, dimensions=(40, 120)
            )
            self._replay_buf = bytearray()
            self._read_thread = threading.Thread(target=self._read_loop, daemon=True)
            self._read_thread.start()
            print(f"[PTY] Spawned claude: pid={self._proc.pid}")

    def _read_loop(self):
        """Read PTY output and broadcast to all WS clients."""
        while self.alive:
            try:
                data = self._proc.read(4096)
                if not data:
                    break
            except EOFError:
                break
            except Exception:
                break

            # Append to replay buffer
            self._replay_buf.extend(data.encode("utf-8") if isinstance(data, str) else data)
            if len(self._replay_buf) > self._replay_max:
                self._replay_buf = self._replay_buf[-self._replay_max:]

            # Broadcast to clients
            msg = json.dumps({"type": "output", "data": data if isinstance(data, str) else data.decode("utf-8", errors="replace")})
            frame = ws_encode_text(msg)
            with self._clients_lock:
                dead = set()
                for client_sock in self._clients:
                    try:
                        client_sock.sendall(frame)
                    except Exception:
                        dead.add(client_sock)
                self._clients -= dead

        # Process exited — notify clients
        exit_msg = json.dumps({"type": "exit", "code": self._proc.exitstatus if self._proc else -1})
        frame = ws_encode_text(exit_msg)
        with self._clients_lock:
            for client_sock in self._clients:
                try:
                    client_sock.sendall(frame)
                except Exception:
                    pass
        print("[PTY] Claude process exited")

    def write(self, data):
        """Write input to the PTY."""
        if self.alive:
            self._proc.write(data)

    def resize(self, rows, cols):
        """Resize the PTY window."""
        if self.alive:
            self._proc.setwinsize(rows, cols)

    def add_client(self, sock):
        """Register a WS client for broadcast and send replay buffer."""
        with self._clients_lock:
            self._clients.add(sock)
        # Send replay buffer so client sees recent output
        if self._replay_buf:
            replay = json.dumps({"type": "output", "data": self._replay_buf.decode("utf-8", errors="replace")})
            try:
                sock.sendall(ws_encode_text(replay))
            except Exception:
                pass

    def remove_client(self, sock):
        with self._clients_lock:
            self._clients.discard(sock)

    def kill(self):
        """Kill the PTY process."""
        with self._lock:
            if self._proc:
                try:
                    self._proc.terminate(force=True)
                except Exception:
                    pass
                self._proc = None


# Global singleton
pty_session = PtySession()


# =============================================================================
# HTTP Handler
# =============================================================================
class RemoteHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/ws_proxy"):
            self._proxy_websocket()
        elif self.path.startswith("/ws_pty"):
            self._handle_ws_pty()
        elif self.path.startswith("/camera_proxy/"):
            self._proxy_video()
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == "/api/transcribe":
            self._handle_transcribe()
        else:
            self.send_error(404, "Not Found")

    # ---- WebSocket PTY endpoint ----
    def _handle_ws_pty(self):
        """WebSocket endpoint for PTY terminal access."""
        # Perform WebSocket handshake
        ws_key = self.headers.get("Sec-WebSocket-Key", "")
        if not ws_key:
            self.send_error(400, "Not a WebSocket request")
            return

        accept = ws_accept_key(ws_key)
        response = (
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept}\r\n"
            "\r\n"
        )
        client_sock = self.request
        client_sock.sendall(response.encode())

        # Auto-spawn Claude if not running
        if not pty_session.alive:
            pty_session.spawn()

        pty_session.add_client(client_sock)
        print(f"[WS_PTY] Client connected")

        try:
            while True:
                opcode, payload = ws_read_frame(client_sock)
                if opcode is None or opcode == 0x8:  # close
                    break
                if opcode == 0x9:  # ping
                    client_sock.sendall(ws_encode_pong(payload))
                    continue
                if opcode == 0xA:  # pong
                    continue
                if opcode != 0x1:  # only handle text frames
                    continue

                try:
                    msg = json.loads(payload.decode("utf-8"))
                except Exception:
                    continue

                msg_type = msg.get("type", "")
                if msg_type == "input":
                    pty_session.write(msg.get("data", ""))
                elif msg_type == "resize":
                    cols = msg.get("cols", 120)
                    rows = msg.get("rows", 40)
                    pty_session.resize(rows, cols)
                elif msg_type == "restart":
                    pty_session.kill()
                    pty_session.spawn()
                elif msg_type == "ping":
                    client_sock.sendall(ws_encode_text(json.dumps({"type": "pong"})))
        except Exception as e:
            print(f"[WS_PTY] Error: {e}")
        finally:
            pty_session.remove_client(client_sock)
            try:
                client_sock.sendall(ws_encode_close())
            except Exception:
                pass
            print(f"[WS_PTY] Client disconnected")

    # ---- Transcription endpoint ----
    def _handle_transcribe(self):
        """POST /api/transcribe — convert audio to text via whisper.cpp."""
        if not WHISPER_CMD or not WHISPER_MODEL:
            self._json_response(503, {"error": "whisper.cpp not installed"})
            return

        content_length = int(self.headers.get("Content-Length", 0))
        if content_length > 10 * 1024 * 1024:
            self._json_response(413, {"error": "Audio too large (max 10MB)"})
            return
        if content_length == 0:
            self._json_response(400, {"error": "No audio data"})
            return

        audio_data = self.rfile.read(content_length)

        try:
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as webm_f:
                webm_f.write(audio_data)
                webm_path = webm_f.name

            wav_path = webm_path.replace(".webm", ".wav")

            # Convert WebM → WAV (16kHz mono)
            ret = subprocess.run(
                ["ffmpeg", "-y", "-i", webm_path, "-ar", "16000", "-ac", "1", "-f", "wav", wav_path],
                capture_output=True, timeout=30
            )
            if ret.returncode != 0:
                self._json_response(500, {"error": "ffmpeg conversion failed"})
                return

            # Run whisper.cpp
            ret = subprocess.run(
                [WHISPER_CMD, "-m", WHISPER_MODEL, "-f", wav_path, "--no-timestamps", "-l", "en"],
                capture_output=True, text=True, timeout=60
            )
            if ret.returncode != 0:
                self._json_response(500, {"error": f"whisper error: {ret.stderr[:200]}"})
                return

            text = ret.stdout.strip()
            # whisper sometimes outputs with leading newline/whitespace
            text = text.strip()
            self._json_response(200, {"text": text})

        except subprocess.TimeoutExpired:
            self._json_response(504, {"error": "Transcription timed out"})
        except Exception as e:
            self._json_response(500, {"error": str(e)})
        finally:
            for p in [webm_path, wav_path]:
                try:
                    os.unlink(p)
                except Exception:
                    pass

    def _json_response(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    # ---- Existing: ROSBridge WebSocket proxy ----
    def _proxy_websocket(self):
        """Transparent WebSocket proxy: relay upgrade + frames to ROSBridge."""
        try:
            upstream = socket.create_connection(
                (ROSBRIDGE_HOST, ROSBRIDGE_PORT), timeout=5
            )
        except Exception as e:
            self.send_error(502, f"Cannot reach ROSBridge: {e}")
            return

        # CRITICAL: reset to blocking mode after connect.
        # create_connection leaves a 5s timeout on ALL subsequent recv/send,
        # which silently kills the relay during any brief traffic lull.
        upstream.settimeout(None)
        upstream.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        upstream.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

        # Rebuild the upgrade request for upstream, targeting /
        raw_headers = f"GET / HTTP/1.1\r\nHost: {ROSBRIDGE_HOST}:{ROSBRIDGE_PORT}\r\n"
        for key in self.headers:
            low = key.lower()
            if low in ("host",):
                continue
            raw_headers += f"{key}: {self.headers[key]}\r\n"
        raw_headers += "\r\n"
        upstream.sendall(raw_headers.encode())

        # Read the upstream 101 response and forward it back to client
        buf = b""
        while b"\r\n\r\n" not in buf:
            chunk = upstream.recv(4096)
            if not chunk:
                self.send_error(502, "ROSBridge closed during handshake")
                upstream.close()
                return
            buf += chunk

        header_end = buf.index(b"\r\n\r\n") + 4
        # Get the raw client socket (unwrap TLS if present)
        client_sock = self.request  # already the (possibly TLS-wrapped) socket
        client_sock.sendall(buf[:header_end])
        leftover = buf[header_end:]

        print(f"[WS_PROXY] ROSBridge relay established")

        # Bidirectional relay
        def relay(src, dst, label, initial=b""):
            try:
                if initial:
                    dst.sendall(initial)
                while True:
                    data = src.recv(16384)
                    if not data:
                        break
                    dst.sendall(data)
            except Exception as e:
                print(f"[WS_PROXY] {label} relay ended: {type(e).__name__}: {e}")
            finally:
                try:
                    dst.shutdown(socket.SHUT_WR)
                except Exception:
                    pass

        # Send any leftover data after the 101 headers to the client
        if leftover:
            client_sock.sendall(leftover)

        t_up = threading.Thread(
            target=relay, args=(upstream, client_sock, "ros->client"), daemon=True
        )
        t_down = threading.Thread(
            target=relay, args=(client_sock, upstream, "client->ros"), daemon=True
        )
        t_up.start()
        t_down.start()
        t_up.join()
        t_down.join()
        print(f"[WS_PROXY] ROSBridge relay closed")
        try:
            upstream.close()
        except Exception:
            pass

    # ---- Existing: Camera video proxy ----
    def _proxy_video(self):
        # Decode %2F etc in query params so web_video_server gets /camera/image_raw
        raw_path = self.path[len("/camera_proxy"):]
        parts = urllib.parse.urlsplit(raw_path)
        decoded_qs = urllib.parse.unquote(parts.query)
        downstream_path = f"{parts.path}?{decoded_qs}" if decoded_qs else parts.path
        try:
            sock = socket.create_connection((VIDEO_HOST, VIDEO_PORT), timeout=5)
            req = f"GET {downstream_path} HTTP/1.0\r\nHost: {VIDEO_HOST}:{VIDEO_PORT}\r\nConnection: close\r\n\r\n"
            sock.sendall(req.encode())

            # Read upstream HTTP headers
            buf = b""
            while b"\r\n\r\n" not in buf:
                chunk = sock.recv(4096)
                if not chunk:
                    self.send_error(502, "No response from camera")
                    sock.close()
                    return
                buf += chunk

            header_end = buf.index(b"\r\n\r\n")
            header_bytes = buf[:header_end]
            body_start = buf[header_end + 4:]

            # Parse upstream headers
            headers = {}
            for line in header_bytes.decode("latin-1").split("\r\n")[1:]:  # skip status line
                if ":" in line:
                    k, v = line.split(":", 1)
                    headers[k.strip().lower()] = v.strip()

            content_type = headers.get("content-type", "application/octet-stream")

            # Send our own proper HTTP response
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Access-Control-Allow-Origin", "*")
            if "content-length" in headers:
                self.send_header("Content-Length", headers["content-length"])
            self.end_headers()

            # Write any body data already read
            if body_start:
                self.wfile.write(body_start)
                self.wfile.flush()

            # Stream remaining data
            while True:
                chunk = sock.recv(8192)
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    break
            sock.close()
        except Exception as e:
            try:
                self.send_error(502, f"Camera proxy error: {e}")
            except Exception:
                pass

    def log_message(self, format, *args):
        msg = args[0] if args else ""
        if "/camera_proxy/" not in str(msg):
            super().log_message(format, *args)


class ThreadedHTTPServer(http.server.HTTPServer):
    def process_request(self, request, client_address):
        t = threading.Thread(target=self._handle, args=(request, client_address))
        t.daemon = True
        t.start()

    def _handle(self, request, client_address):
        try:
            self.finish_request(request, client_address)
        except Exception:
            self.handle_error(request, client_address)
        finally:
            self.shutdown_request(request)


os.chdir(DIRECTORY)

print(f"[CONFIG] Claude CLI: {CLAUDE_CMD or 'NOT FOUND'}")
print(f"[CONFIG] Whisper CLI: {WHISPER_CMD or 'NOT FOUND'}")
print(f"[CONFIG] Whisper model: {WHISPER_MODEL or 'NOT FOUND'}")

# Start HTTPS server in a background thread if certs exist
if os.path.isfile(CERT_FILE) and os.path.isfile(KEY_FILE):
    https_server = ThreadedHTTPServer(("0.0.0.0", HTTPS_PORT), RemoteHandler)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(CERT_FILE, KEY_FILE)
    https_server.socket = ctx.wrap_socket(https_server.socket, server_side=True)
    t = threading.Thread(target=https_server.serve_forever, daemon=True)
    t.start()
    print(f"Serving ainex-remote on https://0.0.0.0:{HTTPS_PORT}")

with ThreadedHTTPServer(("0.0.0.0", PORT), RemoteHandler) as httpd:
    print(f"Serving ainex-remote on http://0.0.0.0:{PORT}")
    httpd.serve_forever()
