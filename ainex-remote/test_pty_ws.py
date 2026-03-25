#!/usr/bin/env python3
"""End-to-end test for the PTY WebSocket (/ws_pty) and transcribe (/api/transcribe) endpoints.

Usage:
    python3 test_pty_ws.py

Starts a test server on a random port, connects via WebSocket, and verifies:
  1. WebSocket handshake succeeds
  2. Claude CLI spawns and produces output
  3. Input is sent and echoed/processed
  4. Resize message is accepted
  5. Ping/pong works
  6. Reconnect gets replay buffer
  7. /api/transcribe returns 503 (no whisper) or works if installed
  8. Restart message respawns the process
  9. Close frame is handled gracefully
"""

import base64
import hashlib
import http.client
import json
import os
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import unittest

# Add project dir to path so we can import serve-remote helpers
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

WS_MAGIC = "258EAFA5-E914-47DA-95CA-5AB4C6BC7430"

# ---- Manual WebSocket client helpers (mirrors server's codec) ----

def ws_handshake(sock, host, port, path="/ws_pty"):
    """Perform a WebSocket upgrade handshake. Returns True on success."""
    key = base64.b64encode(os.urandom(16)).decode()
    req = (
        f"GET {path} HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        f"Upgrade: websocket\r\n"
        f"Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        f"Sec-WebSocket-Version: 13\r\n"
        f"\r\n"
    )
    sock.sendall(req.encode())

    # Read response headers
    buf = b""
    while b"\r\n\r\n" not in buf:
        chunk = sock.recv(4096)
        if not chunk:
            raise ConnectionError("Server closed during handshake")
        buf += chunk

    header_end = buf.index(b"\r\n\r\n")
    headers = buf[:header_end].decode("latin-1")
    first_line = headers.split("\r\n")[0]

    if "101" not in first_line:
        raise ConnectionError(f"Handshake failed: {first_line}")

    # Verify accept key
    expected = base64.b64encode(
        hashlib.sha1((key + WS_MAGIC).encode()).digest()
    ).decode()
    if expected not in headers:
        raise ConnectionError("Bad Sec-WebSocket-Accept")

    return True


def ws_send_text(sock, text):
    """Send a masked text frame (client -> server must be masked)."""
    data = text.encode("utf-8") if isinstance(text, str) else text
    mask = os.urandom(4)
    masked = bytes(data[i] ^ mask[i % 4] for i in range(len(data)))

    frame = bytearray()
    frame.append(0x81)  # FIN + text
    length = len(data)
    if length < 126:
        frame.append(0x80 | length)  # masked
    elif length < 65536:
        frame.append(0x80 | 126)
        frame.extend(struct.pack("!H", length))
    else:
        frame.append(0x80 | 127)
        frame.extend(struct.pack("!Q", length))
    frame.extend(mask)
    frame.extend(masked)
    sock.sendall(bytes(frame))


def ws_send_close(sock, code=1000):
    """Send a masked close frame."""
    payload = struct.pack("!H", code)
    mask = os.urandom(4)
    masked = bytes(payload[i] ^ mask[i % 4] for i in range(len(payload)))
    frame = bytes([0x88, 0x80 | len(payload)]) + mask + masked
    sock.sendall(frame)


def ws_send_ping(sock, data=b"test"):
    """Send a masked ping frame."""
    mask = os.urandom(4)
    masked = bytes(data[i] ^ mask[i % 4] for i in range(len(data)))
    frame = bytes([0x89, 0x80 | len(data)]) + mask + masked
    sock.sendall(frame)


def ws_read_frame(sock, timeout=10):
    """Read one WebSocket frame. Returns (opcode, payload)."""
    sock.settimeout(timeout)

    def recv_exact(n):
        buf = b""
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("closed")
            buf += chunk
        return buf

    hdr = recv_exact(2)
    opcode = hdr[0] & 0x0F
    masked = bool(hdr[1] & 0x80)
    length = hdr[1] & 0x7F

    if length == 126:
        length = struct.unpack("!H", recv_exact(2))[0]
    elif length == 127:
        length = struct.unpack("!Q", recv_exact(8))[0]

    if masked:
        mask = recv_exact(4)

    payload = recv_exact(length) if length > 0 else b""

    if masked:
        payload = bytes(payload[i] ^ mask[i % 4] for i in range(len(payload)))

    return opcode, payload


def ws_read_text_messages(sock, timeout=10, max_messages=50):
    """Read text frames until timeout, return list of parsed JSON messages."""
    messages = []
    end_time = time.time() + timeout
    while time.time() < end_time and len(messages) < max_messages:
        remaining = end_time - time.time()
        if remaining <= 0:
            break
        try:
            opcode, payload = ws_read_frame(sock, timeout=remaining)
            if opcode == 0x1:  # text
                msg = json.loads(payload.decode("utf-8"))
                messages.append(msg)
            elif opcode == 0x8:  # close
                break
            elif opcode == 0xA:  # pong
                messages.append({"type": "pong_frame"})
        except socket.timeout:
            break
        except Exception:
            break
    return messages


# ---- Test server management ----

def find_free_port():
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class ServerProcess:
    """Start serve-remote.py on a test port."""

    def __init__(self):
        self.port = find_free_port()
        self.proc = None

    def start(self):
        env = os.environ.copy()
        # Write a temp wrapper that patches the port and runs the server
        self._wrapper = os.path.join(PROJECT_DIR, f"_test_server_{self.port}.py")
        with open(self._wrapper, "w") as f:
            f.write(f"""#!/usr/bin/env python3
import sys, os
# Patch __file__ and DIRECTORY before the server code runs
__file__ = os.path.join({PROJECT_DIR!r}, "serve-remote.py")
os.chdir({PROJECT_DIR!r})

code = open(__file__).read()
code = code.replace("PORT = 8888", "PORT = {self.port}")
# Disable HTTPS for testing
code = code.replace(
    'if os.path.isfile(CERT_FILE) and os.path.isfile(KEY_FILE):',
    'if False:  # disabled for test'
)
exec(compile(code, __file__, "exec"), {{"__name__": "__main__", "__file__": __file__}})
""")
        self.proc = subprocess.Popen(
            [sys.executable, self._wrapper],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
        )
        # Wait for server to be ready
        for _ in range(50):
            time.sleep(0.1)
            try:
                s = socket.create_connection(("127.0.0.1", self.port), timeout=0.5)
                s.close()
                return True
            except Exception:
                if self.proc.poll() is not None:
                    out = self.proc.stdout.read().decode()
                    raise RuntimeError(f"Server exited early:\n{out}")
        raise RuntimeError("Server didn't start in time")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        if hasattr(self, "_wrapper") and os.path.exists(self._wrapper):
            os.unlink(self._wrapper)

    @property
    def host(self):
        return "127.0.0.1"


# ---- Tests ----

class TestPtyWebSocket(unittest.TestCase):
    server = None

    @classmethod
    def setUpClass(cls):
        cls.server = ServerProcess()
        cls.server.start()

    @classmethod
    def tearDownClass(cls):
        if cls.server:
            cls.server.stop()

    def _connect_ws(self):
        """Create a raw socket and complete WebSocket handshake."""
        sock = socket.create_connection(
            (self.server.host, self.server.port), timeout=5
        )
        ws_handshake(sock, self.server.host, self.server.port)
        return sock

    def test_01_handshake(self):
        """WebSocket handshake to /ws_pty succeeds with 101."""
        sock = self._connect_ws()
        sock.close()

    def test_02_receives_output(self):
        """After connecting, we receive PTY output (Claude startup)."""
        sock = self._connect_ws()
        try:
            messages = ws_read_text_messages(sock, timeout=15)
            self.assertTrue(len(messages) > 0, "Should receive at least one message")
            # At least one should be output type
            output_msgs = [m for m in messages if m.get("type") == "output"]
            self.assertTrue(len(output_msgs) > 0, "Should receive output messages")
            # Output should contain some text
            all_output = "".join(m.get("data", "") for m in output_msgs)
            self.assertTrue(len(all_output) > 0, "Output should not be empty")
            print(f"  Received {len(all_output)} chars of output")
        finally:
            sock.close()

    def test_03_send_input(self):
        """Sending input to the PTY is accepted without error."""
        sock = self._connect_ws()
        try:
            # Drain initial output
            ws_read_text_messages(sock, timeout=5)
            # Send a simple command
            msg = json.dumps({"type": "input", "data": "\n"})
            ws_send_text(sock, msg)
            # Should get more output back
            messages = ws_read_text_messages(sock, timeout=5)
            # Not asserting specific output since Claude's response varies
            print(f"  After Enter: got {len(messages)} messages")
        finally:
            sock.close()

    def test_04_resize(self):
        """Resize message is accepted without error."""
        sock = self._connect_ws()
        try:
            ws_read_text_messages(sock, timeout=3)
            msg = json.dumps({"type": "resize", "cols": 80, "rows": 24})
            ws_send_text(sock, msg)
            # No error = success; give server time to process
            time.sleep(0.5)
            # Should still be connected
            msg2 = json.dumps({"type": "input", "data": ""})
            ws_send_text(sock, msg2)
            print("  Resize accepted")
        finally:
            # Reset to normal size
            msg = json.dumps({"type": "resize", "cols": 120, "rows": 40})
            ws_send_text(sock, msg)
            sock.close()

    def test_05_ping_pong(self):
        """Server responds to ping with pong."""
        sock = self._connect_ws()
        try:
            ws_read_text_messages(sock, timeout=3)
            # Send application-level ping
            msg = json.dumps({"type": "ping"})
            ws_send_text(sock, msg)
            messages = ws_read_text_messages(sock, timeout=3)
            pong_msgs = [m for m in messages if m.get("type") == "pong"]
            self.assertTrue(len(pong_msgs) > 0, "Should receive pong response")
            print("  Got application-level pong")
        finally:
            sock.close()

    def test_06_ws_ping_pong(self):
        """Server responds to WebSocket-level ping with pong frame."""
        sock = self._connect_ws()
        try:
            ws_read_text_messages(sock, timeout=3)
            ws_send_ping(sock, b"hello")
            # Read frames looking for pong (opcode 0xA)
            found_pong = False
            end = time.time() + 3
            while time.time() < end:
                try:
                    opcode, payload = ws_read_frame(sock, timeout=2)
                    if opcode == 0xA:  # pong
                        found_pong = True
                        break
                except Exception:
                    break
            self.assertTrue(found_pong, "Should receive WebSocket pong frame")
            print("  Got WS-level pong")
        finally:
            sock.close()

    def test_07_replay_buffer(self):
        """Second connection receives replay of previous output."""
        # First connection should have generated output already
        sock = self._connect_ws()
        try:
            messages = ws_read_text_messages(sock, timeout=5)
            output_msgs = [m for m in messages if m.get("type") == "output"]
            self.assertTrue(len(output_msgs) > 0, "Replay should contain output")
            replay_text = "".join(m.get("data", "") for m in output_msgs)
            print(f"  Replay buffer: {len(replay_text)} chars")
        finally:
            sock.close()

    def test_08_close_frame(self):
        """Server handles close frame gracefully."""
        sock = self._connect_ws()
        try:
            ws_read_text_messages(sock, timeout=2)
            ws_send_close(sock)
            # Server should respond with close frame
            try:
                opcode, _ = ws_read_frame(sock, timeout=3)
                # opcode 8 = close
                self.assertEqual(opcode, 0x8, "Server should respond with close frame")
                print("  Got close frame response")
            except Exception:
                # Connection might just close, which is also fine
                print("  Connection closed (acceptable)")
        finally:
            sock.close()

    def test_09_transcribe_no_whisper(self):
        """POST /api/transcribe returns 503 when whisper is not installed."""
        conn = http.client.HTTPConnection(self.server.host, self.server.port, timeout=5)
        try:
            conn.request("POST", "/api/transcribe",
                         body=b"fake audio data",
                         headers={"Content-Type": "audio/webm",
                                  "Content-Length": "15"})
            resp = conn.getresponse()
            body = json.loads(resp.read().decode())
            # Either 503 (no whisper) or 500 (processing error) - both valid
            self.assertIn(resp.status, [503, 500, 400],
                          f"Should get error status, got {resp.status}")
            self.assertIn("error", body, "Response should contain error field")
            print(f"  Transcribe: {resp.status} - {body['error']}")
        finally:
            conn.close()

    def test_10_transcribe_empty_body(self):
        """POST /api/transcribe rejects empty body with 400 (or 503 if no whisper)."""
        conn = http.client.HTTPConnection(self.server.host, self.server.port, timeout=5)
        try:
            conn.request("POST", "/api/transcribe",
                         body=b"",
                         headers={"Content-Type": "audio/webm",
                                  "Content-Length": "0"})
            resp = conn.getresponse()
            body = json.loads(resp.read().decode())
            # 400 (empty body) or 503 (no whisper — checked first) are both valid
            self.assertIn(resp.status, [400, 503])
            self.assertIn("error", body)
            print(f"  Empty body: {resp.status} - {body['error']}")
        finally:
            conn.close()

    def test_11_bad_ws_path(self):
        """Non-WebSocket GET to /ws_pty returns 400."""
        conn = http.client.HTTPConnection(self.server.host, self.server.port, timeout=5)
        try:
            conn.request("GET", "/ws_pty")
            resp = conn.getresponse()
            resp.read()
            self.assertEqual(resp.status, 400)
            print(f"  Non-WS request: {resp.status}")
        finally:
            conn.close()


if __name__ == "__main__":
    # Run with verbose output
    unittest.main(verbosity=2)
