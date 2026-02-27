#!/usr/bin/env python3
"""Simple HTTP server to serve ainex-remote control page."""
import http.server
import os

PORT = 8888
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

os.chdir(DIRECTORY)
handler = http.server.SimpleHTTPRequestHandler
with http.server.HTTPServer(("0.0.0.0", PORT), handler) as httpd:
    print(f"Serving ainex-remote on http://0.0.0.0:{PORT}")
    httpd.serve_forever()
