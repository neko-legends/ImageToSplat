#!/usr/bin/env python3
"""Open the treasure chest demo in your browser: one image turned into a splat and four meshes.

    python demo/run_demo.py            # any Python 3.8+, nothing to install
    python demo/run_demo.py --port 8080 --no-browser

Buttons (or keys 1-5) switch between the splat and the meshes, W shows the wireframe. The page loads three.js
and Spark from a CDN, so it needs an internet connection.
"""
import argparse
import functools
import http.server
import socketserver
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGE = "public/viewer/compare.html?set=/demo/chest/chest_set.json"


class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=0, help="Port (default: any free one)")
    ap.add_argument("--no-browser", action="store_true", help="Only print the address")
    a = ap.parse_args()
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(("127.0.0.1", a.port), functools.partial(Handler, directory=str(ROOT))) as httpd:
        url = f"http://127.0.0.1:{httpd.server_address[1]}/{PAGE}"
        print(f"Chest demo: {url}\nCtrl+C to stop.")
        if not a.no_browser:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
