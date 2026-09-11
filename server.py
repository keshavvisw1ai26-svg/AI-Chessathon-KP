"""Local server for the web frontend: serves web/ statically and wraps the real agent.

    uv run python web/server.py    (then open http://127.0.0.1:8360)

The GitHub Pages deployment serves index.html statically; the board talks to this server
on 127.0.0.1, so the agent itself always runs locally.
"""
import json
import sys
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

print("Loading agent (numba compile, up to a minute on first run)...", flush=True)
import agent  # noqa: E402

PORT = 8360


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(ROOT), **k)

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/new":
            agent.PONDERER.stop()
            agent.GAME.__init__()
            out = {"ok": True}
        elif self.path == "/move":
            t = time.perf_counter()
            mv = agent.get_move(body["fen"], int(body.get("time_left_ms", 15000)))
            out = {"move": mv, "time_s": time.perf_counter() - t}
        else:
            self.send_response(404)
            self._cors()
            self.end_headers()
            return
        data = json.dumps(out).encode()
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):  # quiet
        pass


if __name__ == "__main__":
    print("Waiting for the engine to finish compiling (up to ~2 min on a busy machine)...", flush=True)
    agent.COMPILED.wait(240)
    print(f"Engine ready: {agent.COMPILED.is_set()}. Serving on http://127.0.0.1:{PORT}", flush=True)
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
