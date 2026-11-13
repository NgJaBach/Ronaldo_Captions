"""The stadium. Serves the search interface and a tiny JSON API on your own machine (127.0.0.1 only).

  python src/serve.py            # then open the URL it prints (8000, or the next free port)
  python src/serve.py --port 9000

The page opens instantly and shows "Warming up" while the model loads in the background (about a minute).
API:  GET /api/status
      GET /api/meta     moods, tags and teams the page can offer
      GET /api/search?q=...&k=12&offset=0&mode=hybrid|semantic|keyword&max_len=40&since=2024-01-01&until=2025-01-01
                      &after=win|draw|loss&team=Al%20Nassr&within=1&tag=setback&no_ads=1&expand=0
"""
import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from search import Searcher  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PAGE_DIR = ROOT / "design" / "ronaldo-search" / "current"
MAX_K = 100

state = {"searcher": None, "error": None, "total": None, "tags": []}
TEAMS = ["Real Madrid", "Juventus", "Manchester United", "Al Nassr", "Portugal"]
lock = threading.Lock()  # one sqlite connection and one model: take turns


def warm_up() -> None:
    try:
        s = Searcher()
        s._embed_query("warm up")
        state["total"] = len(s.texty)
        try:
            state["tags"] = [t for (t,) in s.con.execute("SELECT tag FROM tags GROUP BY tag ORDER BY COUNT(*) DESC")]
        except Exception:  # tags.py not run yet: the page simply offers no tag filter
            state["tags"] = []
        state["searcher"] = s
        print("Ready. SIUUU.", flush=True)
    except Exception as e:  # shown as a status word in the page, detail in the console
        state["error"] = str(e)
        print("Warm-up failed:", e)


def int_arg(q, name, default=None, lo=0, hi=None):
    try:
        v = int(q.get(name, [default])[0])
    except (TypeError, ValueError):
        return default
    v = max(lo, v)
    return min(v, hi) if hi is not None else v


class Server(ThreadingHTTPServer):
    # On Windows, SO_REUSEADDR lets two programs share one port silently (requests then go to the older one, which may be
    # running old code). Turn it off so a taken port is an error and the next free port is tried instead.
    allow_reuse_address = False


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self) -> None:
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path in ("/", "/index.html"):
            page = PAGE_DIR / "index.html"
            if not page.exists():
                return self._send(404, b"page not built", "text/plain")
            return self._send(200, page.read_bytes(), "text/html; charset=utf-8")
        if u.path == "/api/status":
            return self._json({"ready": state["searcher"] is not None, "error": state["error"], "total": state["total"]})
        if u.path == "/api/search":
            s = state["searcher"]
            if s is None:
                return self._json({"error": "warming up"}, 503)
            query = (q.get("q", [""])[0]).strip()
            if not query:
                return self._json({"results": [], "total": 0})
            mode = q.get("mode", ["hybrid"])[0]
            if mode not in ("hybrid", "semantic", "keyword"):
                mode = "hybrid"
            after = q.get("after", [""])[0]
            tag = q.get("tag", [""])[0]
            team = q.get("team", [""])[0]
            filters = dict(mode=mode, max_len=int_arg(q, "max_len", None, 1, 5000),
                           since=q.get("since", [None])[0] or None, until=q.get("until", [None])[0] or None,
                           after=after if after in ("win", "draw", "loss") else None,
                           team=team if team in TEAMS else None,
                           tag=tag if tag in state["tags"] else None,
                           within=int_arg(q, "within", 1, 0, 2),
                           no_ads=q.get("no_ads", [""])[0] == "1",
                           expand=q.get("expand", ["1"])[0] != "0")
            try:
                with lock:
                    ranked = s.ranking(query, **filters)
                    results = s.page(ranked, int_arg(q, "k", 12, 1, MAX_K), int_arg(q, "offset", 0, 0))
                    mood = s.mood_for(query) if filters["expand"] and mode != "keyword" else None
            except ValueError:  # a malformed date
                return self._json({"error": "bad date"}, 400)
            return self._json({"results": results, "total": len(ranked), "mood": mood})
        if u.path == "/api/meta":
            s = state["searcher"]
            if s is None:
                return self._json({"error": "warming up"}, 503)
            return self._json({"moods": list(s.moods), "tags": state["tags"], "teams": TEAMS})
        self._send(404, b"not found", "text/plain")

    def log_message(self, *a) -> None:
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    srv = None
    for port in range(a.port, a.port + 20):  # another program (VS Code, say) may already hold the port
        try:
            srv = Server(("127.0.0.1", port), Handler)  # loopback only: nothing leaves your machine
            break
        except OSError:  # Windows reports a port held by another program as PermissionError (10013)
            print(f"Port {port} is taken, trying the next one...", flush=True)
    if srv is None:
        sys.exit(f"No free port in {a.port}-{a.port + 19}. Pass one with --port.")
    threading.Thread(target=warm_up, daemon=True).start()
    print(f"Stadium open: http://127.0.0.1:{srv.server_address[1]}  (model warming up, about a minute)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nFull time.")


if __name__ == "__main__":
    main()
