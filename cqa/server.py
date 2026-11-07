"""Local review page: browse chats, compare model answers, save the team's labels.

python run.py serve  ->  http://127.0.0.1:8000   (listens on this computer only)
"""

from __future__ import annotations

import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import config, labels
from .checklist import CRITERIA
from .data import load_dataset
from .evaluate import evaluate
from .model import ModelError
from .results import read_json
from .scoring import ScoreError, final_score

ROOT = config.ROOT
DATASETS = ROOT / "datasets"
WEB = ROOT / "web"
PREDICTION_DIRS = (ROOT / "results", ROOT / "runs")
_lock = threading.Lock()


def _dataset_file(name: str) -> Path:
    p = (DATASETS / name).resolve()
    if p.parent != DATASETS.resolve() or p.suffix != ".json" or not p.exists():
        raise FileNotFoundError(f"dataset {name!r} not found in datasets/")
    return p


def _prediction_file(rel: str) -> Path:
    p = (ROOT / rel).resolve()
    if p.suffix != ".json" or not p.exists() or not any(d.resolve() in p.parents for d in PREDICTION_DIRS):
        raise FileNotFoundError(f"predictions file {rel!r} not found under results/ or runs/")
    return p


def _conv_json(c) -> dict:
    return {"id": c.conversation_id, "language": c.language, "channel": c.channel, "error": c.error,
            "turns": [{"n": t.number, "speaker": t.speaker, "text": t.text, "ts": t.timestamp} for t in c.turns]}


class Handler(BaseHTTPRequestHandler):
    server_version = "CQA/2"

    def log_message(self, fmt, *args):  # quieter console
        pass

    def _send(self, status: int, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n > 5_000_000:
            raise ValueError("request too large")
        return json.loads(self.rfile.read(n).decode("utf-8") or "{}")

    def do_GET(self):
        self._route("GET")

    def do_POST(self):
        # Only same-origin pages may write labels (blocks other websites posting to localhost).
        origin = self.headers.get("Origin")
        if origin and urlparse(origin).netloc != self.headers.get("Host"):
            return self._send(HTTPStatus.FORBIDDEN, {"error": "cross-origin request refused"})
        self._route("POST")

    def _route(self, method: str):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if method == "GET" and url.path in ("/", "/index.html"):
                return self._send(200, (WEB / "index.html").read_bytes(), "text/html; charset=utf-8")
            if method == "GET" and url.path == "/api/checklist":
                return self._send(200, [{"code": c.code, "number": c.number, "title": c.title, "weight": c.weight,
                                         "allowed": list(c.allowed), "definition": c.definition} for c in CRITERIA])
            if method == "GET" and url.path == "/api/datasets":
                out = []
                for p in sorted(DATASETS.glob("*.json")):
                    try:
                        out.append({"name": p.name, "count": len(load_dataset(p))})
                    except Exception as e:  # noqa: BLE001 - show unreadable files instead of hiding them
                        out.append({"name": p.name, "count": 0, "error": str(e)})
                return self._send(200, out)
            if method == "GET" and url.path == "/api/dataset":
                return self._send(200, [_conv_json(c) for c in load_dataset(_dataset_file(q.get("name", "")))])
            if method == "GET" and url.path == "/api/prediction-files":
                files = []
                for d in PREDICTION_DIRS:
                    for p in sorted(d.rglob("*.json")) if d.exists() else []:
                        files.append(p.relative_to(ROOT).as_posix())
                return self._send(200, files)
            if method == "GET" and url.path == "/api/predictions":
                data = read_json(_prediction_file(q.get("file", "")))
                return self._send(200, {r["conversation_id"]: r for r in data.get("results", []) if isinstance(r, dict)})
            if method == "GET" and url.path == "/api/labels":
                _dataset_file(q.get("name", ""))
                return self._send(200, labels.load(labels.labels_path(q["name"]))["labels"])
            if method == "POST" and url.path == "/api/labels":
                name = q.get("name", "")
                convs = {c.conversation_id: c for c in load_dataset(_dataset_file(name))}
                body = self._body()
                conv = convs.get(body.get("conversation_id"))
                if conv is None:
                    raise ValueError("unknown conversation_id")
                with _lock:
                    entry = labels.save_label(name, conv.conversation_id, body.get("criteria") or {},
                                              len(conv.turns), body.get("reviewed", True))
                return self._send(200, entry)
            if method == "POST" and url.path == "/api/score":
                return self._send(200, {"final_score": final_score(self._body().get("values") or {})})
            if method == "POST" and url.path == "/api/evaluate":
                convs = {c.conversation_id: c for c in load_dataset(_dataset_file(q.get("name", "")))}
                conv = convs.get(q.get("id"))
                if conv is None:
                    raise ValueError("unknown conversation_id")
                entry, _ = evaluate(conv, config.model_config(config.load()))
                return self._send(200, entry)
            return self._send(404, {"error": "not found"})
        except FileNotFoundError as e:
            return self._send(404, {"error": str(e)})
        except (ValueError, ScoreError, KeyError) as e:
            return self._send(400, {"error": str(e)})
        except ModelError as e:
            return self._send(502, {"error": str(e)})


def serve(port: int = 8000) -> None:
    DATASETS.mkdir(exist_ok=True)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Review page: http://127.0.0.1:{port}  (Ctrl+C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("stopped")
