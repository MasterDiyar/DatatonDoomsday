"""Model backends, standard library only.

- ollama (default): native /api/chat. Sends the JSON schema as `format` (constrained decoding, so the
  answer always has the right shape) and num_ctx per request (no OLLAMA_CONTEXT_LENGTH needed).
- openai: any OpenAI-compatible /chat/completions API (e.g. a teacher model); key from an env variable.
"""

from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from urllib.parse import urlparse


class ModelError(Exception):
    def __init__(self, message: str, fatal: bool = False):
        super().__init__(message)
        self.fatal = fatal   # configuration problems: retrying the next chat will not help


@dataclass
class ModelConfig:
    backend: str = "ollama"
    base_url: str = "http://127.0.0.1:11434"
    model: str = "qwen2.5:7b-instruct"
    temperature: float = 0.0
    num_ctx: int = 16384
    num_predict: int = 3072
    timeout: int = 600
    seed: int = 42
    api_key_env: str = "CQA_API_KEY"


def _opener(url: str):
    host = (urlparse(url).hostname or "").lower()
    if host in ("127.0.0.1", "localhost", "::1"):
        # Never send local model traffic through a system/corporate proxy.
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener()


def _read_stream(resp, on_chunk) -> dict:
    """Ollama streams one JSON object per line. Collect the content, report progress, return the
    final object with the full content. A plain (non-streamed) JSON body also works."""
    parts, last = [], {}
    for raw in resp:
        line = raw.decode("utf-8").strip()
        if not line:
            continue
        obj = json.loads(line)
        if obj.get("error"):
            return obj
        parts.append((obj.get("message") or {}).get("content", ""))
        last = obj
        if on_chunk:
            on_chunk(len(parts))
    last["message"] = {"role": "assistant", "content": "".join(parts)}
    return last


def _post(url: str, body: dict, headers: dict, timeout: int, on_chunk=None) -> dict:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    try:
        with _opener(url).open(req, timeout=timeout) as resp:
            if body.get("stream"):
                return _read_stream(resp, on_chunk)
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        if e.code in (401, 403):
            raise ModelError(f"HTTP {e.code}: the API rejected the key. {detail}", fatal=True)
        if e.code == 404:
            raise ModelError(f"HTTP 404 at {url}: model or endpoint not found. For Ollama run "
                             f"`ollama pull <model>` and check the name. {detail}", fatal=True)
        if e.code == 400:
            raise ModelError(f"HTTP 400: request rejected. {detail}")
        raise ModelError(f"HTTP {e.code}: {detail}")
    except urllib.error.URLError as e:
        reason = getattr(e, "reason", e)
        if isinstance(reason, (socket.timeout, TimeoutError)):
            raise ModelError(f"timeout after {timeout}s at {url}")
        raise ModelError(f"cannot connect to {url} ({reason}). Is the model server running "
                         f"(`ollama serve`)?", fatal=True)
    except (socket.timeout, TimeoutError):
        raise ModelError(f"timeout after {timeout}s at {url}")
    except json.JSONDecodeError:
        raise ModelError(f"{url} returned a non-JSON response")


def chat(cfg: ModelConfig, messages: list[dict], schema: dict | None, temperature: float | None = None,
         on_chunk=None) -> tuple[str, dict]:
    """Returns (assistant text, usage dict). on_chunk(n_tokens) is called while Ollama generates."""
    temp = cfg.temperature if temperature is None else temperature
    base = cfg.base_url.rstrip("/")
    if cfg.backend == "ollama":
        if base.endswith("/v1"):
            base = base[:-3]
        body = {
            # Streaming: the timeout applies between chunks, and progress can be shown while generating.
            "model": cfg.model, "messages": messages, "stream": True,
            "options": {"temperature": temp, "num_ctx": cfg.num_ctx, "num_predict": cfg.num_predict,
                        "seed": cfg.seed},
        }
        if schema is not None:
            body["format"] = schema
        out = _post(f"{base}/api/chat", body, {}, cfg.timeout, on_chunk)
        if out.get("error"):
            raise ModelError(f"Ollama error: {out['error']}", fatal="not found" in str(out["error"]))
        text = (out.get("message") or {}).get("content", "")
        usage = {"input_tokens": out.get("prompt_eval_count"), "output_tokens": out.get("eval_count"),
                 "done_reason": out.get("done_reason"),
                 "load_seconds": round((out.get("load_duration") or 0) / 1e9, 1),
                 "prompt_seconds": round((out.get("prompt_eval_duration") or 0) / 1e9, 1),
                 "generate_seconds": round((out.get("eval_duration") or 0) / 1e9, 1)}
        if out.get("done_reason") == "length":
            raise ModelError(f"answer cut off at num_predict={cfg.num_predict} tokens; raise num_predict in config.json")
        return text, usage
    if cfg.backend == "openai":
        key = os.environ.get(cfg.api_key_env, "")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        body = {"model": cfg.model, "messages": messages, "temperature": temp,
                "response_format": {"type": "json_object"}}
        out = _post(f"{base}/chat/completions", body, headers, cfg.timeout)
        try:
            text = out["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise ModelError(f"unexpected API response: {str(out)[:300]}")
        u = out.get("usage") or {}
        return text, {"input_tokens": u.get("prompt_tokens"), "output_tokens": u.get("completion_tokens")}
    raise ModelError(f"unknown backend {cfg.backend!r} (use ollama or openai)", fatal=True)
