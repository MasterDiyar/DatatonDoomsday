"""Settings come from config.json in the project root (optional); command-line flags override them.
No .env file and no environment variables are needed for the local Ollama model."""

from __future__ import annotations

import json
from pathlib import Path

from .model import ModelConfig

ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = ROOT / "config.json"

DEFAULTS = {
    "team_name": "The Jigas",
    "submission_model_name": "",   # name written to results.json; defaults to the model id
    "submission_mode": "local",
    "backend": "ollama",
    "base_url": "http://127.0.0.1:11434",
    "model": "qwen2.5:7b-instruct",
    "temperature": 0.0,
    "num_ctx": 16384,
    "num_predict": 3072,
    "timeout": 600,
    "api_key_env": "CQA_API_KEY",
    "rationale_language": "ru",    # язык обоснований в results.json: ru / kk / en
    "closure_check": True,         # отдельная проверка «Завершения» (точнее на DEV)
}


def load(path: Path = CONFIG_FILE) -> dict:
    cfg = dict(DEFAULTS)
    if path.exists():
        user = json.loads(path.read_text(encoding="utf-8-sig"))
        unknown = set(user) - set(DEFAULTS)
        if unknown:
            raise SystemExit(f"config.json: unknown keys {sorted(unknown)}; allowed: {sorted(DEFAULTS)}")
        cfg.update(user)
    return cfg


def model_config(cfg: dict) -> ModelConfig:
    return ModelConfig(backend=cfg["backend"], base_url=cfg["base_url"], model=cfg["model"],
                       temperature=float(cfg["temperature"]), num_ctx=int(cfg["num_ctx"]),
                       num_predict=int(cfg["num_predict"]), timeout=int(cfg["timeout"]),
                       api_key_env=cfg["api_key_env"], rationale_language=cfg["rationale_language"],
                       closure_check=bool(cfg["closure_check"]))
