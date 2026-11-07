"""Turn a conversation into a results.json entry: model call(s) -> validated criteria -> score."""

from __future__ import annotations

import json
import time
from collections import Counter

from . import prompt
from . import facts as fact_check
from .checklist import BY_CODE, CODES, CRITERIA
from .data import Conversation
from .model import ModelConfig, ModelError, chat
from .scoring import ScoreError, check_value, final_score


class OutputError(ValueError):
    pass


_WORDS = {"yes": True, "true": True, "да": True, "иә": True, "no": False, "false": False, "нет": False,
          "жоқ": False, "n/a": None, "na": None, "null": None, "none": None, "": None}


def _coerce(code: str, v):
    c = BY_CODE[code]
    if isinstance(v, str):
        key = v.strip().lower()
        if key in _WORDS:
            v = _WORDS[key]
        else:
            try:
                v = int(float(key))
            except ValueError:
                raise OutputError(f"{code}: unreadable value {v!r}")
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    if c.scale_max is None and type(v) is int and v in (0, 1):
        v = bool(v)
    try:
        check_value(code, v)
    except ScoreError as e:
        raise OutputError(str(e))
    return v


def _fallback_evidence(conv: Conversation, value) -> list[int]:
    if value is None:  # N/A: the request context, usually the customer's first message
        clients = [t.number for t in conv.turns if t.speaker == "CLIENT"]
        return clients[:1] or [1]
    return conv.operator_turns or [1]


def parse_output(text: str, conv: Conversation) -> dict:
    """Model JSON -> {code: {value, rationale, evidence_turns}}. Raises OutputError if unusable."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise OutputError("answer is not JSON")
        try:
            data = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            raise OutputError("answer is not valid JSON")
    crit = data.get("criteria", data) if isinstance(data, dict) else data
    if isinstance(crit, list):  # also accept the submission list shape
        crit = {x.get("criterion_id"): x for x in crit if isinstance(x, dict)}
    if not isinstance(crit, dict):
        raise OutputError("no criteria object in the answer")
    n = len(conv.turns)
    out = {}
    for code in CODES:
        item = crit.get(code)
        if not isinstance(item, dict) or "value" not in item:
            raise OutputError(f"criterion {code} missing")
        value = _coerce(code, item["value"])
        raw_turns = item.get("evidence_turns") or []
        turns = sorted({int(t) for t in raw_turns if isinstance(t, (int, float)) and not isinstance(t, bool)
                        and 1 <= int(t) <= n})
        rationale = str(item.get("rationale") or "").strip() or f"{BY_CODE[code].title}: модель не дала обоснования."
        out[code] = {"value": value, "rationale": rationale, "evidence_turns": turns or _fallback_evidence(conv, value)}
    return out


def _majority(runs: list[dict]) -> dict:
    final = {}
    for code in CODES:
        counts = Counter(json.dumps(r[code]["value"]) for r in runs)
        top = max(counts.values())
        # tie -> value of the earliest run among the most frequent ones
        winner = next(json.dumps(r[code]["value"]) for r in runs if counts[json.dumps(r[code]["value"])] == top)
        final[code] = next(r[code] for r in runs if json.dumps(r[code]["value"]) == winner)
    return final


def to_entry(conv_id: str, criteria: dict) -> dict:
    values = {code: criteria[code]["value"] for code in CODES}
    return {
        "conversation_id": conv_id,
        "status": "ok",
        "criteria": [{"criterion_id": c.code, "value": criteria[c.code]["value"],
                      "rationale": criteria[c.code]["rationale"],
                      "evidence_turns": criteria[c.code]["evidence_turns"]} for c in CRITERIA],
        "score": {"final_score": final_score(values), "critical_flag": values["critical"] is True},
    }


def error_entry(conv_id: str, reason: str) -> dict:
    return {"conversation_id": conv_id, "status": "error", "criteria": [], "score": None, "error": reason}


def evaluate(conv: Conversation, cfg: ModelConfig, votes: int = 1, retries: int = 2, log=None,
             progress=None) -> tuple[dict, dict]:
    """Returns (results.json entry, details for debugging). ModelError(fatal) propagates."""
    if conv.error:
        return error_entry(conv.conversation_id, conv.error), {"error": conv.error}
    msgs = prompt.messages(conv, cfg.rationale_language)
    est = prompt.approx_tokens(msgs[0]["content"] + msgs[1]["content"])
    if est + cfg.num_predict > cfg.num_ctx:
        msg = f"prompt ~{est} tokens + {cfg.num_predict} output exceeds num_ctx {cfg.num_ctx}; raise --num-ctx"
        if log:
            log(f"  warning: {msg}")
    schema = prompt.output_schema() if cfg.backend == "ollama" else None
    runs, details, last_error = [], {"attempts": []}, None
    for i in range(max(1, votes)):
        temp = cfg.temperature if i == 0 else max(cfg.temperature, 0.5)
        for attempt in range(retries + 1):
            started = time.time()
            try:
                text, usage = chat(cfg, msgs, schema, temperature=temp if attempt == 0 else max(temp, 0.3),
                                   on_chunk=progress)
                parsed = parse_output(text, conv)
                runs.append(parsed)
                details["attempts"].append({"ok": True, "seconds": round(time.time() - started, 1), "usage": usage,
                                            "raw": text})
                break
            except ModelError as e:
                if e.fatal:
                    raise
                last_error = str(e)
            except OutputError as e:
                last_error = f"invalid model answer: {e}"
            details["attempts"].append({"ok": False, "seconds": round(time.time() - started, 1), "error": last_error})
            if log:
                log(f"\n  attempt failed: {last_error}" + ("; retrying" if attempt < retries else ""))
    if not runs:
        return error_entry(conv.conversation_id, last_error or "model failed"), details
    final = _majority(runs)
    if cfg.fact_check:
        # Второй короткий вызов: факты с цитатами для 01, 02, 07 (см. cqa/facts.py).
        found, problem = fact_check.ask(conv, cfg)  # ModelError(fatal) propagates like the main call
        if found is None:
            details["fact_check"] = {"error": problem}
            if log:
                log(f"\n  fact check failed, main prompt answers kept: {problem}")
        else:
            checked = fact_check.apply(final, found, conv)
            details["fact_check"] = {"facts": found, "changed": {c: [final[c]["value"], checked[c]["value"]]
                                                                for c in ("contact", "discovery", "closure")
                                                                if final[c]["value"] != checked[c]["value"]}}
            final = checked
    return to_entry(conv.conversation_id, final), details
