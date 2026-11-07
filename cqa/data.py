"""Loading organiser datasets. Records are validated one by one, so a malformed conversation becomes an
error entry with its original ID instead of breaking the whole run."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

SPEAKERS = {
    "operator": "OPERATOR", "agent": "OPERATOR", "manager": "OPERATOR", "support": "OPERATOR",
    "оператор": "OPERATOR",
    "client": "CLIENT", "customer": "CLIENT", "user": "CLIENT", "клиент": "CLIENT",
    "bot": "BOT", "chatbot": "BOT", "бот": "BOT",
    "system": "SYSTEM", "event": "SYSTEM", "система": "SYSTEM",
}


@dataclass
class Turn:
    number: int          # 1-based index in the original transcript array
    speaker: str         # OPERATOR / CLIENT / BOT / SYSTEM (or the raw label upper-cased)
    text: str
    timestamp: str = ""


@dataclass
class Conversation:
    conversation_id: str              # exactly as in the source
    language: str = ""
    channel: str = ""
    turns: list[Turn] = field(default_factory=list)
    error: str | None = None          # set when the record cannot be evaluated

    @property
    def operator_turns(self) -> list[int]:
        return [t.number for t in self.turns if t.speaker == "OPERATOR"]


def raw_records(payload) -> list:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if payload.get("isSuccess") is False:
            raise ValueError("the dataset source reported an error (isSuccess=false)")
        data = payload.get("data")
        if isinstance(data, dict) and isinstance(data.get("records"), list):
            return data["records"]
        if isinstance(data, list):
            return data
        for key in ("records", "conversations", "items"):
            if isinstance(payload.get(key), list):
                return payload[key]
        if "conversation_id" in payload:
            return [payload]
    raise ValueError("no records found: expected a list, data.records or records")


def _text(value) -> str:
    return value if isinstance(value, str) else ("" if value is None else str(value))


def parse_record(raw, position: int) -> Conversation:
    if not isinstance(raw, dict):
        return Conversation(conversation_id=f"<record {position}>", error="record is not an object")
    cid = raw.get("conversation_id")
    if not isinstance(cid, str) or not cid.strip():
        cid = str(cid) if cid not in (None, "") else f"<record {position}>"
        return Conversation(conversation_id=cid, error="missing conversation_id")
    conv = Conversation(conversation_id=cid, language=_text(raw.get("language")).upper(),
                        channel=_text(raw.get("channel")))
    transcript = raw.get("transcript") or raw.get("messages") or raw.get("turns")
    if not isinstance(transcript, list) or not transcript:
        conv.error = "empty or missing transcript"
        return conv
    for i, m in enumerate(transcript, start=1):
        if not isinstance(m, dict):
            conv.error = f"transcript item {i} is not an object"
            return conv
        role = _text(m.get("speaker") or m.get("role") or m.get("author")).strip()
        speaker = SPEAKERS.get(role.lower(), role.upper() or "UNKNOWN")
        text = _text(m.get("text") if "text" in m else m.get("message", m.get("content")))
        ts = _text(m.get("timestamp") or m.get("time") or m.get("ts"))
        conv.turns.append(Turn(number=i, speaker=speaker, text=text, timestamp=ts))
    if not conv.operator_turns:
        conv.error = "no operator messages in the transcript"
    return conv


def load_dataset(path: str | Path) -> list[Conversation]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    conversations, seen = [], set()
    for position, raw in enumerate(raw_records(payload), start=1):
        conv = parse_record(raw, position)
        if conv.conversation_id in seen:
            continue  # one result entry per ID: the first occurrence is evaluated
        seen.add(conv.conversation_id)
        conversations.append(conv)
    return conversations


def dataset_ids(path: str | Path) -> list[str]:
    return [c.conversation_id for c in load_dataset(path)]
