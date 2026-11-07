"""Prompt and output schema. Used for inference AND for building fine-tuning data, so the fine-tuned
model sees exactly the same text at training and evaluation time."""

from __future__ import annotations

import json

from .checklist import CRITERIA, GENERAL_RULES
from .data import Conversation


def system_prompt() -> str:
    lines = [
        "You are a strict quality auditor of a broker's live-chat support. Score the operator's work with the "
        "checklist below.",
        "",
        "GENERAL RULES",
        GENERAL_RULES,
        "",
        "CRITERIA",
    ]
    for c in CRITERIA:
        lines.append(f"[{c.code}] {c.number}. {c.title}: {c.definition}")
    lines += [
        "",
        "OUTPUT",
        "Turns are numbered [1], [2], ... over the whole transcript (customer, bot and system included). "
        "Return only JSON: {\"criteria\": {\"<code>\": {\"rationale\": str, \"evidence_turns\": [int], "
        "\"value\": ...}}} with all 8 codes exactly once.",
        "- rationale: one or two concrete sentences in English explaining the decision.",
        "- evidence_turns: the turn numbers that prove the decision. For a missing action, the turns where it "
        "was expected. For N/A, the turns showing that the N/A condition is met.",
        "- value: true/false/null for yes/no/N/A criteria; one of the listed numbers for consultation and "
        "dialogue. critical is true only if a severe violation is present.",
        "Write the rationale first, then decide the value.",
    ]
    return "\n".join(lines)


def user_prompt(conv: Conversation) -> str:
    head = f"Language: {conv.language or 'unknown'}. Channel: {conv.channel or 'unknown'}.\nTranscript:"
    body = "\n".join(
        f"[{t.number}] {t.speaker}{f' ({t.timestamp})' if t.timestamp else ''}: {t.text}" for t in conv.turns
    )
    return f"{head}\n{body}"


def messages(conv: Conversation) -> list[dict]:
    return [{"role": "system", "content": system_prompt()}, {"role": "user", "content": user_prompt(conv)}]


def output_schema() -> dict:
    """JSON schema for constrained decoding (Ollama `format`). Property order puts rationale first."""
    props = {}
    for c in CRITERIA:
        props[c.code] = {
            "type": "object",
            "properties": {
                "rationale": {"type": "string"},
                "evidence_turns": {"type": "array", "items": {"type": "integer"}},
                "value": {"enum": list(c.allowed)},
            },
            "required": ["rationale", "evidence_turns", "value"],
        }
    return {
        "type": "object",
        "properties": {"criteria": {"type": "object", "properties": props, "required": [c.code for c in CRITERIA]}},
        "required": ["criteria"],
    }


def target_json(criteria: dict) -> str:
    """Assistant answer for fine-tuning, in the same key order the schema enforces."""
    ordered = {
        c.code: {
            "rationale": criteria[c.code]["rationale"],
            "evidence_turns": criteria[c.code]["evidence_turns"],
            "value": criteria[c.code]["value"],
        }
        for c in CRITERIA
    }
    return json.dumps({"criteria": ordered}, ensure_ascii=False)


def approx_tokens(text: str) -> int:
    # Cyrillic averages ~3 characters per token for Qwen tokenizers; conservative estimate.
    return len(text) // 3 + 1
