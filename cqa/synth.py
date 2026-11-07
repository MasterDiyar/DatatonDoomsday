"""Synthetic broker support chats (RU/KZ/EN) with controlled operator behaviour, to enlarge the training set.

Each chat comes from a scenario. Behaviours that unambiguously imply a checklist answer are stored as
`expected`; when the labelling model disagrees with them, `labels-from-results` drops the example.
"""

from __future__ import annotations

import json
import random

from .data import parse_record
from .model import ModelConfig, ModelError, chat

TOPICS_GENERAL = ["commission and tariff plans", "how to open a brokerage account", "trading hours of US and "
                  "KASE exchanges", "documents needed to open an account", "how dividends are credited",
                  "market vs limit orders", "installing and logging in to the mobile app"]
TOPICS_SECURE = ["the client's current portfolio balance", "withdrawing money to a new bank card",
                 "why a specific order of the client was not executed", "changing the phone number on the account",
                 "a statement of the client's transactions", "a password reset", "a blocked account"]

B = {  # behaviour: (instruction, implied checklist values)
    "good_open": ("In the FIRST operator message the operator greets in the client's language AND gives their name.", {"contact": True}),
    "no_intro": ("The first operator message greets but never gives the operator's name, and no name is given later.", {"contact": False}),
    "late_intro": ("The first operator message goes straight to business without a greeting; the name comes only later.", {"contact": False}),
    "full_info": ("The client states the question completely in the first message; nothing needs clarifying.", {"discovery": None}),
    "clarify_done": ("The client's first message is vague; the operator asks a targeted clarifying question before answering.", {"discovery": True}),
    "clarify_skipped": ("The client's first message is vague; the operator does NOT clarify and answers a guess.", {"discovery": False}),
    "id_not_needed": ("It is a general question; no identity verification is needed or done.", {"identification": None}),
    "id_done": ("Before any account data or action the operator asks to verify identity; a SYSTEM event confirms it; then the operator proceeds.", {"identification": True}),
    "id_skipped": ("The operator discloses account data / performs the action WITHOUT any identity verification.", {"identification": False}),
    "answer_full": ("The operator gives a correct, complete answer or a correct next step.", {}),
    "answer_partial": ("The client asks two important questions; the operator answers only one.", {}),
    "answer_wrong": ("The operator's key recommendation is clearly wrong.", {}),
    "client_leaves": ("The client greets and disconnects before formulating any request.", {"consultation": None}),
    "complaint_handled": ("The client is frustrated about a specific inconvenience; the operator responds to that specific problem.", {"empathy": True}),
    "complaint_ignored": ("The client is frustrated about a specific inconvenience; the operator ignores it and answers dryly.", {"empathy": False}),
    "clear_flow": ("Messages are clear; before a longer check the operator warns the client to wait.", {}),
    "silent_wait": ("At one point the operator is silent for 8+ minutes (visible in timestamps) without warning.", {}),
    "confusing": ("Operator messages contradict each other so the client cannot understand what to do.", {}),
    "good_close": ("At the end the operator offers further help or summarizes AND says goodbye.", {"closure": True}),
    "no_goodbye": ("After answering, the operator ends the chat without a goodbye; a SYSTEM event says the operator closed the chat.", {"closure": False}),
    "disconnect": ("A SYSTEM event shows the chat was cut off before it could be completed.", {"closure": None}),
    "clean": ("The operator is never rude, offers no private paid services and discloses nothing confidential.", {"critical": False}),
    "rude": ("At one point the operator insults or belittles the client.", {"critical": True}),
    "private_offer": ("The operator suggests transferring money to the operator personally for investing, or asks for the client's personal phone number for paid private consultations.", {"critical": True}),
    "leak": ("The operator discloses confidential information (e.g. another client's data).", {"critical": True}),
}
LANG = {"RU": "Russian", "KZ": "Kazakh (Cyrillic)", "EN": "English"}


def scenario(i: int, rng: random.Random, prefix: str) -> dict:
    w = lambda pairs: rng.choices([k for k, _ in pairs], [p for _, p in pairs])[0]
    language = w([("RU", 0.4), ("KZ", 0.3), ("EN", 0.3)])
    secure = rng.random() < 0.45
    b = [w([("good_open", 0.6), ("no_intro", 0.25), ("late_intro", 0.15)])]
    if rng.random() < 0.05:
        b += ["client_leaves", "clean"]
    else:
        b.append(w([("full_info", 0.4), ("clarify_done", 0.4), ("clarify_skipped", 0.2)]))
        b.append(w([("id_done", 0.6), ("id_skipped", 0.4)]) if secure else "id_not_needed")
        b.append(w([("answer_full", 0.6), ("answer_partial", 0.25), ("answer_wrong", 0.15)]))
        if rng.random() < 0.45:
            b.append(w([("complaint_handled", 0.6), ("complaint_ignored", 0.4)]))
        b.append(w([("clear_flow", 0.65), ("silent_wait", 0.2), ("confusing", 0.15)]))
        b.append(w([("good_close", 0.6), ("no_goodbye", 0.25), ("disconnect", 0.15)]))
        b.append(w([("clean", 0.82), ("rude", 0.06), ("private_offer", 0.07), ("leak", 0.05)]))
    expected = {}
    for k in b:
        expected.update(B[k][1])
    return {"id": f"{prefix}{i + 1:04d}", "language": language, "with_bot": rng.random() < 0.5,
            "topic": rng.choice(TOPICS_SECURE if secure else TOPICS_GENERAL), "behaviours": b, "expected": expected}


def generate(sc: dict, cfg: ModelConfig) -> dict:
    instruction = (
        "Write a realistic, fictional live-chat transcript between a client and a support operator of an "
        "investment broker. No real personal data.\n"
        f"Language of client and operator: {LANG[sc['language']]}.\nTopic: {sc['topic']}.\n"
        + ("It starts with an automated BOT message before the operator joins.\n" if sc["with_bot"] else "There is no bot.\n")
        + "Operator behaviour (must be clearly visible):\n" + "\n".join(f"- {B[k][0]}" for k in sc["behaviours"])
        + "\nRules: 6-20 turns; speaker is exactly client, operator, bot or system; system turns are short platform "
          "events; ISO 8601 UTC timestamps, increasing. Never mention these instructions or any checklist.\n"
          'Return only JSON: {"transcript":[{"timestamp":"2025-03-04T10:00:00Z","speaker":"client","text":"..."}]}'
    )
    schema = {"type": "object", "properties": {"transcript": {"type": "array", "items": {
        "type": "object", "properties": {"timestamp": {"type": "string"}, "speaker": {"enum": ["client", "operator", "bot", "system"]},
                                         "text": {"type": "string"}}, "required": ["timestamp", "speaker", "text"]}}},
              "required": ["transcript"]}
    text, _ = chat(cfg, [{"role": "user", "content": instruction}], schema if cfg.backend == "ollama" else None, temperature=0.9)
    record = {"conversation_id": sc["id"], "language": sc["language"], "channel": "agent chat",
              "transcript": json.loads(text)["transcript"]}
    conv = parse_record(record, 1)
    if conv.error:
        raise ModelError(conv.error)
    return record
