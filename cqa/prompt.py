"""Промпт и схема ответа. Используются и при оценке, и при подготовке данных для дообучения, поэтому
дообученная модель видит ровно тот же текст, что и при работе."""

from __future__ import annotations

import json

from .checklist import CRITERIA, GENERAL_RULES
from .data import Conversation

RATIONALE_LANGUAGE = {
    "ru": "на русском языке",
    "kk": "на казахском языке",
    "en": "на английском языке",
}


def system_prompt(rationale_language: str = "ru") -> str:
    lang = RATIONALE_LANGUAGE.get(rationale_language, RATIONALE_LANGUAGE["ru"])
    lines = [
        "Ты строгий специалист по контролю качества чата поддержки брокера. Оцени работу оператора по "
        "чек-листу ниже.",
        "",
        "ОБЩИЕ ПРАВИЛА",
        GENERAL_RULES,
        "",
        "КРИТЕРИИ",
    ]
    for c in CRITERIA:
        lines.append(f"[{c.code}] {c.number}. {c.title}: {c.definition}")
    lines += [
        "",
        "ФОРМАТ ОТВЕТА",
        "Реплики пронумерованы [1], [2], … по всему транскрипту (включая клиента, бота и системные сообщения). "
        "Верни только JSON: {\"criteria\": {\"<код>\": {\"rationale\": строка, \"evidence_turns\": [числа], "
        "\"value\": …}}}, все 8 кодов ровно по одному разу.",
        f"- rationale: одно-два конкретных предложения {lang}, почему выбрано это значение; можно коротко "
        "процитировать реплику.",
        "- evidence_turns: номера реплик, которые подтверждают решение. Для пропущенного действия — место, где оно "
        "ожидалось. Для N/A — реплики, показывающие, что условие неприменимости выполнено.",
        "- value: true/false/null для критериев Да/Нет/N/A; одно из перечисленных чисел для consultation и "
        "dialogue. critical = true только при грубой ошибке.",
        "Сначала напиши обоснование, пройдя шаги правила критерия, затем выбери значение.",
    ]
    return "\n".join(lines)


def user_prompt(conv: Conversation) -> str:
    head = f"Язык: {conv.language or 'неизвестен'}. Канал: {conv.channel or 'неизвестен'}.\nТранскрипт:"
    body = "\n".join(
        f"[{t.number}] {t.speaker}{f' ({t.timestamp})' if t.timestamp else ''}: {t.text}" for t in conv.turns
    )
    return f"{head}\n{body}"


def messages(conv: Conversation, rationale_language: str = "ru") -> list[dict]:
    return [{"role": "system", "content": system_prompt(rationale_language)},
            {"role": "user", "content": user_prompt(conv)}]


def output_schema() -> dict:
    """JSON-схема для ограниченной генерации (Ollama `format`). Обоснование идёт первым."""
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
    """Ответ ассистента для дообучения — в том же порядке ключей, что задаёт схема."""
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
    # Кириллица в токенизаторе Qwen — в среднем ~3 символа на токен (оценка с запасом).
    return len(text) // 3 + 1
