"""Правила для фактов, которые надёжнее проверить кодом, чем моделью 7B.

На DEV модель ошибалась в простых фактах: писала «оператор не представился» при «Здравствуйте, я Руслан»
и путала «уточнил» (задал вопрос) с «уточнил» (пояснил). Здесь эти факты определяются по тексту, а модель
решает только то, что требует понимания смысла.
"""

from __future__ import annotations

import re

from .data import Conversation

GREETING = {
    "RU": re.compile(r"здравствуй|добр(ый|ое|ого)\s+(день|вечер|утро|времени)|приветствую|\bпривет", re.I),
    "KZ": re.compile(r"сәлеметсіз|сәлем|қайырлы\s+(күн|таң|кеш)|армысыз", re.I),
    "EN": re.compile(r"\bhello\b|\bhi\b|\bhey\b|good\s+(morning|afternoon|evening)|greetings", re.I),
}
NOT_NAMES = {"Вам", "Вас", "Вы", "Ваш", "Ваша", "Вашего", "Сейчас", "Понял", "Поняла", "Уточню", "Проверю", "Не"}
INTRO = [
    re.compile(r"меня\s+зовут|менің\s+атым|my\s+name\s+is|\[Имя\s+скрыто\]", re.I),
    re.compile(r"(?:^|[\s,.!?])[Яя]\s+(?P<n>[А-ЯЁ][а-яё]+)"),                       # «я Руслан», «Я Лера из поддержки»
    re.compile(r"(?:^|[\s,.!?])[Мм]ен\s+(?P<n>[А-ЯЁӘӨҮҰҚҒҢІҺ][а-яёәөүұқғңіһ]+)"),     # «мен Айбекпін», «Мен Нұрланмын»
    re.compile(r"\bI(?:'m|\s+am)\s+(?P<n>[A-Z][a-z]+)|\bthis\s+is\s+(?P<m>[A-Z][a-z]+)"),  # «I'm Tyler»
]

GENERIC_QUESTION = re.compile(
    r"чем\s+(я\s+)?(могу|можем)\s+(вам\s+)?помочь|с\s+чем\s+помочь|могу\s+(ещё|еще)\s+(чем|что)|"
    r"остались\s+(ещё\s+|еще\s+|дополнительные\s+)?вопрос|(ещё|еще)\s+вопрос|вы\s+на\s+линии|"
    r"how\s+can\s+i\s+help|anything\s+else|any\s+other\s+question|"
    r"қалай\s+көмектесе|тағы\s+сұра|көмек\s+керек\s+пе|"
    r"личный\s+номер|жеке\s+нөмір|personal\s+number", re.I)  # просьба дать личный номер — не уточнение потребности
INFO_REQUEST = re.compile(
    r"(уточните|укажите|подскажите|напишите)[^.?!]{0,40}(как|какой|какая|какие|каком|в\s+какой|где|когда|что|"
    r"название|тикер|номер\s+(поручения|приказа|счёта|телефона)|валют|дат)", re.I)


def contact(conv: Conversation) -> dict | None:
    """01: приветствие на языке клиента + имя в первой реплике оператора."""
    first = next((t for t in conv.turns if t.speaker == "OPERATOR"), None)
    if first is None:
        return None
    text = first.text
    patterns = [GREETING[conv.language]] if conv.language in GREETING else list(GREETING.values())
    greeted = any(p.search(text) for p in patterns)
    other_lang = not greeted and any(p.search(text) for p in GREETING.values())
    named = False
    for p in INTRO:
        for m in p.finditer(text):
            name = m.groupdict().get("n") or m.groupdict().get("m")
            if p is INTRO[0] or (name and name not in NOT_NAMES):
                named = True
    if greeted and named:
        why = "Оператор поздоровался на языке клиента и представился в первой реплике."
    elif named:
        why = ("Оператор представился, но поздоровался не на языке клиента." if other_lang
               else "Оператор представился, но не поздоровался в первой реплике.")
    elif greeted:
        why = "Оператор поздоровался, но не представился (не назвал имя) в первой реплике."
    else:
        why = "В первой реплике оператора нет ни приветствия, ни представления."
    return {"value": greeted and named, "rationale": why, "evidence_turns": [first.number]}


def clarifying_questions(conv: Conversation) -> list[int]:
    """Номера реплик, где оператор задал клиенту вопрос по существу (не «Чем могу помочь?»)."""
    found = []
    for t in conv.turns:
        if t.speaker != "OPERATOR":
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", t.text):
            if GENERIC_QUESTION.search(sentence):
                continue
            if sentence.rstrip().endswith("?") or INFO_REQUEST.search(sentence):
                found.append(t.number)
                break
    return found


def gate_discovery(item: dict, conv: Conversation) -> dict:
    """02: модель выбирает только между вариантами, совместимыми с фактом «был ли вопрос»."""
    asked = clarifying_questions(conv)
    value = item["value"]
    if asked and value is None:
        return {"value": True, "evidence_turns": asked,
                "rationale": f"Оператор задал уточняющий вопрос до ответа (реплика {asked[0]})."}
    if not asked and value is True:
        return {"value": None, "evidence_turns": [1],
                "rationale": "Оператор ничего не уточнял, а данных для ответа хватало с самого начала."}
    return item
