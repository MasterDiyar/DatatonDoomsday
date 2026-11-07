"""Второй, короткий вызов модели: факты с дословными цитатами.

Модель 7B в одном большом промпте ошибалась в простых фактах («не представился» при «я Руслан»). Поэтому
после основной оценки модель отдельно отвечает на узкие вопросы и цитирует текст. Код не ищет ключевые слова:
он только проверяет, что цитата действительно есть в нужной реплике (защита от выдуманных доказательств),
и собирает ответ по формуле чек-листа:
- 01 Контакт = приветствие в первой реплике на языке клиента И имя оператора в первой реплике;
- 02 Потребность: были вопросы о новом -> Да; спрашивал только уже известное -> Нет; вопросов не было ->
  решение основного промпта (Нет или N/A);
- 07 Завершение = (итог или предложение помощи) И прощание; оборвалось до прощания -> N/A.
Если вызов не удался, остаются ответы основного промпта. Это один дополнительный вызов на диалог.
"""

from __future__ import annotations

import json
import re

from . import prompt
from .data import Conversation
from .model import ModelConfig, ModelError, chat

PROMPT = (
    "Прочитай диалог службы поддержки и ответь на вопросы о ФАКТАХ. Ничего не оценивай. Цитаты — только "
    "дословные фрагменты реплик; если факта нет — пустая строка или пустой список.\n"
    "1. greeting_quote: приветствие в ПЕРВОЙ реплике оператора (OPERATOR). Приветствие бота не считается.\n"
    "2. greeting_in_client_language: true, если оператор поздоровался на том же языке, на котором пишет или говорит "
    "клиент.\n"
    "3. name_quote: фрагмент ПЕРВОЙ реплики оператора, где он называет своё имя. Название компании — не имя. "
    "«[Имя скрыто]» в звонках — это имя.\n"
    "4. operator_questions: вопросы, которыми оператор выяснял недостающие сведения о запросе клиента (что именно "
    "нужно, где он смотрит, какой документ, тикер, валюта, что на экране). НЕ включай общие вопросы вроде «Чем "
    "могу помочь?», «Остались вопросы?», просьбы дать личный номер и риторические вопросы. Для каждого: turn — "
    "номер реплики, quote — цитата, already_known — true, если клиент уже сообщил это раньше (в том числе боту).\n"
    "5. summary_or_offer_quote: в последних репликах оператора — итог разговора, договорённость о следующем шаге "
    "или предложение дальнейшей помощи. Слова «спасибо», «пожалуйста», «рад помочь», «отлично» и само прощание — "
    "это НЕ итог.\n"
    "6. goodbye_quote: прощание оператора в конце диалога.\n"
    "7. ended_before_operator_could_close: true, если клиент ушёл, связь прервалась или клиент завершил звонок "
    "раньше, чем оператор мог попрощаться. false, если оператор сам закрыл чат или просто не попрощался."
)
SCHEMA = {
    "type": "object",
    "properties": {
        "greeting_quote": {"type": "string"},
        "greeting_in_client_language": {"type": "boolean"},
        "name_quote": {"type": "string"},
        "operator_questions": {"type": "array", "items": {"type": "object", "properties": {
            "turn": {"type": "integer"}, "quote": {"type": "string"}, "already_known": {"type": "boolean"}},
            "required": ["turn", "quote", "already_known"]}},
        "summary_or_offer_quote": {"type": "string"},
        "goodbye_quote": {"type": "string"},
        "ended_before_operator_could_close": {"type": "boolean"},
    },
    "required": ["greeting_quote", "greeting_in_client_language", "name_quote", "operator_questions",
                 "summary_or_offer_quote", "goodbye_quote", "ended_before_operator_could_close"],
}


def _norm(s: str) -> str:
    return re.sub(r"\W+", " ", str(s or "").lower()).strip()


def _found(quote: str, text: str) -> bool:
    q = _norm(quote)
    return bool(q) and q in _norm(text)


def ask(conv: Conversation, cfg: ModelConfig) -> dict | None:
    """Один короткий вызов модели. None, если ответ не получен (тогда остаётся основная оценка)."""
    try:
        text, _ = chat(cfg, [{"role": "system", "content": PROMPT}, {"role": "user", "content": prompt.user_prompt(conv)}],
                       SCHEMA if cfg.backend == "ollama" else None, temperature=0.0)
        data = json.loads(text)
        # Неполный ответ не трактуется как «фактов нет»: тогда остаются ответы основного промпта.
        if not isinstance(data, dict) or any(k not in data for k in SCHEMA["required"]):
            return None
        return data
    except ModelError as e:
        if e.fatal:
            raise
        return None
    except json.JSONDecodeError:
        return None


def apply(final: dict, facts: dict, conv: Conversation) -> dict:
    """Пересобрать 01, 02 и 07 из проверенных фактов. Возвращает изменённую копию `final`."""
    out = dict(final)
    ops = [t for t in conv.turns if t.speaker == "OPERATOR"]
    if not ops:
        return out
    first, by_number = ops[0], {t.number: t for t in ops}

    # 01 Установление контакта
    greet = str(facts.get("greeting_quote") or "")
    name = str(facts.get("name_quote") or "")
    greeted = _found(greet, first.text)
    in_lang = bool(facts.get("greeting_in_client_language"))
    named = _found(name, first.text)
    if greeted and in_lang and named:
        why = f"Оператор поздоровался («{greet.strip()}») и представился («{name.strip()}») в первой реплике."
    elif not greeted:
        why = "В первой реплике оператора нет приветствия."
    elif not in_lang:
        why = f"Оператор поздоровался («{greet.strip()}») не на языке клиента."
    else:
        why = f"Оператор поздоровался («{greet.strip()}»), но не назвал своё имя в первой реплике."
    out["contact"] = {"value": greeted and in_lang and named, "rationale": why, "evidence_turns": [first.number]}

    # 02 Выявление потребности: только вопросы, которые действительно есть в репликах оператора
    questions = []
    for q in facts.get("operator_questions") or []:
        if not isinstance(q, dict):
            continue
        turn = by_number.get(q.get("turn"))
        if turn is None or not _found(q.get("quote"), turn.text):
            turn = next((t for t in ops if _found(q.get("quote"), t.text)), None)
        if turn is not None:
            questions.append((turn.number, str(q.get("quote")).strip(), bool(q.get("already_known"))))
    if questions:
        new = [q for q in questions if not q[2]]
        if new:
            out["discovery"] = {"value": True, "evidence_turns": sorted({q[0] for q in new}),
                                "rationale": f"До ответа оператор уточнил недостающее: «{new[0][1]}»."}
        else:
            out["discovery"] = {"value": False, "evidence_turns": sorted({q[0] for q in questions}),
                                "rationale": f"Оператор переспросил то, что клиент уже сообщил: «{questions[0][1]}»."}
    elif final["discovery"]["value"] is True:
        out["discovery"] = {"value": None, "evidence_turns": [1],
                            "rationale": "Оператор ничего не уточнял: данных для ответа хватало с самого начала."}

    # 07 Завершение: цитаты ищутся в последних репликах оператора
    tail = ops[-3:]
    summary = str(facts.get("summary_or_offer_quote") or "")
    goodbye = str(facts.get("goodbye_quote") or "")
    goodbye_ok = any(_found(goodbye, t.text) for t in tail)
    rest = _norm(summary).replace(_norm(goodbye), " ") if goodbye_ok else _norm(summary)
    summary_ok = any(_found(summary, t.text) for t in tail) and len(rest.split()) >= 3
    ended = bool(facts.get("ended_before_operator_could_close"))
    tail_turns = [t.number for t in tail[-2:]]
    if summary_ok and goodbye_ok:
        out["closure"] = {"value": True, "evidence_turns": tail_turns,
                          "rationale": f"Оператор подвёл итог или предложил помощь («{summary.strip()}») и попрощался."}
    elif ended and not goodbye_ok:
        last = conv.turns[-1].number
        out["closure"] = {"value": None, "evidence_turns": [last],
                          "rationale": "Диалог оборвался раньше, чем оператор мог его завершить."}
    elif goodbye_ok:
        out["closure"] = {"value": False, "evidence_turns": tail_turns,
                          "rationale": f"Оператор только попрощался («{goodbye.strip()}»), без итога и без предложения помощи."}
    else:
        out["closure"] = {"value": False, "evidence_turns": tail_turns, "rationale": "Оператор не попрощался с клиентом."}
    return out
