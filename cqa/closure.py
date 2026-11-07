"""Отдельная проверка критерия 07 «Завершение».

На DEV модель в общем промпте почти всегда считала простое «Пожалуйста. Хорошего дня!» итогом разговора.
Поэтому завершение проверяется отдельно и узко:
- обрыв связи / уход клиента определяется по системным сообщениям (правила);
- модель получает только последние реплики оператора и должна ПРОЦИТИРОВАТЬ итог, предложение помощи и
  прощание; код проверяет, что цитаты действительно есть в репликах и что «итог» — не просто «спасибо».
Если проверка не удалась, остаётся ответ из основного промпта.
"""

from __future__ import annotations

import json
import re

from .data import Conversation
from .model import ModelConfig, ModelError, chat

CLOSED_BY_OPERATOR = re.compile(r"оператор\s+закрыл|operator\s+closed|agent\s+closed|чатты\s+жапты", re.I)
CUSTOMER_GONE = re.compile(
    r"покинул|прерван|завершил\s+разговор|завершил\s+звонок|отключил|клиент\s+вышел|"
    r"disconnect|connection\s+(to\s+\w+\s+)?lost|lost\s+connection|left\s+the\s+chat|ended\s+the\s+call|"
    r"үзілді|шығып\s+кетті", re.I)
GOODBYE = re.compile(
    r"до\s+свидания|всего\s+доброго|всего\s+хорошего|хорошего\s+(дня|вечера|дня!)|до\s+встречи|"
    r"сау\s+болыңыз|күніңіз\s+сәтті|көріскенше|goodbye|\bbye\b|have\s+a\s+(good|nice|great)|take\s+care", re.I)
OFFER = re.compile(
    r"чем[-\s]?(то|нибудь)?\s*(ещё|еще)?\s*(могу|можем)?\s*помочь|(ещё|еще)\s+вопрос|остались\s+(ещё\s+)?вопрос|"
    r"напишите\s+(нам|ещё|еще|в\s+этот\s+чат)|обращайтесь|если\s+вопрос\s+остан|"
    r"anything\s+else|any\s+other\s+question|feel\s+free|let\s+us\s+know|reach\s+out\s+again|"
    r"тағы\s+(сұра|көмек)|сұрағыңыз\s+болса|(қайта|тағы)\s+жазыңыз", re.I)
# Слова, которые сами по себе не делают итога (вежливость и прощание).
FILLER = re.compile(
    r"пожалуйста|спасибо|благодарю|рад[аы]?\s+помочь|отлично|хорошо|ок|понял[а]?|до\s+свидания|всего\s+доброго|"
    r"всего\s+хорошего|хорошего|дня|вечера|за\s+обращение|за\s+ожидание|рақмет|оқасы\s+жоқ|көмегім\s+тигеніне\s+"
    r"қуаныштымын|сау\s+болыңыз|жақсы|күніңіз\s+сәтті\s+өтсін|you'?re\s+welcome|thanks?(\s+you)?|for\s+reaching\s+out|"
    r"for\s+contacting\s+us|happy\s+to\s+help|glad|great|goodbye|bye|have\s+a\s+good\s+day|take\s+care|of\s+course",
    re.I)

PROMPT = (
    "Ниже последние реплики оператора чата поддержки. Найди в них ДОСЛОВНЫЕ цитаты:\n"
    "- summary: итог разговора или договорённость о следующем шаге (например «вопрос передан, ответ придёт в этот "
    "чат», «перед переводом проверьте комиссию банка»). Просто ответ на вопрос клиента, «спасибо», «пожалуйста», "
    "«рад помочь», «отлично» — это НЕ итог.\n"
    "- offer: предложение дальнейшей помощи («могу ещё чем-то помочь?», «если вопрос останется, напишите нам»).\n"
    "- goodbye: прощание («до свидания», «хорошего дня», «сау болыңыз», «goodbye»).\n"
    "Если чего-то нет — пустая строка. Не придумывай и не перефразируй: только точные фрагменты текста.\n"
    'Ответ — только JSON: {"summary": "", "offer": "", "goodbye": ""}'
)
SCHEMA = {"type": "object", "properties": {"summary": {"type": "string"}, "offer": {"type": "string"},
                                           "goodbye": {"type": "string"}},
          "required": ["summary", "offer", "goodbye"]}


def _norm(s: str) -> str:
    return re.sub(r"[^\w]+", " ", s.lower()).strip()


def _quote_in(quote: str, texts: list[str]) -> bool:
    q = _norm(quote)
    return bool(q) and any(q in _norm(t) for t in texts)


def _substantive(quote: str) -> bool:
    """Итог — это больше, чем вежливые слова: после их удаления должно остаться хотя бы 3 слова."""
    rest = FILLER.sub(" ", quote.lower())
    return len(_norm(rest).split()) >= 3


def check_closure(conv: Conversation, cfg: ModelConfig) -> dict | None:
    turns = conv.turns
    gone_at = next((t.number for t in turns if t.speaker == "SYSTEM" and CUSTOMER_GONE.search(t.text)
                    and not CLOSED_BY_OPERATOR.search(t.text)), None)
    # Реплики оператора после ухода клиента (оператор говорит в пустоту) не учитываются.
    ops = [t for t in turns if t.speaker == "OPERATOR" and (gone_at is None or t.number < gone_at)]
    if not ops:
        return None
    tail = ops[-2:]
    texts = [t.text for t in tail]
    tail_numbers = [t.number for t in tail]

    summary = offer = goodbye = ""
    try:
        listing = "\n".join(f"[{t.number}] {t.text}" for t in tail)
        text, _ = chat(cfg, [{"role": "system", "content": PROMPT}, {"role": "user", "content": listing}],
                       SCHEMA if cfg.backend == "ollama" else None, temperature=0.0)
        data = json.loads(text)
        summary, offer, goodbye = (str(data.get(k) or "") for k in ("summary", "offer", "goodbye"))
    except ModelError as e:
        if e.fatal:
            raise
    except (json.JSONDecodeError, AttributeError):
        pass

    has_goodbye = bool(GOODBYE.search(" ".join(texts))) or (_quote_in(goodbye, texts) and bool(GOODBYE.search(goodbye)))
    has_offer = bool(OFFER.search(" ".join(texts))) or (_quote_in(offer, texts) and _substantive(offer))
    has_summary = _quote_in(summary, texts) and _substantive(summary)

    if has_goodbye and (has_offer or has_summary):
        what = "предложение помощи" if has_offer else "итог"
        quote = (offer if has_offer and offer else summary) or "см. реплику"
        return {"value": True, "evidence_turns": tail_numbers,
                "rationale": f"Оператор дал {what} («{quote.strip()[:120]}») и попрощался."}
    if gone_at is not None and not has_goodbye:
        return {"value": None, "evidence_turns": [gone_at],
                "rationale": f"Диалог оборвался до завершения: системное сообщение [{gone_at}] о том, что клиент ушёл "
                             "или связь прервалась."}
    if has_goodbye:
        return {"value": False, "evidence_turns": tail_numbers,
                "rationale": "Оператор только попрощался/поблагодарил, без итога разговора и без предложения "
                             "дальнейшей помощи."}
    closed = next((t.number for t in turns if t.speaker == "SYSTEM" and CLOSED_BY_OPERATOR.search(t.text)), None)
    return {"value": False, "evidence_turns": tail_numbers + ([closed] if closed else []),
            "rationale": "Оператор не попрощался" + (" и закрыл чат." if closed else ".")}
