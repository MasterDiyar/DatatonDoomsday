"""Evaluate a whole dataset into results/<split>/results.json.

- Resumable: entries already "ok" in the output file are kept; errors and missing chats are (re)run.
- Saved after every chat (atomic write), so an interrupted run loses nothing.
- Every dataset ID gets exactly one entry, in dataset order (errors included).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .config import ROOT
from .data import load_dataset
from .evaluate import error_entry, evaluate
from .model import ModelConfig, ModelError
from .results import build_file, read_json, validate_file, write_json


def run_dataset(data_path: str, out_path: str, settings: dict, mcfg: ModelConfig, ids: set[str] | None = None,
                votes: int = 1, force: bool = False, expect_count: int | None = None, log=print) -> int:
    convs = load_dataset(data_path)
    out = Path(out_path)
    previous = {}
    if out.exists() and not force:
        try:
            previous = {r["conversation_id"]: r for r in read_json(out).get("results", [])}
        except (ValueError, KeyError, TypeError):
            log(f"warning: {out} is unreadable, starting over")
    details_path = ROOT / "runs" / f"{out.parent.name or 'results'}-details.jsonl"
    details_path.parent.mkdir(exist_ok=True)
    entries = {c.conversation_id: previous.get(c.conversation_id) or error_entry(c.conversation_id, "not evaluated yet")
               for c in convs}
    model_name = settings["submission_model_name"] or mcfg.model

    def save():
        write_json(out, build_file(settings["team_name"], model_name, settings["submission_mode"],
                                   [entries[c.conversation_id] for c in convs]))

    todo = [c for c in convs if (ids is None or c.conversation_id in ids)
            and (force or entries[c.conversation_id].get("status") != "ok")]
    log(f"{len(convs)} chats in {data_path}; {len(todo)} to evaluate with {mcfg.model} ({mcfg.backend} at {mcfg.base_url})")
    save()
    started, done, fatal = time.time(), 0, None
    for i, conv in enumerate(todo, start=1):
        t0 = time.time()
        try:
            entry, info = evaluate(conv, mcfg, votes=votes, log=log)
        except ModelError as e:  # fatal: wrong URL / model name / key - stop instead of failing every chat
            fatal = str(e)
            break
        entries[conv.conversation_id] = entry
        with open(details_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"conversation_id": conv.conversation_id, "model": mcfg.model, **info},
                               ensure_ascii=False) + "\n")
        save()
        done += 1
        avg = (time.time() - started) / done
        shown = entry["score"]["final_score"] if entry["status"] == "ok" else "ERROR " + entry["error"]
        flag = " CRITICAL" if entry["status"] == "ok" and entry["score"]["critical_flag"] else ""
        log(f"[{i}/{len(todo)}] {conv.conversation_id}: {shown}{flag}  ({time.time() - t0:.0f}s, "
            f"ETA {avg * (len(todo) - i) / 60:.1f} min)")
    if fatal:
        log(f"\nSTOPPED: {fatal}")
        log("Fix the problem (python run.py check) and run the same command again; finished chats are kept.")
    save()
    data = read_json(out)
    errors = [r for r in data["results"] if r["status"] == "error"]
    problems = validate_file(data, expect_ids=[c.conversation_id for c in convs], expect_count=expect_count)
    log(f"\nWrote {out} - {len(data['results'])} entries, {len(errors)} errors.")
    for r in errors:
        log(f"  error {r['conversation_id']}: {r['error']}")
    if errors:
        log("Run the same command again to retry only the failed chats.")
    for p in problems:
        log(f"  INVALID: {p}")
    return 1 if (fatal or errors or problems) else 0
