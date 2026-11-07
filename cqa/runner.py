"""Evaluate a whole dataset into results/<split>/results.json.

- Resumable: entries already "ok" in the output file are kept; errors and missing chats are (re)run.
- Saved after every chat (atomic write), so an interrupted run loses nothing.
- Every dataset ID gets exactly one entry, in dataset order (errors included).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from . import __version__
from .config import ROOT
from .data import load_dataset
from .evaluate import error_entry, evaluate
from .model import ModelConfig, ModelError
from .results import build_file, read_json, validate_file, write_json


def _live(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()


def run_dataset(data_path: str, out_path: str, settings: dict, mcfg: ModelConfig, ids: set[str] | None = None,
                votes: int = 1, force: bool = False, expect_count: int | None = None, log=print,
                live=None) -> int:
    if live is None:
        live = _live if log is print else (lambda text: None)
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
    log(f"cqa {__version__}; fact check: {'on' if mcfg.fact_check else 'OFF'}; rationale language: {mcfg.rationale_language}")
    log(f"{len(convs)} chats in {data_path}; {len(todo)} to evaluate with {mcfg.model} ({mcfg.backend} at {mcfg.base_url})")
    if not todo and convs:
        log("Nothing to do: every chat already has status ok in this file. Add --force to evaluate again.")
    save()
    started, done, fatal = time.time(), 0, None
    facts_ok = facts_failed = 0
    if todo and mcfg.backend == "ollama":
        log("The first chat also loads the model into memory (can take a minute). "
            "In another window, `ollama ps` shows whether it runs on GPU.")
    for i, conv in enumerate(todo, start=1):
        t0 = time.time()
        state = {"last": 0.0}

        def progress(n, conv=conv, i=i):
            # One updating console line: proves the model is working and shows its speed.
            now = time.time()
            if now - state["last"] >= 1 or n == 1:
                state["last"] = now
                live(f"\r[{i}/{len(todo)}] {conv.conversation_id}: generating... {n} tokens, {now - t0:.0f}s   ")

        size = sum(len(t.text) for t in conv.turns)
        live(f"[{i}/{len(todo)}] {conv.conversation_id}: {len(conv.turns)} turns, {size} chars - waiting for the model...")
        try:
            entry, info = evaluate(conv, mcfg, votes=votes, log=log, progress=progress)
        except ModelError as e:  # fatal: wrong URL / model name / key - stop instead of failing every chat
            live("\n")
            fatal = str(e)
            break
        except KeyboardInterrupt:
            live("\n")
            save()
            log(f"Interrupted (Ctrl+C). {done} chats finished this run are saved in {out}. "
                "Run the same command again to continue from where it stopped.")
            return 130
        entries[conv.conversation_id] = entry
        if "fact_check" in info:
            facts_failed += "error" in info["fact_check"]
            facts_ok += "error" not in info["fact_check"]
        with open(details_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"conversation_id": conv.conversation_id, "model": mcfg.model, **info},
                               ensure_ascii=False) + "\n")
        save()
        done += 1
        avg = (time.time() - started) / done
        shown = entry["score"]["final_score"] if entry["status"] == "ok" else "ERROR " + entry["error"]
        flag = " CRITICAL" if entry["status"] == "ok" and entry["score"]["critical_flag"] else ""
        live("\r" + " " * 70 + "\r")
        log(f"[{i}/{len(todo)}] {conv.conversation_id}: {shown}{flag}  ({time.time() - t0:.0f}s, "
            f"ETA {avg * (len(todo) - i) / 60:.1f} min)")
    if mcfg.fact_check and (facts_ok or facts_failed):
        log(f"\nFact check applied to {facts_ok} of {facts_ok + facts_failed} chats"
            + (f"; failed for {facts_failed} (main prompt answers kept, reasons in {details_path.name})" if facts_failed else "."))
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
