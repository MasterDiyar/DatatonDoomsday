"""CQA evaluator command line. Run `python run.py -h` or `python run.py <command> -h`.

  python run.py check                          test the model connection
  python run.py fetch URL --out datasets/dev.json
  python run.py evaluate datasets/dev.json --split dev        -> results/dev/results.json
  python run.py evaluate datasets/prod.json --split prod      -> results/prod/results.json (exactly 10 checked)
  python run.py validate results/prod/results.json --data datasets/prod.json --expect 10
  python run.py serve                          review page at http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.request
from pathlib import Path

if sys.version_info < (3, 10):
    sys.exit("Python 3.10 or newer is required.")
for stream in (sys.stdout, sys.stderr):  # Windows consoles: print Cyrillic/Kazakh without crashing
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from cqa import config, labels, sft, synth  # noqa: E402
from cqa.data import load_dataset  # noqa: E402
from cqa.model import ModelError, _opener, chat  # noqa: E402
from cqa.results import read_json, validate_file, write_json  # noqa: E402
from cqa.runner import run_dataset  # noqa: E402

ROOT = config.ROOT


def add_model_args(p):
    g = p.add_argument_group("model (defaults from config.json)")
    g.add_argument("--model", help="model id, e.g. qwen2.5:7b-instruct or cqa-qwen7b")
    g.add_argument("--backend", choices=["ollama", "openai"])
    g.add_argument("--base-url", help="e.g. http://127.0.0.1:11434")
    g.add_argument("--num-ctx", type=int, help="context window for Ollama (default 16384)")
    g.add_argument("--temperature", type=float)
    g.add_argument("--timeout", type=int, help="seconds per model call")


def settings_from(args) -> tuple[dict, object]:
    s = config.load()
    for key in ("model", "backend", "base_url", "num_ctx", "temperature", "timeout"):
        v = getattr(args, key, None)
        if v is not None:
            s[key] = v
    for key in ("team", "model_name", "mode"):
        v = getattr(args, key, None)
        if v:
            s[{"team": "team_name", "model_name": "submission_model_name", "mode": "submission_mode"}[key]] = v
    return s, config.model_config(s)


def read_ids(path: str | None) -> set[str] | None:
    if not path:
        return None
    return {line.strip() for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()}


def cmd_check(args) -> int:
    s, m = settings_from(args)
    print(f"Python {sys.version.split()[0]}; config.json {'found' if config.CONFIG_FILE.exists() else 'not found (defaults)'}")
    print(f"backend={m.backend} base_url={m.base_url} model={m.model} num_ctx={m.num_ctx}")
    if m.backend == "ollama":
        base = m.base_url.rstrip("/").removesuffix("/v1")
        try:
            with _opener(base).open(f"{base}/api/tags", timeout=10) as r:
                names = [x["name"] for x in json.loads(r.read().decode("utf-8")).get("models", [])]
        except Exception as e:  # noqa: BLE001
            print(f"FAIL: cannot reach Ollama at {base}: {e}\n -> start it: ollama serve  (or the Ollama app)")
            return 1
        print(f"Ollama is running; installed models: {', '.join(names) or 'none'}")
        if m.model not in names and f"{m.model}:latest" not in names:
            print(f"FAIL: model {m.model!r} is not installed -> ollama pull {m.model}")
            return 1
    t0 = time.time()
    try:
        text, usage = chat(m, [{"role": "user", "content": 'Reply with JSON {"ok": true}'}],
                           {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
                           if m.backend == "ollama" else None)
    except ModelError as e:
        print(f"FAIL: {e}")
        return 1
    print(f"OK: model answered in {time.time() - t0:.1f}s: {text.strip()[:80]}")
    return 0


def cmd_fetch(args) -> int:
    headers = dict(h.split(":", 1) for h in args.header or [])
    req = urllib.request.Request(args.url, headers={k.strip(): v.strip() for k, v in headers.items()})
    with urllib.request.urlopen(req, timeout=120) as r:
        raw = r.read()
    payload = json.loads(raw.decode("utf-8-sig"))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(raw)
    convs = load_dataset(out)
    bad = [c for c in convs if c.error]
    print(f"Saved {out}: {len(convs)} conversations ({len(bad)} invalid)")
    for c in bad:
        print(f"  invalid {c.conversation_id}: {c.error}")
    return 0 if payload is not None else 1


def cmd_evaluate(args) -> int:
    s, m = settings_from(args)
    out = args.out or (f"results/{args.split}/results.json" if args.split else None)
    if not out:
        print("Give --split dev|prod or --out PATH")
        return 2
    expect = args.expect if args.expect is not None else (10 if args.split == "prod" else None)
    return run_dataset(args.data, out, s, m, ids=read_ids(args.ids), votes=args.votes, force=args.force,
                       expect_count=expect)


def cmd_validate(args) -> int:
    data = read_json(args.results)
    ids = [c.conversation_id for c in load_dataset(args.data)] if args.data else None
    problems = validate_file(data, expect_ids=ids, expect_count=args.expect)
    errors = [r for r in data.get("results", []) if isinstance(r, dict) and r.get("status") == "error"]
    for p in problems:
        print("INVALID:", p)
    print(f"{'OK' if not problems else 'FAILED'}: {len(data.get('results', []))} entries, {len(errors)} with status error")
    return 1 if problems else 0


def cmd_labels_from_results(args) -> int:
    convs = {c.conversation_id: len(c.turns) for c in load_dataset(args.data) if not c.error}
    scenarios = read_json(args.scenarios) if args.scenarios else None
    added, skipped, contradicted = labels.from_results(read_json(args.results), args.data, convs, scenarios)
    print(f"labels/{Path(args.data).stem}.json: {added} added (unreviewed), {skipped} skipped, "
          f"{contradicted} dropped (contradict scenario)")
    return 0


def cmd_build_sft(args) -> int:
    pairs = [tuple(p) for p in args.pair]
    stats = sft.build(pairs, args.out, holdout=args.holdout, include_unreviewed=args.include_unreviewed)
    print(json.dumps(stats, indent=2))
    print(f"-> {args.out}/train.jsonl, eval.jsonl, holdout-ids.txt")
    return 0 if stats["train"] else 1


def cmd_compare(args) -> int:
    report = sft.compare(args.results, args.labels, read_ids(args.ids))
    print(f"conversations compared: {report['conversations']}  (no prediction: {len(report['missing_predictions'])})")
    print(f"exact agreement: {report['exact_accuracy']}   final score MAE: {report['score_mae']}")
    for code, acc in report["per_criterion"].items():
        print(f"  {code:15s} {acc}")
    return 0


def cmd_synth(args) -> int:
    s, m = settings_from(args)
    out, sc_out = Path(args.out), Path(args.out).with_name(Path(args.out).stem + "-scenarios.json")
    data = read_json(out) if out.exists() else {"records": []}
    scenarios = read_json(sc_out) if sc_out.exists() else {}
    have = {r["conversation_id"] for r in data["records"]}
    rng = random.Random(args.seed)
    plan = [synth.scenario(i, rng, args.prefix) for i in range(args.count)]
    for sc in plan:
        if sc["id"] in have:
            continue
        try:
            data["records"].append(synth.generate(sc, m))
            scenarios[sc["id"]] = sc
            print(f"{sc['id']} {sc['language']} {','.join(sc['behaviours'])}")
        except (ModelError, ValueError, KeyError, TypeError) as e:
            print(f"{sc['id']}: failed ({e})")
            if isinstance(e, ModelError) and e.fatal:
                break
        write_json(out, data)
        write_json(sc_out, scenarios)
    print(f"{out}: {len(data['records'])} chats; scenarios in {sc_out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("check", help="test the model connection")
    add_model_args(p)
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("fetch", help="download a dataset from the organisers' API")
    p.add_argument("url")
    p.add_argument("--out", required=True)
    p.add_argument("--header", action="append", help='extra HTTP header, e.g. "Authorization: Bearer X"')
    p.set_defaults(fn=cmd_fetch)

    p = sub.add_parser("evaluate", help="score every chat of a dataset into results.json")
    p.add_argument("data")
    p.add_argument("--split", choices=["dev", "prod"], help="writes results/<split>/results.json")
    p.add_argument("--out", help="output path instead of --split")
    p.add_argument("--ids", help="file with conversation_ids to (re)evaluate, one per line")
    p.add_argument("--votes", type=int, default=1, help="model runs per chat, majority vote (default 1)")
    p.add_argument("--force", action="store_true", help="re-evaluate chats that are already ok")
    p.add_argument("--expect", type=int, help="required number of entries (prod: 10)")
    p.add_argument("--team", help="team_name in results.json")
    p.add_argument("--model-name", help="model.name in results.json (default: the model id)")
    p.add_argument("--mode", choices=["local", "external_api", "hybrid"])
    add_model_args(p)
    p.set_defaults(fn=cmd_evaluate)

    p = sub.add_parser("validate", help="check a results.json against format 1.0")
    p.add_argument("results")
    p.add_argument("--data", help="dataset file: every ID must be present exactly once")
    p.add_argument("--expect", type=int)
    p.set_defaults(fn=cmd_validate)

    p = sub.add_parser("serve", help="local review page")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(fn=lambda a: __import__("cqa.server", fromlist=["serve"]).serve(a.port) or 0)

    p = sub.add_parser("labels-from-results", help="import model answers as unreviewed labels")
    p.add_argument("results")
    p.add_argument("--data", required=True)
    p.add_argument("--scenarios", help="synthetic scenarios file: drop contradicting answers")
    p.set_defaults(fn=cmd_labels_from_results)

    p = sub.add_parser("build-sft", help="fine-tuning data from datasets + labels")
    p.add_argument("--pair", nargs=2, action="append", required=True, metavar=("DATASET", "LABELS"))
    p.add_argument("--out", default="training/data")
    p.add_argument("--holdout", type=float, default=0.25, help="share of real chats kept for testing")
    p.add_argument("--include-unreviewed", action="store_true", help="also use unreviewed (model) labels")
    p.set_defaults(fn=cmd_build_sft)

    p = sub.add_parser("compare", help="agreement of results.json with our labels")
    p.add_argument("results")
    p.add_argument("labels")
    p.add_argument("--ids", help="only these IDs, e.g. training/data/holdout-ids.txt")
    p.set_defaults(fn=cmd_compare)

    p = sub.add_parser("synth", help="generate synthetic chats for training")
    p.add_argument("--count", type=int, default=300)
    p.add_argument("--out", default="datasets/synth.json")
    p.add_argument("--prefix", default="synth-")
    p.add_argument("--seed", type=int, default=1)
    add_model_args(p)
    p.set_defaults(fn=cmd_synth)

    args = ap.parse_args(argv)
    try:
        return args.fn(args)
    except FileNotFoundError as e:
        print(f"File not found: {e.filename or e}")
        return 2
    except (ValueError, json.JSONDecodeError) as e:
        print(f"Error: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
