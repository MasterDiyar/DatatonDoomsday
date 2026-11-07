"""Fine-tuning data and accuracy measurement."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from . import prompt
from .checklist import BY_CODE, CODES
from .data import load_dataset
from .labels import load as load_labels
from .results import read_json


def in_holdout(cid: str, fraction: float) -> bool:
    """Deterministic split: the same ID always lands in the same split."""
    h = int(hashlib.sha256(cid.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return h < fraction


def build(pairs: list[tuple[str, str]], out_dir: str, holdout: float = 0.25, include_unreviewed: bool = False,
          synthetic_prefix: str = "synth-", max_tokens: int = 14000) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train, evals, stats = [], [], defaultdict(int)
    for data_path, labels_path in pairs:
        convs = {c.conversation_id: c for c in load_dataset(data_path)}
        labels = load_labels(Path(labels_path))["labels"]
        for cid, label in labels.items():
            conv = convs.get(cid)
            if conv is None or conv.error:
                stats["no_transcript"] += 1
                continue
            if not label.get("reviewed") and not include_unreviewed:
                stats["unreviewed_skipped"] += 1
                continue
            crit = label["criteria"]
            for code in CODES:  # empty rationale/evidence would teach the model to omit them
                if not crit[code]["rationale"]:
                    crit[code]["rationale"] = f"{BY_CODE[code].title}: see evidence turns."
                if not crit[code]["evidence_turns"]:
                    crit[code]["evidence_turns"] = (conv.operator_turns or [1]) if crit[code]["value"] is not None else [1]
            msgs = prompt.messages(conv) + [{"role": "assistant", "content": prompt.target_json(crit)}]
            tokens = prompt.approx_tokens("".join(m["content"] for m in msgs))
            if tokens > max_tokens:
                stats["too_long"] += 1
                continue
            row = {"conversation_id": cid, "language": conv.language, "messages": msgs}
            synthetic = cid.startswith(synthetic_prefix)
            (evals if not synthetic and in_holdout(cid, holdout) else train).append(row)
            stats["synthetic" if synthetic else "real"] += 1
    for name, rows in (("train.jsonl", train), ("eval.jsonl", evals)):
        with open(out / name, "w", encoding="utf-8", newline="\n") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out / "holdout-ids.txt").write_text("".join(r["conversation_id"] + "\n" for r in evals), encoding="utf-8")
    return {"train": len(train), "eval": len(evals), **stats}


def compare(results_path: str, labels_path: str, ids: set[str] | None = None) -> dict:
    """Agreement of predictions with our labels, per criterion and for the final score."""
    preds = {r["conversation_id"]: r for r in read_json(results_path)["results"]}
    labels = load_labels(Path(labels_path))["labels"]
    per = {code: [0, 0] for code in CODES}
    score_err, compared, missing = [], 0, []
    from .scoring import final_score
    for cid, label in labels.items():
        if ids is not None and cid not in ids:
            continue
        p = preds.get(cid)
        if not p or p.get("status") != "ok":
            missing.append(cid)
            continue
        compared += 1
        pv = {c["criterion_id"]: c["value"] for c in p["criteria"]}
        lv = {code: label["criteria"][code]["value"] for code in CODES}
        for code in CODES:
            per[code][1] += 1
            per[code][0] += int(json.dumps(pv[code]) == json.dumps(lv[code]))
        ls = final_score(lv)
        if ls is not None and p["score"]["final_score"] is not None:
            score_err.append(abs(ls - p["score"]["final_score"]))
    total_ok = sum(v[0] for v in per.values())
    total = sum(v[1] for v in per.values())
    return {
        "conversations": compared,
        "missing_predictions": missing,
        "exact_accuracy": round(total_ok / total, 4) if total else None,
        "per_criterion": {code: round(ok / n, 4) if n else None for code, (ok, n) in per.items()},
        "score_mae": round(sum(score_err) / len(score_err), 2) if score_err else None,
    }
