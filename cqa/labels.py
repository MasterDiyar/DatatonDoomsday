"""Team labels (our reference answers) per dataset: labels/<dataset-name>.json

{"dataset": "dev.json", "labels": {"<conversation_id>": {"criteria": {code: {value, rationale,
evidence_turns}}, "reviewed": true, "source": "human|model", "updated": "..."}}}

These are our own labels, used to measure accuracy and to fine-tune. They are not the organisers' answers.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from .checklist import CODES
from .config import ROOT
from .results import read_json, write_json
from .scoring import check_value

LABELS_DIR = ROOT / "labels"


def labels_path(dataset_name: str) -> Path:
    return LABELS_DIR / f"{Path(dataset_name).stem}.json"


def load(path: Path) -> dict:
    if not path.exists():
        return {"dataset": "", "labels": {}}
    data = read_json(path)
    data.setdefault("labels", {})
    return data


def clean_criteria(criteria: dict, n_turns: int) -> dict:
    out = {}
    for code in CODES:
        item = criteria.get(code)
        if not isinstance(item, dict) or "value" not in item:
            raise ValueError(f"{code} missing")
        check_value(code, item["value"])
        turns = sorted({int(t) for t in item.get("evidence_turns") or [] if type(t) is int and 1 <= t <= n_turns})
        out[code] = {"value": item["value"], "rationale": str(item.get("rationale") or "").strip(),
                     "evidence_turns": turns}
    return out


def save_label(dataset_name: str, conversation_id: str, criteria: dict, n_turns: int, reviewed: bool,
               source: str = "human") -> dict:
    path = labels_path(dataset_name)
    data = load(path)
    data["dataset"] = Path(dataset_name).name
    entry = {"criteria": clean_criteria(criteria, n_turns), "reviewed": bool(reviewed), "source": source,
             "updated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    data["labels"][conversation_id] = entry
    write_json(path, data)
    return entry


def from_results(results_file: dict, dataset_name: str, n_turns_by_id: dict, scenarios: dict | None = None,
                 overwrite_reviewed: bool = False) -> tuple[int, int, int]:
    """Import model predictions as unreviewed labels (e.g. teacher answers for synthetic chats).
    Predictions that contradict a synthetic scenario's intended outcome are skipped."""
    path = labels_path(dataset_name)
    data = load(path)
    data["dataset"] = Path(dataset_name).name
    added = skipped = contradicted = 0
    for r in results_file.get("results", []):
        cid = r.get("conversation_id")
        if r.get("status") != "ok" or cid not in n_turns_by_id:
            skipped += 1
            continue
        old = data["labels"].get(cid)
        if old and old.get("reviewed") and not overwrite_reviewed:
            skipped += 1
            continue
        criteria = {c["criterion_id"]: c for c in r["criteria"]}
        expected = (scenarios or {}).get(cid, {}).get("expected", {})
        if any(code in criteria and criteria[code]["value"] != v for code, v in expected.items()):
            contradicted += 1
            continue
        data["labels"][cid] = {"criteria": clean_criteria(criteria, n_turns_by_id[cid]), "reviewed": False,
                               "source": "model",
                               "updated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
        added += 1
    write_json(path, data)
    return added, skipped, contradicted
