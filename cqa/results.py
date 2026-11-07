"""results/<dev|prod>/results.json — EVALUATION RESULTS, JSON FORMAT 1.0."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from .checklist import BY_CODE, CODES
from .scoring import ScoreError, check_value, final_score

MODES = ("local", "external_api", "hybrid")


def build_file(team_name: str, model_name: str, mode: str, entries: list[dict]) -> dict:
    return {"schema_version": "1.0", "team_name": team_name, "model": {"mode": mode, "name": model_name},
            "results": entries}


def write_json(path: str | Path, data) -> None:
    """Atomic UTF-8 write without BOM, so an interrupted run never leaves a broken file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def validate_file(data, expect_ids: list[str] | None = None, expect_count: int | None = None) -> list[str]:
    """Returns a list of problems; empty means the file follows format 1.0."""
    problems = []
    if not isinstance(data, dict):
        return ["root must be a JSON object"]
    if data.get("schema_version") != "1.0":
        problems.append('schema_version must be "1.0"')
    if not isinstance(data.get("team_name"), str) or not data["team_name"].strip():
        problems.append("team_name is empty")
    model = data.get("model")
    if not isinstance(model, dict) or model.get("mode") not in MODES or not str(model.get("name") or "").strip():
        problems.append(f"model must be {{mode: one of {list(MODES)}, name: non-empty}}")
    results = data.get("results")
    if not isinstance(results, list):
        return problems + ["results must be an array"]
    seen = []
    for i, r in enumerate(results):
        where = f"results[{i}]"
        cid = r.get("conversation_id") if isinstance(r, dict) else None
        if not isinstance(cid, str) or not cid:
            problems.append(f"{where}: conversation_id missing")
            continue
        where = f"{where} ({cid})"
        if cid in seen:
            problems.append(f"{where}: duplicate conversation_id")
        seen.append(cid)
        if r.get("status") == "error":
            if r.get("criteria") != [] or r.get("score") is not None or not str(r.get("error") or "").strip():
                problems.append(f"{where}: error entries need criteria [], score null and an error text")
            continue
        if r.get("status") != "ok":
            problems.append(f"{where}: status must be ok or error")
            continue
        crit = r.get("criteria")
        if not isinstance(crit, list) or len(crit) != 8:
            problems.append(f"{where}: exactly 8 criteria required")
            continue
        codes = [c.get("criterion_id") for c in crit]
        for code in CODES:
            if codes.count(code) != 1:
                problems.append(f"{where}: criterion {code} must appear exactly once")
        values = {}
        for c in crit:
            code = c.get("criterion_id")
            if code not in BY_CODE:
                problems.append(f"{where}: unknown criterion {code!r}")
                continue
            try:
                check_value(code, c.get("value", "<missing>"))
                values[code] = c["value"]
            except ScoreError as e:
                problems.append(f"{where}: {e}")
            if not isinstance(c.get("rationale"), str) or not c["rationale"].strip():
                problems.append(f"{where}: {code} rationale is empty")
            turns = c.get("evidence_turns")
            if not isinstance(turns, list) or any(type(t) is not int or t < 1 for t in turns):
                problems.append(f"{where}: {code} evidence_turns must be 1-based integers")
        score = r.get("score")
        if not isinstance(score, dict):
            problems.append(f"{where}: score object missing")
            continue
        if len(values) == 8:
            expected = final_score(values)
            if score.get("final_score") != expected:
                problems.append(f"{where}: final_score {score.get('final_score')} != recomputed {expected}")
            if score.get("critical_flag") is not (values["critical"] is True):
                problems.append(f"{where}: critical_flag must equal the critical criterion value")
    if expect_ids is not None:
        missing = [i for i in expect_ids if i not in seen]
        extra = [i for i in seen if i not in expect_ids]
        if missing:
            problems.append(f"missing conversation_ids: {', '.join(missing)}")
        if extra:
            problems.append(f"conversation_ids not in the dataset: {', '.join(extra)}")
    if expect_count is not None and len(results) != expect_count:
        problems.append(f"expected exactly {expect_count} entries, found {len(results)}")
    return problems
