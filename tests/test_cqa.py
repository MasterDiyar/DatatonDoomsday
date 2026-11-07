"""python -m unittest discover tests   (standard library only; a fake Ollama server stands in for the model)"""

import json
import tempfile
import threading
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from cqa import labels, prompt, sft
from cqa.checklist import CODES
from cqa.data import load_dataset, raw_records
from cqa.evaluate import OutputError, _majority, parse_output, to_entry
from cqa.model import ModelConfig
from cqa.results import build_file, read_json, validate_file, write_json
from cqa.runner import run_dataset
from cqa.scoring import ScoreError, final_score

GUIDE = {"contact": True, "discovery": None, "identification": None, "consultation": 15, "empathy": True,
         "dialogue": 7, "closure": True, "critical": False}
SETTINGS = {"team_name": "T", "submission_model_name": "", "submission_mode": "local"}


def chat_record(cid="c-1", lang="EN"):
    return {"conversation_id": cid, "language": lang, "channel": "agent chat", "transcript": [
        {"timestamp": "00:00", "speaker": "client", "text": "What is the commission?"},
        {"timestamp": "00:01", "speaker": "bot", "text": "Connecting you."},
        {"timestamp": "00:20", "speaker": "operator", "text": "Hello, I am Anna."},
        {"timestamp": "00:40", "speaker": "operator", "text": "It is 0.1%. Anything else? Goodbye!"}]}


def write_dataset(folder: Path, records, name="ds.json") -> Path:
    p = folder / name
    p.write_text(json.dumps({"data": {"records": records}}, ensure_ascii=False), encoding="utf-8")
    return p


def answer(values=GUIDE, turns=(3,)):
    return json.dumps({"criteria": {k: {"rationale": f"because {k}", "evidence_turns": list(turns), "value": v}
                                    for k, v in values.items()}})


class FakeOllama:
    """Minimal /api/chat + /api/tags server. `replies` is a list of (status, body) consumed per call."""

    def __init__(self, replies):
        self.replies, self.requests = list(replies), []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                self._send(200, json.dumps({"models": [{"name": "qwen2.5:7b-instruct"}]}))

            def do_POST(self):
                outer.requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                status, content = outer.replies.pop(0) if len(outer.replies) > 1 else outer.replies[0]
                body = content if status != 200 else json.dumps({"message": {"content": content}, "prompt_eval_count": 1, "eval_count": 1})
                self._send(status, body)

            def _send(self, status, body):
                data = body.encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.cfg = ModelConfig(base_url=f"http://127.0.0.1:{self.httpd.server_port}", timeout=10)

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class Scoring(unittest.TestCase):
    def test_appendix_example_85_38(self):
        self.assertEqual(final_score(GUIDE), 85.38)

    def test_full_marks_100_and_zero(self):
        best = {"contact": True, "discovery": True, "identification": True, "consultation": 20, "empathy": True,
                "dialogue": 10, "closure": True, "critical": True}
        self.assertEqual(final_score(best), 100.0)  # critical flag does not reduce the score
        worst = {**{k: False for k in CODES}, "consultation": 0, "dialogue": 0}
        self.assertEqual(final_score(worst), 0.0)

    def test_na_redistribution(self):
        v = {**GUIDE, "consultation": None, "closure": None}  # applicable 10+15+15 = 40; earned 10+15+10.5
        self.assertEqual(final_score(v), 88.75)

    def test_rejects_disallowed_values(self):
        for bad in ({**GUIDE, "contact": None}, {**GUIDE, "consultation": 12}, {**GUIDE, "dialogue": True},
                    {**GUIDE, "consultation": False}):
            with self.assertRaises(ScoreError):
                final_score(bad)


class Data(unittest.TestCase):
    def test_shapes_speakers_and_bad_records(self):
        with tempfile.TemporaryDirectory() as d:
            recs = [chat_record("a "), {"conversation_id": "b", "transcript": []}, chat_record("a "),
                    {"conversation_id": "c", "transcript": [{"role": "Agent", "message": "hi"}]}]
            p = write_dataset(Path(d), recs)
            p.write_bytes(b"\xef\xbb\xbf" + p.read_bytes())  # Windows BOM
            convs = load_dataset(p)
            self.assertEqual([c.conversation_id for c in convs], ["a ", "b", "c"])  # exact IDs, duplicate dropped
            self.assertIsNone(convs[0].error)
            self.assertEqual([t.speaker for t in convs[0].turns], ["CLIENT", "BOT", "OPERATOR", "OPERATOR"])
            self.assertEqual(convs[0].operator_turns, [3, 4])
            self.assertIn("transcript", convs[1].error)
            self.assertIsNone(convs[2].error)
            self.assertEqual(convs[2].turns[0].speaker, "OPERATOR")
        self.assertEqual(len(raw_records([chat_record()])), 1)


class Prompt(unittest.TestCase):
    def test_turns_numbered_from_1_over_whole_transcript(self):
        with tempfile.TemporaryDirectory() as d:
            conv = load_dataset(write_dataset(Path(d), [chat_record()]))[0]
        text = prompt.user_prompt(conv)
        self.assertIn("[1] CLIENT (00:00): What is the commission?", text)
        self.assertIn("[2] BOT", text)
        schema = prompt.output_schema()["properties"]["criteria"]
        self.assertEqual(schema["required"], list(CODES))
        self.assertEqual(schema["properties"]["consultation"]["properties"]["value"]["enum"], [0, 5, 10, 15, 20, None])
        self.assertEqual(list(schema["properties"]["contact"]["properties"]), ["rationale", "evidence_turns", "value"])


class Parsing(unittest.TestCase):
    def setUp(self):
        with tempfile.TemporaryDirectory() as d:
            self.conv = load_dataset(write_dataset(Path(d), [chat_record()]))[0]

    def test_valid_answer(self):
        out = parse_output(answer(), self.conv)
        self.assertEqual({k: v["value"] for k, v in out.items()}, GUIDE)

    def test_lenient_values_and_evidence(self):
        vals = {**GUIDE, "contact": "Yes", "discovery": "N/A", "consultation": "15", "dialogue": 7.0}
        raw = json.loads(answer(vals, turns=(3, 99, 0)))
        raw["criteria"]["closure"]["evidence_turns"] = []
        out = parse_output("text before " + json.dumps(raw) + " after", self.conv)
        self.assertIs(out["contact"]["value"], True)
        self.assertIsNone(out["discovery"]["value"])
        self.assertEqual(out["contact"]["evidence_turns"], [3])        # out-of-range turns removed
        self.assertEqual(out["closure"]["evidence_turns"], [3, 4])     # fallback: operator turns
        self.assertEqual(out["discovery"]["evidence_turns"], [3])      # model's turns kept

    def test_invalid_answers(self):
        for bad in ("not json", answer({**GUIDE, "consultation": 12}), json.dumps({"criteria": {"contact": {}}})):
            with self.assertRaises(OutputError):
                parse_output(bad, self.conv)

    def test_majority_vote(self):
        a = parse_output(answer(), self.conv)
        b = parse_output(answer({**GUIDE, "consultation": 20}), self.conv)
        c = parse_output(answer({**GUIDE, "consultation": 20, "empathy": False}), self.conv)
        m = _majority([a, b, c])
        self.assertEqual(m["consultation"]["value"], 20)
        self.assertIs(m["empathy"]["value"], True)


class ResultsFormat(unittest.TestCase):
    def entry(self, cid="c-1", values=GUIDE):
        crit = {k: {"value": v, "rationale": "r", "evidence_turns": [1]} for k, v in values.items()}
        return to_entry(cid, crit)

    def test_entry_matches_appendix(self):
        e = self.entry()
        self.assertEqual(list(e), ["conversation_id", "status", "criteria", "score"])
        self.assertEqual([c["criterion_id"] for c in e["criteria"]], list(CODES))
        self.assertEqual(e["score"], {"final_score": 85.38, "critical_flag": False})
        self.assertEqual(validate_file(build_file("T", "m", "local", [e]), ["c-1"], 1), [])

    def test_validator_catches_problems(self):
        e = self.entry()
        dup = build_file("T", "m", "local", [e, e])
        self.assertTrue(any("duplicate" in p for p in validate_file(dup)))
        self.assertTrue(any("missing conversation_ids: c-2" in p for p in validate_file(build_file("T", "m", "local", [e]), ["c-1", "c-2"])))
        self.assertTrue(any("exactly 10" in p for p in validate_file(build_file("T", "m", "local", [e]), expect_count=10)))
        bad = json.loads(json.dumps(e))
        bad["score"]["final_score"] = 90
        bad["criteria"][0]["evidence_turns"] = [0]
        bad["score"]["critical_flag"] = True
        problems = validate_file(build_file("T", "m", "local", [bad]))
        self.assertTrue(any("recomputed" in p for p in problems))
        self.assertTrue(any("1-based" in p for p in problems))
        self.assertTrue(any("critical_flag" in p for p in problems))
        err = {"conversation_id": "x", "status": "error", "criteria": [], "score": None, "error": "boom"}
        self.assertEqual(validate_file(build_file("T", "m", "local", [err])), [])
        self.assertTrue(validate_file(build_file("T", "m", "bad-mode", [err])))

    def test_written_file_is_utf8_without_bom(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r" / "results.json"
            write_json(p, {"x": "Сәлем"})
            raw = p.read_bytes()
            self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))
            self.assertIn("Сәлем", raw.decode("utf-8"))


class EndToEnd(unittest.TestCase):
    def test_run_resume_retry_and_errors(self):
        fake = FakeOllama([(200, "garbage"), (200, answer())])  # first answer invalid -> retried
        try:
            with tempfile.TemporaryDirectory() as d, mock.patch("cqa.runner.ROOT", Path(d)):
                data = write_dataset(Path(d), [chat_record("c-1"), chat_record("c-2"), {"conversation_id": "bad"}])
                out = Path(d) / "results" / "dev" / "results.json"
                code = run_dataset(str(data), str(out), SETTINGS, fake.cfg, log=lambda *a: None)
                res = read_json(out)
                self.assertEqual(code, 1)  # one invalid record
                self.assertEqual([r["status"] for r in res["results"]], ["ok", "ok", "error"])
                self.assertEqual(res["results"][0]["score"]["final_score"], 85.38)
                self.assertEqual(res["model"], {"mode": "local", "name": "qwen2.5:7b-instruct"})
                sent = fake.requests[0]
                self.assertEqual(sent["options"]["num_ctx"], 16384)
                self.assertIn("format", sent)
                n = len(fake.requests)
                run_dataset(str(data), str(out), SETTINGS, fake.cfg, log=lambda *a: None)  # resume: ok entries kept
                self.assertEqual(len(fake.requests), n)
        finally:
            fake.close()

    def test_fatal_model_error_stops_and_keeps_all_ids(self):
        fake = FakeOllama([(404, '{"error":"model not found"}')])
        try:
            with tempfile.TemporaryDirectory() as d, mock.patch("cqa.runner.ROOT", Path(d)):
                data = write_dataset(Path(d), [chat_record("c-1"), chat_record("c-2")])
                out = Path(d) / "results.json"
                logs = []
                self.assertEqual(run_dataset(str(data), str(out), SETTINGS, fake.cfg, expect_count=2, log=logs.append), 1)
                self.assertEqual(len(fake.requests), 1)  # stopped after the first failure
                res = read_json(out)
                self.assertEqual([r["status"] for r in res["results"]], ["error", "error"])
                self.assertEqual(validate_file(res, ["c-1", "c-2"], 2), [])
                self.assertTrue(any("STOPPED" in x for x in logs))
        finally:
            fake.close()


class LabelsAndTraining(unittest.TestCase):
    def test_labels_sft_and_compare(self):
        with tempfile.TemporaryDirectory() as d, mock.patch("cqa.labels.LABELS_DIR", Path(d) / "labels"):
            recs = [chat_record(f"c-{i}") for i in range(8)] + [chat_record("synth-0001")]
            data = write_dataset(Path(d), recs)
            crit = {k: {"value": v, "rationale": "r", "evidence_turns": [3]} for k, v in GUIDE.items()}
            for r in recs[:-1]:
                labels.save_label("ds.json", r["conversation_id"], crit, 4, reviewed=True)
            entry = to_entry("synth-0001", crit)
            added, _, contradicted = labels.from_results(build_file("T", "m", "local", [entry]), "ds.json",
                                                         {"synth-0001": 4}, {"synth-0001": {"expected": {"contact": False}}})
            self.assertEqual((added, contradicted), (0, 1))
            labels.from_results(build_file("T", "m", "local", [entry]), "ds.json", {"synth-0001": 4})
            lp = labels.labels_path("ds.json")
            stats = sft.build([(str(data), str(lp))], str(Path(d) / "sft"), holdout=0.5)
            self.assertEqual(stats["train"] + stats["eval"], 8)       # unreviewed synthetic label skipped
            stats = sft.build([(str(data), str(lp))], str(Path(d) / "sft"), holdout=0.5, include_unreviewed=True)
            self.assertEqual(stats["train"] + stats["eval"], 9)
            row = json.loads((Path(d) / "sft" / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual([m["role"] for m in row["messages"]], ["system", "user", "assistant"])
            self.assertEqual(row["messages"][0]["content"], prompt.system_prompt())
            self.assertEqual(parse_output(row["messages"][2]["content"], load_dataset(data)[0])["consultation"]["value"], 15)
            preds = Path(d) / "pred.json"
            write_json(preds, build_file("T", "m", "local", [to_entry(r["conversation_id"], crit) for r in recs]))
            rep = sft.compare(str(preds), str(lp))
            self.assertEqual(rep["exact_accuracy"], 1.0)
            self.assertEqual(rep["score_mae"], 0.0)


class ReviewServer(unittest.TestCase):
    def test_api_roundtrip(self):
        from cqa import server
        with tempfile.TemporaryDirectory() as d:
            ds = Path(d) / "datasets"
            ds.mkdir()
            write_dataset(ds, [chat_record("c-1")], "dev.json")
            with mock.patch.object(server, "DATASETS", ds), mock.patch("cqa.labels.LABELS_DIR", Path(d) / "labels"):
                httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
                threading.Thread(target=httpd.serve_forever, daemon=True).start()
                base = f"http://127.0.0.1:{httpd.server_port}"
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                get = lambda p: json.loads(opener.open(base + p, timeout=5).read())
                try:
                    self.assertEqual(get("/api/datasets"), [{"name": "dev.json", "count": 1}])
                    self.assertIn(b"Dynamics 365", opener.open(base + "/d365", timeout=5).read())  # embed demo page
                    self.assertEqual(get("/api/dataset?name=dev.json")[0]["turns"][2]["n"], 3)
                    crit = {k: {"value": v, "rationale": "ok", "evidence_turns": [3, 77]} for k, v in GUIDE.items()}
                    req = urllib.request.Request(base + "/api/labels?name=dev.json", method="POST",
                                                 data=json.dumps({"conversation_id": "c-1", "criteria": crit}).encode(),
                                                 headers={"Content-Type": "application/json"})
                    saved = json.loads(opener.open(req, timeout=5).read())
                    self.assertEqual(saved["criteria"]["contact"]["evidence_turns"], [3])
                    self.assertTrue(get("/api/labels?name=dev.json")["c-1"]["reviewed"])
                    with self.assertRaises(urllib.error.HTTPError):
                        opener.open(base + "/api/dataset?name=../config.json", timeout=5)
                    evil = urllib.request.Request(base + "/api/labels?name=dev.json", method="POST", data=b"{}",
                                                  headers={"Origin": "http://evil.example"})
                    with self.assertRaises(urllib.error.HTTPError) as ctx:
                        opener.open(evil, timeout=5)
                    self.assertEqual(ctx.exception.code, 403)
                finally:
                    httpd.shutdown()
                    httpd.server_close()


if __name__ == "__main__":
    unittest.main()


class Robustness(unittest.TestCase):
    def test_dropped_connection_is_retried(self):
        import socket
        calls = {"n": 0}

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                calls["n"] += 1
                if calls["n"] == 1:  # crash mid-request: close without any HTTP response
                    self.connection.shutdown(socket.SHUT_RDWR)
                    return
                data = json.dumps({"message": {"content": answer()}, "done": True}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            with tempfile.TemporaryDirectory() as d, mock.patch("cqa.runner.ROOT", Path(d)):
                data = write_dataset(Path(d), [chat_record("c-1")])
                out = Path(d) / "r.json"
                cfg = ModelConfig(base_url=f"http://127.0.0.1:{httpd.server_port}", timeout=10)
                logs = []
                self.assertEqual(run_dataset(str(data), str(out), SETTINGS, cfg, log=logs.append), 0)
                self.assertEqual(read_json(out)["results"][0]["status"], "ok")
                self.assertTrue(any("connection dropped" in x for x in logs))
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_ctrl_c_saves_and_exits_cleanly(self):
        with tempfile.TemporaryDirectory() as d, mock.patch("cqa.runner.ROOT", Path(d)), \
                mock.patch("cqa.runner.evaluate", side_effect=KeyboardInterrupt):
            data = write_dataset(Path(d), [chat_record("c-1"), chat_record("c-2")])
            out = Path(d) / "r.json"
            logs = []
            self.assertEqual(run_dataset(str(data), str(out), SETTINGS, ModelConfig(), log=logs.append), 130)
            self.assertEqual(len(read_json(out)["results"]), 2)  # file stays complete and valid
            self.assertTrue(any("Interrupted" in x for x in logs))


class ClosureCheck(unittest.TestCase):
    def conv(self, *turns):
        rec = {"conversation_id": "z", "language": "RU", "channel": "agent chat",
               "transcript": [{"timestamp": "", "speaker": s, "text": t} for s, t in turns]}
        with tempfile.TemporaryDirectory() as d:
            return load_dataset(write_dataset(Path(d), [rec]))[0]

    def run_check(self, conv, model_answer):
        from cqa import closure
        fake = lambda *a, **k: (json.dumps(model_answer), {})
        with mock.patch.object(closure, "chat", fake):
            return closure.check_closure(conv, ModelConfig())

    def test_bare_goodbye_is_no_even_if_model_calls_it_a_summary(self):
        c = self.conv(("client", "Где тариф?"), ("operator", "Здравствуйте, я Алина. В «Кабинете»."),
                      ("client", "Спасибо"), ("operator", "Пожалуйста. Хорошего вечера!"))
        r = self.run_check(c, {"summary": "Пожалуйста. Хорошего вечера!", "offer": "", "goodbye": "Хорошего вечера"})
        self.assertIs(r["value"], False)

    def test_summary_or_offer_plus_goodbye_is_yes(self):
        c = self.conv(("client", "Где тариф?"), ("operator", "Я передал вопрос специалистам, ответ придёт в этот чат. До свидания!"))
        r = self.run_check(c, {"summary": "Я передал вопрос специалистам, ответ придёт в этот чат", "offer": "", "goodbye": "До свидания"})
        self.assertIs(r["value"], True)
        c = self.conv(("client", "Где тариф?"), ("operator", "Если вопрос останется, напишите нам. До свидания!"))
        self.assertIs(self.run_check(c, {})["value"], True)  # offer + goodbye found by rules alone

    def test_invented_quote_is_ignored(self):
        c = self.conv(("client", "Где тариф?"), ("operator", "Хорошего дня!"))
        r = self.run_check(c, {"summary": "Вопрос решён, отчёт отправлен клиенту", "offer": "", "goodbye": ""})
        self.assertIs(r["value"], False)

    def test_disconnect_is_na_and_operator_close_is_no(self):
        c = self.conv(("client", "Где форма?"), ("operator", "Здравствуйте. Я Илья. Форма на английском."),
                      ("client", "Ок"), ("system", "Соединение с клиентом прервано."))
        self.assertIsNone(self.run_check(c, {})["value"])
        c = self.conv(("client", "Где форма?"), ("operator", "Тогда не знаю."), ("system", "Оператор закрыл чат."))
        r = self.run_check(c, {})
        self.assertIs(r["value"], False)
        self.assertIn("закрыл чат", r["rationale"])
        # the operator talking after the customer hung up does not count
        c = self.conv(("client", "Понятно, спасибо."), ("operator", "Поручение передано трейдеру."),
                      ("system", "Клиент завершил разговор."), ("operator", "Остались дополнительные вопросы?"))
        self.assertIsNone(self.run_check(c, {})["value"])

    def test_prompt_is_russian_and_rationale_language_switches(self):
        self.assertIn("Оцени работу оператора", prompt.system_prompt())
        self.assertIn("на русском языке", prompt.system_prompt())
        self.assertIn("на казахском языке", prompt.system_prompt("kk"))
