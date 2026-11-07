# DatatonDoomsday — CQA evaluator

Scores live-chat operator work with the **Freedom Broker chat quality checklist** (8 criteria) using a **local model**: Qwen2.5-7B-Instruct on Ollama. It writes `results/dev/results.json` and `results/prod/results.json` in the organisers' **JSON format 1.0**.

- **Python 3.10+ only.** Evaluation needs no `pip install`, Node or `.env`.
- Settings live in `config.json` (team name, model, Ollama address).
- The model's output is forced into the right JSON shape (Ollama structured output). Our code then validates every value and computes the score itself, with N/A redistribution and rounding to two decimals.
- Runs are resumable. Results are saved after every chat, and a failed chat becomes `status: "error"` with its original ID.
- A local review page lets the team correct answers. The corrected answers are used to measure accuracy and to fine-tune the model.

All commands below are for **Windows PowerShell**. Run them from the project folder, one line at a time.

## 1. Install (once)

```powershell
winget install Python.Python.3.12
winget install Ollama.Ollama
```

Close and reopen PowerShell, then:

```powershell
ollama pull qwen2.5:7b-instruct
python run.py check
```

`check` must end with `OK: model answered`. If it says Ollama is not reachable, start the Ollama app (or run `ollama serve` in another window).

Set your team name in `config.json` (`team_name`).

## 2. Evaluate DEV

Download the dataset. Keep the organisers' URL exactly as given:

```powershell
python run.py fetch "PASTE_DEV_URL" --out datasets/dev.json
```

If you already have the file, just copy it to `datasets\dev.json`. Then:

```powershell
python run.py evaluate datasets/dev.json --split dev
```

This prints a score per chat and an ETA, and writes `results/dev/results.json`.

- If it stops or some chats show `ERROR`, run the **same command again**. Finished chats are kept; only failed or missing ones are re-run.
- `--force` re-evaluates everything.
- `--votes 3` runs the model three times per chat and takes the majority (slower).

Publish the results:

```powershell
python run.py validate results/dev/results.json --data datasets/dev.json
git add results/dev/results.json
git commit -m "DEV results"
git push
```

## 3. PROD (the 1-hour window)

```powershell
python run.py fetch "PASTE_PROD_URL" --out datasets/prod.json
python run.py evaluate datasets/prod.json --split prod
python run.py validate results/prod/results.json --data datasets/prod.json --expect 10
git add results/prod/results.json
git commit -m "PROD results"
git push
```

`--split prod` checks that there are exactly 10 entries. Rehearse once on 10 DEV chats to know your timing:

```powershell
python run.py evaluate datasets/dev.json --out runs/rehearsal.json --ids my10.txt
```

`my10.txt` contains 10 conversation IDs, one per line.

## 4. Review page (the team's labels)

```powershell
python run.py serve
```

Open http://127.0.0.1:8000.

1. Pick the dataset and the model answers file (`results/dev/results.json`).
2. For each chat: read it, press **Fill from model**, fix what is wrong, then **Save as reviewed** (Ctrl+S).
3. The page shows the label score next to the model score. Evidence turns are highlighted in the chat. A model answer that differs from your label is shown in red.

Labels are saved in `labels/<dataset>.json`. They are our own reference answers, not the organisers' ground truth.

Measure how often the model agrees with your labels:

```powershell
python run.py compare results/dev/results.json labels/dev.json
```

## 5. Fine-tuning (same model, NVIDIA GPU ≥ 12 GB)

WSL2 (Ubuntu) is recommended for training. Native Windows works too, but uses more GPU memory.

1. **Review DEV labels** on the review page. This is the most important step: the fine-tuned model can only be as good as the labels.
2. *(optional)* **Synthetic chats** to enlarge the data. They are generated, then labelled by the model, and answers that contradict the planned scenario are dropped:
   ```powershell
   python run.py synth --count 300 --out datasets/synth.json
   python run.py evaluate datasets/synth.json --out runs/synth-results.json
   python run.py labels-from-results runs/synth-results.json --data datasets/synth.json --scenarios datasets/synth-scenarios.json
   ```
   A stronger API model can be used for these steps: add `--backend openai --base-url https://... --model <id>` and set the key with `$env:CQA_API_KEY="..."`.
3. **Training data.** The same prompt is used for training and evaluation. 25% of the real chats are held out for testing:
   ```powershell
   python run.py build-sft --pair datasets/dev.json labels/dev.json --pair datasets/synth.json labels/synth.json --include-unreviewed
   ```
   Without `--include-unreviewed`, only reviewed labels are used. The synthetic labels are unreviewed.
4. **Train:**
   ```powershell
   pip install torch --index-url https://download.pytorch.org/whl/cu124
   pip install -r training/requirements.txt
   python training/train_lora.py --dry-run
   python training/train_lora.py
   python training/merge_and_export.py --llama-cpp C:/path/to/llama.cpp
   ollama create cqa-qwen7b -f training/Modelfile
   ```
5. **Check that it helped.** Compare base and fine-tuned on the held-out chats:
   ```powershell
   python run.py evaluate datasets/dev.json --out runs/base.json --ids training/data/holdout-ids.txt
   python run.py evaluate datasets/dev.json --out runs/ft.json --ids training/data/holdout-ids.txt --model cqa-qwen7b
   python run.py compare runs/base.json labels/dev.json --ids training/data/holdout-ids.txt
   python run.py compare runs/ft.json labels/dev.json --ids training/data/holdout-ids.txt
   ```
   Use the better one. Put it in `config.json` (`model`, `submission_model_name`) so `results.json` names the model that was actually used.

## How a chat is scored

| Code | Criterion | Values | Weight |
| --- | --- | --- | ---: |
| contact | 01 Contact establishment | true / false | 10 |
| discovery | 02 Identifying the need | true / false / null | 15 |
| identification | 03 Identification | true / false / null | 20 |
| consultation | 04 Correct and complete consultation | 0, 5, 10, 15, 20 / null | 20 |
| empathy | 05 Empathy | true / false | 15 |
| dialogue | 06 Dialogue management | 0, 4, 7, 10 | 15 |
| closure | 07 Closing | true / false / null | 5 |
| critical | 08 Severe errors (flag, no points) | true / false | — |

- The definitions given to the model are in `cqa/checklist.py`; the prompt is in `cqa/prompt.py`.
- `final_score` = 100 × earned / applicable weight. N/A criteria are excluded, which redistributes their weight.
- `critical_flag` = the value of `critical`. It does not reduce the score.
- `evidence_turns` are 1-based over the whole transcript (customer, bot and system turns included).

## Files

```
run.py            command line (check, fetch, evaluate, validate, serve, compare, synth, build-sft)
config.json       team name, model, Ollama address
cqa/              checklist, prompt, model client, scoring, results format, review server
web/index.html    review page
training/         QLoRA fine-tuning, GGUF export, Ollama Modelfile
tests/            python -m unittest discover tests
datasets/         input data (only sample.json is committed)
results/          dev/results.json and prod/results.json for the organisers
labels/, runs/    team labels and run logs (not committed)
```

Try everything without real data: `python run.py evaluate datasets/sample.json --out runs/sample.json`.
