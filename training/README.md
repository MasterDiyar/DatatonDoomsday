# Собственная локальная модель (QLoRA fine-tune)

Цель — дообучить открытую модель (по умолчанию **Qwen2.5-7B-Instruct**) на оценках по официальному чек-листу и запускать её локально на NVIDIA GPU ≥ 12 ГБ. В `results.json` она указывается как `"mode": "local"`.

Эталонных ответов DEV у нас нет. Поэтому метки создаются так: **учитель** (сильная модель по API) → **ревью командой** в интерфейсе → **дообучение** → **сравнение** с базовой моделью на отложенной части DEV.

> Важно: все метки — «серебряные» (наши), а не Ground Truth организаторов. Качество дообученной модели не выше качества меток. Время на ревью DEV — главное вложение.

Все команды выполняются из корня репозитория. Данные, метки, адаптеры и GGUF лежат в `datasets/` и `training/out/` и в Git не попадают.

## 0. Окружение

```sh
python -m venv .venv && . .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu124 torch
pip install -r training/requirements.txt
# Для GGUF: собрать llama.cpp (https://github.com/ggml-org/llama.cpp) и pip install -r llama.cpp/requirements.txt
# Для запуска: установить Ollama (или использовать llama-server из llama.cpp)
```

## 1. Метки учителя для DEV

DEV сохраните в `datasets/dev.json` (папка в `.gitignore`). В `.env` временно укажите модель-учителя (`MODEL_API_KEY`, `MODEL_BASE_URL`, `MODEL_NAME`).

```sh
node --env-file-if-exists=.env scripts/evaluate-file.mjs --data datasets/dev.json --checklist examples/checklist-official.json --out datasets/teacher-dev.json --verification verified
```

## 2. Ревью командой

1. Запустите `npm start` и импортируйте `datasets/dev.json`.
2. Оцените DEV моделью-учителем в интерфейсе. Либо используйте CLI-файл из шага 1 только как черновик.
3. Ошибочные ответы исправьте через «Апелляции» с решением «изменить».
4. Выгрузите состояние: «Экспорт», или `curl http://127.0.0.1:4173/api/export > datasets/cqa-export.json`.

```sh
# reviewed-ids.txt — ID диалогов, которые команда проверила целиком (в том числе без правок)
node scripts/export-labels.mjs --state datasets/cqa-export.json --reviewed-ids datasets/reviewed-ids.txt --out datasets/labels-dev.json
```

Без ревью можно выгрузить и черновик учителя: `--state datasets/teacher-dev.json`. Но тогда модель просто копирует учителя вместе с его ошибками.

## 3. Синтетические диалоги (аугментация)

65 диалогов DEV мало. Генератор создаёт вымышленные чаты RU/KZ/EN с заданным поведением оператора: нет представления, нет идентификации перед защищённым действием, предложение частной платной услуги, обрыв чата, игнорирование жалобы и т. д. Распределение сбалансировано.

```sh
TEACHER_API_KEY=... TEACHER_MODEL_NAME=... TEACHER_BASE_URL=... \
  node scripts/synthesize-dialogues.mjs --count 400
node --env-file-if-exists=.env scripts/evaluate-file.mjs --data datasets/synth-dialogues.json --checklist examples/checklist-official.json --out datasets/teacher-synth.json --verification verified
node scripts/export-labels.mjs --state datasets/teacher-synth.json --out datasets/labels-synth.json
```

Сценарии сохраняются в `datasets/synth-scenarios.json`. Если метка учителя противоречит однозначному следствию сценария, пример отбрасывается. Пример такого противоречия: по сценарию оператор нагрубил, а учитель поставил «грубых ошибок нет».

## 4. SFT-датасет

```sh
node scripts/build-sft-dataset.mjs --labels datasets/labels-dev.json --labels datasets/labels-synth.json --scenarios datasets/synth-scenarios.json --holdout-fraction 0.25
```

- Промпты строятся функцией `buildModelMessages()` из `src/core.mjs`. Это тот же текст, который оценщик отправляет при работе, байт в байт.
- Цель (ответ ассистента) проверяется `validateModelResults()`, как настоящий ответ модели.
- Отложенная часть — только реальные диалоги DEV, синтетика всегда идёт в обучение. Результат: `datasets/sft/holdout-ids.txt` и `holdout-truth.json`.
- Не меняйте `examples/checklist-official.json` между обучением и оценкой: модель учится именно на этом промпте.

## 5. Обучение (QLoRA, 4-bit)

```sh
python training/train_lora.py --dry-run     # длины в токенах; GPU не нужен
python training/train_lora.py               # → training/out/adapter
```

- По умолчанию: r=16, alpha=32, lr 1e-4, 3 эпохи, batch 1 × accumulation 8, `--max-len 12288`.
- Длинные примеры пропускаются, а не обрезаются.
- Loss считается только по ответу.
- Liger-kernel не создаёт полные логиты словаря, поэтому длинный промпт помещается в 12 ГБ. Если не хватает памяти, уменьшите `--max-len`.
- Другая база: `--base Qwen/Qwen3-8B` (проверьте `--dry-run`, что чат-шаблон совпадает).

## 6. Слияние, GGUF, запуск

```sh
python training/merge_and_export.py --llama-cpp ~/llama.cpp     # → training/out/cqa-q5_k_m.gguf
ollama create cqa-qwen7b-lora -f training/Modelfile
```

`.env` для оценки своей моделью:

```dotenv
ALLOW_LOCAL_MODEL=1
MODEL_API_KEY=local
MODEL_BASE_URL=http://127.0.0.1:11434/v1
MODEL_NAME=cqa-qwen7b-lora
MODEL_TIMEOUT_SECONDS=300
# Для --verification verified нужна ненулевая температура, иначе два прохода совпадут:
MODEL_EXTRA_BODY={"temperature":0.3}
SUBMISSION_MODEL_NAME=Qwen2.5-7B-Instruct + CQA QLoRA (GGUF Q5_K_M)
TEAM_NAME=Your Team
```

Вместо Ollama можно использовать `llama-server -m training/out/cqa-q5_k_m.gguf -c 16384 -ngl 99 --port 8080` с `MODEL_BASE_URL=http://127.0.0.1:8080/v1`.

## 7. Дало ли дообучение что-то? Сравнение на отложенной части

```sh
# (a) базовая модель без дообучения: ollama pull qwen2.5:7b-instruct; OLLAMA_CONTEXT_LENGTH=16384 ollama serve
MODEL_NAME=qwen2.5:7b-instruct node --env-file-if-exists=.env scripts/evaluate-file.mjs --data datasets/dev.json --ids datasets/sft/holdout-ids.txt --checklist examples/checklist-official.json --out datasets/eval-base.json --verification single
node scripts/benchmark.mjs --results datasets/eval-base.json --truth datasets/sft/holdout-truth.json --out datasets/acc-base.json
# (b) дообученная, один проход; (c) дообученная, verified
node --env-file-if-exists=.env scripts/evaluate-file.mjs --data datasets/dev.json --ids datasets/sft/holdout-ids.txt --checklist examples/checklist-official.json --out datasets/eval-ft.json --verification single
node scripts/benchmark.mjs --results datasets/eval-ft.json --truth datasets/sft/holdout-truth.json --out datasets/acc-ft.json
```

Используйте лучший вариант. Если дообученная модель не лучше базовой, отправляйте базовую. В `model.name` указывайте то, что реально использовано.

## 8. Файлы для организаторов

```sh
node --env-file-if-exists=.env scripts/evaluate-file.mjs --data datasets/dev.json --checklist examples/checklist-official.json --out datasets/final-dev.json --verification single
node --env-file-if-exists=.env scripts/export-hackathon.mjs --state datasets/final-dev.json --data datasets/dev.json --dataset dev
# PROD (за 1 час до стоп-кода): 10 диалогов, проверка ровно 10 записей
node --env-file-if-exists=.env scripts/evaluate-file.mjs --data datasets/prod.json --checklist examples/checklist-official.json --out datasets/final-prod.json --verification single
node --env-file-if-exists=.env scripts/export-hackathon.mjs --state datasets/final-prod.json --data datasets/prod.json --dataset prod
git add results && git commit -m "Add results" && git push
```

- Экспортёр переводит внутренние ID критериев в коды формата 1.0: `needs→discovery`, `management→dialogue`, `closing→closure`, `gross_errors→critical`.
- Индексы реплик переводятся из 0-based в 1-based `evidence_turns`.
- `final_score` считается с перераспределением N/A и округляется до двух знаков.
- Ошибки обработки и ответы `pending` записываются как `status: "error"`, а не нулями.
- Если в диалоге `pending` или ошибка, прогоните только его заново (`--ids`) и повторите экспорт.

Порядок времени: заранее прогоните 10 диалогов DEV на своей GPU и замерьте время (CLI печатает секунды на диалог).
