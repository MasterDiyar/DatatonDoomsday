# DatatonDoomsday — оценка качества чатов (CQA)

Инструмент оценивает работу операторов чата по **чек-листу качества чатов Freedom Broker** (8 критериев) с помощью **локальной модели** — Qwen2.5-7B-Instruct в Ollama. На выходе — `results/dev/results.json` и `results/prod/results.json` в формате организаторов **JSON 1.0**.

- **Нужен только Python 3.10+.** Для оценки не нужны `pip install`, Node.js или `.env`.
- Настройки — в `config.json`: название команды, модель, адрес Ollama, язык обоснований.
- Промпт и обоснования — **на русском** (можно переключить на казахский или английский: `"rationale_language": "kk"` / `"en"`). Чаты на RU/KZ/EN модель читает в оригинале.
- Модель отвечает строго в нужной JSON-структуре (structured output Ollama). Код проверяет каждое значение и сам считает итоговый балл: перераспределение весов при N/A, округление до двух знаков.
- **Два вызова модели на диалог.** Основной промпт оценивает все 8 критериев. Второй, короткий вызов отвечает на узкие вопросы о фактах с дословными цитатами: приветствие и имя в первой реплике, уточняющие вопросы оператора, итог и прощание в конце. Код не ищет ключевые слова — он только проверяет, что цитаты действительно есть в диалоге, и собирает «Контакт», «Выявление потребности» и «Завершение» по формуле чек-листа (`cqa/facts.py`). Отключается настройкой `"fact_check": false`.
- Запуск можно прервать и продолжить: результат сохраняется после каждого диалога. Сбойный диалог получает `status: "error"` с исходным ID.
- Локальная страница проверки позволяет команде исправить ответы. Исправленные ответы — это разметка для оценки точности и для дообучения.

Все команды ниже — для **Windows PowerShell**. Запускайте их из папки проекта, по одной строке.

## 1. Установка (один раз)

```powershell
winget install Python.Python.3.12
winget install Ollama.Ollama
```

Закройте и снова откройте PowerShell, затем:

```powershell
ollama pull qwen2.5:7b-instruct
python run.py check
```

`check` должен закончиться строкой `OK: model answered` и показать, что модель работает на `GPU`. Если Ollama недоступна, запустите приложение Ollama (или `ollama serve` в отдельном окне).

В `config.json` укажите название команды (`team_name`).

## 2. Оценка DEV

Скачайте набор. URL организаторов вставьте как есть:

```powershell
python run.py fetch "ВСТАВЬТЕ_DEV_URL" --out datasets/dev.json
```

Если файл уже есть, просто положите его в `datasets\dev.json`. Затем:

```powershell
python run.py evaluate datasets/dev.json --split dev
```

Команда показывает балл по каждому диалогу и оставшееся время, а результат пишет в `results/dev/results.json`.

- Если запуск прервался или у части диалогов `ERROR`, запустите **ту же команду ещё раз**. Готовые диалоги сохраняются, повторно оцениваются только сбойные.
- `--force` переоценивает всё заново.
- `--votes 3` запускает модель трижды на диалог и берёт большинство (медленнее).

Публикация:

```powershell
python run.py validate results/dev/results.json --data datasets/dev.json
git add results/dev/results.json
git commit -m "DEV results"
git push
```

## 3. PROD (окно в 1 час)

```powershell
python run.py fetch "ВСТАВЬТЕ_PROD_URL" --out datasets/prod.json
python run.py evaluate datasets/prod.json --split prod
python run.py validate results/prod/results.json --data datasets/prod.json --expect 10
git add results/prod/results.json
git commit -m "PROD results"
git push
```

`--split prod` проверяет, что записей ровно 10. Заранее прогоните 10 диалогов DEV, чтобы знать время:

```powershell
python run.py evaluate datasets/dev.json --out runs/rehearsal.json --ids my10.txt
```

`my10.txt` — файл с 10 conversation_id, по одному на строку.

## 4. Страница проверки (разметка команды)

```powershell
python run.py serve
```

Откройте http://127.0.0.1:8000.

1. Выберите набор и файл с ответами модели (`results/dev/results.json`).
2. Для каждого диалога: прочитайте его, нажмите **«Заполнить из ответа модели»**, исправьте ошибки, затем **«Сохранить как проверенное»** (Ctrl+S).
3. Страница показывает нашу оценку рядом с оценкой модели. Реплики-доказательства подсвечиваются в диалоге, а ответ модели, который отличается от нашего, выделен красным.

Разметка сохраняется в `labels/<набор>.json`. Это наши эталонные ответы, а не Ground Truth организаторов.

Насколько часто модель совпадает с разметкой:

```powershell
python run.py compare results/dev/results.json labels/dev.json
```

## 5. Дообучение (та же модель, NVIDIA GPU от 12 ГБ)

Для обучения лучше использовать WSL2 (Ubuntu). В Windows тоже работает, но расходует больше видеопамяти.

1. **Проверьте разметку DEV** на странице проверки. Это главный шаг: дообученная модель не будет лучше разметки.
2. *(необязательно)* **Синтетические диалоги** для увеличения данных. Их генерирует модель, затем они размечаются; ответы, противоречащие задуманному сценарию, отбрасываются:
   ```powershell
   python run.py synth --count 300 --out datasets/synth.json
   python run.py evaluate datasets/synth.json --out runs/synth-results.json
   python run.py labels-from-results runs/synth-results.json --data datasets/synth.json --scenarios datasets/synth-scenarios.json
   ```
   Для этих шагов можно взять более сильную модель по API: добавьте `--backend openai --base-url https://... --model <id>` и задайте ключ: `$env:CQA_API_KEY="..."`.
3. **Данные для обучения.** Промпт тот же, что и при оценке. 25% реальных диалогов откладываются для проверки:
   ```powershell
   python run.py build-sft --pair datasets/dev.json labels/dev.json --pair datasets/synth.json labels/synth.json --include-unreviewed
   ```
   Без `--include-unreviewed` берётся только проверенная разметка; синтетическая разметка не проверена.
4. **Обучение:**
   ```powershell
   pip install torch --index-url https://download.pytorch.org/whl/cu124
   pip install -r training/requirements.txt
   python training/train_lora.py --dry-run
   python training/train_lora.py
   python training/merge_and_export.py --llama-cpp C:/path/to/llama.cpp
   ollama create cqa-qwen7b -f training/Modelfile
   ```
5. **Проверьте, что стало лучше.** Сравните базовую и дообученную модель на отложенных диалогах:
   ```powershell
   python run.py evaluate datasets/dev.json --out runs/base.json --ids training/data/holdout-ids.txt
   python run.py evaluate datasets/dev.json --out runs/ft.json --ids training/data/holdout-ids.txt --model cqa-qwen7b
   python run.py compare runs/base.json labels/dev.json --ids training/data/holdout-ids.txt
   python run.py compare runs/ft.json labels/dev.json --ids training/data/holdout-ids.txt
   ```
   Используйте лучшую. Пропишите её в `config.json` (`model`, `submission_model_name`), чтобы в `results.json` была указана реально использованная модель.

## Как считается оценка

| Код | Критерий | Значения | Вес |
| --- | --- | --- | ---: |
| contact | 01 Установление контакта | true / false | 10 |
| discovery | 02 Выявление потребности | true / false / null | 15 |
| identification | 03 Идентификация | true / false / null | 20 |
| consultation | 04 Корректная и полная консультация | 0, 5, 10, 15, 20 / null | 20 |
| empathy | 05 Эмпатия и клиентоориентированность | true / false | 15 |
| dialogue | 06 Управление диалогом, скорость и грамотность | 0, 4, 7, 10 | 15 |
| closure | 07 Завершение | true / false / null | 5 |
| critical | 08 Грубые ошибки (флаг, без баллов) | true / false | — |

- Формулировки и пошаговые правила для модели — в `cqa/checklist.py`, промпт — в `cqa/prompt.py`, проверка фактов — в `cqa/facts.py`.
- `final_score` = 100 × набранные баллы / вес применимых критериев. Критерии с N/A исключаются, их вес перераспределяется.
- `critical_flag` = значение `critical`; балл не уменьшает.
- `evidence_turns` нумеруются с 1 по всему транскрипту (включая клиента, бота и системные сообщения).

## Файлы

```
run.py            командная строка (check, fetch, evaluate, validate, serve, compare, synth, build-sft)
config.json       команда, модель, адрес Ollama, язык обоснований, проверка фактов
cqa/              чек-лист, промпт, клиент модели, проверка фактов, подсчёт балла, формат результатов, сервер проверки
web/index.html    страница проверки
training/         дообучение QLoRA, экспорт в GGUF, Modelfile для Ollama
tests/            python -m unittest discover tests
datasets/         входные данные (в Git только вымышленный sample.json)
results/          dev/results.json и prod/results.json для организаторов
labels/, runs/    разметка команды и журналы запусков (в Git не попадают)
```

Попробовать без реальных данных: `python run.py evaluate datasets/sample.json --out runs/sample.json`.
