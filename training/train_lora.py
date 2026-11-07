"""QLoRA fine-tuning of Qwen2.5-7B-Instruct as the CQA evaluator (single NVIDIA GPU, >=12 GB).

Input: training/data/{train,eval}.jsonl from `python run.py build-sft`. Each row has
"messages" = [system, user, assistant] exactly as the evaluator sends them; loss is on the answer only.

    python training/train_lora.py --dry-run     # token lengths only, no GPU needed
    python training/train_lora.py               # adapter -> training/out/adapter
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--train", default="training/data/train.jsonl")
    p.add_argument("--eval", default="training/data/eval.jsonl")
    p.add_argument("--out", default="training/out")
    p.add_argument("--max-len", type=int, default=8192, help="longer examples are skipped, never truncated")
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def read_jsonl(path):
    f = Path(path)
    return [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()] if f.exists() else []


def encode(tok, rows, max_len, name):
    out, lengths = [], []
    for row in rows:
        msgs = row["messages"]
        prompt_ids = tok.apply_chat_template(msgs[:-1], add_generation_prompt=True, tokenize=True)
        full_ids = tok.apply_chat_template(msgs, tokenize=True)
        if full_ids[: len(prompt_ids)] != prompt_ids:
            sys.exit("The chat template does not prefix-match the generation prompt; use another base model.")
        lengths.append(len(full_ids))
        if len(full_ids) > max_len:
            print(f"[{name}] {row.get('conversation_id')}: {len(full_ids)} tokens > --max-len; skipped")
            continue
        out.append({"input_ids": full_ids, "labels": [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]})
    if lengths:
        s = sorted(lengths)
        print(f"[{name}] rows {len(rows)}, kept {len(out)}; tokens p50={s[len(s) // 2]} max={s[-1]}")
    return out


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        import torch
        w = max(len(x["input_ids"]) for x in batch)
        ids = torch.full((len(batch), w), self.pad_id, dtype=torch.long)
        labels = torch.full((len(batch), w), -100, dtype=torch.long)
        mask = torch.zeros((len(batch), w), dtype=torch.long)
        for i, x in enumerate(batch):
            n = len(x["input_ids"])
            ids[i, :n] = torch.tensor(x["input_ids"])
            labels[i, :n] = torch.tensor(x["labels"])
            mask[i, :n] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}


def main():
    args = parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    train_rows, eval_rows = read_jsonl(args.train), read_jsonl(args.eval)
    if not train_rows:
        sys.exit(f"No rows in {args.train}. Run: python run.py build-sft ...")
    train = encode(tok, train_rows, args.max_len, "train")
    evals = encode(tok, eval_rows, args.max_len, "eval")
    if args.dry_run:
        return

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments, set_seed

    if not torch.cuda.is_available():
        sys.exit("No CUDA GPU found (check `nvidia-smi` and that torch was installed with CUDA).")
    set_seed(args.seed)
    bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=dtype, device_map={"": 0}, attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype))
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    try:
        import liger_kernel  # noqa: F401  (Linux/WSL only) avoids full-vocabulary logits -> less VRAM
        use_liger = True
    except ImportError:
        use_liger = False
        print("liger-kernel not available: if you run out of GPU memory, lower --max-len (e.g. 6144).")

    out = Path(args.out)
    steps = math.ceil(len(train) / args.grad_accum)
    targs = TrainingArguments(
        output_dir=str(out / "checkpoints"), per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum, num_train_epochs=args.epochs, learning_rate=args.lr,
        lr_scheduler_type="cosine", warmup_ratio=0.05, bf16=bf16, fp16=not bf16, optim="paged_adamw_8bit",
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=max(1, steps // 10), eval_strategy="epoch" if evals else "no", save_strategy="epoch",
        save_total_limit=2, load_best_model_at_end=bool(evals), metric_for_best_model="eval_loss" if evals else None,
        greater_is_better=False if evals else None, report_to="none", remove_unused_columns=False,
        use_liger_kernel=use_liger, seed=args.seed, dataloader_num_workers=0)
    trainer = Trainer(model=model, args=targs, train_dataset=Dataset.from_list(train),
                      eval_dataset=Dataset.from_list(evals) if evals else None, data_collator=Collator(tok.pad_token_id))
    trainer.train()
    adapter = out / "adapter"
    trainer.model.save_pretrained(adapter)
    tok.save_pretrained(adapter)
    (adapter / "cqa_training.json").write_text(json.dumps({"base": args.base, "train_rows": len(train),
                                                           "eval_rows": len(evals), "log": trainer.state.log_history},
                                                          indent=2), encoding="utf-8")
    print(f"Adapter saved to {adapter}. Next: python training/merge_and_export.py --llama-cpp <path>")


if __name__ == "__main__":
    main()
