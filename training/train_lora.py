"""QLoRA fine-tuning of an open instruct model as a local CQA evaluator.

Input: datasets/sft/{train,eval}.jsonl produced by scripts/build-sft-dataset.mjs.
Each row has "messages" = [system, user, assistant] in exactly the format the
evaluator sends at inference time. Loss is computed on assistant tokens only.

    python training/train_lora.py --dry-run            # token-length report, no GPU needed
    python training/train_lora.py                      # train, adapter -> training/out/adapter
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct", help="HF model id or local path")
    p.add_argument("--train", default="datasets/sft/train.jsonl")
    p.add_argument("--eval", default="datasets/sft/eval.jsonl")
    p.add_argument("--out", default="training/out")
    p.add_argument("--max-len", type=int, default=12288, help="examples longer than this are skipped, never truncated")
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-liger", action="store_true", help="disable fused cross-entropy kernels")
    p.add_argument("--dry-run", action="store_true", help="only tokenize and report lengths")
    return p.parse_args()


def read_jsonl(path: str) -> list[dict]:
    file = Path(path)
    if not file.exists():
        return []
    return [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines() if line.strip()]


def encode(tokenizer, rows: list[dict], max_len: int, name: str) -> list[dict]:
    """Tokenize with the model's chat template and mask everything except the assistant answer."""
    out, skipped, lengths = [], 0, []
    for row in rows:
        messages = row["messages"]
        assert messages[-1]["role"] == "assistant", "last message must be the assistant target"
        prompt_ids = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, tokenize=True)
        full_ids = tokenizer.apply_chat_template(messages, tokenize=True)
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise SystemExit("Chat template does not prefix-match the generation prompt; choose another base model.")
        lengths.append(len(full_ids))
        if len(full_ids) > max_len:
            skipped += 1
            print(f"[{name}] {row.get('conversation_id')}: {len(full_ids)} tokens > --max-len {max_len}; skipped", file=sys.stderr)
            continue
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        out.append({"input_ids": full_ids, "labels": labels})
    if lengths:
        s = sorted(lengths)
        print(f"[{name}] {len(rows)} rows, kept {len(out)}, skipped {skipped}; tokens p50={s[len(s)//2]} p95={s[min(len(s)-1, int(0.95*len(s)))]} max={s[-1]}")
    return out


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        import torch
        width = max(len(x["input_ids"]) for x in batch)
        ids = torch.full((len(batch), width), self.pad_id, dtype=torch.long)
        labels = torch.full((len(batch), width), -100, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for i, x in enumerate(batch):
            n = len(x["input_ids"])
            ids[i, :n] = torch.tensor(x["input_ids"])
            labels[i, :n] = torch.tensor(x["labels"])
            mask[i, :n] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}


def main() -> None:
    args = parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_rows, eval_rows = read_jsonl(args.train), read_jsonl(args.eval)
    if not train_rows:
        raise SystemExit(f"No training rows in {args.train}. Run scripts/build-sft-dataset.mjs first.")
    train = encode(tokenizer, train_rows, args.max_len, "train")
    evals = encode(tokenizer, eval_rows, args.max_len, "eval")
    if args.dry_run:
        return

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments, set_seed

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required for training (use --dry-run to only inspect the data).")
    set_seed(args.seed)
    bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype
        ),
        torch_dtype=dtype,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))
    model.print_trainable_parameters()

    use_liger = False
    if not args.no_liger:
        try:
            import liger_kernel  # noqa: F401
            use_liger = True
        except ImportError:
            print("liger-kernel not installed: full logits will be materialised; lower --max-len if you hit OOM.", file=sys.stderr)

    steps_per_epoch = math.ceil(len(train) / args.grad_accum)
    out = Path(args.out)
    training_args = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.0,
        bf16=bf16,
        fp16=not bf16,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=max(1, steps_per_epoch // 10),
        eval_strategy="epoch" if evals else "no",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=bool(evals),
        metric_for_best_model="eval_loss" if evals else None,
        greater_is_better=False if evals else None,
        report_to="none",
        remove_unused_columns=False,
        use_liger_kernel=use_liger,
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=Dataset.from_list(train),
        eval_dataset=Dataset.from_list(evals) if evals else None,
        data_collator=Collator(tokenizer.pad_token_id),
    )
    trainer.train()
    adapter_dir = out / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    (adapter_dir / "cqa_training.json").write_text(json.dumps({
        "base": args.base, "train_rows": len(train), "eval_rows": len(evals), "max_len": args.max_len,
        "epochs": args.epochs, "lr": args.lr, "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
        "log_history": trainer.state.log_history,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Adapter saved to {adapter_dir}")


if __name__ == "__main__":
    main()
