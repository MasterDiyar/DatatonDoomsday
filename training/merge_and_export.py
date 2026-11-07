"""Merge the LoRA adapter into the base model and (optionally) convert it to a quantized GGUF
for llama.cpp / Ollama.

    python training/merge_and_export.py --llama-cpp ~/llama.cpp
    -> training/out/merged/            (HF fp16 weights)
    -> training/out/cqa-f16.gguf       (temporary, removed unless --keep-f16)
    -> training/out/cqa-q5_k_m.gguf    (serve this)
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", default="training/out/adapter")
    p.add_argument("--base", default=None, help="defaults to the base recorded at training time")
    p.add_argument("--out", default="training/out")
    p.add_argument("--llama-cpp", default=None, help="path to a built llama.cpp checkout (convert_hf_to_gguf.py, llama-quantize)")
    p.add_argument("--quant", default="Q5_K_M", help="llama-quantize type, e.g. Q4_K_M, Q5_K_M, Q8_0")
    p.add_argument("--keep-f16", action="store_true")
    args = p.parse_args()

    adapter = Path(args.adapter)
    meta_file = adapter / "cqa_training.json"
    base = args.base or (json.loads(meta_file.read_text(encoding="utf-8"))["base"] if meta_file.exists() else None)
    if not base:
        raise SystemExit("Pass --base (no cqa_training.json next to the adapter).")
    out = Path(args.out)
    merged = out / "merged"

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Merging {adapter} into {base} (fp16, on CPU; needs ~16 GB RAM for a 7B model)…")
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16, device_map={"": "cpu"}, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.save_pretrained(merged, safe_serialization=True)
    AutoTokenizer.from_pretrained(adapter).save_pretrained(merged)
    print(f"Merged model saved to {merged}")

    if not args.llama_cpp:
        print("Skip GGUF: pass --llama-cpp /path/to/llama.cpp to convert. vLLM can also serve the merged folder directly.")
        return
    llama = Path(args.llama_cpp).expanduser()
    convert = llama / "convert_hf_to_gguf.py"
    quantize = next((q for q in [llama / "build/bin/llama-quantize", llama / "llama-quantize", llama / "build/bin/Release/llama-quantize.exe"] if q.exists()), None)
    if not convert.exists() or quantize is None:
        raise SystemExit(f"convert_hf_to_gguf.py or llama-quantize not found under {llama}; build llama.cpp first.")
    f16 = out / "cqa-f16.gguf"
    final = out / f"cqa-{args.quant.lower()}.gguf"
    subprocess.run([sys.executable, str(convert), str(merged), "--outfile", str(f16), "--outtype", "f16"], check=True)
    subprocess.run([str(quantize), str(f16), str(final), args.quant], check=True)
    if not args.keep_f16:
        f16.unlink(missing_ok=True)
    print(f"GGUF ready: {final}")
    if shutil.which("ollama"):
        print("Next: ollama create cqa-qwen7b-lora -f training/Modelfile")


if __name__ == "__main__":
    main()
