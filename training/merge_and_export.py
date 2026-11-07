"""Merge the LoRA adapter into the base model and convert it to a quantized GGUF for Ollama.

    python training/merge_and_export.py --llama-cpp C:/tools/llama.cpp
    -> training/out/merged/  (HF fp16)   -> training/out/cqa-q5_k_m.gguf   (load into Ollama)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", default="training/out/adapter")
    p.add_argument("--base", help="defaults to the base model recorded during training")
    p.add_argument("--out", default="training/out")
    p.add_argument("--llama-cpp", help="llama.cpp folder containing convert_hf_to_gguf.py and llama-quantize")
    p.add_argument("--quant", default="Q5_K_M")
    args = p.parse_args()

    adapter, out = Path(args.adapter), Path(args.out)
    meta = adapter / "cqa_training.json"
    base = args.base or (json.loads(meta.read_text(encoding="utf-8"))["base"] if meta.exists() else None)
    if not base:
        sys.exit("Pass --base")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Merging {adapter} into {base} on CPU (needs ~16 GB RAM)...")
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16, device_map={"": "cpu"},
                                                 low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    merged = out / "merged"
    model.save_pretrained(merged, safe_serialization=True)
    AutoTokenizer.from_pretrained(adapter).save_pretrained(merged)
    print(f"Merged model: {merged}")
    if not args.llama_cpp:
        print("Add --llama-cpp <folder> to produce the GGUF file for Ollama.")
        return
    llama = Path(args.llama_cpp).expanduser()
    convert = llama / "convert_hf_to_gguf.py"
    candidates = ["llama-quantize", "llama-quantize.exe", "build/bin/llama-quantize", "build/bin/Release/llama-quantize.exe"]
    quantize = next((llama / c for c in candidates if (llama / c).exists()), None)
    if not convert.exists() or quantize is None:
        sys.exit(f"convert_hf_to_gguf.py / llama-quantize not found in {llama}")
    f16, final = out / "cqa-f16.gguf", out / f"cqa-{args.quant.lower()}.gguf"
    subprocess.run([sys.executable, str(convert), str(merged), "--outfile", str(f16), "--outtype", "f16"], check=True)
    subprocess.run([str(quantize), str(f16), str(final), args.quant], check=True)
    f16.unlink(missing_ok=True)
    print(f"GGUF: {final}\nNext: ollama create cqa-qwen7b -f training/Modelfile")


if __name__ == "__main__":
    main()
