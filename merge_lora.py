"""合并 LoRA 权重并导出为独立模型目录。"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import DEFAULT_BASE_MODEL, DEFAULT_MERGED_DIR
from .runtime import resolve_device, resolve_dtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="合并 LoRA 权重")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--lora-dir", required=True)
    parser.add_argument("--output-dir", default=DEFAULT_MERGED_DIR)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolved_device = resolve_device(args.device)
    resolved_dtype = resolve_dtype(args.dtype, resolved_device)
    tokenizer = AutoTokenizer.from_pretrained(args.lora_dir, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=resolved_dtype,
        device_map="auto" if resolved_device == "cuda" else None,
    )
    merged = PeftModel.from_pretrained(base_model, args.lora_dir)
    merged = merged.merge_and_unload()
    if resolved_device != "cuda":
        merged = merged.to(resolved_device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
