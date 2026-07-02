"""LoRA / SFT 训练入口。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from .collator import DataCollatorForCausalLMSFT
from .config import DEFAULT_BASE_MODEL, DEFAULT_MAX_LENGTH, DEFAULT_OUTPUT_DIR, DEFAULT_SEED
from .data import build_dataset
from .runtime import resolve_device, resolve_dtype
from .task_specs import TASK_AUDIT, TASK_TELEGRAM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepSeek-V4-Fable LoRA/SFT 训练")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--train-file", required=True, help="JSON/JSONL 训练集文件")
    parser.add_argument("--eval-file", default=None, help="可选的验证集文件")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--bf16", action="store_true", help="优先使用 bf16")
    parser.add_argument("--tf32", action="store_true", help="允许 TF32")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--task", choices=["all", TASK_AUDIT, TASK_TELEGRAM], default="all", help="只训练某个任务或全部任务")
    parser.add_argument("--group-by-length", action="store_true", help="按长度分组，减少 padding")
    return parser.parse_args()


def _load_raw_dataset(train_file: str, eval_file: str | None):
    data_files = {"train": train_file}
    if eval_file:
        data_files["validation"] = eval_file
    ext = Path(train_file).suffix.lower().lstrip(".")
    if ext not in {"json", "jsonl"}:
        raise ValueError("仅支持 json 或 jsonl 文件")
    # Hugging Face 的 json 读取器同时兼容标准 JSON 和 JSONL。
    return load_dataset("json", data_files=data_files)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    resolved_device = resolve_device(args.device)
    resolved_dtype = resolve_dtype(args.dtype, resolved_device)
    if args.bf16 and args.dtype == "auto" and resolved_device != "cpu":
        resolved_dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw = _load_raw_dataset(args.train_file, args.eval_file)
    train_dataset = build_dataset(tokenizer, raw["train"], max_length=args.max_length)
    eval_dataset = None
    if "validation" in raw:
        eval_dataset = build_dataset(tokenizer, raw["validation"], max_length=args.max_length)

    if args.task != "all":
        train_dataset = train_dataset.filter(lambda item: item.get("task", args.task) == args.task)
        if eval_dataset is not None:
            eval_dataset = eval_dataset.filter(lambda item: item.get("task", args.task) == args.task)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=resolved_dtype if resolved_device != "cpu" else None,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if resolved_device != "cuda":
        model = model.to(resolved_device)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[item.strip() for item in args.target_modules.split(",") if item.strip()],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        evaluation_strategy="steps" if eval_dataset is not None else "no",
        save_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=bool(eval_dataset is not None),
        metric_for_best_model="eval_loss" if eval_dataset is not None else None,
        greater_is_better=False,
        bf16=bool(args.bf16 and torch.cuda.is_available()),
        fp16=bool(not args.bf16 and torch.cuda.is_available()),
        report_to="none",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        group_by_length=args.group_by_length,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForCausalLMSFT(tokenizer=tokenizer, pad_to_multiple_of=8 if resolved_device == "cuda" else None),
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metadata = {
        "base_model": args.base_model,
        "train_file": os.path.abspath(args.train_file),
        "eval_file": os.path.abspath(args.eval_file) if args.eval_file else None,
        "max_length": args.max_length,
        "task": args.task,
        "device": resolved_device,
        "dtype": str(resolved_dtype),
    }
    Path(args.output_dir, "train_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
