"""把原始数据整理成可训练的标准 JSONL。

支持：
- `messages` 标准对话格式
- `prompt/completion`
- 防御审计的结构化样本
- Telegram 分类的结构化样本

输出会统一成：
{"task":"audit|telegram|generic","messages":[...]}
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

from .task_specs import TASK_CHOICES, infer_task_from_record, normalize_task_name
from .template import infer_and_render_task_prompt
from .template import validate_messages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="准备 SFT 数据")
    parser.add_argument("--input", required=True, help="输入 JSON/JSONL 文件")
    parser.add_argument("--output", required=True, help="输出 JSONL 文件")
    parser.add_argument("--eval-output", default=None, help="可选：验证集输出 JSONL")
    parser.add_argument("--eval-ratio", type=float, default=0.02, help="自动切分验证集比例")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", default=None, choices=sorted(TASK_CHOICES | {"auto"}), help="强制任务类型")
    parser.add_argument("--min-assistant-chars", type=int, default=8, help="过滤过短回答")
    parser.add_argument("--dedupe", action="store_true", help="按标准化消息去重")
    return parser.parse_args()


def _read_records(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix not in {".json", ".jsonl"}:
        raise ValueError("仅支持 json 或 jsonl 文件")

    if suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if isinstance(payload.get("data"), list):
            return payload["data"]
        if isinstance(payload.get("examples"), list):
            return payload["examples"]
        return [payload]
    if isinstance(payload, list):
        return payload
    raise ValueError("无法识别输入 JSON 结构")


def _dedupe_key(messages: list[dict]) -> str:
    return json.dumps(messages, ensure_ascii=False, sort_keys=True)


def _normalize_record(record: dict, forced_task: str | None) -> dict | None:
    task = normalize_task_name(forced_task) if forced_task and forced_task != "auto" else infer_task_from_record(record)

    if "messages" in record:
        messages = record["messages"]
    else:
        messages = infer_and_render_task_prompt(record)

    if not messages:
        return None

    validate_messages(messages)
    assistant_parts = [msg["content"].strip() for msg in messages if msg["role"] == "assistant"]
    assistant_text = "\n".join(part for part in assistant_parts if part)
    if len(assistant_text) < 8:
        return None

    return {
        "task": task,
        "messages": messages,
        "source": record.get("source"),
    }


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    eval_output_path = Path(args.eval_output) if args.eval_output else None

    random.seed(args.seed)
    raw_records = _read_records(input_path)

    normalized: list[dict] = []
    seen: set[str] = set()
    task_counter: Counter[str] = Counter()

    for record in raw_records:
        if not isinstance(record, dict):
            continue
        cleaned = _normalize_record(record, args.task)
        if cleaned is None:
            continue

        key = _dedupe_key(cleaned["messages"])
        if args.dedupe and key in seen:
            continue
        seen.add(key)

        task_counter[cleaned["task"]] += 1
        normalized.append(cleaned)

    if not normalized:
        raise ValueError("没有得到可用样本")

    random.shuffle(normalized)

    if eval_output_path is not None:
        split_index = max(1, int(len(normalized) * (1 - max(0.0, min(args.eval_ratio, 0.9)))))
        train_rows = normalized[:split_index]
        eval_rows = normalized[split_index:]
        _write_jsonl(output_path, train_rows)
        _write_jsonl(eval_output_path, eval_rows)
    else:
        _write_jsonl(output_path, normalized)

    stats = {
        "total": len(normalized),
        "task_counts": dict(task_counter),
        "train_count": len(normalized) if eval_output_path is None else split_index,
        "eval_count": 0 if eval_output_path is None else len(normalized) - split_index,
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
