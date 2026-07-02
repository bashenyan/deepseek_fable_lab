"""训练结果评估。

这个脚本用于两个目标：
- Telegram 分类：计算准确率
- 防御审计：检查 JSON 结构完整性和字段覆盖率

输入格式与训练数据一致，支持 `messages` 或结构化记录。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估训练结果")
    parser.add_argument("--input", required=True, help="评估样本 JSONL")
    parser.add_argument("--task", choices=["telegram", "audit"], required=True)
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _extract_assistant_text(messages: list[dict]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return str(message.get("content", "")).strip()
    return ""


def _eval_telegram(records: list[dict]) -> dict:
    total = 0
    correct = 0
    label_counter = Counter()
    json_errors = 0

    for record in records:
        messages = record.get("messages", [])
        answer = _extract_assistant_text(messages)
        gold = record.get("label")
        if not gold:
            try:
                parsed = json.loads(answer)
                gold = parsed.get("label")
            except Exception:
                gold = None
        if not gold:
            continue

        total += 1
        try:
            parsed = json.loads(answer)
            pred = str(parsed.get("label", "")).strip()
        except Exception:
            json_errors += 1
            pred = ""

        label_counter[gold] += 1
        if pred == gold:
            correct += 1

    return {
        "task": "telegram",
        "total": total,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "json_errors": json_errors,
        "label_distribution": dict(label_counter),
    }


def _eval_audit(records: list[dict]) -> dict:
    total = 0
    valid_json = 0
    missing_fields = Counter()

    for record in records:
        messages = record.get("messages", [])
        answer = _extract_assistant_text(messages)
        if not answer:
            continue

        total += 1
        try:
            parsed = json.loads(answer)
        except Exception:
            continue

        valid_json += 1
        findings = parsed.get("findings", [])
        if not isinstance(findings, list):
            missing_fields["findings_type"] += 1
            continue

        for field in ("summary", "findings"):
            if field not in parsed:
                missing_fields[field] += 1
        for finding in findings:
            if not isinstance(finding, dict):
                missing_fields["finding_object"] += 1
                continue
            for field in ("title", "severity", "evidence", "recommendation"):
                if field not in finding:
                    missing_fields[field] += 1

    return {
        "task": "audit",
        "total": total,
        "json_rate": round(valid_json / total, 4) if total else 0.0,
        "missing_fields": dict(missing_fields),
    }


def main() -> None:
    args = parse_args()
    records = _read_jsonl(Path(args.input))
    if args.task == "telegram":
        result = _eval_telegram(records)
    else:
        result = _eval_audit(records)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
