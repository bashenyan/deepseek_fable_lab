"""数据集读取与特征构建。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from datasets import Dataset

from .template import AssistantSpan, render_fallback_chat, validate_messages


@dataclass
class TrainingExample:
    """单条训练样本。"""

    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]


def _mask_labels_from_spans(offsets: Sequence[tuple[int, int]], spans: Sequence[AssistantSpan], labels: list[int]) -> None:
    """只保留 assistant 内容的监督信号。"""

    for token_index, (start, end) in enumerate(offsets):
        if start == end == 0:
            labels[token_index] = -100
            continue

        keep = any(not (end <= span.start or start >= span.end) for span in spans)
        if not keep:
            labels[token_index] = -100


def tokenize_messages(tokenizer, messages: Sequence[dict], max_length: int) -> TrainingExample:
    """把对话样本转成可训练的因果语言模型特征。"""

    validate_messages(messages)
    text, spans = render_fallback_chat(messages)
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    labels = list(input_ids)
    _mask_labels_from_spans(encoded["offset_mapping"], spans, labels)

    return TrainingExample(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )


def build_dataset(tokenizer, raw_dataset: Dataset, max_length: int) -> Dataset:
    """把原始 JSON 数据集映射成训练特征。"""

    def _map(record: dict) -> dict:
        task = record.get("task", "generic")
        messages = record.get("messages")
        if messages is None and "prompt" in record and "completion" in record:
            messages = [
                {"role": "user", "content": str(record["prompt"])},
                {"role": "assistant", "content": str(record["completion"])},
            ]
        if messages is None:
            raise ValueError("样本必须包含 messages 或 prompt/completion")

        example = tokenize_messages(tokenizer, messages, max_length=max_length)
        return {
            "task": task,
            "input_ids": example.input_ids,
            "attention_mask": example.attention_mask,
            "labels": example.labels,
        }

    columns_to_remove = [name for name in raw_dataset.column_names if name not in {"task"}]
    return raw_dataset.map(_map, remove_columns=columns_to_remove)
