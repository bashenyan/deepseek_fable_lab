"""训练批处理器。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class DataCollatorForCausalLMSFT:
    """对齐输入长度，并用 -100 填充 labels。"""

    tokenizer: Any
    pad_to_multiple_of: int | None = 8

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        labels = [feature.pop("labels") for feature in features]
        batch = self.tokenizer.pad(
            features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        max_len = batch["input_ids"].shape[1]
        padded_labels = []
        for label in labels:
            pad_len = max_len - len(label)
            padded_labels.append(label + [-100] * pad_len)

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch
