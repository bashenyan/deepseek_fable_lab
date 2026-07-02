"""安全任务模板。

这里的任务只面向授权的防御审计和自有数据分类，不包含扫描、利用或绕过。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Finding:
    """单条审计发现。"""

    title: str
    severity: str
    evidence: str
    recommendation: str


def render_defensive_audit_prompt(artifacts: str) -> str:
    """生成防御审计提示词。"""

    return (
        "你是一个授权环境中的防御安全审计助手。"
        "只基于用户提供的配置、响应样本、日志或 OpenAPI 文档进行分析。"
        "不要提供任何利用步骤、payload、绕过方法、扫描策略或攻击性命令。"
        "输出应包含：发现项、风险等级、证据、修复建议。"
        "\n\n"
        f"待审计内容：\n{artifacts.strip()}\n"
    )


def render_telegram_classification_prompt(message_text: str, labels: Sequence[str]) -> str:
    """生成 Telegram 消息分类提示词。"""

    label_text = ", ".join(labels)
    return (
        "你是一个离线消息分类助手，只能基于用户自己的 Telegram Bot 数据进行分类。"
        "不要尝试抓取、扫描、关联或还原任何外部账号信息。"
        f"请从以下标签中选择最合适的一项：{label_text}。"
        "输出格式为 JSON，包含 label、confidence、reason。"
        "\n\n"
        f"消息内容：\n{message_text.strip()}\n"
    )


def normalize_labels(labels: Iterable[str]) -> list[str]:
    """清理分类标签，保持稳定顺序。"""

    cleaned: list[str] = []
    for label in labels:
        item = label.strip()
        if item and item not in cleaned:
            cleaned.append(item)
    if not cleaned:
        raise ValueError("labels 不能为空")
    return cleaned
