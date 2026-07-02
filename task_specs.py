"""任务规范与模板。

这部分把两类安全任务统一成可训练、可评估的标准格式。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import json


TASK_AUDIT = "audit"
TASK_TELEGRAM = "telegram"
TASK_GENERIC = "generic"

TASK_CHOICES = {TASK_AUDIT, TASK_TELEGRAM, TASK_GENERIC}


@dataclass(frozen=True)
class TaskExample:
    """标准化后的训练样本。"""

    task: str
    messages: list[dict]


def canonical_audit_system_prompt() -> str:
    """防御审计任务的固定系统提示。"""

    return (
        "你是一个授权环境中的防御安全审计助手。"
        "只基于用户提供的 OpenAPI、网关配置、HTTP 响应样本、日志或审计报告进行分析。"
        "不要提供任何利用步骤、payload、绕过方法、扫描技巧或攻击性命令。"
        "输出必须包含：发现项、风险等级、证据、修复建议。"
        "如果输入信息不足，要明确说明缺失项并给出下一步补充建议。"
    )


def canonical_telegram_system_prompt() -> str:
    """Telegram 分类任务的固定系统提示。"""

    return (
        "你是一个离线消息分类助手，只能基于用户自己的 Telegram Bot 数据进行分类。"
        "不要尝试抓取、扫描、关联或还原任何外部账号信息。"
        "输出必须是严格 JSON，包含 label、confidence、reason。"
        "label 只能从给定标签中选择。"
    )


def normalize_task_name(value: str | None) -> str:
    """统一任务名，避免标签混乱。"""

    if not value:
        return TASK_GENERIC

    normalized = str(value).strip().lower()
    if normalized in {"aud", "audit", "security", "defense"}:
        return TASK_AUDIT
    if normalized in {"tg", "telegram", "bot", "message"}:
        return TASK_TELEGRAM
    return normalized if normalized in TASK_CHOICES else TASK_GENERIC


def normalize_label_list(labels: Iterable[str]) -> list[str]:
    """去重并保持顺序。"""

    seen: set[str] = set()
    result: list[str] = []
    for label in labels:
        item = str(label).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def infer_task_from_record(record: dict) -> str:
    """从样本内容推断任务类型。"""

    task = normalize_task_name(record.get("task"))
    if task != TASK_GENERIC:
        return task

    messages = record.get("messages")
    if isinstance(messages, list):
        system_text = " ".join(str(item.get("content", "")).lower() for item in messages if isinstance(item, dict) and item.get("role") == "system")
        assistant_text = next(
            (
                str(item.get("content", ""))
                for item in reversed(messages)
                if isinstance(item, dict) and item.get("role") == "assistant"
            ),
            "",
        ).strip()

        if "审计" in system_text or "audit" in system_text or "网关" in system_text:
            return TASK_AUDIT
        if "telegram" in system_text or "分类" in system_text or "消息" in system_text:
            return TASK_TELEGRAM

        try:
            parsed = json.loads(assistant_text)
            if isinstance(parsed, dict):
                if "findings" in parsed or "severity" in parsed:
                    return TASK_AUDIT
                if "label" in parsed and "confidence" in parsed:
                    return TASK_TELEGRAM
        except Exception:
            pass

    label = str(record.get("label", "")).strip()
    if label:
        return TASK_TELEGRAM

    if "asset_name" in record or "audit" in record.get("metadata", {}):
        return TASK_AUDIT

    return TASK_GENERIC
