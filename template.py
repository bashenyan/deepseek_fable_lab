"""对话模板与数据格式化逻辑。

说明：
1. 优先使用 tokenizer 自带 chat template；
2. 如果模型没有模板，则使用这里的兜底模板；
3. 训练时会对 assistant 内容做监督，其余部分全部 mask。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .config import ROLE_PREFIX, ROLE_SUFFIX, TURN_SEPARATOR
from .task_specs import TASK_AUDIT, TASK_TELEGRAM, canonical_audit_system_prompt, canonical_telegram_system_prompt, infer_task_from_record


@dataclass(frozen=True)
class AssistantSpan:
    """assistant 监督区间，使用字符偏移表示。"""

    start: int
    end: int


def validate_messages(messages: Sequence[dict]) -> None:
    """校验消息结构，尽早拦截脏数据。"""

    if not messages:
        raise ValueError("messages 不能为空")

    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise TypeError(f"第 {index} 条消息必须是对象")
        if "role" not in message or "content" not in message:
            raise ValueError(f"第 {index} 条消息缺少 role/content 字段")
        if message["role"] not in ROLE_PREFIX:
            raise ValueError(f"第 {index} 条消息角色不合法: {message['role']}")
        if not isinstance(message["content"], str):
            raise TypeError(f"第 {index} 条消息 content 必须是字符串")


def render_fallback_chat(messages: Sequence[dict]) -> tuple[str, list[AssistantSpan]]:
    """把消息渲染成通用训练文本，并返回 assistant 监督区间。

    这个模板不依赖特殊 tokenizer 词表，适合做通用 LoRA/SFT 基线。
    """

    validate_messages(messages)
    parts: list[str] = []
    spans: list[AssistantSpan] = []
    cursor = 0

    for message in messages:
        prefix = ROLE_PREFIX[message["role"]]
        content = message["content"].strip()
        segment = f"{prefix}{content}{ROLE_SUFFIX}"

        if message["role"] == "assistant":
            start = cursor + len(prefix)
            end = start + len(content)
            spans.append(AssistantSpan(start=start, end=end))

        parts.append(segment)
        cursor += len(segment)
        parts.append(TURN_SEPARATOR)
        cursor += len(TURN_SEPARATOR)

    return "".join(parts).strip(), spans


def render_for_inference(messages: Sequence[dict]) -> str:
    """生成推理 prompt。"""

    prompt, _ = render_fallback_chat(messages)
    return prompt


def render_task_messages(task: str, user_content: str, assistant_content: str, extra_system: str | None = None) -> list[dict]:
    """生成标准任务对话。

    这里确保训练和推理使用一致的任务边界，减少 prompt 漂移。
    """

    if task == TASK_AUDIT:
        system_prompt = canonical_audit_system_prompt()
    elif task == TASK_TELEGRAM:
        system_prompt = canonical_telegram_system_prompt()
    else:
        system_prompt = extra_system or "你是一个严格遵循用户指令的助手。"

    if extra_system and task == TASK_AUDIT:
        system_prompt = f"{system_prompt} {extra_system.strip()}"

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content.strip()},
        {"role": "assistant", "content": assistant_content.strip()},
    ]


def infer_and_render_task_prompt(record: dict) -> list[dict]:
    """从原始记录里推断任务并返回标准消息。"""

    task = infer_task_from_record(record)
    if "messages" in record:
        return list(record["messages"])

    if task == TASK_AUDIT:
        user_content = str(record.get("content") or record.get("input") or record.get("prompt") or "").strip()
        assistant_content = str(record.get("response") or record.get("completion") or record.get("answer") or "").strip()
        return render_task_messages(task, user_content, assistant_content)

    if task == TASK_TELEGRAM:
        user_content = str(record.get("text") or record.get("message") or record.get("content") or "").strip()
        label = str(record.get("label") or "").strip()
        confidence = record.get("confidence", 0.9)
        reason = str(record.get("reason") or record.get("explanation") or "根据消息内容进行离线分类。").strip()
        assistant_content = f'{{"label":"{label}","confidence":{confidence},"reason":"{reason}"}}'
        return render_task_messages(task, user_content, assistant_content)

    return list(record.get("messages", []))


def iter_jsonl_messages(records: Iterable[dict]) -> Iterable[list[dict]]:
    """从通用样本中提取 messages，兼容常见 JSONL 数据集格式。"""

    for record in records:
        if "messages" in record:
            yield record["messages"]
            continue
        if "prompt" in record and "completion" in record:
            yield [
                {"role": "user", "content": str(record["prompt"])},
                {"role": "assistant", "content": str(record["completion"])},
            ]
            continue
        raise ValueError("样本必须包含 messages 或 prompt/completion")
