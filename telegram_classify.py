"""Telegram 消息离线分类入口。

用途：
- 对你自己拥有权限的 Telegram Bot 消息做标签归类
- 支持客服、告警、订单、验证码、广告、垃圾、异常等标签
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .auth import load_auth_config, require_api_key
from .safety_tasks import normalize_labels, render_telegram_classification_prompt

try:
    from fastapi import Depends, FastAPI
    from pydantic import BaseModel, Field
    import uvicorn
except ModuleNotFoundError:  # pragma: no cover - 运行环境未安装可选依赖时回退
    FastAPI = None  # type: ignore[assignment]
    BaseModel = object  # type: ignore[assignment]
    Field = None  # type: ignore[assignment]
    uvicorn = None  # type: ignore[assignment]


DEFAULT_LABELS = [
    "客服",
    "告警",
    "订单",
    "验证码",
    "广告",
    "垃圾",
    "异常",
    "其他",
]


@dataclass
class ClassifyResponse:
    label: str
    confidence: float
    reason: str


if BaseModel is not object:

    class ClassifyRequest(BaseModel):
        text: str
        labels: list[str] = Field(default_factory=lambda: DEFAULT_LABELS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="启动 Telegram 分类服务")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument("--demo-file", default=None, help="可选的 JSONL 消息样本")
    parser.add_argument("--api-key", default=None, help="显式传入 API Key")
    parser.add_argument("--api-key-env", default="DEEPSEEK_FABLE_API_KEY")
    return parser.parse_args()


def local_classify(text: str, labels: list[str]) -> ClassifyResponse:
    """轻量规则分类，便于先把数据和服务链路跑通。"""

    normalized = normalize_labels(labels)
    lowered = text.lower()

    score_map = Counter()
    if any(token in lowered for token in ["验证码", "code", "otp", "verify"]):
        score_map["验证码"] += 3
    if any(token in lowered for token in ["订单", "order", "payment", "paid"]):
        score_map["订单"] += 3
    if any(token in lowered for token in ["告警", "alarm", "alert", "error", "failed"]):
        score_map["告警"] += 3
    if any(token in lowered for token in ["客服", "support", "help", "ticket"]):
        score_map["客服"] += 3
    if any(token in lowered for token in ["buy now", "discount", "promo", "广告", "推广"]):
        score_map["广告"] += 3
    if any(token in lowered for token in ["spam", "垃圾", "bot"]):
        score_map["垃圾"] += 3

    if score_map:
        label = score_map.most_common(1)[0][0]
        confidence = min(0.95, 0.55 + 0.1 * score_map[label])
        reason = "根据关键词与消息语气进行初步分类。"
    else:
        label = "其他" if "其他" in normalized else normalized[0]
        confidence = 0.5
        reason = "未命中明确规则，归入通用类别。"

    if label not in normalized:
        label = normalized[0]

    return ClassifyResponse(label=label, confidence=confidence, reason=reason)


def create_app() -> FastAPI:
    if FastAPI is None:
        raise ModuleNotFoundError("请先安装 fastapi、pydantic 和 uvicorn 以启动分类服务")
    app = FastAPI(title="Telegram Classification API", version="0.1.0")
    auth = load_auth_config()
    api_key_dependency = require_api_key(auth)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/classify", response_model=ClassifyResponse, dependencies=[Depends(api_key_dependency)] if api_key_dependency else [])
    def classify(req: ClassifyRequest) -> ClassifyResponse:
        _prompt = render_telegram_classification_prompt(req.text, req.labels)
        # 当前实现采用本地规则，后续可替换为你训练后的分类模型。
        return local_classify(req.text, req.labels)

    return app


def main() -> None:
    args = parse_args()
    if args.api_key:
        import os

        os.environ.setdefault(args.api_key_env, args.api_key)
    app = create_app()

    if args.demo_file:
        records = [
            json.loads(line)
            for line in Path(args.demo_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        result = [asdict(local_classify(str(item.get("text", "")), DEFAULT_LABELS)) for item in records]
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if uvicorn is None:
        raise ModuleNotFoundError("请先安装 fastapi、pydantic 和 uvicorn 以启动分类服务")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
