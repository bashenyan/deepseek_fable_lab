"""OpenAI 兼容的本地服务封装。"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Literal

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import DEFAULT_BASE_MODEL
from .auth import load_auth_config, require_api_key
from .runtime import pick_input_device, resolve_device, resolve_dtype
from .template import render_for_inference

try:
    from fastapi import Depends, FastAPI
    from pydantic import BaseModel, Field
    import uvicorn
except ModuleNotFoundError:  # pragma: no cover - 运行环境未安装可选依赖时回退
    FastAPI = None  # type: ignore[assignment]
    BaseModel = object  # type: ignore[assignment]
    Field = None  # type: ignore[assignment]
    uvicorn = None  # type: ignore[assignment]


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    stop: list[str] | str | None = None


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[Message]
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    stop: list[str] | str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="启动 DeepSeek-V4-Fable 推理服务")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--model-path", default=None, help="已合并模型目录")
    parser.add_argument("--lora-path", default=None, help="LoRA 适配器目录")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-input-length", type=int, default=8192)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--api-key", default=None, help="显式传入 API Key；不传则读取环境变量 DEEPSEEK_FABLE_API_KEY")
    parser.add_argument("--api-key-env", default="DEEPSEEK_FABLE_API_KEY")
    return parser.parse_args()


def load_model(args: argparse.Namespace):
    resolved_device = resolve_device(args.device)
    resolved_dtype = resolve_dtype(args.dtype, resolved_device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path or args.base_model,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path or args.base_model,
        trust_remote_code=True,
        device_map="auto" if resolved_device == "cuda" else None,
        torch_dtype=resolved_dtype,
    )
    if args.lora_path:
        model = PeftModel.from_pretrained(model, args.lora_path)
    if resolved_device != "cuda":
        model = model.to(resolved_device)
    model.eval()
    return tokenizer, model


def build_prompt(messages: list[dict[str, str]], tokenizer) -> str:
    """优先使用模型自带模板，失败时回退到通用模板。"""

    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return render_for_inference(messages) + "\n### 助手\n"


def create_app(tokenizer, model, max_input_length: int) -> FastAPI:
    if FastAPI is None:
        raise ModuleNotFoundError("请先安装 fastapi、pydantic 和 uvicorn 以启动推理服务")
    app = FastAPI(title="DeepSeek-V4-Fable Service", version="0.1.0")
    auth = load_auth_config()
    api_key_dependency = require_api_key(auth)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models", dependencies=[Depends(api_key_dependency)] if api_key_dependency else [])
    def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": "local-model",
                    "object": "model",
                    "owned_by": "local",
                }
            ],
        }

    def generate(prompt: str, max_tokens: int, temperature: float, top_p: float) -> str:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length,
        ).to(model.device)

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated = output_ids[0][inputs["input_ids"].shape[1] :]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()

    @app.post("/v1/completions", dependencies=[Depends(api_key_dependency)] if api_key_dependency else [])
    def completions(req: CompletionRequest) -> dict[str, Any]:
        text = generate(req.prompt, req.max_tokens, req.temperature, req.top_p)
        now = int(time.time())
        return {
            "id": f"cmpl-{now}",
            "object": "text_completion",
            "created": now,
            "model": req.model or "local-model",
            "choices": [
                {
                    "index": 0,
                    "text": text,
                    "finish_reason": "stop",
                }
            ],
        }

    @app.post("/v1/chat/completions", dependencies=[Depends(api_key_dependency)] if api_key_dependency else [])
    def chat_completions(req: ChatCompletionRequest) -> dict[str, Any]:
        prompt = build_prompt([message.model_dump() for message in req.messages], tokenizer)
        text = generate(prompt, req.max_tokens, req.temperature, req.top_p)
        now = int(time.time())
        return {
            "id": f"chatcmpl-{now}",
            "object": "chat.completion",
            "created": now,
            "model": req.model or "local-model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": text,
                    },
                    "finish_reason": "stop",
                }
            ],
        }

    return app


def main() -> None:
    args = parse_args()
    os.environ.setdefault(args.api_key_env, args.api_key or "")
    tokenizer, model = load_model(args)
    app = create_app(tokenizer, model, args.max_input_length)
    if uvicorn is None:
        raise ModuleNotFoundError("请先安装 fastapi、pydantic 和 uvicorn 以启动推理服务")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
