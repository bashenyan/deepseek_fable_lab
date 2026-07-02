"""授权资产的防御审计入口。

功能定位：
- 输入 OpenAPI / JSON 配置 / HTTP 响应头 / 网关规则
- 输出风险项和修复建议
- 不输出利用步骤、扫描技巧或攻击 payload
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .auth import load_auth_config, require_api_key
from .safety_tasks import render_defensive_audit_prompt

try:
    from fastapi import Depends, FastAPI
    from pydantic import BaseModel, Field
    import uvicorn
except ModuleNotFoundError:  # pragma: no cover - 运行环境未安装可选依赖时回退
    FastAPI = None  # type: ignore[assignment]
    BaseModel = object  # type: ignore[assignment]
    Field = None  # type: ignore[assignment]
    uvicorn = None  # type: ignore[assignment]


@dataclass
class AuditFinding:
    title: str
    severity: str
    evidence: str
    recommendation: str


@dataclass
class AuditResponse:
    asset_name: str
    summary: str
    findings: list[AuditFinding]


if BaseModel is not object:

    class AuditRequest(BaseModel):
        asset_name: str = Field(default="unknown")
        content: str = Field(..., description="OpenAPI、配置、响应头或审计日志")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="启动防御审计服务")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--demo-file", default=None, help="可选的审计样本文件")
    parser.add_argument("--api-key", default=None, help="显式传入 API Key")
    parser.add_argument("--api-key-env", default="DEEPSEEK_FABLE_API_KEY")
    return parser.parse_args()


def local_audit(content: str) -> AuditResponse:
    """轻量本地规则审计，便于没有模型时先跑通流程。"""

    findings: list[AuditFinding] = []
    lowered = content.lower()

    if "access-control-allow-origin: *" in lowered or '"alloworigin": "*"' in lowered:
        findings.append(
            AuditFinding(
                title="CORS 过于宽松",
                severity="medium",
                evidence="发现通配符跨域配置",
                recommendation="限制为明确可信域名，并配合凭证策略和预检缓存控制。",
            )
        )

    if "authorization" not in lowered and "api key" not in lowered and "apikey" not in lowered:
        findings.append(
            AuditFinding(
                title="鉴权信息未明确",
                severity="high",
                evidence="输入内容未体现明确的认证或授权控制。",
                recommendation="确认每个敏感接口都存在强制认证、鉴权和最小权限检查。",
            )
        )

    if "set-cookie" in lowered and "httponly" not in lowered:
        findings.append(
            AuditFinding(
                title="Cookie 缺少 HttpOnly 说明",
                severity="medium",
                evidence="响应头样本中存在 Set-Cookie，但未见 HttpOnly。",
                recommendation="为会话 Cookie 增加 HttpOnly、Secure 和合理的 SameSite 策略。",
            )
        )

    if not findings:
        findings.append(
            AuditFinding(
                title="未发现明显高风险配置",
                severity="info",
                evidence="当前样本未命中已知规则。",
                recommendation="结合 OpenAPI、响应头和真实授权测试结果做进一步人工复核。",
            )
        )

    return AuditResponse(
        asset_name="local",
        summary="这是一个基于规则的初筛结果，适合授权环境的防御审计预览。",
        findings=findings,
    )


def create_app() -> FastAPI:
    if FastAPI is None:
        raise ModuleNotFoundError("请先安装 fastapi、pydantic 和 uvicorn 以启动审计服务")
    app = FastAPI(title="Defensive Audit API", version="0.1.0")
    auth = load_auth_config()
    api_key_dependency = require_api_key(auth)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/audit", dependencies=[Depends(api_key_dependency)] if api_key_dependency else [])
    def audit(req: AuditRequest):
        _prompt = render_defensive_audit_prompt(req.content)
        # 当前实现采用本地规则，后续可替换为你训练后的模型推理。
        result = local_audit(req.content)
        return asdict(AuditResponse(asset_name=req.asset_name, summary=result.summary, findings=result.findings))

    return app


def main() -> None:
    args = parse_args()
    if args.api_key:
        import os

        os.environ.setdefault(args.api_key_env, args.api_key)
    app = create_app()

    if args.demo_file:
        text = Path(args.demo_file).read_text(encoding="utf-8")
        print(json.dumps(asdict(local_audit(text)), ensure_ascii=False, indent=2))
        return

    if uvicorn is None:
        raise ModuleNotFoundError("请先安装 fastapi、pydantic 和 uvicorn 以启动审计服务")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
