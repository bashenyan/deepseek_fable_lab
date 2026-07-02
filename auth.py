"""API 鉴权与访问控制。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

try:
    from fastapi import Header, HTTPException, status
except ModuleNotFoundError:  # pragma: no cover
    Header = None  # type: ignore[assignment]
    HTTPException = Exception  # type: ignore[assignment]
    status = None  # type: ignore[assignment]


@dataclass(frozen=True)
class AuthConfig:
    """运行期 API 鉴权配置。"""

    api_key: str | None
    header_name: str = "X-API-Key"

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)


def load_auth_config(api_key: str | None = None, env_name: str = "DEEPSEEK_FABLE_API_KEY") -> AuthConfig:
    """从显式参数或环境变量加载 API Key。"""

    key = api_key or os.getenv(env_name) or None
    return AuthConfig(api_key=key)


def require_api_key(auth: AuthConfig):
    """生成 FastAPI 依赖项，未配置密钥时自动退化为公开模式。"""

    if not auth.enabled:
        return None

    if Header is None:
        raise ModuleNotFoundError("请先安装 fastapi 才能启用 API 鉴权")

    async def _dependency(x_api_key: Optional[str] = Header(default=None, alias=auth.header_name)):
        if x_api_key != auth.api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unauthorized",
            )

    return _dependency
