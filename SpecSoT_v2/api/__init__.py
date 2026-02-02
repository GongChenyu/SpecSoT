# coding=utf-8
"""
SpecSoT API 模块

提供 HTTP API 服务接口：
- GenerateRequest: 生成请求
- GenerateResponse: 生成响应
- HealthResponse: 健康检查响应
- ModelInfo: 模型信息
- SpecSoTServer: HTTP 服务器
- create_app: 创建 FastAPI 应用
"""

from .protocol import (
    GenerateMode,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    ModelInfo,
)

from .server import (
    SpecSoTServer,
    create_app,
)

__all__ = [
    "GenerateMode",
    "GenerateRequest",
    "GenerateResponse",
    "HealthResponse",
    "ModelInfo",
    "SpecSoTServer",
    "create_app",
]
