# coding=utf-8
"""
SpecSoT HTTP API 服务器

基于 FastAPI 实现的 HTTP 服务，提供：
- POST /generate: 文本生成
- GET /health: 健康检查
- GET /info: 模型信息

使用示例：
    >>> from SpecSoT_v2.api import SpecSoTServer
    >>> server = SpecSoTServer(model, host="0.0.0.0", port=8000)
    >>> server.run()
"""

import time
import logging
from typing import Optional, Any

import torch

from .protocol import (
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    ModelInfo,
)

logger = logging.getLogger(__name__)


def create_app(model: Any):
    """
    创建 FastAPI 应用

    Args:
        model: SpecSoTModel 实例

    Returns:
        FastAPI 应用实例
    """
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError:
        raise ImportError("请安装 fastapi: pip install fastapi uvicorn")

    app = FastAPI(
        title="SpecSoT API",
        description="SpecSoT 推理服务 API",
        version="2.0",
    )

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(request: GenerateRequest) -> GenerateResponse:
        """
        文本生成接口

        Args:
            request: 生成请求

        Returns:
            生成响应
        """
        try:
            start_time = time.time()

            output_ids, stats = model.generate(
                task_prompt=request.prompt,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                enable_parallel=request.enable_parallel,
                use_semantic_constraint=request.use_semantic_constraint,
            )

            # 解码输出
            if output_ids.numel() > 0:
                text = model.tokenizer.decode(
                    output_ids[0].tolist(),
                    skip_special_tokens=True
                )
                tokens = len(output_ids[0])
            else:
                text = ""
                tokens = 0

            elapsed_ms = (time.time() - start_time) * 1000

            return GenerateResponse(
                text=text,
                tokens=tokens,
                time_ms=elapsed_ms,
                mode=stats.get("mode", "unknown"),
                stats=stats,
            )

        except Exception as e:
            logger.error(f"生成失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """健康检查接口"""
        gpu_mem = 0.0
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / (1024 ** 2)

        return HealthResponse(
            status="ok",
            model_loaded=model is not None,
            gpu_memory_mb=gpu_mem,
        )

    @app.get("/info", response_model=ModelInfo)
    async def info() -> ModelInfo:
        """模型信息接口"""
        return ModelInfo(
            base_model=getattr(model, "base_model_path", "unknown"),
            eagle_model=getattr(model, "eagle_model_path", "unknown"),
            use_eagle3=getattr(model, "use_eagle3", True),
            max_seq_len=getattr(model, "max_length", 4096),
        )

    return app


class SpecSoTServer:
    """
    SpecSoT HTTP 服务器

    封装 FastAPI 应用，提供简单的启动接口。

    Attributes:
        model: SpecSoTModel 实例
        host: 监听地址
        port: 监听端口
        app: FastAPI 应用实例
    """

    def __init__(
        self,
        model: Any,
        host: str = "0.0.0.0",
        port: int = 8000,
    ):
        """
        初始化服务器

        Args:
            model: SpecSoTModel 实例
            host: 监听地址
            port: 监听端口
        """
        self.model = model
        self.host = host
        self.port = port
        self.app = create_app(model)

    def run(self):
        """启动服务器"""
        try:
            import uvicorn
        except ImportError:
            raise ImportError("请安装 uvicorn: pip install uvicorn")

        logger.info(f"启动 SpecSoT API 服务: http://{self.host}:{self.port}")
        uvicorn.run(self.app, host=self.host, port=self.port)
