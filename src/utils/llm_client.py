"""统一 LLM 客户端 —— 支持 DeepSeek / Kimi (Moonshot) 等国产模型调度

从 .env 加载配置，提供 call_llm() 统一调用接口。
- model 包含 "deepseek" → DEEPSEEK_BASE_URL + DEEPSEEK_API_KEY
- model 包含 "moonshot" 或 "kimi" → KIMI_BASE_URL + KIMI_API_KEY
- 其他情况 → 默认使用 DEEPSEEK 配置
- 失败自动重试 2 次，再失败则切换备用模型
"""

from __future__ import annotations

import os
import time
from functools import lru_cache

import openai
from dotenv import load_dotenv

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── 加载 .env ────────────────────────────────────────────────────
load_dotenv()


@lru_cache(maxsize=1)
def _get_config() -> dict:
    """缓存的配置加载"""
    return {
        "deepseek_api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
        "deepseek_base_url": os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        "kimi_api_key": os.environ.get("KIMI_API_KEY", ""),
        "kimi_base_url": os.environ.get("KIMI_BASE_URL", "https://api.moonshot.cn/v1"),
        "default_model": os.environ.get("DEFAULT_MODEL", "deepseek-v4-pro"),
    }


def _resolve_credentials(model: str) -> tuple[str, str, str]:
    """根据 model 名解析 api_key / base_url / 实际模型名

    Returns:
        (api_key, base_url, actual_model)
    """
    cfg = _get_config()

    model_lower = model.lower()
    if "deepseek" in model_lower:
        return (
            cfg["deepseek_api_key"],
            cfg["deepseek_base_url"],
            model,
        )
    elif "moonshot" in model_lower or "kimi" in model_lower:
        return (
            cfg["kimi_api_key"],
            cfg["kimi_base_url"],
            model,
        )
    else:
        # 其他/未知模型 → 默认走 DeepSeek
        return (
            cfg["deepseek_api_key"],
            cfg["deepseek_base_url"],
            model,
        )


def _get_fallback_model(model: str) -> str:
    """获取备用模型名"""
    model_lower = model.lower()
    if "deepseek" in model_lower:
        return "moonshot-v1-32k"
    return "deepseek-v4-pro"


def call_llm(
    model: str | None = None,
    messages: list[dict] | None = None,
    max_tokens: int = 2000,
    temperature: float = 0.8,
) -> str:
    """统一 LLM 调用入口

    Args:
        model: 模型名（如 "deepseek-v4-pro"、"moonshot-v1-32k"），默认使用 DEFAULT_MODEL
        messages: 消息列表 [{"role": "user"/"system", "content": "..."}]
        max_tokens: 最大生成 token 数
        temperature: 温度参数

    Returns:
        模型返回的文本内容

    Raises:
        RuntimeError: 所有模型（含备用）均调用失败时抛出
    """
    if messages is None:
        messages = []

    actual_model = model or _get_config()["default_model"]

    # 先尝试主模型，失败后切备用模型
    for attempt_model in (actual_model, _get_fallback_model(actual_model)):
        api_key, base_url, model_name = _resolve_credentials(attempt_model)

        logger.info(
            "[LLM] 调用模型: %s, 消息数: %d, max_tokens: %d",
            model_name, len(messages), max_tokens,
        )

        # 重试 2 次（共 3 次尝试）
        last_exception: Exception | None = None
        for retry in range(3):
            try:
                t0 = time.time()
                client = openai.OpenAI(api_key=api_key, base_url=base_url)
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,  # type: ignore[arg-type]
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                elapsed = time.time() - t0
                content = (response.choices[0].message.content or "").strip()
                logger.info(
                    "[LLM] 调用成功: %s, 耗时 %.1fs, 返回 %d 字",
                    model_name, elapsed, len(content),
                )
                return content
            except Exception as exc:
                last_exception = exc
                logger.warning(
                    "[LLM] %s 第 %d 次调用失败: %s",
                    model_name, retry + 1, exc,
                )
                if retry < 2:
                    time.sleep(1)

        # 该模型 3 次均失败
        if attempt_model == actual_model:
            logger.warning(
                "[LLM] 主模型 %s 全部失败，切换备用模型 %s",
                model_name, _get_fallback_model(actual_model),
            )
        else:
            # 备用模型也失败
            raise RuntimeError(
                f"所有模型调用均失败，最后错误: {last_exception}"
            )

    # 理论上不会走到这里，但让类型检查器安心
    raise RuntimeError("LLM 调用失败")
