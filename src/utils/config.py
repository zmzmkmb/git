"""配置文件 —— API keys、模型选择等（兼容旧版代理引用）"""

import os

from dotenv import load_dotenv

load_dotenv()

# OpenAI / 兼容 API 配置（用于尚未迁移到 llm_client 的旧代码）
OPENAI_API_KEY = os.environ.get("DEEPSEEK_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
OPENAI_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com/v1"))
MODEL_NAME = os.environ.get("DEFAULT_MODEL", os.environ.get("MODEL_NAME", "deepseek-v4-pro"))

# 生成参数
TEMPERATURE = 0.7
MAX_TOKENS = 2048
