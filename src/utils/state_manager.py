"""状态管理器 —— 负责加载/保存 story_state.json，保证持久化"""

import json
import os
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)

# 状态文件默认路径
DEFAULT_STATE_PATH = Path(__file__).resolve().parent.parent / "data" / "story_state.json"

# 初始空状态骨架
INITIAL_STATE: dict[str, Any] = {
    "characters": {},
    "locations": {},
    "power_system": {},
    "foreshadowing": [],
    "timeline": [],
}


def load_state(path: str | Path | None = None) -> dict[str, Any]:
    """从 JSON 文件加载 story state，若文件不存在则返回初始空状态"""
    filepath = Path(path) if path else DEFAULT_STATE_PATH
    if not filepath.exists():
        logger.info("状态文件 %s 不存在，使用初始空状态", filepath)
        return _deep_copy_state(INITIAL_STATE)

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info("已从 %s 加载故事状态（%d 个角色，%d 个地点）",
                     filepath, len(data.get("characters", {})), len(data.get("locations", {})))
        return data
    except (json.JSONDecodeError, IOError) as e:
        logger.error("加载状态文件失败: %s，使用初始空状态", e)
        return _deep_copy_state(INITIAL_STATE)


def save_state(state: dict[str, Any], path: str | Path | None = None) -> None:
    """将 story state 持久化到 JSON 文件"""
    filepath = Path(path) if path else DEFAULT_STATE_PATH
    filepath.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        logger.info("故事状态已保存至 %s", filepath)
    except IOError as e:
        logger.error("保存状态文件失败: %s", e)


def reset_state(path: str | Path | None = None) -> dict[str, Any]:
    """重置状态为初始空状态并保存"""
    state = _deep_copy_state(INITIAL_STATE)
    save_state(state, path)
    return state


def _deep_copy_state(state: dict[str, Any]) -> dict[str, Any]:
    """深拷贝状态以避免引用问题"""
    return json.loads(json.dumps(state))
