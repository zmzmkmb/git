"""Agent 基类 —— 定义统一接口"""

from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """所有 Agent 的基类，对外暴露统一的 run() 方法"""

    def __init__(self, name: str = "BaseAgent"):
        self.name = name

    @abstractmethod
    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """执行 Agent 逻辑

        Args:
            context: 上游传入的上下文数据（比如章节编号、已写文本等）

        Returns:
            dict: 本 Agent 的处理结果
        """
        ...
