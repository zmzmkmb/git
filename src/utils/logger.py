"""日志工具"""

import logging
import sys

# 全局 Logger 实例
_logger: logging.Logger | None = None


def get_logger(name: str = "novel_system") -> logging.Logger:
    """获取（或创建）Logger 实例"""
    global _logger
    if _logger is None:
        _logger = logging.getLogger(name)
        _logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        fmt = logging.Formatter(
            "[%(asctime)s] [%(name)s] %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        _logger.addHandler(handler)
    return _logger
