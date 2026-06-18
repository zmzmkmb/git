"""设定智能补全器 —— 用户提供部分设定，LLM 自动补全其余部分

用法：
  from src.utils.design_completer import auto_complete_design

  partial = {
    "main_characters": [{"name": "主角", "role": "主角", "archetype": "成长型"}],
    "world_setting": {"genre": "玄幻"}
  }
  full_design = auto_complete_design(partial)
"""

from __future__ import annotations

import json
from typing import Any

from src.utils.llm_client import call_llm
from src.utils.logger import get_logger

logger = get_logger(__name__)

AUTO_COMPLETE_SYSTEM_PROMPT = """你是一位资深网文编辑，擅长帮作者完善故事设定。
用户会提供一部分故事设定（可能不完整），你需要根据网文创作规律补全缺失的部分。

补全规则：
1. 保留用户已提供的所有设定，不做修改
2. 只补充缺失的字段：
   - main_characters（如果用户只提供了部分角色，按角色类型补齐4个：主角/金手指/初期对手/女主）
   - world_setting（世界名称、主要势力、流派、风格）
   - power_system（力量体系名称、境界等级）
   - chapter_outlines（按总章节数生成每章大纲）
   - story_outline（故事梗概，如果缺失）
   - foreshadowing_arc（伏笔规划）
3. 补全的内容要合理且完整，符合网文创作规律
4. 返回完整的 JSON，结构必须与以下模板一致

输出格式示例：
{
  "premise": "故事核心设定",
  "story_outline": "一句话梗概",
  "main_characters": [
    {"name": "...", "role": "主角", "archetype": "成长型", "starting_status": "初始状态"},
    {"name": "...", "role": "金手指", "archetype": "引路者", "starting_status": "神秘状态"},
    {"name": "...", "role": "初期对手", "archetype": "敌对者", "starting_status": "高于主角"},
    {"name": "...", "role": "女主", "archetype": "搭档", "starting_status": "相遇时"}
  ],
  "world_setting": {
    "main_world": "...",
    "main_factions": ["..."],
    "genre": "待定",
    "tone": "待定"
  },
  "chapter_outlines": {
    "1": "第1章大纲...",
    "2": "第2章大纲..."
  },
  "foreshadowing_arc": {
    "planted_in": {"1": ["伏笔1"]},
    "resolved_in": {"3": ["伏笔1回收"]}
  }
}

注意：不要输出任何解释文字，只输出 JSON。"""


def auto_complete_design(partial: dict, total_chapters: int = 3,
                          model: str = "deepseek-v4-pro") -> dict:
    """用 LLM 补全不完整的 story design

    Args:
        partial: 用户提供的部分设定（可包含任意字段）
        total_chapters: 需要生成的章数
        model: LLM 模型名

    Returns:
        完整的 story design dict
    """
    # 如果设定已经完整（有 chapter_outlines 就直接用）
    if partial.get("chapter_outlines") and partial.get("main_characters"):
        logger.info("[DesignCompleter] 设定已完整，跳过补全")
        return partial

    user_content = json.dumps(partial, ensure_ascii=False, indent=2)
    user_content += f"\n\n请补全以上设定，生成 {total_chapters} 章的章节大纲。"

    try:
        raw = call_llm(
            model=model,
            messages=[
                {"role": "system", "content": AUTO_COMPLETE_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            max_tokens=4096,
            temperature=0.7,
        )
        logger.info("[DesignCompleter] LLM 补全完成，返回 %d 字符", len(raw))

        # 解析 JSON
        completed = _extract_json(raw)
        if not completed:
            logger.warning("[DesignCompleter] LLM 返回无法解析，使用原始设定")
            return partial

        # 合并：以用户设定为基准，LLM 补充缺失字段
        merged = partial.copy()
        for key in ["world_setting", "story_outline", "premise",
                     "chapter_outlines", "foreshadowing_arc"]:
            if key not in merged and key in completed:
                merged[key] = completed[key]
        # 角色：如果用户一个都没提供，用 LLM 的
        if not merged.get("main_characters") and completed.get("main_characters"):
            merged["main_characters"] = completed["main_characters"]
        # 如果用户只提供了部分角色，补齐
        elif merged.get("main_characters") and completed.get("main_characters"):
            existing_roles = {c.get("role") for c in merged["main_characters"]}
            for c in completed["main_characters"]:
                role = c.get("role", "")
                if role and role not in existing_roles:
                    merged["main_characters"].append(c)
                    existing_roles.add(role)

        logger.info("[DesignCompleter] 合并完成: %d 个角色, %d 章大纲",
                     len(merged.get("main_characters", [])),
                     len(merged.get("chapter_outlines", {})))
        return merged

    except Exception as e:
        logger.error("[DesignCompleter] 补全失败: %s，使用原始设定", e)
        return partial


def _extract_json(text: str) -> dict | None:
    """从 LLM 返回中提取 JSON"""
    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 尝试提取 ```json ... ``` 代码块
    import re
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 尝试找到第一个 { 和最后一个 }
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    return None
