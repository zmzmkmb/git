"""知识库检索器 —— 根据查询从知识库 JSON + 知识图谱中提取相关内容

设计要点：
- 初期用关键词匹配 + 简单评分（命中关键词 + 标签匹配 + 平台匹配）
- 接口稳定：get_writer_context / get_reader_context，便于后续替换为向量检索
- 全文加载 + 内存索引，避免重复 IO
- ★ 已接入知识图谱（graph_retriever），在 JSON KB 基础上提供 KG 增强
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── 尝试加载知识图谱检索器（非必须，失败不阻塞） ──────────────────
try:
    from src.utils.graph_retriever import query_graph as _kg_query
    from src.utils.graph_retriever import get_writing_tips as _kg_tips
    _KG_AVAILABLE = True
except Exception as e:
    logger.warning("知识图谱加载失败: %s，KG 增强功能不可用", e)
    _KG_AVAILABLE = False
    def _kg_query(*args: Any, **kwargs: Any) -> dict:  # type: ignore[no-redef]
        return {}
    def _kg_tips(*args: Any, **kwargs: Any) -> list[str]:  # type: ignore[no-redef]
        return []


# 知识库根目录
KNOWLEDGE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data"
)


@lru_cache(maxsize=4)
def _load_kb(name: str) -> dict:
    """加载并缓存知识库 JSON"""
    path = os.path.join(KNOWLEDGE_DIR, f"{name}_knowledge.json")
    if not os.path.exists(path):
        logger.warning("知识库文件不存在: %s", path)
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _score_match(query: str, keywords: list[str]) -> int:
    """计算 query 与关键词列表的匹配分（命中次数 + 包含）"""
    q = query.lower()
    score = 0
    for kw in keywords:
        if not kw:
            continue
        if kw.lower() in q:
            score += 2  # 完整包含
        # 单字命中（中文支持）
        elif any(ch in q for ch in kw if "\u4e00" <= ch <= "\u9fff"):
            score += 1
    return score


def _pick_template(query: str, writer_kb: dict) -> tuple[str, dict] | tuple[None, None]:
    """从 templates 中选匹配的套路模板

    完全不使用硬编码别名映射，仅通过关键词文本匹配。
    如果 query 是自定义概念（meta/元小说等），直接跳过模板。
    """
    # 检测是否为自定义/元小说概念（跳过套路模板）
    meta_keywords = ["作家", "被困", "meta", "笔下", "小说世界", "角色记忆错乱",
                     "钢笔", "改写现实", "终极秘密", "大纲错误", "文字即真实"]
    is_meta_concept = any(kw in query for kw in meta_keywords)

    if is_meta_concept:
        logger.info("  [KB] 检测到元小说/自定义概念，跳过传统套路模板匹配")
        return None, None

    templates = writer_kb.get("templates", {}) or {}
    best_name: str | None = None
    best_score = -1
    best_data: dict | None = None
    for name, data in templates.items():
        if not isinstance(data, dict):
            continue
        # 收集该模板的关键词（仅用模板自身名称和tags）
        keywords = [name]
        keywords.extend(data.get("tags_index", []) or [])
        score = _score_match(query, keywords)
        if score > best_score:
            best_score = score
            best_name = name
            best_data = data
    if best_score <= 0 or best_data is None:
        return None, None
    return best_name, best_data


def _pick_platform(query: str, writer_kb: dict, explicit: str | None = None) -> str:
    """从 query / 显式参数中识别目标平台"""
    if explicit and explicit in (writer_kb.get("platform_style_rules") or {}):
        return explicit
    for platform in (writer_kb.get("platform_style_rules") or {}).keys():
        if platform in query:
            return platform
    return "番茄"  # 默认


# ════════════════════════════════════════════════════════════════
#  公开接口
# ════════════════════════════════════════════════════════════════


def get_writer_context(query: str, platform: str | None = None) -> dict:
    """为作家模块返回知识库检索结果

    返回：
    {
      "template_name": str | None,
      "template": {...} | None,
      "platform": "番茄",
      "platform_rules": {...},
      "power_paradigm": {...} | None,
      "ai_odor_features": {...},
      "dialogue_samples": [...],
      "ai_odor_pitfalls": [...],
      "opening_hooks": [...]
    }
    """
    writer_kb = _load_kb("writer")
    if not writer_kb:
        return {}

    # 套路模板
    tmpl_name, tmpl_data = _pick_template(query, writer_kb)
    if tmpl_name:
        logger.info("  [KB] 命中作家模板: %s", tmpl_name)

    # 平台
    plat = _pick_platform(query, writer_kb, explicit=platform)
    platform_rules = (writer_kb.get("platform_style_rules") or {}).get(plat, {})

    # 力量体系范式
    power_paradigm = None
    if tmpl_data:
        psp = tmpl_data.get("power_system_paradigm")
        if psp:
            # 优先用模板的范式；否则用全局
            power_paradigm = psp
    if not power_paradigm:
        # 按 query 关键词选范式
        for genre, data in (writer_kb.get("power_system_paradigms") or {}).items():
            if genre in query or (tmpl_name and genre in str(tmpl_data)):
                power_paradigm = data
                break

    # AI 味特征 + 反面清单
    ai_odor = writer_kb.get("ai_odor_features", {}) or {}

    # 对话样本 + 开场钩子
    dialogue_samples = (tmpl_data or {}).get("recommended_dialogue_samples", []) or []
    ai_odor_pitfalls = (tmpl_data or {}).get("ai_odor_pitfalls", []) or []
    opening_hooks = platform_rules.get("preferred_openings", []) or []

    # 套路 key_beats
    key_beats = (tmpl_data or {}).get("key_beats", []) or []
    opening_hook = (tmpl_data or {}).get("opening_hook", "") or ""
    first_chapter_template = (tmpl_data or {}).get("first_chapter_outline_template", "") or ""

    result: dict[str, Any] = {
        "template_name": tmpl_name,
        "template": tmpl_data,
        "platform": plat,
        "platform_rules": platform_rules,
        "power_paradigm": power_paradigm,
        "ai_odor_features": ai_odor,
        "dialogue_samples": dialogue_samples,
        "ai_odor_pitfalls": ai_odor_pitfalls,
        "opening_hooks": opening_hooks,
        "key_beats": key_beats,
        "opening_hook": opening_hook,
        "first_chapter_template": first_chapter_template,
    }

    # ★ 知识图谱增强（如果可用）
    if _KG_AVAILABLE:
        kg_result = _kg_query(query, top_k=30)
        if kg_result:
            result["graph_tips"] = _kg_tips(query, num=15)
            result["graph_summary"] = kg_result.get("text_summary", "")
            result["graph_definitions"] = kg_result.get("definitions", [])
            result["graph_requirements"] = kg_result.get("requirements", [])
            result["graph_suggestions"] = kg_result.get("suggestions", [])
            result["graph_seed_nodes"] = kg_result.get("seed_nodes", [])

    return result


def get_reader_context(query: str, platform: str | None = None) -> dict:
    """为读者模块返回知识库检索结果

    返回：
    {
      "platform": "番茄",
      "persona": {...},
      "satisfaction_map": {...},
      "abandon_signals": [...],
      "ai_odor_dictionary": {...}
    }
    """
    reader_kb = _load_kb("reader")
    writer_kb = _load_kb("writer")
    if not reader_kb:
        return {}

    plat = _pick_platform(query, writer_kb, explicit=platform)
    persona = (reader_kb.get("reader_personas") or {}).get(plat, {})

    # 爽点地图（按 query 匹配模板名）
    satisfaction_map: dict = {}
    for tpl_name, data in (reader_kb.get("satisfaction_map") or {}).items():
        if tpl_name in query or tpl_name.replace("流", "") in query:
            satisfaction_map = data
            break

    result: dict[str, Any] = {
        "platform": plat,
        "persona": persona,
        "satisfaction_map": satisfaction_map,
        "abandon_signals": (reader_kb.get("abandon_warning_signals") or {}).get("high_risk", []),
        "ai_odor_dictionary": reader_kb.get("ai_odor_dictionary", {}),
        "scoring_calibration": reader_kb.get("scoring_calibration", {}),
    }

    # ★ 知识图谱增强（如果可用）
    if _KG_AVAILABLE:
        kg_result = _kg_query(query, top_k=20)
        if kg_result:
            result["graph_tips"] = _kg_tips(query, num=8)
            result["graph_causes"] = kg_result.get("causes", [])  # 弃书因果链

    return result


def get_ai_odor_score(text: str, dictionary: dict | None = None) -> int:
    """基于知识库词典计算 AI 味分（0-100）

    命中 tier_1/tier_2/tier_3 加分；命中 tier_4 减分。
    """
    if not text:
        return 0
    reader_kb = _load_kb("reader")
    dictionary = dictionary or (reader_kb.get("ai_odor_dictionary") or {})
    weights = dictionary.get("weights", {}) or {}

    score = 0
    for tier_key, phrases in dictionary.items():
        if not tier_key.startswith("tier_") or not isinstance(phrases, list):
            continue
        weight = weights.get(tier_key, 0)
        if not weight:
            continue
        hits = sum(text.count(p) for p in phrases)
        score += hits * weight

    # 归一化到 0-100：每 10 分原始 → 10
    score = max(0, min(100, int(score * 1.5)))
    return score


# ════════════════════════════════════════════════════════════════
#  调试 / 测试
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # 简单自测
    q = "玄幻题材快节奏开局"
    print("=== Writer Context ===")
    wctx = get_writer_context(q)
    print(json.dumps({k: v for k, v in wctx.items() if k != "template"},
                     ensure_ascii=False, indent=2))
    print(f"template_name = {wctx.get('template_name')}")

    print("\n=== Reader Context ===")
    rctx = get_reader_context(q)
    print(json.dumps(rctx, ensure_ascii=False, indent=2)[:500])

    print("\n=== AI Odor Score ===")
    test_text = "他仿佛如同潮水般涌来，心中不由得翻江倒海，缓缓地抬起头。"
    print(f"text: {test_text}")
    print(f"score: {get_ai_odor_score(test_text)}")
