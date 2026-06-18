"""知识图谱检索器 —— 加载已构建的三元组，过滤噪音，提供图查询接口

数据源：通过 KG_DATA_DIR 环境变量指定目录
        - triples.json              → 写作技巧知识图谱（必备）
        - feilu_bridge_triples.json → 飞卢桥接知识图谱（可选，自动合并）
        (head, relation, tail, source_doc, source_chunk)

设计要点：
- 启动时加载 triples.json + feilu_bridge_triples.json 到内存，构建 head→edges 索引
- 过滤：仅保留写作相关的 relation 类型
- 查询：关键词匹配 head/tail → 子图扩散 1-hop → 按 relation 分组返回
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from functools import lru_cache
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── 数据路径 ─────────────────────────────────────────────────────
# 优先从环境变量读取，否则用默认路径
_KG_DIR = os.environ.get(
    "KG_DATA_DIR",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "..",
        "..",
        "BaiduNetdiskDownload",
        "网文写作入门小白写作大纲",
        "output",
    ),
)

_TRIPLES_PATH = os.path.normpath(os.path.join(_KG_DIR, "triples.json"))
_FEILU_TRIPLES_PATH = os.path.normpath(os.path.join(_KG_DIR, "feilu_bridge_triples.json"))
_CHUNKS_PATH = os.path.normpath(os.path.join(_KG_DIR, "chunks.json"))

# ── 内嵌核心三元组（项目内打包，作为回退）─────────────────────────
_CORE_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "data"))
_CORE_TRIPLES_PATH = os.path.normpath(os.path.join(_CORE_DIR, "core_triples.json"))

# 数据源选择：auto（自动回退）| core_only（仅用内嵌）| remote_only（仅用外部）
_KG_SOURCE = os.environ.get("KG_SOURCE", "auto")

# ── 写作相关 relation 白名单 ─────────────────────────────────────
_WRITING_RELATIONS: set[str] = {
    "定义", "要求", "建议", "导致", "特征", "方法", "作用",
    "包含", "需要", "指", "影响", "应用", "分类", "原则",
    "属于", "特点", "目的是", "原因", "好处", "技巧",
    "关键", "要点", "注意", "避免", "分为", "包括",
    "例如", "用于", "提高", "增加", "公式",
    # 飞卢桥接关系
    "飞卢实践", "占比", "最常见开局冲突", "冲突强度",
    "飞卢套路符合度", "核心优点", "金手指类型",
    "平均套路符合度", "包含要素", "第一人称", "第三人称",
}

# ── 飞卢核心实体（用于节点匹配优先级加成）─────────────────────────
_FEILU_CORE_ENTITIES: set[str] = {
    "飞卢金手指类型分布", "飞卢开局冲突分布", "飞卢题材分布",
    "飞卢开局风格分布", "飞卢主角身份分布", "飞卢句子风格分布",
    "飞卢视角分布", "飞卢套路公式", "飞卢主角类型偏好",
    "飞卢深度分析案例",
}

# ── 写作核心概念（用于节点匹配优先级加成）─────────────────────────
_WRITING_CORE_CONCEPTS: set[str] = {
    "情节", "写作", "金手指", "文笔", "代入感", "大纲", "节奏",
    "人物设定", "角色", "人物", "小说", "网络小说", "主角",
    "爽点", "伏笔", "悬念", "升级系统", "开头", "开篇",
    "简介", "书名", "对话", "冲突", "描写", "章节",
    "更新", "上架", "签约", "买断", "分成",
    "网文", "玄幻", "女频", "男频", "读者",
    "作品", "故事", "配角", "副本", "世界观", "设定",
    "场景", "开头", "结尾", "节奏把握", "文风",
    "预设目标", "凤头猪肚麒麟尾", "凤头", "猪肚", "麒麟尾",
    "代入", "紧迫感", "追读", "弃书", "更新频率",
    "大纲构思", "故事情节", "书名简介", "作品简介",
    # 飞卢相关概念
    "系统激活", "穿越重生", "退婚打脸", "签到流", "聊天群",
    "系统流", "无敌流", "开局流",
} | _FEILU_CORE_ENTITIES  # 合并飞卢核心实体


# ════════════════════════════════════════════════════════════════
#  数据加载与索引
# ════════════════════════════════════════════════════════════════

@lru_cache(maxsize=1)
def _load_all() -> dict[str, Any]:
    """加载并缓存全部三元组，构建索引

    数据源优先级（由 KG_SOURCE 环境变量控制）：
      - KG_SOURCE=remote_only: 仅用外部 triples.json
      - KG_SOURCE=core_only:   仅用内嵌 core_triples.json
      - KG_SOURCE=auto (默认):  外部优先 → 内嵌回退 → 空

    自动合并 feilu_bridge_triples.json（如果同目录存在）。
    """
    source = os.environ.get("KG_SOURCE", _KG_SOURCE)

    # 确定候选路径
    candidates: list[tuple[str, str]] = []  # (label, path)

    if source == "core_only":
        candidates = [("core_triples.json", _CORE_TRIPLES_PATH)]
    elif source == "remote_only":
        candidates = [("external triples.json", _TRIPLES_PATH)]
    else:  # auto
        candidates = [
            ("external triples.json", _TRIPLES_PATH),
            ("core_triples.json", _CORE_TRIPLES_PATH),
        ]

    # 加载主图谱
    data = _try_load(candidates)
    if data is None:
        logger.warning("[KG] 所有数据源均不可用，返回空")
        return {"triples": [], "head_index": {}, "tail_index": {}}

    # 尝试加载飞卢桥接图谱（可选）
    if os.path.exists(_FEILU_TRIPLES_PATH):
        logger.info("[KG] 发现飞卢桥接图谱: %s", _FEILU_TRIPLES_PATH)
        feilu_data = _load_from_path(_FEILU_TRIPLES_PATH)
        if feilu_data:
            # 合并三元组
            seen_keys: set[tuple[str, str, str]] = set()
            for t in data["triples"]:
                seen_keys.add((t["head"], t["relation"], t["tail"]))
            for t in feilu_data["triples"]:
                key = (t["head"], t["relation"], t["tail"])
                if key not in seen_keys:
                    seen_keys.add(key)
                    data["triples"].append(t)
            # 合并索引
            for head, edges in feilu_data["head_index"].items():
                data["head_index"].setdefault(head, []).extend(edges)
            for tail, edges in feilu_data["tail_index"].items():
                data["tail_index"].setdefault(tail, []).extend(edges)
            logger.info(
                "[KG] 飞卢桥接合并完成: 共 %d 条三元组",
                len(data["triples"]),
            )

    return data


def _try_load(candidates: list[tuple[str, str]]) -> dict[str, Any] | None:
    """逐一尝试候选路径，返回第一个成功加载的数据"""
    for label, path in candidates:
        if os.path.exists(path):
            logger.info("[KG] 从 %s 加载: %s", label, path)
            return _load_from_path(path)
        logger.debug("[KG] %s 不存在: %s", label, path)
    return None


def _load_from_path(path: str) -> dict[str, Any]:
    """从指定路径加载三元组文件并构建索引"""
    all_triples: list[dict] = []
    head_index: dict[str, list[dict]] = defaultdict(list)
    tail_index: dict[str, list[dict]] = defaultdict(list)

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    for r in raw:
        if not isinstance(r, dict):
            continue
        head = (r.get("head") or "").strip()
        relation = (r.get("relation") or "").strip()
        tail = (r.get("tail") or "").strip()

        if not head or not relation or not tail:
            continue

        # 过滤噪音 relation（仅对完整版做，core_triples 已过滤）
        if relation not in _WRITING_RELATIONS:
            continue

        tri = {
            "head": head,
            "relation": relation,
            "tail": tail,
            "source_doc": r.get("source_doc", ""),
            "source_chunk": r.get("source_chunk", -1),
        }
        all_triples.append(tri)
        head_index[head].append(tri)
        tail_index[tail].append(tri)

    logger.info(
        "[KG] 加载完成: %d 条有效三元组, %d 个 head 节点, %d 个 tail 节点",
        len(all_triples), len(head_index), len(tail_index),
    )
    return {
        "triples": all_triples,
        "head_index": dict(head_index),
        "tail_index": dict(tail_index),
    }


@lru_cache(maxsize=1)
def _load_chunks() -> dict[tuple[str, int], str]:
    """懒加载 chunks.json，按 (source_doc, chunk_id) 索引"""
    chunks: dict[tuple[str, int], str] = {}
    if not os.path.exists(_CHUNKS_PATH):
        return chunks
    try:
        with open(_CHUNKS_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for c in raw:
            if not isinstance(c, dict):
                continue
            doc = c.get("doc", "")
            cid = c.get("chunk_id", -1)
            text = c.get("text", "") or ""
            if doc and cid >= 0:
                chunks[(doc, cid)] = text
        logger.debug("[KG] 加载 %d 个原文 chunk", len(chunks))
    except Exception as e:
        logger.warning("[KG] 加载 chunks 失败: %s", e)
    return chunks


# ════════════════════════════════════════════════════════════════
#  查询引擎
# ════════════════════════════════════════════════════════════════

def _tokenize(text: str) -> set[str]:
    """简单中文分词：提取 2-4 字的连续词组 + 单字"""
    cleaned = re.sub(r"[^\u4e00-\u9fff]", "", text)
    tokens: set[str] = set()
    # 2-gram
    for i in range(len(cleaned) - 1):
        tokens.add(cleaned[i:i + 2])
    # 3-gram
    for i in range(len(cleaned) - 2):
        tokens.add(cleaned[i:i + 3])
    # 4-gram
    for i in range(len(cleaned) - 3):
        tokens.add(cleaned[i:i + 4])
    return tokens


def _score_node(node_text: str, query_tokens: set[str]) -> int:
    """计算节点与 query 的匹配分"""
    score = 0
    for tok in query_tokens:
        if tok in node_text:
            score += len(tok) * 2  # 越长匹配权重越高
    # 核心概念加成
    if node_text in _WRITING_CORE_CONCEPTS:
        score += 3
    return score


def query_graph(
    query: str,
    top_k: int = 20,
    expand_hops: int = 1,
) -> dict[str, Any]:
    """从知识图谱中检索与 query 相关的写作技巧

    Args:
        query: 查询文本（如大纲、平台名、套路名）
        top_k: 返回的三元组数量上限
        expand_hops: 从种子节点向外扩散跳数（1 或 2）

    Returns:
        {
            "triples": [{head, relation, tail, source_doc, source_chunk}, ...],
            "definitions": [{概念: 定义}, ...],       # 定义类三元组
            "requirements": [{概念: 要求}, ...],      # 要求/需要类三元组
            "suggestions": [{建议: 原因}, ...],        # 建议/注意/避免类
            "causes": [{行为: 后果}, ...],             # 导致/影响类
            "methods": [{目标: 方法}, ...],            # 方法/用于类
            "seed_nodes": ["匹配到的核心概念", ...],
        }
    """
    data = _load_all()
    all_triples = data["triples"]
    head_index = data["head_index"]
    if not all_triples:
        return {}

    query_tokens = _tokenize(query)

    # ── Step 1: 找种子节点 ──
    seed_scores: list[tuple[int, str]] = []
    for node_text in head_index:
        s = _score_node(node_text, query_tokens)
        if s > 0:
            seed_scores.append((s, node_text))
    seed_scores.sort(key=lambda x: -x[0])

    # 也扫描 tail（有时 query 匹配的是 tail 中的词）
    for node_text in data["tail_index"]:
        s = _score_node(node_text, query_tokens) - 1  # tail 优先级稍低
        if s > 0:
            seed_scores.append((s, node_text))
    # 去重
    seed_scores.sort(key=lambda x: -x[0])
    seen_nodes: set[str] = set()
    seed_nodes: list[str] = []
    for _, node in seed_scores:
        if node not in seen_nodes:
            seen_nodes.add(node)
            seed_nodes.append(node)
        if len(seed_nodes) >= 10:
            break

    if not seed_nodes:
        logger.info("[KG] query '%s' 未匹配到任何节点", query[:50])
        return {}

    logger.info("[KG] query '%s' → 种子节点: %s", query[:50], seed_nodes[:5])

    # ── Step 2: 从种子节点收集三元组 ──
    collected: dict[tuple[str, str, str], dict] = {}
    visited_nodes: set[str] = set(seed_nodes)
    frontier: list[str] = list(seed_nodes)

    for _ in range(expand_hops + 1):
        next_frontier: list[str] = []
        for node in frontier:
            # 以 node 为 head 的出边
            for tri in head_index.get(node, []):
                key = (tri["head"], tri["relation"], tri["tail"])
                if key not in collected:
                    collected[key] = tri
            # 以 node 为 tail 的入边（反向查找：哪些 head 指向 node）
            for tail_key, tri_list in data["tail_index"].items():
                if node in tail_key:
                    for tri in tri_list:
                        head_n = tri["head"]
                        if head_n not in visited_nodes:
                            visited_nodes.add(head_n)
                            next_frontier.append(head_n)
                        key = (tri["head"], tri["relation"], tri["tail"])
                        if key not in collected:
                            collected[key] = tri
        frontier = next_frontier

    logger.info("[KG] 收集 %d 条三元组 (expand=%d hop)", len(collected), expand_hops)

    # ── Step 3: 按 relation 分类 ──
    definitions: list[dict] = []      # 定义 / 特点是 / 指 / 特征
    requirements: list[dict] = []     # 要求 / 需要
    suggestions: list[dict] = []      # 建议 / 注意 / 避免 / 关键 / 要点
    causes: list[dict] = []           # 导致 / 影响
    methods: list[dict] = []          # 方法 / 用于 / 提高 / 增加 / 作用
    structure: list[dict] = []        # 包含 / 包括 / 分为 / 分类 / 属于
    other: list[dict] = []

    relation_groups = {
        "definitions": (definitions, {"定义", "特点是", "指", "特征", "特点", "目的是"}),
        "requirements": (requirements, {"要求", "需要"}),
        "suggestions": (suggestions, {"建议", "注意", "避免", "关键", "要点", "技巧", "原则"}),
        "causes": (causes, {"导致", "影响", "原因", "好处"}),
        "methods": (methods, {"方法", "用于", "提高", "增加", "作用", "应用", "例如", "公式"}),
        "structure": (structure, {"包含", "包括", "分为", "分类", "属于"}),
    }

    # 排序：核心概念优先，短的优先
    def _tri_sort_key(tri: dict) -> tuple[int, int]:
        h = tri["head"]
        is_core = 0 if h in _WRITING_CORE_CONCEPTS else 1
        return (is_core, len(tri["tail"] or ""))  # 先核心概念，再短 tail

    sorted_triples = sorted(collected.values(), key=_tri_sort_key)

    for tri in sorted_triples:
        rel = tri["relation"]
        placed = False
        for (_list, rels) in relation_groups.values():
            if rel in rels:
                _list.append(tri)
                placed = True
                break
        if not placed:
            other.append(tri)

    # ── Step 4: 组装结果 ──
    result = {
        "triples": sorted_triples[:top_k],
        "definitions": definitions[:15],
        "requirements": requirements[:20],
        "suggestions": suggestions[:20],
        "causes": causes[:15],
        "methods": methods[:15],
        "structure": structure[:15],
        "other": other[:10],
        "seed_nodes": seed_nodes,
    }

    # 生成纯文本摘要（方便注入到 prompt）
    result["text_summary"] = _build_text_summary(result)

    logger.info(
        "[KG] 返回 %d 条结果 (定义%d 要求%d 建议%d 因果%d 方法%d 结构%d)",
        len(result["triples"]),
        len(definitions), len(requirements), len(suggestions),
        len(causes), len(methods), len(structure),
    )
    return result


def _build_text_summary(result: dict) -> str:
    """将检索结果拼成一段可注入 prompt 的纯文本摘要"""
    lines: list[str] = []

    if result["definitions"]:
        lines.append("## 相关定义")
        for t in result["definitions"][:8]:
            lines.append(f"  - 「{t['head']}」即：{t['tail']}")

    if result["requirements"]:
        lines.append("\n## 写作要求")
        for t in result["requirements"][:10]:
            lines.append(f"  - 「{t['head']}」需要：{t['tail']}")

    if result["suggestions"]:
        lines.append("\n## 写作建议")
        for t in result["suggestions"][:10]:
            lines.append(f"  - {t['head']}：{t['tail']}")

    if result["causes"]:
        lines.append("\n## 因果关法")
        for t in result["causes"][:8]:
            lines.append(f"  - {t['head']} → {t['tail']}")

    if result["methods"]:
        lines.append("\n## 操作方法")
        for t in result["methods"][:8]:
            lines.append(f"  - {t['head']}：{t['tail']}")

    return "\n".join(lines)


def get_writing_tips(query: str, num: int = 10) -> list[str]:
    """快捷接口：返回纯文本写作技巧列表

    适配直接拼入 Writer / Editor / Reader 的 system prompt
    """
    result = query_graph(query, top_k=num * 3)
    tips: list[str] = []
    seen: set[str] = set()

    for tri in result.get("triples", []):
        h, r, t = tri["head"], tri["relation"], tri["tail"]
        # 跳过太短或太长的结果
        if len(t) < 4 or len(t) > 80:
            continue
        tip = f"{h}({r}): {t}"
        if tip not in seen:
            seen.add(tip)
            tips.append(tip)
        if len(tips) >= num:
            break

    return tips


# ════════════════════════════════════════════════════════════════
#  调试 / 测试
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # 自测
    for q in ["网文写作技巧", "番茄平台快节奏", "如何写好小说开头", "金手指设定方法"]:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        print(f"{'='*60}")
        r = query_graph(q, top_k=15)
        if r.get("seed_nodes"):
            print(f"种子节点: {r['seed_nodes']}")
        if r.get("text_summary"):
            print(r["text_summary"][:800])
        print()
