"""知识图谱查询工具 —— 按 Writer 模块分类抽取 KG 三元组，转为自然语言提示

设计要点：
- 每个模块有专属的关键词集合，用于从图检索器中精准抽取
- 三元组 → 自然语言转述（简洁规则列表，每条 1-2 句）
- 末尾附加"创意平衡提示"，防止 LLM 教条化
- 输出格式统一：模块名 → 可直接注入 System Prompt 的文本块

用法：
  from src.utils.kg_query import get_module_kg
  rules = get_module_kg("rhythm")  # 返回纯文本规则
"""

from __future__ import annotations

from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── 尝试加载知识图谱检索器 ──────────────────────────────────────
try:
    from src.utils.graph_retriever import query_graph as _kg_query
    _KG_AVAILABLE = True
except Exception:
    _KG_AVAILABLE = False
    def _kg_query(*args: Any, **kwargs: Any) -> dict:  # type: ignore[no-redef]
        return {}


# ════════════════════════════════════════════════════════════════
#  模块 → 关键词映射
# ════════════════════════════════════════════════════════════════

_MODULE_KEYWORDS: dict[str, list[str]] = {
    # ① 情绪曲线规划器 —— 节奏、爽点、高潮
    "rhythm": [
        "节奏", "爽点", "高潮", "冲突", "情绪", "波折",
        "张弛", "起承转合", "频率", "密度",
    ],
    # ② 钩子系统 —— 开头、开篇、钩子、悬念
    "hook": [
        "开头", "开篇", "钩子", "开场", "悬念", "期待",
        "凤头", "开头一万字", "书名简介", "作品简介",
    ],
    # ③ 世界观构建器 —— 世界观、力量体系、设定、境界
    "worldbuilding": [
        "世界观", "力量体系", "境界", "修炼", "设定",
        "势力", "背景设定", "人物设定", "总体设定",
    ],
    # ⑤ 对话口吻管理器 —— 对话、口吻、对白
    "dialogue": [
        "对话", "口吻", "对白", "对话流", "角色语气",
    ],
    # ⑥ 正文起草 —— 场景、描写、节奏、渲染
    "drafting": [
        "描写", "场景", "叙事", "渲染", "细节描写",
        "语言描写", "心理描写", "段落", "行文",
    ],
    # ⑧ 自检 —— 避免、禁忌、错误、注意
    "selfcheck": [
        "避免", "禁忌", "错误", "注意", "不要",
        "退稿", "失败", "漏洞",
    ],
    # ⑨ 读者评估 —— 弃书因果、爽点密度、阅读体验
    "reader": [
        "弃书", "读者", "流失", "爽点", "追读", "留存",
        "吸引力", "阅读体验", "代入感", "紧迫感", "满意度",
        "高潮", "冲突", "节奏", "钩子", "悬念",
    ],
    # ⑩ 编辑审校 —— 违规检测、一致性、风格校验
    "editor": [
        "避免", "禁忌", "错误", "注意", "不要",
        "节奏", "爽点", "被动", "口吻", "一致性",
        "段落", "描写", "逻辑", "转折", "冲突",
        "对话", "视角", "风格", "矛盾", "退稿",
        "角色", "行为", "设定", "违反", "漏洞",
    ],
}


# ════════════════════════════════════════════════════════════════
#  核心函数：按模块获取 KG 文本
# ════════════════════════════════════════════════════════════════

def get_module_kg(module: str) -> str:
    """传入模块名，返回该模块专属的结构化提示文本

    Args:
        module: 模块标识，可选值：
            "rhythm"       → 情绪曲线规划器
            "hook"         → 钩子系统
            "worldbuilding"→ 世界观构建器
            "dialogue"     → 对话口吻管理器
            "drafting"     → 正文起草
            "selfcheck"    → 自检（质量否决项）
            "reader"       → 读者评估（弃书因果链 + 爽点评分）
            "editor"       → 编辑审校（硬规则检测 + 软校验理论）

    Returns:
        可直接注入 System Prompt 的自然语言文本块。
        KG 不可用时返回空字符串。
    """
    if not _KG_AVAILABLE:
        logger.debug("[KGQuery] KG 不可用，返回空")
        return ""

    keywords = _MODULE_KEYWORDS.get(module, [])
    if not keywords:
        logger.warning("[KGQuery] 未知模块: %s", module)
        return ""

    # 用模块关键词去查询图
    query_text = " ".join(keywords)
    kg_result = _kg_query(query_text, top_k=60)

    if not kg_result or not kg_result.get("triples"):
        logger.info("[KGQuery] 模块 '%s' 未匹配到 KG 数据", module)
        return ""

    # Reader 模块走专用格式化路径
    if module == "reader":
        abandon_signals, pleasure_criteria = _reader_triples_to_signals(kg_result)
        return _format_reader_block(abandon_signals, pleasure_criteria)

    # Editor 模块走专用格式化路径
    if module == "editor":
        hard_rules, soft_rules = _editor_triples_to_checklist(kg_result)
        return _format_editor_block(hard_rules, soft_rules)

    # 通用模块：按关系类型抽取，转为自然语言规则
    rules = _triples_to_rules(kg_result, module)
    if not rules:
        return ""

    # 组装最终文本
    return _format_module_block(module, rules)


def _triples_to_rules(kg_result: dict, module: str) -> list[str]:
    """从 KG 查询结果中提取并转述为简洁规则列表

    策略：
    - query_graph 已通过种子节点匹配完成相关性筛选，此处不再二次关键词过滤
    - 仅过滤噪音（非写作相关、过短/过长）
    - 按关系类型→自然语言模板转述
    - 高价值类型（定义/要求/建议/避免/技巧/方法/原则/公式）优先
    """
    rules: list[str] = []
    seen: set[str] = set()

    # 关系 → 自然语言模板
    _TEMPLATES = {
        "定义":        "「{h}」是指：{t}",
        "要求":        "「{h}」必须：{t}",
        "建议":        "建议「{h}」：{t}",
        "避免":        "避免「{h}」：{t}",
        "要点":        "「{h}」的要点：{t}",
        "关键":        "「{h}」的关键：{t}",
        "注意":        "注意「{h}」：{t}",
        "技巧":        "「{h}」的写作技巧：{t}",
        "方法":        "「{h}」的实现方法：{t}",
        "原则":        "「{h}」的原则：{t}",
        "特征":        "「{h}」的特征：{t}",
        "特点":        "「{h}」的特点：{t}",
        "作用":        "「{h}」的作用：{t}",
        "导致":        "「{h}」会导致：{t}",
        "目的是":       "「{h}」的目的是：{t}",
        "公式":        "「{h}」的公式：{t}",
    }

    # 高价值关系类型（按模块可能有侧重）
    _HIGH_VALUE = {"定义", "要求", "建议", "避免", "要点", "关键",
                   "技巧", "方法", "原则", "公式"}
    _MED_VALUE = {"注意", "特征", "特点", "作用", "导致", "目的是"}
    _LOW_VALUE = {"包括", "包含", "需要", "分为", "属于", "分类", "指", "用于",
                  "提高", "增加", "影响", "原因", "好处", "应用", "例如"}

    def _is_noisy(head: str, tail: str) -> bool:
        """过滤噪音三元组"""
        if not head or not tail:
            return True
        if len(tail) < 4 or len(tail) > 80:
            return True
        # 非写作相关噪音
        noise_words = ["地震", "火警", "消防", "安全疏散", "电梯", "食品",
                       "家长", "小学生", "Guru", "吸毒", "监狱", "医疗",
                       "消防队", "急救", "炸药", "枪", "逮捕", "报警",
                       "付款", "支付", "违约金", "协议", "合同"]
        combined = f"{head}{tail}"
        if any(nw in combined for nw in noise_words):
            return True
        # 过于抽象、无具体内容
        if head in ("广告", "自我保护", "孩子", "人员"):
            return True
        # 已知作者笔名（"云天空建议专心码字"这种对写作指导无意义）
        _PEN_NAMES = {"云天空", "柳下挥", "夜雨", "乌山云雨", "蒋", "四爷",
                      "隐为者", "写手", "小说作者", "新手写手", "成功写手"}
        if head in _PEN_NAMES:
            return True
        # tail 是纯数字/百分比
        import re
        if re.match(r'^[\d.,%]+$', tail.strip()):
            return True
        return False

    def _relation_priority(rel: str) -> int:
        if rel in _HIGH_VALUE:
            return 0
        if rel in _MED_VALUE:
            return 1
        return 2

    # 从所有分类中提取三元组
    all_items: list[dict] = []
    for category_key in ["definitions", "requirements", "suggestions",
                          "causes", "methods", "structure", "other"]:
        items = kg_result.get(category_key, [])
        if items:
            all_items.extend(items)
    # 兜底：直接从 triples 取
    if not all_items:
        all_items = kg_result.get("triples", [])

    candidates: list[tuple[int, str, str, str]] = []

    for t in all_items:
        if not isinstance(t, dict):
            continue
        h = (t.get("head") or "").strip()
        r = (t.get("relation") or "").strip()
        tl = (t.get("tail") or "").strip()
        if _is_noisy(h, tl):
            continue
        tmpl = _TEMPLATES.get(r)
        if not tmpl:
            continue
        prio = _relation_priority(r)
        candidates.append((prio, r, h, tl))

    # 按优先级排序，同优先级头节点短的优先
    candidates.sort(key=lambda x: (x[0], len(x[2])))

    max_rules = {"selfcheck": 12, "drafting": 10, "rhythm": 10,
                 "hook": 10, "worldbuilding": 8, "dialogue": 8}.get(module, 8)

    for prio, rel, head, tail in candidates:
        tmpl = _TEMPLATES.get(rel, "{h}：{t}")
        rule = tmpl.format(h=head, t=tail)
        dedup_key = f"{head}|{rel}"
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        rules.append(rule)
        if len(rules) >= max_rules:
            break

    return rules


def _reader_triples_to_signals(kg_result: dict) -> tuple[list[str], list[str]]:
    """Reader 专用：将 KG 三元组转为弃书信号 + 爽点参考

    Returns:
        (abandon_signals, pleasure_criteria) — 两个列表
    """
    abandon_signals: list[str] = []
    pleasure_criteria: list[str] = []
    seen_abandon: set[str] = set()
    seen_pleasure: set[str] = set()

    # 已知作者笔名 / 非写作概念头节点
    _PEN_NAMES = {"云天空", "柳下挥", "夜雨", "乌山云雨", "蒋", "四爷",
                  "隐为者", "写手", "小说作者", "新手写手", "成功写手",
                  "老作家", "自由撰稿人", "失落叶", "没有点子", "想清楚"}

    # 弃书相关 tail 关键词（用于筛选因果链）
    _ABANDON_KW = {"读者", "弃书", "流失", "留存", "弃", "收藏", "订阅",
                   "追读", "点击率", "点击", "留存率", "弃文"}

    # 从所有分类中提取三元组
    all_items: list[dict] = []
    for category_key in ["causes", "suggestions", "definitions",
                          "requirements", "methods", "structure", "other"]:
        items = kg_result.get(category_key, [])
        if items:
            all_items.extend(items)
    if not all_items:
        all_items = kg_result.get("triples", [])

    for t in all_items:
        if not isinstance(t, dict):
            continue
        h = (t.get("head") or "").strip()
        r = (t.get("relation") or "").strip()
        tl = (t.get("tail") or "").strip()
        if not h or not tl or len(tl) < 4 or len(tl) > 80:
            continue

        # 过滤笔名 / 噪音头节点
        if h in _PEN_NAMES:
            continue

        combined = f"{h}{tl}"
        # 噪音过滤
        if any(nw in combined for nw in [
            "地震", "火警", "消防", "安全疏散", "电梯", "食品",
            "家长", "小学生", "吸毒", "监狱", "医疗", "炸药",
            "枪", "逮捕", "报警", "付款", "支付", "违约金",
        ]):
            continue

        if r in ("导致", "影响"):
            # 仅保留 tail 中明确提到读者/弃书相关概念的因果链
            if not any(kw in tl for kw in _ABANDON_KW):
                continue
            key = f"{h}|{r}"
            if key not in seen_abandon:
                seen_abandon.add(key)
                abandon_signals.append(f"若出现「{h}」→ {tl}")
        elif r in ("建议", "技巧", "要点", "关键", "方法", "原则", "要求",
                   "定义", "特点", "特征"):
            key = f"{h}|{r}"
            if key not in seen_pleasure:
                seen_pleasure.add(key)
                pleasure_criteria.append(f"「{h}」：{tl}")

    return abandon_signals[:8], pleasure_criteria[:8]


def _format_reader_block(abandon_signals: list[str],
                         pleasure_criteria: list[str]) -> str:
    """将 Reader 信号列表格式化为评估提示文本块"""
    parts: list[str] = []

    if abandon_signals:
        parts.append("【弃书因果链】以下信号出现时，对应提高弃书风险：")
        parts.extend(f"  • {s}" for s in abandon_signals)

    if pleasure_criteria:
        parts.append("\n【爽点密度参考标准】请根据以下维度打分（1-5）：")
        parts.extend(f"  • {s}" for s in pleasure_criteria)

    if not parts:
        return ""

    parts.append(
        "\n  ⚠ 以上标准为评估参考锚点，不是机械检查表。"
        "最终需结合文本整体阅读体验给出综合判断，避免逐条对号入座。"
    )
    return "\n".join(parts)


def _editor_triples_to_checklist(kg_result: dict) -> tuple[list[str], list[str]]:
    """Editor 专用：将 KG 三元组转为硬规则检测项 + 软校验参考

    Returns:
        (hard_rules, soft_rules) — 硬规则列表 + 软校验理论列表
    """
    hard_rules: list[str] = []
    soft_rules: list[str] = []
    seen: set[str] = set()

    # 已知笔名过滤
    _PEN_NAMES = {"云天空", "柳下挥", "夜雨", "乌山云雨", "蒋", "四爷",
                  "隐为者", "写手", "小说作者", "新手写手", "成功写手",
                  "老作家", "自由撰稿人", "失落叶", "唐川先生",
                  "罗伯特·谢克里"}

    # 从所有分类中提取三元组
    all_items: list[dict] = []
    for cat in ["suggestions", "requirements", "causes", "definitions",
                "methods", "structure", "other"]:
        items = kg_result.get(cat, [])
        if items:
            all_items.extend(items)
    if not all_items:
        all_items = kg_result.get("triples", [])

    for t in all_items:
        if not isinstance(t, dict):
            continue
        h = (t.get("head") or "").strip()
        r = (t.get("relation") or "").strip()
        tl = (t.get("tail") or "").strip()
        if not h or not tl or len(tl) < 4 or len(tl) > 80:
            continue
        if h in _PEN_NAMES:
            continue
        combined = f"{h}{tl}"
        if any(nw in combined for nw in [
            "地震", "火警", "消防", "电梯", "食品", "家长",
            "小学生", "吸毒", "监狱", "医疗", "炸药", "枪",
            "逮捕", "报警", "付款", "违约金",
        ]):
            continue

        key = f"{h}|{r}"
        if key in seen:
            continue
        seen.add(key)

        # 硬规则：避免/禁忌/要求/注意 → 可编程检测项
        if r in ("避免", "禁忌", "要求", "注意"):
            hard_rules.append(f"「{h}」规则：{tl}")
        # 软规则：建议/技巧/要点/关键/原则/定义 → 风格/一致性参考
        elif r in ("建议", "技巧", "要点", "关键", "原则", "定义",
                    "特点", "特征", "导致", "作用"):
            soft_rules.append(f"「{h}」→ {tl}")

    return hard_rules[:10], soft_rules[:10]


def _format_editor_block(hard_rules: list[str],
                         soft_rules: list[str]) -> str:
    """将 Editor 检测清单格式化为审校提示文本块"""
    parts: list[str] = []

    if hard_rules:
        parts.append("## 硬规则检测项（代码层可编程扫描）")
        parts.append("以下规则已通过代码层自动检测，结果见 tech_issues 中的 rule 来源：")
        parts.extend(f"  - [{i+1}] {r}" for i, r in enumerate(hard_rules))

    if soft_rules:
        parts.append("\n## 软校验参考（LLM 辅助判断 + KG 理论支撑）")
        parts.append("以下写作理论供你审校时参考，发现违规时输出 structured tech_issue：")
        parts.extend(f"  - {r}" for r in soft_rules)

    if not parts:
        return ""

    parts.append(
        "\n  ⚠ 以上规则为审校参考锚点。硬规则优先用代码扫描，"
        "软规则结合文本语境灵活判断，避免对号入座式打勾。"
    )
    return "\n".join(parts)


def _format_module_block(module: str, rules: list[str]) -> str:
    """将规则列表格式化为最终的提示文本块"""
    # 模块标题
    _TITLES = {
        "rhythm": "【情绪节奏规则】",
        "hook": "【开篇钩子原则】",
        "worldbuilding": "【世界观构建规则】",
        "dialogue": "【对话写作规范】",
        "drafting": "【正文起草要点】",
        "selfcheck": "【质量否决项】",
    }
    title = _TITLES.get(module, "【写作规则】")

    # 格式化规则（每条一行，• 开头）
    rule_lines = "\n".join(f"  • {r}" for r in rules)

    # 创意平衡提示
    balance_note = (
        "\n\n  ⚠ 以上规则为写作质量的质量锚点，不是绝对限制。"
        "遇到戏剧张力与规则冲突时，优先保证阅读爽感。"
    )

    # selfcheck 模块用更明确的"如果出现 X 则标记 issue"格式
    if module == "selfcheck":
        check_lines = []
        for r in rules:
            check_lines.append(f"  • 若出现 [ {r} ] → 标记为硬性质量问题")
        rule_lines = "\n".join(check_lines)
        balance_note = ""

    return f"{title}\n{rule_lines}{balance_note}"


# ════════════════════════════════════════════════════════════════
#  便捷接口：一次获取所有模块的 KG
# ════════════════════════════════════════════════════════════════

def get_all_module_kg() -> dict[str, str]:
    """返回所有 6 个模块的 KG 文本块字典"""
    modules = ["rhythm", "hook", "worldbuilding", "dialogue", "drafting", "selfcheck"]
    result: dict[str, str] = {}
    for m in modules:
        result[m] = get_module_kg(m)
    return result


def get_reader_kg() -> str:
    """便捷接口：获取 Reader 评估专用的 KG 规则文本"""
    return get_module_kg("reader")


def get_editor_kg() -> str:
    """便捷接口：获取 Editor 审校专用的 KG 规则文本"""
    return get_module_kg("editor")


# ════════════════════════════════════════════════════════════════
#  调试 / 测试
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    for mod in ["rhythm", "hook", "worldbuilding", "dialogue", "drafting", "selfcheck"]:
        print(f"\n{'='*60}")
        print(f"  Module: {mod}")
        print(f"{'='*60}")
        text = get_module_kg(mod)
        if text:
            print(text)
        else:
            print("  (无匹配)")
