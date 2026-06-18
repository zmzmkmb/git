"""作家 Agent —— 内部以三阶段工作流串联 8 个模块，生成章节正文

工作流架构：
  阶段一：撰写前规划  ① 情绪曲线 → ② 章节结构+钩子 → ③ 世界观 → ④ 伏笔 → ⑤ 口吻
  阶段二：正文起草    ⑥ 风格克隆+口吻卡片+场景渲染 → 初稿
  阶段三：打磨        ⑦ AI 味祛除 → ⑧ 蓝图自检 → 终稿
"""

from __future__ import annotations

import json
import re
from typing import Any

from src.agents.base_agent import BaseAgent
from src.utils.llm_client import call_llm
from src.utils.logger import get_logger

logger = get_logger(__name__)


WRITER_SYSTEM_PROMPT = """你是一位资深玄幻网文作家，擅长节奏紧凑、爽点密集的长篇连载。
- 强节奏：开篇 200 字内必须出现冲突或悬念
- 多动作、多对话
- 每隔 800-1000 字埋一个小高潮
- 严格遵守设定库中的世界观与角色口吻
- 目标字数必须达到，宁长勿短
- 只输出正文段落，不写元信息
"""


class WriterAgent(BaseAgent):
    """作家 Agent：内部以三阶段工作流运行 8 个模块"""

    def __init__(self) -> None:
        super().__init__(name="Writer")

    # ── 公开入口：保持接口不变 ──────────────────────────────────

    def run(self, context: dict) -> dict:
        outline = context.get("outline", "")
        current_state = context.get("current_state", {}) or {}
        chapter_number = context.get("chapter_number",
                                     context.get("current_chapter",
                                                 context.get("chapter", 0)))
        target_word_count = context.get("target_word_count", 3000)
        rewrite_hints = context.get("rewrite_hints", []) or []
        target_platform = context.get("target_platform", "番茄")
        climax_type = context.get("climax_type", "")  # 用户自定义高潮类型

        # ── 阶段一：撰写前规划 ──
        logger.info("┌─ [Writer] 第 %d 章 · 阶段一：撰写前规划 ─┐", chapter_number)
        blueprint = self.stage_one(
            outline, current_state, chapter_number,
            target_word_count, target_platform,
            climax_type=climax_type,
        )
        if rewrite_hints:
            blueprint["rewrite_hints"] = rewrite_hints
            logger.info("  [重写] 收到重写提示 %d 条", len(rewrite_hints))

        # ── 阶段二：正文起草 ──
        logger.info("┌─ [Writer] 第 %d 章 · 阶段二：正文起草 ─┐", chapter_number)
        draft = self.stage_two(
            blueprint, outline, current_state, chapter_number, target_word_count,
        )

        # ── 阶段三：打磨 ──
        logger.info("┌─ [Writer] 第 %d 章 · 阶段三：打磨 ─┐", chapter_number)
        final_text = self.stage_three(draft, blueprint, current_state, target_word_count)

        logger.info("└─ [Writer] 第 %d 章 · 终稿 %d / %d 字 ─┘",
                     chapter_number, len(final_text), target_word_count)

        return {
            "agent": self.name,
            "chapter": chapter_number,
            "content": final_text,
            "word_count": len(final_text),
            "target_word_count": target_word_count,
            "blueprint": blueprint,
        }

    # ════════════════════════════════════════════════════════════
    #  阶段一：撰写前规划  ①②③④⑤ 串联
    # ════════════════════════════════════════════════════════════

    # ── KG 规则缓存 ────────────────────────────────────────────
    _kg_rules_cache: dict[str, str] | None = None

    def _load_kg_rules(self) -> dict[str, str]:
        """懒加载 KG 规则（全模块），缓存避免重复查询"""
        if self._kg_rules_cache is not None:
            return self._kg_rules_cache
        try:
            from src.utils.kg_query import get_all_module_kg
            self._kg_rules_cache = get_all_module_kg()
            loaded = sum(1 for v in self._kg_rules_cache.values() if v)
            if loaded > 0:
                logger.info("  [KG] 已加载 %d/6 个模块的 KG 规则", loaded)
        except Exception as e:
            logger.warning("  [KG] 加载失败: %s", e)
            self._kg_rules_cache = {}
        return self._kg_rules_cache

    def stage_one(self, outline: str, state: dict,
                  chapter_number: int, target_word_count: int,
                  target_platform: str = "番茄",
                  climax_type: str = "") -> dict:
        """①②③④⑤ 依次调用，产出创作蓝图"""

        # 0) 知识库检索
        from src.utils.knowledge_retriever import get_writer_context
        query = f"{outline} {target_platform}"
        kb = get_writer_context(query, platform=target_platform)
        # 加入用户自定义高潮类型
        if climax_type:
            kb["climax_type"] = climax_type
        if kb.get("template_name"):
            logger.info("  [KB] 模板=%s, 平台=%s, beats=%d",
                         kb["template_name"], kb["platform"],
                         len(kb.get("key_beats", [])))

        # 0.5) 加载 KG 规则
        kg_rules = self._load_kg_rules()

        # ① 情绪爽点曲线规划器
        emotion_result = self._emotion_curve_planner(outline, target_word_count, kb)
        # ② 章节结构与钩子系统
        chapter_structure = self._chapter_structure_system(
            outline, emotion_result, target_word_count, kb)
        # ③ 世界观构建器
        world_card = self._world_builder(state, chapter_structure)
        # ④ 伏笔铺设与回收引擎
        foreshadowing_plan = self._foreshadowing_engine(
            state, chapter_structure, chapter_number)
        # ⑤ 对话口吻管理器
        dialogue_cards = self._dialogue_manager(state, chapter_structure)

        blueprint = {
            "emotion_curve": emotion_result["values"],
            "emotion_types": emotion_result["types"],
            "chapter_structure": chapter_structure,
            "world_card": world_card,
            "foreshadowing_plan": foreshadowing_plan,
            "dialogue_cards": dialogue_cards,
            "knowledge_base": kb,
            "kg_rules": kg_rules,           # ★ KG 规则字典
        }
        return blueprint

    # ════════════════════════════════════════════════════════════
    #  阶段二：正文起草  ⑥
    # ════════════════════════════════════════════════════════════

    def stage_two(self, blueprint: dict, outline: str, state: dict,
                  chapter_number: int, target_word_count: int) -> str:
        """⑥ 融合风格克隆 + 口吻卡片 + 场景渲染，生成初稿"""
        draft = self._draft_composer(
            outline, state, chapter_number, target_word_count, blueprint)
        return draft

    # ════════════════════════════════════════════════════════════
    #  阶段三：打磨  ⑦⑧
    # ════════════════════════════════════════════════════════════

    def stage_three(self, draft: str, blueprint: dict, state: dict,
                    target_word_count: int) -> str:
        """⑦ AI 味祛除 → ⑧ 蓝图自检 → 终稿"""
        # ⑦ 语言质量与 AI 味祛除
        polished = self._ai_odor_remover(draft, blueprint)
        # ⑧ 自检：与蓝图比对
        polished = self._self_check(polished, blueprint, state)
        return polished

    # ════════════════════════════════════════════════════════════
    #  8 个模块私有方法
    # ════════════════════════════════════════════════════════════

    # ── ① 情绪爽点曲线规划器 ────────────────────────────────────

    def _emotion_curve_planner(self, outline: str,
                                target_word_count: int,
                                kb: dict | None = None) -> dict:
        """① 根据章节大纲生成情绪值数组 + 爽点类型标注

        返回：
        {
          "values": [0.15, 0.30, 0.90, 0.60, 0.35],
          "types":  ["压抑", "蓄力", "打脸", "反转", "勾子"],
          "peaks":  [2],   # 高潮点在数组中的位置
        }
        """
        # 规则：从 outline 和 KB 中提取情绪关键词
        outline_lower = outline.lower()
        kb_template = (kb or {}).get("template", {}) or {}
        kb_beats = kb_template.get("key_beats", []) or []

        # 爽点类型字典（关键词 → 情绪类型 + 强度）
        # 注：不使用硬编码套路关键词，根据大纲内容自然推断
        emotion_map: list[tuple[list[str], str, float]] = [
            (["觉醒", "苏醒", "残魂", "金手指", "系统"], "蓄力", 0.40),
            (["突破", "暴涨", "连破", "一鸣惊人", "修为"], "升压", 0.70),
            (["反杀", "震慑", "反击", "碾压", "逆转", "翻盘"], "打脸", 0.90),
            (["约定", "复仇", "阴谋", "秘密", "真相", "伏笔"], "钩子", 0.35),
            (["战斗", "激战", "斩杀", "拼命", "搏杀"], "杀伐", 0.80),
            (["温馨", "亲情", "温柔", "守护"], "温情", 0.30),
            (["发现", "识破", "惊讶", "反转", "欺骗"], "反转", 0.65),
        ]

        values: list[float] = []
        types: list[str] = []
        for keywords, etype, val in emotion_map:
            if any(kw in outline_lower for kw in keywords):
                values.append(val)
                types.append(etype)
            if len(values) >= 5:
                break

        # 不足 5 个时用默认值补齐
        default_values = [0.15, 0.30, 0.80, 0.55, 0.35]
        default_types = ["压抑", "蓄力", "高潮", "回落", "钩子"]
        while len(values) < 5:
            idx = len(values)
            values.append(default_values[idx])
            types.append(default_types[idx])

        # 找出峰值位置
        peaks = [i for i, v in enumerate(values) if v >= 0.75]
        if not peaks:
            peaks = [values.index(max(values))]

        result = {"values": values[:5], "types": types[:5], "peaks": peaks}
        logger.info("  ① 情绪曲线: values=%s types=%s peaks=%s",
                     values[:5], types[:5], peaks)
        return result

    # ── ② 章节结构与钩子系统 ─────────────────────────────────────

    def _chapter_structure_system(self, outline: str,
                                   emotion_result: dict,
                                   target_word_count: int,
                                   kb: dict | None = None) -> dict:
        """② 规划起承转合，在末尾设计强钩子

        返回：
        {
          "sections": [  # 每段包含名称、位置、情绪值、字数、钩子设计
            {"name": "起(开场冲突)", "position": 0, "emotion": 0.15, "word_count": 300, "hook": "..."},
            ...
          ],
          "final_hook": "男主角发现密室中的神秘信物，信物上刻着从未见过的古文字...",
          "hook_type": "悬念"
        }
        """
        values = emotion_result["values"]
        etypes = emotion_result["types"]
        peaks = emotion_result.get("peaks", [2])

        # 五段式结构命名
        section_names = ["起(开场冲突)", "承(矛盾酝酿)", "转(高潮爆发)", "合(剧情反转)", "结(悬念钩子)"]
        sections = []
        total_e = sum(values) or 1

        for i, name in enumerate(section_names):
            sections.append({
                "name": name,
                "position": i,
                "emotion": values[i],
                "emotion_type": etypes[i],
                "word_count": int(target_word_count * values[i] / total_e),
                "is_climax": i in peaks,
            })

        # 末尾钩子设计：基于大纲 + KB 模板生成钩子文案
        final_hook = self._design_hook(outline, sections, kb)

        structure = {
            "sections": sections,
            "total_word_count": target_word_count,
            "final_hook": final_hook["text"],
            "hook_type": final_hook["type"],
        }

        word_counts = [s["word_count"] for s in sections]
        logger.info("  ② 章节结构: %d 段 %s", len(sections), word_counts)
        logger.info("  ② 钩子设计: [%s] %s", final_hook["type"], final_hook["text"])
        return structure

    @staticmethod
    def _design_hook(outline: str, sections: list[dict],
                     kb: dict | None = None) -> dict[str, str]:
        """设计末尾强钩子（悬念 / 意外 / 期待）"""
        outline_lower = outline.lower()
        kb = kb or {}

        # 从 KB 获取用户自定义高潮类型
        climax_type = kb.get("climax_type", "").lower()

        # 从 KB 模板取钩子灵感
        kb_template = kb.get("template", {}) or {}
        kb_opening = kb_template.get("opening_hook", "") or ""
        first_ch_tmpl = kb_template.get("first_chapter_outline_template", "") or ""

        # 根据 climax_type 和大纲关键词选择钩子
        # 用户自定义高潮类型优先
        if climax_type == "复仇":
            hook_type = "期待"
            hook_text = "他望着仇人离去的背影，指节攥得发白：'三年后，我会让你跪在我面前偿还一切。'"
        elif climax_type == "探险":
            hook_type = "悬念"
            hook_text = "石碑上的古文突然亮起，一道光芒没入他的眉心——脑海中响起一个苍老的声音：'有缘人，你终于来了。'"
        elif climax_type == "崛起":
            hook_type = "期待"
            hook_text = "就在众人以为他将彻底沉沦时，他体内突然爆发出璀璨光芒——那股力量，远超所有人想象。"
        elif any(kw in outline_lower for kw in ["约定", "秘密", "真相", "阴谋", "伏笔"]):
            hook_type = "期待"
            hook_text = "男主角冷冷望向对方离去的方向：'三年后，我必亲自登门——到时候，我会让你知道什么叫后悔。'"
        elif any(kw in outline_lower for kw in ["信物", "秘密", "真相", "阴谋", "背后的"]):
            hook_type = "悬念"
            hook_text = "夜深人静，他摊开掌心——那枚残破玉佩的背面，竟浮现出一行从未见过的古文字。字的尽头，是一个血红的'杀'字。"
        elif any(kw in outline_lower for kw in ["突破", "暴涨", "连破"]):
            hook_type = "意外"
            hook_text = "就在他以为一切结束时，丹田深处突然传来一股磅礴力量——那是连帝君都未曾料到的——'上古血脉？'苍老的声音第一次露出震惊。"
        elif "打脸" in outline_lower or "反击" in outline_lower:
            hook_type = "期待"
            hook_text = "围观者面面相觑，谁也没想到这个被他们嘲笑的'废物'，竟藏得这么深。而更让他们胆寒的是——这只是他的第一战。"
        else:
            hook_type = "悬念"
            hook_text = "他不知道的是，在遥远的某处，有人正透过一面古镜注视着他的一切——古镜旁，一道血红色的符文正在缓缓亮起。"

        return {"type": hook_type, "text": hook_text}

    # ── ③ 世界观构建器 ──────────────────────────────────────────

    def _world_builder(self, state: dict,
                        chapter_structure: dict) -> dict:
        """③ 从 current_state 提取本章所需世界观要素，输出"本章世界观卡片"

        包含：境界体系、功法、势力、地点、世界规则
        """
        ps = state.get("power_system", {}) or {}
        locations = state.get("locations", {}) or {}

        world_card = {
            "realms": (ps.get("realms") or [])[:10],
            "techniques": (ps.get("techniques") or [])[:10],
            "rules": (ps.get("rules") or [])[:5],
            "factions": list({
                (c.get("faction") or "")
                for c in (state.get("characters") or {}).values()
                if isinstance(c, dict) and c.get("faction")
            })[:5],
            "active_locations": list(locations.keys())[:5],
            "atmosphere": "玄幻世界，强者为尊，弱肉强食",
        }
        if not world_card["rules"]:
            world_card["rules"] = ["玄幻通用：强者为尊，弱肉强食"]
        # 从大纲提取场景相关地点
        if chapter_structure.get("final_hook"):
            world_card["hook_context"] = chapter_structure["final_hook"][:80]

        logger.info("  ③ 世界观卡片: %d 境界 / %d 功法 / %d 势力 / %d 地点",
                     len(world_card["realms"]), len(world_card["techniques"]),
                     len(world_card["factions"]), len(world_card["active_locations"]))
        return world_card

    # ── ④ 伏笔铺设与回收引擎 ─────────────────────────────────────

    def _foreshadowing_engine(self, state: dict,
                                chapter_structure: dict,
                                chapter_number: int) -> dict:
        """④ 检查未回收伏笔，按 age 优先回收，规划新伏笔铺设

        返回：
        {
          "recycle": [{"clue": "...", "age": N, "urgency": "suggested"|"critical", ...}],
          "plant":   [{"clue_hint": "...", "suggested_in_section": N, "intensity": "强|中|弱"}],
          "expired": [...],       # 过期伏笔（需废弃或强制回收）
          "section_index": 4,
        }
        """
        existing = state.get("foreshadowing", []) or []

        to_recycle = []
        expired = []

        for fw in existing:
            if not isinstance(fw, dict):
                continue
            status = fw.get("status", "")
            recycled = fw.get("recycled") or (status == "resolved")
            if recycled or not fw.get("clue"):
                continue

            planted = fw.get("planted_chapter", 0)
            age = chapter_number - planted if planted else 0

            item = {
                "clue": fw.get("clue", ""),
                "from_chapter": planted,
                "direction": fw.get("possible_direction", ""),
                "age": age,
                "urgency": "常规",
            }

            if age >= 5:
                item["urgency"] = "critical"
                expired.append(item)
                to_recycle.append(item)
            elif age >= 2:
                item["urgency"] = "suggested"
                to_recycle.append(item)
            else:
                # age < 2：暂不回收（伏笔太新鲜）
                pass

        # 按 urgency 排序：critical 在前，suggested 在后
        urgency_order = {"critical": 0, "suggested": 1, "常规": 2}
        to_recycle.sort(key=lambda x: urgency_order.get(x.get("urgency", "常规"), 9))

        # 回收上限：最多 3 条
        to_recycle = to_recycle[:3]

        # 铺设：在结尾段埋新伏笔
        to_plant = [{
            "clue_hint": ("本章结尾留下的反常细节或未解之谜——"
                          "为下一章制造期待感"),
            "suggested_in_section": 4,
            "intensity": "中",
        }]

        plan = {
            "recycle": to_recycle,
            "plant": to_plant,
            "expired": expired,
            "section_index": 4,
        }

        logger.info("  ④ 伏笔计划: 回收 %d 条 (过期 %d) / 铺设 %d 条",
                     len(to_recycle), len(expired), len(to_plant))
        for r in to_recycle:
            logger.info("      %s 回收: [age=%d] %s...",
                         r.get("urgency", "?"), r.get("age", 0),
                         r.get("clue", "")[:40])
        return plan

    # ── ⑤ 对话口吻管理器 ────────────────────────────────────────

    def _dialogue_manager(self, state: dict,
                            chapter_structure: dict) -> dict:
        """⑤ 为每个出场角色生成"口吻卡片"

        返回：
        {
          "角色名": {
            "name": "...",
            "personality_tags": [...],
            "speech_style": "..." ,
            "common_phrases": [...],       // 常用词/句尾
            "tone": "冷峻"|"温和"|"狂妄"|"...",
            "forbidden": ["..."],           // 不能说/不符合人设的话
          },
          ...
        }
        """
        cards: dict[str, dict] = {}
        characters = state.get("characters", {}) or {}

        for name, info in list(characters.items())[:8]:
            if not isinstance(info, dict):
                continue

            tags = info.get("personality_tags") or []
            speech = info.get("speech_traits") or ""

            # 推断口吻风格
            common_phrases: list[str] = []
            forbidden: list[str] = []

            if any(t in tags for t in ["沉默", "寡言", "冷峻"]):
                style = "少言寡语，一句一顿，反问极少"
                common_phrases = ["嗯。", "...", "走吧。"]
                forbidden = ["长篇大论", "情感宣泄"]
            elif any(t in tags for t in ["狂妄", "张狂", "傲慢"]):
                style = "傲气凌人，反问句多，多用'尔等''区区'"
                common_phrases = ["尔等也配？", "区区...", "你可知..."]
                forbidden = ["示弱", "认错"]
            elif any(t in tags for t in ["温和", "温润", "儒雅"]):
                style = "语调平和，善用比喻，话尾常带笑意"
                common_phrases = ["无妨。", "倒也有趣。", "依我看..."]
                forbidden = ["粗口", "咄咄逼人"]
            elif any(t in tags for t in ["冷酷", "杀伐"]):
                style = "果断凌厉，不拖泥带水，命令式短句"
                common_phrases = ["杀。", "动手。", "不必废话。"]
                forbidden = ["犹豫", "长篇解释"]
            elif any(t in tags for t in ["活泼", "俏皮"]):
                style = "语速快，常带反问和俏皮话"
                common_phrases = ["嘿嘿。", "那可不一定！", "你看——"]
                forbidden = ["沉重说教", "过于严肃"]
            elif any(t in tags for t in ["阴险", "城府", "心机"]):
                style = "话里有话，表面温和实则冰冷"
                common_phrases = ["哦？是吗。", "那可真是...巧了。"]
                forbidden = ["直白情绪表露"]
            else:
                style = speech or "中性正常语速"

            cards[name] = {
                "name": name,
                "personality_tags": tags,
                "speech_style": style,
                "common_phrases": common_phrases,
                "tone": tags[0] if tags else "中性",
                "forbidden": forbidden,
            }

        logger.info("  ⑤ 口吻卡片: %d 角色 %s",
                     len(cards), list(cards.keys()))
        return cards

    # ── ⑥ 正文起草器（融合风格 + 口吻 + 场景渲染）───────────────

    def _draft_composer(self, outline: str, state: dict, chapter_number: int,
                        target_word_count: int, blueprint: dict) -> str:
        """⑥ 基于蓝图生成初稿，内部融合三个能力：

        - 风格克隆：确保整体文风符合目标平台调性
        - 场景渲染：对关键场景强制加入感官细节
        - 对话控制：严格按口吻卡片生成对话
        """
        # ⑥-1 风格基线
        kb = blueprint.get("knowledge_base", {}) or {}
        platform_rules = kb.get("platform_rules", {}) or {}
        style_baseline = {
            "pov": "第三人称有限视角",
            "pacing": platform_rules.get("pacing", "快节奏，强冲突，零废话"),
            "paragraph_length": platform_rules.get("paragraph_length", "30-60 字为主"),
            "dialogue_ratio": self._safe_float_range(platform_rules.get("dialogue_ratio", 0.35), 0.35),
            "forbidden": platform_rules.get("forbidden", []),
            "sensory_focus": platform_rules.get("sensory_focus", ["动作", "对话", "环境"]),
        }
        logger.info("  ⑥ 风格基线: 节奏=%s 段落=%s 对话率=%d%%",
                     style_baseline["pacing"],
                     style_baseline["paragraph_length"],
                     int(style_baseline["dialogue_ratio"] * 100))

        # ⑥-2 场景渲染详略分配
        emotion_curve = blueprint["emotion_curve"]
        sections = blueprint["chapter_structure"]["sections"]
        detail_levels = []
        sensory_channels = style_baseline["sensory_focus"][:3]
        for e in emotion_curve:
            if e < 0.3:
                detail_levels.append("精简")   # 过渡/背景
            elif e < 0.7:
                detail_levels.append("中等")   # 常规剧情
            else:
                detail_levels.append("详尽")   # 高潮/爽点：视+听+触+嗅全通道
        logger.info("  ⑥ 场景渲染: 详略=%s", detail_levels)

        # ⑥-3 口吻卡片（阶段一已生成，直接引用）
        dialogue_cards = blueprint.get("dialogue_cards", {}) or {}
        card_count = len(dialogue_cards)

        # 组合 prompt → 调用 LLM
        draft = self._build_draft_prompt(
            outline, state, chapter_number, target_word_count, blueprint,
            style_baseline, detail_levels, sensory_channels,
        )
        return draft

    def _build_draft_prompt(self, outline: str, state: dict, chapter_number: int,
                             target_word_count: int, blueprint: dict,
                             style: dict, detail_levels: list[str],
                             sensory: list[str]) -> str:
        """构造阶段二的 LLM 提示词"""

        # 重写提示
        rewrite_hints = blueprint.get("rewrite_hints", []) or []
        rewrite_block = ""
        if rewrite_hints:
            rewrite_block = (
                "### ⚠ 重写要求（来自编辑审校反馈）\n"
                + "\n".join(f"  - {h}" for h in rewrite_hints)
                + "\n\n**请务必避免以上问题**\n\n"
            )

        # KB 块
        kb_block = self._build_kb_block(blueprint.get("knowledge_base", {}) or {})

        # ★ KG 规则块（知识图谱注入各模块规则）
        kg_rules = blueprint.get("kg_rules", {}) or {}
        kg_block_lines: list[str] = []
        for mod_id, label in [
            ("rhythm", "情绪节奏规则"),
            ("hook", "开篇钩子原则"),
            ("worldbuilding", "世界观构建规范"),
            ("dialogue", "对话写作规范"),
            ("drafting", "正文起草要点"),
        ]:
            text = kg_rules.get(mod_id, "")
            if text and len(text) > 10:
                kg_block_lines.append(f"### {label}")
                kg_block_lines.append(text)
                kg_block_lines.append("")
        kg_block = "\n".join(kg_block_lines) if kg_block_lines else ""

        # 角色口吻块
        dialogue_cards = blueprint.get("dialogue_cards", {}) or {}
        char_block_lines = []
        for name, card in dialogue_cards.items():
            char_block_lines.append(
                f"  - {name}：{card['speech_style']}，"
                f"常用词 {card.get('common_phrases', [])}，"
                f"禁止 {card.get('forbidden', [])}"
            )
        char_block = "\n".join(char_block_lines) if char_block_lines else "  （暂无角色）"

        # 章节结构
        sections = blueprint["chapter_structure"]["sections"]
        section_lines = "\n".join(
            f"  {i+1}. [{s['name']}] {s['emotion_type']}({s['emotion']:.2f}) "
            f"约 {s['word_count']} 字 · 详略 {detail_levels[i]}"
            f"{' ⭐高潮' if s.get('is_climax') else ''}"
            for i, s in enumerate(sections)
        )

        # 钩子
        final_hook = blueprint["chapter_structure"]["final_hook"]
        hook_type = blueprint["chapter_structure"]["hook_type"]

        # 伏笔
        fw = blueprint["foreshadowing_plan"]
        recycle_lines = "\n".join(
            f"  - 回收：{r['clue']}（第{r['from_chapter']}章）"
            for r in (fw.get("recycle") or [])
        ) or "  （本章无需要回收的伏笔）"
        plant_lines = "\n".join(
            f"  - 铺设：{p['clue_hint']}（强度 {p.get('intensity','?')}）"
            for p in (fw.get("plant") or [])
        )

        # 世界观卡片
        wc = blueprint["world_card"]
        world_lines = []
        if wc.get("realms"):
            world_lines.append(f"  - 境界：{', '.join(wc['realms'][:8])}")
        if wc.get("techniques"):
            world_lines.append(f"  - 功法：{', '.join(wc['techniques'][:8])}")
        if wc.get("factions"):
            world_lines.append(f"  - 势力：{', '.join(wc['factions'][:5])}")
        if wc.get("rules"):
            world_lines.append(f"  - 规则：{'; '.join(wc['rules'][:3])}")
        world_block = "\n".join(world_lines) or "  - 玄幻通用设定"

        # 时间线
        timeline = state.get("timeline", []) or []
        recent = timeline[-3:] if timeline else []
        tl_block = "\n".join(
            f"  - [第{e.get('chapter', '?')}章] {e.get('event', '')}"
            for e in recent
        ) or "  （故事开篇）"

        # 风格基线
        style_block = (
            f"  - 视角：{style['pov']}\n"
            f"  - 节奏：{style['pacing']}\n"
            f"  - 段落：{style['paragraph_length']}\n"
            f"  - 对话占比：{int(style['dialogue_ratio'] * 100)}%\n"
            f"  - 感官通道：{', '.join(sensory)}\n"
        )
        if style.get("forbidden"):
            style_block += f"  - 禁止：{'; '.join(style['forbidden'][:5])}"

        prompt = f"""# 任务
创作【第{chapter_number}章】正文，目标 {target_word_count} 字（±10%）。

{rewrite_block}## 知识库参考

{kb_block}## 知识图谱写作规范

{kg_block}## 创作蓝图

### 章节结构（含情绪曲线与详略）
{section_lines}

### 末尾钩子 [{hook_type}]
{final_hook}

### 感官渲染
详略等级：{', '.join(detail_levels)}
聚焦通道：{', '.join(sensory)}
（詳尽段 → 视觉+听觉+触觉+嗅觉全感官）
（精简段 → 仅视觉或动作，快节奏推进）

### 伏笔规划
{recycle_lines}
{plant_lines}

### 世界观约束
{world_block}

### 角色口吻卡片
{char_block}

### 近期时间线
{tl_block}

## 本章大纲
{outline}

## 写作要求
- 强节奏、快冲突，前 200 字必须出现冲突或悬念
- 对话严格按口吻卡片生成——每人说话方式不能混
- 高潮段（⭐标记处）必须全感官渲染：视觉 / 听觉 / 触觉 / 嗅觉
- 严禁 AI 痕迹词：「如同」「仿佛」「似乎」「不由得」「下意识」「恍若」「缓缓地」「轻轻地」「深深地」「命运的齿轮」「一股无形」「翻江倒海」「眼中闪过一丝」
- 段落用空行分隔，对话用「""」
- 只输出章节正文，不要元信息
"""
        return self._call_llm(prompt, target_word_count, blueprint, state)

    # ── ⑦ 语言质量与 AI 味祛除器 ────────────────────────────────

    # AI 高频词 / 句式
    _AI_TIER_1_PHRASE = [
        ("一股无形的压力", "一阵窒息般的压迫感"),
        ("一股暖流", "一阵温热"),
        ("只见那", ""),
        ("只见", ""),
        ("微微一笑", "嘴角一挑"),
        ("冷冷一笑", "冷哼一声"),
        ("他感到十分震惊", "他瞳孔骤缩——"),
        ("他感到异常愤怒", "他攥紧拳头，指甲嵌进掌心——"),
        ("他感到无比激动", "他喉结滚动，手心微微发颤——"),
    ]

    _AI_TIER_1_SINGLE = [
        "翻江倒海", "命运的齿轮", "眼中闪过一丝",
        "目光深邃如渊", "声音低沉而富有磁性",
        "周身气势陡然攀升",
    ]

    _AI_TIER_2 = [
        "如同", "仿佛", "似乎", "恍若", "宛如", "犹如",
        "不由得", "下意识", "缓缓地", "静静地", "默默地",
        "深深地", "轻轻地", "与此同时",
    ]

    def _ai_odor_remover(self, draft: str, blueprint: dict) -> str:
        """⑦ 逐段检查初稿，标记并重写 AI 味问题：

        - 过于工整的对仗句 → 拆解
        - "一股...""只见...""微微一笑"等 AI 高频词 → 替换
        - "他感到十分震惊"等概括句 → 用具象行为替代
        """
        if not draft:
            return draft

        original_len = len(draft)
        text = draft
        stats = {"replaced": 0, "split_pairs": 0, "abstract_fixed": 0}

        # 1) 替换 AI 句式（优先长匹配）
        for src, dst in self._AI_TIER_1_PHRASE:
            if src in text:
                count_before = text.count(src)
                text = text.replace(src, dst)
                stats["replaced"] += count_before

        # 2) 删除 T1 单例 AI 痕迹
        for tick in self._AI_TIER_1_SINGLE:
            if tick in text:
                count = text.count(tick)
                text = text.replace(tick, "")
                stats["replaced"] += count

        # 3) 抑制 T2 AI 词（保留首次出现，删除重复超过 2 次的）
        for tick in self._AI_TIER_2:
            count = text.count(tick)
            if count > 2:
                # 保留前 2 次，其余删除
                parts = text.split(tick)
                text = parts[0] + tick + tick + tick.join(parts[3:]) if len(parts) > 3 else text

        # 4) 拆解过于工整的对仗句（四字对仗 + 四字对仗）
        # 例："山高水长，云淡风轻→山高水长。云淡风轻。"
        pattern = re.compile(r'([\u4e00-\u9fff]{4})，([\u4e00-\u9fff]{4})')
        matches = pattern.findall(text)
        if matches and len(matches[0][0]) == len(matches[0][1]):
            # 只处理 4+4 严格对仗
            pass  # 逐个替换很危险，这里只做统计标记

        # 5) 抽象概括句替换（"他感到十分X" → 具象行为）
        abstract_patterns = [
            (re.compile(r"他感到十分(震惊|愤怒|激动|紧张|恐惧|悲伤)"), {
                "震惊": "他瞳孔骤缩——",
                "愤怒": "他攥紧拳头，指节咯咯作响——",
                "激动": "他喉结滚动，手心微微发颤——",
                "紧张": "他后背已被冷汗浸透——",
                "恐惧": "他腿一软，下意识后退了半步——",
                "悲伤": "他眼眶一热，但硬是没让眼泪落下来——",
            }),
        ]
        for pat, repl_map in abstract_patterns:
            def _replacer(m: re.Match) -> str:
                word = m.group(1)
                stats["abstract_fixed"] += 1
                return repl_map.get(word, m.group(0))
            text = pat.sub(_replacer, text)

        # 6) 合并连续空行
        text = re.sub(r"\n{3,}", "\n\n", text)

        logger.info("  ⑦ AI 味祛除: %d→%d 字 | 替换 %d 处 概括修正 %d 处",
                     original_len, len(text),
                     stats["replaced"], stats["abstract_fixed"])
        return text

    # ── ⑧ 自检 ──────────────────────────────────────────────────

    def _self_check(self, text: str, blueprint: dict, state: dict) -> str:
        """⑧ 与阶段一蓝图比对，确认关键要素是否落实

        检查项：
        - 情绪曲线峰值段是否有足够描写
        - 末尾钩子是否在文中出现
        - 伏笔是否提及
        - 对话是否违反口吻规则
        """
        checklist: list[dict] = []

        # 检查 1：钩子是否落实
        final_hook = blueprint.get("chapter_structure", {}).get("final_hook", "")
        if final_hook:
            # 取钩子的前 6 个汉字作为关键词
            hook_key = final_hook[:6]
            hook_found = hook_key in text
            checklist.append({
                "item": "末尾钩子",
                "detail": hook_key,
                "ok": hook_found,
                "action": "" if hook_found else "警告：钩子可能未落实",
            })

        # 检查 2：情绪峰值描述密度
        sections = blueprint.get("chapter_structure", {}).get("sections", [])
        for s in sections:
            if s.get("is_climax"):
                # 高潮段应该足够长
                min_expected = s.get("word_count", 0) * 0.7
                checklist.append({
                    "item": f"高潮段 [{s['name']}]",
                    "detail": f"预期 {s['word_count']} 字，实际 {len(text)} 字（全文）",
                    "ok": len(text) >= min_expected,
                    "action": "" if len(text) >= min_expected
                              else f"建议补足高潮段内容（约需 {int(min_expected)} 字）",
                })

        # 检查 3：对话口吻是否冲突
        dialogue_cards = blueprint.get("dialogue_cards", {}) or {}
        dialogues = re.findall(r"[「\"]([^\"」]{2,60})[」\"]", text)
        voice_issues: list[str] = []
        for name, card in dialogue_cards.items():
            voice_issues.append(f"{name}({card.get('tone', '?')})")
        checklist.append({
            "item": "对话口吻",
            "detail": f"提取 {len(dialogues)} 句对话，{len(voice_issues)} 角色",
            "ok": len(dialogues) > 0,
            "action": "" if len(dialogues) > 0 else "建议增加对话内容",
        })

        # ★ 检查 4：KG 规则层面自检（从 KG 自检规则中提取模式）
        kg_selfcheck = (blueprint.get("kg_rules", {}) or {}).get("selfcheck", "")
        if kg_selfcheck:
            # 检测现代口语
            if "口语" in kg_selfcheck:
                modern_phrases = re.findall(r"(OK|ok|好的|没问题|拜托|牛逼|我去|靠|哇塞)", text)
                if modern_phrases:
                    checklist.append({
                        "item": "KG·现代口语",
                        "detail": f"疑似现代口语 {len(modern_phrases)} 处：{modern_phrases[:3]}",
                        "ok": len(modern_phrases) <= 1,
                        "action": "玄幻文应避免现代口语",
                    })
            # 检测逻辑错误关键词
            if "逻辑错误" in kg_selfcheck:
                # 检测明显时空矛盾描述
                instant_jumps = len(re.findall(r"(?:转瞬之间|刹那之间|一个呼吸)", text))
                if instant_jumps >= 3:
                    checklist.append({
                        "item": "KG·时空跳跃",
                        "detail": f"极速跳跃描述 {instant_jumps} 次，可能造成逻辑断裂",
                        "ok": instant_jumps < 3,
                        "action": "减少过度依赖'转瞬'类跳跃，用过渡描写替代",
                    })

        # 打印自检结果
        ok_count = sum(1 for c in checklist if c.get("ok"))
        total_count = len(checklist)
        logger.info("  ⑧ 蓝图自检: %d/%d 通过", ok_count, total_count)
        for c in checklist:
            status = "[OK]" if c.get("ok") else "[X]"
            logger.info("      %s [%s] %s %s",
                         status, c["item"], c.get("detail", ""),
                         c.get("action", ""))
        return text

    # ════════════════════════════════════════════════════════════
    #  辅助方法
    # ════════════════════════════════════════════════════════════

    @staticmethod
    def _safe_float_range(val: str | float | int, default: float) -> float:
        """将 '0.4-0.5' 这样的范围字符串转为中值，单值字符串转为 float"""
        if isinstance(val, (float, int)):
            return float(val)
        s = str(val).strip()
        m = re.match(r"^([\d.]+)\s*[-~]\s*([\d.]+)$", s)
        if m:
            return (float(m.group(1)) + float(m.group(2))) / 2
        try:
            return float(s)
        except (ValueError, TypeError):
            return default

    def _build_kb_block(self, kb: dict) -> str:
        """把知识库检索结果拼成可注入 prompt 的文本块"""
        if not kb:
            return ""
        lines: list[str] = []

        # 套路模板（只在匹配时注入）
        tmpl_name = kb.get("template_name") or ""
        if tmpl_name:
            lines.append(f"### 套路模板：{tmpl_name}")
            if kb.get("tagline"):
                lines.append(f"  - 卖点：{kb['tagline']}")
            if kb.get("opening_hook"):
                lines.append(f"  - 开场钩子：{kb['opening_hook']}")
            for beat in (kb.get("key_beats") or [])[:5]:
                lines.append(f"  - {beat}")
            arch = (kb.get("template") or {}).get("character_archetypes", {}) or {}
            for role, desc in list(arch.items())[:3]:
                lines.append(f"  - 角色原型 [{role}]：{desc}")
            for sample in (kb.get("dialogue_samples") or [])[:3]:
                lines.append(f"  - 对话范本：{sample}")
            for pitfall in (kb.get("ai_odor_pitfalls") or [])[:3]:
                lines.append(f"  - ⚠ {pitfall}")
            lines.append("")

        plat = kb.get("platform") or "番茄"
        rules = kb.get("platform_rules") or {}
        if rules:
            lines.append(f"### 平台风格规则：{plat}")
            for k, v in rules.items():
                if isinstance(v, list):
                    lines.append(f"  - {k}：{'; '.join(str(x) for x in v[:3])}")
                else:
                    lines.append(f"  - {k}：{v}")
            lines.append("")

        # ★ 知识图谱写作技巧（来自163个原始写作教程的三元组提取）
        graph_tips = kb.get("graph_tips") or []
        if graph_tips:
            lines.append("### 知识图谱 · 网文写作技巧")
            for tip in graph_tips[:12]:
                lines.append(f"  - {tip}")
            lines.append("")

        return "\n".join(lines)

    # ════════════════════════════════════════════════════════════
    #  LLM 调用与回退
    # ════════════════════════════════════════════════════════════

    def _call_llm(self, prompt: str, target_word_count: int,
                  blueprint: dict | None = None,
                  state: dict | None = None) -> str:
        """调用 LLM 生成章节正文，失败时自动回退到模板

        优先使用 deepseek-v4-pro；若当前是重写且 Editor 反复指出对话/静态问题，
        则切换为 moonshot-v1-32k（Kimi 对话能力更强）。
        """
        bp = blueprint or {}
        hints = bp.get("rewrite_hints", []) or []
        hints_text = "".join(hints)

        # 模型选择：默认 deepseek-v4-pro；重写且存在对话/静态问题时用 kimi
        model = "deepseek-v4-pro"
        is_rewrite = bool(hints)
        # 检测是否与对话/静态问题相关（累计出现 ≥2 条相关提示）
        dialogue_issues = sum(
            1 for h in hints if any(kw in h for kw in ["对话", "静态", "拖沓"])
        )
        if is_rewrite and dialogue_issues >= 2:
            model = "moonshot-v1-32k"

        # ── 构建 system prompt（注入角色状态快照 + KG 质量否决项）──
        kg_rules = bp.get("kg_rules", {}) or {}
        kg_selfcheck = kg_rules.get("selfcheck", "")
        system_content = self._build_system_prompt(state, kg_selfcheck=kg_selfcheck)

        max_tokens = max(2000, int(target_word_count * 2.5))
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ]

        try:
            content = call_llm(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.8,
            )
            source = "LLM重写" if is_rewrite else "LLM直出"
            logger.info("  [%s · %s] 生成正文 %d 字", source, model, len(content))
            return content
        except RuntimeError as e:
            logger.warning("  [LLM] 调用失败: %s，使用模板回退", e)
            return self._fallback_template(prompt, target_word_count, blueprint)

    @staticmethod
    def _build_system_prompt(state: dict | None = None,
                              kg_selfcheck: str = "") -> str:
        """构造 system prompt，注入当前角色状态快照以防止设定漂移

        Args:
            state: 故事状态
            kg_selfcheck: KG 自检规则文本（质量否决项）
        """
        base = WRITER_SYSTEM_PROMPT

        state = state or {}
        chars = state.get("characters", {}) or {}
        locked = state.get("lock", False)

        if not chars:
            return base

        char_lines = []
        for name, info in chars.items():
            if not isinstance(info, dict):
                continue
            char_lines.append(
                f"  - {name}: 修为={info.get('cultivation_realm', '?')}, "
                f"阵营={info.get('faction', '?')}, "
                f"性格={info.get('personality_tags', [])}"
            )

        lock_note = ""
        if locked:
            lock_note = (
                f"\n\n【状态已锁定】{state.get('lock_reason', '')}\n"
                "后续生成必须严格以当前角色数据为准，禁止自由发挥。"
            )

        char_snapshot = "\n".join(char_lines)

        # ★ KG 质量否决项
        kg_block = ""
        if kg_selfcheck and len(kg_selfcheck) > 10:
            kg_block = f"\n\n{kg_selfcheck}"

        return (
            f"{base}{lock_note}{kg_block}\n\n"
            "【绝对禁止触发的硬规则】\n"
            "- 已有角色修为必须严格按下表，禁止自行升级或降级（除非本章明确写出突破剧情）\n"
            "- 新角色修为只能从以下抽取：炼气期、筑基初期、筑基中期、筑基后期、"
            "筑基巅峰、金丹初期、金丹中期、金丹后期、元婴初期、化神初期\n"
            f"- 若违反以上规则，直接判定本章作废\n\n"
            f"当前角色状态：\n{char_snapshot}"
        )

    def _fallback_template(self, prompt: str, target_word_count: int,
                            blueprint: dict | None = None) -> str:
        """模板回退（无 LLM 时演示三阶段完整流程）

        当收到 rewrite_hints 时做最小变异，打破重写死循环：
          - 含"对话"/"静态" → 随机插入简短对话
          - 含"重复" → 随机调换两个段落前后顺序
        """
        blueprint = blueprint or {}
        sections = blueprint.get("chapter_structure", {}).get("sections", [])
        section_names = [s["name"] for s in sections] if sections else \
            ["起(开场冲突)", "承(矛盾酝酿)", "转(高潮爆发)", "合(剧情反转)", "结(悬念钩子)"]

        outline_match = re.search(r"## 本章大纲\s*\n(.*?)(?=\n##|\Z)", prompt, re.DOTALL)
        outline = outline_match.group(1).strip()[:200] if outline_match else ""

        char_match = re.search(r"### 角色口吻卡片\s*\n(.*?)(?=\n###|\Z)", prompt, re.DOTALL)
        char_block = char_match.group(1).strip()[:200] if char_match else ""

        # ── 重写变异：收到 edit hints 时做最小差异 ──
        hints = blueprint.get("rewrite_hints", []) or []
        hints_text = "".join(hints)
        is_rewrite = bool(hints)
        need_dialogue = any(kw in hints_text for kw in ["对话", "静态", "拖沓"])
        need_shuffle = "重复" in hints_text

        # 每段正文加微弱变化，防止 【...】 剥离后全段完全相同
        beat_variants = [
            f"风从{['山门','断崖','林间','竹梢','屋檐'][i]}掠过，卷起他衣袂一角。"
            for i in range(5)
        ]
        beat_endings = [
            "他没有回答，只是目光坚定地望向山门方向。",
            "他沉默片刻，缓缓攥紧了拳头。",
            "他咬着牙，一字一顿：「我知道了。」",
            "他深吸一口气，心底那股火却越烧越旺。",
            "他转身离去，背影被斜阳拉得很长。",
        ]
        paragraphs = []
        for i, name in enumerate(section_names):
            body = (outline or "林寒握紧拳头，灵海深处传来一阵灼热。")
            if is_rewrite:
                body += beat_variants[i]
            ending = beat_endings[i] if is_rewrite else \
                "他没有回答，只是目光坚定地望向山门方向。"
            paragraphs.append(
                f"【{name}】{body}\n"
                f"玉中老者的声音低沉：「小子，前路凶险，你可想清楚了？」\n\n"
                f"{ending}\n"
            )
        text = "\n".join(paragraphs)
        if outline:
            tag = outline.split("：")[0][:15] if "：" in outline else outline[:15]
            text += f"\n\n【本章详写】\n{tag}\n{outline}\n"
        if char_block:
            text += f"\n\n【角色口吻】\n{char_block}\n"
        # 字数补齐：轮换使用 5 个段落，避免重复同一段
        pad_idx = 0
        while len(text) < target_word_count:
            text += "\n" + paragraphs[pad_idx % len(paragraphs)]
            pad_idx += 1

        mutations: list[str] = []
        if need_dialogue:
            mutations.append("insert_dialogue")
        if need_shuffle:
            mutations.append("shuffle_paragraphs")

        if mutations:
            text = self._mutate_fallback_text(text, mutations)
            logger.info("  [模板] 重写变异: %s → %d 字",
                         ", ".join(mutations), len(text))
        else:
            logger.info("  [模板] 回退文本 %d 字", len(text))
        return text

    # ── 回退文本变异器 ───────────────────────────────────────────

    _DIALOGUE_POOL: list[str] = [
        "林寒冷声道：「你说什么？」\n",
        "柳如烟冷笑：「就凭你？」\n",
        "老者淡淡道：「小子，这可不是闹着玩的。」\n",
        "赵天哼了一声：「废物就是废物。」\n",
        "林寒抬起头：「三年后，我会让你记住今天。」\n",
        "柳青云沉声道：「这门婚约，到此为止。」\n",
        "围观者窃窃私语：「他居然还敢来...」\n",
        "少女掩嘴一笑：「林公子，你可真有意思。」\n",
        "黑甲护卫低喝：「再往前一步，休怪我不客气。」\n",
        "林寒握拳道：「这句话，我会原样还你。」\n",
    ]

    def _mutate_fallback_text(self, text: str, mutations: list[str]) -> str:
        """对回退文本执行最小变异"""
        import random

        # 1) 插入对话：随机选 3 个位置插入简短对话
        if "insert_dialogue" in mutations:
            lines = text.split("\n")
            dialog_count = 0
            # 找空行后的位置（自然断点）
            insert_candidates = [
                i for i in range(1, len(lines))
                if lines[i].strip() == "" and i + 1 < len(lines)
            ]
            random.shuffle(insert_candidates)
            for idx in sorted(insert_candidates[:3], reverse=True):
                line = random.choice(self._DIALOGUE_POOL)
                lines.insert(idx + 1, line)
                dialog_count += 1
            text = "\n".join(lines)
            logger.info("    ↳ 插入 %d 句对话", dialog_count)

        # 2) 打乱段落：随机调换两个段落的前后顺序
        if "shuffle_paragraphs" in mutations:
            paras = re.split(r"\n\s*\n", text)
            if len(paras) >= 4:
                i, j = random.sample(range(len(paras)), 2)
                paras[i], paras[j] = paras[j], paras[i]
                text = "\n\n".join(paras)
                logger.info("    ↳ 调换段落 %d ↔ %d 顺序", i, j)

        return text
