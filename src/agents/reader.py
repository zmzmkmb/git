"""读者 Agent —— 模拟真实网文读者视角，评估吸引力与 AI 味"""


from __future__ import annotations

import json
import re

from src.agents.base_agent import BaseAgent
from src.utils.llm_client import call_llm
from src.utils.logger import get_logger

logger = get_logger(__name__)


# 不同平台的读者偏好画像
PLATFORM_PROFILES: dict[str, dict] = {
    "番茄": {
        "core_audience": "下沉市场，男性偏多",
        "preferences": [
            "节奏极快，前 200 字必须有冲突或悬念",
            "爽点密度高：每章至少 1-2 个小高潮",
            "主角要'爽'：打脸、装逼、扮猪吃虎",
            "反感大段心理描写与景物铺陈",
            "对 AI 感最强的词（如同/仿佛/似乎/命运）极度敏感",
        ],
        "abandon_threshold": "开篇 500 字内无任何动作或冲突 → 高流失",
    },
    "起点": {
        "core_audience": "资深男频读者",
        "preferences": [
            "世界观与设定严谨，逻辑自洽",
            "力量体系完整，等级清晰",
            "剧情推进与世界观展开并重",
            "对话风格化、人物塑造立体",
        ],
        "abandon_threshold": "升级过快或战力崩坏 → 高流失",
    },
    "晋江": {
        "core_audience": "女性向读者",
        "preferences": [
            "情感细腻，人物心理活动丰富",
            "感情线推进明确，CP 感强",
            "对话质量与情绪张力",
            "可接受较长心理描写",
        ],
        "abandon_threshold": "情感线停滞或人物 OOC → 中等流失",
    },
}


READER_SYSTEM_PROMPT = """你是一位资深网文读者，对 AI 生成的内容有极高的辨识力。
请以真实读者的视角评估章节正文，输出严格 JSON：

{
  "attraction_score": 0-100,    // 吸引力评分：是否让人想继续读
  "ai_odor_score": 0-100,       // AI 味指数：越高越像 AI 写的
  "abandon_risk": "low"|"medium"|"high",  // 弃读风险
  "abandon_index": 0-100,       // 综合弃书指数（结合爽点密度+因果链缺陷）
  "pleasure_density": 0-100,    // 爽点密度评分（基于爽点出现频率与质量）
  "pleasure_gaps": ["爽点缺失的具体位置或段落描述"],  // 爽点缺失位置
  "feedback": "具体反馈（指出最影响阅读体验的 1-2 个点）",
  "abandon_reason": "弃书归因（结合因果链中最严重的缺陷，一句话说明）",
  "suggestions": ["具体改进建议1", "具体改进建议2"]
}

评分参考：
- attraction_score 80+：节奏紧凑、爽点密集、对话真实
- attraction_score 60-80：可读但有拖沓之处
- attraction_score <60：节奏慢、冲突弱、读不下去
- ai_odor_score 60+：明显 AI 痕迹（如同/仿佛/似乎堆砌、对仗工整、形容词罗列）
- ai_odor_score 30-60：偶有 AI 痕迹但可读
- ai_odor_score <30：人类质感强
- abandon_index 70+：高风险弃书，需大幅修改
- abandon_index 40-70：中等风险，建议针对性优化
- abandon_index <40：低风险，内容有黏性
- pleasure_density 80+：爽点密集，每 800 字左右有反转或小高潮
- pleasure_density 50-80：爽点尚可，但部分段落拖沓
- pleasure_density <50：爽点稀疏，读者易疲劳

只输出 JSON，不要任何解释。
"""


class ReaderAgent(BaseAgent):
    """读者 Agent：基于 LLM 评估吸引力与 AI 味"""

    # ── KG 规则缓存 ────────────────────────────────────────────
    _kg_rules_cache: str | None = None

    def __init__(self) -> None:
        super().__init__(name="Reader")

    def _load_kg_rules(self) -> str:
        """加载 Reader 专用的 KG 评估规则（弃书因果链 + 爽点参考）"""
        if self._kg_rules_cache is not None:
            return self._kg_rules_cache
        try:
            from src.utils.kg_query import get_reader_kg
            self._kg_rules_cache = get_reader_kg()
            if self._kg_rules_cache:
                logger.info("  [Reader] 已加载 KG 评估规则 (%d 字符)",
                            len(self._kg_rules_cache))
            else:
                self._kg_rules_cache = ""
        except Exception as e:
            logger.warning("  [Reader] KG 加载失败: %s", e)
            self._kg_rules_cache = ""
        return self._kg_rules_cache

    def run(self, context: dict) -> dict:
        chapter_text = context.get("chapter_text", "")
        chapter_number = context.get("chapter_number",
                                      context.get("current_chapter",
                                                  context.get("chapter", 0)))
        target_platform = context.get("target_platform", "番茄")
        # 知识库检索（query 由 outline 决定，没有则用平台名）
        outline_hint = context.get("outline", "") or context.get("template_hint", "")
        from src.utils.knowledge_retriever import get_reader_context, get_ai_odor_score
        kb_query = f"{outline_hint} {target_platform}"
        kb = get_reader_context(kb_query, platform=target_platform)

        if not chapter_text:
            logger.warning("Reader: 章节文本为空")
            return self._empty_result(chapter_number)

        # 加载 KG 评估规则
        kg_rules = self._load_kg_rules()

        prompt = self._build_prompt(chapter_text, target_platform, kb, kg_rules)
        result = self._call_llm_json(prompt)
        result.setdefault("agent", self.name)
        result.setdefault("chapter", chapter_number)

        # 规则层兜底：基于知识库 AI 味词典（带权重）计算
        rule_ai = get_ai_odor_score(chapter_text,
                                     kb.get("ai_odor_dictionary"))
        llm_ai = int(result.get("ai_odor_score", 0) or 0)
        final_ai = max(rule_ai, llm_ai)
        result["ai_odor_score"] = final_ai

        # 弃读风险联动
        if result.get("abandon_risk") not in ("low", "medium", "high"):
            result["abandon_risk"] = self._derive_abandon_risk(
                int(result.get("attraction_score", 50) or 50),
                final_ai,
            )

        # 继续读概率（与吸引力正相关，与 AI 味负相关）
        attraction = int(result.get("attraction_score", 50) or 50)
        result["continuation_probability"] = max(
            0, min(100, int(attraction * 0.85 - final_ai * 0.35))
        )

        # 把知识库命中信息也带上（用于日志）
        result["kb_platform"] = kb.get("platform")
        result["kb_satisfaction_matched"] = bool(kb.get("satisfaction_map"))

        logger.info("  [Reader] 吸引力 %d | AI味 %d | 弃读风险 %s | 弃书指数 %d | 爽点密度 %d | 续读 %d%% | KB平台=%s 爽点命中=%s",
                     attraction, final_ai, result.get("abandon_risk", "low"),
                     result.get("abandon_index", 0),
                     result.get("pleasure_density", 0),
                     result["continuation_probability"],
                     result["kb_platform"], result["kb_satisfaction_matched"])
        return result

    # ── Prompt 构建 ──────────────────────────────────────────────

    def _build_prompt(self, text: str, platform: str,
                       kb: dict | None = None, kg_rules: str = "") -> str:
        profile = (kb or {}).get("persona") or PLATFORM_PROFILES.get(
            platform, PLATFORM_PROFILES["番茄"])
        # 兼容 KB persona 字段 vs 旧字段
        core = profile.get("core", profile.get("core_audience", ""))
        prefs = profile.get("preferences", [])
        pref_lines = "\n".join(f"  - {p}" for p in prefs)

        # 爽点地图（仅命中时）
        sat_map = (kb or {}).get("satisfaction_map", {}) or {}
        sat_block = ""
        if sat_map:
            sat_block = "\n## 爽点地图\n"
            for k, v in sat_map.items():
                sat_block += f"  - {k}：{v}\n" if not isinstance(v, list) else f"  - {k}：" + " / ".join(str(x) for x in v[:3]) + "\n"

        # 弃书信号
        abandon = (kb or {}).get("abandon_signals", []) or []
        abandon_block = ""
        if abandon:
            abandon_block = "\n## 高风险弃书信号\n" + "\n".join(f"  - {s}" for s in abandon[:5])

        # KG 评估规则（弃书因果链 + 爽点密度参考）
        kg_block = ""
        if kg_rules and len(kg_rules) > 20:
            kg_block = f"\n## 知识图谱评估标准（基于同类网文读者弃书原因）\n{kg_rules}\n"

        return f"""# 读者评估任务
平台：{platform}
读者画像：{core}
{sat_block}{abandon_block}{kg_block}

## 该平台读者偏好
{pref_lines}

## 章节正文（{len(text)} 字）
{text[:6000]}

## 任务
请以该平台真实读者的视角，评估章节的吸引力与 AI 味。
输出中必须包含：
  - attraction_score：综合吸引力
  - abandon_index：综合弃书指数（0-100），结合爽点密度和 KG 因果链中最严重的缺陷给出
  - pleasure_density：爽点密度评分（0-100），关注每 800 字是否有小爽点/反转
  - pleasure_gaps：指出爽点缺失的具体位置
  - abandon_reason：弃书归因，一句话说明最严重的归因
严格按 JSON 格式输出。"""

    # ── LLM 调用 + JSON 解析 ─────────────────────────────────────

    def _call_llm_json(self, prompt: str) -> dict:
        """使用 Kimi (moonshot-v1-32k) 评估章节质量，失败回退规则"""
        try:
            content = call_llm(
                model="moonshot-v1-32k",
                messages=[
                    {"role": "system", "content": READER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=300,
                temperature=0.5,
            )
            return self._parse_json(content)
        except RuntimeError as e:
            logger.warning("  [Reader] LLM 调用失败: %s，使用规则回退", e)
            return self._fallback(prompt)

    @staticmethod
    def _parse_json(text: str) -> dict:
        text = re.sub(r"^```(?:json)?\s*", "", text.strip())
        text = re.sub(r"\s*```$", "", text.strip())
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return {}
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}

    # ── 规则兜底 ────────────────────────────────────────────────

    @staticmethod
    def _rule_ai_odor(text: str) -> int:
        """基于正文真实扫描的 AI 味检测（每命中 +3，上限 100）"""
        if not text:
            return 0
        ai_ticks = [
            "微微一笑", "只见", "一股", "心中暗道", "不由得",
            "顿时", "便见", "当下", "当即",
        ]
        hits = sum(text.count(t) for t in ai_ticks)
        return min(100, hits * 3)

    @staticmethod
    def _rule_dialogue_ratio(text: str) -> float:
        """计算对话引号内容字数占正文的比例"""
        if not text:
            return 0.0
        dialogues = re.findall(r'[「"「]([^「」""」]*)[」""」]', text)
        total_dialogue_chars = sum(len(d.strip()) for d in dialogues)
        return total_dialogue_chars / len(text)

    @staticmethod
    def _derive_abandon_risk(attraction: int, ai_odor: int) -> str:
        if attraction < 50 or ai_odor >= 70:
            return "high"
        if attraction < 70 or ai_odor >= 45:
            return "medium"
        return "low"

    def _fallback(self, prompt: str) -> dict:
        """无 LLM 时基于正文真实扫描的评估"""
        text_match = re.search(r"## 章节正文（\d+ 字）\n(.*)$", prompt, re.DOTALL)
        text = text_match.group(1) if text_match else ""
        wc = len(text)

        # ── AI 味检测 ──
        ai = self._rule_ai_odor(text)

        # ── 节奏检测：对话占比目标 40%，偏差 >10% 扣分 ──
        dialogue_ratio = self._rule_dialogue_ratio(text)
        rhythm_penalty = max(0, abs(dialogue_ratio - 0.4) - 0.10) * 100  # 超出 10% 偏差开始扣

        # ── 吸引力评分 ──
        base = 65  # 基础分
        # AI 味扣分
        base -= min(30, ai // 3)
        # 对话偏差扣分
        base -= min(15, int(rhythm_penalty))
        # 字数不足扣分
        if wc < 1500:
            base -= 20
        elif wc < 2500:
            base -= 8
        # 字数达标奖励
        if wc >= 2500:
            base += 5
        # 对话合理奖励
        if 0.20 <= dialogue_ratio <= 0.55:
            base += 8
        attraction = max(0, min(100, base))

        # ── 具体反馈 ──
        feedback_parts = []
        suggestions = []
        if ai >= 60:
            feedback_parts.append(f"AI 味较重（{ai}分）")
            suggestions.append("减少'微微一笑''只见''不由得'等痕迹词")
        elif ai >= 30:
            feedback_parts.append(f"偶有 AI 痕迹（{ai}分）")
        if dialogue_ratio < 0.20:
            feedback_parts.append("对话过少，节奏可能拖沓")
            suggestions.append(f"增加对话密度（当前对话占比 {int(dialogue_ratio*100)}%，目标 40%）")
        elif dialogue_ratio > 0.60:
            feedback_parts.append("对话过于密集，可加入描写调节")
            suggestions.append("在对话间插入环境和动作描写")
        if wc < 2000:
            feedback_parts.append(f"字数偏少（{wc}字）")
            suggestions.append("补充情节或细节描写")
        if not suggestions:
            suggestions = ["整体可读，建议用 LLM 获取更精准反馈"]
        feedback = "；".join(feedback_parts) if feedback_parts else "（规则评估）整体可读"

        return {
            "attraction_score": attraction,
            "ai_odor_score": ai,
            "abandon_risk": self._derive_abandon_risk(attraction, ai),
            "abandon_index": max(0, min(100, 100 - attraction + ai // 2)),
            "pleasure_density": max(0, min(100, attraction - ai // 3)),
            "pleasure_gaps": [],
            "abandon_reason": feedback,
            "feedback": feedback,
            "suggestions": suggestions,
        }

    @staticmethod
    def _empty_result(chapter_number: int) -> dict:
        return {
            "agent": "Reader",
            "chapter": chapter_number,
            "attraction_score": 0,
            "ai_odor_score": 100,
            "abandon_risk": "high",
            "abandon_index": 100,
            "pleasure_density": 0,
            "pleasure_gaps": [],
            "abandon_reason": "章节文本为空",
            "continuation_probability": 0,
            "feedback": "章节文本为空",
            "suggestions": ["请先生成章节正文"],
        }
