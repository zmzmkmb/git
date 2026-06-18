"""编辑 Agent —— 审核章节质量，检查设定一致性与网文质量"""

from __future__ import annotations

import json
import re
from typing import Any

from src.agents.base_agent import BaseAgent
from src.utils.llm_client import call_llm
from src.utils.logger import get_logger

logger = get_logger(__name__)


EDITOR_SYSTEM_PROMPT = """你是一位严苛的网文主编，负责审校连载小说的章节正文。
你需要从以下两个维度对章节进行评估：

【一】一致性检查（hard_errors）
- 角色修为是否前后一致（不能本章筑基，下文突然金丹）
- 已确立的人物关系、势力归属、功法设定是否被违反
- 时间线是否合理（角色不可能瞬间跨越千里）
- 已知伏笔是否被合理回收或埋下

【二】技术性问题（tech_issues，每条为结构化对象）
每条 tech_issue 包含：
  - type: "pace" | "style" | "logic" | "dialogue" | "sensory" | "hook" | "rule"
  - severity: "critical" | "major" | "minor"
  - description: 问题描述（一句话）
  - quote: 原文节选（20-50 字，标明问题所在位置）
  - suggestion: 改进建议

常见 tech_issue 类型参考：
- pace（节奏）：长段无转折、爽点密度不够
- style（风格）：AI 痕迹词密集、被动句过多、口吻混乱
- logic（逻辑）：角色行为前后矛盾、情节不合理
- dialogue（对话）：对话流冗长、无动作穿插、口吻未区分
- sensory（感官）：缺乏动作/视觉/听觉描写
- hook（钩子）：开篇无冲突、结尾无悬念
- rule（规则违反）：违反下方提供的 KG 写作规范

【输出格式】严格 JSON：
{
  "decision": "pass" | "revise" | "rewrite",
  "hard_errors": ["..."],
  "tech_issues": [
    {"type": "pace", "severity": "major", "description": "...", "quote": "...", "suggestion": "..."}
  ],
  "quality_score": 0-100
}

判定标准：
- quality_score >= 75 且 hard_errors 为空 → "pass"
- 50 <= quality_score < 75 或有 1-2 条 tech_issues → "revise"
- quality_score < 50 或有 hard_errors → "rewrite"

只输出 JSON，不要任何解释。
"""


class EditorAgent(BaseAgent):
    """编辑 Agent：基于 LLM 的一致性与质量审核"""

    # ── KG 规则缓存 ────────────────────────────────────────────
    _kg_rules_cache: dict[str, list[str]] | None = None

    def __init__(self) -> None:
        super().__init__(name="Editor")

    def _load_kg_rules(self) -> dict[str, list[str]]:
        """加载 Editor 专用的 KG 审校规则（硬规则 + 软规则）"""
        if self._kg_rules_cache is not None:
            return self._kg_rules_cache
        result: dict[str, list[str]] = {"hard": [], "soft": []}
        try:
            from src.utils.kg_query import get_editor_kg
            text = get_editor_kg()
            if text:
                logger.info("  [Editor] 已加载 KG 审校规则 (%d 字符)", len(text))
                # 从文本中拆出硬规则和软规则
                parts = text.split("##")
                for p in parts:
                    if "硬规则检测" in p:
                        lines = [l.strip() for l in p.split("\n")
                                 if l.strip().startswith("- [") or l.strip().startswith("「")]
                        result["hard"] = lines[:10]
                    elif "软校验参考" in p:
                        lines = [l.strip() for l in p.split("\n")
                                 if l.strip().startswith("「")]
                        result["soft"] = lines[:10]
            self._kg_rules_cache = result
        except Exception as e:
            logger.warning("  [Editor] KG 加载失败: %s", e)
            self._kg_rules_cache = result
        return self._kg_rules_cache

    def run(self, context: dict) -> dict:
        chapter_text = context.get("chapter_text", "")
        current_state = context.get("current_state", {}) or {}
        chapter_number = context.get("chapter_number",
                                     context.get("current_chapter",
                                                 context.get("chapter", 0)))

        if not chapter_text:
            logger.warning("Editor: 章节文本为空")
            return self._empty_result(chapter_number)

        # 加载 KG 审校规则
        kg_rules = self._load_kg_rules()

        # 构建 prompt
        prompt = self._build_prompt(chapter_text, current_state, chapter_number, kg_rules)
        # 调用 LLM（带 JSON 解析与回退）
        result = self._call_llm_json(prompt)
        result.setdefault("agent", self.name)
        result.setdefault("chapter", chapter_number)

        # 规则层兜底：用硬规则补充 hard_errors（设定冲突检测）
        rule_errors = self._rule_consistency_check(chapter_text, current_state)
        if rule_errors:
            existing = set(result.get("hard_errors", []) or [])
            for e in rule_errors:
                if e not in existing:
                    result.setdefault("hard_errors", []).append(e)

        # KG 硬规则检测：长段落无转折、被动句占比
        kg_hard_issues = self._rule_kg_hard_check(chapter_text, kg_rules)
        if kg_hard_issues:
            existing_tech = result.get("tech_issues", []) or []
            existing_descs = {(t.get("description"), t.get("type"))
                              for t in existing_tech if isinstance(t, dict)}
            for issue in kg_hard_issues:
                key = (issue.get("description", ""), issue.get("type", ""))
                if key not in existing_descs:
                    result.setdefault("tech_issues", []).append(issue)

        # ★ 任何 hard_error 都强制重写（硬错是阻断发布的致命问题）
        hard_count = len(result.get("hard_errors", []) or [])
        if hard_count >= 1:
            result["decision"] = "rewrite"
            result["rewrite_reason"] = (
                f"硬错 {hard_count} 条: "
                + "; ".join((result.get("hard_errors") or [])[:3])
            )

        logger.info("  [Editor] 评分 %d，决策 %s，hard=%d, tech=%d",
                     result.get("quality_score", 0), result.get("decision", "pass"),
                     len(result.get("hard_errors", [])),
                     sum(1 for t in (result.get("tech_issues") or []) if isinstance(t, dict)))
        return result

    # ── Prompt 构建 ──────────────────────────────────────────────

    def _build_prompt(self, text: str, state: dict, chapter_number: int,
                       kg_rules: dict[str, list[str]] | None = None) -> str:
        # 摘要设定
        chars = state.get("characters", {}) or {}
        char_summary = []
        for name, info in list(chars.items())[:10]:
            if not isinstance(info, dict):
                continue
            char_summary.append(
                f"  - {name}：修为 {info.get('cultivation_realm', '?')}，"
                f"势力 {info.get('faction', '?')}，性格 "
                f"{','.join(info.get('personality_tags', []) or []) or '无'}"
            )
        char_block = "\n".join(char_summary) if char_summary else "  （暂无角色）"

        ps = state.get("power_system", {}) or {}
        realms = ", ".join((ps.get("realms") or [])[:10]) or "无"
        techniques = ", ".join((ps.get("techniques") or [])[:10]) or "无"

        # 字数（用于判断爽点密度）
        word_count = len(text)

        # KG 审校规则注入
        kg_block = ""
        if kg_rules:
            hard_rules = (kg_rules.get("hard") or [])[:5]
            soft_rules = (kg_rules.get("soft") or [])[:5]
            kg_lines: list[str] = []
            if hard_rules:
                kg_lines.append("## KG 强制质量规则（硬规则 - 代码层可编程扫描）")
                for hr in hard_rules:
                    kg_lines.append(f"  - {hr}")
            if soft_rules:
                kg_lines.append("\n## KG 写作规范参考（软校验 - LLM 判断）")
                for sr in soft_rules:
                    kg_lines.append(f"  - {sr}")
            if kg_lines:
                kg_lines.append(
                    "\n  ⚠ 发现违反软规则时，输出 type 为 rule 的 tech_issue。"
                )
                kg_block = "\n" + "\n".join(kg_lines) + "\n"

        return f"""# 审校任务
请审核【第{chapter_number}章】正文（{word_count} 字）。

## 当前设定库（一致性基准）

### 角色
{char_block}

### 力量体系
- 境界：{realms}
- 功法：{techniques}

## 章节正文
{text[:6000]}
{kg_block}
## 任务
1. **一致性**：检查正文是否与设定库冲突（修为/势力/功法/人际关系/时间线）
2. **技术性**：评估节奏、爽点密度、动作戏、感官描写、钩子设计，以及 KG 规则遵守情况
3. **决策**：pass（通过）/ revise（修订）/ rewrite（重写）
4. **评分**：0-100，参考网文头部作品的及格线为 75

只输出 JSON。"""

    # ── LLM 调用 + 鲁棒 JSON 解析 ────────────────────────────────

    def _call_llm_json(self, prompt: str) -> dict:
        """使用 DeepSeek 审校章节，失败回退规则"""
        try:
            content = call_llm(
                model="deepseek-v4-pro",
                messages=[
                    {"role": "system", "content": EDITOR_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1500,
                temperature=0.3,
            )
            return self._parse_json(content)
        except RuntimeError as e:
            logger.warning("  [Editor] LLM 调用失败: %s，使用规则回退", e)
            return self._fallback(prompt)

    @staticmethod
    def _parse_json(text: str) -> dict:
        """鲁棒 JSON 解析：先尝试直接 parse，否则抽取首个 JSON 块"""
        # 去除 markdown 代码块
        text = re.sub(r"^```(?:json)?\s*", "", text.strip())
        text = re.sub(r"\s*```$", "", text.strip())
        # 抽取第一个 { ... } 块
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return {}
        try:
            return json.loads(m.group(0))
        except Exception:
            # 尝试宽松解析
            try:
                return json.loads(m.group(0).replace("\u3000", " "))
            except Exception:
                return {}

    # ── 规则兜底 ────────────────────────────────────────────────

    def _fallback(self, prompt: str) -> dict:
        """无 LLM 时基于正文真实扫描的评分"""
        # 提取章节文本
        text_match = re.search(r"## 章节正文\s*\n(.*)$", prompt, re.DOTALL)
        text = text_match.group(1) if text_match else ""
        wc = len(text)
        ch_match = re.search(r"[第]?\s*(\d+)\s*[章节]", prompt)
        chapter = int(ch_match.group(1)) if ch_match else 0

        tech_issues: list[dict] = []
        hard_errors: list[str] = []

        # ── 1. 字数检查 ──
        if wc < 1500:
            tech_issues.append({
                "type": "pace",
                "severity": "minor",
                "description": f"字数仅 {wc}，未达基本长度要求",
                "quote": f"全文共 {wc} 字",
                "suggestion": "建议扩充至 2000 字以上，确保情节充分展开",
            })
        if wc < 800:
            hard_errors.append(f"字数严重不足（{wc}），无法评估质量")

        # ── 2. 对话检查 ──
        dialogue_chars = len(re.findall(r'[「"][^「」"]+[」"]', text))
        dialogue_ratio = dialogue_chars / max(1, wc)
        if dialogue_ratio < 0.10:
            tech_issues.append({
                "type": "dialogue",
                "severity": "major",
                "description": f"对话占比极低（{dialogue_ratio:.0%}），章节可能过于静态",
                "quote": f"对话内容仅 {dialogue_chars} 字 / 全文 {wc} 字",
                "suggestion": "在叙事段落之间插入对话，用角色互动推动情节",
            })

        # ── 3. 连续重复段落检测（完全相同的 ≥30 字片段） ──
        dup_errors = self._scan_duplicates(text, min_len=30)
        if dup_errors:
            hard_errors.append(
                f"检测到 {len(dup_errors)} 处连续重复段落（完全相同的 ≥30 字片段）"
            )
            for de in dup_errors[:2]:
                hard_errors.append(f"  重复标记: 「{de[:30]}...」")

        # ── 4. AI 痕迹词检测 ──
        ai_ticks = ["微微一笑", "只见", "一股", "心中暗道", "不由得",
                    "顿时", "便见", "当下", "当即"]
        ai_hits = sum(text.count(t) for t in ai_ticks)
        if ai_hits >= 10:
            tech_issues.append({
                "type": "style",
                "severity": "major",
                "description": f"AI 痕迹词出现 {ai_hits} 次，质感较差",
                "quote": f"全文 {ai_hits} 处 AI 痕迹词",
                "suggestion": "用具体动作描写替代'微微一笑''只见'等套路表达",
            })
        elif ai_hits >= 5:
            tech_issues.append({
                "type": "style",
                "severity": "minor",
                "description": f"AI 痕迹词 {ai_hits} 次，可适当减少",
                "quote": f"全文 {ai_hits} 处 AI 痕迹词",
                "suggestion": "检查'不由得''顿时'等词，替换为更具体的描写",
            })

        # ── 5. KG 硬规则检测（长段落无转折 + 被动句） ──
        tech_issues.extend(self._rule_kg_hard_check(text))

        # ── 6. 综合评分 ──
        base_score = 70
        base_score -= min(15, wc // 100 - (wc // 100) % 5) if wc < 2000 else 0  # 字数不足扣分
        base_score -= min(25, ai_hits * 2)         # AI 词扣分
        base_score -= min(15, dialogue_ratio < 0.10 and 15)  # 对话过少
        base_score -= len(dup_errors) * 10          # 重复段落严重扣分
        base_score = max(0, min(100, base_score))

        decision = ("pass" if base_score >= 75 and not hard_errors else
                    "rewrite" if hard_errors else
                    "revise" if base_score >= 50 else "rewrite")
        return {
            "decision": decision,
            "hard_errors": hard_errors,
            "tech_issues": tech_issues,
            "quality_score": base_score,
        }

    @staticmethod
    def _scan_duplicates(text: str, min_len: int = 30) -> list[str]:
        """扫描正文中完全相同的连续重复段落（≥ min_len 字）

        比较前先移除两类标记，避免模板回退文本被误判为重复内容：
        - r'【[^】]*】' 移除所有 【...】 节拍标记
        - 移除「本章详写」「角色口吻」等引用性重述块（其后的正文与节拍正文重复）
        """
        if not text or len(text) < min_len * 2:
            return []

        # 移除 【...】 标记
        clean_text = re.sub(r'【[^】]*】', '', text)
        # 移除 「本章详写」「角色口吻」等引用重复块（保留标签行本身无意义）
        clean_text = re.sub(r'【本章详写】\s*\n[\s\S]*?(?=\n\[|【|\Z)', '', clean_text)
        clean_text = re.sub(r'【角色口吻】\s*\n[\s\S]*?(?=\n\[|【|\Z)', '', clean_text)

        dupes: list[str] = []
        # 按段切分（空行分隔），用前缀+后缀联合签名避免长前缀同质化误判
        paras = [p.strip() for p in re.split(r"\n\s*\n", clean_text) if len(p.strip()) >= min_len]
        key_counts: dict[str, int] = {}
        key_samples: dict[str, str] = {}
        for p in paras:
            # 联合签名：前 60 字 + 后 40 字，防止共用的长前缀导致假阳性
            key = p[:60] + "|" + p[-40:] if len(p) >= 100 else p[:80]
            key_counts[key] = key_counts.get(key, 0) + 1
            if key not in key_samples:
                key_samples[key] = p[:60]
        for key, cnt in key_counts.items():
            if cnt >= 4:
                dupes.append(key_samples[key])
        # 跨段滑窗扫描：仅报告出现 ≥4 次的 ≥50 字片段（避免模板正文的同质内容误报）
        for i in range(0, len(clean_text) - 50 * 2, max(1, len(clean_text) // 4)):
            snippet = clean_text[i : i + 50]
            # snippet 本身有意义（非纯空白/标点）
            if not re.search(r'[\u4e00-\u9fff]{3,}', snippet):
                continue
            count = clean_text.count(snippet)
            if count >= 4 and len(snippet.strip()) >= 50:
                existing = [d for d in dupes if d in snippet or snippet in d]
                if not existing:
                    dupes.append(snippet[:60])
                break
        return dupes[:5]

    @staticmethod
    def _rule_consistency_check(text: str, state: dict) -> list[str]:
        """规则层一致性检查：检测明显的人物修为/势力矛盾"""
        errors: list[str] = []
        chars = state.get("characters", {}) or {}
        # 把正文按段落切，对每段检查
        for name, info in list(chars.items()):
            if not isinstance(info, dict):
                continue
            realm = info.get("cultivation_realm") or ""
            faction = info.get("faction") or ""
            if not realm and not faction:
                continue
            # 简化的修为跳变检测：若文中出现更高阶修为词且未提到突破
            higher_realms = ["金丹", "元婴", "化神"]
            for hr in higher_realms:
                if hr in text and realm and hr != realm:
                    # 仅当同一名字附近出现时报警
                    if name in text and hr in text:
                        idx_name = text.find(name)
                        idx_realm = text.find(hr)
                        if 0 <= idx_name - idx_realm < 50 or 0 <= idx_realm - idx_name < 50:
                            errors.append(
                                f"角色「{name}」原修为 {realm}，"
                                f"本章出现 {hr}，可能存在修为跳变"
                            )
                            break
        return errors

    @staticmethod
    def _rule_kg_hard_check(text: str,
                            kg_rules: dict[str, list[str]] | None = None) -> list[dict]:
        """KG 驱动的硬规则检测：长段落无转折 + 被动句占比

        Returns:
            [{"type": str, "severity": str, "description": str,
              "quote": str, "suggestion": str}, ...]
        """
        issues: list[dict] = []
        if not text:
            return issues

        # ── 1. 长段落无转折检测（连续 ≥5 句无明显转折词） ──
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.strip()) > 100]
        for idx, para in enumerate(paragraphs[:10]):  # 只看前 10 段
            sentences = re.split(r'[。！？\n]', para)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
            # 转折/冲突词集合
            TURN_WORDS = {"但是", "然而", "却", "突然", "忽然", "猛地", "没想到",
                          "出乎意料", "岂料", "不料", "谁知", "就在这时",
                          "一瞬间", "转瞬之间", "刹那间"}
            no_turn_count = 0
            longest_no_turn = 0
            max_start = 0
            for si, sent in enumerate(sentences):
                if any(tw in sent for tw in TURN_WORDS):
                    no_turn_count = 0
                else:
                    if no_turn_count == 0:
                        max_start = si
                    no_turn_count += 1
                    longest_no_turn = max(longest_no_turn, no_turn_count)
            if longest_no_turn >= 5:
                # 取问题段落的前 50 字作为引用
                start_sent = sentences[max_start] if max_start < len(sentences) else ""
                quote = start_sent[:50] if start_sent else para[:50]
                issues.append({
                    "type": "pace",
                    "severity": "major",
                    "description": f"第 {idx+1} 段连续 {longest_no_turn} 句无转折，读者易疲劳",
                    "quote": quote,
                    "suggestion": "每 3-4 句插入一个转折/冲突/悬念，保持阅读节奏",
                })

        # ── 2. 被动句占比检测 ──
        passive_markers = ["被", "给", "遭到", "受到", "挨", "叫", "让"]
        passive_count = 0
        total_sentences = 0
        sentences = re.split(r'[。！？\n]', text)
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 4:
                continue
            total_sentences += 1
            if any(pm in sent for pm in passive_markers):
                passive_count += 1
        if total_sentences > 0:
            passive_ratio = passive_count / total_sentences
            if passive_ratio > 0.25:
                # 找第一个被动句例子
                example = ""
                for sent in sentences:
                    if any(pm in sent for pm in passive_markers) and len(sent) > 10:
                        example = sent.strip()[:50]
                        break
                issues.append({
                    "type": "style",
                    "severity": "minor",
                    "description": f"被动句占比 {passive_ratio:.0%}（{passive_count}/{total_sentences}），超过 25% 建议线",
                    "quote": example or "（无示例）",
                    "suggestion": "将部分被动句改为主动句，增强文字的力度和画面感",
                })

        return issues[:5]

    @staticmethod
    def _empty_result(chapter_number: int) -> dict:
        return {
            "agent": "Editor",
            "chapter": chapter_number,
            "decision": "rewrite",
            "hard_errors": ["章节文本为空"],
            "tech_issues": [],
            "quality_score": 0,
        }

    # ── 辅助方法 ────────────────────────────────────────────────

    @staticmethod
    def _is_tech_issue_dict(item: Any) -> bool:
        return isinstance(item, dict) and "type" in item
