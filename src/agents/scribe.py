"""书记员 Agent —— 用 LLM 分析章节正文，提取世界观信息并增量合并到 state"""

import json
import re
from copy import deepcopy

from src.agents.base_agent import BaseAgent
from src.utils.llm_client import call_llm
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── LLM 系统提示词 ──────────────────────────────────────────────

SCRIBE_SYSTEM_PROMPT = """你是一名专业的网文设定分析师（书记员）。你的任务是从玄幻小说章节中提取关键世界观信息。

## 提取规则

### 1. 新角色 (new_characters)
对章节中新出现或首次被详细描写的角色，提取：
- name: 角色姓名
- aliases: 别名/称号列表
- cultivation_realm: 当前修为境界（如"锻体境""金丹期"等）
- faction: 所属势力/宗门
- personality_tags: 性格标签列表（如["坚毅","寡言"]）
- speech_traits: 口吻特征，根据对话总结（如"语气冷峻，言简意赅"），无对话时可为空字符串
- first_appearance: 首次出现时的简述

### 2. 新地点 (new_locations)
对章节中新出现的地点，提取：
- name: 地点名称
- type: 类型（城市/秘境/宗门/山脉/洞府/战场/其他）
- description: 特征描述

### 3. 力量体系扩展 (power_system)
- new_realms: 本章新出现的境界名称列表
- new_techniques: 本章新出现的功法/武技/秘术列表
- new_rules: 本章新揭示的修炼规则或世界观法则

### 4. 伏笔 (foreshadowing)
识别当前章节埋下的线索：
- clue: 伏笔内容描述
- possible_direction: 可能回收方向

### 5. 时间线事件 (timeline_events)
按发生顺序记录本章重要事件：
- event: 事件描述
- chapter: 所属章节号

## 输出格式
只输出一个 JSON 对象，不要包含任何额外文字。结构如下：
{
  "new_characters": [...],
  "new_locations": [...],
  "power_system": { "new_realms": [...], "new_techniques": [...], "new_rules": [...] },
  "foreshadowing": [...],
  "timeline_events": [...],
  "warnings": []
}
"""


class ScribeAgent(BaseAgent):
    """书记员 Agent：分析章节，增量更新世界观设定"""

    def __init__(self) -> None:
        super().__init__(name="Scribe")

    # ── 公开入口 ─────────────────────────────────────────────────

    def run(self, context: dict) -> dict:
        current_chapter = context.get("current_chapter", 0)
        chapter_text = context.get("chapter_text", "")
        current_state = context.get("current_state", {})

        # 分析章节
        extracted = self._analyze_chapter(current_chapter, chapter_text)

        # 合并到当前状态
        merged_state = self._merge_state(deepcopy(current_state), extracted, current_chapter)

        return {
            "agent": self.name,
            "chapter": current_chapter,
            "state": merged_state,
            "extracted": extracted,
            "warnings": extracted.get("warnings", []),
        }

    # ── 内部方法 ─────────────────────────────────────────────────

    def _analyze_chapter(self, chapter: int, text: str) -> dict:
        """调用 LLM 分析章节文本，返回提取的结构化信息"""
        # 输出格式约束
        format_reminder = f"""
当前分析章节: 第{chapter}章
输出格式：只输出一个合法 JSON 对象，不要包含 markdown 代码块标记，不要包含任何额外文字。
字段若未提取到，请使用空数组 [] 或空对象 {{}}。
"""
        user_prompt = f"请分析以下章节内容：\n\n{text}\n\n{format_reminder}"

        try:
            raw = call_llm(
                model="deepseek-v4-pro",
                messages=[
                    {"role": "system", "content": SCRIBE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=2048,
                temperature=0.3,
            )
            logger.info("LLM 原始返回 (前200字符): %s", raw[:200])
            return self._parse_json(raw)
        except RuntimeError as e:
            logger.error("LLM 调用失败: %s，使用模拟提取结果", e)
            return self._mock_extract(chapter, text)

    def _parse_json(self, raw: str) -> dict:
        """从 LLM 返回中提取 JSON 对象"""
        # 尝试直接解析
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # 尝试提取 ```json ... ``` 代码块
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # 尝试提取第一个 { ... } 块
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        logger.warning("无法从 LLM 返回中解析 JSON，返回空结果")
        return {}

    def _merge_state(self, state: dict, extracted: dict, chapter: int) -> dict:
        """将提取结果增量合并到 current_state，检测冲突并记录 warning"""

        warnings = extracted.get("warnings", [])

        # 初始化结构
        state.setdefault("characters", {})
        state.setdefault("locations", {})
        state.setdefault("power_system", {})
        state.setdefault("foreshadowing", [])
        state.setdefault("timeline", [])

        # ── 合并角色 ──
        for char in extracted.get("new_characters", []):
            name = char.get("name", "").strip()
            if not name:
                continue
            if name in state["characters"]:
                # 冲突检测
                existing = state["characters"][name]
                for key in ("cultivation_realm", "faction"):
                    new_val = char.get(key, "")
                    old_val = existing.get(key, "")
                    if new_val and old_val and new_val != old_val:
                        msg = (f"[第{chapter}章] 角色「{name}」的 {key} 冲突: "
                               f"旧值「{old_val}」→ 新值「{new_val}」，已以新值为准")
                        warnings.append(msg)
                        logger.warning(msg)
                # 合并：更新标量字段，合并列表字段
                existing["aliases"] = list(set(existing.get("aliases", []) + char.get("aliases", [])))
                existing["personality_tags"] = list(set(existing.get("personality_tags", []) + char.get("personality_tags", [])))
                for k in ("cultivation_realm", "faction", "speech_traits", "first_appearance"):
                    if char.get(k):
                        existing[k] = char[k]
            else:
                state["characters"][name] = {
                    "name": name,
                    "aliases": char.get("aliases", []),
                    "cultivation_realm": char.get("cultivation_realm", ""),
                    "faction": char.get("faction", ""),
                    "personality_tags": char.get("personality_tags", []),
                    "speech_traits": char.get("speech_traits", ""),
                    "first_appearance": char.get("first_appearance", f"第{chapter}章"),
                }

        # ── 合并地点 ──
        for loc in extracted.get("new_locations", []):
            loc_name = loc.get("name", "").strip()
            if not loc_name:
                continue
            if loc_name in state["locations"]:
                if loc.get("description"):
                    state["locations"][loc_name]["description"] = loc.get("description",
                                                                          state["locations"][loc_name].get("description", ""))
                if loc.get("type"):
                    state["locations"][loc_name]["type"] = loc.get("type",
                                                                   state["locations"][loc_name].get("type", ""))
            else:
                state["locations"][loc_name] = {
                    "name": loc_name,
                    "type": loc.get("type", ""),
                    "description": loc.get("description", ""),
                    "first_appearance": f"第{chapter}章",
                }

        # ── 合并力量体系 ──
        ps = extracted.get("power_system", {})
        for realm in ps.get("new_realms", []):
            if realm and realm not in state["power_system"].get("realms", []):
                state["power_system"].setdefault("realms", [])
                state["power_system"]["realms"].append(realm)
        for tech in ps.get("new_techniques", []):
            if tech and tech not in state["power_system"].get("techniques", []):
                state["power_system"].setdefault("techniques", [])
                state["power_system"]["techniques"].append(tech)
        for rule in ps.get("new_rules", []):
            if rule and rule not in state["power_system"].get("rules", []):
                state["power_system"].setdefault("rules", [])
                state["power_system"]["rules"].append(rule)

        # ── 合并伏笔 ──
        for fw in extracted.get("foreshadowing", []):
            clue = fw.get("clue", "")
            if clue:
                state["foreshadowing"].append({
                    "clue": clue,
                    "possible_direction": fw.get("possible_direction", ""),
                    "planted_chapter": chapter,
                })

        # ── 合并时间线 ──
        for event in extracted.get("timeline_events", []):
            desc = event.get("event", "")
            if desc:
                state["timeline"].append({
                    "event": desc,
                    "chapter": chapter,
                })

        # ── 伏笔回收：5 章前的伏笔若在当前章节再次出现关键词，自动标记 resolved ──
        current_fw_keywords = set()
        for fw in extracted.get("foreshadowing", []):
            clue = fw.get("clue", "")
            for kw in ("造化诀", "九玉", "三年之约", "青云真人", "万魂幡",
                       "天衍帝君", "玄清真人", "残阳玉", "寒渊玉", "玄黄玉",
                       "北冥玉", "南明玉", "造化玉"):
                if kw in clue:
                    current_fw_keywords.add(kw)
        for old_fw in state["foreshadowing"]:
            if old_fw.get("status") == "resolved":
                continue
            planted = old_fw.get("planted_chapter", 0)
            # 同章不回收
            if planted >= chapter:
                continue
            # 5 章之前埋下的伏笔，再次出现关键词则回收
            if chapter - planted < 3:
                continue
            clue_text = old_fw.get("clue", "")
            for kw in current_fw_keywords:
                if kw in clue_text:
                    old_fw["status"] = "resolved"
                    old_fw["resolved_chapter"] = chapter
                    msg = (f"[第{chapter}章] 伏笔已回收: 「{clue_text[:40]}」 "
                           f"（关键词: {kw}）")
                    warnings.append(msg)
                    logger.info(msg)
                    break

        # 附加 warnings
        state["_last_warnings"] = warnings
        return state

    # ── 模拟提取（当 LLM 不可用时）────────────────────────────────

    # 常见中文姓氏
    _COMMON_SURNAMES: set[str] = {
        "林", "苏", "叶", "萧", "楚", "秦", "云", "白", "沈", "陆",
        "赵", "王", "李", "张", "刘", "陈", "杨", "黄", "周", "吴",
        "徐", "孙", "马", "朱", "胡", "郭", "何", "高", "罗", "郑",
        "梁", "谢", "宋", "唐", "韩", "曹", "许", "邓", "冯", "慕容",
    }
    
    # 噪声词 —— 以地点后缀结尾但不是地名的常见词
    _LOCATION_NOISE: set[str] = {
        "翻江倒海", "人山人海", "刀山火海", "一片林", "说", "道说",
    }
    # 角色名噪声：常见动词/形容词伪装
    _CHARACTER_NOISE: set[str] = {
        "苏醒", "楚了", "林寒", "赵天",  # 可能是真实角色，加到白名单（不做噪声排除）
    }
    # 不能作为角色名结尾的动词/副词
    _CHARACTER_BAD_ENDS: set[str] = {
        "道", "说", "答", "问", "怒", "笑", "冷", "喝", "叹", "惊", "声", "喊",
    }

    def _mock_extract(self, chapter: int, text: str) -> dict:
        """基于规则模拟提取，用于离线测试"""
        import re as _re

        result: dict = {
            "new_characters": [],
            "new_locations": [],
            "power_system": {"new_realms": [], "new_techniques": [], "new_rules": []},
            "foreshadowing": [],
            "timeline_events": [],
            "warnings": [],
        }

        # ── 按句子分割（保留标点以辅助分析） ──
        raw_sentences = _re.split(r"[。！？\n]", text)
        sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 3]
        full_text_joined = "".join(sentences)

        # ── 1. 角色提取 ──
        found_names: set[str] = set()
        _surnames_alt = "(?:" + "|".join(self._COMMON_SURNAMES) + ")"

        # 方法 A: 前缀模式 —— "少年XXX" "老者XXX" 等
        # 后续必须是非汉字（标点/空格/动名词）才算合法名字，避免吞掉整个短语
        prefix_patterns = "(?:少年|老者|少女|中年|青年|村长|宗主|长老|帝君)"
        for m in _re.finditer(rf"{prefix_patterns}({_surnames_alt}[\u4e00-\u9fff]{{0,1}})(?=[，。！？、：；\s\"'「」])", full_text_joined):
            name = m.group(1)
            if self._looks_like_name(name):
                found_names.add(name)

        # 方法 B: 姓氏+名 —— 要求名字后紧跟非汉字/动词，避免片段匹配
        for m in _re.finditer(
            rf"({_surnames_alt}[\u4e00-\u9fff]{{1,2}})(?=[，。！？、：；\s\"'「」]|的|说|道|问|笑|喝|喊|怒|叹|独自|沉默|心中|猛|深吸|缓缓|已经|突然|没有|向着|回到|背|走|看|皱|咬|没|继续|从)",
            full_text_joined,
        ):
            name = m.group(1)
            if self._looks_like_name(name):
                found_names.add(name)

        # 方法 C: 对话标识 —— "XXX说/道/问/答/笑/喊" 或 "XXX："
        speech_markers = r"(说|道|问道|答道|大喝|冷喝|低语|喃喃|冷笑|怒道|笑道|叹道|开口|喝道|喊道|问道|答道)"
        for m in _re.finditer(rf"([\u4e00-\u9fff]{{2,3}})(?:冷|淡|轻|沉|缓|急|大|怒|微|笑)?(?:声)?{speech_markers}", full_text_joined):
            name = m.group(1)
            if self._looks_like_name(name):
                found_names.add(name)

        # 方法 D: 冒号前的人名（"XXX：" / "XXX:"）
        for m in _re.finditer(r"([\u4e00-\u9fff]{2,3})[：:]", full_text_joined):
            name = m.group(1)
            if self._looks_like_name(name):
                found_names.add(name)

        # 构建角色信息
        faction_keywords = ["宗", "殿", "谷", "门", "阁", "派", "盟", "府", "族"]
        for name in found_names:
            char_info = {"name": name, "aliases": [], "cultivation_realm": "",
                         "faction": "", "personality_tags": [], "speech_traits": "",
                         "first_appearance": f"第{chapter}章"}

            # 检测势力
            for fk in faction_keywords:
                for sent in sentences:
                    if name in sent and fk in sent:
                        m = _re.search(rf"([\u4e00-\u9fff]{{2,4}}{fk})", sent)
                        if m:
                            char_info["faction"] = m.group(1)
                            break
                if char_info["faction"]:
                    break

            # 检测修为 —— 角色名后紧跟的境界描述
            realm_match = _re.search(rf"{name}[\u4e00-\u9fff]*?(锻体|炼气|筑基|金丹|元婴|化神|渡劫|大乘)", full_text_joined)
            if realm_match:
                char_info["cultivation_realm"] = realm_match.group(1) + "期"

            # 检测性格标签
            if _re.search(rf"{name}.*?(坚毅|沉默|冷静|狂妄|阴狠|温和|暴躁|贪婪|正直)", full_text_joined):
                for m in _re.finditer(rf"{name}.*?(坚毅|沉默|冷静|狂妄|阴狠|温和|暴躁|贪婪|正直)", full_text_joined):
                    tag = m.group(1)
                    if tag not in char_info["personality_tags"]:
                        char_info["personality_tags"].append(tag)

            # 检测口吻（角色对话附近的描述）
            speech_context = _re.search(rf"{name}[\u4e00-\u9fff]*?(冷|淡|轻|沉|缓|急|怒)?(?:声)?(?:说|道|开口)[\u4e00-\u9fff]*?", full_text_joined)
            if speech_context:
                char_info["speech_traits"] = speech_context.group(0)[:20]

            result["new_characters"].append(char_info)

        # ── 2. 地点提取 ──
        locations = self._extract_locations(text, found_names)
        result["new_locations"] = locations

        # ── 3. 功法/武技提取 ──
        # 优先匹配书名号引出的功法名
        for m in _re.finditer(r"《([\u4e00-\u9fff]{2,8})》", full_text_joined):
            tech = m.group(1)
            techs = result["power_system"].setdefault("new_techniques", [])
            if tech not in techs:
                techs.append(tech)
        # 然后匹配 X诀 / X功 / X拳 / X掌 / X印 等专门名词（要求 X 至少 1 个字）
        for m in _re.finditer(r"([\u4e00-\u9fff]{1,3}(?:诀|功|拳|掌|印|心法|神功))", full_text_joined):
            tech = m.group(1)
            techs = result["power_system"].setdefault("new_techniques", [])
            if tech not in techs:
                techs.append(tech)

        # ── 4. 修为境界匹配 ──
        realm_patterns = [
            r"([\u4e00-\u9fff]{2,4}(?:境|期|阶|层))",
            r"(炼气|筑基|金丹|元婴|化神|渡劫|大乘|锻体)(?:期|境|层)?"
        ]
        for pat in realm_patterns:
            for m in _re.finditer(pat, full_text_joined):
                realm = m.group(1)
                realms = result["power_system"].setdefault("new_realms", [])
                if realm not in realms:
                    realms.append(realm)

        # ── 5. 时间线（前 8 句重要事件） ──
        count = 0
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 8 and count < 8:
                result["timeline_events"].append({"event": sent[:80], "chapter": chapter})
                count += 1

        # ── 6. 伏笔识别（关键词触发） ──
        # 关键伏笔词：伏笔、悬念、留下、待、终有一日、起、造化、十年、三百年
        foreshadowing_triggers = [
            ("造化诀", "天衍帝君传承的功法"),
            ("九玉", "散落九州的九枚造化玉碎片"),
            ("三年之约", "林寒复仇的三年期限"),
            ("青云真人", "终极大反派伏笔"),
            ("万魂幡", "禁忌法宝伏笔"),
            ("天衍帝君", "金手指导师背景"),
            ("玄清真人", "师尊伏笔"),
            ("残阳玉", "第一枚造化玉"),
            ("寒渊玉", "第二枚造化玉"),
            ("玄黄玉", "第三枚造化玉"),
            ("北冥玉", "第七枚造化玉"),
            ("南明玉", "第八枚造化玉"),
            ("造化玉", "第九枚造化玉"),
            ("化神", "修为大境界跃升"),
            ("渡劫", "修为大境界跃升"),
        ]
        for keyword, direction in foreshadowing_triggers:
            if keyword in text:
                # 提取含关键词的句子作为伏笔
                for sent in sentences:
                    if keyword in sent and 5 < len(sent) < 60:
                        result["foreshadowing"].append({
                            "clue": f"第{chapter}章：{sent[:60]}",
                            "possible_direction": direction,
                        })
                        break  # 每关键词每章只取 1 条

        logger.info("模拟提取完成: %d 角色, %d 地点, %d 境界, %d 伏笔",
                     len(result["new_characters"]), len(result["locations", ""]) if False else len(result["new_locations"]),
                     len(result["power_system"].get("new_realms", [])),
                     len(result["foreshadowing"]))
        return result

    @staticmethod
    def _looks_like_name(text: str) -> bool:
        """判断文本是否像一个中文名字——更严格的过滤"""
        import re as _re
        if not text or len(text) < 2 or len(text) > 4:
            return False
        # 排除纯数字/标点/英文
        if _re.search(r"[0-9a-zA-Z，。！？、；：""''（）\s]", text):
            return False
        # 排除以动词/副词结尾的假名（如 林寒冷→冷, 云震怒→怒）
        _bad_ends = {"道", "说", "答", "问", "怒", "笑", "冷", "喝", "叹", "惊", "声", "喊"}
        if text[-1] in _bad_ends:
            return False
        # 排除常见非名字短语
        noise = {
            "就是", "不过", "只是", "缓缓", "忽然", "突然", "依旧", "已经", "可以",
            "因为", "所以", "但是", "然后", "于是", "接着", "向前", "向后",
            "这里", "那里", "自己", "它们", "所有", "什么", "怎么", "这个", "那个",
            "一声", "一阵", "一股", "一下", "一道", "心中", "体内", "经脉",
            "灵力", "天地", "万物", "整个", "成为", "没有", "这老", "小子",
            "莫非", "原来", "难道", "还", "不要", "才能",
            "楚了", "音低", "冷冷", "轻轻", "慢慢", "默默", "静静", "重重",
            "不少", "这次", "刚才", "此刻", "此时", "片刻", "一时",
            "刚刚", "今日", "明日", "顷刻", "转瞬", "刹那",
            "老子", "老夫", "本座", "在下", "本人", "吾",
            "声中", "声低", "声的", "声说", "道说", "答道", "问说",
        }
        if text in noise:
            return False
        # 明确排除动词误判
        _verb_noise = {
            "苏醒", "说道", "问道", "答道", "怒道", "笑道", "喝道", "冷道",
            "见过", "看过", "听见", "感到", "觉得", "以为",
            "走过", "来过", "去过", "来过", "对着", "望着", "看着",
            "想到", "想起", "想到", "看出", "听到",
        }
        if text in _verb_noise:
            return False
        # 排除包含噪声子串的假名（如"楚了他"包含"楚了"）
        _noise_substrings = ["楚了", "音低", "声低"]
        for ns in _noise_substrings:
            if ns in text:
                return False
        # 必须以常见姓氏/前缀开头
        surnames = {
            "林", "苏", "叶", "萧", "楚", "秦", "云", "白", "沈", "陆",
            "赵", "王", "李", "张", "刘", "陈", "杨", "黄", "周", "吴",
            "徐", "孙", "马", "朱", "胡", "郭", "何", "高", "罗", "郑",
            "梁", "谢", "宋", "唐", "韩", "曹", "许", "邓", "冯", "慕",
            "冷", "霍", "顾", "丁", "方", "任", "姜", "范", "江", "钟",
        }
        if text[0] not in surnames:
            return False
        return True

    # ── 地点提取 ─────────────────────────────────────────────────
    _LOCATION_SUFFIXES: list[tuple[str, str]] = [
        ("山脉", "山脉"), ("谷", "山谷"), ("城", "城市"),
        ("坊市", "坊市"), ("殿", "宗门"), ("阁", "建筑"),
        ("府", "洞府"), ("洞", "洞府"), ("宗", "宗门"),
        ("门", "宗门"), ("派", "宗门"), ("界", "秘境"),
        ("域", "领域"), ("崖", "悬崖"), ("峰", "山峰"),
        ("林", "森林"), ("镇", "城镇"), ("村", "村落"),
        ("秘境", "秘境"),
    ]

    def _extract_locations(self, text: str, known_characters: set[str] | None = None) -> list[dict]:
        """识别以特定后缀结尾的 2-6 字地名词组，排除角色名和噪声词"""
        import re as _re
        known = known_characters or set()
        result: list[dict] = []
        seen: set[str] = set()

        for suffix, loc_type in self._LOCATION_SUFFIXES:
            for m in _re.finditer(rf"([\u4e00-\u9fff]{{1,5}}{_re.escape(suffix)})", text):
                name = m.group(1)
                if len(name) < 2 or len(name) > 6:
                    continue
                if name in seen:
                    continue
                if name in self._LOCATION_NOISE:
                    continue
                if name in known:
                    continue
                # 上下文验证：对短后缀（2 字以内）要求有方位词佐证
                suffix_main = suffix[-2:] if len(suffix) >= 2 else suffix
                if len(suffix_main) <= 2:
                    # 检查前是否有 在/去/到/进/入/从/望/向/至/来到/踏入/走出/离开
                    has_prefix = bool(_re.search(
                        rf"(?:在|去|到|进|入|从|望|向|至|来到|踏入|走出|离开)"
                        rf"[\u4e00-\u9fff]{{0,3}}{_re.escape(name)}", text))
                    # 检查后是否有 中/内/外/里/之中/之内
                    has_suffix = bool(_re.search(
                        rf"{_re.escape(name)}(?:中|内|外|里|之中|之内)", text))
                    if not has_prefix and not has_suffix:
                        continue

                seen.add(name)
                result.append({
                    "name": name,
                    "type": loc_type,
                    "description": f"文本中出现的地点（后缀 {suffix}）",
                })

        return result
