"""主控 Agent —— 负责调度整个写作流水线"""

import json
import os
from collections import Counter
from datetime import datetime
from typing import Any

from src.agents.base_agent import BaseAgent
from src.agents.writer import WriterAgent
from src.agents.scribe import ScribeAgent
from src.agents.editor import EditorAgent
from src.agents.reader import ReaderAgent
from src.utils.state_manager import load_state, save_state, reset_state
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ════════════════════════════════════════════════════════════════
#  大纲加载：优先用 src/data/story_design.json（30章玄幻全本），
#  否则回退到内置 DEFAULT_OUTLINES（3章演示）
# ════════════════════════════════════════════════════════════════

STORY_DESIGN_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "story_design.json"
)


def _load_story_design() -> tuple[str, dict[int, str]] | None:
    """从 story_design.json 加载 200 字梗概 + 30 章大纲"""
    if not os.path.exists(STORY_DESIGN_PATH):
        return None
    try:
        with open(STORY_DESIGN_PATH, "r", encoding="utf-8") as f:
            design = json.load(f)
        outline = design.get("story_outline", "")
        chapter_outlines = design.get("chapter_outlines", {}) or {}
        # JSON 键为字符串，转 int
        normalized = {int(k): v for k, v in chapter_outlines.items()}
        if not outline or not normalized:
            return None
        return outline, normalized
    except Exception as e:
        logger.warning("加载 story_design.json 失败: %s", e)
        return None


# 默认大纲数组 —— 演示用 3 章通用剧情
DEFAULT_OUTLINES: dict[int, str] = {
    1: (
        "故事开场。主角的日常生活被一件突发事件打破，一个神秘人物的出现带来了关键线索。"
        "本章需要：快速建立世界观，用事件钩住读者，结尾留下悬念。"
    ),
    2: (
        "主角追查上一章的线索，初遇核心冲突，获得第一个助力或工具。"
        "本章需要：丰富世界观设定，推动主线发展，展现主角的特质与潜力。"
    ),
    3: (
        "冲突升级，主角面对真正的挑战，发现事情并不简单，更大的伏笔被埋下。"
        "本章需要：紧张感递进，展现角色的成长弧光，结尾为后续章节制造强大期待。"
    ),
}


# 审校日志存放目录
LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "logs")


def _ensure_logs_dir() -> None:
    os.makedirs(LOGS_DIR, exist_ok=True)


def _save_chapter_log(chapter: int, payload: dict) -> str:
    """保存单章的 writer/scribe/editor/reader 全量日志到 JSON 文件"""
    _ensure_logs_dir()
    path = os.path.join(LOGS_DIR, f"chapter_{chapter:03d}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


class Orchestrator(BaseAgent):
    """主控 Agent，接受故事梗概，循环生成 N 章"""

    def __init__(self, outline: str = "", total_chapters: int = 3,
                 chapter_outlines: dict[int, str] | None = None,
                 target_word_count: int = 3000,
                 target_platform: str = "番茄",
                 output_dir: str = "",
                 output_file: str = "",
                 reset: bool = False,
                 max_rewrite: int = 1,
                 custom_settings: dict | None = None) -> None:
        super().__init__(name="Orchestrator")
        self.target_word_count = target_word_count
        self.target_platform = target_platform
        self.max_rewrite = max_rewrite
        self.custom_settings = custom_settings or {}
        self.output_dir = output_dir
        self.output_file = output_file

        # 解析自定义角色名
        self.protagonist = self.custom_settings.get("protagonist", "")
        self.mentor = self.custom_settings.get("mentor", "")
        self.antagonist = self.custom_settings.get("antagonist", "")
        self.heroine = self.custom_settings.get("heroine", "")
        self.climax_type = self.custom_settings.get("climax_type", "")

        # 加载故事设计
        auto_design = _load_story_design()
        if auto_design:
            loaded_outline, loaded_chapters = auto_design
            self.outline = outline or loaded_outline
            self.chapter_outlines = chapter_outlines or loaded_chapters
            if total_chapters < 1 or total_chapters == 3:
                self.total_chapters = total_chapters if total_chapters != 3 else len(loaded_chapters)
            else:
                self.total_chapters = total_chapters
        else:
            self.outline = outline
            self.chapter_outlines = chapter_outlines or {}
            self.total_chapters = total_chapters

        # 如果调用方已经传入了明确的 chapter_outlines，跳过自动生成
        # 否则才根据自定义角色名或默认设定生成大纲
        if chapter_outlines and len(chapter_outlines) > 0:
            pass  # 使用传入的 chapter_outlines
        elif self._has_custom_characters():
            self._generate_chapter_outlines_from_design()
        elif not self.chapter_outlines:
            self._generate_generic_outlines()

        # 健康趋势跟踪（按章号顺序）
        self.health_trend: list[dict] = []

        # 加载或重置故事状态
        if reset:
            self.story_state = reset_state()
        else:
            self.story_state = load_state()

        # 初始化各子 Agent
        self.writer = WriterAgent()
        self.scribe = ScribeAgent()
        self.editor = EditorAgent()
        self.reader = ReaderAgent()

    def _has_custom_characters(self) -> bool:
        """检查是否有自定义角色名"""
        return any([
            self.protagonist,
            self.mentor,
            self.antagonist,
            self.heroine
        ])

    def _load_story_design_characters(self) -> dict:
        """从 story_design.json 加载角色信息"""
        if not os.path.exists(STORY_DESIGN_PATH):
            return {}
        try:
            with open(STORY_DESIGN_PATH, "r", encoding="utf-8") as f:
                design = json.load(f)
            characters = design.get("main_characters", []) or []
            char_map = {}
            for c in characters:
                role = c.get("role", "")
                char_map[role] = {
                    "name": c.get("name", ""),
                    "archetype": c.get("archetype", ""),
                    "starting_status": c.get("starting_status", ""),
                }
            return char_map
        except Exception:
            return {}

    def _generate_chapter_outlines_from_design(self) -> None:
        """根据 story_design.json 的角色设定生成章节大纲"""
        design_chars = self._load_story_design_characters()

        # 确定角色名（命令行参数优先）
        protag = self.protagonist or design_chars.get("主角", {}).get("name", "") or "主角"
        mentor = self.mentor or design_chars.get("金手指", {}).get("name", "") or "神秘老者"
        antagonist = self.antagonist or design_chars.get("初期对手", {}).get("name", "") or "反派"
        heroine = self.heroine or design_chars.get("女主", {}).get("name", "") or "女主"

        # 获取角色模板
        protag_arch = design_chars.get("主角", {}).get("archetype", "成长型")
        mentor_arch = design_chars.get("金手指", {}).get("archetype", "引路者")
        antagonist_arch = design_chars.get("初期对手", {}).get("archetype", "敌对者")
        heroine_arch = design_chars.get("女主", {}).get("archetype", "搭档")

        # 根据角色模板和用户指定的 climax_type 生成不同类型的开局
        self._generate_outlines_by_archetype(
            protag, mentor, antagonist, heroine,
            protag_arch, mentor_arch, antagonist_arch, heroine_arch
        )

    def _generate_outlines_by_archetype(self, protag: str, mentor: str,
                                          antagonist: str, heroine: str,
                                          protag_arch: str, mentor_arch: str,
                                          antagonist_arch: str, heroine_arch: str) -> None:
        """根据高潮类型生成章节大纲（纯通用，无硬编码套路）"""
        if self.climax_type == "复仇":
            self._generate_revenge_outlines(protag, mentor, antagonist, heroine)
        elif self.climax_type == "探险":
            self._generate_exploration_outlines(protag, mentor, antagonist, heroine)
        elif self.climax_type == "崛起":
            self._generate_rise_outlines(protag, mentor, antagonist, heroine)
        else:
            # 默认为通用开局
            self._generate_rise_outlines(protag, mentor, antagonist, heroine)

    def _generate_revenge_outlines(self, protag: str, mentor: str,
                                     antagonist: str, heroine: str) -> None:
        """生成复仇流章节大纲"""
        self.chapter_outlines = {
            1: (
                f"{protag}的生活被一场变故彻底改变，{antagonist}的出现让他陷入绝境。"
                f"在最危急的时刻，{mentor}出手相救，并告诉{protag}他身上隐藏着巨大的潜力。"
                f"{protag}下定决心要变强，讨回公道。"
                f"本章需要：开场快速进入冲突，建立共情，结尾留下金手指激活的悬念。"
            ),
            2: (
                f"{protag}开始跟随{mentor}修炼，意外发现了自己独特的天赋。"
                f"为突破瓶颈，他独自前往危险之地寻找机缘，遭遇了一场惊险战斗。"
                f"本章需要：战斗场面紧凑，展现{protag}的潜力，结尾发现关于{antagonist}势力的线索。"
            ),
            3: (
                f"{protag}在追寻线索时，偶遇神秘的{heroine}。两人发现各自的目标"
                f"似乎指向同一件事。{antagonist}的势力开始注意到{protag}。"
                f"本章需要：新角色的引入要自然，埋下阵营对立与情感纠葛的种子。"
            ),
        }

    def _generate_exploration_outlines(self, protag: str, mentor: str,
                                        antagonist: str, heroine: str) -> None:
        """生成探险流章节大纲"""
        self.chapter_outlines = {
            1: (
                f"{protag}在偶然中发现了一处古老遗迹的入口。在探索中遭遇危险，"
                f"被残魂状态的{mentor}所救，获知一个关于上古传承的秘密。"
                f"本章需要：开场即营造探险氛围，遗迹环境的细节描写，结尾留下关于更大秘密的钩子。"
            ),
            2: (
                f"{protag}深入遗迹，发现记载着超凡功法的石碑。{mentor}解读后，"
                f"{protag}尝试修炼，实力突飞猛进。守护遗迹的神秘生灵出现，对他进行考验。"
                f"本章需要：探险氛围层层递进，功法领悟过程的悬念感，守护者的测试。"
            ),
            3: (
                f"{protag}离开遗迹后，在附近的据点遇到了正在被追杀的{heroine}。"
                f"{heroine}因知道某个秘密而被{antagonist}的势力追杀。"
                f"{protag}出手相助，两人结伴而行。"
                f"本章需要：动作场面干脆利落，{heroine}的性格鲜明，结尾留下{antagonist}势力的伏笔。"
            ),
        }

    def _generate_rise_outlines(self, protag: str, mentor: str,
                                 antagonist: str, heroine: str) -> None:
        """生成通用章节大纲"""
        self.chapter_outlines = {
            1: (
                f"开场：{protag}生活在某处，看似平凡却暗藏不凡。一个突发事件——{antagonist}的出现"
                f"或某个机缘——打破了他的平静生活。在关键时刻，{mentor}现身，揭示了一个关于{protag}的秘密。"
                f"本章需要：快速建立世界观和角色印象，用事件钩住读者，结尾留下悬念。"
            ),
            2: (
                f"{protag}开始探索新的力量。他遇到了{heroine}，两人之间的交集"
                f"为剧情注入了新的变数。{antagonist}的势力在暗中活动。"
                f"本章需要：丰富世界观设定，推进主线，展现{protag}的成长与抉择。"
            ),
            3: (
                f"冲突升级。{protag}与{antagonist}的势力正面交锋，初战告捷"
                f"但发现真正的威胁远超想象。更大的秘密被揭开一角。"
                f"本章需要：紧张感递进，第一场关键对抗，结尾为后续留下强大悬念。"
            ),
        }

    def _generate_generic_outlines(self) -> None:
        """生成通用章节大纲（最小化套路）"""
        self.chapter_outlines = {
            1: (
                "【开篇】主角处于困境但未绝望，意外获得改变命运的机遇。"
                "本章需要：开场即困境展现主角心性，机遇出现时制造期待感，结尾留下悬念钩子。"
            ),
            2: (
                "【成长】主角利用机遇快速成长，初次展现实力。"
                "本章需要：成长过程有波折而非一帆风顺，结尾遇到新挑战或发现新线索。"
            ),
            3: (
                "【考验】主角面临第一次重大考验，与某个对手产生冲突。"
                "本章需要：冲突要有深层原因而非简单打脸，结尾留下更大阴谋的伏笔。"
            ),
        }

    def run(self, context: dict | None = None) -> dict:
        """运行主控循环"""
        context = context or {}
        results: list[dict] = []

        print(f"\n{'='*60}")
        print(f"  网文自动写作系统 启动")
        print(f"  故事梗概: {self.outline}")
        print(f"  总章节数: {self.total_chapters}")
        print(f"  目标字数: {self.target_word_count}/章")
        print(f"  目标平台: {self.target_platform}")
        print(f"  已有角色数: {len(self.story_state.get('characters', {}))}")
        print(f"{'='*60}\n")

        for chapter in range(1, self.total_chapters + 1):
            print(f"--- 第 {chapter} 章 ---")

            chapter_outline = self.chapter_outlines.get(chapter, "")
            if not chapter_outline:
                logger.warning("第 %d 章没有预设大纲，使用空字符串", chapter)

            # ① 作家写正文
            writer_result = self._run_writer(chapter, chapter_outline, context)
            chapter_text = writer_result["content"]

            # ② 书记员更新设定
            scribe_result = self._run_scribe(chapter, chapter_text)

            # 持久化
            save_state(self.story_state)

            # ③ 编辑审校
            editor_result = self._run_editor(chapter, chapter_text)

            # ④ Editor 决策为 rewrite → 重写（最多 max_rewrite 次）
            rewrite_round = 0
            while (editor_result.get("decision") == "rewrite"
                   and rewrite_round < self.max_rewrite):
                rewrite_round += 1
                print(f"  [Editor] 触发重写（第 {rewrite_round} 次），根据错误列表重写")
                logger.info("  [Editor] 触发重写（第 %d 次）", rewrite_round)
                writer_result = self._run_writer(
                    chapter, chapter_outline, context,
                    rewrite_hints=(editor_result.get("hard_errors", [])
                                   + editor_result.get("tech_issues", [])),
                )
                chapter_text = writer_result["content"]
                # 重新审校
                editor_result = self._run_editor(chapter, chapter_text)

            # ⑤ 读者反馈
            reader_result = self._run_reader(chapter, chapter_text)

            # ⑥ 保存全章日志
            log_payload = {
                "chapter": chapter,
                "outline": chapter_outline,
                "writer": writer_result,
                "scribe": scribe_result,
                "editor": editor_result,
                "reader": reader_result,
                "rewrite_rounds": rewrite_round,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            log_path = _save_chapter_log(chapter, log_payload)
            logger.info("  [Log]  本章日志已保存: %s", log_path)

            # ⑦ 章节健康报告（per-chapter summary）
            health = self._chapter_health_report(chapter, log_payload)
            log_payload["health_report"] = health
            self.health_trend.append(health)
            self._print_chapter_health(chapter, health)

            # 重新保存（含 health_report）
            _save_chapter_log(chapter, log_payload)

            # ⑧ 保存章节正文到输出路径
            self._save_chapter_output(chapter, chapter_text, health)

            results.append(log_payload)
            print()

        # 最终状态摘要
        self._print_summary()

        # ⑧ 全本趋势报告
        self._print_trend_report()
        self._save_trend_report(results)

        return {
            "outline": self.outline,
            "total_chapters": self.total_chapters,
            "final_state": self.story_state,
            "results": results,
            "health_trend": self.health_trend,
        }

    # ════════════════════════════════════════════════════════════
    #  各 Agent 调用与打印
    # ════════════════════════════════════════════════════════════

    def _run_writer(self, chapter: int, outline: str, ctx: dict,
                    rewrite_hints: list[str] | None = None) -> dict:
        writer_ctx = {
            "outline": outline,
            "current_state": self.story_state,
            "chapter_number": chapter,
            "target_word_count": self.target_word_count,
            "current_chapter": chapter,
            "chapter": chapter,
            "target_platform": self.target_platform,
            "climax_type": self.climax_type,  # 传递用户自定义高潮类型
            **({"rewrite_hints": rewrite_hints} if rewrite_hints else {}),
            **ctx,
        }
        result = self.writer.run(writer_ctx)
        wc = result.get("word_count", len(result["content"]))
        target = result.get("target_word_count", self.target_word_count)
        ratio = wc / target if target else 0
        tag = " [重写]" if rewrite_hints else ""
        print(f"  [Writer] {tag} 生成正文 {wc} / {target} 字 ({ratio:.0%})")
        preview = result["content"][:60].replace("\n", " / ")
        print(f"            开头: {preview}...")
        return result

    def _run_scribe(self, chapter: int, chapter_text: str) -> dict:
        scribe_ctx = {
            "current_chapter": chapter,
            "chapter_text": chapter_text,
            "current_state": self.story_state,
        }
        result = self.scribe.run(scribe_ctx)
        self.story_state = result["state"]
        save_state(self.story_state)
        chars = self.story_state.get("characters", {})
        locs = self.story_state.get("locations", {})
        realms = self.story_state.get("power_system", {}).get("realms", [])
        fw = self.story_state.get("foreshadowing", [])
        warnings = result.get("warnings", [])
        print(f"  [Scribe]  角色 {len(chars)} | 地点 {len(locs)} | "
              f"境界 {len(realms)} | 伏笔 {len(fw)} | 警告 {len(warnings)}")
        if warnings:
            for w in warnings[:3]:
                print(f"            [!] {w}")
        return result

    def _run_editor(self, chapter: int, chapter_text: str) -> dict:
        editor_ctx = {
            "chapter_text": chapter_text,
            "current_state": self.story_state,
            "chapter_number": chapter,
            "current_chapter": chapter,
            "chapter": chapter,
        }
        result = self.editor.run(editor_ctx)
        decision = result.get("decision", "pass")
        score = result.get("quality_score", 0)
        hard_n = len(result.get("hard_errors", []) or [])
        tech_n = len(result.get("tech_issues", []) or [])
        print(f"  [Editor]  决策 {decision} | 评分 {score} | "
              f"hard={hard_n} tech={tech_n}")
        # 打印错误明细
        for h in (result.get("hard_errors") or [])[:3]:
            print(f"            [x] {h}")
        for t in (result.get("tech_issues") or [])[:3]:
            print(f"            - {t}")
        return result

    def _run_reader(self, chapter: int, chapter_text: str) -> dict:
        reader_ctx = {
            "chapter_text": chapter_text,
            "chapter_number": chapter,
            "current_chapter": chapter,
            "chapter": chapter,
            "target_platform": self.target_platform,
            "outline": self.chapter_outlines.get(chapter, ""),
        }
        result = self.reader.run(reader_ctx)
        print(f"  [Reader]  吸引力 {result.get('attraction_score', 0)} | "
              f"AI味 {result.get('ai_odor_score', 0)} | "
              f"弃读 {result.get('abandon_risk', '?')} | "
              f"续读 {result.get('continuation_probability', 0)}% | "
              f"KB平台={result.get('kb_platform', '?')} 爽点命中={result.get('kb_satisfaction_matched', False)}")
        if result.get("feedback"):
            print(f"            -> {result['feedback']}")
        for s in (result.get("suggestions") or [])[:2]:
            print(f"            -> {s}")
        return result

    # ════════════════════════════════════════════════════════════
    #  摘要
    # ════════════════════════════════════════════════════════════

    def _print_summary(self) -> None:
        print("="*60)
        print("  全章节写作完成")
        print("="*60)
        chars = self.story_state.get("characters", {})
        if chars:
            print("  角色列表:")
            for name, info in chars.items():
                print(f"    - {name}: {info.get('cultivation_realm', '?')}, "
                      f"{info.get('faction', '?')}, "
                      f"{info.get('personality_tags', [])}")
        locs = self.story_state.get("locations", {})
        if locs:
            print(f"  地点 ({len(locs)}): {', '.join(locs.keys())}")
        timeline = self.story_state.get("timeline", [])
        if timeline:
            print(f"  时间线事件: {len(timeline)} 条")
        foreshadowing = self.story_state.get("foreshadowing", [])
        if foreshadowing:
            print(f"  伏笔: {len(foreshadowing)} 条")
            for fw in foreshadowing[:3]:
                print(f"    - [{fw.get('planted_chapter', '?')}章] "
                      f"{fw.get('clue', '')[:60]}")
        print()

    # ════════════════════════════════════════════════════════════
    #  章节健康报告
    # ════════════════════════════════════════════════════════════

    def _chapter_health_report(self, chapter: int, payload: dict) -> dict:
        """单章健康报告：汇总 4 Agent 输出"""
        writer = payload.get("writer", {})
        scribe = payload.get("scribe", {})
        editor = payload.get("editor", {})
        reader = payload.get("reader", {})
        rewrite_rounds = payload.get("rewrite_rounds", 0)

        # 字数达成率
        wc = writer.get("word_count", 0)
        target = writer.get("target_word_count", self.target_word_count)
        wc_ratio = round(wc / target, 2) if target else 0.0

        # 设定增量（基于 state 自身前后对比）
        extracted = scribe.get("extracted", {}) or {}
        new_chars = extracted.get("new_characters", []) or []
        new_locs = extracted.get("new_locations", []) or []
        ps = extracted.get("power_system", {}) or {}
        new_realms = ps.get("new_realms", []) or []
        new_tech = ps.get("new_techniques", []) or []
        new_fw = extracted.get("foreshadowing", []) or []
        new_tl = extracted.get("timeline_events", []) or []
        scribe_warnings = scribe.get("warnings", []) or []

        # 健康评级（4 Agent 联动）
        quality = editor.get("quality_score", 0)
        decision = editor.get("decision", "pass")
        hard_n = len(editor.get("hard_errors", []) or [])
        tech_n = len(editor.get("tech_issues", []) or [])
        attraction = reader.get("attraction_score", 0)
        ai_odor = reader.get("ai_odor_score", 0)
        continuation = reader.get("continuation_probability", 0)
        abandon = reader.get("abandon_risk", "low")

        # 综合分
        composite = round(
            (quality * 0.4) + (attraction * 0.4) + ((100 - ai_odor) * 0.2),
            1
        )

        # 评级
        if composite >= 80 and hard_n == 0:
            grade = "A"
        elif composite >= 65 and hard_n == 0:
            grade = "B"
        elif composite >= 50:
            grade = "C"
        else:
            grade = "D"

        # 硬错误类型聚类
        hard_types = Counter()
        for h in (editor.get("hard_errors") or []):
            if "修为" in h or "境界" in h:
                hard_types["修为不一致"] += 1
            elif "姓名" in h or "称呼" in h:
                hard_types["称谓错乱"] += 1
            elif "时间" in h or "顺序" in h:
                hard_types["时间线错乱"] += 1
            elif "势力" in h or "宗门" in h:
                hard_types["势力归属"] += 1
            else:
                hard_types["其他"] += 1

        # 口吻一致性快速校验
        bp = writer.get("blueprint", {}) or {}
        dialogue_cards = bp.get("dialogue_cards", []) or []
        content = writer.get("content", "") or ""
        out_chars = sorted({
            name for name in (self.story_state.get("characters", {}) or {}).keys()
            if name in content
        })

        return {
            "chapter": chapter,
            "composite_score": composite,
            "grade": grade,
            "word_count": wc,
            "target_word_count": target,
            "word_completion": wc_ratio,
            "rewrite_rounds": rewrite_rounds,
            "editor": {
                "decision": decision,
                "quality_score": quality,
                "hard_errors": hard_n,
                "tech_issues": tech_n,
                "hard_error_types": dict(hard_types),
            },
            "reader": {
                "attraction_score": attraction,
                "ai_odor_score": ai_odor,
                "abandon_risk": abandon,
                "continuation_probability": continuation,
                "kb_platform": reader.get("kb_platform"),
                "kb_satisfaction_matched": reader.get("kb_satisfaction_matched"),
            },
            "scribe": {
                "delta_chars": len(new_chars),
                "delta_locs": len(new_locs),
                "delta_realms": len(new_realms),
                "delta_techniques": len(new_tech),
                "delta_foreshadowing": len(new_fw),
                "delta_timeline": len(new_tl),
                "warnings": len(scribe_warnings),
            },
            "voice_consistency": {
                "planned_cards": len(dialogue_cards),
                "actual_chars": out_chars,
            },
            "foreshadowing": {
                "planted_total": len(self.story_state.get("foreshadowing", []) or []),
                "resolved_total": sum(
                    1 for fw in (self.story_state.get("foreshadowing", []) or [])
                    if fw.get("status") == "resolved"
                ),
            },
        }

    def _print_chapter_health(self, chapter: int, h: dict) -> None:
        grade = h["grade"]
        grade_sym = {"A": "[A]", "B": "[B]", "C": "[C]", "D": "[D]"}.get(grade, "[?]")
        print(f"  +- [ 章节健康报告 ] 第 {chapter} 章 {'-'*25}")
        print(f"  │ {grade_sym} 综合评级: {grade}（{h['composite_score']} / 100）")
        wc = h["word_count"]
        target = h["target_word_count"]
        print(f"  │ 字数: {wc}/{target} ({int(h['word_completion']*100)}%) | "
              f"重写轮次: {h['rewrite_rounds']}")
        ed = h["editor"]
        print(f"  │ [Editor] 决策={ed['decision']} 评分={ed['quality_score']} "
              f"硬错={ed['hard_errors']} 技错={ed['tech_issues']}")
        if ed["hard_error_types"]:
            types_str = ", ".join(f"{k}:{v}" for k, v in ed["hard_error_types"].items())
            print(f"  │   硬错分类 → {types_str}")
        rd = h["reader"]
        print(f"  │ [Reader] 吸引={rd['attraction_score']} AI味={rd['ai_odor_score']} "
              f"续读={rd['continuation_probability']}% 弃读={rd['abandon_risk']}")
        sc = h["scribe"]
        print(f"  │ [Scribe] Δ角色={sc['delta_chars']} Δ地点={sc['delta_locs']} "
              f"Δ境界={sc['delta_realms']} Δ功法={sc['delta_techniques']} "
              f"Δ伏笔={sc['delta_foreshadowing']} Δ事件={sc['delta_timeline']} "
              f"警告={sc['warnings']}")
        vc = h["voice_consistency"]
        if vc["actual_chars"]:
            print(f"  │ [口吻] 规划卡片={vc['planned_cards']} 张 / 实际出场={','.join(vc['actual_chars'])}")
        fw = h["foreshadowing"]
        print(f"  │ [伏笔] 累计铺设={fw['planted_total']} 累计回收={fw['resolved_total']}")
        print(f"  └─────────────────────────────────────")

    # ════════════════════════════════════════════════════════════
    #  全本趋势报告
    # ════════════════════════════════════════════════════════════

    def _print_trend_report(self) -> None:
        if not self.health_trend:
            return
        n = len(self.health_trend)
        composites = [h["composite_score"] for h in self.health_trend]
        ai_odors = [h["reader"]["ai_odor_score"] for h in self.health_trend]
        continuations = [h["reader"]["continuation_probability"] for h in self.health_trend]
        rewrites = sum(h["rewrite_rounds"] for h in self.health_trend)
        hard_total = sum(h["editor"]["hard_errors"] for h in self.health_trend)
        grades = Counter(h["grade"] for h in self.health_trend)
        avg_composite = round(sum(composites) / n, 1)
        avg_ai = round(sum(ai_odors) / n, 1)
        avg_cont = round(sum(continuations) / n, 1)
        rewrite_rate = round(rewrites / n * 100, 1)

        # AI 味趋势：后半段 < 前半段 = 改善
        half = n // 2
        ai_first = sum(ai_odors[:half]) / max(1, half) if half else 0
        ai_later = sum(ai_odors[half:]) / max(1, n - half) if n - half else 0
        ai_trend = "↓ 改善" if ai_later < ai_first else ("↑ 恶化" if ai_later > ai_first else "→ 平稳")

        # 续读趋势
        cont_first = sum(continuations[:half]) / max(1, half) if half else 0
        cont_later = sum(continuations[half:]) / max(1, n - half) if n - half else 0
        cont_trend = "↑ 提升" if cont_later > cont_first else ("↓ 下降" if cont_later < cont_first else "→ 平稳")

        # 伏笔回收率
        last_fw = self.health_trend[-1]["foreshadowing"]
        planted = last_fw["planted_total"]
        resolved = last_fw["resolved_total"]
        recycle_rate = round(resolved / planted * 100, 1) if planted else 0.0

        # 角色数增长
        first_chars = self.health_trend[0]["scribe"]["delta_chars"] if self.health_trend else 0

        print()
        print("=" * 60)
        print("  [全本趋势报告] " + str(n) + " 章")
        print("=" * 60)
        print(f"  综合分均值:        {avg_composite} / 100")
        print(f"  AI 味均值:         {avg_ai} / 100    趋势 {ai_trend}")
        print(f"  续读概率均值:      {avg_cont} %      趋势 {cont_trend}")
        print(f"  重写率:            {rewrite_rate}%   （共 {rewrites} 次重写）")
        print(f"  累计硬错误:        {hard_total} 条")
        print(f"  评级分布:          " + "  ".join(
            f"{g}:{grades.get(g,0)}" for g in ("A", "B", "C", "D")
        ))
        print(f"  伏笔铺设 / 回收:   {planted} / {resolved}  回收率 {recycle_rate}%")
        print(f"  状态健康度:        角色={len(self.story_state.get('characters', {}))} "
              f"地点={len(self.story_state.get('locations', {}))} "
              f"境界={len(self.story_state.get('power_system', {}).get('realms', []))} "
              f"时间线={len(self.story_state.get('timeline', []))}")

        # 健康度条形图
        print()
        print("  综合分折线（每章一格）：")
        line = "  "
        for h in self.health_trend:
            score = int(h["composite_score"])
            if score >= 80:
                mark = "█"
            elif score >= 65:
                mark = "▓"
            elif score >= 50:
                mark = "▒"
            else:
                mark = "░"
            line += mark
        print(line + f"  ← {composites[0]} → {composites[-1]}")

        # AI 味折线
        print("  AI 味折线（每章一格，越低越好）：")
        line = "  "
        for h in self.health_trend:
            score = int(h["reader"]["ai_odor_score"])
            if score <= 20:
                mark = "█"
            elif score <= 40:
                mark = "▓"
            elif score <= 60:
                mark = "▒"
            else:
                mark = "░"
            line += mark
        print(line + f"  ← {int(ai_odors[0])} → {int(ai_odors[-1])}")

        # 续读折线
        print("  续读概率折线（每章一格，越高越好）：")
        line = "  "
        for h in self.health_trend:
            score = int(h["reader"]["continuation_probability"])
            if score >= 70:
                mark = "█"
            elif score >= 55:
                mark = "▓"
            elif score >= 40:
                mark = "▒"
            else:
                mark = "░"
            line += mark
        print(line + f"  ← {int(continuations[0])} → {int(continuations[-1])}")
        print()

    def _save_trend_report(self, results: list[dict]) -> str:
        """保存全本趋势报告到 JSON"""
        _ensure_logs_dir()
        path = os.path.join(LOGS_DIR, "trend_report.json")
        report = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "outline": self.outline,
            "total_chapters": self.total_chapters,
            "platform": self.target_platform,
            "trend": self.health_trend,
            "summary": {
                "avg_composite": round(
                    sum(h["composite_score"] for h in self.health_trend)
                    / max(1, len(self.health_trend)), 1),
                "avg_ai_odor": round(
                    sum(h["reader"]["ai_odor_score"] for h in self.health_trend)
                    / max(1, len(self.health_trend)), 1),
                "avg_continuation": round(
                    sum(h["reader"]["continuation_probability"] for h in self.health_trend)
                    / max(1, len(self.health_trend)), 1),
                "total_rewrites": sum(h["rewrite_rounds"] for h in self.health_trend),
                "total_hard_errors": sum(h["editor"]["hard_errors"] for h in self.health_trend),
                "final_state": {
                    "characters": len(self.story_state.get("characters", {})),
                    "locations": len(self.story_state.get("locations", {})),
                    "realms": len(self.story_state.get("power_system", {}).get("realms", [])),
                    "foreshadowing_planted": len(self.story_state.get("foreshadowing", []) or []),
                },
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info("  [Trend] 全本趋势报告已保存: %s", path)
        return path

    # ════════════════════════════════════════════════════════════
    #  章节正文输出
    # ════════════════════════════════════════════════════════════

    def _save_chapter_output(self, chapter: int, text: str,
                              health: dict | None = None) -> str | None:
        """将本章正文保存到输出路径（纯文本）

        如果指定了 output_file（单个文件），所有章节追加写入。
        否则按 chapter_N.txt 格式写入 output_dir 目录。
        """
        # 确定输出路径
        if self.output_file:
            # 单文件模式：追加写入
            out_path = self.output_file
            mode = "a" if chapter > 1 else "w"
        elif self.output_dir:
            # 目录模式：每章独立文件
            out_dir = self.output_dir
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"chapter_{chapter:03d}.txt")
            mode = "w"
        else:
            return None  # 未指定输出路径

        try:
            with open(out_path, mode, encoding="utf-8") as f:
                if mode == "a":
                    f.write("\n\n" + "=" * 60 + "\n\n")
                header = f"第 {chapter} 章\n{'='*40}\n\n"
                f.write(header)
                f.write(text)
                if health:
                    f.write(f"\n\n{'='*40}\n")
                    f.write(f"[生成报告] 字数={health.get('word_count', len(text))} | "
                            f"Editor评分={health.get('editor', {}).get('quality_score', '?')} | "
                            f"读者吸引力={health.get('reader', {}).get('attraction_score', '?')}")
            print(f"  [Output] 章节已保存: {out_path}")
            return out_path
        except Exception as e:
            logger.error("保存章节输出失败: %s", e)
            return None
