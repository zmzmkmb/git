"""网文自动写作系统 - 主入口

用法示例：
  # 基本用法：3章，字数3000，默认设定
  python -m src.main --reset

  # 自定义角色名 + 高潮类型（崛起/复仇/探险）
  python -m src.main --protagonist "主角名" --mentor "导师名" --antagonist "反派名" --heroine "女主名" --climax "崛起" --chapters 5 --reset

  # 使用外部设定文件（JSON格式）
  python -m src.main --design "my_story.json" --output "./output"

  # 使用外部设定文件并让 AI 补全缺失部分
  python -m src.main --design "my_story_partial.json" --auto-complete --chapters 10 --output "./output"

  # 每章字数控制
  python -m src.main --words 5000 --chapters 3 --reset

  # 指定输出路径
  python -m src.main --output "E:/我的小说/第一章.txt" --chapters 1 --reset
"""

import argparse
import json
import os
from src.agents.orchestrator import Orchestrator

# 默认设定文件路径
DEFAULT_STORY_DESIGN = os.path.join(
    os.path.dirname(__file__), "data", "story_design.json"
)


def load_custom_design(path: str) -> dict | None:
    """加载用户自定义设定文件"""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[错误] 加载设定文件失败: {e}")
        return None


def resolve_output_path(args) -> str:
    """根据参数确定输出路径"""
    if args.output:
        return os.path.abspath(args.output)

    # 默认输出到 src/data/output/
    default_dir = os.path.join(os.path.dirname(__file__), "data", "output")
    os.makedirs(default_dir, exist_ok=True)
    return default_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="网文自动写作系统 - 多Agent协作生成网文",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python -m src.main --reset --chapters 1
  python -m src.main --design my_design.json --auto-complete --chapters 10 --output ./my_novel
  python -m src.main --protagonist "叶凡" --mentor "老疯子" --antagonist "王腾" --climax "崛起" -n 5 -w 3000 --reset
        """
    )

    # --- 章节控制 ---
    parser.add_argument("--chapters", "-n", type=int, default=3,
                        help="总章节数（默认3）")
    parser.add_argument("--words", "-w", type=int, default=3000,
                        help="每章目标字数（默认3000）")

    # --- 设定来源 ---
    parser.add_argument("--design", "-d", type=str, default="",
                        help="设定JSON文件路径（默认使用 data/story_design.json）")
    parser.add_argument("--auto-complete", action="store_true",
                        help="自动补全设定（当设定文件不完整时，用AI补全缺失部分）")

    # --- 角色参数（覆盖设定文件） ---
    parser.add_argument("--protagonist", type=str, default="",
                        help="主角名称")
    parser.add_argument("--mentor", type=str, default="",
                        help="金手指/导师名称")
    parser.add_argument("--antagonist", type=str, default="",
                        help="反派角色名称")
    parser.add_argument("--heroine", type=str, default="",
                        help="女主名称")
    parser.add_argument("--climax", "-c", type=str, default="",
                        help="高潮类型：崛起/复仇/探险（默认自动推断）")

    # --- 平台与输出 ---
    parser.add_argument("--platform", "-p", type=str, default="番茄",
                        help="目标平台：番茄/起点/晋江（默认番茄）")
    parser.add_argument("--output", "-o", type=str, default="",
                        help="输出路径（文件夹或文件路径，默认 src/data/output/）")
    parser.add_argument("--reset", "-r", action="store_true",
                        help="重置故事状态，从头开始")

    args = parser.parse_args()

    # ================================================================
    #  1. 加载设定
    # ================================================================
    design_path = args.design if args.design else DEFAULT_STORY_DESIGN
    custom_design = load_custom_design(design_path)

    # 如果启用了 auto-complete 且设定不完整，用 LLM 补全
    if args.auto_complete and custom_design:
        from src.utils.design_completer import auto_complete_design
        print("  [AI补全] 正在智能补全故事设定...")
        custom_design = auto_complete_design(
            custom_design,
            total_chapters=args.chapters,
        )
        # 保存补全后的设定到原文件
        try:
            with open(design_path, "w", encoding="utf-8") as f:
                json.dump(custom_design, f, ensure_ascii=False, indent=2)
            print(f"  [AI补全] 补全后的设定已保存至: {design_path}")
        except Exception:
            pass

    # ================================================================
    #  2. 构建用户设定
    # ================================================================
    user_settings = {
        "protagonist": args.protagonist,
        "mentor": args.mentor,
        "antagonist": args.antagonist,
        "heroine": args.heroine,
        "climax_type": args.climax,
    }

    # 故事梗概
    outline = ""
    if custom_design:
        outline = custom_design.get("story_outline", "")
    if not outline:
        outline = "一个关于成长与抉择的故事。"

    # ================================================================
    #  3. 输出路径
    # ================================================================
    output_path = resolve_output_path(args)
    print(f"\n{'='*60}")
    print(f"  网文自动写作系统")
    print(f"{'='*60}")
    print(f"  总章节数: {args.chapters}")
    print(f"  目标字数: {args.words}/章")
    print(f"  目标平台: {args.platform}")
    print(f"  输出路径: {output_path}")

    # 显示设定
    if any(user_settings.values()):
        print(f"  角色设定:")
        for k, v in user_settings.items():
            if v:
                print(f"    - {k}: {v}")

    if custom_design:
        chars = custom_design.get("main_characters", [])
        if chars:
            print(f"  故事角色:")
            for c in chars:
                name = c.get("name", "未知")
                role = c.get("role", "")
                arch = c.get("archetype", "")
                status = c.get("starting_status", "")
                desc = f"{name} ({role}: {arch})"
                if status:
                    desc += f" [{status}]"
                print(f"    - {desc}")
        gen = custom_design.get("world_setting", {}).get("genre", "")
        if gen:
            print(f"  类型: {gen}")

    print(f"{'='*60}\n")

    # ================================================================
    #  4. 运行
    # ================================================================

    # 从 custom_design 提取章节大纲（如果存在）
    chapter_outlines_from_design = None
    if custom_design:
        raw_outlines = custom_design.get("chapter_outlines") or {}
        if raw_outlines:
            chapter_outlines_from_design = {int(k): v for k, v in raw_outlines.items()}

    orchestrator = Orchestrator(
        outline=outline,
        chapter_outlines=chapter_outlines_from_design,
        total_chapters=args.chapters,
        target_word_count=args.words,
        target_platform=args.platform,
        output_dir=output_path if os.path.isdir(output_path) or not os.path.exists(output_path) else os.path.dirname(output_path),
        output_file=output_path if os.path.isfile(output_path) else "",
        reset=args.reset,
        custom_settings=user_settings,
    )
    orchestrator.run()


if __name__ == "__main__":
    main()
