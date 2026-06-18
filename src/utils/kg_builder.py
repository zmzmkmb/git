"""KG Builder —— 从策略文件自动抽取三元组，增量构建知识图谱

用法：
    python -m src.utils.kg_builder --input-dir ./prompt_engineering --output ./triples.json

设计要点：
- 扫描 input-dir 下的 .md / .yaml / .yml 文件
- 用 LLM 按统一 schema 抽取三元组 (head, relation, tail)
- 增量合并到已有 triples.json（如有），去重后输出
- 可集成到 CI：KG 更新自动化
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any

try:
    from src.utils.llm_client import call_llm
except ImportError:
    call_llm = None  # type: ignore[assignment]
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Schema 定义 ──────────────────────────────────────────────────
# LLM 抽取时使用的统一 schema
EXTRACTION_SYSTEM_PROMPT = """你是一个知识图谱三元组抽取专家。
请从以下文本中抽取出所有写作相关的知识三元组。

输出格式：严格 JSON 数组
[
  {
    "head": "主体概念（简洁，如'爽点'而非'爽点的定义是什么'）",
    "relation": "关系类型（从以下列表中选择最匹配的）",
    "tail": "客体描述（完整语句，10-60字）",
    "source_doc": "来源文件名（传入的 filename 参数）",
    "source_chunk": 0
  }
]

关系类型列表（请严格选其一）：
- 定义：对概念下定义
- 要求：对某事物的硬性要求/必要条件
- 建议：推荐的做法或技巧
- 避免：应该回避的做法
- 导致：因果关系（某行为会导致某后果）
- 特征/特点：事物的特征描述
- 方法：实现某目标的具体方法
- 原则：需要遵循的原则
- 注意：需要留意的事项
- 作用：某事物的作用或功能
- 技巧：写作技巧
- 要点：关键要点
- 包含/包括：组成关系
- 分为：分类关系

抽取规则：
1. head 必须是简洁的名词性概念（≤8字）
2. tail 是完整的陈述句（10-60字）
3. 只抽取与网文/写作/创作直接相关的三元组
4. 不抽取常识性、非写作内容
5. head 和 tail 都必须完整，不可用省略号
6. 如果文本中没有可抽取的三元组，返回空数组 []
"""


# ── 文件扫描 ──────────────────────────────────────────────────

def scan_input_files(input_dir: str) -> list[dict[str, Any]]:
    """扫描 input_dir 下所有 .md/.yaml 文件，返回 [{path, name, content}]"""
    results: list[dict[str, Any]] = []
    if not os.path.isdir(input_dir):
        logger.warning("输入目录不存在: %s", input_dir)
        return results

    for root, _dirs, files in os.walk(input_dir):
        for fname in files:
            if not (fname.endswith(".md") or fname.endswith(".yaml") or fname.endswith(".yml")):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception as e:
                logger.warning("  跳过 %s: %s", fname, e)
                continue
            if len(content.strip()) < 50:
                logger.debug("  跳过 %s: 内容过短", fname)
                continue
            results.append({"path": fpath, "name": fname, "content": content})
            logger.info("  扫描: %s (%d 字符)", fname, len(content))
    return results


# ── LLM 抽取 ──────────────────────────────────────────────────

def extract_triples_from_file(file_info: dict[str, Any],
                               model: str = "moonshot-v1-32k") -> list[dict]:
    """对单个文件调用 LLM 抽取三元组"""
    content = file_info["content"]
    fname = file_info["name"]

    # 截断过长内容（LLM 上下文限制）
    max_chars = 6000
    if len(content) > max_chars:
        logger.info("  %s 过长 (%d 字符)，截取前 %d 字符", fname, len(content), max_chars)
        content = content[:max_chars]

    user_prompt = f"""文件名：{fname}

文本内容：
{content}

请从中抽取所有写作相关的知识三元组，输出 JSON 数组。"""

    try:
        resp = call_llm(
            model=model,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=2000,
            temperature=0.1,
        )
        triples = _parse_llm_response(resp, fname)
        logger.info("  %s → 抽取 %d 条三元组", fname, len(triples))
        return triples
    except Exception as e:
        logger.warning("  %s LLM 调用失败: %s", fname, e)
        return []


def _parse_llm_response(resp: str, fname: str) -> list[dict]:
    """解析 LLM 返回的 JSON"""
    # 去除 markdown 代码块
    text = re.sub(r"^```(?:json)?\s*", "", resp.strip())
    text = re.sub(r"\s*```$", "", text.strip())
    # 抽取 JSON 数组
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        logger.warning("  %s: LLM 返回中未找到 JSON 数组", fname)
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        logger.warning("  %s: JSON 解析失败", fname)
        return []

    validated: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        h = (item.get("head") or "").strip()
        r = (item.get("relation") or "").strip()
        t = (item.get("tail") or "").strip()
        if not h or not r or not t or len(t) < 4:
            continue
        validated.append({
            "head": h,
            "relation": r,
            "tail": t,
            "source_doc": item.get("source_doc", fname),
            "source_chunk": item.get("source_chunk", 0),
        })
    return validated


# ── 增量合并 ──────────────────────────────────────────────────

def merge_triples(existing: list[dict], new_triples: list[dict]) -> list[dict]:
    """去重合并：以 (head, relation, tail) 为唯一键"""
    seen: set[tuple[str, str, str]] = set()
    merged: list[dict] = []

    for t in existing:
        key = (t.get("head", ""), t.get("relation", ""), t.get("tail", ""))
        if key not in seen:
            seen.add(key)
            merged.append(t)

    for t in new_triples:
        key = (t.get("head", ""), t.get("relation", ""), t.get("tail", ""))
        if key not in seen:
            seen.add(key)
            merged.append(t)

    return merged


# ── 主流程 ──────────────────────────────────────────────────

def build(input_dir: str, output_path: str, model: str = "moonshot-v1-32k") -> int:
    """主构建函数

    Returns:
        本次新增的三元组数量
    """
    logger.info("=" * 50)
    logger.info("KG Builder 启动")
    logger.info("  输入目录: %s", input_dir)
    logger.info("  输出路径: %s", output_path)
    logger.info("  LLM 模型: %s", model)
    logger.info("=" * 50)

    # Step 1: 扫描输入文件
    files = scan_input_files(input_dir)
    if not files:
        logger.warning("未找到可处理的策略文件")
        return 0

    # Step 2: 逐文件 LLM 抽取
    all_new: list[dict] = []
    for fi in files:
        triples = extract_triples_from_file(fi, model=model)
        all_new.extend(triples)

    if not all_new:
        logger.info("未抽取出任何三元组")
        return 0

    # Step 3: 加载已有数据（增量合并）
    existing: list[dict] = []
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            logger.info("加载已有 triples.json: %d 条", len(existing))
        except Exception:
            logger.warning("加载已有文件失败，将新建")

    merged = merge_triples(existing, all_new)
    added = len(merged) - len(existing)

    # Step 4: 写出
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    logger.info("写入完成: %s (%d 条, 新增 %d 条)",
                 output_path, len(merged), added)
    return added


# ── CLI ──────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="KG Builder — 从策略文件自动抽取三元组",
    )
    parser.add_argument(
        "--input-dir", "-i",
        default="./prompt_engineering",
        help="策略文件目录（含 .md/.yaml），默认 ./prompt_engineering",
    )
    parser.add_argument(
        "--output", "-o",
        default="./kg_output/triples.json",
        help="输出路径，默认 ./kg_output/triples.json",
    )
    parser.add_argument(
        "--model", "-m",
        default="moonshot-v1-32k",
        help="LLM 模型名，默认 moonshot-v1-32k",
    )
    args = parser.parse_args()
    build(args.input_dir, args.output, args.model)


if __name__ == "__main__":
    main()
