# PA_Agent · 网文自动写作系统

> 基于多 Agent 协作的 AI 小说自动生成框架。
> 故事背景示例：四合院官场 · 1978（改革开放初期至南巡后）

## 项目简介

`PA_Agent` 是一个模块化的多 Agent 写作流水线。通过 `Orchestrator` 调度
`Writer`（写作）、`Scribe`（速记）、`Editor`（编辑）、`Reader`（读者反馈）
四个角色，循环产出章节正文，并对人物弧线、伏笔、节奏做持续校准。

- 设定驱动：可加载外部 JSON 设定文件，也可使用内置默认大纲
- 长篇支持：内置 30 章玄幻/官场全本模板
- 可扩展：每个 Agent 都是独立模块，可替换或新增

## 目录结构

```
.
├── src/
│   ├── main.py                 # 入口脚本（CLI）
│   ├── agents/                 # 各 Agent 实现
│   │   ├── base_agent.py       # Agent 基类
│   │   ├── orchestrator.py     # 主控：调度流水线
│   │   ├── writer.py           # 写作 Agent
│   │   ├── scribe.py           # 速记 Agent
│   │   ├── editor.py           # 编辑 Agent
│   │   └── reader.py           # 读者反馈 Agent
│   ├── utils/                  # 工具模块
│   │   ├── config.py           # 配置加载
│   │   ├── llm_client.py       # LLM 客户端
│   │   ├── kg_builder.py       # 知识图谱构建
│   │   ├── kg_query.py         # 知识图谱查询
│   │   ├── knowledge_retriever.py
│   │   ├── graph_retriever.py
│   │   ├── design_completer.py # 设定补全
│   │   ├── state_manager.py    # 状态管理
│   │   ├── logger.py
│   │   └── data/               # 工具自带的参考数据
│   └── data/                   # 设定与运行时数据
│       ├── global_setting.json # 世界观/人物/势力
│       ├── story_design.json   # 章节大纲
│       ├── meta_novel_design.json
│       └── logs/               # （被 .gitignore 忽略）
├── kg_data/                    # 知识图谱原始数据
├── files/                      # Aider 编辑器记忆（被忽略）
├── requirements.txt            # Python 依赖
├── .env.example                # 环境变量模板
├── .gitignore
└── .gitattributes
```

## 快速开始

### 准备环境

```bash
# 建议 Python 3.11+
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

### 配置 API Key

复制 `.env.example` 为 `.env`，填入你的 LLM 服务密钥：

```env
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxx
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
KIMI_API_KEY=sk-xxxxxxxxxxxx
KIMI_BASE_URL=https://api.moonshot.cn/v1
DEFAULT_MODEL=deepseek-v4-pro
```

> [!WARNING]
> `.env` 已加入 `.gitignore`，**永远不会被提交到仓库**。

### 运行

```bash
# 3 章演示版本，使用内置默认设定
python -m src.main --reset

# 5 章 + 自定义角色
python -m src.main --protagonist "何雨柱" --mentor "易中海" \
  --antagonist "许大茂" --heroine "冉秋叶" --climax "崛起" \
  --chapters 5 --reset

# 加载外部设定文件
python -m src.main --design "my_story.json" --chapters 10 --output "./output"

# 让 AI 补全缺失的设定
python -m src.main --design "my_story_partial.json" --auto-complete --chapters 10

# 控制每章字数
python -m src.main --words 5000 --chapters 3 --reset
```

## Agent 流水线

```
            ┌──────────────────────┐
            │   Orchestrator   │  ← 主控
            └──────┬──────────────┘
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
   ┌────────┐ ┌────────┐ ┌────────┐
   │ Writer │→ │ Scribe │→ │ Editor │ → Reader 反馈 → 下一轮
   └────────┘ └────────┘ └────────┘
     草稿       摘要/伏笔    润色/校对
```

每章生成后，`Orchestrator` 会把章节摘要、人物状态、伏笔、读者评分
写入 `src/data/` 下的 JSON 文件，供下一章上下文使用。

## 自定义与扩展

| 想做什么 | 看这里 |
|---|---|
| 改章节大纲 | `src/data/story_design.json` |
| 改世界观/人物 | `src/data/global_setting.json` |
| 替换 LLM | `src/utils/llm_client.py` |
| 新增 Agent | 继承 `src/agents/base_agent.py`，加入 `orchestrator` 流水线 |
| 改提示词 | 各 Agent 文件顶部的 `SYSTEM_PROMPT` 变量 |

## 依赖

```
openai>=1.0.0
python-dotenv>=1.0.0
```

> 完整环境（PyQt6、numpy 等）只用于本地开发调试，**未纳入版本控制**。
> 部署/克隆后请按 `requirements.txt` 自行安装。

## 许可证

MIT
