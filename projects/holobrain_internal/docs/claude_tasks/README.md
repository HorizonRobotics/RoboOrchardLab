# Claude Session Handover 交接文档目录

本目录存放**跨 session 的完整状态交接文档**。每一份对应一个 session 结束时的快照，供下一位 Claude 无缝接手。

## 命名规则

`YYYY-MM-DD_<主题>_<状态>.md`
- 日期在前，便于按时间排序（`ls -1` 就是时间序）
- 主题描述：如 `robotwin_eval`、`train_ckpt_backup` 等
- 状态标签：
  - `env_ready` — 环境就绪
  - `blocked_<原因>` — 阻塞在某一点
  - `completed` — 完成
  - `resolved` — 已被后续 session 完全解决（可忽略）

## 阅读顺序

下次接手时：
1. `ls -1 *.md`，找**日期最新**且文件名里**没有 `_resolved`** 的那一份；
2. 从头读一遍（每份文档开头都有 TL;DR / §0 概览段）；
3. 再回本 `README.md` 看**同期活着**的其他交接（可能是并行工作流）。

## 当前文档

| 文件 | 主题 | 状态 |
|---|---|---|
| `2026-07-22_robotwin_eval_env_ready_blocked_curobo.md` | RoboTwin 2.0 评估 checkpoint_11 | 🟠 阻塞在 curobo；`robotwin_holobrain_eval` env 已装完、模型能 to cuda |

## 与相邻文档的关系

- **plan 文件**：`~/.claude/plans/*.md` —— 每个 session 的详细实现计划；比 handover 更长。当前有效的：`breezy-floating-star.md`（本 session）。
- **memory 索引**：`~/.claude/projects/-home-users-kun01-wu-labs-git-repo-robo-orchard-lab/memory/MEMORY.md` —— 单条事实。与 handover 互补：handover 讲**故事**，memory 讲**事实**。
- **项目 tutorial**：`../{01_..11_}*.md` —— 面向新同学的项目全流程 tutorial（12 篇，2026-07-22 已完成）。
