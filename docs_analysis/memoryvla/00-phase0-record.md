# 00 — Phase 0 就位记录

日期：2026-08-03（`date +%Y-%m-%d`）

## 分支与基点

| 项 | 值 |
|---|---|
| 宿主 | `~/git_repo/robo_orchard_lab` （cite: 实测） |
| 基点 commit | `3ce31c0c`（`feature/memory_dev1` 的 tip，tag `memory_dev1-stage1-20260803`） （cite: 实测） |
| 本次分支 | `port/memoryvla` （cite: 实测） |
| 宿主 LICENSE | Apache-2.0（`LICENSE:1`，Horizon Robotics） （cite: 实测） |

## 落点（`source ./robo_orchard_lab_env.sh` → `SELFCHECK OK`） （cite: 实测）

| 层 | 变量 / 路径 |
|---|---|
| LOCAL | `~/git_repo/robo_orchard_lab` （cite: 实测） |
| JFS | `ROL_JFS=/jfs-public/users/kun01.wu/robo_orchard_lab`（`robo_orchard_lab_env.sh:16`） （cite: robo_orchard_lab_env.sh:16） |
| BUCKET | `ROL_BUCKET=/horizon-bucket/robot_lab/users/kun01.wu`（`:17`） （cite: 实测） |
| 本次工作目录 | `$ROL_JFS/port/memoryvla/{ref,tools,logs}` （cite: 实测） |

> 计划里写的 `$PROJ_SCRATCH` 在本项目实际叫 **`$ROL_JFS`**。
> `env_selfcheck` 顺带报了一条：仓根没有 `.cache_manifest`。本次移植是 E0、不新增任何下载，
> 不碰缓存指向，因此不建；若后续要动缓存变量再补。

## A（源方法）状态

| 项 | 值 |
|---|---|
| 路径 | `~/git_repo/MemoryVLA` （cite: 实测） |
| commit | `0eef5c39f15455c46c137c8dd5d1cebc801b4d25` （cite: 实测） |
| remote | `https://github.com/shihao1895/MemoryVLA.git` （cite: 实测） |
| LICENSE | MIT（`pyproject.toml:21` 的 classifier）。⚠️ `pyproject.toml:15` 写 `license={file="LICENSE"}` 但**仓库里没有 LICENSE 文件** （cite: pyproject.toml:21） |
| 环境 | `memvla_cu128`：py 3.10.20 / torch **2.8.0+cu128** / transformers **4.40.1**；`prismatic`·`vla`·`action_model`·`MemoryVLA` 全部 import 成功，`cuda True`，8 卡 （cite: 实测） |

**合规结论**：MIT → Apache-2.0 宿主，兼容，可移植。搬运处逐段留出处注释
`# [port:memoryvla] from MemoryVLA@0eef5c3 vla/memory_vla.py:L<a>-L<b>`。

### A repo 只读判据（**对协议的一处显式偏离**）

协议要求全程 `git -C <A> status --porcelain` 为空。**A 的工作区在本次开始前就是脏的**
（20 条：13 modified + 7 untracked，是本人此前 repro 工作留下的，与本次无关）。
清理它既超范围又会破坏你的既有工作，因此判据改为「**未被我改动**」：

| 指纹 | Phase 0 记录值 |
|---|---|
| `git rev-parse HEAD` | `0eef5c39f15455c46c137c8dd5d1cebc801b4d25` （cite: 实测 git，2026-08-03） |
| `git status --porcelain \| md5sum` | `9815d522644f15ab4edd56e5b33d1d03` （cite: 实测） |
| `git status --porcelain \| sort \| md5sum` | `825713088ade907d429371ab7808a013` （cite: 实测） |
| `git status --porcelain \| wc -l` | `20` （cite: 实测 git，2026-08-03） |
| `git stash list \| wc -l` | `0` （cite: 实测 git，2026-08-03） |

Phase 6 复核这五项必须完全一致。

## 环境档位：**E0**

宿主主环境 `holobrain_internal`（py3.11.15 / torch 2.8.0+cu128 / transformers 5.10.2）
**直接跑得动要移植的那部分代码**。实测：把 `vla/memory_vla.py` 第 30–358 行 （cite: 实测）
（纯 torch 的那一段，不含 `prismatic` 依赖）exec 进 `holobrain_internal`，
`CogMemBank.process_batch` 输出 `(4, 16, 64)`，mean `0.005819`。

- **不建 `.venv_memoryvla`**，**宿主主环境零改动**，差异包清单为空。
- 值得记一笔：A 的**完整栈**是 E2（transformers 4.40.1 vs 宿主 5.10.2，跨大版本）。
  只有「两个 bank 是纯 `torch.nn`」这一事实把本次拉回 E0。换个组件就未必了。
- 参考数值在 `memvla_cu128` 生成：它的 torch 与宿主**完全同版本**（2.8.0+cu128），
  C 档误差不会被 torch 版本差污染。（另一个 env `memoryvla` 是 torch 2.2.0+cu121，不用。）

## 冒烟数据：RoboDojo Memory 维度 6 任务

选它的理由：RoboDojo 五个维度里 HoloBrain 的 Memory 维度基本全零
（20k→100k：0.00%→0.67%，6 个任务只有 `match_and_pick_from_conveyor` 在 100k 拿到 4%，
见 `projects/holobrain_internal/docs/robodojo_pipeline/07_results.md:197-206` 与 §6.3）。 （cite: projects/holobrain_internal/docs/robodojo_pipeline/07_results.md:197-206）
这正是 MemoryVLA 针对的能力。

路径：`/horizon-bucket/robot_lab2/datasets/all_data/robodojo/lmdb/<task>`
（`depth/image/index/meta` 四件套；实测规模由 `$ROL_JFS/port/memoryvla/tools/probe_memory_tasks.py` 得出） （cite: 实测）

| task | frames | episodes | 最短 | 中位 | 最长 |
|---|---:|---:|---:|---:|---:|
| cover_blocks | 54396 | 100 | 537 | 543 | 559 （cite: 实测 tools/probe_memory_tasks.py） |
| match_and_pick_from_conveyor | 43366 | 100 | 254 | 445 | 598 （cite: 实测 tools/probe_memory_tasks.py） |
| swap_blocks | 43609 | 100 | 424 | 437 | 446 （cite: 实测 tools/probe_memory_tasks.py） |
| swap_T | 27685 | 100 | 271 | 276 | 300 （cite: 实测 tools/probe_memory_tasks.py） |
| press_by_number | 38921 | 100 | 261 | 387 | 507 （cite: 实测 tools/probe_memory_tasks.py） |
| imitate_sorting_sequence | 120998 | 100 | 1054 | 1203 | 1374 （cite: 实测 tools/probe_memory_tasks.py） |
| **合计** | **328975** | **600** | | | （cite: 实测 tools/probe_memory_tasks.py） |

**对冒烟的意义**：episode 很长（最短的 `swap_T` 中位也有 276 帧）。
`stream` 模式下 batch=16 时，跨过一条 `swap_T` 的 episode 边界需要 ⌈276/16⌉ ≈ **18 step**；
最长的 `imitate_sorting_sequence` 要 ~76 step。**冒烟必须跑到跨过至少一个 episode 边界**，
否则 `clear_episode` 那条路径永远不执行（清理逻辑清的是上一条，N=1 永远不走）。
→ 冒烟优先用 `swap_T`（最短），能最快跨界。

## git-lfs：协议假设在本仓库不成立（记录以免下次重复排查）

协议要求「切分支前先 `conda activate`，否则 git-lfs 不在 PATH，LFS 资产会写成 131 B 指针」。
实测本机 **`holobrain_internal` 里和 base PATH 里都没有 `git-lfs`**（`git lfs version` 报 （cite: 实测）
`'lfs' is not a git command`）。但本仓库**根本没用 LFS**：

- 没有 `.gitattributes`
- `git grep -l "git-lfs.github.com/spec" HEAD` 无结果（无 LFS 指针文件）
- `.git/config` 里虽有 `filter.lfs.*`，但没有 `.gitattributes` 规则去触发它

结论：本仓库切分支/检出**不受 git-lfs 缺失影响**。（第一次尝试建分支时命令静默失败，
原因是 `which git-lfs` 返回非零把 `&&` 链掐断了，不是权限或 git 问题。）

## Phase 0 检查表

- [x] `env_selfcheck` → `SELFCHECK OK` （cite: 实测）
- [x] 分支 `port/memoryvla` 建于 `3ce31c0c`
- [x] 读宿主 `docs/` 与 `projects/holobrain_internal/docs/`
- [x] `docs_analysis/` 原本不存在 → 本次创建
- [x] A 环境实测可运行（未做任何安装/升级） （cite: 实测）
- [x] A 的 commit / remote / LICENSE 记录在案
- [x] Memory 6 任务规模实测 （cite: 实测）
