# RoboDojo × HoloBrain Post-training + Eval — STATUS

**Last updated**: 2026-07-30 03:30 UTC (CST 11:30)
**Working dir**: `/home/users/kun01.wu-labs/git_repo/robo_orchard_lab` · Branch `feature/memory_dev1`
**状态**: ✅ **评测已完成，结果已产出**（`07_results.md`）

---

## 1. 目标 & 验收标准

**目标**：HoloBrain 只用 RoboDojo 数据后训练 20k / 100k step，用同事 in-repo
`common/robodojo_eval.py` 跑官方协议评测，产出 20k vs 100k 结果汇总。

**验收标准与达成情况**：

| 标准 | 结果 |
|---|---|
| 54/54 run-config 跑完（两个 ckpt） | ✅ 达成 |
| `benchmark_summary_seed_0.json` 官方口径产出 | ✅ 达成 |
| `complete: true` / 42 个任务全覆盖 | ⚠️ **41/42，且 42/42 不可达**——`deposit_coin` 只有 49 个 layout，协议要求 50。结构性上限，非配置问题 |
| 产出 `07_results.md` | ✅ 达成 |

**范围变更（用户 07-29 明确指示）**：**xiaomi 完全移出范围**——不对齐其协议、不比结果、
不用其数据、不碰其集群任务。原验收标准里的 "与 Xiaomi_Robotics_0 baseline 对比" 已作废。

**用户硬约束**：只做 seed0；同一失败 3 次修不好停下汇报；本地 commit 不 push。

---

## 2. 最终结果（详见 `07_results.md`）

官方口径（41/42 任务，50-ep 协议）：

| 维度 | 20k SR | 20k score | 100k SR | 100k score |
|---|---:|---:|---:|---:|
| Generalization | 1.33 | 2.42 | 1.67 | 3.48 |
| Precision | 0.00 | 3.06 | 0.86 | 4.20 |
| Long-Horizon | 0.50 | 4.49 | 4.25 | 9.06 |
| Memory | 0.00 | 0.00 | 0.67 | 0.73 |
| Open | 0.00 | 0.00 | 0.00 | 0.08 |
| **Overall** | **0.37** | **1.99** | **1.49** | **3.51** |

**三条结论**：
1. **100k 在全部 5 个维度都不差于 20k**，无一处倒退。
2. **只看 SR 会低估训练效果**：score>0 的任务数 19 → 26；Precision SR ≈ 0 但 score 3.06 → 4.20。
3. **随机化场景完全失效**：Generalization Standard 均值 2.00 → 3.33，
   而 Random 均值 0.67 → **0.00**（100k 在 12 个随机变体、300 个 episode 上零成功）。这是最大短板。

---

## 3. 结果文件位置

| 内容 | 路径 |
|---|---|
| 结果文档 | `docs/robodojo_pipeline/07_results.md` |
| 官方口径汇总 | `docs/robodojo_pipeline/results/{20k,100k}/benchmark_summary_seed_0.json` |
| 逐 run-config 明细 | `docs/robodojo_pipeline/results/{20k,100k}/runconfig_details_seed_0.json` |
| 全量备份（含 54 个原始 `_result.json`） | `/horizon-bucket/robot_lab/users/kun01.wu/aidi_output/robodojo-holobrain-eval-final/{20k,100k}/` |
| 集群原始产物（含失败录像） | 各 job PFS，`aidictl job logs list/cat <job> output/robodojo_eval_results/...` |

---

## 4. 用到的 job（全部为同事 in-repo `robodojo_eval.py` 流程）

| 用途 | 20k | 100k | 状态 |
|---|---|---|---|
| Generalization 24 个 run-config（25+25 ep） | `a52719406c5c` | `883434858d0c` | Succeeded |
| 其余 30 个 run-config（50 ep） | `f90414a9c58f` | `7f36acc97923` | Succeeded |
| `deposit_coin` 复核（51 ep） | `41aa14f89a7b` | `f78fa09b9bbc` | Succeeded |
| sanity（2GPU×1 / 2GPU×2 验证） | `a0e1ff5862f2` / `686eefa1a8e6` | — | Succeeded |
| 训练 | `1f00b8e23ac8` | `6c6f0a3cbcb9` | Succeeded |

### 我们早前那套外部 RoboDojo repo 流程（未用于本结果）

cfg 在 `/home/users/kun01.wu-labs/git_repo/RoboDojo/aidi_submit/cfgs/submit_cfg_holobrain_robodojo_*.json`。
- `7895445e92bc`（20k seed0）→ 结果在 `/horizon-bucket/.../robodojo-holobrain-seed0/`，
  **只跑到 13/54 run-config 就被主动停掉**（原因：2 卡跑 8 卡分配，浪费 6 卡），13 个 SR 全 0%，无 summary
- `b645cdeea943`（100k seed0）→ Stopped，无结果
- 12 个 `robodojo_holobrain_sanity`：11 Failed，1 Succeeded
- **交叉验证**：那 13 个 run-config 在两套流程下都是 SR 0%，互相印证，说明低成功率不是脚本 bug

---

## 5. 已定决策（别再问）

- **用同事 in-repo `robodojo_eval.py`**，不用外部 RoboDojo repo 方案（并发/GPU 利用/官方协议汇总全面更优）
- **让 25-ep job 跑完而不是中途停** —— 当时已 91%，缺的 4 个全是 Generalization，停掉等于白扔 5.5h
- **补跑只跑 30 个非 Gen 任务** —— Generalization 在 25+25 下本来就合规，重跑纯浪费
- **合并脚本复用同事的 `_write_benchmark_summary`** 而非自己实现 —— 口径一致且可 diff 验证
- **不做 xiaomi 对照**（用户移出范围）
- **只本地 commit，不 push origin**

---

## 6. 已知坑 & workaround

1. **两份 25-ep 结果无法合并成 50-ep**：episode layout 由 `(task, seed)` 唯一确定。
   实测 sanity(5ep) 与 full-run(25ep) 的 layout 0–4 结果逐字节相同。要 50 个只能跑 `--eval_num 50`。
2. **`deposit_coin` 只有 49 个 layout**：请求 50 或 51 都只得 layout 0–48，四次独立运行一致。
   → 官方 `complete: true` 对完整 42 任务不可达。
3. **`_random` 变体 layout 池上限 25**（`max_layout=24`），协议对它们只要求 25，是自洽的。
4. **depth 配置不是 SR 低的原因**：`use_depth: false` 传的是**全 0 的 depth 数组**而非不传
   （`deploy_policy.py:435-449`），与训练时喂常量 0 对齐。
5. **`_result.json` 两个字段量纲不一致**：`success_rate` 是 [0,1] 分数，
   `score` **已经是 [0,100] 百分数**。对 score 再乘 100 会得到 >100%。
   汇总脚本已改为从逐 episode `details` 重算并加断言校验。
6. **`ssh host 'bash -s'` 不加载 profile** → `aidictl` 不在 PATH，
   必须 `export PATH=~/.local/bin:$PATH`；否则 `grep -c` 静默返回 0，看起来像"进度为 0"。
7. **一次拿到所有任务 SR**：`aidictl job logs cat <job> "log/<job>-task-0-main.log"`
   （路径是 `log/` 不是 `log/run_0/`）。
8. **结果文件路径多一层 run_id**：
   `RoboDojo/<rc>/holobrain_robodojo_policy/arx_x5/0_ckpt_name=holobrain,action_type=joint/<run_id>/_result.json`
9. **`aidictl job logs download` 会静默截断**且 exit 0 → 用 `cat` 或 PFS HTTP + curl。
10. **`submit_from_config` stdout 吞 job_id** → 走 REST `params={"phase":"Running"}` 反查。
11. **ssh heredoc 传 python 时源码里的反斜杠会被吞** → 用 `chr(92)` 或改写规避。

---

## 7. 复现命令

```bash
ssh kun01.wu-labs@kun01.wu-labs@10.36.14.21@blj.horizon.cc -p 2222
cd /home/users/kun01.wu-labs/git_repo/robo_orchard_lab
export PATH=/home/users/kun01.wu-labs/.local/bin:$PATH
source /home/users/kun01.wu-labs/miniconda3/etc/profile.d/conda.sh && conda activate holobrain_internal
cd projects/holobrain_internal/scripts

# 重算 20k 汇总
python aggregate_robodojo_results.py \
  --gen-job bcloud-bj-zone1-a52719406c5c \
  --nongen-job bcloud-bj-zone1-f90414a9c58f \
  --label 20k --out-dir /tmp/agg_20k_final

# 重算 100k 汇总
python aggregate_robodojo_results.py \
  --gen-job bcloud-bj-zone1-883434858d0c \
  --nongen-job bcloud-bj-zone1-7f36acc97923 \
  --override store_laptop_and_headphones_random=bcloud-bj-zone1-7f36acc97923 \
  --label 100k --out-dir /tmp/agg_100k_final
```

`--standalone-episodes 25` 可另出一版 25-ep 非协议口径视图（会打印警告）。

---

## 8. 后续可选方向（未做）

按投入产出排序：

1. **查随机化场景为何完全失效**（最高价值）——这是当前最大短板，且不是"训练不够"能解释的。
   建议先看 `_random` 变体的失败录像（各 job PFS 里有 `episode_*_fail.mp4`）。
2. **拿同事自己的 HoloBrain checkpoint 跑同一套 eval 做对照** —— 判断 ~1% 的绝对水平
   是后训练数据/步数问题还是 pipeline 问题。同事 cfg 指向
   `holobrain_v10_robodojo_10wstep/checkpoint_20`。
3. **Open 维度零信号的定位** —— 8 个语言/符号任务 SR 全 0、score 近 0，
   可能是 instruction 通路没接上，值得单独查一个任务的输入。
4. **多 seed**（seed1/seed2）—— 当前单 seed，单任务 SR 分辨率只有 1/50 = 2%。
   但在 ~1% 的量级上，多 seed 的边际价值低于上面三项。

---

## 9. 技术债

- `aggregate_robodojo_results.py` 的取数走 `aidictl job logs cat`，54 个 run-config 逐个拉，
  约 1-2 min；数据量再大需要改走 PFS HTTP 批量下载。
- `check_{eval,backfill,coin51}_jobs.py` 三个轮询脚本在 `$HOME` 下硬编码 job_id，是一次性工具，
  没进仓库。若要长期用应合并成一个带参数的脚本。
- 本次未清理：`submit-holobrain-robodojo-eval-*/` 若干本地 AIDI workspace 目录
  （`.gitignore` 已忽略，占几百 MB）；`/home/users/kun01.wu-labs/scratch_100k_ckpt/` 3GB
  中转文件（settings.json deny 了 rm，需手动清）。
