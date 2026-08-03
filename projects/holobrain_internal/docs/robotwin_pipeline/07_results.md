# 07 — RoboTwin 2.0 评测结果

> ⚠️ **本页所有数字都来自 `checkpoint_11` = 训练 step 60000 的中间 ckpt。**
> 训练实际跑满了 step 100000（`checkpoint_19`），**终版权重从未在 RoboTwin 上评测过**。
> 不要把本页当作"这轮后训练的最终成绩"引用。原因见 [01_training.md](01_training.md) §5。

配套机器可读 JSON：`results/20260723/summary.json`、`results/20260724/summary.json`
（与 bucket 归档里的 `summary.json` 逐字节相同）。

---

## 1. 口径

| 项 | 值 | 来源 |
|---|---|---|
| Benchmark | RoboTwin 2.0 | `envs/` 含 v2.0 独有任务；`sapien==3.0.0b1 + mplib==0.2.1` |
| Task config | `demo_clean` | `--task_config`；domain randomization **全关** |
| Embodiment | `aloha-agilex`（双臂 + 双 D435） | `task_config/demo_clean.yml:embodiment` |
| 任务数 | **16** | submit cfg 的 `--task_names` |
| 每任务 trial | **50** | `--test_num 50` |
| Instruction type | `unseen` | `deploy_policy.yml:instruction_type` |
| Seed 起点 | `st_seed = 100000×(1+0) = 100000` | `deploy_policy.yml:seed=0`；`script/eval_policy.py:167` |
| 每次 policy 输出取前几步 | 32 | `holobrain_robotwin_policy/deploy_policy.py:156-157` |
| 成功判定 | RoboTwin 各 task 自带的 `check_success()` | `envs/<task>.py` |

**`mean` 的定义**：`robotwin_eval.py:153` 对 `results` 求**简单平均**（不按 trial 数加权）。
因为 16 个任务的 `test_num` 都是 50，所以这里与 trial 池化平均等价。

---

## 2. 两次完整评测

同一份权重、同一套参数，跑了两次（差别只有 docker 镜像）。

| # | Task | 20260723 | 20260724 | Δ (pp) |
|---|---|---:|---:|---:|
| 1 | handover_mic | 50/50 = **100.0%** | 49/50 = **98.0%** | −2 |
| 2 | adjust_bottle | 43/50 = **86.0%** | 43/50 = **86.0%** | 0 |
| 3 | dump_bin_bigbin | 36/50 = **72.0%** | 37/50 = **74.0%** | +2 |
| 4 | open_laptop | 35/50 = **70.0%** | 35/50 = **70.0%** | 0 |
| 5 | open_microwave | 29/50 = **58.0%** | 28/50 = **56.0%** | −2 |
| 6 | stack_bowls_three | 25/50 = **50.0%** | 27/50 = **54.0%** | +4 |
| 7 | beat_block_hammer | 25/50 = **50.0%** | 26/50 = **52.0%** | +2 |
| 8 | place_empty_cup | 22/50 = **44.0%** | 23/50 = **46.0%** | +2 |
| 9 | place_cans_plasticbox | 17/50 = **34.0%** | 23/50 = **46.0%** | **+12** |
| 10 | place_dual_shoes | 17/50 = **34.0%** | 14/50 = **28.0%** | −6 |
| 11 | rotate_qrcode | 15/50 = **30.0%** | 19/50 = **38.0%** | +8 |
| 12 | blocks_ranking_rgb | 15/50 = **30.0%** | 15/50 = **30.0%** | 0 |
| 13 | lift_pot | 4/50 = **8.0%** | 4/50 = **8.0%** | 0 |
| 14 | blocks_ranking_size | 3/50 = **6.0%** | 4/50 = **8.0%** | +2 |
| 15 | stack_blocks_three | 3/50 = **6.0%** | 2/50 = **4.0%** | −2 |
| 16 | move_pillbottle_pad | 2/50 = **4.0%** | 2/50 = **4.0%** | 0 |
| | **mean** | **42.625%** | **43.875%** | +1.25 |
| | **池化** | 341/800 | 351/800 | |

两次合计 **692 / 1600 = 43.25%**。

排序按 20260724 那次。两次的排序几乎一致，只有
`place_cans_plasticbox`（第 9 ↔ 第 6 档）和 `place_dual_shoes` 互换了相对位置。

---

## 3. 分层解读

| 档位 | 任务 | 共同点 |
|---|---|---|
| **高 (≥70%)** | handover_mic、adjust_bottle、dump_bin_bigbin、open_laptop | 单目标、抓取或推拉一次即完成，对末端位姿精度容忍度高 |
| **中 (28–58%)** | open_microwave、stack_bowls_three、beat_block_hammer、place_empty_cup、place_cans_plasticbox、rotate_qrcode、blocks_ranking_rgb、place_dual_shoes | 单目标但要求较准的落点，或两阶段（抓→放） |
| **低 (≤8%)** | lift_pot、blocks_ranking_size、stack_blocks_three、move_pillbottle_pad | **多物体精确摆放 / 双臂协同**：要么需要两臂同步（lift_pot），要么需要连续 3 次不能失误的精确堆叠或排序 |

低档 4 个任务的失败是**结构性的**而非抖动 —— 两次跑分别是
2–4/50 和 2–4/50，一致地贴地板。这类任务需要的是**长程无累积误差**，
一次 32 步的 action chunk 里任何一步偏了后面就救不回来。

---

## 4. 三条读数注意事项

### 4.1 噪声量级：±5–10 pp 不能当真实差异

同一权重跑两次，逐任务差 0–12 pp。这不是 bug，是**统计噪声 + 环境不确定性**叠加的结果。

量级估算：50 次伯努利试验、`p≈0.4` 时，单次估计的标准误是
`sqrt(0.4×0.6/50) ≈ 6.9 pp`；两次独立测量之**差**的标准误约 `sqrt(2)×6.9 ≈ 9.8 pp`。
所以 95% 区间约 **±19 pp** —— 观察到的最大差异（`place_cans_plasticbox` +12 pp）
**完全落在噪声内**。

→ **想比较两个 checkpoint，50 trial 不够。** 要么加大 `--test_num`，
要么固定多个 seed 各跑一遍取合并，否则得到的是抖动不是结论。

### 4.2 seed 是过滤出来的，而且两次不完全相同

`script/eval_policy.py:225-277` 的 `while succ_seed < test_num`：

1. 用 expert（`TASK_ENV.play_once()`）试跑当前 seed；不合法（`UnStableError` 或异常）则
   `now_seed += 1` **跳过，不计数**；
2. 合法 → 让 policy 跑一次；无论成败 `succ_seed += 1, now_seed += 1`。

所以实际执行的 50 个 seed **不是连号 100000..100049**，而是过滤后的 50 个，
末位 seed 落在 100049–100145 不等（`lift_pot` 要试到 100145 才凑够 50 个可行 seed，
说明该任务本身可行 seed 就稀疏）。

**关键**：两次跑的末位 seed 并不一样 —— 如 `place_dual_shoes` 是 100070 vs 100081、
`rotate_qrcode` 是 100064 vs 100067。这说明 **expert 过滤本身不是确定性的**，
两次面对的 episode 集合并不完全相同。这就是 §4.1 那些差异的直接来源。

> **勘误**：`../eval_robotwin_ckpt11.md` §4 曾写"在同一 (task, task_config, seed, test_num) 组合下，
> 不同 checkpoint 面对的是完全相同的 seed，可以直接横向对比"。**该结论与本次实测不符**，
> 已在那份文档里标注更正。

### 4.3 失败的任务会被静默丢出统计

`robotwin_eval.py:71-89`：只有子进程 `returncode == 0` 才把结果写进 `results`；
非零时只打一行 `Fail to eval task[...]`，该任务**不进 `results`**，
而 `mean = sum(results.values()) / len(results)`（`:153`）照算。

→ **`mean` 只有配合 `num_tasks` 才可信**。若 `num_tasks < 16`，说明有任务整个挂掉，
此时的 `mean` 是幸存者偏差，不能与别的跑对比。

本页两次跑 `num_tasks` 都是 **16**，均值是干净的（已用逐任务三处交叉核对确认）。

---

## 5. 原始数据在哪

```
/horizon-bucket/robot_lab/users/kun01.wu/robo_orchard_lab/eval_results/
    robotwin-holobrain-ckpt11-20260723/     job bcloud-bj-zone1-51192e413238
    robotwin-holobrain-ckpt11-20260724/     job bcloud-bj-zone1-a74cf470f80b
        README.md                           这次跑的完整参数与取舍说明
        summary.json                        16 任务汇总
        main.log                            job 主 stdout
        <task>/log.txt                      eval_policy.py 全量 stdout（逐 episode）
        <task>/_result.txt                  该 task 成功率
```

每份 35 个文件 / 139 MB，2026-07-31 从 AIDI 取回，`tree_verify.py --mode equal` 校验通过，
md5 清单在 `/jfs-public/users/kun01.wu/robo_orchard_lab/manifests/archive_eval_results_*.md5`。

⚠️ **归档不含 episode 视频**（每任务 50 段、两次共 800 段）。视频只在 AIDI 侧、有留存期，
到期不可再取 —— 这是明确的取舍，不是归档不完整。详见各归档目录的 `README.md`。

---

## 6. 可以往下做什么

1. **补测 step-100000 的终版权重** —— 目前最大的空白。终版 `checkpoint_19` 还在
   `/jfs-public/users/kun01.wu/robo_orchard_lab/workspace/checkpoints/checkpoint_19/`。
   按 §4.1，要判断 60k→100k 是涨是跌，**必须跑两遍**（或加大 test_num），单次不足以下结论。
2. **低档 4 个任务做定向分析** —— 它们贴地板且稳定，是最有信息量的失败样本。
   `<task>/log.txt` 里有逐 episode 记录可查失败模式。
3. **提高统计功效** —— 后续做 ckpt 对比时把 `--test_num` 提到 100+，
   或用多个 `seed` 值（`deploy_policy.yml:seed` → `st_seed = 100000×(1+seed)`）跑多轮合并。
