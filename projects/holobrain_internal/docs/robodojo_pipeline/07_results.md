# RoboDojo 评测结果：HoloBrain RoboDojo-only 后训练 20k vs 100k

## 1. 结论摘要

- **100k 在全部 5 个维度上都不差于 20k**，SR 和 score 双指标一致；没有任何维度倒退。
  overall SR **0.37% → 1.49%**，overall score **1.99% → 3.51%**。
- **绝对水平很低**：42 个任务里 SR > 0 的只有 20k 4 个 / 100k 9 个。多数任务整体成功率为 0。
- **只看 SR 会低估训练效果**：score（部分完成度）在 SR 恒为 0 的维度上依然明显上升，
  例如 Precision 的 SR 两边都接近 0，但 score 3.06% → 4.20%。score > 0 的任务数
  **19 → 26**，说明模型学到了任务的前半段但跨不过完成线。
- **随机化场景几乎完全失效**：Generalization 的 Standard 均值 2.00% → 3.33%，
  而 Random 均值 0.67% → **0.00%**。100k 在所有 12 个随机化变体上一次都没成功。
- 最强的单任务是 `put_bottles_into_dustbin`（SR 4% → 30%，score 23.9% → 45.1%）和
  `stack_bowls`（SR 10% → 12%）。
- 覆盖度：54/54 run-config 全部跑完；官方口径计分 **41/42** 任务。缺的 `deposit_coin`
  是**benchmark 结构性上限**（该任务只有 49 个 layout，协议要求 50），补不齐，详见 §7。

---

## 2. 评测配置

| 项 | 值 |
|---|---|
| 策略 | `holobrain_robodojo_policy`（in-repo，`common/robodojo_eval.py`） |
| 20k checkpoint | `holobrain_robodojo_posttrain_v9/checkpoint_20000` |
| 100k checkpoint | `holobrain_robodojo_posttrain_v9_100k/checkpoint_100000` |
| 训练数据 | 仅 RoboDojo |
| 环境 / 本体 | `arx_x5`，`action_type=joint` |
| seed | 0（**单 seed**） |
| 每任务 episode | 50（官方协议） |
| 集群 | `project-5090-robot-lab-bcloud-bj`，8 GPU × 2 process = 16 worker |

### 协议要点

- **42 个计分任务**，分 5 个维度；**54 个 run-config**：Generalization 的 12 个任务各有
  标准 `X` 与随机 `X_random` 两个变体（24 个），其余 30 个任务各一个。
- 每个任务需 **50 个 episode**：Generalization 取 `X` 25 个 + `X_random` 25 个；
  其余 30 个任务从单一 run-config 取 50 个。
- 两项指标：
  - **SR（success_rate）** —— 整体成功率，二值判定。
  - **score** —— 部分完成度，按任务内子目标给分。在低成功率区间比 SR 敏感得多。

### 数据来源

| 部分 | job |
|---|---|
| Generalization 24 个 run-config（25+25） | 20k `bcloud-bj-zone1-a52719406c5c` / 100k `bcloud-bj-zone1-883434858d0c` |
| 其余 30 个 run-config（50 ep） | 20k `bcloud-bj-zone1-f90414a9c58f` / 100k `bcloud-bj-zone1-7f36acc97923` |
| `deposit_coin` 复核（51 ep，确认 layout 上限） | 20k `bcloud-bj-zone1-41aa14f89a7b` / 100k `bcloud-bj-zone1-f78fa09b9bbc` |

> 100k 的 `store_laptop_and_headphones_random` 在 25-ep 那轮只拿到 24 个 episode，
> 已改从 50-ep job 取数（`--override`），因此 100k 的 Generalization 是完整 12/12。

### 结果文件位置

| 内容 | 路径 |
|---|---|
| 官方口径汇总（本文数字的来源） | `docs/robodojo_pipeline/results/{20k,100k}/benchmark_summary_seed_0.json` |
| 逐 run-config 明细（SR/score/episode/来源 job） | `docs/robodojo_pipeline/results/{20k,100k}/runconfig_details_seed_0.json` |
| 全量备份（含 54 个原始 `_result.json`） | `/horizon-bucket/robot_lab/users/kun01.wu/robo_orchard_lab/eval_results/robodojo-holobrain-eval-final/{20k,100k}/` |
| 集群原始产物（含失败录像 mp4） | 各 job PFS，`aidictl job logs list/cat <job_id> output/robodojo_eval_results/...` |

汇总由 `projects/holobrain_internal/scripts/aggregate_robodojo_results.py` 合并两个 job 的
per-run-config `_result.json` 后，调用 `robodojo_eval._write_benchmark_summary` 产出——
统计口径与 in-repo evaluator 完全一致（已用 sanity job 做过 `diff` 校验）。

---

## 3. 5 维度汇总

单位均为百分比。Δ = 100k − 20k。

| 维度 | 任务数 | 20k SR | 20k score | 100k SR | 100k score | Δ SR | Δ score |
|---|---:|---:|---:|---:|---:|---:|---:|
| Generalization | 12/12 | 1.33 | 2.42 | **1.67** | **3.48** | +0.33 | +1.06 |
| Precision | 7/8 † | 0.00 | 3.06 | **0.86** | **4.20** | +0.86 | +1.14 |
| Long-Horizon | 8/8 | 0.50 | 4.49 | **4.25** | **9.06** | +3.75 | +4.58 |
| Memory | 6/6 | 0.00 | 0.00 | **0.67** | **0.73** | +0.67 | +0.73 |
| Open | 8/8 | 0.00 | 0.00 | 0.00 | 0.08 | 0.00 | +0.08 |
| **Overall** | **41/42** | **0.37** | **1.99** | **1.49** | **3.51** | **+1.12** | **+1.52** |

† `deposit_coin` 只有 49 个可用 layout，协议要求 50，因此被判为 incomplete 而未计入。
这不是本次配置问题，**41/42 是该 benchmark 的结构性上限**，详见 §7。

> ### ⚠️ 哪些差异可以解读，哪些不可以
>
> 单 seed、每任务 50 episode，**单任务 SR 的最小分辨率是 1/50 = 2%**。因此：
>
> - **不要解读**维度级的小差异。例如 Generalization「1.33% → 1.67%」相差 0.33%，
>   远小于单次成功/失败带来的抖动，**不能据此说 Generalization 变好了**。
>   同理 Memory「0.00% → 0.67%」实际只是 6 个任务里某一个多成功了 2 个 episode。
> - **可以解读**的是幅度明显超过分辨率的单任务变化，例如
>   `put_bottles_into_dustbin` 2/50 → 15/50（SR 4% → 30%）、
>   `organize_table` score 5.5 → 13.5。
> - **可以解读**的是方向的一致性：5 个维度、SR 与 score 两个指标**全部同向**，
>   这个整体模式比任何单个数字都更可信。
> - **可以解读**的是 Random 半边的归零（12 个任务 × 25 ep = 300 个 episode 零成功），
>   样本量足够大。
>
> 要让维度级小差异变得可解读，需要多 seed（seed1/seed2）或提高每任务 episode 数。

### 补充口径：把 `deposit_coin` 按其 49 个 episode 纳入（42/42）

上表是 `robodojo_eval.py` 的官方输出。由于 `deposit_coin` 永远拿不到第 50 个 episode，
下表补一个"用它实际的 49 个 episode 纳入统计"的版本，便于看全部 42 个任务：

| 维度 | 任务数 | 20k SR | 20k score | 100k SR | 100k score |
|---|---:|---:|---:|---:|---:|
| Generalization | 12/12 | 1.33 | 2.42 | 1.67 | 3.48 |
| Precision | 8/8 | 0.00 | 2.83 | 0.75 | 3.78 |
| Long-Horizon | 8/8 | 0.50 | 4.49 | 4.25 | 9.06 |
| Memory | 6/6 | 0.00 | 0.00 | 0.67 | 0.73 |
| Open | 8/8 | 0.00 | 0.00 | 0.00 | 0.08 |
| **Overall** | **42/42** | **0.37** | **1.95** | **1.47** | **3.42** |

两种口径的差异都在 0.1 个百分点以内，**结论完全一致**。

---

## 4. Generalization：Standard vs Random

每个 Generalization 任务的 50 个 episode 由标准场景 25 个 + 随机化场景 25 个组成。
下表拆开看两半的 SR。

| task | 20k Standard | 20k Random | 100k Standard | 100k Random |
|---|---:|---:|---:|---:|
| stack_bowls | **20.0** | 0.0 | **24.0** | 0.0 |
| push_T | 0.0 | 0.0 | 0.0 | 0.0 |
| pack_objects_into_box | 0.0 | 0.0 | 0.0 | 0.0 |
| fold_clothes | 0.0 | 0.0 | **4.0** | 0.0 |
| hang_mugs | 0.0 | 0.0 | 0.0 | 0.0 |
| sweep_blocks | 0.0 | 0.0 | 0.0 | 0.0 |
| pour_liquid_into_cup | 0.0 | **8.0** | **8.0** | 0.0 |
| make_toast | 0.0 | 0.0 | 0.0 | 0.0 |
| arrange_largest_number | 0.0 | 0.0 | 0.0 | 0.0 |
| sort_nesting_dolls_by_size | 0.0 | 0.0 | 0.0 | 0.0 |
| store_laptop_and_headphones | **4.0** | 0.0 | 0.0 | 0.0 |
| stack_blocks | 0.0 | 0.0 | **4.0** | 0.0 |
| **均值** | **2.00** | **0.67** | **3.33** | **0.00** |
| **Δ（Random − Standard）** | **−1.33** | | **−3.33** | |

**读法**：训练让标准场景变好（2.00 → 3.33），但随机化场景反而归零。
100k 在 12 个随机化变体上 300 个 episode 里一次都没成功。这说明后训练学到的是
偏特定布局的策略，对布局扰动没有泛化能力——是当前最突出的短板。

---

## 5. 逐任务明细

SR / score 均为百分比。加粗 = SR > 0。

### Generalization（每任务 25+25 episode）

| task | 20k SR | 20k score | 100k SR | 100k score |
|---|---:|---:|---:|---:|
| stack_bowls | **10.0** | 14.5 | **12.0** | 17.4 |
| push_T | 0.0 | 0.0 | 0.0 | 0.0 |
| pack_objects_into_box | 0.0 | 0.8 | 0.0 | 3.1 |
| fold_clothes | 0.0 | 0.4 | **2.0** | 2.0 |
| hang_mugs | 0.0 | 1.8 | 0.0 | 2.1 |
| sweep_blocks | 0.0 | 0.0 | 0.0 | 0.0 |
| pour_liquid_into_cup | **4.0** | 4.0 | **4.0** | 4.0 |
| make_toast | 0.0 | 1.5 | 0.0 | 1.0 |
| arrange_largest_number | 0.0 | 0.5 | 0.0 | 1.1 |
| sort_nesting_dolls_by_size | 0.0 | 0.0 | 0.0 | 0.0 |
| store_laptop_and_headphones | **2.0** | 4.0 | 0.0 | 6.0 |
| stack_blocks | 0.0 | 1.5 | **2.0** | 5.0 |

### Precision（每任务 50 episode）

| task | 20k SR | 20k score | 100k SR | 100k score |
|---|---:|---:|---:|---:|
| fasten_screws | 0.0 | 0.8 | 0.0 | 0.4 |
| plug_in_charger | 0.0 | 0.0 | **2.0** | 2.0 |
| insert_tubes | 0.0 | 8.0 | 0.0 | 7.2 |
| pour_balls_into_vase | 0.0 | 0.0 | **4.0** | 4.0 |
| play_Xylophone | 0.0 | 0.0 | 0.0 | 0.0 |
| deposit_coin † | 0.0 | 1.2 | 0.0 | 0.8 |
| insert_key | 0.0 | 9.0 | 0.0 | 12.6 |
| build_tower | 0.0 | 3.6 | 0.0 | 3.2 |

† 49/50 episode，未计入维度均值。

### Long-Horizon（每任务 50 episode）

| task | 20k SR | 20k score | 100k SR | 100k score |
|---|---:|---:|---:|---:|
| put_bottles_into_dustbin | **4.0** | 23.9 | **30.0** | 45.1 |
| fill_pen_holder | 0.0 | 2.1 | 0.0 | 2.8 |
| classify_objects | 0.0 | 3.0 | 0.0 | 4.3 |
| play_tic_tac_toe | 0.0 | 0.4 | 0.0 | 2.2 |
| fill_egg_holder | 0.0 | 1.0 | 0.0 | 0.6 |
| organize_table | 0.0 | 5.5 | 0.0 | 13.5 |
| make_kong | 0.0 | 0.0 | **4.0** | 4.0 |
| play_stacking_toy | 0.0 | 0.0 | 0.0 | 0.0 |

### Memory（每任务 50 episode）

| task | 20k SR | 20k score | 100k SR | 100k score |
|---|---:|---:|---:|---:|
| cover_blocks | 0.0 | 0.0 | 0.0 | 0.4 |
| match_and_pick_from_conveyor | 0.0 | 0.0 | **4.0** | 4.0 |
| swap_blocks | 0.0 | 0.0 | 0.0 | 0.0 |
| swap_T | 0.0 | 0.0 | 0.0 | 0.0 |
| press_by_number | 0.0 | 0.0 | 0.0 | 0.0 |
| imitate_sorting_sequence | 0.0 | 0.0 | 0.0 | 0.0 |

### Open（每任务 50 episode）

| task | 20k SR | 20k score | 100k SR | 100k score |
|---|---:|---:|---:|---:|
| align_blocks | 0.0 | 0.0 | 0.0 | 0.0 |
| general_pickup | 0.0 | 0.0 | 0.0 | 0.0 |
| stack_blocks_by_language | 0.0 | 0.0 | 0.0 | 0.4 |
| solve_equation | 0.0 | 0.0 | 0.0 | 0.0 |
| classify_objects_by_language | 0.0 | 0.0 | 0.0 | 0.2 |
| pick_from_conveyor_by_image | 0.0 | 0.0 | 0.0 | 0.0 |
| store_tools_in_toolbox | 0.0 | 0.0 | 0.0 | 0.0 |
| pour_by_language | 0.0 | 0.0 | 0.0 | 0.0 |

---

## 6. 观察与解读

### 6.1 训练是有效的，但只在少数任务上

score > 0 的任务数 **19 → 26**，且 score 提升最大的几个任务：

| task | score 20k → 100k | Δ |
|---|---|---:|
| put_bottles_into_dustbin | 23.9 → 45.1 | **+21.2** |
| organize_table | 5.5 → 13.5 | +8.0 |
| insert_key | 9.0 → 12.6 | +3.6 |
| pour_balls_into_vase | 0.0 → 4.0 | +4.0 |
| match_and_pick_from_conveyor | 0.0 → 4.0 | +4.0 |
| make_kong | 0.0 → 4.0 | +4.0 |

退化的只有 3 个且幅度都小于 1：`insert_tubes` −0.8、`make_toast` −0.5、`build_tower` −0.4，
在单 seed、50 episode 的噪声范围内。

### 6.2 Open 维度几乎完全没有信号

8 个 Open 任务（语言指令、图像指定目标、符号推理）在两个 checkpoint 上 SR 全为 0，
score 也几乎全为 0（100k 只有 `stack_blocks_by_language` 0.4、
`classify_objects_by_language` 0.2）。RoboDojo-only 后训练没有带来任何开放指令能力。

### 6.3 Memory 维度接近全零

6 个任务里只有 `match_and_pick_from_conveyor`（100k, SR 4%）非零。

### 6.4 泛化短板最明显

见第 4 节：随机化场景 SR 归零。这比整体 SR 低更值得关注——它说明问题不只是"训练不够"，
而是策略过拟合到了特定布局。

---

## 7. 数据完整性与已知缺口

| 项 | 状态 |
|---|---|
| run-config 覆盖 | 54/54（两个 checkpoint 都是） |
| 计分任务（官方口径） | **41/42** |
| 唯一缺口 | `deposit_coin` 只有 49 个可用 layout（协议要求 50）——**结构性，无法补齐** |

### `deposit_coin` 为什么永远只有 49 个 episode

**`deposit_coin` 这个任务只有 49 个可用 layout。** 已用 4 次独立运行确认：

| job | 请求 | 实得 | layout 范围 |
|---|---|---|---|
| `f90414a9c58f`(20k) | `--eval_num 50` | 49 | 0–48（连续） |
| `7f36acc97923`(100k) | `--eval_num 50` | 49 | 0–48（连续） |
| `41aa14f89a7b`(20k) | `--eval_num 51` | 49 | 0–48（连续） |
| `f78fa09b9bbc`(100k) | `--eval_num 51` | 49 | 0–48（连续） |

请求 51 个仍然只得 49 个、且 layout 连续无空洞，说明这是 benchmark 侧的 layout 池上限，
不是丢帧、不是超时、也不是本次配置问题（四次都 `exit_code=0` 干净退出）。

**推论：官方协议的 `complete: true` 对完整 42 任务 benchmark 是不可达的**——协议要求非
Generalization 任务各 50 个 episode，而 `deposit_coin` 最多只能给 49 个。**41/42 就是上限**，
任何人用这套 `robodojo_eval.py` 跑都一样，包括同事原版流程。

同类现象：Generalization 的 `_random` 变体 layout 池上限是 25（`max_layout=24`），
所以协议对它们只要求 25 个——这部分是自洽的。

`deposit_coin` 在两个 checkpoint 上 SR 都是 0%、score 分别 1.22% / 0.82%，
纳入与否都不改变任何结论（见 §3 的两种口径对照）。

### 其他限制

- **单 seed（seed 0）**，没有跨 seed 方差，单任务 SR 的分辨率是 1/50 = 2%。
- **没有外部 baseline 对照**，本文只做 20k vs 100k 的内部比较。
- Generalization 的 `_random` 变体每个只有 25 个可用 layout（`max_layout=24`），
  这是 benchmark 侧的上限，不是本次配置问题。

### 一个容易踩的单位陷阱

per-run-config 的 `_result.json` 里两个字段量纲不一致：`success_rate` 是 [0,1] 的分数，
而 `score` **已经是 [0,100] 的百分数**。直接对 `score` 再乘 100 会得到 >100% 的结果。
汇总脚本现在统一从逐 episode 的 `details` 重算，并对这个约定做断言校验。

---

## 8. 复现

```bash
ssh <devbox>
cd /home/users/kun01.wu-labs/git_repo/robo_orchard_lab
export PATH=/home/users/kun01.wu-labs/.local/bin:$PATH
source /home/users/kun01.wu-labs/miniconda3/etc/profile.d/conda.sh && conda activate holobrain_internal
cd projects/holobrain_internal/scripts

# 20k
python aggregate_robodojo_results.py \
  --gen-job bcloud-bj-zone1-a52719406c5c \
  --nongen-job bcloud-bj-zone1-f90414a9c58f \
  --label 20k --out-dir /tmp/agg_20k_final

# 100k
python aggregate_robodojo_results.py \
  --gen-job bcloud-bj-zone1-883434858d0c \
  --nongen-job bcloud-bj-zone1-7f36acc97923 \
  --override store_laptop_and_headphones_random=bcloud-bj-zone1-7f36acc97923 \
  --label 100k --out-dir /tmp/agg_100k_final
```

产物：`benchmark_summary_seed_0.json`（官方口径）、`runconfig_details_seed_0.json`
（逐 run-config SR/score/episode 数/来源 job）。

提交用的配置：`common/aidi_submit_config/submit_cfg_robodojo_eval_kun_{20k,100k}{,_50ep,_coin51}.json`。
