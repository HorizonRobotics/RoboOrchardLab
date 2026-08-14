# `num_envs > 1` 评测塌陷:调查记录与挂起理由(2026-08-14)

`num_envs=1` 正常,`num_envs=2` 与 `4` 下 cover_blocks **八个 layout 全部 0.0**,
包括单 env 下完整成功的那个。本文件记录:修掉了什么、**排除了什么(附证据)**、
新增了哪些观测量与它们的基准值、以及**剩下唯一那个可测的量**。

⚠️ **结论先行:故障不在策略侧。** 观测完整且正确地送达每个 env,策略从正确位姿起步规划,
但机械臂实际只走了单 env 的 1/3~1/5。根因在**执行/仿真侧**。

⚠️ **本条已挂起。** `num_envs>1` 是省时间的优化,不是交付物;2×2 与 stride32 一律
`num_envs=1`,而单 env 的正确性在 E5/E6/E7 的 n1 上反复复现。继续查需要往 RoboDojo
仿真内部插探针,成本上了一个台阶。

## 1. 修掉的两个真实缺陷(`6b48fe29`)

| 缺陷 | 症状 | 判据 |
|---|---|---|
| 所有 env 共用一个记忆库 | `bank_lengths` 只有一个 key | 现在 `bank_keys` 每 env 一个,2/4 env 均确认 |
| `_env_step` 被 `num_envs` 放大 | 2 env 下 `env_step=1600`,而 episode 只有 800 帧 | 现在 `env_step_by_env` 每 env 都是 800 |

根因是同一个:ws 传输层**从不调用** `update_obs_batch`/`get_action_batch`
(`model_client.py:81-101` 把批留在客户端后逐个发 INFER;`model_server._handle_infer`
绑的是 `update_obs`+`get_action`)。身份本来就在 obs 里(`eval_env.py:285`),
`update_obs` 读一下即可,**传输层不用动**。

## 2. 已排除的十二项(每项都有据)

| # | 排除项 | 证据 |
|---|---|---|
| 1 | 记忆库共用 | `bank_keys` 每 env 一个 key(2/4 env) |
| 2 | 计数器被 `num_envs` 缩放 | `env_step_by_env` 每 env = 800 |
| 3 | env 身份 / `uuid` 生成 | `check_multi_env_isolation.py` 16 项 + 4 阴性对照 |
| 4 | 「机械臂几乎不动」是唯一模式 | E3 的特征(44.7 vs 93.9)不存在;4 env 下两种极端**同时**出现 |
| 5 | 仿真步数变少 | 每条 episode 801 帧、每 env 25 次前向、`env_step` 800,与单 env 一致 |
| 6 | **完整观测被误路由** | `obs_dup = 0`(2 env 与 4 env) |
| 7 | **图像/本体感觉错配** | `obs_dup_image_only = 0`(2 env 与 4 env) |
| 8 | 批处理动作代码与单 env 不同 | `take_action` 就是 `take_action_batch([a],[0])`,**同一份实现** |
| 9 | `have_empty` 让插值循环提前退出 | `interpolation_nums = int(obs_manager.collect_interval)` 是**常数** ⇒ 队列等长同步消耗 ⇒「有一个空」等价「全部空」 |
| 10 | `push` 的 env↔控制序列错位 | `push` 内有 `assert len(env_idx_list) == len(control_queue_list)`,错位会**大声报错**,日志里没有 |
| 11 | 记忆库是病因 | E2d:**无记忆库的 baseline 也塌**(判决 B)。⚠️ 只有 1 个 live layout,单独看很弱,但与第 6/7 项合起来足够 |
| 12 | 显存 | 峰值 12.0 / 13.0 / 14.3 GB(1/2/4 env),每 env 边际 ~720~750 MiB |

⚠️ 第 9、10 项是**读代码**得出的排除。这一周读代码**生成并消灭了候选,一次都没产出答案**;
每一次真正的进展都来自测量。把它们记为「已排除」而不是「已理解」。

## 3. 新增的三个观测量与基准值

都在**单次前向内、按 env** 计算,因此不受 episode 长短与 layout 影响
(`action_path` 受影响,这是它的弱点)。实现见 `deploy_policy._record_obs` /
`_record_motion`,测试见 `check_obs_routing.py`(14 项 + 3 个阴性对照)。

| 量 | 含义 | 单 env 基准 |
|---|---|---|
| `action_path` | 策略**指令**的累计位移(episode 累计) | **92.0~93.9**(逐条 77.5~106.0) |
| `obs_jump` | 机械臂**实际**走的距离(每次前向之间) | **2.92~3.11**(逐条 1.88~3.54) |
| `act_gap` | chunk 起点离「算它时那个位姿」多远 | **0.535~0.652**(逐条 0.31~0.65) |
| `obs_dup` | 与另一 env 的观测逐字节相同的次数 | **0** |
| `obs_dup_image_only` | 图像撞了但关节没撞(错配对) | **0** |
| `obs_dup_state_only` | 关节撞了但图像没撞 | 开局的 home 位姿碰撞,**良性** |

⚠️ 签名必须**同时**含图像与关节:所有机器人开局都在同一 home 位姿,只哈希关节会在
每条 episode 第一帧误报重复 —— 恰好在这个读数最需要可信的时候失效。

## 4. 实测结果

`cover_blocks` · mem 包 · chunk 模式 · seed 0。E6 与 E7 是**两次独立运行**。

| | `action_path`(指令) | `obs_jump`(实际) | `act_gap` | `obs_dup` | `dup_image_only` |
|---|---|---|---|---|---|
| **n1** | 92.0 | **2.92 / 3.11** | 0.535 / 0.652 | 0 | 0 |
| n2 env0 | 82.5 | **0.677 / 0.627** | 1.033 / 1.009 | 0 | 0 |
| n2 env1 | 164.9 | **1.307 / 1.167** | 1.084 / 1.030 | 0 | 0 |
| n4 env0 | 963.0 | 1.42 / 1.66 | **12.444 / 9.497** | 0 | 0 |
| n4 env1 | 70.8 | 0.519 / 0.560 | 1.087 / 1.066 | 0 | 0 |
| n4 env2 | 751.4 | 1.864 / 2.679 | **9.034 / 9.489** | 0 | 0 |
| n4 env3 | 41.4 | 0.714 / 0.673 | 1.026 / 1.025 | 0 | 0 |

三条读法:

1. **每个 env 都比单 env 动得少**(0.52~2.68 对 2.92~3.11),而且多 env 的值**异常地紧**
   —— n2 的 env0 三条 episode 是 0.649 / 0.669 / 0.712(离散 10%),而单 env 自然离散近 2 倍。
   ⇒ 一个**固定机制**在限制它,不是任务动力学。
2. **指令与实际反向**:动得最少的 env,`action_path` 反而最高。这正是闭环策略够不到目标时
   不断加大修正量的表现。
3. **`act_gap` 的分裂在两次运行间完全复现**(4 env 下 env0/env2 ≈ 9.5,env1/env3 ≈ 1.03)。
   ⇒ 确定性,不是噪声。偶数槽位是个线索。

因果链(第 3 步之后是推论,不是测量):
**指令欠执行 → 臂落到训练中从未见过的位姿 → 策略输出退化(`act_gap` 爆、`action_path` 虚高)→ 成绩 0**。
所以 `act_gap=9.5` 是**结果不是原因**;根因是第一步。

## 5. 剩下唯一那个可测的量

**每个动作实际执行了多少步仿真,按 env 分开。**

`take_action_batch` 把一个动作展开成 `interpolation_nums` 个控制步
(前 80% 线性插值到目标、后 20% 保持),`push` 入队后
`while not have_empty(env_idx_list): self.step(env_idx_list=...)`。
如果多 env 下每个动作实际推进的步数少于 `interpolation_nums`,臂就到不了目标位姿 ——
这与全部实测一致,而第 9 项只排除了「`have_empty` 提前退出」这一种机制,**没有排除结论本身**。

做法:在 `robodojo_pod_tree.sh` 里给那个 stepping 循环加一个按 env 的计数器,
与 `interpolation_nums` 一起打进日志。这是 pod 端补丁器已经支持的形状
(它已经打了 5 处锚点补丁,每处都在打完后自证)。

⚠️ **判据要打在产物上**:计数器与 `interpolation_nums` 的比值,按 env 分开;
单 env 必须是 1.0,否则探针本身有问题。

## 6. 相关文件

- `deploy_policy.py`:`_record_obs` · `_record_motion` · `_obs_signature` · `_init_runtime_state`
- `scripts/check_obs_routing.py`(14 项 + 3 阴性对照)· `check_multi_env_isolation.py`(16 + 4)
- `scripts/aidi_eval_e2_numenvs.sh`:E2c/E5/E6/E7 都用它,`E2_DRYRUN=1` 可本地全流程干跑,
  `E2_DRYRUN_BREAK=1` 伪造该轮要抓的那个失败
- run_id:`...-e2c_numenvs-20260813-1400-01` · `...-e2d_basectl-20260813-1600-01` ·
  `...-e5_numenvs_motion-20260814-0230-01` · `...-e6_obs_routing-20260814-0400-01` ·
  `...-e7_split_sig-20260814-0530-01`
