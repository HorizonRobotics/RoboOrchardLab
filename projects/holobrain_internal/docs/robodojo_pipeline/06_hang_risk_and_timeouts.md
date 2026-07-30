# 06 — WebSocket 保活 + zenity 死循环风险（重要）

> **范围说明**：本文的 hang / timeout 分析主要基于 xiaomi baseline 的实测。
> xiaomi 本身已移出范围，但其中关于 Isaac Sim 卡死、wall-time、超时阈值的结论
> 对现行 in-repo 流程同样适用（现行流程的实测亦印证：单任务可跑到 90 min 才结束）。


> **状态修订 (UTC 07-28 05:23)**：**Latent risk，非 active incident**。此前判断「HoloBrain 参考 job `bcloud-bj-zone1-7895445e92bc` 已死锁」的结论**不成立**——独立复核实测：
> - 该 job 已完成 **8/54 tasks**（不是 6/54）
> - 文档原来说 hang 在 `cover_blocks` (task 7)——实测 `cover_blocks` **已正常 PASS**（`[MAIN] eval finished` + `wall_clock=6497s (108min)`）
> - 该 job 当前正在跑第 9 个 task `fasten_screws` (step 1608/1900)，log 每几秒都在推进
> - 之前误判可能是把 Isaac Sim 正常 shutdown 时的 `carb.windowing-glfw.plugin` warning 当成了 hang 指纹
>
> xiaomi seed0 v7 job `bcloud-bj-zone1-2eeb1fee4778` 的死锁**是真的**（本文作者已 stop）。**风险机制客观存在**（ping timeout + zenity fork + 无 per-task timeout），只是 HoloBrain 通路上尚未触发。
>
> **本文只做诊断与「配置层」缓解方案。不涉及 RoboDojo / XPolicyLab 源码修改**（那需要在 RoboDojo repo 侧开 issue）。
>
> 相关证据：`~/git_repo/RoboDojo/aidi_submit/STATUS.md` §"v7 死锁分析"、`aidi_submit/HANDOFF.md` §"🔥 07-28"。

---

## 1. 结论摘要

HoloBrain 评测通路与 xiaomi 评测通路共享**同一套 RoboDojo 基础设施**（Isaac Sim 容器 + `smoke_all_tasks.sh` + `XPolicyLab/client_server/ws/*`），因此**共享同一个死锁风险机制**。**xiaomi 通路上已经被触发**（seed0 v7 job 死锁 8h40min），**HoloBrain 通路上目前尚未触发**（reconnect 机制救回来了），但触发条件相同。

- 参考 job `bcloud-bj-zone1-7895445e92bc`（03_eval.md 标注为「默认 job，54 task × 25 ep」）：截至 UTC 07-28 05:23，已 PASS 8/54，第 9 个 task 活跃推进中。因 `wall_time=2880min` 只够跑 ~26-28 task（handoff 时已预警），非 hang 问题。
- xiaomi seed0 v7 job `bcloud-bj-zone1-2eeb1fee4778` 已经死锁 8h40min（task 11/54 `fill_pen_holder`，258 次 `zenity: not found` 死循环），07-28 10:19 CST 被手动 stop。
- 两 job 的**基础设施相同**，触发触发链路相同，只是概率不同。

---

## 2. Root Cause（三层）

### 层 1：WebSocket keepalive ping timeout（触发器）

`XPolicyLab/client_server/ws/model_server.py:37-38` + `protocol/client.py:61-62`：

```python
ws_ping_interval_s: float = 20.0
ws_ping_timeout_s:  float = 20.0
```

`websockets.serve(..., ping_interval=20, ping_timeout=20, max_size=None)`。**deploy.yml 没有 override 通道**，只能在源码里改。

**为什么会 timeout**：`_handle_infer`（`model_server.py:229`）在 `_model_lock` 加锁的整段执行 forward pass。HoloBrain 是 Qwen2.5-VL-3B + 10-step diffusion，冷启动 GPU shader compile 时单次 `get_action` 可以超 20s；xiaomi 是 Qwen3-VL 30-frame chunk generate，也常有 >20s call。任何单次 forward 超过 20s → server 事件循环无法及时回 pong → `ConnectionClosedError` 抛出。

**验证证据**（HoloBrain 参考 job 每 task log 里 `keepalive ping timeout` 次数）：

| Task | HoloBrain 参考 job | 说明 |
|---|---|---|
| align_blocks | 1 | 已 PASS |
| build_tower | 2 | 已 PASS |
| classify_objects | 4 | 已 PASS |
| arrange_largest_number_random | **9** | 已 PASS（reconnect 9 次都成功）|
| cover_blocks | 2 | **已 PASS**（此前误判为 hang）|

结论：**Ping timeout 是常态**，HoloBrain 客户端 reconnect 逻辑到目前为止 100% 救回来。**xiaomi 通路**有极小概率走进 zenity 死循环（`env0 step=0` 时首个 forward 就断连的场景，Isaac Sim 走另一条 except 分支）。**HoloBrain 通路目前未观察到此模式**，但基础设施相同，不能保证未来不触发。

### 层 2：Isaac Sim → `zenity: not found`（放大器）

Isaac Sim / Omniverse Kit 在 headless 容器里遇到内部 error 时会 fallback 尝试 `zenity` GUI 弹窗（Linux 桌面通用工具）。**HoloBrain / xiaomi / RoboDojo 一方 python 代码里都没有 `zenity` 调用**（本次 audit 全 grep 确认），来源是 Isaac Sim 的 C++ 层 crash-reporter。

- xiaomi v7 `fill_pen_holder.log`: **258** 次 `sh: 1: zenity: not found`（8h40m）
- HoloBrain 参考 job 每个 task log: 5-8 次（已收敛，未成死循环）

区别：xiaomi 的死循环是因为 `env0 step=0` 时就 WS 断连（policy 首个 action 从未产出），Isaac 侧某个 except handler 反复 fork zenity 且不检查 exit code；正常 task 是 policy 已经跑几帧后再断，Isaac 走另一条路径不 fork。

**HoloBrain 与 xiaomi 同基础镜像**（`docker.hobot.cc/imagesys/kun01.wu/robodojo-*`），同样没装 zenity，同样有触发 zenity fork 的 Isaac 版本 → **共享风险**。

### 层 3：`smoke_all_tasks.sh` 无 per-task hard timeout（真正病根）

`scripts/internal/smoke_all_tasks.sh:340-344`：

```bash
ROBODOJO_RUN_ID="${task_run_id}" \
ROBODOJO_FATAL_RESTART_COUNT=0 \
"${eval_cmd[@]}" \
  > "${log_path}" 2>&1
```

直接 exec，无 `timeout` 包裹。全 repo `grep -n 'timeout\|kill\|SIGKILL' scripts/` 无 per-task 相关命中。`scripts/eval_policy.sh` 的 bash retry loop 只处理 rc=99/134/139（PhysX fatal），对**无 exit code 的 silent hang 无效**（rc=124 timeout kill 不在处理列表）。

结论：**只要一个 task hang，整个 benchmark（后续 43-48 个 task）永不开始**。这是本次事故最贵的成本。

---

## 3. 三种缓解方案（按侵入性排序）

### 方案 A：cfg cmd 层「zenity 屏蔽 + 顶层 timeout」（推荐，零源码修改）

在 `submit_cfg_holobrain_robodojo_seed0*.json` 的 `cmd` 字段最前面加 3 行：

```bash
# 1) 屏蔽 zenity fork bomb（sh 里所有 zenity 调用秒返回 true）
ln -sf /bin/true /usr/local/bin/zenity 2>/dev/null || true

# 2) 让每 task 单独有 wall-clock 保护
export ROBODOJO_TASK_WALL_S=3600   # 1h per-task 硬上限，超时 SIGKILL

# 3) 整个 benchmark 也包一层 timeout（保险丝）
timeout --preserve-status --kill-after=60 172800 \
    bash scripts/robodojo.sh benchmark ...
```

**代价**：
- ln zenity → /bin/true 是无害操作（正常镜像也应该没这个）
- 顶层 timeout 172800s = 48h 与 AIDI wall_time 一致，只是把「AIDI SIGTERM」换成「timeout SIGTERM」，让 `trap EXIT` 走的 flush rsync 有 60s 窗口跑完
- **单独 export 环境变量 `ROBODOJO_TASK_WALL_S` 不生效** —— 因为 `smoke_all_tasks.sh` 里根本没读它。需要方案 B 或 C 才能真正做 per-task。

**只用方案 A 的效果**：zenity 不再死循环；但如果任一 task 内部 hang（比如 Isaac 卡在 CUDA），仍会耗完剩余 wall_time。**A 是双保险，不是主 fix**。

### 方案 B：本地 fork `smoke_all_tasks.sh` 加 timeout（推荐，配合 A）

不改上游 repo，在 cfg cmd 里 sed patch pod 侧的 smoke_all_tasks.sh：

```bash
# smoke_all_tasks.sh:340-344 附近，把
#   "${eval_cmd[@]}" \
#     > "${log_path}" 2>&1
# 替换成
#   timeout -k 30 "${ROBODOJO_TASK_WALL_S:-3600}" "${eval_cmd[@]}" \
#     > "${log_path}" 2>&1

sed -i 's|"${eval_cmd\[@\]}" \\|timeout -k 30 "${ROBODOJO_TASK_WALL_S:-3600}" "${eval_cmd[@]}" \\|' \
    ${WORKING_PATH}/scripts/internal/smoke_all_tasks.sh
```

**效果**：
- rc=124 时被 record_result 标为 FAIL，自动进下一个 task
- 单 task 上限 1h（可从 env 调整）
- 完全不动 repo，只在 pod 里 patch

**验证**：sed 后 `grep 'timeout -k 30' ${WORKING_PATH}/scripts/internal/smoke_all_tasks.sh` 应命中。

### 方案 C：绕开 smoke_all_tasks.sh，自写 wrapper（最干净，最费事）

在 `aidi_submit/scripts/` 下写一个 `run_holobrain_backfill.sh`，遍历 `task_inventory --only-runnable` 的输出，每个 task 独立 `timeout -k 30 3600 bash scripts/robodojo.sh eval ...`，自己维护 summary。

**优点**：完全不动 repo，逻辑最清晰。
**缺点**：需要重写 record_result / write_summaries 逻辑，多 100+ 行代码，且需要跟 upstream `smoke_all_tasks.sh` 的行为对齐（比如 `--dimension`, `--resume`, `--tasks-file` 语义）。

---

## 4. 上游 issue 建议（给 RoboDojo repo owners）

以下 3 项需要在 RoboDojo repo 侧修，本文只是 issue 索引：

1. **`XPolicyLab/client_server/ws/model_server.py:37-38` + `protocol/client.py:61-62`**：把 `ws_ping_interval_s` / `ws_ping_timeout_s` 提到 deploy.yml 里可配置（推荐 `interval=60, timeout=120`；或直接 `ping_interval=None` 关掉，客户端已有 reconnect 逻辑，靠 request-level 120s timeout 就够）。
2. **`scripts/internal/smoke_all_tasks.sh:340`**：加 `--task-timeout SEC` CLI 参数，默认 3600s，实现里包 `timeout -k 30 $SEC "${eval_cmd[@]}"`；rc=124 视为 FAIL 而非致命错误。
3. **`docker/*` 或 base image**：install `zenity` 或 provide `/usr/local/bin/zenity -> /bin/true` shim，防 Isaac Sim 的 crash-reporter 死循环。

---

## 5. 建议给 HoloBrain 团队的行动项

### 5.1 立刻做（不改代码）

1. **~~Stop 参考 job `bcloud-bj-zone1-7895445e92bc`~~** — **决定：不 stop**（UTC 07-28 05:23 复核）
   - 复核发现该 job 未 hang，仍在活跃跑（第 9 个 task fasten_screws 进行中）
   - 8 tasks 累计 12h 成果，中断浪费更大
   - wall_time 到期时预期完成 ~26-28 tasks 的 partial coverage，与 xiaomi 对齐可用
   - 该 job 让它自然跑到 wall_time，不干预

2. **下次提交 seed1/seed2/100k ckpt eval 时**，在 cmd 开头加两行 shim（cost 低、无破坏性）：
   ```bash
   # 屏蔽 zenity fork bomb（防 xiaomi 那种死循环）
   ln -sf /bin/true /usr/local/bin/zenity 2>/dev/null || true

   # per-task 1h 硬超时，rc=124 会被 record_result 记为 FAIL 后进下一个 task
   sed -i 's|"${eval_cmd\[@\]}" \\|timeout -k 30 3600 "${eval_cmd[@]}" \\|' \
       ${WORKING_PATH}/scripts/internal/smoke_all_tasks.sh
   ```
   把这两行加到 `submit_cfg_holobrain_robodojo_seed{0,1,2}.json` 或 100k ckpt 版本的 cmd 里，位置在 IsaacLab sed patch 之后、`conda activate RoboDojo` 之前。

3. **改 bg_monitor 按 run_id 过滤计数**（现有实现数的是 `eval_result/RoboDojo/*/` 顶层目录数，会因为过去 run 而误报 54/54）：
   ```bash
   n_res=$(find $BUCKET_ROOT/eval_result -path "*${RUN_ID}*_result.json" 2>/dev/null | wc -l)
   ```

### 5.2 中期做（等 upstream 修好之后）

1. 提 PR 到 RoboDojo repo 完成 §4 三项
2. Rerun 参考 job，验证 54 task 能真正跑完 → 更新 `03_eval.md` 顶部的 job id 与 sample count 断言

### 5.3 文档修正建议

- **`03_eval.md` 顶部**（第 5 行）：把「默认 job：`bcloud-bj-zone1-7895445e92bc`（seed0 full eval，54 task × 25 ep）」改成 hedged 表述（当前 6/54 已 PASS，其余因 WS-hang 未完成）。
- **`03_eval.md` §8「中止 / 中断」**：新增 §8.5「WebSocket keepalive 死锁」子节，链接本文。
- **`05_troubleshooting.md`**：加 §16「WS keepalive → zenity 死循环」，指向本文。

---

## 6. 时间线（用于事后复盘）

| 时间 (UTC) | 事件 |
|---|---|
| 07-27 14:57 | HoloBrain 100k train job `bcloud-bj-zone1-6c6f0a3cbcb9` 提交 |
| 07-27 16:54 | HoloBrain seed0 eval job `bcloud-bj-zone1-7895445e92bc` 提交，Queuing |
| 07-27 21:49 | seed0 job 拿到 GPU，run_id `2026-07-27_21-49-05_smoke` 第一个 task 开跑 |
| 07-28 02:19 | Monitor poll: 5/54 PASS，第 6 个 in-progress |
| 07-28 03:49 | task 7 `cover_blocks` 正常 PASS（`[MAIN] eval finished` `wall_clock=6497s`）|
| 07-28 ~04-05 | xiaomi v7 job 被作者 stop（真实死锁），本文档基于其经验写成 |
| 07-28 05:12 | 本文档首版发布，误判 HoloBrain 也 hang，建议 stop seed0 |
| 07-28 05:23 | **独立复核**：seed0 实测 8/54 PASS 且第 9 个活跃推进，非 hang。文档修订，撤回 stop 建议 |
| 07-28 xx:xx | seed0 继续跑到 wall_time (48h) 到期，预期完成 26-28 tasks partial coverage |

---

## 7. 参考文献

- xiaomi 事故报告：`/home/users/kun01.wu-labs/git_repo/RoboDojo/aidi_submit/STATUS.md` §「v7 死锁分析」+ §「v8 时间预算」
- xiaomi handoff：`/home/users/kun01.wu-labs/git_repo/RoboDojo/aidi_submit/HANDOFF.md` §「🔥 07-28」
- xiaomi 死循环 log 完整原件：`/horizon-bucket/robot_lab/users/kun01.wu/aidi_output/robodojo-xiaomi-seed0/smoke_results/2026-07-27_18-47-34_smoke/logs/fill_pen_holder.log`（258 次 zenity）
- HoloBrain cover_blocks 正常完成的 log（此前误判为 hang 的证据反例）：`/horizon-bucket/robot_lab/users/kun01.wu/aidi_output/robodojo-holobrain-seed0/smoke_results/2026-07-27_21-49-05_smoke/logs/cover_blocks.log`
- 通路详解：本目录 `03_eval.md`
- 已知坑：本目录 `05_troubleshooting.md`（本文补 §16 空白）
