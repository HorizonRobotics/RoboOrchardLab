# RoboDojo × HoloBrain Post-training + Eval — Session STATUS

**Session**: `d2f037a8-f216-4a38-9d98-14d2d7f15cb9` (recovery from `10f5c967` + `ff050624`)
**Last updated**: 2026-07-28 22:35 CST (14:35 UTC)
**Working dir**: `/home/users/kun01.wu-labs/git_repo/robo_orchard_lab`  ·  Branch `feature/memory_dev1`

---

## 1. 目标 & 验收标准

**目标**：HoloBrain 只用 RoboDojo 数据后训练，在集群做完整评测，与 Xiaomi_Robotics_0 baseline 对比 per-task success rate。

**验收标准**：至少 1 组后训练 ckpt 的 seed0 × 25-ep 覆盖 25+ 个 task（partial coverage 按 xiaomi 精度可接受），产出 per-task SR 对比 markdown。

**用户硬约束**：
- 只做 seed0（不做 seed1/seed2）
- 遇同一失败 3 次修不好停下汇报
- 结果没出来前先不 commit（等最后一并提）

---

## 2. 当前真实进度（工件为准，2026-07-28 14:35 UTC）

| 任务 | Job ID | State | 完成度 | 证据路径 |
|---|---|---|---|---|
| 20k train | `bcloud-bj-zone1-1f00b8e23ac8` | Succeeded (07-27) | ✅ 100% | `/horizon-bucket/robot_lab/users/kun01.wu/aidi_output/holobrain_robodojo_posttrain_v9/checkpoint_20000/` (2.83 GB, md5 `a71cb164...`) |
| 100k train | `bcloud-bj-zone1-6c6f0a3cbcb9` | Succeeded (07-27 21:06 UTC) | ✅ 100% | ckpt 已从 cluster PFS 拉到 `/horizon-bucket/.../holobrain_robodojo_posttrain_v9_100k/checkpoint_100000/` (2.83 GB, md5 `29f0d902...`) |
| seed0 eval on **20k** ckpt | `bcloud-bj-zone1-7895445e92bc` | Running (07-27 起) | 🔄 24%+ (task 13/54, `fold_clothes_random`, mtime 07-28 13:42 UTC) | `/horizon-bucket/.../robodojo-holobrain-seed0/eval_result/RoboDojo/*/` |
| seed0 eval on **100k** ckpt | `bcloud-bj-zone1-b645cdeea943` | **Queuing** (07-28 22:28 CST create) | 🔄 0% (刚提) | `/horizon-bucket/.../robodojo-holobrain-seed0-100k/` (将由 job 自动创建) |

**Xiaomi baseline** 3 seed 数据齐（part partial）在 `/horizon-bucket/robot_lab/users/kun01.wu/aidi_output/robodojo-xiaomi-seed{0,1,2}/`，Step 5 收尾用。

---

## 3. 本 session 已完成、已验证的动作

1. ✅ **恢复上下文**：读 3 个 plan (`session-10f5c967-handoff.md` / `keen-wondering-engelbart.md` / `bubbly-cooking-dahl.md`) + 8 篇 memory，用 REST + `aidictl` 校验现场（Explore subagent 汇总的"stall"是过时快照，实际 eval 一直在推进）。
2. ✅ **100k ckpt 上 bucket**：
   - `aidictl job logs list bcloud-bj-zone1-6c6f0a3cbcb9 output/checkpoints` 找到 checkpoint_{18,19,20}（`total_limit=3`）
   - `aidictl job logs download` 拉 `checkpoint_20/{model.safetensors,model.config.json}` (~1 min for 3 GB)
   - `cp` 到 `/horizon-bucket/.../holobrain_robodojo_posttrain_v9_100k/checkpoint_100000/` (~13 s on FUSE)
   - 补 4 附件（`robodojo_processor.json` / `robodojo_inference.config.json` / `urdf/` / `ckpt -> xuewu.lin/ckpt`）从 20k deploy pkg cp
   - `md5sum` 校验：20k `a71cb164...`, 100k `29f0d902...`（不同，说明真是 100k 权重）
3. ✅ **提 seed0_100k eval job**：
   - 新 cfg `~/git_repo/RoboDojo/aidi_submit/cfgs/submit_cfg_holobrain_robodojo_seed0_100k.json`
   - 用 python 精准替换（3 处 ckpt path、output_dir、`--ckpt`）
   - **加了 shim**：`ln -sf /bin/true /usr/local/bin/zenity` + `smoke_all_tasks.sh` 加 `timeout -k 30 3600` per-task hard cap（[[robodojo-eval-timeout-shim]]）
   - `python aidi_submit/submit.py ...` 提交，REST 反查得 job_id `bcloud-bj-zone1-b645cdeea943`（stdout 吞 id 已按 memory `holobrain-aidi-submit-conventions` §2.4 反查）
4. ✅ **Session-only monitor cron**：`d03eb196`，`13 */7 * * *`（每 7h 的 :13 分），检查 20k eval + 100k eval + 100k archive 状态，7 天自动过期
5. ✅ **新 memory**：`aidi-train-output-not-on-bucket.md` + `MEMORY.md` pointer —— 记录训练 ckpt 不自动落 bucket 的常识

---

## 4. 剩余下一步（具体到文件 / 函数）

### 短期（等 eval）
- **观察 seed0_100k 从 Queuing → Running**：`aidictl job status bcloud-bj-zone1-b645cdeea943` 或 REST。5090 队列前面有排队 (07-27 handoff 提到 0 free / 5 waiting，现在情况可能改善)。
- **验 Step 2**：seed0_100k 起来后 5 min 内看 `/horizon-bucket/.../robodojo-holobrain-seed0-100k/benchmark.log` 是否出现 `[smoke_all_tasks] tasks=54` + `RUN align_blocks`；若 30 min 内没起来看 `aidictl job logs cat bcloud-bj-zone1-b645cdeea943 log` 找 python traceback。
- **20k eval 继续跑**：预计 13:42 UTC 之后再 ~30h 到 wall_time，届时 task 会推到 ~35/54（若无 stall）。

### 收尾（依赖 24-48h 后）
- **Step 5 提取 SR，产 markdown**：
  - 位置：`projects/holobrain_internal/docs/robodojo_pipeline/07_results.md`
  - 数据源（4 组）：
    - 20k: `/horizon-bucket/.../robodojo-holobrain-seed0/eval_result/RoboDojo/*/HoloBrain/arx_x5_holobrain/*/2026-07-*_smoke_<task>/_result.json`
    - 100k: `/horizon-bucket/.../robodojo-holobrain-seed0-100k/eval_result/RoboDojo/*/HoloBrain/arx_x5_holobrain/*/2026-07-*_smoke_<task>/_result.json`
    - xiaomi baseline s0/s1/s2: `/horizon-bucket/.../robodojo-xiaomi-seed{0,1,2}/eval_result/RoboDojo/*/`
  - 表格列：task_name | 20k SR | 100k SR | xiaomi-s0 SR | xiaomi-s1 SR | xiaomi-s2 SR | 备注(partial/timeout/n_ep)
  - 用 python 一次性扫，别手抄。
- **Step 5.5 commit（Q3 = 等结果一并提）**：
  - **先撤 `dataset_specs.py` 的 filter_list narrowing**（那是本地临时；100k/seed0 走独立 `dataset_specs_robodojo.py`，不需要动 shared spec）
  - 分 3 commit：
    (a) `projects/holobrain_internal/common/aidi_submit_config/*.json` + `configs/{dataset_specs_robodojo.py,deploy_specs.py}` + `submit_cfg.json`（robodojo 训练+评测）
    (b) `diagnose_agilex_instruction_dup.py` + agilex 相关 handoff doc（诊断，不修）
    (c) `docs/**` + `scripts/eval_robotwin_ckpt11.sh`（新文档栈）
  - `.gitignore` / `setup.py` / `.claude/settings.json` / `config_holobrain_common.py` 一行 pyright suppression 可跟 (a) 合。

### 后续可选
- **加 seed0_100k 进 monitor cron 报告**：cron `d03eb196` 已经写了 filter 兼容 `robodojo_holobrain_seed0*`。
- **如果结果证明 100k 显著优于 20k**：可考虑提 seed1/seed2 拿到置信区间。**用户目前 no**，除非结果强烈到值得。

---

## 5. 已定决策（用户已回答，别再问）

| Q | 决策 | 时间 |
|---|---|---|
| 100k eval 策略 | 并行提 seed0_100k（20k eval 继续跑，两组数据都要） | 2026-07-28 |
| Monitor cron | Session-only, 每 7h（`d03eb196`, 7 天自动过期） | 2026-07-28 |
| Commit 时机 | 等评测结果出来一并提；不 commit `dataset_specs.py` 的 filter_list narrowing | 2026-07-28 |
| seed 数量 | 只做 seed0（不做 seed1/seed2） | 2026-07-27 |
| Eval num | `--eval-num 25`（不 100，节约 wall_time） | 2026-07-27 |
| 未来 eval submit_cfg | 加 zenity + timeout shim（[[robodojo-eval-timeout-shim]]） | 2026-07-28 06:01 |
| 20k seed0 是否停 | 不停，让它跑到 wall_time（partial coverage OK） | 2026-07-28 06:01 |

## 6. 已知坑 & workaround

1. **AIDI 训练 ckpt 不自动落 bucket**：只在 `/job_data`（pod-local，job 结束就没）+ cluster PFS（HTTP via aidictl）。要拿必须 `aidictl download + cp`。见 [[aidi-train-output-not-on-bucket]]。
2. **aidisdk submit stdout 吞 job_id**：用 REST `/job/list?user_name=` 反查最新一条。见 [[holobrain-aidi-submit-conventions]] §2.4。
3. **aidictl list 有 15 min 缓存**：反查刚提交的 job 要走 REST。
4. **`rsync -aL` 里 symlink target 到 dev-machine 会 dangling**：RoboDojo/robo_orchard_lab/robo_orchard_lab/ 必须实拷贝。
5. **IsaacLab pin 2.4.31 exact 不匹配 image 里的 2.4.30**：submit_cfg cmd 里 sed patch 强改成 `{}`。
6. **numpy 1.26.4 硬 pin**：mplib 0.2.1 要 <2；numpydantic 1.10 要 ≥2 但 eval 路径不 hit，故 mplib 胜。
7. **`smoke_all_tasks.sh:340` 无 per-task timeout**：加 `timeout -k 30 3600` shim，防单 task 死拖整批（xiaomi seed0 v7 的 8h40m zenity 死循环前车之鉴）。
8. **`dataset_specs.py` filter_list narrowing 不能 commit**：那是本地实验用；robodojo 训练走 `dataset_specs_robodojo.py` 独立 spec，不 touch shared。
9. **`diagnose_agilex_instruction_dup.py` 是诊断脚本，P0 lmdb dup-open bug 未修**：不阻塞本次任务，标记 tech-debt。修法看 `projects/holobrain_internal/docs/claude_tasks/2026-07-22_agilex_...md` 里的 5 个方案（推荐 A：`dataset_factory._build_typed_datasets` dedup）。
10. **20k 前 12 task PASS 但 SR=0.0** 是 subagent 转述里说的（我没直接验，只看 `smoke_results/*.json` counts.PASS=12）；PASS=可跑完 25 ep 不 crash ≠ SR>0。真实 per-task SR 要看 `_result.json`。

## 7. 复现当前状态所需的命令

```bash
# 环境
cd /home/users/kun01.wu-labs/git_repo/robo_orchard_lab
git checkout feature/memory_dev1  # 已在
source /home/users/kun01.wu-labs/miniconda3/etc/profile.d/conda.sh
conda activate holobrain_internal

# ① 拿最新 job 状态
python3 <<'PY'
import requests, json
tok = open("/home/users/kun01.wu-labs/.aidisdk/config.yaml").read().split("token:")[1].split("\n")[0].strip()
r = requests.get("http://computing.aidi.hobot.cc/infra/api/v1alpha/computing-apiserver/job/list",
    headers={"Authorization": tok}, params={"limit":50,"user_name":"kun01.wu-labs"}, timeout=15)
for j in r.json().get("data",{}).get("list") or []:
    n = j["job_name"]
    if any(t in n for t in ("robodojo_holobrain_seed0","holobrain_robodojo_posttrain")):
        s = j.get("job_status",{}) or {}
        print(f'{j["job_id"]} | {s.get("phase")} | {s.get("create_time")} | {n[:60]}')
PY

# ② 看 seed0 (20k) 进度
grep -c '\[smoke_all_tasks\] RUN' /horizon-bucket/robot_lab/users/kun01.wu/aidi_output/robodojo-holobrain-seed0/benchmark.log
ls -td /horizon-bucket/robot_lab/users/kun01.wu/aidi_output/robodojo-holobrain-seed0/eval_result/RoboDojo/*/ | head -3

# ③ 看 seed0 (100k) 进度（Queuing 阶段先看 job/list）
ls -la /horizon-bucket/robot_lab/users/kun01.wu/aidi_output/robodojo-holobrain-seed0-100k/ 2>/dev/null

# ④ 看 100k ckpt 已就位
ls -lh /horizon-bucket/robot_lab/users/kun01.wu/aidi_output/holobrain_robodojo_posttrain_v9_100k/checkpoint_100000/

# ⑤ 若需在集群 stop seed0 (20k)
# aidictl job stop bcloud-bj-zone1-7895445e92bc

# ⑥ 若需 aidictl 拉 100k tboard 曲线
aidictl job logs download bcloud-bj-zone1-6c6f0a3cbcb9 tboardlog
# then: tensorboard --logdir ./tboardlog_bcloud-.../ --port 6006
```

---

## 8. Gap Analysis（Phase 5 主动补的）

按用户 template 6 层，各分"必须补"vs"可以以后再说"，投入产出排序：

### 目标层
- **原目标仍成立**：post-train + eval + 对 xiaomi baseline，需求没漂。**没有更简单达成路径**：eval 靠集群 GPU 与 baseline 对齐的仿真环境，不可代替。**没有 over-engineering**：100k 是自然延伸，shim 是安全网。

### 正确性
- **必须补**：Step 5 提 SR 时**必须过滤 `_result.json` 里 partial 的（reward 未落到 predicate 就 wall_time 到期）**；否则把 timeout 的 task 当 SR=0 会误判 policy 差。看 `_result.json` 里 `episodes[].steps` 若 == `step_lim` 且 `success=false` 才是 hard-fail；`stopped=true` 且 `success=false` 是 partial。
- **必须补**：Step 5 对比 xiaomi 时**只做 pair-wise（HoloBrain 和 xiaomi 都覆盖的 task）**；xiaomi 3 seed 也是 partial，两边分母不同直接比 mean 会误导。
- **可选**：加 policy-server 起来时的 5 min "smoke sample check"（前 3 ep 看 action 是不是全零 → 立刻 abort）。目前没做，seed0 (20k) 前 3 ep 里 SR 有没有 >0 需要看 `_result.json`；如果全 0，说明 policy 根本没学会（不是 partial 问题）。

### 一致性
- **必须补**：commit 前撤 `dataset_specs.py` filter_list narrowing。已经在 STATUS 里标了。
- **可选**：`config_holobrain_common.py` 那行 pyright suppression 是否要 commit（是本地 lint 用；review 时问）。
- **可选**：`docs/robodojo_pipeline/06_hang_risk_and_timeouts.md` 的"cover_blocks hang"结论**已作废**，正式收尾时要在文档尾部加"2026-07-28 revised"，避免误导。

### 可运行性
- **必须补**：`.claude/settings.json` 的 deny list 里的 `rm *` 会让下次 scratch 清理失败（本 session 已遇到）。scratch dir `/home/users/kun01.wu-labs/scratch_100k_ckpt/` 现在还占 3 GB，用户下次登录时手动清（或加白名单）。
- **可选**：`env HOLOBRAIN_DATA_BASE` 在 `dataset_specs_robodojo.py` 硬默认 `./data`。集群 pod 里靠 to_upload 时的 `ln -sfn all_data ...` 拿到，dev machine 也 OK。**不换机就不会出问题**。

### 规模与性能
- 当前 pipeline 上限：eval 单 task ~1.33 h × 54 task = 72 h 单 seed，与 wall_time=48h 差 24h → 结构性 partial。**决方案**：future 若要 full coverage 见 xiaomi 拆并发（每组 8 task × 6 job × 12h）；不是本次范围。
- 100k train 26h（20k 8-10h），wall_time=72h 富余大，无风险。

### 可观测性
- **必须补**：写死一份 `benchmark.log tail` 加 result_json 扫描 的一句脚本进 `docs/robodojo_pipeline/04_commands_cheatsheet.md`（快速 diagnostic）。本次 status 里 §7 已经有，可搬。
- **可选**：monitor cron `d03eb196` 触发时把 output 追加到 `~/.claude/logs/robodojo_monitor.log`（要改 cron prompt 写文件）。**决**：不做，7 天过期本身就是限制。

### 技术债
- **本次引入**：
  - scratch dir `/home/users/kun01.wu-labs/scratch_100k_ckpt/` 没删（rm 权限被 deny）—— 用户手动清
  - `dataset_specs.py` filter_list narrowing 未回滚（等 commit 时处理）
  - `06_hang_risk_and_timeouts.md` 首段"stop it"结论已过时（已在 memory 记 revised，但文档未改）
- **既存未修**：agilex instruction lmdb dup-open P0 bug（`diagnose_agilex_instruction_dup.py` 只诊断，不修）—— 用户没让修

### 元层面
- **本次中断暴露**：session-only cron 撑不过网络掉线；durable cron 是选项但用户目前偏好不 pollute repo。**折中**：cron 不 durable，但**每次 session 都把「监控目标 + cron_id + 触发规则」写进 STATUS.md**，恢复时看 STATUS 就知道要不要重设。已经这样做了。
- **让下次恢复更便宜**：本次靠 `session-10f5c967-handoff.md` 复活，效率高（Explore agent 找到就完全够用）。**关键动作**：这类多任务并行 workload 一定要写 handoff 到 `.claude/plans/session-<id>-handoff.md`，别只靠 memory（memory 是 point-in-time，plan 里的 job_id/bucket path 是当下事实）。
- 3 个 Failed sanity（07-27 14:23/14:37/15:33/15:56/16:16）在 REST 里被 aidictl 屏蔽（TSV 只列 5 条）—— 恢复时靠 REST 才看得全。**决方案**：STATUS 里指定 recovery 命令用 REST 而非 aidictl。已写。
