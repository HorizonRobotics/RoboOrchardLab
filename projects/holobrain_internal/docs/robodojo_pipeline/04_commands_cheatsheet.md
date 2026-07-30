# 04 — 命令 Cheatsheet

所有需要跑的命令，按场景组织。**绝大部分命令来自实操 session 10f5c967**（100k train + sanity + seed0 eval）。

---

## 0. 环境准备

```bash
# conda env
source /home/users/kun01.wu-labs/miniconda3/etc/profile.d/conda.sh
conda activate holobrain_internal   # dev machine 上的提交环境（RoboOrchardJob-AIDISubmit 等在此）

# 项目根（现行流程：训练与评测都在这里）
cd /home/users/kun01.wu-labs/git_repo/robo_orchard_lab

# 外部 RoboDojo repo —— 仅旧评测流程需要，现行流程用不到
# cd /home/users/kun01.wu-labs/git_repo/RoboDojo
```

---

## 1. 提交 AIDI job

### 1.1 训练（robo_orchard_lab 侧）

```bash
cd /home/users/kun01.wu-labs/git_repo/robo_orchard_lab

# sanity（5k step，验证 pipeline）
RoboOrchardJob-AIDISubmit submit_from_config \
    --config projects/holobrain_internal/common/aidi_submit_config/submit_cfg_robodojo_train_sanity.json

# 20k baseline
RoboOrchardJob-AIDISubmit submit_from_config \
    --config projects/holobrain_internal/common/aidi_submit_config/submit_cfg_robodojo_train.json

# 100k full
RoboOrchardJob-AIDISubmit submit_from_config \
    --config projects/holobrain_internal/common/aidi_submit_config/submit_cfg_robodojo_train_100k.json
```

### 1.2 评测（现行：in-repo `robodojo_eval.py`）

```bash
cd /home/users/kun01.wu-labs/git_repo/robo_orchard_lab
C=projects/holobrain_internal/common/aidi_submit_config

# sanity（4 task × 5 ep，2 GPU）—— 验证 pipeline 通不通
RoboOrchardJob-AIDISubmit submit_from_config --config $C/submit_cfg_robodojo_eval_kun_20k_sanity.json

# 第一批：54 run-config × 25 ep（8 GPU × 2 proc = 16 worker，约 6.5h）
RoboOrchardJob-AIDISubmit submit_from_config --config $C/submit_cfg_robodojo_eval_kun_20k.json
RoboOrchardJob-AIDISubmit submit_from_config --config $C/submit_cfg_robodojo_eval_kun_100k.json

# 第二批：30 个非 Generalization 任务 × 50 ep（协议要求，约 9h）
RoboOrchardJob-AIDISubmit submit_from_config --config $C/submit_cfg_robodojo_eval_kun_20k_50ep.json
RoboOrchardJob-AIDISubmit submit_from_config --config $C/submit_cfg_robodojo_eval_kun_100k_50ep.json
```

**为什么要提两批**：官方协议每个任务要 50 个 episode。12 个 Generalization 任务是
「标准 25 + 随机 25」，`--eval_num 25` 那批就已满足；其余 30 个任务要从单个 run-config
取 50 个，必须另跑一批 `--eval_num 50`。两批结果由
`scripts/aggregate_robodojo_results.py` 合并，详见 [07_results.md](07_results.md) §2。

**旧流程（已弃用，仅作历史参考）**：

```bash
# cd /home/users/kun01.wu-labs/git_repo/RoboDojo
# RoboOrchardJob-AIDISubmit submit_from_config \
#     --config aidi_submit/cfgs/submit_cfg_holobrain_robodojo_seed0.json
```

**注意**：`submit_from_config` 会打印 `Command executed:` 但**吃掉 job_id**（[[../../CLAUDE.md]] / skill `aidi-cloud-submit` §2.4 陷阱）。要立即查 job_id 见 §2。

---

## 2. 查 job_id / 状态

`aidictl job list` 有 **15 分钟缓存**，不适合刚提交的 job。用 REST API：

```bash
python3 <<'PY'
import requests
token = open("/home/users/kun01.wu-labs/.aidisdk/config.yaml").read().split("token:")[1].split("\n")[0].strip()

# 最近 20 条
r = requests.get(
    "http://computing.aidi.hobot.cc/infra/api/v1alpha/computing-apiserver/job/list",
    headers={"Authorization": token},  # 注意：不是 Bearer，就是 raw token
    params={"limit": 20, "user_name": "kun01.wu-labs"},
)
for j in r.json()["data"]["list"]:
    s = j["job_status"]
    print(f"{j['job_id']} | {s.get('phase'):10s} | {j['job_name'][:60]}")
PY
```

### 单 job 状态查询（不受缓存影响）

```bash
python3 <<'PY'
import requests
token = open("/home/users/kun01.wu-labs/.aidisdk/config.yaml").read().split("token:")[1].split("\n")[0].strip()
JOB_ID = "bcloud-bj-zone1-6c6f0a3cbcb9"
r = requests.get("http://computing.aidi.hobot.cc/infra/api/v1alpha/computing-apiserver/job/get",
                 headers={"Authorization": token}, params={"job_id": JOB_ID})
d = r.json()["data"]
s = d["job_status"]
print(f"phase={s['phase']}  create={s.get('create_time')}  start={s.get('start_time')}  end={s.get('end_time')}")
PY
```

---

## 3. 拉日志 / 抓训练进度

### 3.1 列出 log 目录

```bash
aidictl job logs ls <job_id> log
# 或 output / tboardlog
aidictl job logs ls <job_id> output
```

### 3.2 Tail 训练主 log

```bash
JOB_ID=bcloud-bj-zone1-6c6f0a3cbcb9
# 训练 log 一般在 task-1（rank 0）
aidictl job logs tail $JOB_ID log/$JOB_ID-task-1-main.log

# 抓每步 loss
aidictl job logs tail $JOB_ID log/$JOB_ID-task-1-main.log | grep "GlobalStep\[" | tail -10

# 抓 checkpoint 落地
aidictl job logs cat $JOB_ID log/$JOB_ID-task-1-main.log | grep "Save checkpoint" | tail -10

# 抓训练速度和 ETA
aidictl job logs tail $JOB_ID log/$JOB_ID-task-1-main.log \
    | grep -E "Training Speed|Estimated Remaining" | tail -5
```

### 3.3 Tail 评测 log

```bash
JOB_ID=bcloud-bj-zone1-7895445e92bc
aidictl job logs tail $JOB_ID log/$JOB_ID-task-0-main.log \
    | grep -E "RUN |Success nums|wall_clock" | tail -20
```

### 3.4 下载全部 log

```bash
aidictl job logs download $JOB_ID output/ --dest ~/tmp_output/
aidictl job logs download $JOB_ID log/    --dest ~/tmp_logs/
```

### 3.5 拿日志 URL（浏览器打开）

```bash
aidictl job logs url $JOB_ID
# 输出 log/output/tboardlog 三个 URL
```

---

## 4. 读评测结果

> **现行流程的结果不在 bucket，在 job 自己的 PFS**（`output/robodojo_eval_results/`），
> 要用 `aidictl job logs list/cat` 取。下面 §4.1–§4.3 里的 bucket 路径属于旧流程。
>
> **已汇总好的最终结果**（推荐直接用，不必自己解析）：
> - `docs/robodojo_pipeline/results/{20k,100k}/benchmark_summary_seed_0.json` —— 官方口径
> - `docs/robodojo_pipeline/results/{20k,100k}/runconfig_details_seed_0.json` —— 逐 run-config
> - 结论见 [07_results.md](07_results.md)
>
> **一次拿到所有任务的 SR**（比逐个解析 `_result.json` 快得多）：
> ```bash
> aidictl job logs cat <job_id> "log/<job_id>-task-0-main.log" | grep 'finished: success_rate'
> ```
> 注意路径是 `log/` 而不是 `log/run_0/`。
>
> **单个 run-config 的原始结果**，路径比想象的多一层 run_id：
> ```
> output/robodojo_eval_results/RoboDojo/<run_config>/holobrain_robodojo_policy/arx_x5/
>   0_ckpt_name=holobrain,action_type=joint/<run_id>/_result.json
> ```
>
> **重新汇总**（合并两批 job，产出官方口径 summary）：
> ```bash
> cd projects/holobrain_internal/scripts
> python aggregate_robodojo_results.py --gen-job <25ep_job> --nongen-job <50ep_job> \
>     --label 20k --out-dir /tmp/agg_20k
> ```
>
> 不要用 `aidictl job logs download` 拉大文件——会静默截断且照样 exit 0。

旧流程的评测 job 会把结果 rsync 到
`/horizon-bucket/robot_lab/users/kun01.wu/aidi_output/robodojo-holobrain-seed0/`
（**注意该目录只有 13/54 run-config，那个 job 被提前停掉了**）。

### 4.1 单 task result

```bash
BUCKET=/horizon-bucket/robot_lab/users/kun01.wu/aidi_output/robodojo-holobrain-seed0
TASK=align_blocks

# 找 _result.json
find $BUCKET/eval_result/RoboDojo/$TASK -name "_result.json"

# 解析：SR / score / detail
python3 -c "
import json, glob
p = glob.glob('$BUCKET/eval_result/RoboDojo/$TASK/HoloBrain/*/0_ckpt_name*/2026-07-27*/_result.json')[0]
d = json.load(open(p))
print(f'SR={d[\"success_rate\"]:.3f}  score={d[\"score\"]:.3f}  eval_time={d[\"eval_time\"]}')
"
```

### 4.2 所有 task 汇总

```bash
BUCKET=/horizon-bucket/robot_lab/users/kun01.wu/aidi_output/robodojo-holobrain-seed0

python3 <<'PY'
import json, os, glob
BUCKET = "/horizon-bucket/robot_lab/users/kun01.wu/aidi_output/robodojo-holobrain-seed0"

# smoke_results/<run_id>.json (整批 task 的 status 汇总)
d = json.load(open(f"{BUCKET}/smoke_results/2026-07-27_21-49-05_smoke.json"))
print(f"counts: {d['counts']}")
print(f"{'task':<38}{'status':<8}{'ep':<5}{'elapsed_sec':<12}")
for r in d.get('results', []):
    print(f"{r['task']:<38}{r['status']:<8}{r['eval_time']:<5}{r['elapsed_sec']:<12}")

# 每 task 的 _result.json (SR / score)
print()
base = f"{BUCKET}/eval_result/RoboDojo"
print(f"{'task':<38}{'SR':<8}{'score':<10}{'ep':<6}")
for t in sorted(os.listdir(base)):
    p = glob.glob(f"{base}/{t}/HoloBrain/*/0_ckpt_name*/2026-07-27*/_result.json")
    if not p: continue
    dd = json.load(open(p[0]))
    print(f"{t:<38}{dd.get('success_rate',0.0):<8.3f}{dd.get('score',0.0):<10.3f}{dd.get('eval_time',0):<6}")
PY
```

### 4.3 看视频（下载到本地）

```bash
BUCKET=/horizon-bucket/robot_lab/users/kun01.wu/aidi_output/robodojo-holobrain-seed0
TASK=align_blocks
mkdir -p ~/tmp_videos/$TASK
rsync -av $BUCKET/eval_result/RoboDojo/$TASK/HoloBrain/*/0_ckpt_name*/2026-07-27*/episode_0000000_cam_*.mp4 \
       ~/tmp_videos/$TASK/
ls ~/tmp_videos/$TASK/
```

---

## 5. Stop / urgent job

```bash
# 停 job
aidictl job stop <job_id>

# 加急（提高优先级）
aidictl job urgent <job_id>

# 取消加急
aidictl job urgent --cancel <job_id>
```

---

## 6. 查集群队列（判断是否 Queuing 太久）

```bash
aidictl queue ls --type gpu -f "top=5"
# 输出：queue_name, cpu/gpu allocated/free, waiting_jobs
```

`project-5090-robot-lab-bcloud-bj` 是 kun 用的队列。**free_gpu=0 且 waiting > 3** 时新 job 会 Queuing 几小时。

---

## 7. Docker image 相关

### 7.1 拉镜像

```bash
docker login docker.hobot.cc   # 一次性
docker pull docker.hobot.cc/imagesys/kun01.wu/robodojo-holobrain:ubuntu22.04-gcc11.4-cu128-torch280-holobrain-20260727-v6
```

### 7.2 起 dev container

见 skill `internal-docker` 或 [[../CLAUDE.md]]。核心命令：

```bash
docker run -it --rm --gpus all \
    -v /home/users/kun01.wu-labs/git_repo:/git_repo \
    -v /horizon-bucket:/horizon-bucket:ro \
    docker.hobot.cc/imagesys/kun01.wu/robodojo-holobrain:...-v6 \
    bash
```

### 7.3 commit + push（迭代 image）

```bash
CONTAINER_ID=$(docker ps -q -l)
docker commit $CONTAINER_ID docker.hobot.cc/imagesys/kun01.wu/robodojo-holobrain:...-v7
docker push docker.hobot.cc/imagesys/kun01.wu/robodojo-holobrain:...-v7
```

---

## 8. AIDI SDK config 位置

```
/home/users/kun01.wu-labs/.aidisdk/config.yaml
```

含 API `token`（拉 REST API 时用），server URL 等。别放到 repo 里。

---

## 9. 常用 monitor cron

设定每 7h 自动 poll（session 10f5c967 用的策略）：

```bash
# 在 Claude Code session 里
CronCreate cron="17 0,7,10,17 * * *" recurring=true prompt="Monitor RoboDojo HoloBrain 集群 jobs..."
CronList
CronDelete id=<cron_id>
```

---

## 10. Handoff 备忘

Session 10f5c967 里，**关键 job ID**：

| Job | ID | 用途 |
|---|---|---|
| 5k sanity 训练 | `bcloud-bj-zone1-4fb0ee2ff3d4` | 验证 pipeline |
| 20k baseline 训练 | `bcloud-bj-zone1-1f00b8e23ac8` | 首个 checkpoint 来源 |
| 100k full 训练 | `bcloud-bj-zone1-6c6f0a3cbcb9` | 主训练 |
| sanity smoke eval | `bcloud-bj-zone1-805a64eaab5f` | 2 task PASS |
| seed0 full eval | `bcloud-bj-zone1-7895445e92bc` | 54 task × 25 ep |

---

## 11. 从 dev machine 上快速起本地 policy server（不走 AIDI）

（用于开发调试 —— 本仓库项目 CLAUDE.md 未详述，若需要参见 skill `local-train`）

```bash
cd /home/users/kun01.wu-labs/git_repo/RoboDojo
docker run -it --gpus=all \
    -v $PWD:/workspace \
    -v /horizon-bucket:/horizon-bucket:ro \
    -w /workspace \
    docker.hobot.cc/imagesys/kun01.wu/robodojo-holobrain:...-v6 \
    bash -c "
      source /opt/miniconda3/etc/profile.d/conda.sh
      conda activate RoboDojo
      bash scripts/robodojo.sh benchmark \
          --policy-dir XPolicyLab/policy/HoloBrain \
          --ckpt checkpoint_20000 \
          --env-cfg arx_x5_holobrain \
          --eval-num 1 \
          --policy-gpu 0 --env-gpu 1
    "
```

（本地跑一整 task 15 min，估计 5090 24GB VRAM 才够；单卡 4090 24GB 也行但 concurrency=1）

---

## 12. 备忘 — 需 confirm 才能跑的破坏性操作

- **删 workspace_folder**：`clear_workspace=true` 会自动清；手工 `rm -rf submit-holobrain-*/`
- **改 bucket 里 checkpoint_20000/model.safetensors**：会影响正在跑的 eval！建议先复制到 `checkpoint_20000_v2/` 再改 symlink
- **rebuild docker image**：确认 tag suffix 加 `v7`/`v8`，不要覆盖 `v6`
