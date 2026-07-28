# 05 — FAQ、易踩的坑、你还没想到的地方

按「触发场景 → 症状 → 原因 → 修复」组织。每条尽量给可复现的诊断命令。

---

## A. 提交侧（Dev → AIDI）

### A1. `submit_from_config` 打了 `Command executed:` 但没 job_id

**症状**：stdout 只有 reproducing 命令，没 `job_id = bcloud-bj-zone1-xxx`。

**原因**：aidisdk 用自己的 logger（`aidisdk.*`），默认没接 root logger → INFO 级 job_id 消息被吞。

**修复**（陷阱 1，详见 [[aidi-cloud-submit]] §2.4）：
```bash
python3 <<'PY'
import logging, sys
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s %(levelname)s] %(message)s")
for name in ("aidisdk", "urllib3", "requests"):
    logging.getLogger(name).setLevel(logging.DEBUG)
from robo_orchard_jobs.job_submit.aidi.aidi_submit_job import AIDISDKSubmitConfig
cfg = AIDISDKSubmitConfig(
    job_config_path="submit-holobrain/job_config.yaml",
    queue_name="project-5090-robot-lab-bcloud-bj",
    job_type="train",
)
cfg.command_impl()
PY
```
末尾能看到 `http request code: 200 return: {"code":0,"data":{"job_id":"bcloud-bj-zone1-..."}}`。

或者直接走 REST API 反查（陷阱 2）：
```bash
python3 <<'PY'
import requests, os
token = open(os.path.expanduser("~/.aidisdk/config.yaml")).read().split("token:")[1].split("\n")[0].strip()
r = requests.get("http://computing.aidi.hobot.cc/infra/api/v1alpha/computing-apiserver/job/list",
                 headers={"Authorization": token},
                 params={"limit": 20, "user_name": os.environ.get("USER","kun01.wu-labs")})
for j in r.json().get("data",{}).get("list") or []:
    s = j["job_status"]
    print(j["job_id"], "|", s.get("phase"), "|", j["job_name"][:60])
PY
```

### A2. `aidictl job list` 找不到刚提交的 job

**症状**：提交 5 min 后 `aidictl job list --name <keyword>` 空返回。

**原因**：`aidictl` 走 aidi 平台 list API，有 **15 分钟缓存**。

**修复**：用 A1 的 REST `job/list` 或 `job/get`（by-id 无缓存）。

### A3. 反复提交 → 僵尸 job 消耗集群

**症状**：因为 A1/A2 看不到 job_id，agent 反复重跑，最后集群里躺着 5+ 个同名 job 都在 Running。

**修复**：
1. 只要 `Command executed:` 打出来就假设提交成功
2. 用 REST 反查确认
3. 重跑前必先 `aidictl job stop <old_id> -y`
4. 一天 debug 结束用 REST `job/list` 扫自己 24h 内的 job 检查僵尸

### A4. `to_upload` 里的 symlink 变 dangling

**症状**：pod 里 `/running_package/code_package/robo_orchard_lab` 是坏 symlink 指向 dev 机路径。

**原因**：`rsync -aL` 理论上 follow symlinks，但实测某些跨 fs 场景下 target 会 dangling。

**修复**：把 symlink target **实拷贝**成真实目录。见 `robodojo_pipeline/05_troubleshooting.md` §2。

```bash
rm -rf ~/git_repo/RoboDojo/robo_orchard_lab
cp -r ~/git_repo/robo_orchard_lab/robo_orchard_lab ~/git_repo/RoboDojo/robo_orchard_lab
```

### A5. `workspace_folder` 撑爆

**症状**：dev 机磁盘满了，`du -sh submit-*/` 每个几十 GB。

**原因**：`clear_workspace` 默认 `false` → 每次提交在旧目录基础上 pile up。

**修复**：**所有 submit_cfg 都写 `"clear_workspace": true`**。已 pile 的手动 `rm -rf submit-*/`。

### A6. 忘 mount 某个 bucket

**症状**：pod 里 `ln -s /horizon-bucket/robot_lab2/... assets` 报 `No such file`。

**原因**：`input_bucket` 只写了 `"robot_lab"`，没写 `"robot_lab2"`。

**修复**：`"input_bucket": "robot_lab,robot_lab2"`（逗号分隔）。多个都要写。

### A7. `project_id` / `queue_name` 组合错

**症状**：`403 Forbidden` 或 `Queue not found`。

**修复**：
- HoloBrain 系用 `queue_name=project-5090-robot-lab-bcloud-bj` + `project_id=horizon-labs`
- 不要用 `Robot-lab`（那是 gendata 用的）
- 查你有权限的队列：`aidictl queue ls --type gpu`

### A8. 提交时 cwd 错

**症状**：`FileNotFoundError: robo_orchard_lab`（在提交时）。

**原因**：`to_upload` 相对路径 = **提交时 cwd**（不是 JSON 所在目录）。

**修复**：`cd ~/git_repo/robo_orchard_lab && RoboOrchardJob-AIDISubmit submit_from_config --config <full_path>`。

---

## B. 集群 pod 内

### B1. Wall-time 到期，训练/评测未完

**症状**：Job 状态 `Failed`，log 里最后是 `Terminated` 或 `SIGTERM received`。

**修复**：
- **训练**：`SaveCheckpoint(total_limit=3)` 保证有 latest ckpt，续训用另一 job (`--resume_from`)。或者提前把 `save_step_freq` 调小以便有中间 ckpt。
- **评测**：接受 partial coverage，或降 `--eval-num` 25→15。若真跑不完考虑分组并发（详见 `robodojo_pipeline/05_troubleshooting.md` §9）。
- **wall_time 单位**：**分钟**（源码 typo "minitus"），2880=48h, 4320=72h, 14400=240h。

### B2. checkpoint `total_limit=3` 把中间 ckpt 删了

**症状**：训练完想拿 step 30k 的 ckpt，发现只剩最新 3 份。

**修复**：
- 训练中定期 `rsync` 中间 ckpt 出到 bucket 长期目录
- 或改 `train.py` 里 `SaveCheckpoint(total_limit=None)`
- 或加大 `save_step_freq`（省 IO + 少产生 ckpt）

### B3. `${WORKING_PATH}` 不展开

**症状**：pod log 里出现 `ln -s /path ${WORKING_PATH}/data` 字面字符串，报 `No such file: ${WORKING_PATH}/...`。

**原因**：`cmd` 里的字符串**在 pod 的 bash 里展开**。若你把 `${WORKING_PATH}` 单引号了或写成 `\${WORKING_PATH}` 转义，就不会展开。

**修复**：JSON 里写 `"${WORKING_PATH}/data"`（双引号是 JSON 语法，bash 会正常展开变量）。

### B4. `accelerate` 没找到 → 命令拼错

**症状**：pod log `accelerate: command not found`。

**原因**：镜像里 `accelerate` 不在 `PATH`（比如你镜像的 conda env 不是 activate 状态）。

**修复**：`cmd` 前面加 `source /opt/miniconda3/etc/profile.d/conda.sh; conda activate <env>`；或者用绝对路径 `/opt/miniconda3/envs/<env>/bin/accelerate`。

### B5. GPU OOM

**症状**：`CUDA out of memory`。

**修复**：
- 别提 `batch_size`，横向扩：加 `num_workers` 或 `gpu_per_worker`
- 5090 32GB 上 HoloBrain 3B + batch=16 已经跑满，batch=32 会 OOM
- 或用 gradient accumulation（`accelerate` 支持 `--gradient_accumulation_steps`）

### B6. IsaacLab pin sed patch 忘打

**症状**：pod 一起 IsaacSim 就 exit，log `isaacsim.asset.importer.urdf 2.4.31 not found`。

**修复**：`cmd` 里加 sed patch（见 `robodojo_pipeline/05_troubleshooting.md` §3 或本目录 [04_dual_env_client_server.md](04_dual_env_client_server.md) §3）。

### B7. numpy 版本冲突

**症状**：`import numpydantic` 报 numpy 2.x required 但 mplib 报 numpy<2.0 required。

**修复**：镜像里硬 pin `numpy==1.26.4`（mplib 优先），只要不真实调 numpydantic 就没事。

### B8. Isaac Sim GLFW warning

**症状**：eval log 出现 `Failed to startup plugin carb.windowing-glfw.plugin`。

**通常无害**：headless 模式下正常。只有**连续 warning + 之后无 `[MAIN] eval finished`** 才要担心。

### B9. 训练 SR=0 但 loss 好看

**症状**：`total_loss=0.098` 但 eval SR=0/25。

**原因**：loss 是 diffusion 去噪 loss，不等价于 rollout 成功率。20k step 的 loss 不足以代表 policy 好。

**修复**：定期挑中间 ckpt 起 sanity eval（`--eval-num 1` × 2-3 task）。100k step 后（loss≈0.05）才检验。

---

## C. Bucket / 路径类

### C1. 写 bucket 权限拒绝

**症状**：`mkdir /horizon-bucket/robot_lab/datasets/new_dir` → Permission denied。

**修复**：只能在 `/horizon-bucket/<bucket>/users/<你的 bucket 用户名>/` 下写。**你的 bucket 用户名 = SSO 用户名去掉 `-labs`**（`kun01.wu`，不是 `kun01.wu-labs`）。见 memory `[[kun-wu-bucket-workspace]]`。

### C2. Bucket 路径大小写 / 前后加 slash

**症状**：`ln -s /horizon-bucket/robot_lab/... target` 建立后 pod 里 target 不可读。

**修复**：
- bucket 名严格小写
- 结尾**别加** `/`（除非 target 也带 `/`）
- fuse 挂载有时对末尾 `/` 敏感

### C3. `ckpt/` symlink 找不到 Qwen VLM base

**症状**：`HoloBrainProcessor.load` 报 `ckpt/Qwen2.5-VL-3B-Instruct not found`。

**原因**：`HoloBrainProcessor.load` 会 `os.chdir(model_dir)`，之后走 `./ckpt/Qwen...` 相对路径解析。所以：
- **model_dir 里必须有 `ckpt/` symlink**（指向 `/horizon-bucket/.../xuewu.lin/ckpt`）
- **同时 `${WORKING_PATH}/ckpt` 也要有**（eval client cwd 是 WORKING_PATH，也会走相对路径）

**修复**：两处都建 symlink。见 `robodojo_pipeline/05_troubleshooting.md` §5.4-5.5。

### C4. Deploy package 少文件

**症状**：`_load_model` 报 `model.config.json not found` 或 `robodojo_processor.json not found`。

**修复**：Deploy package 必须包含（见 memory `[[holobrain-checkpoint-layouts]]`）：
- `model.safetensors`
- `model.config.json`
- `<sim>_processor.json`（如 `robodojo_processor.json`）
- `<sim>_inference.config.json`
- `ckpt/` symlink → Qwen VLM base
- `urdf/` 目录

---

## D. 日志 / 监控类

### D1. 训练 log 看不到 loss

**症状**：`aidictl job logs tail <id> log/<id>-task-1-main.log` 只有 setup 消息，没 GlobalStep。

**原因**：
- 你 tail 的可能不是 rank 0（loss 只在 rank 0 打）—— rank 0 通常在 task-1
- 或 accelerate mixed_precision + tqdm 干扰了 stdout buffer

**修复**：
```bash
JOB_ID=bcloud-bj-zone1-xxx
aidictl job logs ls $JOB_ID log     # 看所有 log 文件
aidictl job logs tail $JOB_ID log/$JOB_ID-task-1-main.log | grep -E "GlobalStep|loss" | tail -20
```

### D2. 找不到 output/ 目录

**症状**：`aidictl job logs ls <id> output` 报 `Not found`。

**原因**：job 还在 Running，output 只在 Succeeded/Failed 后归档。

**修复**：
- 用 `aidictl job status <id>` 看 phase
- Running 中要看产物，直接查 bucket（`/horizon-bucket/.../aidi_output/<...>`），前提是代码写到了 bucket 而非 /job_data

### D3. TensorBoard 打不开

**症状**：`aidictl job logs download <id> tboardlog/` 拉下来，但 tb 打不出图。

**原因**：TB event files 需要 `pip install tensorboard`（dev 侧）。

**修复**：
```bash
pip install tensorboard
tensorboard --logdir ~/tmp/tboardlog/ --port 6006
```

---

## E. Docker / Image 类

### E1. `docker push` 失败 401 Unauthorized

**症状**：`docker push docker.hobot.cc/imagesys/kun01.wu/...` → 401。

**修复**：
```bash
docker logout docker.hobot.cc
docker login docker.hobot.cc         # SSO 密码
docker push ...
```

### E2. Push 大 image 卡住

**症状**：`docker push` 一直 `Preparing`。

**修复**：
```bash
docker system prune -f
sudo systemctl restart docker
docker push ...
```

### E3. 镜像 tag 混乱

**建议**：tag 命名带日期 + 版本，如 `<base>-<yy-mm-dd>-v6`，从不覆盖旧 tag。若要给同一天迭代，用 `-v6a`、`-v6b`。

### E4. 集群 pod 拉 image 慢

**症状**：Running Phase 前长时间 hang，log 无输出。

**原因**：node 冷缓存拉 12+ GB image。

**修复**：无需处理，等 5-10 min。同 image 复用后续 job 会秒起。

---

## F. 集群队列 / 优先级

### F1. Queuing 太久

**症状**：submit 后 job 一直 Queuing。

**诊断**：
```bash
aidictl queue ls --type gpu -f "top=5"
# 看 project-5090-robot-lab-bcloud-bj 行：allocated / free / waiting
```

**修复**：
- 若 free>0 且 waiting=0，可能是 aidi scheduler 卡了，等
- 若 free=0 且 waiting>3，只能等；有能力就抢别的队列
- 若自己有多 Queuing job，`aidictl job stop` 掉次要的
- **加急**：`aidictl job urgent <job_id> -y`（priority=5 已是最高，加急 = 抬到 5）

### F2. 加急被拒或没效果

**规则**（见 [[aidi-cloud-submit]] §2.3）：
- 加急只对**资源 ≤ 4 机 8 卡**的 job 有效
- 已经 priority=5 的 job，加急无变化
- 加急是**抬优先级**，不改队列位置；上游资源不足仍要等

**加急没有「审批流程」**——是 CLI 一键操作。

---

## G. 通信 / 双 env 类

### G1. Policy server 起不来 timeout

**症状**：`wait_for_policy_server.sh` 报 `timeout 600s`。

**诊断**：
```bash
# pod 里
ss -ltnp | grep $PORT              # 端口是否 LISTEN
ps aux | grep setup_policy_server  # 进程是否活
tail policy_server.log             # 报错？
```

**常见原因**：
- Checkpoint 加载失败（路径错、大小不对）
- Import 报错（缺 dep）
- GPU 已被别的进程占（`nvidia-smi`）
- flash_attn 装了但版本不匹配 → HoloBrain v6 image 里有 `_patch_holobrain_vlm_attn_to_sdpa()` workaround

### G2. Client 报连不上 server

**症状**：`ConnectionRefused` on `ws://localhost:$PORT`。

**原因**：Server 死了 / 端口冲突 / 防火墙。

**修复**：
- 用 `get_free_port.sh` 动态分配端口（避免与其他 job 撞）
- 检查 `nc -z localhost $PORT` 是否通
- 检查 server 进程 `ps aux | grep setup_policy_server`

### G3. INFER 返回 shape 不对

**症状**：`EvalEnv.validate_action_dict` 报 `expected dim=6, got dim=X`。

**原因**：`deploy.yml` 里 `action_dim` 与 embodiment 不匹配。

**修复**：查 `env_cfg/robot/dual_x5.yml` 得到 arm_dim + ee_dim，改 `deploy.yml`。

---

## H. 你可能没想到的

### H1. `python_launcher: accelerate` + `num_workers=1` 也会加分布式参数

**注意**：只要 `python_launcher=accelerate`，即使 `num_workers=1` 也会拼 `accelerate launch --num_machines 1 --num-processes N`。若你只想 py3 直调，改 `python_launcher: python3`。

### H2. 提交后 dev 侧 `aidi_job_submit.json` 覆盖

每次 submit 都会**覆写** dev cwd 下的 `aidi_job_submit.json`。若你连续提交两个不同 cfg，只留最后一份。想保留就手动改名。

### H3. `job_password="1227"` 的由来

`accelerate --main_process_port 1227` 是硬编码（`submit_config.py:158`）。`job_password` 用 `1227` 只是**巧合的记忆点**，本身不必和 port 相同。见 job_password 字段说明。

### H4. 多个 conda env 同时激活会串环境变量

**别在同一进程里 `conda activate A; conda activate B`**。总在**新的子 shell** 里 activate，用括号 `( conda activate B; ... )` 或后台 `&` 子进程。

### H5. `to_upload` 的 rsync 会保留 mtime

若你 dev 侧改了文件但未重新 rsync 到 workspace_folder，`clear_workspace=false` 时不会覆盖 —— 因为 rsync 只更新 newer 的文件。设 `clear_workspace=true` 才彻底重来。

### H6. `set -e` 与后台进程的坑

评测 `cmd` 里用了 `set -euo pipefail` 但也起了 bg rsync：
```bash
(while true; do rsync ...; sleep 60; done) &
```
如果 bg 里某次 rsync 失败会不会中断主 shell？不会——bg 进程的 exit code 不影响父 shell。但若你在**前台**用 `rsync ... || true`，`set -e` 也不生效（`||` 兜底了）。这就是 `2>/dev/null || true` 的意义。

### H7. 训练完的 ckpt 不 export 就用不了

**易犯错**：训练完直接把 `/job_data/checkpoints/checkpoint_N/` 拷到 bucket 就想跑 eval → **不 work**。它是 accelerate state，只能 resume 训练。必须 export 出 deploy package。见 memory `[[holobrain-checkpoint-layouts]]` + `../09_export_and_eval.md`。

### H8. Bucket fuse 挂载的 IO 特性

- **顺序读**很快（十几百 MB/s）
- **随机小写**很慢（fuse 每次要走 metadata + 上传）
- **不适合放 lmdb 训练数据以外的高频写文件**（如 tmp / cache）
- 训练 tmp 用 `/dev/shm` 或 `/tmp`（pod 本地 SSD）

### H9. Job kill 后 bucket 上的临时文件不会自动清

**别忘手动清**：
```bash
ls -la /horizon-bucket/robot_lab/users/kun01.wu/aidi_output/*/  # 找没归档的临时目录
```

### H10. 同 user 多 job 抢 dev 侧 workspace_folder

**症状**：你和别人（或你自己两个 session）同时 submit 用同一 `workspace_folder` → 相互覆盖。

**修复**：每份 cfg 起独立 workspace_folder 名，如加时间戳后缀 `submit-holobrain-robodojo-eval-<yy-mm-dd>`。

### H11. `holobrain_internal` dev env 和镜像的 env 混淆

**别混淆**：
- Dev 上的 `~/miniconda3/envs/holobrain_internal` — **只有 submit 用**，含 `robo_orchard_jobs`
- 镜像里的 `/opt/miniconda3/envs/holobrain` — **训练/评测跑的**，含 torch + transformers + robo_orchard_lab
- 两者名字近但**用途完全不同**

### H12. `dataset_specs` 是配置字符串路径

`config_holobrain_common.py` 里的 `dataset_specs="configs/dataset_specs_robodojo.py"` 是**相对 pod 里 cwd 的路径**（cwd=WORKING_PATH），指向 `to_upload` 拉过来的 `configs/dataset_specs_robodojo.py`。改数据集就改这个 arg。

### H13. 单 pod 上 fork subprocess 会继承 CUDA 上下文

**症状**：policy server + env client 都跑起来后 `nvidia-smi` 显示两个 python 进程各在 GPU 0/1，但 GPU 0 上偶尔看到 env client 的 shadow。

**原因**：subprocess 若在 CUDA 初始化后 fork，会继承 CUDA context。

**修复**：`multiprocessing.set_start_method("spawn")`（HoloBrain train.py 已经这么做了）。

### H14. dev 上想复现 pod 内执行

**技巧**：
```bash
# submit 时把 execute 改成 false
"execute": false
# 会写 workspace_folder/{run.sh, run_local.sh}, 但不提交
# 你可以本地 docker run --gpus all ... <image> bash -c "cd /workspace && bash run.sh"
```

### H15. 集群 pod 时区

Pod 通常是 UTC。若你的 log 里日期看着比预期早 8h，是时区差 —— log timestamp 是 UTC，你 shell 是 CST。

### H16. `job_type` 影响调度器

`job_type=train`（GPU 场景）vs `prediction`（无 GPU）会走不同的 scheduler pool。**评测 job 也要写 `train`**（不是 `eval`），因为需要 GPU；`job_type=eval` 是给 CPU-only 后处理用的。源码见 `job_config.py:33-43` 枚举 + `:253` 自动映射。

### H17. AIDI console URL 拿法

```bash
aidictl job logs url <job_id>
# 输出 log / output / tboardlog 三个 URL，浏览器打开
```

---

## I. 心法 / 元规则

### I1. 「3-strike fatigue」

同类 error 修 3 次没成，**停下汇报**。判据：报错 top-line 一样，或修改文件相近。**不是**「修一个 import 报下一个 import」（那算前进）。见 `robodojo_pipeline/05_troubleshooting.md` §10。

### I2. 破坏性操作先看

- **删 workspace_folder**：确认没同事在共用
- **改 bucket 里的 ckpt**：影响正在 Running 的 eval！先复制到 `ckpt_v2/`
- **rebuild image**：tag 加 `v7`，从不覆盖 `v6`
- **`aidictl job stop`**：确认是你的 job，不是别人的

### I3. 每次 debug 结束扫僵尸 job

```bash
python3 <<'PY'
# 参见 [04_commands_cheatsheet §2]，list 自己 24h 内的 job，看有无重复 Running
PY
```

### I4. 别信训练 loss 曲线

Diffusion 去噪 loss ≠ rollout SR。用 eval 侧 `success_rate` 做真判据。

### I5. 用 memory 而非重记

Session 之间靠 `~/.claude/projects/-.../memory/*.md` 传递结论。项目习惯写进 `[[holobrain-aidi-submit-conventions]]` 之类，避免同一个坑重复踩。

---

## 相关文件

- `robodojo_pipeline/05_troubleshooting.md` — 一次任务里踩过的 15 类具体坑
- `robodojo_pipeline/04_commands_cheatsheet.md` — 所有命令模板
- `[[aidi-cloud-submit]]` skill §2.4 — AIDI SDK 三大陷阱详版
- `[[internal-docker]]` skill — 镜像迭代流程
- `[[robodojo-holobrain-eval-image-v6]]` memory — v6 image 完整配方
- `[[kun-wu-bucket-workspace]]` memory — bucket 目录结构
- `[[holobrain-checkpoint-layouts]]` memory — ckpt 两种形态
- `[[holobrain-aidi-submit-conventions]]` memory — 提交约定
