# 03 — submit_cfg.json 生命周期详解

**目标**：拿具体的 `submit_cfg.json`（这里以 `projects/holobrain_internal/common/aidi_submit_config/submit_cfg.json` 为例），完整回答：

1. 任务提交后进入哪个目录？
2. 什么时候激活哪个 conda env？
3. 到底跑了什么程序？
4. JSON 里每个字段映射到集群侧什么行为？
5. 我没写在 JSON 里的字段，默认值是什么？在哪查？

---

## 0. 参照 JSON

```json
// projects/holobrain_internal/common/aidi_submit_config/submit_cfg.json
{
    "job_name": "holobrain_alldata",
    "workspace_folder": "submit-holobrain",
    "docker_image": "docker.hobot.cc/imagesys/robotlab-mani:ubuntu2204-...-trasnformers5102",
    "input_bucket": "robot_lab,robot_lab2",
    "output_bucket": "robot_lab,robot_lab2",
    "num_workers": 2,
    "gpu_per_worker": 8,
    "wall_time": 14400,
    "cmd": [
        "ulimit -n 65536",
        "ln -s /horizon-bucket/robot_lab2/datasets/all_data ${WORKING_PATH}/data",
        "ln -s /horizon-bucket/robot_lab2/datasets/all_data/urdf/urdf_v20260711 ${WORKING_PATH}/urdf",
        "ln -s /horizon-bucket/robot_lab/users/xuewu.lin/ckpt ${WORKING_PATH}/ckpt",
        "export PYTHONPATH=${WORKING_PATH}/robo_orchard_lab:$PYTHONPATH"
    ],
    "python_launcher": "accelerate",
    "python_executable": "train.py --workspace /job_data --logging_dir /job_tboard --config configs/config_holobrain_common.py",
    "to_upload": [
        "robo_orchard_lab",
        "projects/holobrain_internal/common/configs",
        "projects/holobrain_internal/common/train.py",
        "projects/holobrain_internal/common/holobrain_utils.py"
    ],
    "job_password": "1227",
    "queue_name": "project-5090-robot-lab-bcloud-bj",
    "project_id": "horizon-labs"
}
```

---

## 1. 目录结构：Dev 侧 vs Pod 侧

### Dev 侧（提交时的 cwd = repo root）

```
~/git_repo/robo_orchard_lab/                    ← cwd, submit_from_config 在这跑
├── robo_orchard_lab/                          ← ✓ 在 to_upload
├── projects/holobrain_internal/common/
│   ├── configs/                               ← ✓ 在 to_upload
│   ├── train.py                               ← ✓ 在 to_upload
│   └── holobrain_utils.py                     ← ✓ 在 to_upload
├── aidi_job_submit.json                       ← ⚡ submit 时生成的快照
└── submit-holobrain/                          ← ⚡ workspace_folder, submit 时生成
    ├── robo_orchard_lab/                      ← rsync -aL 拷贝进来
    ├── configs/
    ├── train.py
    ├── holobrain_utils.py
    ├── get_rank.py                            ← 分布式工具（robo_orchard_jobs 自动拷）
    ├── ssh_launcher.py
    ├── url2IP.py
    ├── job_config.yaml                        ← ⚡ jinja 生成，给 aidisdk 吃
    ├── run.sh                                 ← ⚡ jinja 生成，pod 入口
    └── run_local.sh                           ← ⚡ jinja 生成，run.sh 调这个
```

**注意 `to_upload` 里的相对路径**：都是**相对 dev 侧 cwd** 解析，而 cwd 是 repo root（不是 JSON 所在目录）。所以你提交时必须 `cd ~/git_repo/robo_orchard_lab`，否则 `robo_orchard_lab/` 找不到 → `FileNotFoundError`。

**注意 rsync 时的目录扁平化**：`to_upload: ["projects/holobrain_internal/common/configs"]` 会把 `configs/` **直接放到** `workspace_folder/configs/`（不是 `workspace_folder/projects/holobrain_internal/common/configs/`）。这就是为什么 `python_executable` 里写 `--config configs/config_holobrain_common.py` 而不是 `--config projects/holobrain_internal/common/configs/config_holobrain_common.py`。

### Pod 侧

```
$WORKING_PATH = /running_package/code_package         ← 上面 workspace 解压到这里
├── robo_orchard_lab/     configs/     train.py       ← 同 dev
├── holobrain_utils.py    get_rank.py  ssh_launcher.py  url2IP.py
├── job_config.yaml       run.sh       run_local.sh
├── data          ← ⚡ `cmd` 里 ln -s /horizon-bucket/... 建的软链
├── urdf          ← 同
└── ckpt          ← 同

/job_data/                    ← AIDI 分配的产物目录（会被归档到 output/）
    checkpoints/
        checkpoint_0/  checkpoint_1/  ...

/job_tboard/                  ← TensorBoard（归档到 tboardlog/）

/horizon-bucket/robot_lab/   /horizon-bucket/robot_lab2/    ← fuse 挂载
```

---

## 2. Pod 里发生了什么（逐行拆解）

pod 拉完镜像、mount bucket、解压 tar 后，AIDI 执行：

```bash
bash $WORKING_PATH/run.sh
```

### 2.1 run.sh（多 pod 分支，num_workers=2）

由 `robo_orchard_jobs/job_submit/distributed/jinja2_templates/run.sh.jinja2` 渲染：

```bash
set -e
cd ${WORKING_PATH}                                # cwd = /running_package/code_package
python3 url2IP.py                                 # 解 hostname → IP，落 /job_data/mpi_hosts
python3 ssh_launcher.py --monitor --nworker 2 \
    --ngpus 8 -H /job_data/mpi_hosts \
    'bash run_local.sh'                           # 通过 ssh 到每个 pod 起 run_local.sh
```

`ssh_launcher.py` 起来后每个 pod（含 rank 0 自己）都会执行 `bash run_local.sh`。

### 2.2 run_local.sh（accelerate 分支）

由 `run_local.sh.jinja2` 渲染：

```bash
set -e
python3 get_rank.py --launcher accelerate         # 生成 ./rank 文件
NODE_INFO=`cat ./rank`                            # e.g. "--machine_rank 0 --main_process_ip 10.x.x.x"
echo "NODE_INFO: ", $NODE_INFO

# 以下由 generate_cmd_str() 拼接，来自你 JSON 的 cmd + python_executable：

ulimit -n 65536
ln -s /horizon-bucket/robot_lab2/datasets/all_data ${WORKING_PATH}/data
ln -s /horizon-bucket/robot_lab2/datasets/all_data/urdf/urdf_v20260711 ${WORKING_PATH}/urdf
ln -s /horizon-bucket/robot_lab/users/xuewu.lin/ckpt ${WORKING_PATH}/ckpt
export PYTHONPATH=${WORKING_PATH}/robo_orchard_lab:$PYTHONPATH

 accelerate launch  --num_machines 2 --num-processes 16  --multi-gpu --gpu-ids 0,1,2,3,4,5,6,7  $NODE_INFO --main_process_port 1227 train.py --workspace /job_data --logging_dir /job_tboard --config configs/config_holobrain_common.py
```

**几个关键实施细节**（源码 `robo_orchard_jobs/job_submit/submit_config.py:121-162`）：

- `accelerate launch` 前的空格是源码 f-string 拼出来的（无害）
- `--gpu-ids 0,1,...,7` 数量 = `gpu_per_worker` 值
- `--num-processes 16` = `gpu_per_worker × num_workers`
- 只有 `num_workers > 1` 才追加 `$NODE_INFO --main_process_port 1227`
- 只有 `gpu_per_worker > 1` 才加 `--multi-gpu`
- **`--main_process_port` 硬编码 1227**（源码 line 158），如果与你的 firewall/其他服务冲突需要 patch 源码

### 2.3 conda env 什么时候激活？

**这个 submit_cfg 里根本没激活任何 conda env！**

因为：
- 训练镜像 `robotlab-mani:...` 的 **`PATH` 已经把 `accelerate` / `python` 指向了目标 env**（镜像 baked）
- 或者 `/opt/miniconda3/bin/python` 是默认 python
- 训练不需要在多 env 之间切

**评测的 submit_cfg 就不一样了**（见 [04_dual_env_client_server.md](04_dual_env_client_server.md)）：需要显式
```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate RoboDojo
```

**判断规则**：镜像里如果只有一个 conda env（或 base 就够用），不用激活；有多个 env 且要切换，必须 `source /opt/miniconda3/etc/profile.d/conda.sh` 再 `conda activate <name>`。

### 2.4 到底跑了什么程序？

对本示例：`accelerate launch <accelerate args> train.py --workspace /job_data ...`

即 `robo_orchard_lab/projects/holobrain_internal/common/train.py`（在 pod 里叫 `${WORKING_PATH}/train.py`）。

`train.py` 的 main（简化）：
```python
accelerator = Accelerator(...)          # 拉起分布式
config = load_config(args.config, kwargs=args.kwargs)
dataset = build_dataset(config.dataset_specs)
model   = build_model(config)
trainer = SimpleTrainer(
    model=model, dataloader=..., optimizer=..., scheduler=...,
    hooks=[StatsMonitor, LossMovingAverageTracker, SaveCheckpoint(total_limit=3)],
    max_step=config.max_step,
    workspace=args.workspace,           # /job_data
)
trainer.fit()
```

产物：
- `/job_data/checkpoints/checkpoint_N/` — accelerate state
- `/job_tboard/` — TB scalars
- stdout → `output/<job_id>-task-<n>-main.log`（AIDI 归档）

---

## 3. 字段 → 行为 一一对照表

| JSON 字段 | pydantic 默认值 | 集群侧作用 |
|---|---|---|
| `job_name` | **required** | AIDI 显示名；实际 job_id 会被 aidisdk 附上 `_<uuid>` |
| `workspace_folder` | `"workspace_to_submit"` | dev 侧本地目录名；pod 侧 = `$WORKING_PATH` |
| `clear_workspace` | `false` | true = submit 前先 `rm -rf workspace_folder/`（避免 stale） |
| `docker_image` | **required** | AIDI 拉这个 image 起 pod |
| `input_bucket` | `None` | 逗号分隔或 list，fuse 挂载到 pod |
| `output_bucket` | `None` | 同上，用于**可写**的 bucket（源码约束「output 应与 input 不同」但实操两者相同也行） |
| `num_workers` | `1` | pod 数（跨机） |
| `gpu_per_worker` | `0` | 每 pod GPU 数，0 → JOB_TYPE 自动变 "prediction"；上限 8 |
| `cpu_per_worker` | `8` | 每 pod CPU 数，1-24 |
| `cpu_mem_ratio` | `8` | 内存 = CPU 数 × ratio (GB)，1-16 |
| `wall_time` | `7200` | **分钟**（源码 typo "minitus"），到点 SIGTERM |
| `python_launcher` | `"python3"` | `"accelerate"` 时自动拼 accelerate launch 命令；`"python3"` 时把 python_executable 直接接在 `python3 ` 后 |
| `python_executable` | `None` | 跟在 launcher 后的整串（`train.py --arg1 ...`） |
| `cmd` | `None` | str 或 list[str]，会**先于** python_executable 逐行写入 run_local.sh |
| `to_upload` | `None` | 每项 rsync -aL 到 workspace_folder（**follow_symlinks=true**，坑见 05） |
| `queue_name` | **required** | 决定集群 / 项目 / 机型 |
| `project_id` | **required** | 计费单位；`horizon-labs` 是本项目常用；混错 → 403 |
| `job_password` | `"aidi_job_passwd"` | 加密 tar 密码；本项目习惯设 `"1227"` |
| `priority` | `5` | 1-5，5=最高（默认）；改小让位其他 job |
| `use_aidisdk` | `true` | 走 `aidisdk_job_submit`；false 走老 `aidi-inf-cli job submit` |
| `execute` | `true` | false = 只准备本地 workspace，不真提交（dry-run 用） |
| `job_type` | `"train"` | AIDI 侧 job 分类：debug / packing / **train** / prelabel / rl / prediction / filter / eval / data-process |

**字段来源**：`robo_orchard_jobs/job_submit/aidi/job_config.py::JobSubmitParamForAIDI`（继承 `submit_config.py::JobSubmitParams`）。

### 3.1 查默认值的方法

**方式 A：读 pydantic model 源码**
```bash
python -c "
from robo_orchard_jobs.job_submit.aidi.job_config import JobSubmitParamForAIDI
for name, field in JobSubmitParamForAIDI.model_fields.items():
    print(f'{name:20s} default={field.default!r:20s} desc={field.description}')"
```

（要在 `holobrain_internal` env 下跑）

**方式 B：跑一次 `--execute false` 看生成的 job_config.yaml**
```bash
# 临时把 execute 改成 false，重跑 submit_from_config：
# 会生成 workspace_folder/job_config.yaml，里面所有字段都被显式填充（默认值 + 你 override 的值）
cat submit-holobrain/job_config.yaml
```

**方式 C：读 `aidi_job_submit.json`**
每次 submit 会在 cwd 写一份 `aidi_job_submit.json` 快照，用 `exclude_none=True` 打印**所有你设了的字段**（默认字段不出现 → 找不到就是走默认）。

**方式 D：读源码路径**
`~/miniconda3/envs/holobrain_internal/lib/python3.11/site-packages/robo_orchard_jobs/job_submit/{aidi/job_config.py, submit_config.py}` —— 权威真相。

### 3.2 关键默认值一览（照顾懒得跑上面命令的场景）

```
job_password         "aidi_job_passwd"      # 建议自己改一个（本项目习惯 "1227"）
num_workers          1
gpu_per_worker       0                       # !! 默认无 GPU，训练/评测必须显式设
cpu_per_worker       8
cpu_mem_ratio        8                       # → 内存 = 8 × 8 = 64 GB / pod
wall_time            7200                    # 分钟 = 120 h = 5 天
workspace_folder     "workspace_to_submit"
clear_workspace      false                   # !! 建议每份 cfg 都写 true
python_launcher      "python3"
priority             5
use_aidisdk          true
execute              true
job_type             "train"
```

---

## 4. python_launcher 展开表

`generate_cmd_str()` 逻辑（`submit_config.py:121-162`）：

### 4.1 `python_launcher = "python3"`

```bash
{cmd 每行}
python3 {python_executable}
```

即 `python3 train.py --arg1 ...`。**单进程**。若要多进程要自己在 `python_executable` 里写 `-m torch.distributed.launch ...`。

### 4.2 `python_launcher = "accelerate"`

```bash
{cmd 每行}
accelerate launch  --num_machines {num_workers} --num-processes {gpu_per_worker × num_workers} \
  [--multi-gpu if >1 process] \
  [--gpu-ids 0,1,...,{gpu_per_worker-1} if gpu_per_worker > 0] \
  [$NODE_INFO --main_process_port 1227 if num_workers > 1] \
  {python_executable}
```

**具体例子**：`num_workers=2, gpu_per_worker=8, python_executable="train.py ..."`
→ `accelerate launch --num_machines 2 --num-processes 16 --multi-gpu --gpu-ids 0,1,2,3,4,5,6,7 $NODE_INFO --main_process_port 1227 train.py ...`

**注意**：
- port 1227 是硬编码（`submit_config.py:158`），也是 `job_password` 常用值的来源
- `$NODE_INFO` 在 pod 里由 `get_rank.py` 生成

### 4.3 什么时候用哪个

| 场景 | launcher | 原因 |
|---|---|---|
| 单机单卡 debug | python3 | 不需要 accelerate |
| 训练（多机多卡） | accelerate | 自动分布式 |
| 评测（policy server + env client 各占 1 卡） | python3 + cmd 里手写 | 评测用 `multiprocessing.Process` 分卡，非 DDP |
| 数据处理 job | python3 | 通常单进程 |

---

## 5. 完整生命周期示意（对着本 JSON）

```
t=0  DEV: cd ~/git_repo/robo_orchard_lab
        RoboOrchardJob-AIDISubmit submit_from_config --config .../submit_cfg.json
     │
t=1  DEV: JobSubmitParamForAIDI 读 JSON
        - job_name="holobrain_alldata"
        - workspace_folder="submit-holobrain"
        - to_upload=[...4 items...]
        - python_launcher="accelerate", python_executable="train.py ..."
        - 默认 clear_workspace=false → 若已存在则 pile up ⚠️
     │
t=2  DEV: 写 aidi_job_submit.json 快照
     │
t=3  DEV: prepare_workspace()
        - mkdir submit-holobrain/
        - rsync -aL robo_orchard_lab/  → submit-holobrain/robo_orchard_lab/
        - rsync -aL projects/holobrain_internal/common/configs/ → submit-holobrain/configs/
        - rsync -aL projects/holobrain_internal/common/train.py → submit-holobrain/train.py
        - rsync -aL projects/holobrain_internal/common/holobrain_utils.py → submit-holobrain/
     │
t=4  DEV: rsync 3 个 dist utils → submit-holobrain/{get_rank,ssh_launcher,url2IP}.py
     │
t=5  DEV: 写 submit-holobrain/job_config.yaml（YAML 版参数）
     │
t=6  DEV: jinja render → submit-holobrain/run.sh, run_local.sh
     │
t=7  DEV: subprocess.check_call([
          "RoboOrchardJob-AIDISubmit", "aidisdk_job_submit",
          "--job_config_path", "submit-holobrain/job_config.yaml",
          "--queue_name", "project-5090-robot-lab-bcloud-bj",
          "--job_type", "train",
        ])
        → aidisdk 内部：tar submit-holobrain/ + 加密（password="1227"）
        → upload to OSS
        → POST http://computing.aidi.hobot.cc/.../job/create
        → 收 200 return job_id=bcloud-bj-zone1-<xxx>  ⚠️ 日志被吞
     │
t=8  DEV: subprocess return 0 → logger.info("Command executed: ...") → return
     │
t=9  AIDI: job Queuing → Running（可能几 min 到几 h）
     │
t=10 AIDI: 分配 2 pod × 8 × RTX 5090, mount robot_lab + robot_lab2
     │
t=11 AIDI: 拉 docker.hobot.cc/imagesys/robotlab-mani:...trasnformers5102（2-10 min）
     │
t=12 POD: 解密 tar → /running_package/code_package/
     │
t=13 POD: env WORKING_PATH=/running_package/code_package bash run.sh
        run.sh 逻辑（num_workers=2 分支）：
          - cd /running_package/code_package
          - python3 url2IP.py                          # → /job_data/mpi_hosts
          - python3 ssh_launcher.py --monitor --nworker 2 --ngpus 8 \
              -H /job_data/mpi_hosts 'bash run_local.sh'
                → 到每个 pod (含自己) 起 run_local.sh
     │
t=14 POD (each): bash run_local.sh
        - python3 get_rank.py --launcher accelerate   # 生成 ./rank
        - NODE_INFO=$(cat ./rank)
        - ulimit -n 65536
        - ln -s /horizon-bucket/robot_lab2/datasets/all_data ${WORKING_PATH}/data
        - ln -s ... urdf, ckpt
        - export PYTHONPATH=${WORKING_PATH}/robo_orchard_lab:$PYTHONPATH
        - accelerate launch --num_machines 2 --num-processes 16 --multi-gpu \
             --gpu-ids 0,1,...,7 $NODE_INFO --main_process_port 1227 \
             train.py --workspace /job_data --logging_dir /job_tboard \
                      --config configs/config_holobrain_common.py
     │
t=15 POD: train.py 主循环，每 save_step_freq 步写 /job_data/checkpoints/checkpoint_N/
     │
t=16 POD: 达 max_step 正常退出，或 wall_time=14400 min（240 h）SIGTERM
     │
t=17 AIDI: /job_data → output/; /job_tboard → tboardlog/; stdout → log/
        Phase → Succeeded (or Failed)
     │
t=18 DEV: aidictl job logs download <job_id> output/ --dest ...
        或直接从 bucket 读（如果代码是写到 /horizon-bucket/...）
```

---

## 6. 常见问题

### Q: 我改了 `configs/config_holobrain_common.py`，但集群跑的还是旧版
- 你确定 `to_upload` 里有 `projects/holobrain_internal/common/configs` 吗？
- 你确定提交时 cwd 是 repo root 吗？
- 检查 `submit-holobrain/configs/config_holobrain_common.py` 是不是最新的（rsync 时应该已更新）
- 检查 `docker_image` 里有没有旧版被 bake 进去（一般不会，但 `configs/` 若曾经进过 image 就会覆盖你的 upload）

### Q: 我加了个 python 依赖，怎么弄进 pod？
- **不要**改 `cmd` 里加 `pip install`（每 job 都重新装，慢，且 mirror 不稳）
- 正确做法：起 dev container → `pip install` → `docker commit + push` → 新 image tag → 改 `docker_image`
- 见 [[internal-docker]] skill

### Q: 集群里 cwd 是哪里？
- `run.sh` 里 `cd ${WORKING_PATH}` 后 = `/running_package/code_package`
- 但你在 python 里如果 `os.chdir(...)` 或 `subprocess(cwd=...)` 会变
- HoloBrain `train.py` 靠 accelerate 提供的 cwd，写产物用绝对路径 `/job_data`

### Q: pod 里的 python 版本？
- 由 `docker_image` 决定，与镜像里 `/opt/miniconda3` 或系统 python 一致
- 训练镜像 `robotlab-mani` 默认 py3.11（跟 dev 侧 holobrain_internal 一致）
- 评测镜像 `robodojo-holobrain-v6` 有两个 env，分别 py3.11 (holobrain) 和 py3.10 (RoboDojo)

### Q: `${WORKING_PATH}` 什么时候展开？
- **在 pod 的 shell 里**，不是 dev 侧。所以你 JSON 里写 `${WORKING_PATH}` 是原样传给 pod 的 shell，pod shell 展开。
- 若你 dev 侧本地 shell 就 `echo` 这个字符串会展开为空 —— 因为 dev shell 没有 `WORKING_PATH`。

### Q: `python_executable` 里能放 shell 语法（`&&`, `$VAR` 展开）吗？
- 可以，最终会被拼进 `bash run_local.sh` 里，是 bash 环境
- 但注意 accelerate 分支下 `python_executable` 会被 accelerate launch 后直接执行，是**单进程 python 命令**；如果你要多命令，用 `cmd` 而不是塞 `&&` 进 `python_executable`

---

下一篇 [04_dual_env_client_server.md](04_dual_env_client_server.md) 讲评测里怎么在一个 pod 起两个 conda env。
