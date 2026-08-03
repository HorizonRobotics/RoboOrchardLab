# 01 — 训练侧完整通路

从 AIDI submit 到 `/job_data/checkpoints/checkpoint_N/` 落盘，逐层拆解。

**默认 job**：`bcloud-bj-zone1-6c6f0a3cbcb9`（100k step 训练，v9 warm-start，2 pod × 8 GPU × batch 16）

> **⚠️ 2026-08-03 合并 `feature/sem_internal` 之后，有两处变更会影响本文的准确性**：
> ① 仓库默认配置已从 v9 切到 v10（VLM 换成 Qwen3.5-2B、`patch_size` 28→32），本文所有
> `config_holobrain_common.py:<行号>` 引用都已漂移，对照表见
> [`../04_config_system.md`](../04_config_system.md) 顶部；
> ② robodojo 的 processor 导出名改成了 `robodojo_arx_x5a_processor`，见
> [02_deploy_package.md](02_deploy_package.md) 顶部。
> **本文记录的仍是 v9 那次的实况，没有改写。**

---

## 1. AIDI 提交端

### 1.1 submit_cfg_robodojo_train_100k.json

**文件**：`projects/holobrain_internal/common/aidi_submit_config/submit_cfg_robodojo_train_100k.json`

| 字段 | 值 | 作用 |
|---|---|---|
| `job_name` | `holobrain_robodojo_posttrain_v9_100k` | AIDI job name |
| `workspace_folder` | `submit-holobrain-robodojo-100k` | AIDI 会先把 to_upload 拷到这个本地目录 |
| `clear_workspace` | `true` | 每次提交前先清空这个 workspace |
| `docker_image` | `docker.hobot.cc/imagesys/robotlab-mani:ubuntu2204-gcc11.4-cu128-nccl2277-torch280-erdma-trasnformers5102` | 训练镜像（有 typo `trasnformers` 别改）|
| `input_bucket` | `robot_lab,robot_lab2` | RW mount 两个 bucket |
| `output_bucket` | `robot_lab` | 结果落 `robot_lab` |
| `num_workers` | `2` | 2 pod（分布式训练 world_size=16） |
| `gpu_per_worker` | `8` | 每 pod 8 卡 |
| `wall_time` | `14400` | 分钟（240h，训练留有余量） |
| `queue_name` | `project-5090-robot-lab-bcloud-bj` | 5090 集群 |
| `project_id` | `horizon-labs` | 计费 |
| `to_upload` | `["robo_orchard_lab", "projects/holobrain_internal/common", "projects/holobrain_internal/common/aidi_submit_config"]` | 会打包成 `/running_package/code_package/` 里的对应子目录 |
| `cmd` | 见下方 | 集群侧真正的执行脚本 |

### 1.2 cmd 关键段（run.sh）

集群侧 pod 上，AIDI 会创建 `run.sh`（内容 = submit_cfg 的 `cmd` 字段），实测生成为：

```bash
# submit-holobrain-robodojo-100k/run.sh  L1-9
set -euo pipefail
cd /running_package/code_package
ln -sfn /horizon-bucket/robot_lab2/datasets/all_data/robodojo ./data/robodojo   # LMDB 数据入口
ln -sfn /horizon-bucket/robot_lab2/datasets/all_data/urdf/urdf_v20260711 ./urdf # URDF 入口
export PYTHONPATH=/running_package/code_package:$PYTHONPATH
accelerate launch --config_file projects/holobrain_internal/common/aidi_submit_config/accelerate_multi_node.yaml \
  projects/holobrain_internal/common/train.py \
  --config projects/holobrain_internal/common/configs/config_holobrain_common.py:v9 \
  --data_config projects/holobrain_internal/common/configs/data_configs/config_robodojo_dataset.py \
  --dataset_specs projects/holobrain_internal/common/configs/dataset_specs_robodojo.py \
  --workspace /job_data --tboard_dir /job_tboard \
  --max_step 100000 --save_step_freq 5000 --with_depth_loss false --seed 0
```

### 1.3 提交命令

```bash
cd /home/users/kun01.wu-labs/git_repo/robo_orchard_lab
RoboOrchardJob-AIDISubmit submit_from_config \
    --config projects/holobrain_internal/common/aidi_submit_config/submit_cfg_robodojo_train_100k.json
```

**注意**：`RoboOrchardJob-AIDISubmit submit_from_config` 的 stdout 会**吃掉 job_id**（见 [[../../CLAUDE.md]] / skill `aidi-cloud-submit` §2.4），要立刻反查用：

```bash
python3 <<'PY'
import requests
token = open("/home/users/kun01.wu-labs/.aidisdk/config.yaml").read().split("token:")[1].split("\n")[0].strip()
r = requests.get("http://computing.aidi.hobot.cc/infra/api/v1alpha/computing-apiserver/job/list",
    headers={"Authorization": token}, params={"limit": 20, "user_name": "kun01.wu-labs"})
for j in r.json()["data"]["list"][:5]:
    print(j["job_id"], j["job_name"])
PY
```

---

## 2. 训练入口 `train.py`

**文件**：`projects/holobrain_internal/common/train.py`

### 2.1 main 骨架

```python
# train.py:59-241 (骨架)
def main():
    args = parse_args()                                                        # L59
    accelerator = build_accelerator(args)                                      # L228

    cfg = load_config(args.config)                                             # L72 → holobrain_utils.load_config
    data_cfg = load_config(args.data_config)
    dataset_specs = load_config(args.dataset_specs)

    train_dataset = build_training_dataset(cfg, data_cfg, dataset_specs)      # L199-233 in dataset_factory.py
    train_loader  = build_dataloader(train_dataset, batch_size=cfg.batch_size,
                                     num_workers=cfg.num_workers, collate=collate_batch_dict)
    model = cfg.build_model()                                                  # L104 in config_holobrain_common.py
    optimizer, scheduler = cfg.build_optimizer(model)                          # L460-510

    if args.load_ckpt or cfg.vlm_pretrain:
        load_checkpoint(model, args.load_ckpt or cfg.vlm_pretrain)             # L127-158 in holobrain_utils.py

    trainer = SimpleTrainer(                                                   # L174-204
        model=model, dataloader=train_loader,
        optimizer=optimizer, scheduler=scheduler,
        accelerator=accelerator,
        batch_processor=MyBatchProcessor(),                                    # L49-56
        hooks=[SaveCheckpoint(save_step_freq=args.save_step_freq, total_limit=3),
               LossTracker(...), StatsTracker(...)],
        workspace=args.workspace, tboard_dir=args.tboard_dir,
        max_step=args.max_step, grad_clip_norm=10,                             # L180-181
    )
    trainer()                                                                  # calls HookBasedTrainer.__call__
```

### 2.2 MyBatchProcessor

**位置**：`projects/holobrain_internal/common/train.py:49-56`

```python
class MyBatchProcessor(SimpleBatchProcessor):
    def forward_backward(self, model, batch, accelerator):
        model_outs = model(batch)                # HoloBrain_Qwen2_5_VL.forward
        loss_dict  = model.loss(model_outs, batch)  # HoloBrainActionLoss
        total_loss = sum(loss_dict.values())
        accelerator.backward(total_loss)
        return {"total_loss": total_loss, **loss_dict}
```

这个 `loss_dict` 就是 log 里看到的六个 loss key 的来源。

### 2.3 Accelerator 构造

**位置**：`projects/holobrain_internal/common/train.py:228-241`

- 从 `--workspace=/job_data` 决定 checkpoint 保存根目录
- multi-node config `projects/holobrain_internal/common/aidi_submit_config/accelerate_multi_node.yaml`：
  - `distributed_type: MULTI_GPU`
  - `mixed_precision: bf16`
  - `num_processes: 16` (2 pod × 8 GPU, AIDI 会填 env vars)

---

## 3. Config 加载：v9 override

**文件**：`projects/holobrain_internal/common/configs/config_holobrain_common.py`

### 3.1 base config

```python
# config_holobrain_common.py:19-44
class ConfigHolobrainCommon:
    hist_steps = 1                       # L20
    pred_steps = 64                      # L21
    chunk_size = 4                       # L22
    state_dims = 8                       # L165 → [joint_val, x, y, z, qw, qx, qy, qz]
    batch_size = 16                      # per-GPU
    num_workers = 8
    base_lr = 1e-4                       # L34
    ...
```

### 3.2 v9 override（100k 训练用的）

```python
# config_holobrain_common.py:85-94
class ConfigHolobrainV9(ConfigHolobrainCommon):
    embed_dim = 384                      # L87 (vs 512 for v10+)
    num_decoder_layers = 10              # L88
    vlm_pretrain = "https://..../holobrain_pretrain_v9/checkpoint_50/model.safetensors"  # L89
    # 加载时 holobrain_utils.load_checkpoint 会 filelock 下载到 cache
```

### 3.3 build_model 与 build_optimizer

```python
# config_holobrain_common.py:104
def build_model(self):
    from robo_orchard_lab.models.holobrain.structure import HoloBrain_Qwen2_5_VL
    return HoloBrain_Qwen2_5_VL(
        vlm_pretrain=self.vlm_pretrain,
        embed_dim=self.embed_dim,
        num_decoder_layers=self.num_decoder_layers,
        data_preprocessor=..., backbone_3d=..., action_decoder=...,
    )

# config_holobrain_common.py:460-510
def build_optimizer(self, model):
    # 3 param groups:
    # group 0: VLM (vlm.*), lr = base_lr = 1e-4
    # group 1: main body, lr = base_lr = 1e-4
    # group 2: pretrained VLM head, lr = base_lr * 0.1 = 1e-5
    # warmup: 500 iters, 0.001× → 1×
    # decay: ×0.1 at step 90000
```

---

## 4. Dataset 侧（RoboDojo LMDB）

### 4.1 dataset_specs

**文件**：`projects/holobrain_internal/common/configs/dataset_specs_robodojo.py`

```python
# dataset_specs_robodojo.py:34, 55-71
DATA_BASE = "./data/robodojo"                       # → /horizon-bucket/robot_lab2/datasets/all_data/robodojo
TRAINING_DATASETS = [
    {"ref": "config_robodojo_dataset.py:dataset_config",
     "kwargs": {"embodiment": "arx_x5a",
                "lmdb_glob": f"{DATA_BASE}/lmdb/*"}},   # L60-63
]
```

### 4.2 RoboDojoLmdbDataset

**文件**：`robo_orchard_lab/dataset/robodojo/robodojo_lmdb_dataset.py`

```python
# robodojo_lmdb_dataset.py:151-252
def __getitem__(self, idx):
    sample = self._read_lmdb(idx)
    return {
        # 3 cams × float32 HxWx3 (unresized here)
        "cam_left_wrist":  ndarray(H,W,3) uint8,
        "cam_right_wrist": ndarray(H,W,3) uint8,
        "cam_head":        ndarray(H,W,3) uint8,
        # No depth: RoboDojo LMDB has no depth channel
        "instruction":     str,                         # from L140-149 (random 1 of N)
        "hist_joint":      ndarray(1, 14) float32,      # hist_steps=1, 14 joints (7 per arm, gripper=joint 6/13)
        "future_joint":    ndarray(64, 14) float32,     # pred_steps=64
        "hist_ee_pose":    ndarray(1, 2, 7),            # per-arm [x,y,z,qw,qx,qy,qz]
        "future_ee_pose":  ndarray(64, 2, 7),
    }
```

### 4.3 build_transforms 主要 pipeline

**文件**：`projects/holobrain_internal/common/configs/data_configs/config_robodojo_dataset.py:82-287`

```python
Compose([
    AddItems(...),                         # L316-370 in transforms.py (加入 URDF/kinematics 元信息)
    SimpleStateSampling(...),              # 从 hist+future 拼采样窗口
    Resize(size=(W=308, H=252)),           # L563-611
    ToTensor(),                            # L637-648
    ConvertDataType(image=torch.bfloat16), # L651-673
    MultiArmKinematics(urdf=...),          # L687-980 (跑 FK, 生成 joint_relative_pos + robot_state[14,8])
    ItemSelection(keys=[...]),             # L676-684
])
```

- **Resize 后**：3 cam × `(3, 252, 308)` bf16。Qwen2.5-VL patch_size=28 → 9×11 grid = **99 image tokens/cam**
- **`joint_state_to_robot_state`** (`transforms.py:982-1006`) 输出 `[..., 14, 8]`：每个 joint 8 维 = `[joint_value, xyz(3), quat(4)]`（by FK）

### 4.4 batch 后 collate

**文件**：`robo_orchard_lab/dataset/collates.py:38-65`

```python
def collate_batch_dict(batch: list[dict]) -> dict:
    # 每个 key 独立 stack；非 tensor 保持 list
    return {"cam_head":   torch.stack([b["cam_head"] for b in batch]),    # [B,3,252,308] bf16
            "cam_left_wrist":  ...,
            "cam_right_wrist": ...,
            "instruction": [b["instruction"] for b in batch],              # list[str]
            "robot_state":     torch.stack(...),                            # [B, hist_steps=1, 14, 8]
            "future_joint":    torch.stack(...),                            # [B, 64, 14]
            "joint_relative_pos": torch.stack(...),                         # [14, 14]  static per-URDF
            "joint_mask":      torch.stack(...),                            # [B, 14]
            ...}
```

---

## 5. Model forward — HoloBrain_Qwen2_5_VL

**文件**：`robo_orchard_lab/models/holobrain/structure.py`

### 5.1 类 & forward 入口

```python
# structure.py:119-205, 225-232
class HoloBrain_Qwen2_5_VL(nn.Module):
    def __init__(self, vlm_pretrain, embed_dim=384, num_decoder_layers=10, ...):
        self.vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(...)   # bf16, flash_attn_2
        # LM 裁到前 4 层
        self.vlm.language_model.layers = self.vlm.language_model.layers[:4]  # L150 附近
        self.feat_map = nn.Linear(vlm_hidden, embed_dim)                     # L155
        self.text_template = TextTemplate(...)                                # L51-116
        self.action_decoder = HoloBrainActionDecoder(embed_dim=embed_dim, ...)

    def forward(self, inputs):                    # L225-232
        image_feature, text_dict = self._forward_vlm(inputs)                 # L467-489
        feature_3d = self.extract_feature_3d(image_feature, inputs)          # L249-268
        return {"feature_maps": image_feature,
                "feature_3d":   feature_3d,
                "text_dict":    text_dict}

    def loss(self, model_outs, inputs):           # L234-239
        return self.action_decoder.loss(model_outs, inputs)
```

### 5.2 VLM 输出的 reshape

```python
# structure.py:292-346 _vlm_outputs_handler
# 输入：VLM hidden [B, seq_len, hidden]
# 拆 image tokens vs text tokens
# image tokens → reshape 到 [B, N_cams=3, C=384, h=9, w=11]  ← 这就是 3 视角特征图
# text tokens  → text_dict{"embed": [B, T, 384], "mask": [B, T]}
```

### 5.3 3D 特征提取

```python
# structure.py:249-268 extract_feature_3d
# 输入：image_feature [B, N_cams, 384, 9, 11] + inputs["cam_intrinsic"/"cam_extrinsic"]
# 走 Swin 3D backbone (config_holobrain_common.py:247-268)
# + DepthFusionSpatialEnhancer (spatial_enhancer.py:170-255)
# 输出：feature_3d [B, ..., 384]
```

---

## 6. Loss — HoloBrainActionLoss

**文件**：`robo_orchard_lab/models/holobrain/loss.py`

### 6.1 类结构

```python
# loss.py:29-142
class HoloBrainActionLoss(nn.Module):
    def forward(self, model_outs, inputs, text_dict):
        pred    = model_outs["pred"]         # [B*P, T=64, 14, 8]  P=num_parallel=4
        target  = model_outs["target"]       # [B*P, T=64, 14, 8]
        timesteps = model_outs["timesteps"]  # [B*P]

        # timestep-weighted smooth_l1 (β=0.04), best-of-4-parallel selection
        loss = self._loss_func(pred, target, timesteps)          # L151-211
        loss_angle, loss_xyz, loss_rot = robot_state_loss(loss)  # L113 (split [1, 3, 4→3] over dim -1)

        # FK loss: same predictions run through FK to get ee_pose, then compute loss again
        pred_fk = fk(pred);  target_fk = fk(target)
        loss_fk = self._loss_func(pred_fk, target_fk, timesteps)
        loss_angle_fk, loss_xyz_fk, loss_rot_fk = robot_state_loss(loss_fk)

        return {                                                 # L135-141 — 6 个 key 与 log 完全对应
            "loss_angle":     loss_angle,
            "loss_xyz":       loss_xyz,
            "loss_rot":       loss_rot,
            "loss_angle_fk":  loss_angle_fk,
            "loss_xyz_fk":    loss_xyz_fk,
            "loss_rot_fk":    loss_rot_fk,
        }
```

### 6.2 timestep 权重（diffusion）

```python
# loss.py:151-211  内部关键
# timestep_loss_weight = 1000/(t+1)  → 强化 t 小（去噪后期）的样本
# parallel_loss_weight = 0.1  → best-of-4 hard 选择加软策略
# smooth_l1(β=0.04)
```

### 6.3 state_loss_weights 覆盖

joint 6 和 joint 13（gripper joint）loss 权重被拉到 0.1，其他 1.0（`config_holobrain_common.py:284-408` 里 decoder cfg）。

---

## 7. Checkpoint 保存

**文件**：`robo_orchard_lab/pipeline/hooks/checkpoint.py`

### 7.1 SaveCheckpoint hook

```python
# checkpoint.py:78-219
class SaveCheckpoint(TrainerHookMixin):
    def __init__(self, save_step_freq=5000, save_epoch_freq=None,
                 save_when_loop_end=True, total_limit=3):
        ...

    def _on_step_end(self, ctx):                                 # L155
        if ctx.global_step % self.save_step_freq == 0:
            path = f"{ctx.workspace}/checkpoints/checkpoint_{self.n_saved}"
            ctx.accelerator.save_state(path)                     # accelerate state (all files below)
            # 触发 pre-hook (torch_model.py:202) 会额外写 model.config.json
            ctx.accelerator.save_model(model, f"{path}/model")   # pytorch_model.bin
            self._prune_old(ctx.workspace, total_limit=3)        # 只留最新 3 份
            self.n_saved += 1
```

### 7.2 checkpoint_N 目录内容

每一份 checkpoint（如 `/job_data/checkpoints/checkpoint_9/`）包含：

```
checkpoint_9/
├── model.safetensors           # 2.83 GB, accelerate save_state 主 weight (bf16)
├── model.config.json           # from pre-hook, model 结构 json
├── optimizer.bin               # AdamW state
├── scheduler.bin               # LR scheduler state
├── sampler.bin                 # DataLoader sampler 状态（保证 resume 数据顺序不变）
├── random_states_0.pkl         # torch/numpy/py random state per process
├── custom_checkpoint_0.pkl     # TrainerProgressState (global_step, epoch, ...)
└── model/
    └── pytorch_model.bin       # accelerate save_model 的另一份 weight (fp32/bf16)
```

**注意**：`accelerate save_state` + `save_model` 双写造成两份 weight。`model.safetensors` 是 deploy 时唯一需要的，其余 6 个是 resume 训练时才需要。见 [02_deploy_package.md](02_deploy_package.md)。

### 7.3 total_limit=3 的坑

`save_step_freq=5000, max_step=100000` → 会存 20 份 checkpoint，但 `total_limit=3` 会让老 checkpoint 被删。**若要挑 fine-grained 曲线上的中间 step，训练完成前必须及时抓 checkpoint_N 到 bucket 外**，或者显式设 `total_limit=None`。

---

## 8. LossTracker → TensorBoard

**文件**：`robo_orchard_lab/pipeline/hooks/loss_tracker.py:102-142`

`_on_step_end` 每步把 batch_processor 返回的每个 loss key 写到 TB：

```python
# loss_tracker.py:134-138
for k, v in ctx.batch_result.items():
    self.writer.add_scalar(f"Loss/{k}", v, ctx.global_step)
self.writer.add_scalar("Loss/Total_Loss", ctx.batch_result["total_loss"], ctx.global_step)
```

同时日志文件里打（就是 log 里看到的 `loss_angle[0.03] ... total_loss[0.13]`）。

---

## 9. 训练进度信号（从 log 抓什么）

用 `aidictl job logs tail <job_id> log/<job_id>-task-1-main.log` 后 grep：

| 目标 | 关键 pattern |
|---|---|
| 每步 loss | `GlobalStep\[[0-9]+/99999\]` |
| 训练速度 | `Training Speed: [0-9.]+ samples/sec` |
| 剩余时间 | `Estimated Remaining Time: [0-9:]+` |
| checkpoint 落地 | `Save checkpoint at the end of step [0-9]+` |
| depth loss（本训禁用） | `loss_depth` — 应该看不到 |

100k 训练实测在 5090 上 **~275 samples/sec, ~0.93 s/step**，26h 完成。

---

## 相关文件汇总

**Submit / launch**
- `projects/holobrain_internal/common/aidi_submit_config/submit_cfg_robodojo_train_100k.json:1-29`
- `projects/holobrain_internal/common/aidi_submit_config/accelerate_multi_node.yaml`

**Entry point**
- `projects/holobrain_internal/common/train.py:59` main
- `projects/holobrain_internal/common/train.py:49-56` `MyBatchProcessor`
- `projects/holobrain_internal/common/train.py:174-204` `SimpleTrainer` 构造
- `projects/holobrain_internal/common/train.py:228-241` Accelerator 构造
- `projects/holobrain_internal/common/holobrain_utils.py:72-80` `load_config`
- `projects/holobrain_internal/common/holobrain_utils.py:127-158` `load_checkpoint`（warm-start）

**Config**
- `projects/holobrain_internal/common/configs/config_holobrain_common.py:19-44` base cfg
- `projects/holobrain_internal/common/configs/config_holobrain_common.py:85-94` v9 override
- `projects/holobrain_internal/common/configs/config_holobrain_common.py:104` `build_model`
- `projects/holobrain_internal/common/configs/config_holobrain_common.py:225-268` preprocessor + backbone_3d
- `projects/holobrain_internal/common/configs/config_holobrain_common.py:269-283` DepthFusionSpatialEnhancer
- `projects/holobrain_internal/common/configs/config_holobrain_common.py:284-408` decoder cfg
- `projects/holobrain_internal/common/configs/config_holobrain_common.py:409-453` robot encoder
- `projects/holobrain_internal/common/configs/config_holobrain_common.py:460-510` optimizer + scheduler

**Dataset**
- `projects/holobrain_internal/common/configs/dataset_specs_robodojo.py:34,55-71`
- `projects/holobrain_internal/common/configs/data_configs/config_robodojo_dataset.py:26-79` `dataset_config["arx_x5a"]`
- `projects/holobrain_internal/common/configs/data_configs/config_robodojo_dataset.py:82-287` transforms
- `projects/holobrain_internal/common/configs/dataset_factory.py:199-233` `build_training_dataset`
- `robo_orchard_lab/dataset/robodojo/robodojo_lmdb_dataset.py:40-252`
- `robo_orchard_lab/dataset/collates.py:38-65` `collate_batch_dict`
- `robo_orchard_lab/dataset/horizon_manipulation/transforms.py:316-1006`

**Model**
- `robo_orchard_lab/models/holobrain/structure.py:51-489` HoloBrain_Qwen2_5_VL
- `robo_orchard_lab/models/holobrain/action_decoder.py:179-717` HoloBrainActionDecoder
- `robo_orchard_lab/models/holobrain/robot_state_encoder.py:80-` RobotStateEncoder
- `robo_orchard_lab/models/holobrain/loss.py:29-211` HoloBrainActionLoss
- `robo_orchard_lab/models/torch_model.py:202-` save_state pre-hook
- `robo_orchard_lab/models/bip3d/spatial_enhancer.py:170-255` DepthFusionSpatialEnhancer

**Pipeline**
- `robo_orchard_lab/pipeline/trainer.py:49-179` `SimpleTrainer.__init__`
- `robo_orchard_lab/pipeline/hook_based_trainer.py:368-` `HookBasedTrainer.__call__` 主循环
- `robo_orchard_lab/pipeline/batch_processor/simple.py:61-227` `SimpleBatchProcessor`
- `robo_orchard_lab/pipeline/hooks/checkpoint.py:78-219` `SaveCheckpoint`
- `robo_orchard_lab/pipeline/hooks/loss_tracker.py:102-142` `LossTracker._on_step_end`
