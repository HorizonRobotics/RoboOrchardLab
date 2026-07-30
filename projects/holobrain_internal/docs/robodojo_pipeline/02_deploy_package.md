# 02 — 从 accelerate state 到 deploy package

**问题**：训练完得到的是 `checkpoint_9/` 里 8 个文件的 accelerate state；但评测时 `HoloBrainProcessor.load` / `ModelMixin.load_model` 只需要 5 样：`model.safetensors` + `model.config.json` + `<sim>_processor.json` + `<sim>_inference.config.json` + `urdf/`。这一步就是**手工组装**（AIDI 训练侧不会自动做）。

**Memory 参照**：[[holobrain-checkpoint-layouts]]。

---

## 1. 两种 checkpoint 布局

### 1.1 accelerate state（训练侧原生保存）

一个 checkpoint_N/ 里：

```
checkpoint_9/
├── model.safetensors               2.83 GB  bf16 flat state_dict, from save_state
├── model.config.json               <1 KB    (自动生成 by torch_model.py:202 pre-hook)
├── optimizer.bin                   AdamW state
├── scheduler.bin                   LR scheduler
├── sampler.bin                     DataLoader sampler position
├── random_states_0.pkl             torch/numpy/py random state
├── custom_checkpoint_0.pkl         TrainerProgressState (step, epoch)
└── model/
    └── pytorch_model.bin           accelerate save_model 输出的另一份 weight
```

**用途**：**只能 resume 训练**，不能直接送评测。因为缺少 processor.json（tokenizer / instruction template 定义）和 inference.config.json（推理时 diffusion scheduler / chunk size / num_inference_timesteps）。

### 1.2 deploy package（评测端需要的）

```
checkpoint_20000/                      ← 目录名不重要，惯例用 step 数
├── model.safetensors                  ← 从 checkpoint_N/ 拷来
├── model.config.json                  ← 从 checkpoint_N/ 拷来
├── robodojo_processor.json            ← 手工组装，见 §3
├── robodojo_inference.config.json     ← 手工组装，见 §4
├── urdf/                              ← ARX X5 dual arm URDF
│   └── robotwin2_dual_arm_arx_x5a.urdf
└── ckpt -> /horizon-bucket/robot_lab/users/xuewu.lin/ckpt   ← symlink!
```

**关键：`ckpt` 是 symlink**（相对路径 `./ckpt/Qwen2.5-VL-3B-Instruct` 得能在 processor.load 时的 cwd 里找到）。

---

## 2. `model.safetensors` 的来源

选择 checkpoint step：

- **20k**（`checkpoint_9`，step 19999）：`total_loss=0.098`，最好曲线
- **50k**（假设 `checkpoint_9` in 100k train run，step 49999）：`total_loss ≈ 0.07`

生成 md5 做防篡改校验：

```bash
BUCKET_SRC=/job_data/checkpoints/checkpoint_9    # 在集群 pod 里
BUCKET_DST=/horizon-bucket/robot_lab/users/kun01.wu/robo_orchard_lab/ckpts/holobrain_robodojo_posttrain_v9/checkpoint_20000
cp -v "$BUCKET_SRC/model.safetensors" "$BUCKET_DST/"
cp -v "$BUCKET_SRC/model.config.json" "$BUCKET_DST/"
md5sum "$BUCKET_DST/model.safetensors"     # 20k 版实测 md5=a71cb164...
```

**从 dev machine 拉**（20k job 训练结束后）：

```bash
# aidictl job logs download 会 rsync log/output dir 下来
aidictl job logs download bcloud-bj-zone1-1f00b8e23ac8 output/ --dest ~/tmp_ckpt/
# 或直接从 bucket 读（AIDI output 会自动 rsync 到 bucket）
ls /horizon-bucket/robot_lab/users/kun01.wu-labs/aidi_output/*/checkpoints/checkpoint_9/
```

---

## 3. `robodojo_processor.json` 的来源

**Processor** 负责：把训练时的 dataset transforms 打包成一个 JSON，评测时 `HoloBrainProcessor.load` 会重建同样的 transforms（含 URDF 路径、image size、instruction template 等），保证训练/评测的输入分布一致。

**位置**：`robo_orchard_lab/models/holobrain/processor.py`（`HoloBrainProcessor.dump` / `load`）

### 3.1 生成方法（在训练 pod 里跑一次）

参考 `projects/holobrain_internal/common/configs/config_holobrain_common.py` 里 `build_processor` 相关代码。惯例是训练完在 pod 上做：

```python
from projects.holobrain_internal.common.configs.config_holobrain_common import ConfigHolobrainV9
from projects.holobrain_internal.common.configs.data_configs.config_robodojo_dataset import dataset_config

cfg = ConfigHolobrainV9()
data_cfg = dataset_config["arx_x5a"]
processor = cfg.build_processor(data_cfg)
processor.dump("./robodojo_processor.json")
```

### 3.2 内容示意

```json
{
  "type": "HoloBrainProcessor",
  "transforms": [
    {"type": "AddItems", "items": {...}},
    {"type": "SimpleStateSampling", "hist_steps": 1, "pred_steps": 64},
    {"type": "Resize", "size": [308, 252]},
    {"type": "ToTensor"},
    {"type": "ConvertDataType", "target_dtype": {"cam_head": "bfloat16", ...}},
    {"type": "MultiArmKinematics", "urdf": "./urdf/robotwin2_dual_arm_arx_x5a.urdf"},
    {"type": "ItemSelection", "keys": ["cam_head", "cam_left_wrist", ...]}
  ],
  "instruction_template": "The robot task is: {}",
  "chunk_size": 4,
  "num_parallel_samples": 4,
  ...
}
```

---

## 4. `robodojo_inference.config.json` 的来源

**Inference config** 负责：定义评测时的推理超参（不同于训练！）——用哪个 diffusion scheduler、几步 denoise、要不要 parallel sampling、chunk size。

**内容示意**：

```json
{
  "type": "HoloBrainInferenceConfig",
  "scheduler": {
    "type": "DPMSolverMultistepScheduler",
    "num_inference_timesteps": 10,
    "prediction_type": "sample"
  },
  "chunk_size": 4,
  "num_parallel_samples": 1,          // 训练=4，推理=1（单个 rollout）
  "action_type": "joint"
}
```

**生成**：同 processor，见 `config_holobrain_common.py:284-408` 里 decoder cfg 的 `train_scheduler` vs `test_scheduler` 区分。

---

## 5. `urdf/` 的来源

**必需**：`robotwin2_dual_arm_arx_x5a.urdf`（RoboTwin 系列 URDF）。

- 训练时 `MultiArmKinematics` (`transforms.py:687`) 需要 URDF 做 FK
- 评测时 `HoloBrainProcessor.load` → `MultiArmKinematics` 也需要（保持 processor.json 相对路径 `./urdf/...` 可解析）

拷贝：

```bash
BUCKET_URDF=/horizon-bucket/robot_lab2/datasets/all_data/urdf/urdf_v20260711
mkdir -p $BUCKET_DST/urdf
cp $BUCKET_URDF/robotwin/arx_x5a/robotwin2_dual_arm_arx_x5a.urdf $BUCKET_DST/urdf/
```

---

## 6. `ckpt` symlink 的来源

**为什么需要**：HoloBrainProcessor.load 会 `os.chdir(model_dir)` 然后加载 `./ckpt/Qwen2.5-VL-3B-Instruct/` 作为 VLM base。所以 model_dir 里必须有一个 `ckpt` 目录（或 symlink）指向 Qwen VLM 权重。

```bash
BUCKET_DST=/horizon-bucket/robot_lab/users/kun01.wu/robo_orchard_lab/ckpts/holobrain_robodojo_posttrain_v9/checkpoint_20000
ln -sfn /horizon-bucket/robot_lab/users/xuewu.lin/ckpt $BUCKET_DST/ckpt
# 确认 target
ls -la $BUCKET_DST/ckpt/Qwen2.5-VL-3B-Instruct/    # 必须能列出 config.json / model.safetensors 等
```

**同 AIDI cmd 里也要建 ${WORKING_PATH}/ckpt symlink**（评测 client 端的 cwd 是 WORKING_PATH 而非 model_dir）。见 `submit_cfg_holobrain_robodojo_seed0.json` cmd L15。

---

## 7. 常见 export 错误

| 错误 | 原因 | 修复 |
|---|---|---|
| `FileNotFoundError: robodojo_processor.json` | 忘了 dump processor | 在训练 pod 上跑 §3.1 的 Python snippet |
| `HoloBrainProcessor.load 找不到 urdf` | processor.json 里 `urdf` key 用了绝对路径 | 重新 dump 用相对路径 `./urdf/...` |
| `Qwen2.5-VL not found` | `checkpoint_20000/ckpt` symlink 断了 | 确认 target `xuewu.lin/ckpt/Qwen2.5-VL-3B-Instruct/` 存在 |
| `state_dict key mismatch` | 训练 config (embed_dim=384) vs 部署 config 不一致 | 用同一个 config 版本（v9）生成 processor + train |
| `flash_attn 2 CUDA arch mismatch` | 部署 image 里 flash_attn 未装 | 用 v6 image，或改 `_patch_holobrain_vlm_attn_to_sdpa` (`XPolicyLab/policy/HoloBrain/model.py:355`) 强制 SDPA |

---

## 8. 手工 export 完整 checklist

```bash
# 1) 挑 ckpt
STEP=20000; SRC_JOB=bcloud-bj-zone1-1f00b8e23ac8
# 假设 ckpt 已 rsync 到 bucket
SRC=/horizon-bucket/robot_lab/users/kun01.wu-labs/plat_gpu/2026-07-27/*/${SRC_JOB}/*/output/checkpoints/checkpoint_9

# 2) 建 deploy 目录
DST=/horizon-bucket/robot_lab/users/kun01.wu/robo_orchard_lab/ckpts/holobrain_robodojo_posttrain_v9/checkpoint_${STEP}
mkdir -p $DST/urdf

# 3) 拷 model
cp -v $SRC/model.safetensors  $DST/
cp -v $SRC/model.config.json  $DST/

# 4) 生成 processor + inference config（在 dev machine 或训练 pod 上）
python3 -c "
from projects.holobrain_internal.common.configs.config_holobrain_common import ConfigHolobrainV9
from projects.holobrain_internal.common.configs.data_configs.config_robodojo_dataset import dataset_config
cfg = ConfigHolobrainV9(); data_cfg = dataset_config['arx_x5a']
cfg.build_processor(data_cfg).dump('$DST/robodojo_processor.json')
cfg.build_inference_config().dump('$DST/robodojo_inference.config.json')
"

# 5) URDF
cp /horizon-bucket/robot_lab2/datasets/all_data/urdf/urdf_v20260711/robotwin/arx_x5a/robotwin2_dual_arm_arx_x5a.urdf \
   $DST/urdf/

# 6) ckpt symlink
ln -sfn /horizon-bucket/robot_lab/users/xuewu.lin/ckpt $DST/ckpt

# 7) 校验
ls -la $DST/
md5sum $DST/model.safetensors      # 记录 md5
```

Deploy package 就绪，可以直接被 `submit_cfg_holobrain_robodojo_seed0.json` 的 cmd L16 (`ln -sfn ...checkpoint_20000 XPolicyLab/policy/HoloBrain/checkpoints/checkpoint_20000`) 消费。
