# 11 · 术语表

> **阅读前置**：无（可当查手册用）
>
> 分为「通用缩写」与「项目内部命名」两部分。

---

## 11.1 通用缩写

| 缩写 | 全称 | 含义 |
|------|------|------|
| **VLM** | Vision-Language Model | 视觉语言大模型。HoloBrain 用 Qwen2.5-VL / Qwen3-VL |
| **VLA** | Vision-Language-Action | 在 VLM 基础上加动作头，形成机器人策略模型的一种范式（如 OpenVLA, RT-2） |
| **DDPM** | Denoising Diffusion Probabilistic Models | 扩散模型的经典训练方案。HoloBrain 训练时用 `DDPMScheduler(prediction_type="sample")` 直接预测干净轨迹 x₀ |
| **DPM-Solver** | Diffusion Probabilistic Model Solver | 一种加速扩散推理的 ODE/SDE 求解器；HoloBrain 推理用 `DPMSolverMultistepScheduler` |
| **CFG** | Classifier-Free Guidance | 扩散模型里通过随机 drop 条件让模型学出条件/无条件两分支，推理时用差分放大条件影响。HoloBrain 里对 state 分支有 CFG-style dropout（`state_drop_rate=0.2`），对语言 token 无 |
| **FK** | Forward Kinematics | 正向运动学：从关节角推出末端位姿 |
| **EE** | End-Effector | 机器人末端执行器（爪子、夹爪、手腕） |
| **DiT** | Diffusion Transformer | 把 Transformer 用作扩散网络主体的架构；`AdaRMSNorm(zero=True)` 输出六元组门控是 DiT 的标准 modulation |
| **RMSNorm** | Root Mean Square Norm | LayerNorm 的一个变体，只做方差归一化。用于 Qwen 系列与 HoloBrain 全网 |
| **AdaLN** | Adaptive LayerNorm | 让 norm 的 `scale, shift` 依赖某个条件（如时间步 embedding）；`AdaRMSNorm` 就是 RMS 版 |
| **Rotary** | Rotary Position Embedding (RoPE) | 用旋转矩阵编码位置的注意力位置方案，Qwen 与 HoloBrain 的 `RotaryAttention` 都用它 |
| **SDPA** | Scaled Dot-Product Attention | PyTorch 2 提供的 fused attention 实现（`F.scaled_dot_product_attention`），是 `RotaryAttention` 的底层 |
| **LMDB** | Lightning Memory-Mapped Database | 一种基于 mmap 的键值存储；HoloBrain 用它保存图像/深度/元数据 |
| **URDF** | Unified Robot Description Format | XML 格式的机器人描述文件，`pytorch_kinematics` 从里面解析出运动学链 |
| **CoT** | Chain-of-Thought | 让模型在给出答案前先输出一段推理过程。`with_cot=True` 时 HoloBrain 走 `_generate_vlm` 生成 subtask 描述再交给 decoder |
| **FSDP / DDP** | Fully Sharded Data Parallel / Distributed Data Parallel | 多卡训练策略；由 accelerate 决定用哪种 |
| **SafeTensors** | HuggingFace 的一种更安全的 tensor 序列化格式 | HoloBrain checkpoint 主要以 `.safetensors` 形式保存 |
| **msgpack** | 二进制 JSON 变体 | GenieSim3 WebSocket 通信用它序列化 numpy array |
| **Xvfb** | X Virtual Framebuffer | Linux 上无显示器情况下起虚拟显示；Isaac Sim / Behavior-1K eval 都要它 |
| **HL-Gauss** | Histogram + Gaussian smoothing 的 categorical regression loss | value 模型（`config_holobrain_value_common.py`）里的 `loss_mode="hlgauss"`，把回归转成 51 类分类 + 高斯软标签 |

## 11.2 项目内部命名

以下按字母序（英文名，中文简称）。

### 张量与形状相关

- **`B / batch_size`**：批大小；由 config 决定，默认 16。
- **`num_cams`**：单个样本的相机数；LIBERO=2, RoboTwin/AgiBot=3~4。一个 batch 内保证一致（由 `DistributedBatchFlagSampler`）。
- **`num_joint`**：单个 embodiment 的关节数；同一 batch 内一致；LIBERO=1（EE + gripper 合成 1 joint）, RoboTwin_aloha_v2=14, AgiBot=20 或 18（no_head 后）。
- **`num_link`**：`HoloBrainRobotStateEncoder` 里对应 num_joint 的别名；同义。
- **`state_dims`**：动作/状态每关节的通道数，**恒为 8** = `[jval, x, y, z, qw, qx, qy, qz]`（关节角度 + EE 位置 + EE 四元数）。
- **`hist_steps`**：历史观测步数；默认 1。
- **`pred_steps`**：预测动作步数；默认 64。
- **`chunk_size`**：一个动作 token 打包多少步；默认 4；`num_chunk = pred_steps // chunk_size = 16`。
- **`embed_dims`**：主 hidden 维度；v0=256，v9=384。
- **`num_hist_chunk`**：`hist_steps // hist_chunk_size`；`HoloBrainRobotStateEncoder` 用 `chunk_size = min(8, hist_steps)`。
- **`h_ / w_`**：VLM 图像 token 的空间网格；`h_ = H // qwen_patch_size`（Qwen2.5 patch=28 时 252/28=9）。
- **`num_traj / num_test_traj`**：推理时每 sample 采样几条轨迹（提供多模态候选）；默认 1，某些 config 会开成 4~8。
- **`num_parallel_training_sample / num_parallel / P`**：训练时把 batch 复制 P 份，每份加不同噪声用于 winner-takes-all；默认 4。

### Config / batch 里的 key

- **`imgs`**：`[B, num_cams, H, W, 3]` uint8 或 float，BGR 由 `ImageChannelFlip` 转 RGB，再由 `BaseDataPreprocessor.channel_flip=True` 再确保 RGB。
- **`depths`**：`[B, num_cams, H, W]`，米制，`with_depth=True` 时启用。
- **`text`**：list[str]，chat template 前是原始 instruction，`TextTemplate` 之后是完整 Qwen chat prompt。
- **`hist_robot_state`**：`[B, hist_steps, num_joint, 8]`，历史状态（含 FK）。
- **`pred_robot_state`**：`[B, pred_steps, num_joint, 8]`，训练时的 GT 未来轨迹。
- **`joint_scale_shift`**：`[B, num_joint, 2]`，per-joint `(scale, shift)`；只对通道 0 起作用。
- **`joint_relative_pos`**：`[B, num_joint, num_joint]` long，关节图上两两最短路径距离（Floyd-Warshall 预算）。
- **`joint_mask`**：`[B, num_joint]` bool，True=激活。
- **`pred_mask`**：`[B, pred_steps]` bool，True=真实预测步，False=padding。
- **`state_loss_weights`**：`[B, pred_steps, num_joint, 8]`，per-step per-joint per-channel 权重。
- **`fk_loss_weight`**：同上，控制 `loss_*_fk` 的权重；None 表示不算 FK loss。
- **`kinematics`**：Python list of `MultiArmKinematics` 实例（每样本一个）；collate 时不 stack。
- **`embodiedment_mat`**：`[B, 4, 4]`，`T_base2ego`；由 `GetProjectionMat` 写入。
- **`projection_mat`**：`[B, num_cams, 4, 4]`，`intrinsic @ T_world2cam @ T_base2world @ inv(T_base2ego)`。
- **`ee_frame_alignment`**：`[B, 4, 4]`，LIBERO 里把 EE 坐标绕 z 转 180° 对齐。
- **`reference_imgs`**：`[B, N_ref, H, W, 3]`，可选任务参考图，`LoadReferenceImages` 提供。
- **`subtask`**：list[str]，`training_with_subtask=True` 时拼进 prompt。
- **`uuid`**：list[str]，样本对应 episode id。
- **`noise_type`**：list[str]，扩散噪声模式（`local_joint / global_pose / ...`）。
- **`mobile_traj`**：`[B, pred_steps, mobile_traj_state_dims]`，`with_mobile=True` 时移动底盘轨迹。
- **`remaining_actions / delay_horizon`**：RTC async 推理用的滑动窗口。

### 训练超参 / 开关

- **`num_vlm_layers`**：截断 VLM 只保留前 N 层。None=全保留。v9 默认 4。
- **`freeze_vlm`**：是否冻结 VLM 全部参数（含 vision tower）。
- **`with_cot`**：是否走 `vlm.generate(...)` 自回归 CoT 路径。
- **`training_with_subtask`**：prompt 里是否拼 subtask 描述。
- **`decoder_layers`**：decoder block 层数；v9 是 10。
- **`operation_order`**：list[str]，控制 decoder 每 block 的 op 序列。
- **`temporal_attn_drop`**：`temp_cross_attn` mask 概率；默认 0.05。
- **`state_drop_rate`**：`MultiModalAttention` 的 state 分支概率化 drop；默认 0.2。
- **`teacher_forcing_rate`**：给一个样本以此概率替换 leading 若干步为 clean GT；默认 0.02。
- **`teacher_forcing_mean_steps`**：Poisson span 均值；默认 `pred_steps // 4`。
- **`num_parallel_training_sample`**：每 sample 复制多少份看不同噪声；默认 4。
- **`parallel_loss_weight`**：winner-takes-all 里非最优轨迹的权重；默认 0.1。
- **`timestep_loss_weight`**：`_loss_func` 里用 `w / (t+1)` 加权；默认 1000。
- **`num_inference_timesteps`**：DPM-Solver 步数；默认 10。
- **`noise_type`**：`local/global × joint/pose` 四选一，控制 `sample_noise` 生成噪声方式。
- **`prediction_type`**（HoloBrain）：`absolute/relative × joint × pose` 组合，控制 `get_prediction` 如何把网络输出映射回目标空间。
- **`pred_scaled_joint`**：网络是否直接输出归一化通道 0；默认 False（输出物理量）。

### Sampler / Dataset

- **`ConcatDatasetWithFlag`**：`torch.utils.data.ConcatDataset` 的子类，每 sub-dataset 有一个 int `flag`。
- **`DistributedBatchFlagSampler`**：按 flag 组内攒够 `batch_size` 才发一批；rank 分片在 sampler 内做。
- **`DistributedMixedBatchFlagSampler`**：可选，允许一个 batch 内按 `dataset_batch_ratios` 混合多个 embodiment。
- **`sample_weight`**：每个 spec 里可以写；被 `_finalize_dataset_sample_weights` 打成 list 传给 `WeightedRandomSampler`。

### 处理器 / 推理

- **`HoloBrainProcessor`**：处理器（不是 HF 的），负责 `MultiArmManipulationInput ↔ dict` 与 transforms。
- **`HoloBrainInferencePipeline`**：把 processor + model 组合的推理入口，来自 `robo_orchard_lab.inference.basic.InferencePipeline`。
- **`MultiArmManipulationInput / Output`**：推理侧数据类；字段见 [06 章 6.8](./06_model_architecture.md#68-holobrainprocessor-与数据结构)。
- **`Struct2Dict`**：把 `MultiArmManipulationInput` 转成模型 batch dict 的第一步。
- **`RTCInferencePlugin`**：实时融合插件；`realworld_eval.py` 用它把新一次预测和上次的 `remaining_actions` 拼起来。
- **`async_inference_plugin`**：decoder 的可选属性；非空时启用 RTC。
- **`load_impl="native" / "accelerate"`**：`ModelMixin.load_model` 的两种模式；native 用 safetensors 直接加载，accelerate 走 accelerate 的分片加载。

### 工具函数

- **`apply_scale_shift(state, joint_scale_shift, inverse, scale_only)`**：只对通道 0 做仿射，见 [05 章 5.8](./05_dataset_pipeline.md#58-归一化--统计量)。
- **`recompute(pred, inputs)`**：反归一化 → FK → 拼回，`loss_*_fk` 用它。
- **`apply_joint_mask(state, mask, constant_value=-1)`**：mask 掉的关节位置写常数 -1。
- **`forward_kinematics(joint_state, inputs)`**：一段 hist 或 pred 关节 → 8 维 robot_state。

### Trainer / hooks

- **`SimpleTrainer` / `HookBasedTrainer`**：训练器；SimpleTrainer 是 thin wrapper。
- **`MyBatchProcessor`**：`train.py` 里的 batch processor，`sum` 所有 `loss` key 的 mean。
- **`SaveCheckpointConfig / StatsMonitorConfig / LossMovingAverageTrackerConfig`**：user hook；见 [08 章 8.6](./08_loss_and_training.md#86-hook-系统)。
- **`GradientClippingHookConfig / OptimizerHookConfig / ValidationHookConfig`**：`HookBasedTrainer` 自动挂的三个基础 hook。
- **`ActionMetric`**：验证时用的指标类，输出 `average_joint / final_joint / average_xyz / final_xyz / average_quat / final_quat / jerk / jerk_xyz` 三张 AsciiTable。

### AIDI / 云端

- **`aidi_submit_config/*.json`**：AIDI 集群作业配置；`python_launcher / python_executable / image / uploads / mounts / n_workers / gpus`。
- **`/job_data`**：AIDI 环境下的持久输出目录，`--workspace /job_data` 保证 checkpoint 落到共享盘。
- **`/job_tboard`**：AIDI 环境下的 TensorBoard 目录。
- **`CLUSTER`**：环境变量；`train.py` 里 `if_cluster = os.environ.get("CLUSTER") is not None` 用来切换本地/云端行为（resume 目录不同）。
- **`HOLOBRAIN_DATA_BASE`**：环境变量，覆盖 `dataset_specs.py::DATA_BASE`。
- **`HOLOBRAIN_MODEL_DIR`**：GenieSim3 eval 用；指向导出模型目录。
- **`ROBOTWIN_DIR / LIBERO_ROOT / LIBERO_PLUS_ROOT / ORCHARD_ISAAC_DIR / ROBOCHALLENGE_INFERENCE_REPO`**：各 eval 脚本约定的仿真代码根目录环境变量。

---

**回到 →** [README](./README.md)
