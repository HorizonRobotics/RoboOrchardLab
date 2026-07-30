# 05 — Troubleshooting & 已知坑

> **范围说明**：本文的排障记录来自外部 RoboDojo repo 流程与 xiaomi baseline 的实操。
> 现行评测改用 in-repo `robodojo_eval.py`（见 [03_eval.md](03_eval.md) 顶部说明），
> xiaomi 已移出范围。Isaac Sim / 镜像 / rsync 这类底层坑仍然适用。


Session 10f5c967 期间 v1→v6 image + 20k train + sanity + seed0 eval 逐步跑通时踩过的坑，按类型汇总。

**Memory 参照**：[[robodojo-holobrain-eval-image-v6]] · [[holobrain-aidi-submit-conventions]] · [[kun-wu-bucket-workspace]]

---

## 1. AIDI SDK 三大陷阱（skill `aidi-cloud-submit` §2.4）

| 陷阱 | 现象 | 应对 |
|---|---|---|
| **log 吃 job_id** | `submit_from_config` 打了 `Command executed:` 但 stdout 没 job_id | 用 REST API `job/list` 反查（[04_commands_cheatsheet.md](04_commands_cheatsheet.md) §2） |
| **`aidictl list` 15min 缓存** | 刚提交的 job 不出现 | 用 REST `job/get` 端点（无缓存） |
| **重复提交倍数消耗**| 手抖多次跑同一命令会重复起 | 只要 `Command executed:` 打了就先假设 submitted，用 REST 反查再决定 |

---

## 2. rsync `-aL` symlink 坑

**症状**：AIDI 打包 `to_upload` 时用 `rsync -aL`（follow symlinks，将 target 内容 tar 进去），但**实测容器内 symlink target 会 dangling** 到 dev-machine 路径。

**例**：`RoboDojo/robo_orchard_lab -> ~/git_repo/robo_orchard_lab/robo_orchard_lab`（symlink）→ 上传后 pod 里 `/running_package/code_package/robo_orchard_lab` 是个 dangling symlink 指向 dev-machine 路径。

**修复**：把 target **实拷贝**成真实目录：

```bash
rm -rf /home/users/kun01.wu-labs/git_repo/RoboDojo/robo_orchard_lab
cp -r /home/users/kun01.wu-labs/git_repo/robo_orchard_lab/robo_orchard_lab \
      /home/users/kun01.wu-labs/git_repo/RoboDojo/robo_orchard_lab
```

（cp -r 而非 rsync 因为 rsync -aL 也可能踩坑）

---

## 3. IsaacLab pin sed patch

**症状**：pod 启动时 IsaacSim Kit 立即 exit，log 里 `isaacsim.asset.importer.urdf 2.4.31 not found`。

**原因**：v6 image 里该 extension 是 2.4.30，`IsaacLab/apps/isaaclab.python.kit` 硬 pin `{version = "2.4.31", exact = true}`。

**修复**：submit_cfg 里 cmd L19-20 放开 pin：

```bash
sed -i 's|isaacsim.asset.importer.urdf" = {version = "2.4.31", exact = true}|isaacsim.asset.importer.urdf" = {}|' \
    /home/users/kun01.wu-labs/git_repo/RoboDojo/third_party/IsaacLab/apps/isaaclab.python.kit \
    ${WORKING_PATH}/third_party/IsaacLab/apps/isaaclab.python.kit 2>/dev/null || true
```

两条路径都 sed 是因为 image 里和 workspace 里各有一份。`2>/dev/null || true` 保护单条失败不影响另一条。

---

## 4. numpy < 2.0 硬 pin 冲突

**症状**：`import numpydantic` 报错 `numpy 2.x required` 但 mplib 又要 numpy<2.0。

**原因**：v6 image 里：
- `numpydantic 1.10` 要 numpy >= 2.0
- `mplib 0.2.1` 要 numpy < 2.0

**修复**：手工 pin numpy 到 `1.26.4`，让 mplib 优先。**只要 numpydantic 不被真正调用就没事**（HoloBrainProcessor.load 走的路径不 hit numpydantic）。

```bash
pip install numpy==1.26.4
```

如果 build image 时报 numpydantic ImportError，正常，跳过即可。

---

## 5. RoboDojo eval submit_cfg 必需 workaround 清单

1. **`clear_workspace: true`** — 否则 stale workspace 会 pile up（曾撑到 76 GB）
2. **IsaacLab pin sed patch** — 见 §3
3. **`arx_x5_holobrain` env_cfg 对应 `_robot_info.json` entry + Assets Eval_Layout symlink**：
   ```bash
   # Bucket 侧
   ln -sfn /horizon-bucket/robot_lab/users/kun01.wu/datasets/RoboDojo/Assets/Eval_Layout/RoboDojo/arx_x5 \
           /horizon-bucket/robot_lab/users/kun01.wu/datasets/RoboDojo/Assets/Eval_Layout/RoboDojo/arx_x5_holobrain

   # RoboDojo 侧
   # XPolicyLab/utils/robot/_robot_info.json 里加 arx_x5_holobrain entry
   # {"arx_x5_holobrain": {"arm_dim": [6, 6], "ee_dim": [1, 1], ...}}   # 与 arx_x5 同结构
   ```
4. **Bucket-side `checkpoint_20000/ckpt` symlink**：
   ```bash
   ln -sfn /horizon-bucket/robot_lab/users/xuewu.lin/ckpt \
           /horizon-bucket/robot_lab/users/kun01.wu/aidi_output/holobrain_robodojo_posttrain_v9/checkpoint_20000/ckpt
   ```
   HoloBrainProcessor.load 走 `in_cwd(model_dir)`，`./ckpt/Qwen2.5-VL-3B-Instruct` 必须相对可解析
5. **`ckpt` symlink in `${WORKING_PATH}`** — eval-client 的 cwd 是 WORKING_PATH（不是 model_dir），也要能找到 `./ckpt/...`
6. **背景 rsync 循环**（[03_eval.md](03_eval.md) §1.2 L26-32）保险，`trap EXIT` flush
7. **别加 `--fail-fast`** — 会掩盖后续 task 的独立错误

---

## 6. v6 image dep 排查历史（按发现顺序）

从 xiaomi `-l3` image 派生，逐个 dep import 报错时加装：

| # | 缺失包 | 修复 |
|---|---|---|
| 1 | `robo_orchard_core` | 本地 `~/git_repo/robo_orchard_core/` 复制到 /opt/ + editable install，pydantic<=2.10.6 硬 pin |
| 2 | `pytorch3d` | 0.7.9 CPU-only（`FORCE_CUDA=0` + 本地 editable，GitHub clone 不通） |
| 3 | `scipy` | 走 hobot pypi mirror |
| 4 | `pytorch_kinematics` | mirror |
| 5 | `datasets` | HF datasets |
| 6 | `sqlalchemy` | mirror |
| 7 | `pyzstd` | mirror |
| 8 | `sortedcontainers` | mirror |
| 9 | `msgpack_numpy` | **XPolicyLab client_server/ws/protocol/codec.py 硬依赖** |
| 10 | `omni.kit.usd` | IsaacLab sed patch（见 §3）|
| 11 | `Eval_Layout/arx_x5_holobrain` | bucket symlink（见 §5.3） |
| 12 | `Qwen2.5-VL-3B-Instruct` 路径 | `checkpoint_20000/ckpt` symlink（见 §5.4） |

**v6 image 里 baked 的关键包**（RoboDojo env 里已有的不算，holobrain env 里额外的）：
- torch 2.8.0+cu128 / torchvision / torchaudio
- transformers 5.10.2 / accelerate 1.14 / diffusers 0.39 / safetensors / peft / websockets
- pytorch3d 0.7.9 CPU-only
- robo_orchard_core (editable)
- pytorch_kinematics / scipy / sklearn / imageio / lmdb / timm / omegaconf / mmengine / urdf_parser_py
- datasets / tensorboard / hydra / rich / pyarrow / sqlalchemy / pyzstd / sapien 3.0.3 / mplib 0.2.1
- **msgpack_numpy** / sortedcontainers / rosbags / mcap
- numpy < 2.0 硬 pin
- 全套 aws sdk / opentelemetry / ray 等（来自 pip freeze 训练镜像 221 包）

---

## 7. 训练 checkpoint total_limit=3 坑

**症状**：训练完成后想挑中间某个 step 的 ckpt 做曲线分析，发现前面的 ckpt 都被删了。

**原因**：`SaveCheckpoint(total_limit=3)` 只保留最新 3 份（`pipeline/hooks/checkpoint.py:78-219`）。

**避免**：
- 训练中及时抓 ckpt 到 bucket 外（`aidictl job logs download <id> output/` 或直接 rsync from bucket）
- 或提交训练时显式改 `total_limit=None`（改 `train.py:174-204` 里 SaveCheckpoint 参数）

---

## 8. 训练 SR=0 但 loss 好看

**症状**：20k train, `total_loss=0.098`（vs 100k 训练早期已 0.13），但 sanity eval 全 fail。

**原因**：`loss_angle + loss_xyz + loss_rot`（+ fk 分量）是**每步的 diffusion 去噪 loss**，衡量 policy 在噪声动作序列上的重建能力，与真实 rollout 无关。

**判断**：需要 eval 侧的 `success_rate` 和 `score` 作为下游 metric。100k train 完成后 (`total_loss ≈ 0.05`) 才是有意义的 policy 检验点。

**建议**：
- 定期挑中间 ckpt 起 sanity smoke eval（`--eval-num 1` × 2-3 task），10 min 内看质变
- 别信训练 loss 曲线单独下降就说明 policy 好

---

## 9. seed0 eval wall-time 撞 48h

**症状**：seed0 job wall_time=2880 min 到期时只跑完 ~25-28 task（xiaomi 亦然）。

**原因**：每 task 25 ep × ~5min/ep = 125min，54 task × 125min = 112h ≫ 48h。

**应对**（P1 决策）：
- **接受 partial coverage**：与 xiaomi baseline 对齐每 task 对比
- **分组并发**：拆 6 个 job × 每组 9 task × 25 ep × 12h → 12h 内跑完（需要 6 × 8 = 48 GPU）
- **降 `--eval-num`**：25 → 15 或 10（xiaomi 是 100，我们 25 已激进），换取更全 task coverage

---

## 10. 3-strike fatigue 心法

**规则**：同一类 error 修 3 次没成，停下汇报。

**判断「同一类」**：
- 报错 top-line 一样（如 `ImportError: msgpack_numpy`）
- 或修改的文件/行数相近（如三次都在同一段 sed 修 IsaacLab pin）
- **不是**「修一个 import 报下一个 import」，那算 pipeline 前进不算重复

**为什么**：v6 image 的迭代经验说明——单个 dep 问题通常 5-10 min 内能解，但 3 次都修不好说明**根因不是缺包**，可能是 config、env 变量、docker layer 冲突。硬修只会耗 token。

---

## 11. 常见 submit_cfg 错误

| 错误 | 症状 | 修复 |
|---|---|---|
| `queue_name` 拼错 | `Queue not found` | 用 `project-5090-robot-lab-bcloud-bj`（skill `aidi-cloud-submit`） |
| `project_id=Robot-lab` | Job 起来但计费错 | 应为 `horizon-labs`（`Robot-lab` 是 gendata 用的） |
| 忘 mount `robot_lab2` | 训练 pod 上 lmdb 数据不可见 | `input_bucket: "robot_lab,robot_lab2"` |
| workspace_folder 重名 | 多次 submit 相互覆盖 | 每次改后缀（如 `-v2`, `-100k`） |
| `wall_time` 单位错 | 分钟数搞错 | 是分钟数（源码 typo "minitus"），`wall_time=2880` = 48h |

---

## 12. 集群 GPU queue 长时间不到手

**症状**：`bcloud-bj-zone1-7895445e92bc` seed0 job Queuing 了 5h 才到 GPU。

**诊断**：
```bash
aidictl queue ls --type gpu -f "top=5"
# 看 project-5090-robot-lab-bcloud-bj: allocated/free/waiting
```

**应对**：
- 如果 free=0 且 waiting > 3，只能等
- 如果自己有多个 Queuing job 抢队列，考虑 `aidictl job stop` 掉次要 job
- **提高优先级**：`aidictl job urgent <job_id>`（数值大 = 高优，`priority=5` 已是最高）

---

## 13. Isaac Sim 起不来 / GLFW error

**症状**：eval 客户端 log 出现 `Failed to startup plugin carb.windowing-glfw.plugin` 后 hang 或退出。

**通常无害**：headless 模式下 GLFW 不启用是正常的，warning 不影响 sim。**只有连续 warning + 之后无 `[MAIN] eval finished`** 才需要担心。

---

## 14. GPU OOM

**症状**：训练 log 里 `CUDA out of memory`。

**排查**：
- batch_size × num_gpu × bf16_footprint > VRAM
- v9 model + Qwen2.5-VL-3B 在 5090 32GB 上 batch=16 刚好；batch=32 会 OOM
- **不要提 batch_size**，横向扩 GPU（加 pod / 加 gpu_per_worker）

---

## 15. Docker push 失败

**症状**：`docker push ...` 报 `401 Unauthorized` 或 `manifest blob unknown`。

**修复**：
```bash
docker logout docker.hobot.cc
docker login docker.hobot.cc            # 重新登录
# 确认 tag namespace 正确：docker.hobot.cc/imagesys/kun01.wu/...
docker push ...
```

如果 push 大 image 卡在 `Preparing`：`docker system prune -f` 后重启 daemon。

---

## 相关 memory

- [[robodojo-holobrain-eval-image-v6]] — v6 image 完整配方
- [[holobrain-aidi-submit-conventions]] — 提交约定
- [[holobrain-checkpoint-layouts]] — accelerate state vs deploy package
- [[kun-wu-bucket-workspace]] — bucket 目录结构
- [[robotwin-eval-blocked-on-curobo]] — 前一次 RoboTwin eval 的坑（curobo 装不上）
