# HoloBrain 新人上手教程

> 面向零基础同学：读完能理解 **数据 → 模型 → 训练 → 推理/评估** 全流程，能跑通最小示例，能改 config 与网络。
>
> 所有结论均基于当前 HEAD 下的真实源码；对代码中未明确处，一律标注 **「待确认」**。
>
> 教程覆盖代码位于：
> - `projects/holobrain_internal/common/` — 项目侧入口、配置、评估脚本
> - `robo_orchard_lab/models/holobrain/` — 模型核心实现
> - `robo_orchard_lab/dataset/` — 数据集与 transforms
> - `robo_orchard_lab/pipeline/` — 训练器 / hook 系统

## 阅读顺序

按下面顺序读，一次一篇，每篇 15–30 分钟：

| 编号 | 文件 | 内容简介 |
|------|------|----------|
| 01 | [01_overview.md](./01_overview.md) | 项目是什么、解决什么、核心思想、端到端一张图 |
| 02 | [02_repo_structure.md](./02_repo_structure.md) | 仓库目录导览与代码阅读顺序 |
| 03 | [03_env_and_quickstart.md](./03_env_and_quickstart.md) | 环境依赖、软链、一条命令跑通训练 / 导出 |
| 04 | [04_config_system.md](./04_config_system.md) | Python 配置系统 + `dataset_factory` + specs |
| 05 | [05_dataset_pipeline.md](./05_dataset_pipeline.md) | LMDB 磁盘布局 → transforms → batch dict |
| 06 | [06_model_architecture.md](./06_model_architecture.md) | VLM + Decoder + Encoder + Layers 逐模块 |
| 07 | [07_forward_pass.md](./07_forward_pass.md) | 端到端一次 forward 的分步走读 |
| 08 | [08_loss_and_training.md](./08_loss_and_training.md) | 训练循环、loss、优化器、hook、accelerate |
| 09 | [09_export_and_eval.md](./09_export_and_eval.md) | `export.py` + 7 个 eval 脚本 + 推理服务 |
| 10 | [10_logging_and_debug.md](./10_logging_and_debug.md) | TensorBoard、可视化、常见坑与二次开发切入点 |
| 11 | [11_glossary.md](./11_glossary.md) | 术语表 |

## 推荐路径

- **只想跑通一次训练**：01 → 03 → 08。
- **想改数据集**：01 → 04 → 05 → 10。
- **想改网络**：01 → 06 → 07 → 08。
- **想部署上线**：01 → 03 → 09。

## 文档约定

- 中文正文；文件路径、类名、shape 保持英文原字。
- 引用源码用形如 `robo_orchard_lab/models/holobrain/action_decoder.py:480-592` 的 `文件:起-止` 记法，方便点击跳转。
- 张量 shape 用符号维度：`B / num_cams / hist_steps / pred_steps / num_joint / num_chunk / embed_dims / state_dims` 等，与源码变量名一致；配一组示例值（`B=16, num_cams=2, pred_steps=64, num_chunk=16, embed_dims=384, state_dims=8`）帮助建立直觉。
- 代码块只贴 5–15 行的关键段并配注释，避免整段粘贴。
- 对代码中未明确、未在仓库中找到实现的行为，用 **「待确认」** 标注，不猜测。
