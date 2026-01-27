# LMDB Packer for InternA1 Dataset

这个文件夹包含用于将 InternA1 数据集从 LeRobot 格式转换为 LMDB 格式的工具和脚本。LMDB（Lightning Memory-Mapped Database）是一种高效的键值存储数据库，适用于大规模机器人数据集的存储和访问。

## 文件说明

### `lmdb_pack_InternA1.py`
主要的打包脚本，用于将 LeRobot 数据集转换为 LMDB 格式。

**功能特性：**
- 支持多种机器人类型：ARX Lift-2、AgileX Split Aloha 和 Genie-1
- 处理多模态数据：RGB 图像、关节状态、动作、末端执行器姿态等
- 进行正向运动学计算，将关节状态转换为笛卡尔坐标
- 图像预处理：调整大小、JPEG 压缩
- 过滤静止帧以减少数据冗余
- 批量处理以提高效率

**使用方法：**
```bash
python3 robo_orchard_lab/dataset/interna1/packer/lmdb_packer/lmdb_pack_InternA1.py \
        --input_path /horizon-bucket/robot_lab2/datasets/InternData-A1/sim_updated_lerobotv30/arrange_the_tableware \
        --output_path dataset/lmdb_dataset_ARX_Lift2_arrange_the_tableware_ep_startid_100_ep_endid_101_static \
        --start_idx 100 \
        --end_idx 101

python3 robo_orchard_lab/dataset/interna1/packer/lmdb_packer/lmdb_pack_InternA1.py \
        --input_path /horizon-bucket/robot_lab2/datasets/InternData-A1/sim_updated_lerobotv30/pour_water_right_arm \
        --output_path dataset/lmdb_dataset_ARX_Lift2_pour_water_right_arm_ep_startid_100_ep_endid_101 \
        --start_idx 100 \
        --end_idx 101
```

**参数说明：**
- `--input_path`: LeRobot 数据集路径
- `--output_path`: 输出 LMDB 数据库路径
- `--start_idx`: 开始处理的 episode 索引
- `--end_idx`: 结束处理的 episode 索引



### `viz_lmdb_InternA1.py`
用于可视化打包后 LMDB 数据集的脚本。

**功能特性：**
- 从 LMDB 数据集中导出视频
- 随机选择 episodes 进行可视化
- 生成 HTML 链接页面
- 支持集群环境下的文件复制

**使用方法：**
```bash
python3 robo_orchard_lab/dataset/interna1/packer/lmdb_packer/viz_lmdb_InternA1.py \
        --lmdb_dataset_path dataset/lmdb_dataset_ARX_Lift2_arrange_the_tableware_ep_startid_100_ep_endid_101_static \
        --output_path dataset/lmdb_dataset_ARX_Lift2_arrange_the_tableware_ep_startid_100_ep_endid_101_static/viz

python3 robo_orchard_lab/dataset/interna1/packer/lmdb_packer/viz_lmdb_InternA1.py \
        --lmdb_dataset_path dataset/lmdb_dataset_ARX_Lift2_pour_water_right_arm_ep_startid_100_ep_endid_101 \
        --output_path dataset/lmdb_dataset_ARX_Lift2_pour_water_right_arm_ep_startid_100_ep_endid_101/viz
```

**参数说明：**
- `--lmdb_dataset_path`: LMDB 数据集路径
- `--output_path`: 可视化输出路径

### `submit_lmdb_pack_InternA1.py`
用于批量提交打包任务到集群的脚本。

**功能特性：**
- 自动扫描数据集目录
- 生成提交配置文件和运行脚本
- 支持分批处理大量 episodes
- 集成集群作业管理系统

**使用方法：**
直接运行脚本，它会自动处理所有符合条件的任务：
```bash
python submit_lmdb_pack_InternA1.py
```

脚本会为每个数据集生成相应的 `submit_pack.json` 和 `run_pack.sh` 文件，并提交到集群。

## 数据流程

1. **数据准备**: 确保 LeRobot 数据集可用
2. **打包**: 使用 `lmdb_pack_InternA1.py` 将数据转换为 LMDB 格式
3. **验证**: 使用 `viz_lmdb_InternA1.py` 可视化结果
4. **批量处理**: 使用 `submit_lmdb_pack_InternA1.py` 在集群上批量处理

## 依赖项

- PyTorch
- NumPy
- OpenCV
- LeRobot
- RoboOrchard Core
- 其他相关库（详见脚本中的 import）

## 注意事项

- 确保有足够的磁盘空间存储 LMDB 数据库
- 对于大型数据集，建议在集群环境中运行
- 可视化脚本会随机选择 episodes，可能需要多次运行以覆盖更多数据
- 打包过程中会进行内存优化，避免 OOM 错误