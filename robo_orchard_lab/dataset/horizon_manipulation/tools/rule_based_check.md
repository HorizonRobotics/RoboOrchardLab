# Rule-Based Check 使用说明

本文以当前仓库中的实现为准，说明 `horizon_manipulation` 数据检查链路里已经落地的 rule-based check：它检查什么、怎么跑、结果怎么解读，以及哪些行为容易被误解。

## 1. 背景与整体流程

当前 rule-based check 的入口是：

```bash
python3 -m robo_orchard_lab.dataset.horizon_manipulation.tools.mcap_checker
```

这条链路目前主要分为三个模块：

- `check_config.py`
  - 定义默认阈值，并支持从 JSON 覆盖
- `check_models.py`
  - 定义 `EpisodeData`、`RuleResult`、`EpisodeInspection`、`EpisodeReport`
- `check_rules.py`
  - 负责逐条执行规则并生成结构化结果

一次检查任务的核心流程如下：

1. 扫描 `input_path` 下符合 `user_names / task_names / date_prefix` 条件的 episode。
2. 解析每个 episode 的 MCAP，提取图像、深度、机器人状态、末端位姿等 topic。
3. 以 `/observation/cameras/left/color_image/image_raw` 的时间戳作为基准时间轴。
4. 其他 topic 按最近时间戳对齐到基准时间轴，并统计：
   - 原始消息条数
   - FPS
   - 起止时间与时长
   - 对齐时间差
5. 组装 `EpisodeData`，执行 rule-based rules。
6. 输出结构化结果和日志；当启用视频导出时，还会生成 review 视频和人工复核页面。

episode 级别状态的聚合口径如下：

- `fail`
  - 存在任意 `severity=blocking` 且 `status=fail` 的规则
- `warning`
  - 没有 blocking fail，但存在 `warning` 或非 blocking 的 `fail` 信号
- `pass`
  - 没有任何 `warning` / `fail` 信号

## 2. 默认阈值

默认阈值定义在 `InspectConfig` 中，当前实现如下：

| 配置项 | 默认值 | 含义 |
| --- | ---: | --- |
| `camera_topics_mean_fps_limit` | `25.0` | 相机 topic 平均 FPS 下限 |
| `camera_topics_min_fps_limit` | `10.0` | 相机 topic 最小 FPS 下限 |
| `robot_state_topics_mean_fps_limit` | `180.0` | 机器人状态 topic 平均 FPS 下限 |
| `robot_state_topics_min_fps_limit` | `50.0` | 机器人状态 topic 最小 FPS 下限 |
| `timestamp_limit` | `0.5` | 时间戳回退、起止时间差、时长差的容忍阈值，单位秒 |
| `alignment_time_diff_limit` | `0.5` | 对齐后最大时间差阈值，单位秒 |
| `joint_limit_tolerance` | `0.1` | URDF 关节上下限容忍量 |
| `joint_change_tolerance` | `0.1` | 相邻帧关节跳变阈值 |
| `master_slave_joint_tolerance` | `0.1` | master / follower 关节差阈值 |
| `ee_pose_position_tolerance` | `0.02` | EE 位置差阈值，单位米 |
| `ee_pose_orientation_tolerance` | `0.05` | EE 朝向差阈值，单位弧度 |

## 3. 检查范围

### 3.1 必检 topic

当前 `required_topics` 明确包含以下 12 路：

```text
/observation/cameras/left/color_image/image_raw
/observation/cameras/middle/color_image/image_raw
/observation/cameras/right/color_image/image_raw
/observation/cameras/left/depth_image/image_raw
/observation/cameras/middle/depth_image/image_raw
/observation/cameras/right/depth_image/image_raw
/observation/robot_state/left/joint
/observation/robot_state/right/joint
/observation/robot_state/left_master/joint
/observation/robot_state/right_master/joint
/observation/robot_state/left/end_pose
/observation/robot_state/right/end_pose
```

### 3.2 参与解析但不在必检集合中的 topic

下面这些 topic 当前会被解析或参与渲染，但不在 `required_topics` 里：

- `*/camera_info`
- `/tf_static`

这意味着它们缺失时，不一定触发独立的 rule id，更可能表现为：

- 渲染失败
- 运行时异常
- 仅在日志里体现

## 4. 检查项说明

### 4.1 Topic 完整性

| Rule ID | 级别 | 触发条件 | 关键指标 |
| --- | --- | --- | --- |
| `missing_topic` | blocking fail | `required_topics` 中有 topic 根本没有在 MCAP 中出现 | `missing_topics` |
| `empty_stream` | blocking fail | topic 出现过，但有效消息数小于等于 0 | `empty_topics` |

补充说明：

- `missing_topic` 现在基于 `observed_topics` 判断，强调“有没有出现过”。
- `empty_stream` 强调“topic 存在，但没有可用数据”。
- 这两条规则现在是显式区分的，不再混在一起。

### 4.2 FPS

| Rule ID | 级别 | 触发条件 | 关键指标 |
| --- | --- | --- | --- |
| `fps_out_of_range` | blocking fail | topic 平均 FPS 小于配置下限 | `mean_fps`、`mean_fps_limit`、`min_fps` |
| `interval_spike_or_drop_frame` | blocking fail | 平均 FPS 仍达标，但最小 FPS 低于阈值，通常表示局部卡顿或掉帧 | `min_fps`、`min_fps_limit` |

补充说明：

- `fps_out_of_range` 更偏向整体频率过低。
- `interval_spike_or_drop_frame` 更偏向局部尖峰、局部掉帧。
- 两者都只检查相机流和机器人状态流，不检查 `camera_info`。
- 不同传感器流的原始消息数本来就不一定完全一致，因此当前实现不再把“原始 count 与基准步数不相等”作为独立 warning。

### 4.3 时间戳与对齐

| Rule ID | 级别 | 触发条件 | 关键指标 |
| --- | --- | --- | --- |
| `timestamp_non_monotonic` | blocking fail | 基准时间轴出现大于 `timestamp_limit` 的倒退 | `min_delta` |
| `start_ts_mismatch` | non-blocking warning | 某 topic 起始时间与基准时间轴起点相差超过阈值 | `actual`、`expected`、`delta` |
| `end_ts_mismatch` | non-blocking warning | 某 topic 结束时间与基准时间轴终点相差超过阈值 | `actual`、`expected`、`delta` |
| `duration_mismatch` | non-blocking warning | 某 topic 时长与基准时间轴时长差超过阈值 | `actual`、`expected`、`delta` |
| `alignment_time_diff_out_of_range` | non-blocking warning | 对齐到基准时间轴后，最近邻匹配的时间差过大 | `max_time_diff`、`mean_time_diff`、`limit` |

补充说明：

- 这些规则里的“基准时间轴”都指向 `left color image`。
- `timestamp_non_monotonic` 直接检查基准时间轴自身。
- `start/end/duration` 规则比较的是各 topic 自己的原始起止时间，与基准时间轴的差值。
- `alignment_time_diff_out_of_range` 比较的是对齐之后的最近邻时间差。

### 4.4 机器人状态一致性

| Rule ID | 级别 | 触发条件 | 关键指标 |
| --- | --- | --- | --- |
| `joint_limit_violation` | blocking fail | 任意关节超出对应 URDF 上下限，且超出容忍量 | `joint_name`、`actual`、`lower_limit`、`upper_limit`、`max_violation` |
| `joint_jump_violation` | non-blocking warning | 相邻帧关节变化量超过阈值 | `max_joint_delta` |
| `master_slave_joint_gap` | non-blocking warning | master 与 follower 关节差过大 | `max_gap` |
| `fk_ee_pose_mismatch` | non-blocking warning | 录制的 EE pose 与由 joint FK 推出的 EE pose 差异过大 | `position_gap`、`orientation_gap` |

补充说明：

- `joint_limit_violation` 按 14 维录制关节布局，对齐到 URDF 中对应 joint 的上下限后判断，并额外应用 `joint_limit_tolerance`。
- `fk_ee_pose_mismatch` 同时比较位置和朝向两部分。
- `orientation_gap` 现在是 recorded pose 与 FK pose 的真实旋转角误差，单位弧度。

## 5. 用法

### 5.1 本地命令行

```bash
ulimit -n 65536
python3 -m robo_orchard_lab.dataset.horizon_manipulation.tools.mcap_checker \
    --input_path ${input_path} \
    --output_path ${output_path}/${user_names}-${task_names}-${date_prefix} \
    --urdf "./urdf/piper_description_dualarm_new.urdf" \
    --user_names ${user_names} \
    --task_names ${task_names} \
    --date_prefix ${date_prefix} \
    --inspect_config ${inspect_config_path} \
    --num_workers 10 \
    [--enable_ffmpeg_log]
```

常用参数说明：

- `--input_path`
  - 原始数据根目录，支持逗号分隔多个根路径
- `--output_path`
  - 检查结果输出目录
- `--urdf`
  - FK 使用的 URDF
- `--user_names`
  - 用户名过滤，逗号分隔
- `--task_names`
  - 任务名过滤，逗号分隔
- `--date_prefix`
  - 日期或日期时间前缀过滤，逗号分隔
- `--inspect_config`
  - 自定义阈值 JSON 文件路径
- `--enable_ffmpeg_log`
  - 打开 ffmpeg 控制台日志，默认关闭

当前 CLI 没有直接暴露以下内部参数：

- `static_threshold`
- `head_time_to_filter`
- `tile_time_to_filter`
- `skip_video_export`

如果通过 Python 直接实例化 `PiperMcapChecker` 使用这些参数，规则行为可能和命令行默认行为不完全一致。

### 5.2 自定义阈值配置

`--inspect_config` 对应一个 JSON 文件。它是局部覆盖模式，未写的字段继续沿用默认值。

示例：

```json
{
  "camera_topics_mean_fps_limit": 24.0,
  "camera_topics_min_fps_limit": 8.0,
  "robot_state_topics_mean_fps_limit": 160.0,
  "robot_state_topics_min_fps_limit": 40.0,
  "timestamp_limit": 0.3,
  "alignment_time_diff_limit": 0.3,
  "joint_change_tolerance": 0.08
}
```

建议：

- 先只改确实需要收紧或放宽的字段，不要把默认配置整份复制一遍。
- 字段名必须与 `InspectConfig` 中的名字完全一致。
- 时间类阈值单位都是秒。

### 5.3 集群提交

可以直接用示例模板：

```bash
RoboOrchardJob-AIDISubmit submit_from_config \
    --config robo_orchard_lab/dataset/horizon_manipulation/tools/submit_check.json
```

如果通过 `tools/app.py` 的页面生成 check 任务：

- `input_path`、`user_names`、`task_names`、`date_prefix` 会根据筛选结果自动回填
- `check` 任务的 `output_path` 在 job 容器内固定为 `/job_data`

## 6. 输出结果怎么看

### 6.1 核心产物

检查任务会产出以下文件：

- `inspect_mcap_result.log`
  - 合并后的 `PASS / ERROR` 列表
- `inspect_full_log.log`
  - 最完整的检查日志，包含运行参数、inspect config、topic summary、详细 rule 命中信息
- `inspect_error_log.log`
  - 从 full log 提取出的 signal 信息
- `inspect_error_log.html`
  - `inspect_error_log.log` 的 HTML 版本，适合人工点击查看

如果视频成功渲染，还会有：

- 每个 episode 的单独 MP4
- `concat_videos.mp4`
- `manual_review.html`
- `manual_review_timeline.json`
- `manual_review_failures.json`

### 6.2 建议排查顺序

建议按下面顺序看结果：

1. 先看 `inspect_mcap_result.log`
   - 快速知道本批有哪些 episode 被打到 `ERROR`
2. 再看 `inspect_error_log.html`
   - 快速聚焦有信号的 block
3. 如需看上下文，再打开 `inspect_full_log.log`
   - 里面能看到完整 topic summary 和每条规则的详细指标
4. 如果需要人眼确认，再打开 `manual_review.html`
   - 用 concat 视频回放并做人工标记

## 7. 使用时最容易误解的点

### 7.1 `episode_status` 与 `inspect_mcap_result.log` 不是一回事

rule-based check 内部会给每个 episode 计算 `episode_status=pass/warning/fail`，但 `inspect_mcap_result.log` 里的 `PASS / ERROR` 使用的是简化后的快速筛选口径，只看：

- 缺 topic
- camera / robot_state 的 FPS 问题
- 解析异常

这意味着：

- 某个 episode 在 `rule_results` 中可能已经是 `warning` 或 `fail`
- 但在 `inspect_mcap_result.log` 里仍然可能显示为 `PASS`

典型例子包括：

- `master_slave_joint_gap`
- `joint_jump_violation`
- `fk_ee_pose_mismatch`
- `joint_limit_violation`

因此：

- `inspect_mcap_result.log` 更适合做快速筛选
- 真正判断 rule-based check 的命中情况，请以 `inspect_full_log.log` 或 `inspect_error_log.html` 里的规则明细为准

### 7.2 人工复核页面的预置失败项，同样沿用快速筛选口径

`manual_review_failures.json` 的初始种子也是从这套 `PASS / ERROR` 快速筛选逻辑推出来的，不是所有 `warning/fail` 规则都会自动进入人工复核列表。

如果团队希望把更多规则自动带入人工复核，需要单独调整这部分映射逻辑。

### 7.3 时间戳相关规则都依赖左目彩色图的时间轴

当前实现用的是：

```text
/observation/cameras/left/color_image/image_raw
```

作为基准时间轴。

这意味着：

- 左目彩色图自身时间戳异常，会放大后续时间对齐相关信号
- `start/end/duration/alignment` 的判断，本质上都是“相对左目彩色图”的偏差

如果后续更换基准流，规则结果会整体变化。

### 7.4 静态段过滤会影响部分时间相关规则

类实现中仍然保留了：

- `static_threshold`
- `head_time_to_filter`
- `tile_time_to_filter`

当 `static_filter_applied=True` 时，以下规则会被跳过：

- `start_ts_mismatch`
- `end_ts_mismatch`
- `duration_mismatch`
- `alignment_time_diff_out_of_range`

当前命令行入口没有直接暴露这些参数；如果通过 Python 直接启用静态段过滤，需要意识到这会改变时间相关规则的输出。

### 7.5 `joint_limit_violation` 现在按 URDF 真实限位检查

当前实现会把录制用的 14 维 joint layout 映射回 FK 链对应的 URDF joints，并按每个 joint 的上下限做检查。

判定逻辑可以理解为：

```text
lower_limit - joint_limit_tolerance <= joint <= upper_limit + joint_limit_tolerance
```

一旦超限，日志里会给出：

- `joint_name`
- `step_index`
- `joint_index`
- `actual`
- `lower_limit`
- `upper_limit`
- `max_violation`

### 7.6 `fk_ee_pose_mismatch` 的 `orientation_gap` 是真实旋转角误差

当前实现中：

- `position_gap`
  - 是位置向量的最大绝对差，单位米
- `orientation_gap`
  - 是 recorded pose 与 FK pose 之间的最大旋转角差，单位弧度

因此 full log 中的：

```text
angle difference: ... radians
```

可以直接按真实角误差理解。

### 7.7 `camera_info` 和 `/tf_static` 不是显式规则检查项

它们会影响：

- 相机内参
- 外参
- 可视化渲染

但它们目前不在 `required_topics` 中，因此：

- 缺失时不一定触发独立 rule id
- 更可能表现为运行时错误、渲染失败，或只在日志里暴露

如果后续希望把这类问题纳入 rule-based check，需要额外补专门规则。

### 7.8 数据压缩大小目前还不是结构化规则

full log 中目前会输出：

- 总大小
- 平均 MB/s
- 当平均大小超过 `100 MB/s` 时打印 warning

但这部分当前还没有进入 `rule_results`，也不会直接影响 `episode_status` 和 `inspect_mcap_result.log`。

## 8. 推荐实践

- 日常批量筛查时：
  - 先跑默认配置，优先看 `inspect_error_log.html`
- 某任务确认有稳定采样频率差异时：
  - 只在 `inspect_config` 里局部调整对应阈值
- 想快速确认是否是局部掉帧时：
  - 重点看 `interval_spike_or_drop_frame`
- 想确认运动质量时：
  - 重点看 `joint_jump_violation`、`master_slave_joint_gap`、`fk_ee_pose_mismatch`
- 想判断规则结果是否可靠时：
  - 不要只看 `PASS / ERROR`，一定要回到 full log / error log 看明细指标

## 9. 总结

当前这条链路的价值不只是“多了几条规则”，更关键的是把检查流程变成了：

- 可配置
- 可结构化输出
- 可追溯到具体 rule id 和具体指标
- 能串上视频和人工复核

同时也要注意，它更适合做“结构化自动初筛 + 人工复核辅助”，而不是完全替代人工判断的最终裁决器。
