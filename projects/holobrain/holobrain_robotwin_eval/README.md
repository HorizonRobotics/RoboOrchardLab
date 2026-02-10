# Prepare RoboTwin env
```bash
git clone https://github.com/RoboTwin-Platform/RoboTwin.git
cd RoboTwin
git checkout e71140e9734e69686daa420a9be8b75a20ff4587  # TODO: Support the latest version
```
Follow the setup instructions in RoboTwin to set up the evaluation environment.


# Run Evaluation

```bash
export HYDRA_FULL_ERROR=1
python3 projects/holobrain/holobrain_robotwin_eval/policy.py \
  --task_config demo_clean
  --task_names place_empty_cup,adjust_bottle,stack_blocks_three \
  --model_config ${MODEL_TO_EVAL} \
  --model_processor robotwin2_0_processor \
  --robotwin_dir ${ROBOTWIN_DIR} \
  --num_workers 8  # use 8 gpus
```
