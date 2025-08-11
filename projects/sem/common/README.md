# Training
## Local run
```bash
cd projects/sem/common

# single gpu
python3 train.py --config configs/config_sem_common.py $@

# multi gpu
accelerate launch --multi-gpu --num-processes 4 --gpu-ids 0,1,2,3 train.py --config configs/config_sem_common.py
```

## Cluster run
```bash
RoboOrchardJob-AIDISubmit submit_from_config --config projects/sem/common/submit_cfg.json
```


# Do evaluation in Robotwin2.0 Envs
## Local run
```bash
export CUDA_VISIBLE_DEVICES=0,1  # use two gpus
export ROBOTWIN_DIR=$WORKING_PATH/robotwin  # update to the robotwin directory
cp -r projects/sem/common/sem_robotwin_policy $ROBOTWIN_DIR
cp -r projects/sem/common/robotwin_eval.py $ROBOTWIN_DIR

cd $ROBOTWIN_DIR
task_names=place_empty_cup,stack_blocks_three
task_config=demo_clean
model_config="xxx"  # local directory or the http url, e.g. "http://pfs-svcspawner.bcloud-bj-zone1.hobot.cc/user/homespace/xuewu.lin/plat_gpu/2025-08-09/14-02/sem_robotwin2_tempjoint_originvlm_newweightinit-20250809-140220.982428/output/checkpoints/checkpoint_10/"
vlm_ckpt_dir="/horizon-bucket/robot_lab/users/xuewu.lin/ckpt"
urdf_dir="/horizon-bucket/robot_lab/users/xuewu.lin/urdf"

python3 robotwin_eval.py \
    --task_names ${task_names} \
    --task_config ${task_config} \
    --model_config ${model_config} \
    --vlm_ckpt_dir ${vlm_ckpt_dir} \
    --urdf_dir ${urdf_dir} \
    --test_num 100
```

## Cluster run
```bash
RoboOrchardJob-AIDISubmit submit_from_config --config projects/sem/common/submit_cfg_robotwin_eval.json
```
