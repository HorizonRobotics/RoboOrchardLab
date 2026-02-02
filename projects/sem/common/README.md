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


## Export processor and model from python config
```bash
cd projects/sem/common
python3 export.py --config configs/config_sem_common.py $@
```

# Do evaluation in Orchard Isaac Envs
## Local run
### Prerequisites

Docker: `hub.hobot.cc/auto/robot_lab/mengao.zhao:ubuntu22.04-gcc11.4-py3.11-cuda12.8-isaac_lab-v2.0.2-sem-ext-v0.2`

Before running the evaluation, make sure you have:

- Cloned the `robo_orchard` repository
```bash
git clone git@jh-gitlab.hobot.cc:dep/robot-lab/robo_orchard.git
cd robo_orchard
git checkout feature/sim_dev # TODO: replace with a tagged version
```
- Properly set up Xvfb and x11vnc for Isaac Sim offscreen rendering
```bash
id=1 # id can be any unused display number.
Xvfb :$id -screen 0 1920x1200x24 -ac +extension GLX +render -noreset &
x11vnc -display :$id -forever -bg -ncache
```
### Run Evaluation
```bash
export ORCHARD_ISAAC_DIR=$WORKING_PATH/robo_orchard  # update to the robo_orchard directory
cp projects/sem/common/isaac_eval.py $ORCHARD_ISAAC_DIR
cp -r projects/sem/common/isaac_task_config $ORCHARD_ISAAC_DIR
cp -r robo_orchard_lab $ORCHARD_ISAAC_DIR/python/robo_orchard_lab
export PYTHONPATH=python/robo_orchard_isaac:$PYTHONPATH
export PYTHONPATH=python/robo_orchard_planner:$PYTHONPATH
export PYTHONPATH=python/robo_orchard_lab:$PYTHONPATH
cd $ORCHARD_ISAAC_DIR

vlm_ckpt_dir=/horizon-bucket/robot_lab/users/xuewu.lin/ckpt
urdf_dir=/horizon-bucket/robot_lab/users/xuewu.lin/urdf
seed=100000
task_names=stack_block_two,place_mouse_pad
model_config="xxx"  # local directory or the http url
model_processor=isaac_pick_place_processor
multi_task_config=isaac_task_config/multi_task_setting.yaml

DISPLAY=:$id python3 isaac_eval.py \
    --task_names ${task_names} \
    --model_config ${model_config} \
    --vlm_ckpt_dir ${vlm_ckpt_dir} \
    --urdf_dir ${urdf_dir} \
    --model_processor ${model_processor} \
    --seed ${seed} \
    --model_prefix model_0 \
    --test_num 100 \
    --multi_task_config ${multi_task_config}
```

## Cluster run
```bash
RoboOrchardJob-AIDISubmit submit_from_config --config projects/sem/common/submit_cfg_isaac_eval.json
```

# Do evaluation in LIBERO Envs
## Local run
### Prerequisites
Before running the evaluation, make sure you have:
- Cloned the LIBERO repo:
```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -r requirements.txt # Please note the specific Torch version requirements.
```
### Run Evaluation
```bash
export CUDA_VISIBLE_DEVICES=0,1  # Specify GPUs (e.g., use 2 GPUs)
export PYTHONPATH=$PYTHONPATH:.  # Ensure eval_policy.py can be found
export LIBERO_DIR=$WORKING_PATH/LIBERO 
cp -r projects/sem/common/sem_libero_policy $LIBERO_DIR
cp -r projects/sem/common/libero_eval.py $LIBERO_DIR
cp projects/sem/libero/eval_policy.py $LIBERO_DIR
cp projects/sem/libero/libero_utils.py $LIBERO_DIR

model_config="[http://pfs-svcspawner.bcloud-bj-zone1.hobot.cc/user/homespace/](http://pfs-svcspawner.bcloud-bj-zone1.hobot.cc/user/homespace/)..." # URL or local path
vlm_ckpt_dir="/horizon-bucket/robot_lab/users/xuewu.lin/ckpt"
urdf_dir="/horizon-bucket/robot_lab/users/xuewu.lin/urdf"

# Option 1: Run a specific benchmark suite (e.g., libero_goal)
python3 libero_eval.py \
    --model_config ${model_config} \
    --model_prefix model_0 \
    --vlm_ckpt_dir ${vlm_ckpt_dir} \
    --urdf_dir ${urdf_dir} \
    --model_processor libero_processor \
    --task_suite libero_goal \
    --num_trials_per_task 50 \
    --save_video True

# Run ALL benchmark suites (spatial, object, goal, 10). Set --task_suite to -1
python3 libero_eval.py \
    --model_config ${model_config} \
    --model_prefix model_0 \
    --vlm_ckpt_dir ${vlm_ckpt_dir} \
    --urdf_dir ${urdf_dir} \
    --model_processor libero_processor \
    --num_trials_per_task 50 \
    --save_video True
```
## Cluster run
```bash
RoboOrchardJob-AIDISubmit submit_from_config --config projects/sem/common/submit_cfg_libero_eval.json # Adjust relevant config parameters accordingly
```

# Docker image
- **New version image (supports Qwen3):**  
  `docker.hobot.cc/imagesys/robot_lab:ubuntu22.04-gcc11.4-py3.10-cuda11.8-torch260-robotwin2-transformer4571-20251030`

- **Old version image (deprecated):**  
  `docker.hobot.cc/imagesys/robot_lab:ubuntu22.04-gcc11.4-py3.10-cuda11.8-torch241-robotwin2-20250918`