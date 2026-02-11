# HoloBrain-0


## Prepare Data
## :file_folder: Prepare Data
### [RoboTwin2.0](https://github.com/RoboTwin-Platform/RoboTwin)
Follow the instructions in the RoboTwin code repository to download the required assets and generate data.
Then, use the following command to package the data into LMDB format for training.

```bash
cd path/to/robo_orchard_lab

# require data format from the robotwin2.0 master branch before commit e71140e9734e69686daa420a9be8b75a20ff4587
python3 -m robo_orchard_lab.dataset.robotwin.robotwin_packer.py \
    --input_path path/to/robotwin_data \
    --output_path "projects/holobrain/data/lmdb" \
    --task_names ${task_names} \
    --config_name demo_clean
```


## :rocket: Run Training
```bash
cd projects/holobrain
CONFIG=configs/config_holobrain_qwen_common.py # or configs/config_holobrain_gd_common.py

# train with single-gpu
python3 train.py --config ${CONFIG}

# train with multi-gpu multi-machine
# example: 2 machines × 8 gpus
accelerate launch  \
    --num_machines 2 \
    --num-processes 16  \
    --multi-gpu \
    --gpu-ids 0,1,2,3,4,5,6,7  \
    --machine_rank ${current_rank} \
    --main_process_ip ${main_process_ip} \
    --main_process_port 1227 \
    train.py \
    --workspace ./workspace \
    --config ${CONFIG}
```


## :package: Export Model and Processors

```bash
cd projects/holobrain
CONFIG=configs/config_holobrain_qwen_common.py # or configs/config_holobrain_gd_common.py

python3 export.py --config ${CONFIG} --workspace ./workspace
```


## :bar_chart: Run Evaluation

### RoboTwin2.0
Refer to [robotwin_eval](projects/holobrain/holobrain_robotwin_eval/README.md).
