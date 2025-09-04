# SEM: Enhancing Spatial Understanding for Robust Robot Manipulation

<div align="center" class="authors">
    <a href="https://scholar.google.com/citations?user=pfXQwcQAAAAJ&hl=en" target="_blank">Xuewu Lin</a>,
    <a href="https://wzmsltw.github.io/" target="_blank">Tianwei Lin</a>,
    <a href="https://scholar.google.com/citations?user=F2e_jZMAAAAJ&hl=en" target="_blank">Lichao Huang</a>,
    <a href="https://openreview.net/profile?id=~HONGYU_XIE2" target="_blank">Hongyu Xie</a>,
    <a href="" target="_blank">Yiwei Jin</a>,
    <a href="https://scholar.google.com/citations?user=m3IK258AAAAJ&hl=zh-CN&oi=ao" target="_blank">Keyu Li</a>,
    <a href="https://scholar.google.com/citations?user=HQfc8TEAAAAJ&hl=en" target="_blank">Zhizhong Su</a>
</div>

<div align="center" style="line-height: 3;">
  <a href="https://arxiv.org/abs/2505.16196" target="_blank" style="margin: 2px;">
    <img alt="Paper" src="https://img.shields.io/badge/Paper-Arxiv-red" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://huggingface.co/HorizonRobotics/SEM-RoboTwin-Tiny" target="_blank" style="margin: 2px;">
    <img alt="Model" src="https://img.shields.io/badge/Model-HuggingFace-red" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>


## Prepare pre-trained weights

The current project adopts the SEM-GroundingDINO architecture, utilizing GroundingDINO-tiny as the pre-trained model.
```bash
cd projects/sem/robotwin
mkdir ckpt

# Swin-Tiny
wget https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swint_ogc_mmdet-822d7e9d.pth -O ckpt/groundingdino_swint_ogc_mmdet-822d7e9d.pth
python tools/ckpt_rename.py ckpt/groundingdino_swint_ogc_mmdet-822d7e9d.pth --output ./ckpt
```

Download bert config and pretrain weights from [huggingface](https://huggingface.co/google-bert/bert-base-uncased/tree/main).

```text
projects/sem/robotwin
└──ckpt
    ├──groundingdino_swint_ogc_mmdet-822d7e9d.pth
    ├──groundingdino_swint_ogc_mmdet-822d7e9d-rename.pth  # generated after rename
    └──bert-base-uncased
        ├──config.json
        ├──tokenizer_config.json
        ├──tokenizer.json
        ├──pytorch_model.bin
        ...
```

## Prepare Python Dependency

First prepare [robotwin dependency](https://github.com/TianxingChen/RoboTwin/blob/main/INSTALLATION.md), then install the following packages:

```text
# It is recommended to use CUDA 11.8.
torch==2.4.1
torchmetrics==1.6.1 
torchvision==0.19.1
transformers==4.49.0
lmdb==1.6.2 
safetensors==0.5.3 
accelerate==1.4.0 
diffusers==0.32.2 
timeout-decorator==0.5.0
requests==2.32.3 
h5py==3.13.0
```


## Prepare the data
First, run RoboTwin to obtain the expert data from the simulation.

```bash
git clone https://github.com/RoboTwin-Platform/RoboTwin.git
cd RoboTwin
git checkout e71140e9734e69686daa420a9be8b75a20ff4587  # TODO: Support the latest version
```
Follow the instructions in the RobotWin code repository to download the required assets and generate data.
Then, use the following command to package the data into LMDB format for training.
```bash
cd path/to/robo_orchard_lab

# for the robotwin2.0 master branch before commit e71140e9734e69686daa420a9be8b75a20ff4587 or the Challenge-Cup-2025 branch
python robo_orchard_lab/dataset/robotwin/robotwin_packer.py \
    --input_path path/to/robotwin_data \
    --output_path "projects/sem/robotwin/data/lmdb" \
    --task_names ${task_names} \
    --config_name demo_clean
```

Make sure the resulting data path is as follows:
```text
projects/sem/robotwin
└──data
    └──lmdb
        ├──depth
        ├──image
        ├──index
        └──meta
```

## Run

Update the [URDF file directory in the config](./config_sem_robotwin.py#L21) to use the URDF provided by RoboTwin.
```bash
cd projects/sem/robotwin
CONFIG=config_sem_robotwin.py

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
    --workspace /job_data \
    --config ${CONFIG}

# eval with single-gpu (for Robotwin2.0 master branch)
# following instruction in ./sem_policy/README.md
```


## Experimental Results

Below are the results of the single-task models trained on 100 episodes.
Training Configuration: 2 machines × 8 GPUs (100k steps for all tasks).

| Task Category  | Hammer Beat | Block Handover | Bottle Adjust | Blocks Stack (easy/hard) | Container Place | Bottles Pick | Dual Shoes Place | Dual Bottles Pick (easy/hard) | Empty Cup Place | Pick Apple | Put Apple Cabinet | Mug hanging (easy/hard) | Shoe Place  | Mean    |
|----------------|-------------|----------------|---------------|--------------------------|-----------------|--------------|------------------|-------------------------------|-----------------|------------|-------------------|-------------------------|-------------|---------|
| **SEM**            | 97.0±1.4    | 96.7±1.2       | 69.0±5.1      | 76.3±2.6 / 89.7±1.7      | 56.3±4.2        | 56.3±5.3     | 51.7±2.6         | 98.0±0.8 / 60.7±0.5           | 87.0±2.2        | 98.7±0.5   | 73.3±0.9          | 12.3±4.8 / 6.0±1.6      | 88.3±1.2    | 69.8±2.7|


## :handshake: Acknowledgement
[RoboTwin](https://github.com/TianxingChen/RoboTwin)
